//! `int`/`PrimExpr` arithmetic and physical layout coordinates for the layout proofs.

use tvm::analysis::Analyzer;
use tvm::ir::{PrimExpr, Var};
use tvm::tvm_ffi::{self as tvm_ffi, Any, Array, Map, String as FfiString};

use super::super::layout::{expr_any, int_any, op_binary};
use super::super::util::{ffi_error, ffi_text, oref, prim, repr_text, unsupported, AResult};
use super::parse::sorted_strings;
use super::TileRegion;

/// `Var(name, "int32")` as a primitive expression.
pub(super) fn int32_var(name: &str) -> AResult<PrimExpr> {
    prim(&oref(Var::new(name, "int32")?))
}

/// TVM's `int -> PrimExpr` FFI fallback conversion (`TypeTraits<PrimExpr>`).
pub(super) fn int_expr(value: i64) -> AResult<PrimExpr> {
    let converted: Any =
        tvm_ffi::cached_global_func!("prim.convert").call_tuple((int_any(value),))?;
    Ok(PrimExpr::try_from(converted)?)
}

/// `int` or `PrimExpr` arithmetic exactly as Python evaluates it.
#[derive(Clone)]
pub enum Val {
    Int(i64),
    Expr(PrimExpr),
}

impl Val {
    fn any(&self) -> Any {
        match self {
            Val::Int(value) => int_any(*value),
            Val::Expr(expr) => expr_any(expr),
        }
    }

    /// `str(value)`.
    fn text(&self) -> AResult<String> {
        match self {
            Val::Int(value) => Ok(value.to_string()),
            Val::Expr(expr) => repr_text(&oref(expr.clone())),
        }
    }

    /// The value as one FFI argument of PrimExpr type.
    fn expr(&self) -> AResult<PrimExpr> {
        match self {
            Val::Int(value) => int_expr(*value),
            Val::Expr(expr) => Ok(expr.clone()),
        }
    }
}

fn py_floordiv(a: i64, b: i64) -> i64 {
    let quotient = a / b;
    if a % b != 0 && ((a < 0) != (b < 0)) {
        quotient - 1
    } else {
        quotient
    }
}

fn py_floormod(a: i64, b: i64) -> i64 {
    a - b * py_floordiv(a, b)
}

fn val_op(name: &str, lhs: &Val, rhs: &Val) -> AResult<Val> {
    if let (Val::Int(a), Val::Int(b)) = (lhs, rhs) {
        return Ok(Val::Int(match name {
            "_OpAdd" => a + b,
            "_OpSub" => a - b,
            "_OpMul" => a * b,
            "_OpFloorDiv" => py_floordiv(*a, *b),
            "_OpFloorMod" => py_floormod(*a, *b),
            _ => unreachable!("integer operator {name}"),
        }));
    }
    Ok(Val::Expr(op_binary(name, lhs.any(), rhs.any())?))
}

pub(super) fn vadd(lhs: &Val, rhs: &Val) -> AResult<Val> {
    val_op("_OpAdd", lhs, rhs)
}

pub(super) fn vsub(lhs: &Val, rhs: &Val) -> AResult<Val> {
    val_op("_OpSub", lhs, rhs)
}

pub(super) fn vmul(lhs: &Val, rhs: &Val) -> AResult<Val> {
    val_op("_OpMul", lhs, rhs)
}

pub(super) fn vfloordiv(lhs: &Val, rhs: &Val) -> AResult<Val> {
    val_op("_OpFloorDiv", lhs, rhs)
}

pub(super) fn vfloormod(lhs: &Val, rhs: &Val) -> AResult<Val> {
    val_op("_OpFloorMod", lhs, rhs)
}

pub(super) fn vint(value: i64) -> Val {
    Val::Int(value)
}

pub(super) fn vexpr(expr: &PrimExpr) -> Val {
    Val::Expr(expr.clone())
}

/// `analyzer.can_prove_equal(lhs, rhs)` with Python's int conversion.
pub(super) fn prove_equal(analyzer: &Analyzer, lhs: &Val, rhs: &Val) -> AResult<bool> {
    Ok(analyzer.can_prove_equal(&lhs.expr()?, &rhs.expr()?)?)
}

/// `analyzer.simplify(value)` with Python's int conversion.
pub(super) fn simplify_val(analyzer: &Analyzer, value: &Val) -> AResult<PrimExpr> {
    Ok(analyzer.simplify(&value.expr()?)?)
}

// ----------------------------------------------------------------------
// Physical layout coordinates.
// ----------------------------------------------------------------------

/// One `layout.apply(...)` result: physical axis name -> coordinate expression.
pub(super) type Physical = Vec<(String, PrimExpr)>;

pub(super) fn physical_get<'a>(physical: &'a Physical, axis: &str) -> Option<&'a PrimExpr> {
    physical
        .iter()
        .find(|(name, _)| name == axis)
        .map(|(_, value)| value)
}

/// `tuple(sorted(physical))`.
pub(super) fn physical_axes(physical: &Physical) -> Vec<String> {
    sorted_strings(
        &physical
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>(),
    )
}

/// `set(physical) == {...}`.
pub(super) fn axes_are(physical: &Physical, expected: &[&str]) -> bool {
    let mut expected: Vec<String> = expected.iter().map(|axis| (*axis).to_owned()).collect();
    expected.sort();
    expected.dedup();
    physical_axes(physical) == expected
}

/// `set(physical).issubset({...})`.
pub(super) fn axes_subset(physical: &Physical, allowed: &[&str]) -> bool {
    physical
        .iter()
        .all(|(name, _)| allowed.contains(&name.as_str()))
}

fn logical_region_indices_at(region: &TileRegion, coordinates: &[Val]) -> AResult<Vec<PrimExpr>> {
    let expected_rank = region.extents.iter().filter(|extent| **extent != 1).count();
    if coordinates.len() != expected_rank {
        return unsupported(format!(
            "Tile logical coordinate rank {} does not match region rank {expected_rank}",
            coordinates.len()
        ));
    }
    let mut next = coordinates.iter();
    let mut indices = Vec::new();
    for (minimum, extent) in region.mins.iter().zip(region.extents.iter()) {
        let offset = if *extent == 1 {
            vint(0)
        } else {
            next.next().expect("coordinate per logical axis").clone()
        };
        indices.push(op_binary("_OpAdd", expr_any(minimum), offset.any())?);
    }
    Ok(indices)
}

pub(super) fn physical_layout_coordinates_at(
    region: &TileRegion,
    coordinates: &[Val],
    op_name: &str,
) -> AResult<Physical> {
    let coordinate_text = coordinates
        .iter()
        .map(Val::text)
        .collect::<AResult<Vec<_>>>()?
        .join(", ");
    let indices = logical_region_indices_at(region, coordinates)?;
    let buffer_type = region.buffer.buffer_type();
    let mapped = (|| -> tvm_ffi::Result<Map<FfiString, PrimExpr>> {
        let Some(layout) = buffer_type.layout.clone() else {
            return Err(ffi_error(
                "'NoneType' object has no attribute 'canonicalize'",
            ));
        };
        layout
            .canonicalize()?
            .apply_with_shape(&Array::new(indices), &buffer_type.shape)
    })();
    match mapped {
        Ok(map) => Ok(map
            .iter()
            .map(|(axis, value)| (ffi_text(&axis), value))
            .collect()),
        Err(error) => unsupported(format!(
            "TilePrimitiveCall({op_name}): cannot apply physical layout at logical ({coordinate_text}): {}",
            error.message()
        )),
    }
}

pub(super) fn physical_layout_coordinates(
    region: &TileRegion,
    row: &Val,
    col: &Val,
    op_name: &str,
) -> AResult<Physical> {
    physical_layout_coordinates_at(region, &[row.clone(), col.clone()], op_name)
}
