//! Helpers shared by the tile emitters.

use tvm::ir::{IntImm, PrimExpr, PrimType, Var};
use tvm::prim::Cast;
use tvm::tvm_ffi::object::ObjectRef;

use super::super::analyze::layout::{expr_any, op_binary};
use super::super::analyze::tile_forms::TileRegion;
use super::super::analyze::util::{
    as_var, dtype_text, ffi_error, ffi_text, oref, prim, unsupported, AResult, Failure,
};
use super::{Emitter, NestedLoadSite, SplitArgument};

/// The uncalled `buffer_load_factory` a GEMM operand reference receives.
#[derive(Clone)]
pub(super) enum LoadFactory {
    /// `lambda: None`: let the reference install its implicit loader.
    None,
    /// The current buffer loader, read when the reference runs.
    Current,
    Reject(String),
}

impl LoadFactory {
    pub(super) fn resolve(&self, emitter: &Emitter<'_>) -> Option<NestedLoadSite> {
        match self {
            LoadFactory::None => None,
            LoadFactory::Current => Some(
                emitter
                    .nested_load_site
                    .clone()
                    .unwrap_or(NestedLoadSite::Exact),
            ),
            LoadFactory::Reject(message) => Some(NestedLoadSite::Reject(message.clone())),
        }
    }
}

/// One captured pure closure: its lines and move-captured arguments.
pub(super) type Capture = (Vec<String>, Vec<SplitArgument>);

/// `tirx.IntImm("int64", value)`.
pub(super) fn int64_imm(value: i64) -> AResult<PrimExpr> {
    Ok(IntImm::new("int64", value)?.into())
}

/// `tirx.Var(name, dtype)` as a primitive expression together with its identity.
pub(super) fn tir_var(name: &str, dtype: &str) -> AResult<(ObjectRef, PrimExpr)> {
    let variable = oref(Var::new(name, dtype)?);
    let expr = prim(&variable)?;
    Ok((variable, expr))
}

/// `minimum + tirx.Cast(minimum.ty, coordinate)`.
pub(super) fn offset_index(minimum: &PrimExpr, coordinate: &PrimExpr) -> AResult<PrimExpr> {
    let dtype = dtype_text(minimum.dtype());
    let cast: PrimExpr = Cast::new(PrimType::new(&dtype)?, coordinate.clone())?.into();
    op_binary("_OpAdd", expr_any(minimum), expr_any(&cast))
}

/// The pure TIR coordinates of one linear element index.
pub(super) fn linear_coordinates(linear: &PrimExpr, extents: &[i64]) -> AResult<Vec<PrimExpr>> {
    let mut coordinates = Vec::new();
    for (axis, extent) in extents.iter().enumerate() {
        let stride: i64 = extents[axis + 1..].iter().product();
        let mut coordinate = linear.clone();
        if stride != 1 {
            coordinate = op_binary(
                "_OpFloorDiv",
                expr_any(&coordinate),
                expr_any(&int64_imm(stride)?),
            )?;
        }
        if *extent != 1 {
            coordinate = op_binary(
                "_OpFloorMod",
                expr_any(&coordinate),
                expr_any(&int64_imm(*extent)?),
            )?;
        } else {
            coordinate = int64_imm(0)?;
        }
        coordinates.push(coordinate);
    }
    Ok(coordinates)
}

/// `str(var.name)` of a TIR variable expression.
pub(super) fn var_name(expr: &PrimExpr) -> AResult<String> {
    match as_var(&oref(expr.clone())) {
        Some(var) => Ok(ffi_text(&var.name)),
        None => Err(Failure::Ffi(ffi_error(
            "tile coordinate is not a TIR variable",
        ))),
    }
}

pub(super) fn tile_logical_region_indices(
    region: &TileRegion,
    logical_coordinates: &[PrimExpr],
) -> AResult<Vec<PrimExpr>> {
    let expected = region.extents.iter().filter(|extent| **extent != 1).count();
    if logical_coordinates.len() != expected {
        return unsupported(format!(
            "Logical tile coordinate rank {} does not match region rank {expected}",
            logical_coordinates.len()
        ));
    }
    let mut coordinates = logical_coordinates.iter();
    let mut expanded = Vec::new();
    for extent in &region.extents {
        expanded.push(if *extent == 1 {
            int64_imm(0)?
        } else {
            coordinates.next().expect("logical coordinate").clone()
        });
    }
    if expanded.len() != region.extents.len() {
        return unsupported("Internal tile coordinate rank mismatch");
    }
    region
        .mins
        .iter()
        .zip(expanded.iter())
        .map(|(minimum, coordinate)| offset_index(minimum, coordinate))
        .collect()
}

pub(super) fn v2_tile_element(dtype: &str) -> Option<&'static str> {
    Some(match dtype {
        "bool" => "v2::mem::variant::Bool",
        "int8" => "v2::reg::variant::I8",
        "uint8" => "v2::reg::variant::U8",
        "float8_e4m3fn" => "v2::tile::variant::E4m3",
        "float8_e8m0fnu" => "v2::tile::variant::E8m0",
        "int16" => "v2::reg::variant::I16",
        "uint16" => "v2::reg::variant::U16",
        "float16" => "v2::reg::variant::F16",
        "bfloat16" => "v2::reg::variant::Bf16",
        "int32" => "v2::reg::variant::I32",
        "uint32" => "v2::reg::variant::U32",
        "float32" => "v2::reg::variant::F32",
        "int64" => "v2::reg::variant::I64",
        "uint64" => "v2::reg::variant::U64",
        "float64" => "v2::reg::variant::F64",
        _ => return None,
    })
}

/// `v2::tile::variant::<Scope>` of a whole-tile copy.
pub(super) fn v2_tile_scope(exec_scope: &str) -> Option<&'static str> {
    Some(match exec_scope {
        "thread" => "v2::tile::variant::Thread",
        "warp" => "v2::tile::variant::Warp",
        "warpgroup" => "v2::tile::variant::Warpgroup<4>",
        "cta" => "v2::tile::variant::Cta",
        _ => return None,
    })
}

impl Emitter<'_> {
    pub(super) fn emit_captured_closure(
        &mut self,
        name: &str,
        signature: &str,
        body: &[String],
        arguments: &[SplitArgument],
    ) {
        self.emit_line(&format!("let {name} = {{"));
        self.indent += 1;
        for argument in arguments {
            self.emit_line(&format!(
                "let {}: {} = {};",
                argument.name, argument.rust_type, argument.call_code
            ));
        }
        self.emit_line(signature);
        self.indent += 1;
        for line in body {
            self.emit_line(line);
        }
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("};");
    }
}
