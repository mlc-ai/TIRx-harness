//! Call operands, configuration values, literal readers and the checks every
//! tile resolver shares.

use crate::tvm_compat::int_value;
use tvm::analysis::Analyzer;
use tvm::ir::StringImmObj;
use tvm::ir::TensorRegionObj;
use tvm::ir::{FloatImm, FloatImmObj, IntImm, IntImmObj, PrimExpr, Range};
use tvm::tirx::{BufferVar, TilePrimitiveCallObj};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{Any, ObjectRefCore, String as FfiString, TypeIndex};

use super::super::layout::expr_any;
use super::super::util::{
    buffer_dtype, buffer_scope, dtype_of, ffi_error, ffi_text, int_imm_expr, kind, repr_text,
    unmodeled, unsupported, AResult, Failure,
};
use super::{TileOperand, TileRegion, TileScalar, TILE_OP_PREFIX};

const FLOAT_DTYPES: [&str; 4] = ["float16", "bfloat16", "float32", "float64"];
pub(super) const BOOL: &str = "bool";
pub(super) const EXEC_SCOPES: [&str; 3] = ["thread", "warp", "warpgroup"];
pub(super) const ELEMENTWISE_EXEC_SCOPES: [&str; 4] = ["thread", "warp", "warpgroup", "cta"];

pub(super) fn is_float_dtype(dtype: &str) -> bool {
    FLOAT_DTYPES.contains(&dtype)
}

pub(super) fn unmodeled_tile_form<T>(op_name: &str, message: impl Into<String>) -> AResult<T> {
    unmodeled(format!("tile:{TILE_OP_PREFIX}{op_name}"), message)
}

pub(super) fn sorted_strings(items: &[String]) -> Vec<String> {
    let mut sorted = items.to_vec();
    sorted.sort();
    sorted.dedup();
    sorted
}

// ----------------------------------------------------------------------
// Decoded FFI operands and configuration values.
// ----------------------------------------------------------------------

/// One decoded `tvm_ffi.Any`: a literal or an IR object.
#[derive(Clone)]
pub enum AnyValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Object(ObjectRef),
}

pub fn any_value(value: &Any) -> AResult<AnyValue> {
    let index = value.type_index();
    if index == TypeIndex::kTVMFFINone as i32 {
        return Ok(AnyValue::None);
    }
    if index == TypeIndex::kTVMFFIBool as i32 {
        return Ok(AnyValue::Bool(bool::try_from(value.clone())?));
    }
    if index == TypeIndex::kTVMFFIInt as i32 {
        return Ok(AnyValue::Int(i64::try_from(value.clone())?));
    }
    if index == TypeIndex::kTVMFFIFloat as i32 {
        return Ok(AnyValue::Float(f64::try_from(value.clone())?));
    }
    if index == TypeIndex::kTVMFFISmallStr as i32 || index == TypeIndex::kTVMFFIStr as i32 {
        return Ok(AnyValue::Str(ffi_text(&FfiString::try_from(
            value.clone(),
        )?)));
    }
    Ok(AnyValue::Object(ObjectRef::try_from(value.clone())?))
}

impl AnyValue {
    /// Diagnostic text for a decoded operand.
    pub fn plain_text(&self) -> AResult<String> {
        Ok(match self {
            AnyValue::None => "None".to_owned(),
            AnyValue::Bool(value) => value.to_string(),
            AnyValue::Int(value) => value.to_string(),
            AnyValue::Float(value) => format!("{:?}", *value),
            AnyValue::Str(value) => value.clone(),
            AnyValue::Object(node) => repr_text(node)?,
        })
    }

    /// `type(value).__name__`.
    pub fn type_name(&self) -> String {
        match self {
            AnyValue::None => "NoneType".to_owned(),
            AnyValue::Bool(_) => "bool".to_owned(),
            AnyValue::Int(_) => "int".to_owned(),
            AnyValue::Float(_) => "float".to_owned(),
            AnyValue::Str(_) => "str".to_owned(),
            AnyValue::Object(node) => kind(node).unwrap_or("Object").to_owned(),
        }
    }
}

// ----------------------------------------------------------------------
// Literal readers.
// ----------------------------------------------------------------------

pub(super) fn static_int_any(analyzer: &Analyzer, value: &Any, field: &str) -> AResult<i64> {
    let decoded = any_value(value)?;
    let expr = match &decoded {
        AnyValue::Int(value) => return Ok(*value),
        AnyValue::Bool(value) => return Ok(i64::from(*value)),
        AnyValue::Object(node) => PrimExpr::try_from(Any::from(node.clone())).ok(),
        _ => None,
    };
    if let Some(expr) = expr {
        let simplified = analyzer.simplify(&expr)?;
        if let Some(value) = int_imm_expr(&simplified) {
            return Ok(value);
        }
    }
    unsupported(format!(
        "TilePrimitiveCall({field}): requires a static integer, got {}",
        decoded.plain_text()?
    ))
}

pub(super) fn static_int_expr(analyzer: &Analyzer, value: &PrimExpr, field: &str) -> AResult<i64> {
    static_int_any(analyzer, &expr_any(value), field)
}

pub(super) fn literal_bool(value: &Any, field: &str) -> AResult<bool> {
    let decoded = any_value(value)?;
    if let AnyValue::Bool(value) = decoded {
        return Ok(value);
    }
    if let AnyValue::Object(node) = &decoded {
        if let Some(imm) = node.as_node::<IntImmObj>() {
            return Ok(int_value(imm)? != 0);
        }
    }
    unsupported(format!(
        "TilePrimitiveCall({field}): expected a static bool, got {}",
        decoded.plain_text()?
    ))
}

pub(super) fn literal_string(value: &Any, field: &str) -> AResult<String> {
    let decoded = any_value(value)?;
    if let AnyValue::Str(value) = &decoded {
        return Ok(value.clone());
    }
    if let AnyValue::Object(node) = &decoded {
        if let Some(imm) = node.as_node::<StringImmObj>() {
            return Ok(ffi_text(&imm.value));
        }
    }
    unsupported(format!(
        "TilePrimitiveCall({field}): expected a string, got {}",
        decoded.plain_text()?
    ))
}

pub(super) fn literal_int(value: &Any, field: &str) -> AResult<i64> {
    let decoded = any_value(value)?;
    match &decoded {
        AnyValue::Bool(_) => {
            return unsupported(format!(
                "TilePrimitiveCall({field}): expected a static integer, got bool"
            ))
        }
        AnyValue::Int(value) => return Ok(*value),
        AnyValue::Object(node) => {
            if let Some(imm) = node.as_node::<IntImmObj>() {
                return Ok(int_value(imm)?);
            }
        }
        _ => {}
    }
    unsupported(format!(
        "TilePrimitiveCall({field}): expected a static integer, got {}",
        decoded.plain_text()?
    ))
}

pub(super) fn literal_float(value: &Any, field: &str) -> AResult<f64> {
    let decoded = any_value(value)?;
    match &decoded {
        AnyValue::Bool(_) => {
            return unsupported(format!(
                "TilePrimitiveCall({field}): expected a static float, got bool"
            ))
        }
        AnyValue::Int(value) => return Ok(*value as f64),
        AnyValue::Float(value) => return Ok(*value),
        AnyValue::Object(node) => {
            if let Ok(expr) = PrimExpr::try_from(Any::from(node.clone())) {
                let simplified = Analyzer::new()?.simplify(&expr)?;
                if let Some(imm) = simplified.as_node::<FloatImmObj>() {
                    return Ok(imm.value);
                }
                if let Some(imm) = simplified.as_node::<IntImmObj>() {
                    return Ok(int_value(imm)? as f64);
                }
            }
        }
        _ => {}
    }
    unsupported(format!(
        "TilePrimitiveCall({field}): expected a static float, got {}",
        decoded.plain_text()?
    ))
}

// ----------------------------------------------------------------------
// Operands and common checks.
// ----------------------------------------------------------------------

fn memory_scope(buffer: &BufferVar) -> String {
    let scope = buffer_scope(buffer);
    if scope.starts_with("shared") {
        "shared".to_owned()
    } else {
        scope
    }
}

pub(super) fn operand(analyzer: &Analyzer, value: &Any, role: &str) -> AResult<TileOperand> {
    let decoded = any_value(value)?;
    match &decoded {
        AnyValue::Object(node) => {
            if let Some(region) = node.as_node::<TensorRegionObj>() {
                let ranges: Vec<Range> = region.region.iter().collect();
                let mut extents = Vec::new();
                for (axis, item) in ranges.iter().enumerate() {
                    extents.push(static_int_expr(
                        analyzer,
                        &item.extent,
                        &format!("{role}.extent[{axis}]"),
                    )?);
                }
                if extents.iter().any(|extent| *extent <= 0) {
                    return unsupported(format!(
                        "TilePrimitiveCall({role}): region extents must be positive, got {:?}",
                        &extents
                    ));
                }
                let buffer = BufferVar::try_from(region.source.clone())?;
                return Ok(TileOperand::Region(TileRegion {
                    dtype: buffer_dtype(&buffer),
                    memory_scope: memory_scope(&buffer),
                    buffer,
                    mins: ranges.iter().map(|item| item.min.clone()).collect(),
                    extents,
                }));
            }
            // Current TIRx exposes buffers as `Var`, so a bare buffer reaches the
            // typed-scalar path below.
        }
        AnyValue::Bool(value) => {
            let expr: PrimExpr = IntImm::new(BOOL, i64::from(*value))?.into();
            return Ok(TileOperand::Scalar(TileScalar {
                expr: expr_any(&expr),
                dtype: BOOL.to_owned(),
            }));
        }
        AnyValue::Int(value) => {
            // Pick the narrowest type that represents the literal exactly.
            let dtype = if (-(1i64 << 31)..(1i64 << 31)).contains(value) {
                "int32"
            } else {
                // Every i64 fits int64, so no uint64 literal arises here.
                "int64"
            };
            let expr: PrimExpr = IntImm::new(dtype, *value)?.into();
            return Ok(TileOperand::Scalar(TileScalar {
                expr: expr_any(&expr),
                dtype: dtype.to_owned(),
            }));
        }
        AnyValue::Float(value) => {
            let expr: PrimExpr = FloatImm::new("float64", *value)?.into();
            return Ok(TileOperand::Scalar(TileScalar {
                expr: expr_any(&expr),
                dtype: "float64".to_owned(),
            }));
        }
        _ => {}
    }
    let AnyValue::Object(node) = &decoded else {
        return Err(Failure::Ffi(ffi_error(&format!(
            "'{}' object has no attribute 'ty'",
            decoded.type_name()
        ))));
    };
    let dtype = dtype_of(node)?;
    if dtype.is_empty() {
        return unsupported(format!(
            "TilePrimitiveCall({role}): expected a buffer region or typed scalar, got {}",
            decoded.type_name()
        ));
    }
    Ok(TileOperand::Scalar(TileScalar {
        expr: value.clone(),
        dtype,
    }))
}

pub(super) fn destination(analyzer: &Analyzer, value: &Any, op_name: &str) -> AResult<TileRegion> {
    match operand(analyzer, value, &format!("{op_name}.dst"))? {
        TileOperand::Region(region) => Ok(region),
        TileOperand::Scalar(_) => unsupported(format!(
            "TilePrimitiveCall({op_name}): destination must be a buffer"
        )),
    }
}

/// The call metadata (scope, dispatch and config).
pub(super) struct CommonFacts {
    pub(super) scope: String,
    pub(super) dispatch: Option<String>,
    pub(super) config: Vec<(String, Any)>,
}

impl CommonFacts {
    pub(super) fn get(&self, key: &str) -> Option<&Any> {
        self.config
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    pub(super) fn has(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub(super) fn keys(&self) -> Vec<String> {
        self.config.iter().map(|(name, _)| name.clone()).collect()
    }

    pub(super) fn unknown_keys(&self, known: &[&str]) -> Vec<String> {
        sorted_strings(
            &self
                .config
                .iter()
                .map(|(name, _)| name.clone())
                .filter(|name| !known.contains(&name.as_str()))
                .collect::<Vec<_>>(),
        )
    }
}

pub(super) fn check_common(
    call: &TilePrimitiveCallObj,
    op_name: &str,
    exec_scopes: &[&str],
) -> AResult<CommonFacts> {
    let scope = call.scope.name()?.to_owned();
    if !exec_scopes.contains(&scope.as_str()) {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): exec scope {:?} is not implemented",
            &scope
        ));
    }
    if call.workspace.len() != 0 {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): non-empty workspace is not implemented"
        ));
    }
    let dispatch = Option::<FfiString>::from(call.dispatch.clone()).map(|text| ffi_text(&text));
    let config = call
        .config
        .iter()
        .map(|(key, value)| (ffi_text(&key), value))
        .collect();
    Ok(CommonFacts {
        scope,
        dispatch,
        config,
    })
}

pub(super) fn require_arity(
    call: &TilePrimitiveCallObj,
    op_name: &str,
    count: usize,
) -> AResult<()> {
    if call.args.len() != count {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): expected {count} args, got {}",
            call.args.len()
        ));
    }
    Ok(())
}

pub(super) fn same_logical_shape(
    op_name: &str,
    destination: &TileRegion,
    sources: &[&TileRegion],
) -> AResult<()> {
    let expected = destination.logical_shape();
    for source in sources {
        let shape = source.logical_shape();
        if shape != expected {
            return unsupported(format!(
                "TilePrimitiveCall({op_name}): logical shape mismatch {:?} != {:?}",
                &expected, &shape
            ));
        }
    }
    Ok(())
}

pub(super) fn require_broadcastable_to(
    op_name: &str,
    destination: &TileRegion,
    sources: &[&TileRegion],
) -> AResult<()> {
    let destination_shape = &destination.extents;
    for source in sources {
        let source_shape = &source.extents;
        if source_shape.len() > destination_shape.len() {
            return unsupported(format!(
                "TilePrimitiveCall({op_name}): source rank {} exceeds destination rank {} for broadcast: {:?} -> {:?}",
                source_shape.len(),
                destination_shape.len(),
                source_shape,
                destination_shape));
        }
        let rank_padding = destination_shape.len() - source_shape.len();
        for (axis, source_extent) in source_shape.iter().enumerate() {
            let destination_extent = destination_shape[rank_padding + axis];
            if *source_extent != 1 && *source_extent != destination_extent {
                return unsupported(format!(
                    "TilePrimitiveCall({op_name}): source shape {:?} cannot right-align broadcast to {:?}; source axis {axis} has extent {source_extent}, expected 1 or {destination_extent}",
                    source_shape,
                    destination_shape));
            }
        }
    }
    Ok(())
}

pub(super) fn require_float_regions(op_name: &str, regions: &[&TileRegion]) -> AResult<()> {
    let bad: Vec<String> = regions
        .iter()
        .filter(|region| !is_float_dtype(&region.dtype))
        .map(|region| region.dtype.clone())
        .collect();
    if !bad.is_empty() {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): native tile numeric lowering requires float16/bfloat16/float32, got {:?}",
            &bad));
    }
    Ok(())
}

pub(super) fn validate_elementwise_storage(
    op_name: &str,
    regions: &[&TileRegion],
) -> AResult<String> {
    let scopes = sorted_strings(
        &regions
            .iter()
            .map(|region| region.memory_scope.clone())
            .collect::<Vec<_>>(),
    );
    if scopes != ["local"] && scopes != ["shared"] {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): elementwise operands must all reside in local or all reside in shared memory, got {:?}",
            &scopes));
    }
    Ok(scopes[0].clone())
}
