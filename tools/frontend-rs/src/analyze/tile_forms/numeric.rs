//! cast / unary / elementwise / reduction.

use tvm::ir::TensorRegionObj;
use tvm::ir::{FloatImmObj, PrimExpr, PrimType};
use tvm::prim::Cast;
use tvm::tirx::{Layout, TileLayout, TilePrimitiveCallObj};
use tvm::tvm_ffi::{
    self as tvm_ffi, Any, Array, DLDataType, DLDataTypeExt, ObjectRefCast, ObjectRefCore,
};

use super::super::layout::{axis_name, expr_any};
use super::super::util::{unsupported, AResult};
use super::super::Ctx;
use super::parse::{
    any_value, check_common, destination, is_float_dtype, literal_bool, literal_string, operand,
    require_arity, require_broadcastable_to, require_float_regions, sorted_strings, static_int_any,
    static_int_expr, unmodeled_tile_form, validate_elementwise_storage, AnyValue,
    ELEMENTWISE_EXEC_SCOPES,
};
use super::{ParsedTileCall, TileAttr, TileOpKind, TileOperand, TileRegion, TileScalar};
use crate::tables::is_integer_dtype;

const REDUCTION_EXEC_SCOPES: [&str; 4] = ["thread", "warp", "warpgroup", "cta"];
const TILE_SCALAR_DTYPES: [&str; 13] = [
    "float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8",
    "uint16", "uint32", "uint64", "bool",
];
const FILL_DTYPES: [&str; 14] = [
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
    "float8_e4m3fn",
];
const CAST_DTYPES: [&str; 15] = [
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
    "float8_e4m3fn",
    "float8_e8m0fnu",
];

fn is_arithmetic_dtype(dtype: &str) -> bool {
    is_float_dtype(dtype) || is_integer_dtype(dtype)
}

pub(super) fn resolve_cast(ctx: &Ctx, call: &TilePrimitiveCallObj) -> AResult<ParsedTileCall> {
    let analyzer = &ctx.analyzer;
    let op_name = "cast";
    require_arity(call, op_name, 2)?;
    let facts = check_common(call, op_name, &ELEMENTWISE_EXEC_SCOPES)?;
    if !facts.config.is_empty() {
        return unsupported(format!(
            "TilePrimitiveCall(cast): unsupported config keys {:?}",
            &sorted_strings(&facts.keys())
        ));
    }
    let destination = destination(analyzer, &call.args.get(0)?, op_name)?;
    let TileOperand::Region(source) = operand(analyzer, &call.args.get(1)?, "cast.src")? else {
        return unsupported("TilePrimitiveCall(cast): source must be a buffer");
    };
    require_broadcastable_to(op_name, &destination, &[&source])?;
    let unsupported_dtypes: Vec<String> = [&destination, &source]
        .iter()
        .filter(|region| !CAST_DTYPES.contains(&region.dtype.as_str()))
        .map(|region| region.dtype.clone())
        .collect();
    if !unsupported_dtypes.is_empty() {
        return unmodeled_tile_form(
            op_name,
            format!(
                "TilePrimitiveCall(cast): dtype is not implemented for {:?}",
                &unsupported_dtypes
            ),
        );
    }
    let storage_scope = validate_elementwise_storage(op_name, &[&destination, &source])?;
    Ok(ParsedTileCall::new(
        TileOpKind::Cast,
        &facts.scope,
        destination,
        vec![TileOperand::Region(source)],
        vec![("storage_scope", TileAttr::Str(storage_scope))],
    ))
}

fn is_unary_with_bias_scale(kind: TileOpKind) -> bool {
    matches!(
        kind,
        TileOpKind::Sqrt | TileOpKind::Exp | TileOpKind::Exp2 | TileOpKind::Log2
    )
}

fn widest_float_dtype(operands: &[&TileOperand]) -> AResult<String> {
    let mut widest = operands[0].dtype().to_owned();
    let mut widest_bits = DLDataType::try_from_str(&widest)?.bits;
    for operand in &operands[1..] {
        let bits = DLDataType::try_from_str(operand.dtype())?.bits;
        if bits > widest_bits {
            widest = operand.dtype().to_owned();
            widest_bits = bits;
        }
    }
    Ok(widest)
}

pub(super) fn resolve_unary_elementwise(
    ctx: &Ctx,
    call: &TilePrimitiveCallObj,
    kind: TileOpKind,
) -> AResult<ParsedTileCall> {
    let analyzer = &ctx.analyzer;
    let op_name = kind.value();
    let expected_arity = if is_unary_with_bias_scale(kind) { 4 } else { 2 };
    require_arity(call, op_name, expected_arity)?;
    let facts = check_common(call, op_name, &ELEMENTWISE_EXEC_SCOPES)?;
    if !facts.config.is_empty() {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): unsupported config keys {:?}",
            &sorted_strings(&facts.keys())
        ));
    }
    let destination = destination(analyzer, &call.args.get(0)?, op_name)?;
    let supported_dtypes: &[&str] = if kind == TileOpKind::Fill {
        &FILL_DTYPES
    } else {
        &TILE_SCALAR_DTYPES
    };
    if !supported_dtypes.contains(&destination.dtype.as_str()) {
        return unmodeled_tile_form(
            op_name,
            format!(
                "TilePrimitiveCall({op_name}): destination dtype {} is not implemented",
                destination.dtype
            ),
        );
    }

    if kind == TileOpKind::Fill {
        let TileOperand::Scalar(mut value) = operand(analyzer, &call.args.get(1)?, "fill.value")?
        else {
            return unsupported("TilePrimitiveCall(fill): value must be a typed scalar");
        };
        if !supported_dtypes.contains(&value.dtype.as_str()) {
            return unmodeled_tile_form(
                op_name,
                format!(
                    "TilePrimitiveCall(fill): scalar dtype {} is not implemented",
                    value.dtype
                ),
            );
        }
        if value.dtype != destination.dtype {
            let expr = PrimExpr::try_from(value.expr.clone())?;
            let cast: PrimExpr = Cast::new(PrimType::new(&destination.dtype)?, expr)?.into();
            value = TileScalar {
                expr: expr_any(&cast),
                dtype: destination.dtype.clone(),
            };
        }
        let storage_scope = validate_elementwise_storage(op_name, &[&destination])?;
        return Ok(ParsedTileCall::new(
            kind,
            &facts.scope,
            destination,
            vec![TileOperand::Scalar(value)],
            vec![
                ("storage_scope", TileAttr::Str(storage_scope)),
                ("rounding_mode", TileAttr::Str("rn".to_owned())),
            ],
        ));
    }

    let TileOperand::Region(source) =
        operand(analyzer, &call.args.get(1)?, &format!("{op_name}.src"))?
    else {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): source must be a buffer"
        ));
    };
    require_broadcastable_to(op_name, &destination, &[&source])?;
    let storage_scope = validate_elementwise_storage(op_name, &[&destination, &source])?;

    if kind == TileOpKind::Zero {
        if !TILE_SCALAR_DTYPES.contains(&source.dtype.as_str()) {
            return unmodeled_tile_form(
                op_name,
                format!(
                    "TilePrimitiveCall(zero): source dtype {} is not implemented",
                    source.dtype
                ),
            );
        }
        // CUDA zero accepts a source region for dispatch/layout selection but its
        // numerical definition does not read it.
        return Ok(ParsedTileCall::new(
            kind,
            &facts.scope,
            destination,
            Vec::new(),
            vec![
                ("storage_scope", TileAttr::Str(storage_scope)),
                ("rounding_mode", TileAttr::Str("rn".to_owned())),
            ],
        ));
    }

    let mut regions: Vec<TileRegion> = vec![destination.clone(), source.clone()];
    let mut operands: Vec<TileOperand> = vec![TileOperand::Region(source.clone())];
    let mut scale_operand: Option<TileScalar> = None;
    let mut bias_operand: Option<TileOperand> = None;
    let mut has_scale = false;
    let mut has_bias = false;
    if is_unary_with_bias_scale(kind) {
        let bias_value = call.args.get(2)?;
        let scale_value = call.args.get(3)?;
        if !matches!(any_value(&scale_value)?, AnyValue::None) {
            let is_float_imm = matches!(any_value(&scale_value)?, AnyValue::Object(node) if node.as_node::<FloatImmObj>().is_some());
            if !is_float_imm {
                return unsupported(format!(
                    "TilePrimitiveCall({op_name}): scale must be a static FloatImm"
                ));
            }
            let TileOperand::Scalar(scale) =
                operand(analyzer, &scale_value, &format!("{op_name}.scale"))?
            else {
                return unsupported(format!(
                    "TilePrimitiveCall({op_name}): scale must be a typed scalar"
                ));
            };
            operands.push(TileOperand::Scalar(scale.clone()));
            scale_operand = Some(scale);
            has_scale = true;
        }
        if !matches!(any_value(&bias_value)?, AnyValue::None) {
            let accepted = matches!(any_value(&bias_value)?, AnyValue::Object(node)
                if node.as_node::<TensorRegionObj>().is_some() || node.as_node::<FloatImmObj>().is_some());
            if !accepted {
                return unsupported(format!(
                    "TilePrimitiveCall({op_name}): bias must be a buffer region or static FloatImm"
                ));
            }
            let bias = operand(analyzer, &bias_value, &format!("{op_name}.bias"))?;
            if let TileOperand::Region(region) = &bias {
                require_broadcastable_to(op_name, &destination, &[region])?;
                regions.push(region.clone());
            }
            operands.push(bias.clone());
            bias_operand = Some(bias);
            has_bias = true;
        }
    }

    let region_refs: Vec<&TileRegion> = regions.iter().collect();
    require_float_regions(op_name, &region_refs)?;
    let bad: Vec<String> = operands
        .iter()
        .filter_map(TileOperand::scalar)
        .filter(|scalar| !is_float_dtype(&scalar.dtype))
        .map(|scalar| scalar.dtype.clone())
        .collect();
    if !bad.is_empty() {
        return unmodeled_tile_form(
            op_name,
            format!(
                "TilePrimitiveCall({op_name}): scalar dtype is not implemented, got {:?}",
                &bad
            ),
        );
    }
    let mut compute_operands: Vec<TileOperand> = regions
        .iter()
        .map(|region| TileOperand::Region(region.clone()))
        .collect();
    if let Some(TileOperand::Scalar(bias)) = &bias_operand {
        compute_operands.push(TileOperand::Scalar(bias.clone()));
    }
    let compute_refs: Vec<&TileOperand> = compute_operands.iter().collect();
    let compute_dtype = widest_float_dtype(&compute_refs)?;
    let bias_scalar = match &bias_operand {
        Some(TileOperand::Scalar(scalar)) => Some(scalar.clone()),
        _ => None,
    };
    for (label, scalar) in [("scale", &scale_operand), ("bias", &bias_scalar)] {
        if let Some(scalar) = scalar {
            if scalar.dtype != compute_dtype {
                return unsupported(format!(
                    "TilePrimitiveCall({op_name}): {label} dtype {} does not match compute dtype {compute_dtype}",
                    scalar.dtype
                ));
            }
        }
    }

    let storage_scope = validate_elementwise_storage(op_name, &region_refs)?;
    Ok(ParsedTileCall::new(
        kind,
        &facts.scope,
        destination,
        operands,
        vec![
            ("storage_scope", TileAttr::Str(storage_scope)),
            ("rounding_mode", TileAttr::Str("rn".to_owned())),
            ("has_scale", TileAttr::Bool(has_scale)),
            ("has_bias", TileAttr::Bool(has_bias)),
        ],
    ))
}

pub(super) fn resolve_elementwise(
    ctx: &Ctx,
    call: &TilePrimitiveCallObj,
    kind: TileOpKind,
) -> AResult<ParsedTileCall> {
    let analyzer = &ctx.analyzer;
    let op_name = kind.value();
    require_arity(call, op_name, if kind == TileOpKind::Fma { 4 } else { 3 })?;
    let facts = check_common(call, op_name, &ELEMENTWISE_EXEC_SCOPES)?;
    let unknown = facts.unknown_keys(&["rounding_mode"]);
    if !unknown.is_empty() {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): unsupported config keys {:?}",
            &unknown
        ));
    }
    let destination = destination(analyzer, &call.args.get(0)?, op_name)?;
    let mut operands: Vec<TileOperand> = Vec::new();
    for index in 1..call.args.len() {
        operands.push(operand(
            analyzer,
            &call.args.get(index)?,
            &format!("{op_name}.src{index}"),
        )?);
    }
    let regions: Vec<&TileRegion> = operands.iter().filter_map(TileOperand::region).collect();
    require_broadcastable_to(op_name, &destination, &regions)?;
    if matches!(kind, TileOpKind::Sub | TileOpKind::Fdiv) && operands[0].scalar().is_some() {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): the left operand must be a buffer region"
        ));
    }
    if matches!(
        kind,
        TileOpKind::Add
            | TileOpKind::Sub
            | TileOpKind::Mul
            | TileOpKind::Maximum
            | TileOpKind::Fdiv
    ) && operands.iter().all(|operand| operand.scalar().is_some())
    {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): both source operands cannot be scalars"
        ));
    }
    let integer_arithmetic = matches!(kind, TileOpKind::Add | TileOpKind::Sub | TileOpKind::Mul);
    if integer_arithmetic {
        let mut arithmetic_dtypes: Vec<String> = vec![destination.dtype.clone()];
        arithmetic_dtypes.extend(operands.iter().map(|operand| operand.dtype().to_owned()));
        let bad: Vec<String> = arithmetic_dtypes
            .iter()
            .filter(|dtype| !is_arithmetic_dtype(dtype))
            .cloned()
            .collect();
        if !bad.is_empty() {
            return unmodeled_tile_form(
                op_name,
                format!(
                    "TilePrimitiveCall({op_name}): arithmetic dtype is not implemented, got {:?}",
                    &bad
                ),
            );
        }
        let integer_dtypes = sorted_strings(
            &arithmetic_dtypes
                .iter()
                .filter(|dtype| is_integer_dtype(dtype))
                .cloned()
                .collect::<Vec<_>>(),
        );
        let has_float = arithmetic_dtypes.iter().any(|dtype| is_float_dtype(dtype));
        if !integer_dtypes.is_empty() && has_float {
            return unmodeled_tile_form(
                op_name,
                format!(
                    "TilePrimitiveCall({op_name}): mixed integer/float arithmetic is not implemented"
                ),
            );
        }
        if integer_dtypes.len() > 1 {
            return unmodeled_tile_form(
                op_name,
                format!(
                    "TilePrimitiveCall({op_name}): integer operands and destination must have one dtype, got {:?}",
                    &integer_dtypes),
            );
        }
    } else {
        let mut checked: Vec<&TileRegion> = vec![&destination];
        checked.extend(regions.iter().copied());
        let specialized: Vec<String> = checked
            .iter()
            .filter(|region| region.dtype.starts_with("float") && !is_float_dtype(&region.dtype))
            .map(|region| region.dtype.clone())
            .collect();
        if !specialized.is_empty() {
            return unmodeled_tile_form(
                op_name,
                format!(
                    "TilePrimitiveCall({op_name}): native tile numeric lowering requires float16/bfloat16/float32/float64, got {:?}",
                    &specialized),
            );
        }
        require_float_regions(op_name, &checked)?;
    }
    let mut storage_regions: Vec<&TileRegion> = vec![&destination];
    storage_regions.extend(regions.iter().copied());
    let storage_scope = validate_elementwise_storage(op_name, &storage_regions)?;
    for operand in &operands {
        let Some(scalar) = operand.scalar() else {
            continue;
        };
        let allowed = if integer_arithmetic {
            is_arithmetic_dtype(&scalar.dtype)
        } else {
            is_float_dtype(&scalar.dtype)
        };
        if !allowed {
            return unsupported(format!(
                "TilePrimitiveCall({op_name}): scalar dtype {} is not implemented",
                scalar.dtype
            ));
        }
    }
    let mut requested_rounding: Option<String> = None;
    if let Some(value) = facts.get("rounding_mode") {
        let mode = literal_string(value, &format!("{op_name}.rounding_mode"))?;
        if !["rn", "rm", "rp", "rz"].contains(&mode.as_str()) {
            return unsupported(format!(
                "TilePrimitiveCall({op_name}): unsupported rounding mode {:?}",
                &mode
            ));
        }
        requested_rounding = Some(mode);
    }
    let directed = requested_rounding
        .as_deref()
        .is_some_and(|mode| mode != "rn");
    if directed
        && (destination.dtype == "float64"
            || operands.iter().any(|operand| operand.dtype() == "float64"))
    {
        return unmodeled_tile_form(
            op_name,
            format!("TilePrimitiveCall({op_name}): directed float64 rounding is not implemented"),
        );
    }
    Ok(ParsedTileCall::new(
        kind,
        &facts.scope,
        destination,
        operands,
        vec![
            ("storage_scope", TileAttr::Str(storage_scope)),
            (
                "rounding_mode",
                TileAttr::Str(requested_rounding.unwrap_or_else(|| "rn".to_owned())),
            ),
        ],
    ))
}

fn thread_owner_axes(region: &TileRegion) -> AResult<Vec<String>> {
    let Some(layout) = region.buffer.buffer_type().layout.clone() else {
        return Ok(Vec::new());
    };
    if layout.clone().try_cast::<TileLayout>().is_err() {
        return Ok(Vec::new());
    }
    let canonical = layout.canonicalize()?.try_cast::<TileLayout>()?;
    let mut axes = Vec::new();
    for item in canonical.shard()?.iter() {
        let is_thread: Any = tvm_ffi::cached_global_func!("tirx.AxisIsThreadAxis")
            .call_tuple((item.axis.clone(),))?;
        if bool::try_from(is_thread)? {
            let name = axis_name(&item.axis)?;
            if !axes.contains(&name) {
                axes.push(name);
            }
        }
    }
    Ok(axes)
}

pub(super) fn resolve_reduction(
    ctx: &Ctx,
    call: &TilePrimitiveCallObj,
    kind: TileOpKind,
) -> AResult<ParsedTileCall> {
    let analyzer = &ctx.analyzer;
    let op_name = kind.value();
    require_arity(call, op_name, 4)?;
    let facts = check_common(call, op_name, &REDUCTION_EXEC_SCOPES)?;
    let unknown = facts.unknown_keys(&["thread_reduce"]);
    if !unknown.is_empty() {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): unsupported config keys {:?}",
            &unknown
        ));
    }
    let thread_reduce = match facts.get("thread_reduce") {
        Some(value) => literal_bool(value, &format!("{op_name}.thread_reduce"))?,
        None => false,
    };
    let destination = destination(analyzer, &call.args.get(0)?, op_name)?;
    let TileOperand::Region(source) =
        operand(analyzer, &call.args.get(1)?, &format!("{op_name}.src"))?
    else {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): source must be a buffer"
        ));
    };
    if destination.dtype == source.dtype
        && (source.dtype == "bool"
            || (source.dtype.starts_with("float") && !is_float_dtype(&source.dtype)))
    {
        return unmodeled_tile_form(
            op_name,
            format!(
                "TilePrimitiveCall({op_name}): source/destination must have the same supported numeric dtype, got {}/{}",
                source.dtype, destination.dtype
            ),
        );
    }
    if destination.dtype != source.dtype || !is_arithmetic_dtype(&source.dtype) {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): source/destination must have the same supported numeric dtype, got {}/{}",
            source.dtype, destination.dtype
        ));
    }
    let raw_axes_value = call.args.get(2)?;
    let raw_axes_array = Array::<Any>::try_from(raw_axes_value)?;
    let mut raw_axes: Vec<i64> = Vec::new();
    for value in raw_axes_array.iter() {
        raw_axes.push(static_int_any(
            analyzer,
            &value,
            &format!("{op_name}.axes"),
        )?);
    }
    let rank = source.extents.len() as i64;
    let mut axes: Vec<i64> = Vec::new();
    for raw_axis in raw_axes {
        let axis = if raw_axis < 0 {
            raw_axis + rank
        } else {
            raw_axis
        };
        if axis < 0 || axis >= rank {
            return unsupported(format!(
                "TilePrimitiveCall({op_name}): axis {raw_axis} is outside rank {rank}"
            ));
        }
        if !axes.contains(&axis) {
            axes.push(axis);
        }
    }
    let spatial_shape: Vec<i64> = source
        .extents
        .iter()
        .enumerate()
        .filter(|(axis, _)| !axes.contains(&(*axis as i64)))
        .map(|(_, extent)| *extent)
        .collect();
    let mut spatial_logical_shape: Vec<i64> = spatial_shape
        .iter()
        .copied()
        .filter(|extent| *extent != 1)
        .collect();
    if spatial_logical_shape.is_empty() {
        spatial_logical_shape = vec![1];
    }
    let expected_count: i64 = if spatial_shape.is_empty() {
        1
    } else {
        spatial_shape.iter().product()
    };
    let destination_shape = destination.logical_shape();
    let exact_shape = destination_shape == spatial_logical_shape;
    let replicated_shape = destination_shape.len() >= spatial_logical_shape.len()
        && destination_shape[..spatial_logical_shape.len()] == spatial_logical_shape[..]
        && destination.element_count() % expected_count == 0;
    if !exact_shape && !replicated_shape {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): destination logical shape {:?} does not match reduced shape {:?}",
            &destination.logical_shape(),
            &spatial_logical_shape));
    }
    let output_replication = destination.element_count() / expected_count;
    let accum = literal_bool(&call.args.get(3)?, &format!("{op_name}.accum"))?;
    let scopes = sorted_strings(&[
        destination.memory_scope.clone(),
        source.memory_scope.clone(),
    ]);
    if scopes.len() != 1 {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): reduction source/destination scopes must match, got {:?}",
            &scopes));
    }
    let storage_scope = scopes[0].clone();
    if storage_scope != "local" && storage_scope != "shared" {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): unsupported storage scope {:?}",
            &storage_scope
        ));
    }
    if storage_scope == "local" && facts.scope == "cta" {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): local reduction does not support CTA scope"
        ));
    }
    if thread_reduce && (storage_scope != "local" || facts.scope != "warp") {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): thread_reduce=True requires warp-scope local buffers"
        ));
    }
    let source_owner_axes = thread_owner_axes(&source)?;
    let destination_owner_axes = thread_owner_axes(&destination)?;
    let implicit_collective = storage_scope == "local"
        && facts.scope == "warp"
        && source_owner_axes
            .iter()
            .any(|axis| !destination_owner_axes.contains(axis));
    let collective = thread_reduce || implicit_collective;
    let mut collective_width: i64 = 32;
    if implicit_collective {
        let source_layout = source
            .buffer
            .buffer_type()
            .layout
            .clone()
            .and_then(|layout| layout.try_cast::<TileLayout>().ok());
        let destination_layout = destination
            .buffer
            .buffer_type()
            .layout
            .clone()
            .and_then(|layout| layout.try_cast::<TileLayout>().ok());
        let (Some(source_layout), Some(destination_layout)) = (source_layout, destination_layout)
        else {
            return unsupported(format!(
                "TilePrimitiveCall({op_name}): warp collective requires TileLayout operands"
            ));
        };
        // `TileLayout.is_swizzle()` is always False (only a ComposeLayout can be a
        // swizzle), so a swizzled-collective rejection is unreachable here.
        let source_canonical = Layout::from(source_layout)
            .canonicalize()?
            .try_cast::<TileLayout>()?;
        let destination_canonical = Layout::from(destination_layout)
            .canonicalize()?
            .try_cast::<TileLayout>()?;
        let mut source_lane_shards = Vec::new();
        for item in source_canonical.shard()?.iter() {
            if axis_name(&item.axis)? == "laneid" {
                source_lane_shards.push(item);
            }
        }
        let mut destination_lane_replicas = Vec::new();
        for item in destination_canonical.replica()?.iter() {
            if axis_name(&item.axis)? == "laneid" {
                destination_lane_replicas.push(item);
            }
        }
        if source_lane_shards.is_empty() || destination_lane_replicas.is_empty() {
            return unsupported(format!(
                "TilePrimitiveCall({op_name}): local warp collective requires a laneid shard-to-replica layout"
            ));
        }
        let mut source_lane_span: i64 = 1;
        for item in &source_lane_shards {
            let stride = static_int_expr(
                analyzer,
                &item.stride,
                &format!("{op_name}.source_lane_stride"),
            )?;
            let extent = static_int_expr(
                analyzer,
                &item.extent,
                &format!("{op_name}.source_lane_extent"),
            )?;
            source_lane_span += stride.abs() * (extent - 1);
        }
        if source_lane_span != 32 {
            return unsupported(format!(
                "TilePrimitiveCall({op_name}): local warp collective source lane span must be 32, got {source_lane_span}"
            ));
        }
        collective_width = 1;
        for item in &destination_lane_replicas {
            collective_width *= static_int_expr(
                analyzer,
                &item.extent,
                &format!("{op_name}.destination_lane_replica"),
            )?;
        }
        if ![1, 2, 4, 8, 16, 32].contains(&collective_width) {
            return unsupported(format!(
                "TilePrimitiveCall({op_name}): local warp collective width must be one of 1/2/4/8/16/32, got {collective_width}"
            ));
        }
    }
    let mut parsed = ParsedTileCall::new(
        kind,
        &facts.scope,
        destination,
        vec![TileOperand::Region(source)],
        vec![
            ("storage_scope", TileAttr::Str(storage_scope)),
            ("collective", TileAttr::Bool(collective)),
            ("collective_width", TileAttr::Int(collective_width)),
            ("thread_reduce", TileAttr::Bool(thread_reduce)),
            ("output_replication", TileAttr::Int(output_replication)),
        ],
    );
    parsed.axes = axes;
    parsed.accum = accum;
    Ok(parsed)
}
