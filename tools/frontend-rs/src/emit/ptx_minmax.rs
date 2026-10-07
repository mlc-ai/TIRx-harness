//! Validation and emission support for ptx_minmax.

use crate::analyze::util::{unmodeled, unsupported, AResult};
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::ptx_compare::required_marker;
use crate::emit::register_call::{require_register_call, RegisterCall, PTX_TYPE_MARKERS};
use crate::emit::{Emitter, RustValue};

/// `any(call.modifier(name) for name in names)` (short-circuit order kept).
fn any_modifier(decoded: &DecodedPtx, names: &[&str]) -> AResult<bool> {
    for name in names {
        if !decoded.modifier(name)?.is_empty() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `resolve_ptx_minmax`.
pub fn resolve_ptx_minmax(decoded: &DecodedPtx) -> AResult<RegisterCall> {
    let op_name = decoded.op_name.as_str();
    require_register_call(decoded, false)?;
    let destination = decoded.scalar_operand("d")?;
    let three_source = op_name == "tirx.ptx.max3" || op_name == "tirx.ptx.min3";
    let source_names: &[&str] = if three_source {
        &["a", "b", "c"]
    } else {
        &["a", "b"]
    };
    let mut sources = Vec::new();
    for name in source_names {
        sources.push(decoded.scalar_operand(name)?);
    }
    let instruction = if op_name == "tirx.ptx.max" || op_name == "tirx.ptx.max3" {
        "max"
    } else {
        "min"
    };
    let ptx_type = decoded.modifier("type")?;

    if three_source && ptx_type != "f32" {
        return unsupported(format!("{op_name} requires type f32"));
    }
    let variant: String;
    let rust_type: &'static str;
    if matches!(ptx_type, "f32" | "f16" | "f16x2" | "bf16" | "bf16x2") {
        let subnormal = if decoded.modifier("ftz")? == "ftz" {
            "Ftz"
        } else {
            "PreserveSubnormal"
        };
        let nan = if decoded.modifier("nan")? == "NaN" {
            "PropagateNan"
        } else {
            "IgnoreNan"
        };
        let absolute = !decoded.modifier("abs")?.is_empty();
        let xor_sign = !decoded.modifier_or_empty("xorsign").is_empty();
        let modifiers = format!("v2::reg::variant::{subnormal}, v2::reg::variant::{nan}");
        if ptx_type == "f32" {
            variant = format!(
                "v2::reg::variant::F32MinMax<{modifiers}, {absolute}, {xor_sign}, {}>",
                sources.len()
            );
            rust_type = "f32";
        } else {
            let marker = required_marker(PTX_TYPE_MARKERS, ptx_type)?;
            variant = format!(
                "v2::reg::variant::HalfMinMax<v2::reg::variant::{marker}, {modifiers}, {xor_sign}>"
            );
            rust_type = if ptx_type.ends_with("x2") {
                "u32"
            } else {
                "u16"
            };
        }
    } else if ptx_type == "f64" {
        if any_modifier(decoded, &["ftz", "nan", "xorsign", "abs", "relu"])? {
            return unsupported(format!("{op_name}.f64 accepts no modifiers"));
        }
        variant = "v2::reg::variant::F64".to_owned();
        rust_type = "f64";
    } else if matches!(
        ptx_type,
        "s16" | "s32" | "s64" | "u16" | "u32" | "u64" | "s16x2" | "u16x2"
    ) {
        let signed = ptx_type.starts_with('s');
        let packed = ptx_type.ends_with("x2");
        rust_type = match (packed, signed, &ptx_type[1..]) {
            (true, _, _) => "u32",
            (false, true, "16") => "i16",
            (false, true, "32") => "i32",
            (false, true, _) => "i64",
            (false, false, "16") => "u16",
            (false, false, "32") => "u32",
            (false, false, _) => "u64",
        };
        let marker = required_marker(PTX_TYPE_MARKERS, ptx_type)?;
        variant = if decoded.modifier("relu")?.is_empty() {
            format!("v2::reg::variant::{marker}")
        } else {
            format!(
                "v2::reg::variant::IntegerMinMax<v2::reg::variant::{marker}, v2::reg::variant::Relu>"
            )
        };
    } else {
        return unmodeled(
            format!("call:{op_name}"),
            format!("{op_name}.{ptx_type} has no reviewed NumSim register variant"),
        );
    }

    let argument_types = vec![rust_type; sources.len()];
    Ok(RegisterCall::new(
        destination,
        sources,
        instruction,
        variant,
        argument_types,
        rust_type,
    )
    .with_predicate_region(
        "minmax",
        format!("{op_name} predicate must lower to bool or integer"),
    ))
}

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = resolve_ptx_minmax(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_register_call(decoded, &parts, source_op_id)?;
    Ok(None)
}
