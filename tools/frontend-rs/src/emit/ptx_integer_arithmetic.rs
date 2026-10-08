//! Validation and emission support for ptx_integer_arithmetic.

use crate::analyze::util::{unsupported, AResult};
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::register_call::{require_register_call, RegisterCall};
use crate::emit::{Emitter, RustValue};

const INTEGER_TYPES: [&str; 6] = ["u16", "u32", "u64", "s16", "s32", "s64"];
const SIGNED_TYPES: [&str; 3] = ["s16", "s32", "s64"];
const WIDE_TYPES: [&str; 4] = ["u16", "u32", "s16", "s32"];
const TYPES_32: [&str; 2] = ["u32", "s32"];

/// `(dtype, rust_type, marker, bits)`.
fn type_info(ptx_type: &str, allowed: &[&str]) -> AResult<(String, String, String, i64)> {
    if !allowed.contains(&ptx_type) {
        return unsupported(format!("unsupported PTX integer type {:?}", ptx_type));
    }
    let signed = ptx_type.starts_with('s');
    let bits: i64 = ptx_type[1..]
        .parse()
        .expect("validated PTX integer type width");
    Ok((
        format!("{}{bits}", if signed { "int" } else { "uint" }),
        format!("{}{bits}", if signed { "i" } else { "u" }),
        format!("{}{bits}", if signed { "I" } else { "U" }),
        bits,
    ))
}

/// `resolve_ptx_integer_arithmetic`.
pub fn resolve_ptx_integer_arithmetic(decoded: &DecodedPtx) -> AResult<RegisterCall> {
    let op_name = decoded.op_name.as_str();
    let operands = require_register_call(decoded, false)?;
    if operands.destinations.len() != 1 {
        return unsupported(format!("{op_name} requires exactly one destination"));
    }
    let destination = operands.destinations[0].clone();

    let instruction: &'static str;
    let variant: String;
    let argument_rust_types: Vec<String>;
    let result_rust_type: String;
    if matches!(
        op_name,
        "tirx.ptx.div" | "tirx.ptx.rem" | "tirx.ptx.sad" | "tirx.ptx.neg_int"
    ) {
        let allowed: &[&str] = if op_name == "tirx.ptx.neg_int" {
            &SIGNED_TYPES
        } else {
            &INTEGER_TYPES
        };
        let (_dtype, rust_type, marker, _bits) = type_info(decoded.modifier("type")?, allowed)?;
        argument_rust_types = vec![rust_type.clone(); operands.sources.len()];
        instruction = match op_name {
            "tirx.ptx.div" => "div",
            "tirx.ptx.rem" => "rem",
            "tirx.ptx.sad" => "sad",
            _ => "neg",
        };
        variant = format!("v2::reg::variant::{marker}");
        result_rust_type = rust_type;
    } else if op_name == "tirx.ptx.mul_wide" || op_name == "tirx.ptx.mad_wide" {
        if decoded.modifier("mode")? != "wide" {
            return unsupported(format!("{op_name} requires the wide mode"));
        }
        let ptx_type = decoded.modifier("type")?;
        let (_source_dtype, source_rust_type, marker, bits) = type_info(ptx_type, &WIDE_TYPES)?;
        let signed = ptx_type.starts_with('s');
        result_rust_type = format!("{}{}", if signed { "i" } else { "u" }, bits * 2);
        if op_name == "tirx.ptx.mul_wide" {
            argument_rust_types = vec![source_rust_type.clone(), source_rust_type];
            instruction = "mul";
            variant = format!("v2::reg::variant::MulWide<v2::reg::variant::{marker}>");
        } else {
            argument_rust_types = vec![
                source_rust_type.clone(),
                source_rust_type,
                result_rust_type.clone(),
            ];
            instruction = "mad";
            variant = format!("v2::reg::variant::MadWide<v2::reg::variant::{marker}>");
        }
    } else if op_name == "tirx.ptx.mul24" || op_name == "tirx.ptx.mad24" {
        let mode = decoded.modifier("mode")?;
        if !(mode == "hi" || mode == "lo") {
            return unsupported(format!("{op_name} has unsupported mode {:?}", mode));
        }
        let ptx_type = decoded.modifier("type")?;
        let (_dtype, rust_type, marker, _bits) = type_info(ptx_type, &TYPES_32)?;
        let mode_marker = if mode == "hi" { "Hi" } else { "Lo" };
        if op_name == "tirx.ptx.mul24" {
            instruction = "mul24";
            variant = format!(
                "v2::reg::variant::Mul24<v2::reg::variant::{marker}, v2::reg::variant::{mode_marker}>"
            );
        } else {
            let saturation = decoded.modifier("sat")?;
            if !(saturation.is_empty() || saturation == "sat")
                || (saturation == "sat" && (mode, ptx_type) != ("hi", "s32"))
            {
                return unsupported(format!("{op_name} .sat applies only to the hi.s32 form"));
            }
            instruction = "mad24";
            let clamp = if saturation.is_empty() {
                "NoSat"
            } else {
                "Sat"
            };
            variant = format!(
                "v2::reg::variant::Mad24<v2::reg::variant::{marker}, v2::reg::variant::{mode_marker}, v2::reg::variant::{clamp}>"
            );
        }
        argument_rust_types = vec![rust_type.clone(); operands.sources.len()];
        result_rust_type = rust_type;
    } else {
        let atype = decoded.modifier("atype")?;
        let (_a_dtype, a_rust, a_marker, _bits) = type_info(atype, &TYPES_32)?;
        let btype = decoded.modifier("btype")?;
        let (_b_dtype, b_rust, b_marker, _bits) = type_info(btype, &TYPES_32)?;
        let unsigned_accumulator = atype == "u32" && btype == "u32";
        result_rust_type = if unsigned_accumulator { "u32" } else { "i32" }.to_owned();
        argument_rust_types = vec![a_rust, b_rust, result_rust_type.clone()];
        if op_name == "tirx.ptx.dp2a" {
            let mode = decoded.modifier("mode")?;
            if !(mode == "hi" || mode == "lo") {
                return unsupported(format!("{op_name} has unsupported mode {:?}", mode));
            }
            let mode_marker = if mode == "hi" { "Hi" } else { "Lo" };
            instruction = "dp2a";
            variant = format!(
                "v2::reg::variant::Dp2a<v2::reg::variant::{a_marker}, v2::reg::variant::{b_marker}, v2::reg::variant::{mode_marker}>"
            );
        } else {
            instruction = "dp4a";
            variant = format!(
                "v2::reg::variant::Dp4a<v2::reg::variant::{a_marker}, v2::reg::variant::{b_marker}>"
            );
        }
    }

    let sources = operands.sources;
    if sources.len() != argument_rust_types.len() {
        return unsupported(format!(
            "{op_name} schema has {} sources, expected {}",
            sources.len(),
            argument_rust_types.len()
        ));
    }
    Ok(RegisterCall::new(
        destination,
        sources,
        instruction,
        variant,
        argument_rust_types,
        result_rust_type,
    )
    .with_predicate_region(
        "integer_arithmetic",
        "integer arithmetic predicate must be bool or integer".to_owned(),
    ))
}

/// `resolve_ptx_integer_arithmetic`, keeping the validated form.

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = resolve_ptx_integer_arithmetic(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_register_call(decoded, &parts, source_op_id)?;
    Ok(None)
}
