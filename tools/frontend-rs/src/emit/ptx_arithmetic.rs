//! Validation and emission support for ptx_arithmetic.

use crate::analyze::util::{dtype_of, unsupported, AResult};
use crate::decode::ptx::{unmodeled_form, DecodedPtx};
use crate::decode::Decoded;
use crate::emit::ptx_compare::required_marker;
use crate::emit::register_call::{require_register_call, RegisterCall, PTX_TYPE_MARKERS};
use crate::emit::{Emitter, RustValue};

const MIXED_VECTOR_UP_CALLS: [&str; 2] = ["tirx.ptx.add_mixed_vec_up", "tirx.ptx.sub_mixed_vec_up"];
const MIXED_VECTOR_DOWN_CALLS: [&str; 6] = [
    "tirx.ptx.add_mixed_vec_down_f16",
    "tirx.ptx.add_mixed_vec_down_bf16",
    "tirx.ptx.sub_mixed_vec_down_f16",
    "tirx.ptx.sub_mixed_vec_down_bf16",
    "tirx.ptx.mul_mixed_vec_down_f16",
    "tirx.ptx.mul_mixed_vec_down_bf16",
];
const MIXED_VECTOR_CROSS_MUL_CALLS: [&str; 2] = [
    "tirx.ptx.mul_mixed_vec_bf16_f16",
    "tirx.ptx.mul_mixed_vec_f16_bf16",
];

fn is_mixed_vector_call(name: &str) -> bool {
    MIXED_VECTOR_UP_CALLS.contains(&name)
        || MIXED_VECTOR_DOWN_CALLS.contains(&name)
        || MIXED_VECTOR_CROSS_MUL_CALLS.contains(&name)
        || name == "tirx.ptx.fma_mixed_vec"
}

fn instruction_of(op_name: &str) -> &'static str {
    match op_name {
        "tirx.ptx.add"
        | "tirx.ptx.add_half"
        | "tirx.ptx.add_int"
        | "tirx.ptx.add_mixed_vec_up"
        | "tirx.ptx.add_mixed_vec_down_f16"
        | "tirx.ptx.add_mixed_vec_down_bf16" => "add",
        "tirx.ptx.mul_int"
        | "tirx.ptx.mul"
        | "tirx.ptx.mul_half"
        | "tirx.ptx.mul_mixed_vec_down_f16"
        | "tirx.ptx.mul_mixed_vec_down_bf16"
        | "tirx.ptx.mul_mixed_vec_bf16_f16"
        | "tirx.ptx.mul_mixed_vec_f16_bf16" => "mul",
        "tirx.ptx.mad_int" | "tirx.ptx.mad_f" => "mad",
        "tirx.ptx.sub_int"
        | "tirx.ptx.sub"
        | "tirx.ptx.sub_half"
        | "tirx.ptx.sub_mixed_vec_up"
        | "tirx.ptx.sub_mixed_vec_down_f16"
        | "tirx.ptx.sub_mixed_vec_down_bf16" => "sub",
        "tirx.ptx.fma" | "tirx.ptx.fma_half" | "tirx.ptx.fma_mixed_vec" => "fma",
        "tirx.ptx.copysign" => "copysign",
        "tirx.ptx.div_f" => "div",
        _ => unreachable!("arithmetic family op name"),
    }
}

fn round_marker(rounding: &str) -> Option<&'static str> {
    match rounding {
        "" | "rn" => Some("Rn"),
        "rz" => Some("Rz"),
        "rm" => Some("Rm"),
        "rp" => Some("Rp"),
        _ => None,
    }
}

/// `repr(dict(call.modifiers))`.
fn modifiers_repr(decoded: &DecodedPtx) -> String {
    let items: Vec<String> = decoded
        .modifiers
        .iter()
        .map(|(slot, token)| format!("{:?}: {:?}", slot, token))
        .collect();
    format!("{{{}}}", items.join(", "))
}

/// `set(modifiers) == {...}`.
fn modifier_slots_are(decoded: &DecodedPtx, expected: &[&str]) -> bool {
    let mut actual: Vec<&str> = decoded
        .modifiers
        .iter()
        .map(|(slot, _)| slot.as_str())
        .collect();
    actual.sort_unstable();
    actual.dedup();
    let mut expected: Vec<&str> = expected.to_vec();
    expected.sort_unstable();
    actual == expected
}

/// The one closed engine signature for a PTX 9.4 mixed-vector form.
struct MixedVectorSignature {
    source_names: &'static [&'static str],
    expected_destination: &'static str,
    expected_sources: Vec<&'static str>,
    variant: String,
    argument_rust_types: Vec<&'static str>,
    result_rust_type: &'static str,
}

fn mixed_vector_signature(decoded: &DecodedPtx) -> AResult<MixedVectorSignature> {
    let op_name = decoded.op_name.as_str();
    let dtype = decoded.modifier_or_empty("dtype");
    let atype = decoded.modifier_or_empty("atype");
    let ctype = decoded.modifier_or_empty("ctype");
    let rounding = decoded.modifier_or_empty("rnd");
    let ftz = decoded.modifier_or_empty("ftz");

    if MIXED_VECTOR_UP_CALLS.contains(&op_name) {
        if !modifier_slots_are(decoded, &["rnd", "dtype", "atype", "ctype"])
            || round_marker(rounding).is_none()
            || dtype != "f32x2"
            || !(atype == "f16x2" || atype == "bf16x2")
            || ctype != "f32x2"
        {
            return unmodeled_form(
                decoded,
                &format!(
                    "has unsupported mixed-vector modifiers {}",
                    modifiers_repr(decoded)
                ),
            );
        }
        let source = if atype == "f16x2" { "F16x2" } else { "Bf16x2" };
        let variant = format!(
            "v2::reg::variant::MixedF32x2<v2::reg::variant::{source}, v2::reg::variant::{}>",
            round_marker(rounding).expect("validated rounding")
        );
        return Ok(MixedVectorSignature {
            source_names: &["a", "c"],
            expected_destination: "uint64",
            expected_sources: vec!["uint32", "uint64"],
            variant,
            argument_rust_types: vec!["u32", "u64"],
            result_rust_type: "u64",
        });
    }

    if op_name == "tirx.ptx.fma_mixed_vec" {
        if !modifier_slots_are(decoded, &["rnd", "dtype", "atype", "btype", "ctype"])
            || !matches!(rounding, "rn" | "rz" | "rm" | "rp")
            || dtype != "f32x2"
            || !(atype == "f16x2" || atype == "bf16x2")
            || decoded.modifier_or_empty("btype") != "f32x2"
            || ctype != "f32x2"
        {
            return unmodeled_form(
                decoded,
                &format!(
                    "has unsupported mixed-vector modifiers {}",
                    modifiers_repr(decoded)
                ),
            );
        }
        let source = if atype == "f16x2" { "F16x2" } else { "Bf16x2" };
        let variant = format!(
            "v2::reg::variant::MixedF32x2<v2::reg::variant::{source}, v2::reg::variant::{}>",
            round_marker(rounding).expect("validated rounding")
        );
        return Ok(MixedVectorSignature {
            source_names: &["a", "b", "c"],
            expected_destination: "uint64",
            expected_sources: vec!["uint32", "uint64", "uint64"],
            variant,
            argument_rust_types: vec!["u32", "u64", "u64"],
            result_rust_type: "u64",
        });
    }

    if MIXED_VECTOR_DOWN_CALLS.contains(&op_name) {
        let is_f16 = op_name.ends_with("_f16");
        let expected_keys: &[&str] = if is_f16 {
            &["rnd", "dtype", "atype", "ctype", "ftz"]
        } else {
            &["rnd", "dtype", "atype", "ctype"]
        };
        if !modifier_slots_are(decoded, expected_keys)
            || rounding != "rz"
            || ftz != (if is_f16 { "ftz" } else { "" })
            || dtype != (if is_f16 { "f16x2" } else { "bf16x2" })
            || atype != "f32x2"
            || ctype != "f32x2"
        {
            return unmodeled_form(
                decoded,
                &format!(
                    "has unsupported mixed-vector modifiers {}",
                    modifiers_repr(decoded)
                ),
            );
        }
        let destination = if is_f16 { "F16x2" } else { "Bf16x2" };
        return Ok(MixedVectorSignature {
            source_names: &["a", "c"],
            expected_destination: "uint32",
            expected_sources: vec!["uint64", "uint64"],
            variant: format!("v2::reg::variant::MixedF32x2Down<v2::reg::variant::{destination}>"),
            argument_rust_types: vec!["u64", "u64"],
            result_rust_type: "u32",
        });
    }

    let (expected_dtype, expected_atype, expected_ctype, destination, source) =
        if op_name == "tirx.ptx.mul_mixed_vec_bf16_f16" {
            ("bf16x2", "bf16x2", "f16x2", "Bf16x2", "F16x2")
        } else {
            ("f16x2", "f16x2", "bf16x2", "F16x2", "Bf16x2")
        };
    if !modifier_slots_are(decoded, &["dtype", "atype", "ctype"])
        || (dtype, atype, ctype) != (expected_dtype, expected_atype, expected_ctype)
    {
        return unmodeled_form(
            decoded,
            &format!(
                "has unsupported mixed-vector modifiers {}",
                modifiers_repr(decoded)
            ),
        );
    }
    Ok(MixedVectorSignature {
        source_names: &["a", "c"],
        expected_destination: "uint32",
        expected_sources: vec!["uint32", "uint32"],
        variant: format!(
            "v2::reg::variant::MixedLowMul<v2::reg::variant::{destination}, v2::reg::variant::{source}>"
        ),
        argument_rust_types: vec!["u32", "u32"],
        result_rust_type: "u32",
    })
}

fn owned(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

/// `resolve_ptx_arithmetic`.
pub fn resolve_ptx_arithmetic(decoded: &DecodedPtx) -> AResult<RegisterCall> {
    let op_name = decoded.op_name.as_str();
    let operands = require_register_call(decoded, false)?;
    if operands.destinations.len() != 1 {
        return unsupported(format!("{op_name} requires exactly one destination"));
    }
    let destination = operands.destinations[0].clone();

    let instruction = instruction_of(op_name);
    let mixed = if is_mixed_vector_call(op_name) {
        Some(mixed_vector_signature(decoded)?)
    } else {
        None
    };
    let source_names: &[&str] = match &mixed {
        Some(signature) => signature.source_names,
        None if instruction == "fma" || instruction == "mad" => &["a", "b", "c"],
        None => &["a", "b"],
    };
    let sources = operands.sources;
    if sources.len() != source_names.len() {
        return unsupported(format!(
            "{op_name} schema has {} sources, expected {}",
            sources.len(),
            source_names.len()
        ));
    }
    let ptx_type = decoded.modifier_or_empty("type");
    let source_type = decoded.modifier_or_empty("srctype");
    let rounding = decoded.modifier_or_empty("rnd");
    let ftz = decoded.modifier_or_empty("ftz") == "ftz";
    let sat = decoded.modifier_or_empty("sat") == "sat";
    let Some(round_marker) = round_marker(rounding) else {
        return unsupported(format!(
            "{op_name} has unsupported rounding modifier {:?}",
            rounding
        ));
    };

    let destination_dtype = dtype_of(&destination)?;
    let expected_destination: String;
    let expected_sources: Vec<String>;
    let variant: String;
    let argument_rust_types: Vec<String>;
    let result_rust_type: String;
    if let Some(signature) = mixed {
        expected_destination = signature.expected_destination.to_owned();
        expected_sources = owned(&signature.expected_sources);
        variant = signature.variant;
        argument_rust_types = owned(&signature.argument_rust_types);
        result_rust_type = signature.result_rust_type.to_owned();
    } else if op_name == "tirx.ptx.div_f" {
        let mode = decoded.modifier_or_empty("mode");
        if ptx_type == "f32" && matches!(mode, "rn" | "rz" | "rm" | "rp" | "approx" | "full") {
            let marker = match mode {
                "approx" => "Approx",
                "full" => "Full",
                _ => self::round_marker(mode).expect("directed rounding mode"),
            };
            let subnormal = if ftz { "Ftz" } else { "PreserveSubnormal" };
            variant = format!(
                "v2::reg::variant::F32Arithmetic<v2::reg::variant::{marker}, v2::reg::variant::{subnormal}>"
            );
        } else if ptx_type == "f64" && matches!(mode, "rn" | "rz" | "rm" | "rp") && !ftz {
            variant = format!(
                "v2::reg::variant::F64Arithmetic<v2::reg::variant::{}>",
                self::round_marker(mode).expect("directed rounding mode")
            );
        } else {
            return unmodeled_form(
                decoded,
                &format!(
                    "div.{mode}{}.{ptx_type} has no deterministic NumSim representative",
                    if ftz { ".ftz" } else { "" }
                ),
            );
        }
        expected_destination = if ptx_type == "f64" {
            "float64"
        } else {
            "float32"
        }
        .to_owned();
        expected_sources = vec![expected_destination.clone(); 2];
        argument_rust_types = vec![ptx_type.to_owned(); 2];
        result_rust_type = ptx_type.to_owned();
    } else if matches!(
        op_name,
        "tirx.ptx.add_int" | "tirx.ptx.sub_int" | "tirx.ptx.mul_int" | "tirx.ptx.mad_int"
    ) {
        // The decoder checks the canonical table's type/modifier domain.
        let marker = required_marker(PTX_TYPE_MARKERS, ptx_type)?;
        let high = decoded.modifier_or_empty("mode") == "hi";
        let mut integer_variant = format!("v2::reg::variant::{marker}");
        if high || sat {
            let mode = if high { "Hi" } else { "Lo" };
            let clamp = if sat { "Sat" } else { "NoSat" };
            integer_variant = format!(
                "v2::reg::variant::IntegerArithmetic<{integer_variant}, v2::reg::variant::{mode}, v2::reg::variant::{clamp}>"
            );
        }
        variant = integer_variant;
        let packed = ptx_type.ends_with("x2");
        let signed = ptx_type.starts_with('s') && !packed;
        let width = if packed { "32" } else { &ptx_type[1..] };
        expected_destination = format!("{}{width}", if signed { "int" } else { "uint" });
        result_rust_type = format!("{}{width}", if signed { "i" } else { "u" });
        expected_sources = vec![expected_destination.clone(); source_names.len()];
        argument_rust_types = vec![result_rust_type.clone(); source_names.len()];
    } else if matches!(
        op_name,
        "tirx.ptx.add_half" | "tirx.ptx.sub_half" | "tirx.ptx.mul_half" | "tirx.ptx.fma_half"
    ) {
        // The canonical TVM schema owns legality (e.g. no BF16 FTZ/sat,
        // and FMA's mutually exclusive sat/relu and FTZ/OOB forms).
        let marker = required_marker(PTX_TYPE_MARKERS, ptx_type)?;
        let subnormal = if ftz { "Ftz" } else { "PreserveSubnormal" };
        let clamp = if sat {
            "Sat"
        } else if !decoded.modifier_or_empty("relu").is_empty() {
            "Relu"
        } else {
            "NoSat"
        };
        let oob = !decoded.modifier_or_empty("oob").is_empty();
        variant = format!(
            "v2::reg::variant::HalfArithmetic<v2::reg::variant::{marker}, v2::reg::variant::{subnormal}, v2::reg::variant::{clamp}, {oob}>"
        );
        let packed = ptx_type.ends_with("x2");
        expected_destination = if packed { "uint32" } else { "uint16" }.to_owned();
        result_rust_type = if packed { "u32" } else { "u16" }.to_owned();
        expected_sources = vec![expected_destination.clone(); source_names.len()];
        argument_rust_types = vec![result_rust_type.clone(); source_names.len()];
    } else if !source_type.is_empty() {
        if ptx_type != "f32" {
            return unsupported(format!(
                "{op_name} mixed-precision sources require an f32 destination"
            ));
        }
        if !matches!(source_type, "f16" | "bf16") || !matches!(instruction, "add" | "sub" | "fma") {
            return unmodeled_form(
                decoded,
                &format!(
                    "{instruction}.{ptx_type}.{source_type} has no exact NumSim engine variant"
                ),
            );
        }
        expected_destination = "float32".to_owned();
        expected_sources = if instruction == "fma" {
            owned(&["uint16", "uint16", "float32"])
        } else {
            owned(&["uint16", "float32"])
        };
        if ftz {
            return unsupported(format!("{op_name} mixed-precision form cannot use ftz"));
        }
        let clamp = if sat { "Sat" } else { "NoSat" };
        variant = format!(
            "v2::reg::variant::MixedF32<v2::reg::variant::{}, v2::reg::variant::{round_marker}, v2::reg::variant::{clamp}>",
            required_marker(PTX_TYPE_MARKERS, source_type)?
        );
        argument_rust_types = if instruction == "fma" {
            owned(&["u16", "u16", "f32"])
        } else {
            owned(&["u16", "f32"])
        };
        result_rust_type = "f32".to_owned();
    } else if op_name == "tirx.ptx.copysign" {
        if !modifier_slots_are(decoded, &["type"]) || !(ptx_type == "f32" || ptx_type == "f64") {
            return unmodeled_form(
                decoded,
                &format!("copysign.{ptx_type} has no exact NumSim variant"),
            );
        }
        expected_destination = if ptx_type == "f32" {
            "float32"
        } else {
            "float64"
        }
        .to_owned();
        expected_sources = vec![expected_destination.clone(); 2];
        variant = format!(
            "v2::reg::variant::{}",
            if ptx_type == "f32" { "F32" } else { "F64" }
        );
        result_rust_type = if ptx_type == "f32" { "f32" } else { "f64" }.to_owned();
        argument_rust_types = vec![result_rust_type.clone(); 2];
    } else if ptx_type == "f32" {
        expected_destination = "float32".to_owned();
        expected_sources = vec![expected_destination.clone(); sources.len()];
        let subnormal = if ftz { "Ftz" } else { "PreserveSubnormal" };
        let clamp = if sat { "Sat" } else { "NoSat" };
        variant = format!(
            "v2::reg::variant::F32Arithmetic<v2::reg::variant::{round_marker}, v2::reg::variant::{subnormal}, v2::reg::variant::{clamp}>"
        );
        argument_rust_types = vec!["f32".to_owned(); sources.len()];
        result_rust_type = "f32".to_owned();
    } else if ptx_type == "f32x2" {
        expected_destination = "uint64".to_owned();
        expected_sources = vec![expected_destination.clone(); sources.len()];
        if sat {
            return unsupported(format!("{op_name} f32x2 form cannot use sat"));
        }
        let subnormal = if ftz { "Ftz" } else { "PreserveSubnormal" };
        variant = format!(
            "v2::reg::variant::F32x2Arithmetic<v2::reg::variant::{round_marker}, v2::reg::variant::{subnormal}>"
        );
        argument_rust_types = vec!["u64".to_owned(); sources.len()];
        result_rust_type = "u64".to_owned();
    } else if ptx_type == "f64" {
        expected_destination = "float64".to_owned();
        expected_sources = vec![expected_destination.clone(); sources.len()];
        if ftz || sat {
            return unsupported(format!("{op_name} f64 form cannot use ftz or sat"));
        }
        variant = format!("v2::reg::variant::F64Arithmetic<v2::reg::variant::{round_marker}>");
        argument_rust_types = vec!["f64".to_owned(); sources.len()];
        result_rust_type = "f64".to_owned();
    } else {
        return unsupported(format!(
            "{op_name} has unsupported arithmetic type {:?}",
            ptx_type
        ));
    }

    let mut actual_sources: Vec<String> = Vec::new();
    for source in &sources {
        actual_sources.push(dtype_of(source)?);
    }
    if destination_dtype != expected_destination || actual_sources != expected_sources {
        return unsupported(format!(
            "{op_name}.{ptx_type} expects {:?} -> {expected_destination}, got {:?} -> {destination_dtype}",
            &expected_sources,
            &actual_sources));
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
        "arithmetic",
        format!("{op_name} predicate must lower to bool or integer"),
    ))
}

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = resolve_ptx_arithmetic(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_register_call(decoded, &parts, source_op_id)?;
    Ok(None)
}
