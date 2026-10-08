//! Rust spelling tables shared by the analysis and the emitter.

use crate::analyze::util::{unsupported, AResult};
use crate::schema::Schema;

/// The engine load and store functions (below `v2::`), called by many lowerings.
pub const MEM_LD: &str = "mem::ld";
pub const MEM_ST: &str = "mem::st";

pub const INTEGER_RUST_TYPES: [&str; 8] = ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"];
const SIGNED_INTEGER_RUST_TYPES: [&str; 4] = ["i8", "i16", "i32", "i64"];

pub fn is_integer_rust_type(rust_type: &str) -> bool {
    INTEGER_RUST_TYPES.contains(&rust_type)
}

fn is_signed_integer_rust_type(rust_type: &str) -> bool {
    SIGNED_INTEGER_RUST_TYPES.contains(&rust_type)
}

pub fn stmt_rust_scalar_by_dtype(dtype: &str) -> Option<&'static str> {
    Some(match dtype {
        "handle" => "u64",
        "bool" => "bool",
        "int8" => "i8",
        "int16" => "i16",
        "int32" => "i32",
        "int64" => "i64",
        "uint8" => "u8",
        "uint16" => "u16",
        "uint32" => "u32",
        "uint64" => "u64",
        "uint128" => "U64x2",
        "int128" => "U64x2",
        "uint64x2" => "U64x2",
        "uint32x2" => "u64",
        "float16x2" => "u32",
        "bfloat16x2" => "u32",
        "float32x2" => "u64",
        "float32x4" => "F32x4",
        "float16" => "f32",
        "bfloat16" => "f32",
        "float32" => "f32",
        "float64" => "f64",
        "float8_e4m3fn" => "f32",
        "float8_e8m0fnu" => "f32",
        "float4_e2m1fn" => "f32",
        _ => return None,
    })
}

pub fn expr_rust_type_by_dtype(dtype: &str) -> Option<&'static str> {
    if dtype == "float4_e2m1fn" {
        return None;
    }
    stmt_rust_scalar_by_dtype(dtype)
}

pub fn expr_rust_type(schema: &Schema, dtype: &str) -> AResult<String> {
    if let Some(rust_type) = expr_rust_type_by_dtype(dtype) {
        return Ok(rust_type.to_owned());
    }
    match vector_rust_type(schema, dtype) {
        Some(rust_type) => Ok(rust_type),
        None => unsupported(format!(
            "Rust scalar lowering is not implemented for {dtype}"
        )),
    }
}

pub fn dtype_by_rust_type(rust_type: &str) -> Option<&'static str> {
    Some(match rust_type {
        "bool" => "bool",
        "i8" => "int8",
        "i16" => "int16",
        "i32" => "int32",
        "i64" => "int64",
        "u8" => "uint8",
        "u16" => "uint16",
        "u32" => "uint32",
        "u64" => "uint64",
        "U64x2" => "uint64x2",
        "f32" => "float32",
        "f64" => "float64",
        "F32x4" => "float32x4",
        _ => return None,
    })
}

pub fn is_low_precision_float(dtype: &str) -> bool {
    matches!(
        dtype,
        "float16" | "bfloat16" | "float8_e4m3fn" | "float8_e8m0fnu"
    )
}

pub fn scope_marker(scope: &str) -> &'static str {
    match scope {
        "cta" => "Cta",
        "cluster" => "Cluster",
        "gpu" => "Gpu",
        "sys" => "Sys",
        _ => unreachable!("validated scope"),
    }
}

pub fn round_marker(rounding_mode: &str) -> Option<&'static str> {
    Some(match rounding_mode {
        "rn" => "Rn",
        "rm" => "Rm",
        "rp" => "Rp",
        "rz" => "Rz",
        _ => return None,
    })
}

/// The Rust storage type of a fixed-width vector dtype, with the semantic spelling.
pub fn vector_rust_type(schema: &Schema, dtype: &str) -> Option<String> {
    let (_, _, _, total_bits) = schema.vector_dtype_abi(dtype)?;
    if dtype == "float32x4" {
        return Some("F32x4".to_owned());
    }
    packed_vector_rust_type(total_bits).map(str::to_owned)
}

fn packed_vector_rust_type(total_bits: i64) -> Option<&'static str> {
    Some(match total_bits {
        16 => "u16",
        32 => "u32",
        64 => "u64",
        128 => "U64x2",
        _ => return None,
    })
}

/// `dtype_abi.dtype_itemsize`.
pub fn dtype_itemsize(schema: &Schema, dtype: &str) -> Option<i64> {
    if let Some(bits) = schema.scalar_dtype_bits.get(dtype) {
        return Some(bits / 8);
    }
    schema
        .vector_dtype_abi(dtype)
        .map(|(_, _, _, total_bits)| total_bits / 8)
}

/// `dtype_itemsize`, rejecting a dtype without one.
pub fn dtype_byte_len(schema: &Schema, dtype: &str) -> AResult<i64> {
    match dtype_itemsize(schema, dtype) {
        Some(len) => Ok(len),
        None => unsupported(format!("raw pointer dtype {:?} is not implemented", dtype)),
    }
}

/// The v2 memory type of `dtype` without a PTX carrier.
pub fn v2_memory_type_rust(schema: &Schema, dtype: &str) -> AResult<String> {
    let marker = match dtype {
        "int8" => Some("v2::reg::variant::I8"),
        "int16" => Some("v2::reg::variant::I16"),
        "int32" => Some("v2::reg::variant::I32"),
        "int64" => Some("v2::reg::variant::I64"),
        "uint8" => Some("v2::reg::variant::U8"),
        "uint16" => Some("v2::reg::variant::U16"),
        "uint32" => Some("v2::reg::variant::U32"),
        "uint64" => Some("v2::reg::variant::U64"),
        "uint128" | "int128" => Some("v2::mem::variant::U64x2"),
        "float16" => Some("v2::mem::variant::F16"),
        "bfloat16" => Some("v2::mem::variant::Bf16"),
        "float32" => Some("v2::reg::variant::F32"),
        "float64" => Some("v2::reg::variant::F64"),
        "bool" => Some("v2::mem::variant::Bool"),
        "float32x4" => Some("v2::mem::variant::F32x4"),
        "uint64x2" => Some("v2::mem::variant::U64x2"),
        _ => None,
    };
    if let Some(marker) = marker {
        return Ok(marker.to_owned());
    }
    let carrier = vector_rust_type(schema, dtype);
    let marker = match carrier.as_deref() {
        Some("u16") => Some("v2::reg::variant::U16"),
        Some("u32") => Some("v2::reg::variant::U32"),
        Some("u64") => Some("v2::reg::variant::U64"),
        Some("U64x2") => Some("v2::mem::variant::U64x2"),
        Some("F32x4") => Some("v2::mem::variant::F32x4"),
        _ => None,
    };
    match marker {
        Some(marker) => Ok(marker.to_owned()),
        None => unsupported(format!("no v2 memory carrier for dtype {:?}", dtype)),
    }
}

pub struct IntegerBinarySemantics {
    pub integer_lowering: &'static str,
    pub rust_token: &'static str,
    pub float_lowering: Option<&'static str>,
}

/// The Rust lowering of one binary node kind, failing closed for unknown kinds.
pub fn integer_binary_semantics(kind: &str) -> AResult<IntegerBinarySemantics> {
    let (integer_lowering, rust_token, float_lowering) = match kind {
        "Add" => ("wrapping", "add", Some("operator")),
        "Sub" => ("wrapping", "sub", Some("operator")),
        "Mul" => ("wrapping", "mul", Some("operator")),
        "Div" => ("checked", "div", Some("operator")),
        "Mod" => ("checked", "rem", None),
        "FloorDiv" => ("floor", "div", None),
        "FloorMod" => ("floor", "rem", None),
        "Min" => ("minmax", "min", Some("minmax")),
        "Max" => ("minmax", "max", Some("minmax")),
        _ => {
            return unsupported(format!(
                "integer expression operation {kind} is not registered"
            ))
        }
    };
    Ok(IntegerBinarySemantics {
        integer_lowering,
        rust_token,
        float_lowering,
    })
}

fn error_expression(message: &str, error_context: &str) -> String {
    let literal = json_string(message);
    if error_context == "python" {
        format!("PyValueError::new_err({literal})")
    } else {
        format!("EngineError::message({literal})")
    }
}

/// Scalar bitwise code shared by ordinary expressions and live wait predicates.
/// TVM requires matching operand dtypes; Rust's shift methods take a u32 count.
pub fn render_bitwise(name: &str, operands: &[String]) -> String {
    match (name, operands) {
        ("bitwise_and", [a, b]) => format!("({a}) & ({b})"),
        ("bitwise_or", [a, b]) => format!("({a}) | ({b})"),
        ("bitwise_xor", [a, b]) => format!("({a}) ^ ({b})"),
        ("bitwise_not", [a]) => format!("!({a})"),
        ("shift_left", [a, b]) => format!("({a}).wrapping_shl(({b}) as u32)"),
        ("shift_right", [a, b]) => format!("({a}).wrapping_shr(({b}) as u32)"),
        _ => unreachable!("expected decoded scalar bitwise operands for {name}"),
    }
}

/// One integer binary node as Rust: `(code, requires_statement)`.
pub fn render_integer_binary(
    kind: &str,
    lhs: &str,
    rhs: &str,
    rust_type: &str,
    nonnegative_floor: bool,
    error_context: &str,
    error_label: Option<&str>,
) -> AResult<(String, bool)> {
    if !is_integer_rust_type(rust_type) {
        return unsupported(format!("{kind} currently requires an integer dtype"));
    }
    let semantics = integer_binary_semantics(kind)?;
    let token = semantics.rust_token;
    match semantics.integer_lowering {
        "wrapping" => return Ok((format!("({lhs}).wrapping_{token}({rhs})"), false)),
        "minmax" => return Ok((format!("({lhs}).{token}({rhs})"), false)),
        _ => {}
    }
    let operation = if token == "div" {
        "division"
    } else {
        "remainder"
    };
    if semantics.integer_lowering == "checked" {
        let message = error_label
            .map(str::to_owned)
            .unwrap_or_else(|| format!("invalid truncating {operation}"));
        let error = error_expression(&message, error_context);
        return Ok((
            format!("({lhs}).checked_{token}({rhs}).ok_or_else(|| {error})?"),
            true,
        ));
    }
    if is_signed_integer_rust_type(rust_type) {
        if nonnegative_floor {
            let operator = if token == "div" { "/" } else { "%" };
            return Ok((format!("({lhs}) {operator} ({rhs})"), true));
        }
        let helper = if token == "div" {
            "floor_div_i64"
        } else {
            "floor_mod_i64"
        };
        let helper_call = format!("{helper}(({lhs}) as i64, ({rhs}) as i64)");
        let mut code = if error_context == "engine" {
            format!("{helper_call}?")
        } else {
            let message = error_label
                .map(str::to_owned)
                .unwrap_or_else(|| format!("invalid signed floor {operation}"));
            format!(
                "{helper_call}.map_err(|error| PyValueError::new_err(format!(\"{{}}: {{}}\", {}, error)))?",
                json_string(&message)
            )
        };
        if rust_type != "i64" {
            code = format!("({code}) as {rust_type}");
        }
        return Ok((code, true));
    }
    let message = error_label
        .map(str::to_owned)
        .unwrap_or_else(|| format!("invalid unsigned floor {operation}"));
    let error = error_expression(&message, error_context);
    Ok((
        format!("({lhs}).checked_{token}({rhs}).ok_or_else(|| {error})?"),
        true,
    ))
}

/// One float binary node as Rust.
pub fn render_float_binary(kind: &str, lhs: &str, rhs: &str, rust_type: &str) -> AResult<String> {
    let semantics = integer_binary_semantics(kind)?;
    match semantics.float_lowering {
        Some("operator") => {
            let operator = match semantics.rust_token {
                "add" => "+",
                "sub" => "-",
                "mul" => "*",
                _ => "/",
            };
            Ok(format!("({lhs}) {operator} ({rhs})"))
        }
        Some("minmax") => Ok(if rust_type == "f32" {
            format!("cuda_f32_{}({lhs}, {rhs})", semantics.rust_token)
        } else {
            format!("({lhs}).{}({rhs})", semantics.rust_token)
        }),
        _ => unsupported(format!("{kind} currently requires an integer dtype")),
    }
}

/// `json.dumps(text)`.
pub fn json_string(text: &str) -> String {
    crate::analyze::util::Json::String(text.to_owned()).to_string()
}

/// The stable public key for one kernel-local binding (`host_abi.rs`).
pub fn phase_binding_name(kernel_index: i64, local_name: &str, kernel_count: i64) -> String {
    if kernel_count == 1 {
        local_name.to_owned()
    } else {
        format!("k{kernel_index}:{local_name}")
    }
}

pub fn is_integer_dtype(dtype: &str) -> bool {
    crate::dtypes::integer_dtypes().contains(&dtype)
}
