//! Validation and emission of the matrix instruction family.

use crate::analyze::util::{
    as_buffer, capitalize, dtype_of, ffi_error, int_imm_expr, not_covered, oref, prim, repr_text,
    same, simplify, static_int, static_string, unmodeled, unsupported, AResult, Failure,
};
use crate::analyze::Ctx;
use crate::decode::projected_buffer;
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::matrix_variants::{DenseMmaEngine, DenseMmaVariant, MmaTypes, SparseMmaVariant};
use crate::emit::register_call::require_register_call;
use crate::emit::{abi, Emitter, RustValue};
use crate::tables::is_integer_dtype;
use crate::tables::{dtype_byte_len, v2_memory_type_rust, MEM_LD, MEM_ST};
use crate::tvm_compat::int_value;
use tvm::ir::{CallObj, FloatImmObj, IntImm, IntImmObj, PrimExpr, TensorLoadObj, VarObj};
use tvm::tirx::BufferVar;
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

/// The engine functions (below `v2::`) the MMA lowerings call.
pub const MMA_SYNC: &str = "matrix::mma_sync";
pub const MMA_SP_SYNC: &str = "matrix::mma_sp_sync";

pub const PTX_DENSE_MMA_CALLS: [&str; 5] = [
    "tirx.ptx.mma",
    "tirx.ptx.mma_f16acc",
    "tirx.ptx.mma_f16c_f32d",
    "tirx.ptx.mma_int",
    "tirx.ptx.mma_f64",
];
pub const PTX_SPARSE_MMA_CALLS: [&str; 7] = [
    "tirx.ptx.mma_sp_f16acc",
    "tirx.ptx.mma_sp_f16acc_pair",
    "tirx.ptx.mma_sp_int_pair",
    "tirx.ptx.mma_sp_int_all",
    "tirx.ptx.mma_sp",
    "tirx.ptx.mma_sp_pair",
    "tirx.ptx.mma_sp_all",
];

const PTX_LEGACY_DTYPE: &[(&str, &str)] = &[
    ("fp16", "float16"),
    ("f16", "float16"),
    ("bf16", "bfloat16"),
    ("fp32", "float32"),
    ("f32", "float32"),
    ("fp64", "float64"),
    ("tf32", "tf32"),
    ("s8", "int8"),
    ("u8", "uint8"),
    ("s4", "int4"),
    ("u4", "uint4"),
    ("b1", "int1"),
    ("s32", "int32"),
    ("e4m3", "float8_e4m3fn"),
    ("e5m2", "float8_e5m2"),
];
const MMA_ACCUMULATOR_DTYPES: [&str; 4] = ["float16", "float32", "float64", "int32"];

const PTX_MATRIX_DTYPE: &[(&str, &str)] = &[
    ("f16", "float16"),
    ("bf16", "bfloat16"),
    ("f32", "float32"),
    ("f64", "float64"),
    ("tf32", "tf32"),
    ("s8", "int8"),
    ("u8", "uint8"),
    ("s4", "int4"),
    ("u4", "uint4"),
    ("b1", "int1"),
    ("s32", "int32"),
    ("e4m3", "float8_e4m3fn"),
    ("e5m2", "float8_e5m2"),
];

pub struct LegacyMatrixCall {
    pub op_name: String,
    pub args: Vec<ObjectRef>,
    pub m: Option<i64>,
    pub n: Option<i64>,
    pub k: Option<i64>,
    pub d_dtype: Option<String>,
    pub a_dtype: Option<String>,
    pub b_dtype: Option<String>,
    pub c_dtype: Option<String>,
    pub a_layout: Option<String>,
    pub b_layout: Option<String>,
    pub has_c: bool,
    pub saturate: bool,
    pub bit_op: Option<String>,
    pub local_size: Option<i64>,
}

impl LegacyMatrixCall {
    fn new(op_name: &str, args: Vec<ObjectRef>) -> Self {
        Self {
            op_name: op_name.to_owned(),
            args,
            m: None,
            n: None,
            k: None,
            d_dtype: None,
            a_dtype: None,
            b_dtype: None,
            c_dtype: None,
            a_layout: None,
            b_layout: None,
            has_c: false,
            saturate: false,
            bit_op: None,
            local_size: None,
        }
    }
}

fn legacy_static_bool(ctx: &Ctx, value: &ObjectRef, field: &str) -> AResult<bool> {
    let simplified = simplify(&ctx.analyzer, &prim(value)?)?;
    let is_bool =
        int_imm_expr(&simplified).is_some() && dtype_of(&oref(simplified.clone()))? == "bool";
    if !is_bool {
        return unsupported(format!(
            "{field} must specialize to bool, got {}",
            repr_text(value)?
        ));
    }
    Ok(int_imm_expr(&simplified).expect("bool immediate") != 0)
}

/// `re.fullmatch(r"m(\d+)n(\d+)k(\d+)", raw)`.
fn parse_mma_shape(raw: &str) -> Option<(i64, i64, i64)> {
    let rest = raw.strip_prefix('m')?;
    let (m, rest) = rest.split_once('n')?;
    let (n, k) = rest.split_once('k')?;
    let digits = |text: &str| -> Option<i64> {
        if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        text.parse().ok()
    };
    Some((digits(m)?, digits(n)?, digits(k)?))
}

fn legacy_shape(value: &ObjectRef, field: &str) -> AResult<(i64, i64, i64)> {
    let raw = static_string(value, field)?;
    let raw = raw.trim_start_matches('.');
    match parse_mma_shape(raw) {
        Some(shape) => Ok(shape),
        None => unsupported(format!("{field} must be mMxNxK, got {:?}", raw)),
    }
}

fn legacy_ptx_dtype(value: &ObjectRef, field: &str) -> AResult<String> {
    let raw = static_string(value, field)?;
    let raw = raw.trim_start_matches('.');
    Ok(PTX_LEGACY_DTYPE
        .iter()
        .find(|(key, _)| *key == raw)
        .map_or_else(|| raw.to_owned(), |(_, dtype)| (*dtype).to_owned()))
}

fn require_legacy_dtypes(op_name: &str, args: &[ObjectRef], expected: &[&[&str]]) -> AResult<()> {
    let mut actual = Vec::new();
    for argument in args {
        actual.push(dtype_of(argument)?);
    }
    if actual.len() != expected.len() {
        return unsupported(format!(
            "{op_name} expects {} arguments, got {}",
            expected.len(),
            actual.len()
        ));
    }
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        if !want.contains(&got.as_str()) {
            let mut allowed: Vec<&str> = want.to_vec();
            allowed.sort_unstable();
            return unsupported(format!(
                "{op_name} argument {index} must be one of {:?}, got {got}",
                &allowed
            ));
        }
    }
    Ok(())
}

pub fn resolve_legacy_matrix_call(
    ctx: &Ctx,
    node: &ObjectRef,
    op_name: &str,
) -> AResult<LegacyMatrixCall> {
    let Some(call) = node.as_node::<CallObj>() else {
        return Err(Failure::Ffi(ffi_error(&format!(
            "{op_name} lowering received a non-Call node"
        ))));
    };
    let args: Vec<ObjectRef> = call.args.iter().map(oref).collect();

    if op_name == "tirx.ptx_legacy.mma" {
        let result_dtype = dtype_of(node)?;
        if !result_dtype.is_empty() {
            return unsupported(format!("{op_name} must return void, got {result_dtype}"));
        }
        if args.len() != 13 && args.len() != 14 {
            return unsupported(format!(
                "{op_name} expects 13 or 14 arguments, got {}",
                args.len()
            ));
        }
        let integer_offset: &[&str] = &["int32", "int64"];
        require_legacy_dtypes(
            op_name,
            &args[..13],
            &[
                &[""],
                &[""],
                &[""],
                &[""],
                &[""],
                &[""],
                &["handle"],
                integer_offset,
                &["handle"],
                integer_offset,
                &["handle"],
                integer_offset,
                &["bool"],
            ],
        )?;
        let (m, n, k) = legacy_shape(&args[0], &format!("{op_name}.shape"))?;
        let bit_op = if args.len() == 13 {
            None
        } else {
            Some(static_string(&args[13], &format!("{op_name}.operator"))?)
        };
        let c_dtype = legacy_ptx_dtype(&args[5], &format!("{op_name}.c_dtype"))?;
        let a_layout = static_string(&args[1], &format!("{op_name}.a_layout"))?;
        let b_layout = static_string(&args[2], &format!("{op_name}.b_layout"))?;
        let a_dtype = legacy_ptx_dtype(&args[3], &format!("{op_name}.a_dtype"))?;
        let b_dtype = legacy_ptx_dtype(&args[4], &format!("{op_name}.b_dtype"))?;
        let saturate = legacy_static_bool(ctx, &args[12], &format!("{op_name}.saturate"))?;
        let mut parsed = LegacyMatrixCall::new(op_name, args);
        parsed.m = Some(m);
        parsed.n = Some(n);
        parsed.k = Some(k);
        parsed.a_layout = Some(a_layout);
        parsed.b_layout = Some(b_layout);
        parsed.d_dtype = Some(c_dtype.clone());
        parsed.a_dtype = Some(a_dtype);
        parsed.b_dtype = Some(b_dtype);
        parsed.c_dtype = Some(c_dtype);
        parsed.has_c = true;
        parsed.saturate = saturate;
        parsed.bit_op = bit_op;
        return Ok(parsed);
    }

    let result_dtype = dtype_of(node)?;
    if !MMA_ACCUMULATOR_DTYPES.contains(&result_dtype.as_str()) {
        return unsupported(format!(
            "{op_name} accumulator dtype must be one of {:?}, got {:?}",
            &MMA_ACCUMULATOR_DTYPES, &result_dtype
        ));
    }
    if op_name == "tirx.mma_fill" || op_name == "tirx.mma_fill_legacy" {
        if args.len() != 3 {
            return unsupported(format!("{op_name} expects 3 arguments, got {}", args.len()));
        }
        let local_size = static_int(
            &ctx.analyzer,
            &prim(&args[0])?,
            &format!("{op_name}.local_size"),
            "must specialize to an integer",
        )?;
        if local_size <= 0 {
            return unsupported(format!("{op_name}.local_size must be positive"));
        }
        if dtype_of(&args[1])? != "handle" || !is_integer_dtype(&dtype_of(&args[2])?) {
            return unsupported(format!(
                "{op_name} requires a handle destination and integer offset"
            ));
        }
        let mut parsed = LegacyMatrixCall::new(op_name, args);
        parsed.d_dtype = Some(result_dtype);
        parsed.local_size = Some(local_size);
        return Ok(parsed);
    }

    if args.len() != 6 {
        return unsupported(format!("{op_name} expects 6 arguments, got {}", args.len()));
    }
    let m = static_int(
        &ctx.analyzer,
        &prim(&args[0])?,
        &format!("{op_name}.m"),
        "must specialize to an integer",
    )?;
    let n = static_int(
        &ctx.analyzer,
        &prim(&args[1])?,
        &format!("{op_name}.n"),
        "must specialize to an integer",
    )?;
    if (m, n) != (16, 16) {
        return unsupported(format!("{op_name} supports only m=16, n=16"));
    }
    if dtype_of(&args[2])? != "handle" || dtype_of(&args[3])? != "handle" {
        return unsupported(format!("{op_name} source and destination must be handles"));
    }
    if !is_integer_dtype(&dtype_of(&args[4])?) || !is_integer_dtype(&dtype_of(&args[5])?) {
        return unsupported(format!(
            "{op_name} source offset and destination stride must be integers"
        ));
    }
    let mut parsed = LegacyMatrixCall::new(op_name, args);
    parsed.m = Some(m);
    parsed.n = Some(n);
    parsed.d_dtype = Some(result_dtype);
    Ok(parsed)
}

pub fn legacy_dense_variant(call: &LegacyMatrixCall) -> AResult<DenseMmaVariant> {
    let variant = (|| {
        MmaTypes {
            d: call.d_dtype.as_deref()?,
            a: call.a_dtype.as_deref()?,
            b: call.b_dtype.as_deref()?,
            c: call.c_dtype.as_deref()?,
        }
        .legacy_dense(
            (call.m?, call.n?, call.k?),
            (call.a_layout.as_deref()?, call.b_layout.as_deref()?),
            call.saturate,
            call.bit_op.as_deref(),
        )
    })();
    let Some(variant) = variant else {
        let render = |value: &Option<i64>| value.map_or("None".to_owned(), |v| v.to_string());
        let text = |value: &Option<String>| value.clone().unwrap_or_else(|| "None".to_owned());
        return unsupported(format!(
            "{} has no engine ABI for m{}n{}k{} {}.{} {}/{}/{}/{}",
            call.op_name,
            render(&call.m),
            render(&call.n),
            render(&call.k),
            text(&call.a_layout),
            text(&call.b_layout),
            text(&call.d_dtype),
            text(&call.a_dtype),
            text(&call.b_dtype),
            text(&call.c_dtype)
        ));
    };
    Ok(variant)
}

/// `_v2_matrix_type`.
pub fn v2_matrix_type(dtype: &str) -> AResult<&'static str> {
    Ok(match dtype {
        "float16" => "Fp16",
        "bfloat16" => "Bf16",
        "tf32" => "Tf32",
        "float32" => "Fp32",
        "int8" => "I8",
        "uint8" => "U8",
        "int4" => "I4",
        "uint4" => "U4",
        "float8_e4m3fn" => "E4M3",
        "float8_e5m2" => "E5M2",
        _ => {
            return Err(Failure::Ffi(ffi_error(&format!(
                "_v2_matrix_type has no marker for {dtype}"
            ))))
        }
    })
}

#[allow(clippy::too_many_arguments)]
pub fn v2_dense_variant_values(
    form: &DenseMmaVariant,
    m: i64,
    k: i64,
    d_dtype: &str,
    a_dtype: &str,
    b_dtype: &str,
    c_dtype: &str,
    a_layout: &str,
    b_layout: &str,
    has_c: bool,
    saturate: bool,
    bit_op: Option<&str>,
    rounding: &str,
) -> AResult<String> {
    let v = "v2::matrix::variant";
    let c = if has_c {
        format!("{v}::WithC")
    } else {
        format!("{v}::NoC")
    };
    Ok(match form.engine {
        DenseMmaEngine::F32B16 => {
            format!("{v}::F32B16<{k}, {v}::{}, {c}>", v2_matrix_type(a_dtype)?)
        }
        DenseMmaEngine::F16F16 => format!("{v}::F16<{k}, {c}>"),
        DenseMmaEngine::F32Tf32 => format!("{v}::F32Tf32<{k}, {c}>"),
        DenseMmaEngine::F16F8 | DenseMmaEngine::F32F8 => {
            let accumulator = if form.engine == DenseMmaEngine::F16F8 {
                "Fp16"
            } else {
                "Fp32"
            };
            format!(
                "{v}::F8<{k}, {v}::{accumulator}, {v}::{}, {v}::{}, {c}>",
                v2_matrix_type(a_dtype)?,
                v2_matrix_type(b_dtype)?
            )
        }
        DenseMmaEngine::F64 => {
            let mode = if rounding.is_empty() || rounding == "rn" {
                String::new()
            } else {
                format!(", v2::reg::variant::{}", capitalize(rounding))
            };
            format!("{v}::F64<{m}, {k}, {c}{mode}>")
        }
        DenseMmaEngine::PackedInteger => {
            if a_dtype == "int1" {
                let operation = match bit_op {
                    Some("xor") => "Xor",
                    Some("and") => "And",
                    _ => return Err(Failure::Ffi(ffi_error("b1 MMA without a validated bit op"))),
                };
                format!("{v}::Binary<{m}, {k}, {v}::{operation}, {c}>")
            } else {
                let family = if a_dtype == "int8" || a_dtype == "uint8" {
                    "PackedI8"
                } else {
                    "PackedI4"
                };
                format!(
                    "{v}::{family}<{m}, {k}, {v}::{}, {v}::{}, {c}, {}>",
                    v2_matrix_type(a_dtype)?,
                    v2_matrix_type(b_dtype)?,
                    if saturate { "true" } else { "false" }
                )
            }
        }
        DenseMmaEngine::M8n8k4F16 => {
            let accumulator = match (d_dtype, c_dtype) {
                ("float16", "float16") => "M8N8K4F16F16",
                ("float32", "float16") => "M8N8K4F32F16",
                ("float32", "float32") => "M8N8K4F32F32",
                _ => {
                    return Err(Failure::Ffi(ffi_error(
                        "m8n8k4 MMA without a validated accumulator pair",
                    )))
                }
            };
            let a_layout_marker = if a_layout == "row" { "Row" } else { "Col" };
            let b_layout_marker = if b_layout == "row" { "Row" } else { "Col" };
            format!("{v}::{accumulator}<{v}::{a_layout_marker}, {v}::{b_layout_marker}, {c}>")
        }
    })
}

pub fn v2_dense_variant(call: &LegacyMatrixCall, form: &DenseMmaVariant) -> AResult<String> {
    v2_dense_variant_values(
        form,
        call.m.expect("legacy m"),
        call.k.expect("legacy k"),
        call.d_dtype.as_deref().expect("legacy d_dtype"),
        call.a_dtype.as_deref().expect("legacy a_dtype"),
        call.b_dtype.as_deref().expect("legacy b_dtype"),
        call.c_dtype.as_deref().expect("legacy c_dtype"),
        call.a_layout.as_deref().expect("legacy a_layout"),
        call.b_layout.as_deref().expect("legacy b_layout"),
        call.has_c,
        call.saturate,
        call.bit_op.as_deref(),
        "",
    )
}

pub fn v2_sparse_variant_values(
    form: &SparseMmaVariant,
    k: i64,
    a_dtype: &str,
    b_dtype: &str,
    c_dtype: &str,
    saturate: bool,
    ordered_metadata: bool,
) -> AResult<String> {
    let v = "v2::matrix::variant";
    let ordered = if ordered_metadata { ", true" } else { "" };
    let saturate_text = if saturate { "true" } else { "false" };
    Ok(match a_dtype {
        "float16" => {
            let accumulator = if c_dtype == "float16" { "Fp16" } else { "Fp32" };
            format!("{v}::SparseF16<{k}, {v}::{accumulator}{ordered}>")
        }
        "bfloat16" => format!("{v}::SparseBf16<{k}{ordered}>"),
        "tf32" => format!("{v}::SparseTf32<{k}{ordered}>"),
        "int8" | "uint8" => format!(
            "{v}::SparseI8<{k}, {v}::{}, {v}::{}, {saturate_text}{ordered}>",
            v2_matrix_type(a_dtype)?,
            v2_matrix_type(b_dtype)?
        ),
        "int4" | "uint4" => format!(
            "{v}::SparseI4<{k}, {v}::{}, {v}::{}, {saturate_text}{ordered}>",
            v2_matrix_type(a_dtype)?,
            v2_matrix_type(b_dtype)?
        ),
        "float8_e4m3fn" | "float8_e5m2" => format!(
            "{v}::SparseF8<{v}::{}, {v}::{}{ordered}>",
            v2_matrix_type(a_dtype)?,
            v2_matrix_type(b_dtype)?
        ),
        _ => {
            return Err(Failure::Ffi(ffi_error(&format!(
                "unhandled sparse MMA engine {:?}",
                form.engine
            ))))
        }
    })
}

// ----------------------------------------------------------------------
// Decoded PTX warp MMA calls.
// ----------------------------------------------------------------------

fn decoded_mma_call(decoded: &DecodedPtx) -> AResult<()> {
    let op_name = decoded.op_name.as_str();
    require_register_call(decoded, false)?;
    for (name, expected) in [("sync", "sync"), ("aligned", "aligned")] {
        let actual = decoded.modifier(name)?;
        if actual != expected {
            return unsupported(format!(
                "{op_name} requires {name}={:?}, got {:?}",
                expected, actual
            ));
        }
    }
    Ok(())
}

fn decoded_mma_shape(decoded: &DecodedPtx) -> AResult<(i64, i64, i64)> {
    let token = decoded.modifier("shape")?;
    match parse_mma_shape(token) {
        Some(shape) => Ok(shape),
        None => unsupported(format!(
            "{} has malformed shape modifier {:?}",
            decoded.op_name, token
        )),
    }
}

fn decoded_mma_dtype(decoded: &DecodedPtx, modifier: &str) -> AResult<String> {
    let token = decoded.modifier(modifier)?;
    match PTX_MATRIX_DTYPE.iter().find(|(key, _)| *key == token) {
        Some((_, dtype)) => Ok((*dtype).to_owned()),
        None => unsupported(format!(
            "{} has unsupported {modifier} modifier {:?}",
            decoded.op_name, token
        )),
    }
}

fn decoded_register_group(decoded: &DecodedPtx, name: &str) -> AResult<Vec<ObjectRef>> {
    let registers = decoded.operand(name)?;
    if registers.is_empty() {
        return unsupported(format!(
            "{}.{name} register group must not be empty",
            decoded.op_name
        ));
    }
    let mut result = Vec::new();
    for register in registers {
        match register {
            Some(value) => result.push(value.clone()),
            // The register contract rejects sink lanes before this group is read.
            None => return crate::analyze::util::not_covered("sunk lane in an MMA register group"),
        }
    }
    Ok(result)
}

fn is_typed_positive_zero_literal(value: &ObjectRef, dtype: &str) -> AResult<bool> {
    let literal_dtype = dtype_of(value)?;
    let allowed = literal_dtype == dtype
        || ((dtype == "float16" || dtype == "int32") && literal_dtype == "uint32");
    if !allowed {
        return Ok(false);
    }
    if let Some(imm) = value.as_node::<IntImmObj>() {
        return Ok(int_value(imm)? == 0);
    }
    if let Some(imm) = value.as_node::<FloatImmObj>() {
        let scalar = imm.value;
        return Ok(scalar == 0.0 && scalar.is_sign_positive());
    }
    Ok(false)
}

fn decoded_dense_c_group(decoded: &DecodedPtx, c_dtype: &str) -> AResult<(Vec<ObjectRef>, bool)> {
    let values = decoded.operand("c")?;
    if values.is_empty() {
        return unsupported(format!(
            "{}.c operand group must not be empty",
            decoded.op_name
        ));
    }
    let mut present = Vec::new();
    for value in values {
        match value {
            Some(value) => present.push(value.clone()),
            // A sink operand has no dtype to check for a positive-zero literal.
            None => {
                return crate::analyze::util::not_covered("sunk lane in an MMA accumulator group")
            }
        }
    }
    let mut all_zero = true;
    for value in &present {
        if !is_typed_positive_zero_literal(value, c_dtype)? {
            all_zero = false;
            break;
        }
    }
    Ok((present, !all_zero))
}

pub struct DenseMmaParts {
    pub d: Vec<ObjectRef>,
    pub a: Vec<ObjectRef>,
    pub b: Vec<ObjectRef>,
    pub c: Vec<ObjectRef>,
    pub has_c: bool,
    pub variant: String,
}

pub fn decoded_dense_mma_parts(decoded: &DecodedPtx) -> AResult<DenseMmaParts> {
    decoded_mma_call(decoded)?;
    let op_name = decoded.op_name.as_str();
    if !PTX_DENSE_MMA_CALLS.contains(&op_name) {
        return Err(Failure::Ffi(ffi_error(&format!(
            "dense PTX MMA emitter received {op_name}"
        ))));
    }
    let (m, n, k) = decoded_mma_shape(decoded)?;
    let d_dtype = decoded_mma_dtype(decoded, "dtype")?;
    let a_dtype = decoded_mma_dtype(decoded, "atype")?;
    let b_dtype = decoded_mma_dtype(decoded, "btype")?;
    let c_dtype = decoded_mma_dtype(decoded, "ctype")?;
    let a_layout = decoded.modifier("alayout")?.to_owned();
    let b_layout = decoded.modifier("blayout")?.to_owned();
    let saturate = decoded.modifier_or_empty("satfinite") == "satfinite";
    let bit_op = match decoded.modifier_or_empty("bitop") {
        "" => None,
        token => Some(token.to_owned()),
    };
    let rounding = decoded.modifier_or_empty("rnd");

    let d = decoded_register_group(decoded, "d")?;
    let a = decoded_register_group(decoded, "a")?;
    let b = decoded_register_group(decoded, "b")?;
    let (c, has_c) = decoded_dense_c_group(decoded, &c_dtype)?;
    let form = MmaTypes {
        d: &d_dtype,
        a: &a_dtype,
        b: &b_dtype,
        c: &c_dtype,
    }
    .dense(
        (m, n, k),
        (&a_layout, &b_layout),
        saturate,
        bit_op.as_deref(),
    );
    let Some(form) = form else {
        return unmodeled(
            format!("call:{op_name}"),
            format!(
                "{op_name} has no engine ABI for {} {a_layout}.{b_layout} {d_dtype}/{a_dtype}/{b_dtype}/{c_dtype}",
                decoded.modifier("shape")?
            ),
        );
    };
    let actual_counts = [d.len(), a.len(), b.len(), c.len()];
    let expected_counts = [
        form.d_count as usize,
        form.a_count as usize,
        form.b_count as usize,
        form.c_count as usize,
    ];
    if actual_counts != expected_counts {
        return unsupported(format!(
            "{op_name} register counts {:?} do not match engine ABI {:?}",
            &actual_counts, &expected_counts
        ));
    }
    let variant = v2_dense_variant_values(
        &form,
        m,
        k,
        &d_dtype,
        &a_dtype,
        &b_dtype,
        &c_dtype,
        &a_layout,
        &b_layout,
        has_c,
        saturate,
        bit_op.as_deref(),
        rounding,
    )?;
    Ok(DenseMmaParts {
        d,
        a,
        b,
        c,
        has_c,
        variant,
    })
}

pub struct SparseMmaParts {
    pub d: Vec<ObjectRef>,
    pub a: Vec<ObjectRef>,
    pub b: Vec<ObjectRef>,
    pub c: Vec<ObjectRef>,
    pub metadata: ObjectRef,
    pub selector: i64,
    pub variant: String,
}

pub fn decoded_sparse_mma_parts(ctx: &Ctx, decoded: &DecodedPtx) -> AResult<SparseMmaParts> {
    decoded_mma_call(decoded)?;
    let op_name = decoded.op_name.as_str();
    if !PTX_SPARSE_MMA_CALLS.contains(&op_name) {
        return Err(Failure::Ffi(ffi_error(&format!(
            "sparse PTX MMA emitter received {op_name}"
        ))));
    }
    let ordered_metadata = decoded.modifier("spvariant")? == "sp::ordered_metadata";
    let (m, n, k) = decoded_mma_shape(decoded)?;
    let d_dtype = decoded_mma_dtype(decoded, "dtype")?;
    let a_dtype = decoded_mma_dtype(decoded, "atype")?;
    let b_dtype = decoded_mma_dtype(decoded, "btype")?;
    let c_dtype = decoded_mma_dtype(decoded, "ctype")?;
    let a_layout = decoded.modifier("alayout")?.to_owned();
    let b_layout = decoded.modifier("blayout")?.to_owned();
    let saturate = decoded.modifier_or_empty("satfinite") == "satfinite";

    let d = decoded_register_group(decoded, "d")?;
    let a = decoded_register_group(decoded, "a")?;
    let b = decoded_register_group(decoded, "b")?;
    let c = decoded_register_group(decoded, "c")?;
    let metadata_group = decoded_register_group(decoded, "e")?;
    if metadata_group.len() != 1 {
        return unsupported(format!(
            "{op_name}.e expects one metadata register, got {}",
            metadata_group.len()
        ));
    }
    let selector = static_int(
        &ctx.analyzer,
        &prim(&decoded.scalar_operand("f")?)?,
        &format!("{op_name}.selector"),
        "must specialize to an integer",
    )?;
    let selector_limit = if op_name.ends_with("_all") {
        0
    } else if op_name.ends_with("_pair") {
        1
    } else {
        3
    };
    if selector < 0 || selector > selector_limit {
        return unsupported(format!(
            "{op_name}.selector must be in [0, {selector_limit}], got {selector}"
        ));
    }
    if (m, n, a_layout.as_str(), b_layout.as_str()) != (16, 8, "row", "col") {
        return unsupported(format!("{op_name} sparse engine requires m16n8 row.col"));
    }
    let form = MmaTypes {
        d: &d_dtype,
        a: &a_dtype,
        b: &b_dtype,
        c: &c_dtype,
    }
    .sparse(k, saturate);
    let Some(form) = form else {
        return unmodeled(
            format!("call:{op_name}"),
            format!(
                "{op_name} has no sparse engine ABI for {} {d_dtype}/{a_dtype}/{b_dtype}/{c_dtype}",
                decoded.modifier("shape")?
            ),
        );
    };
    let actual_counts = [d.len(), a.len(), b.len(), c.len()];
    let expected_counts = [
        form.c_count as usize,
        form.a_count as usize,
        form.b_count as usize,
        form.c_count as usize,
    ];
    if actual_counts != expected_counts {
        return unsupported(format!(
            "{op_name} register counts {:?} do not match engine ABI {:?}",
            &actual_counts, &expected_counts
        ));
    }
    let variant = v2_sparse_variant_values(
        &form,
        k,
        &a_dtype,
        &b_dtype,
        &c_dtype,
        saturate,
        ordered_metadata,
    )?;
    Ok(SparseMmaParts {
        d,
        a,
        b,
        c,
        metadata: metadata_group[0].clone(),
        selector,
        variant,
    })
}

/// The parsed parts of one PTX warp MMA call.
pub enum MmaParts {
    Dense(DenseMmaParts),
    Sparse(SparseMmaParts),
}

/// The parsed parts.
fn mma_parts(ctx: &Ctx, decoded: &DecodedPtx) -> AResult<MmaParts> {
    let parts = if PTX_DENSE_MMA_CALLS.contains(&decoded.op_name.as_str()) {
        MmaParts::Dense(decoded_dense_mma_parts(decoded)?)
    } else {
        MmaParts::Sparse(decoded_sparse_mma_parts(ctx, decoded)?)
    };
    Ok(parts)
}

/// The parsed legacy call.

fn matrix_register_storage(dtype: &str) -> AResult<(i64, i64)> {
    match dtype {
        "float16" | "bfloat16" => Ok((4, 2)),
        "int8" | "uint8" | "int4" | "uint4" | "int1" | "float8_e4m3fn" | "float8_e5m2" => {
            Ok((4, 1))
        }
        "float32" | "tf32" | "int32" => Ok((4, 4)),
        "float64" => Ok((8, 8)),
        _ => unsupported(format!("matrix register storage does not support {dtype}")),
    }
}

impl<'a> Emitter<'a> {
    fn matrix_physical_pointer(&mut self, expression: &ObjectRef, label: &str) -> AResult<String> {
        let mut value = self.emit_expr(expression)?;
        if value.rust_type != "PhysicalPtr" {
            value = self.emit_raw_generic_pointer(value, "ctx.active_mask()")?;
        }
        if value.rust_type != "PhysicalPtr" {
            return unsupported(format!("{label} must resolve to a physical address"));
        }
        Ok(value.code)
    }

    fn matrix_named_pointer(&mut self, expression: &ObjectRef, label: &str) -> AResult<String> {
        let mut buffer = projected_buffer(expression)?;
        if buffer.is_none() && expression.as_node::<VarObj>().is_some() {
            for (candidate, _) in &self.buffers {
                if same(&self.buffer_bindings.storage_key(candidate)?, expression) {
                    buffer = Some(candidate.clone());
                    break;
                }
            }
        }
        let code = match &buffer {
            Some(buffer) => self.matrix_buffer_base_address(buffer)?,
            None => self.matrix_physical_pointer(expression, label)?,
        };
        let name = self.control_name("matrix_physical_pointer");
        self.emit_line(&format!("let {name} = ({code}).clone();"));
        Ok(name)
    }

    /// `kernel.emit_buffer_address(buffer, zeros)`.
    fn matrix_buffer_base_address(&mut self, buffer: &BufferVar) -> AResult<String> {
        let mut zeros: Vec<PrimExpr> = Vec::new();
        for _ in buffer.buffer_type().shape.iter() {
            zeros.push(IntImm::new("int32", 0)?.into());
        }
        Ok(self.emit_buffer_address(buffer, &zeros)?.code)
    }

    fn matrix_warp_i64(&mut self, expression: &ObjectRef) -> AResult<String> {
        let value = self.emit_expr(expression)?;
        let value = self.as_i64(value)?;
        let value = self.as_warp_value(value);
        let name = self.control_name("matrix_warp_integer");
        self.emit_line(&format!("let {name}: WarpValue<i64> = {};", value.code));
        Ok(name)
    }

    fn decoded_register_address(&mut self, register: &ObjectRef, label: &str) -> AResult<String> {
        let load = register
            .as_node::<TensorLoadObj>()
            .expect("validated TensorLoad");
        let Some(buffer) = as_buffer(&oref(load.source.clone())) else {
            return crate::analyze::util::not_covered("MMA register source is not a typed buffer");
        };
        self.record_global_write(&buffer)?;
        let indices: Vec<PrimExpr> = load.indices.iter().collect();
        let pointer = self.emit_buffer_address(&buffer, &indices)?;
        let pointer_name = self.control_name(label);
        self.emit_line(&format!("let {pointer_name} = {};", pointer.code));
        Ok(abi::address(
            "v2::Register",
            &abi::cloned(&pointer_name),
            None,
        ))
    }

    fn decoded_register_addresses(
        &mut self,
        registers: &[ObjectRef],
        label: &str,
    ) -> AResult<String> {
        let mut rendered = Vec::new();
        for (index, register) in registers.iter().enumerate() {
            rendered.push(self.decoded_register_address(register, &format!("{label}_{index}"))?);
        }
        Ok(format!("[{}]", rendered.join(", ")))
    }

    fn decoded_register_value(
        &mut self,
        register: &ObjectRef,
        label: &str,
        f64: bool,
        access_mask: Option<&str>,
    ) -> AResult<String> {
        let mut bits = self.emit_as_unsigned_bits(
            register,
            if f64 { 64 } else { 32 },
            "warp MMA",
            label,
            access_mask,
        )?;
        if f64 {
            bits = self.emit_from_unsigned_bits(bits, "float64", 64, "warp MMA", label)?;
        }
        Ok(abi::register(&bits.code))
    }

    /// Snapshot every source before the
    /// instruction writes any destination.
    fn decoded_register_values(
        &mut self,
        registers: &[ObjectRef],
        label: &str,
        f64: bool,
    ) -> AResult<String> {
        let name = self.control_name(label);
        let mut values = Vec::new();
        for (index, register) in registers.iter().enumerate() {
            values.push(self.decoded_register_value(
                register,
                &format!("{label}_{index}"),
                f64,
                None,
            )?);
        }
        self.emit_line(&format!("let {name} = [{}];", values.join(", ")));
        Ok(name)
    }

    /// `emit_ptx_warp_mma`.
    fn emit_ptx_warp_mma(
        &mut self,
        decoded: &DecodedPtx,
        parts: &MmaParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "warp_mma",
            &format!(
                "{} predicate must lower to bool or integer",
                decoded.op_name
            ),
        )?;
        let mask = region.mask.clone();
        self.emit_ptx_warp_mma_body(decoded, parts, source_op_id)?;
        self.close_predicated_region(region);
        let destinations = decoded.operand("d")?.to_vec();
        let Some(Some(first)) = destinations.first() else {
            return not_covered("warp MMA without a register destination");
        };
        let result_dtype = dtype_of(first)?;
        self.finish_predicated_destinations(
            decoded,
            &destinations,
            &result_dtype,
            &mask,
            source_op_id,
            false,
        )
    }

    fn emit_ptx_warp_mma_body(
        &mut self,
        decoded: &DecodedPtx,
        parts: &MmaParts,
        source_op_id: i64,
    ) -> AResult<()> {
        if let MmaParts::Dense(parts) = parts {
            let f64 = decoded.modifier("atype")? == "f64";
            let mut operands = vec![
                self.decoded_register_addresses(&parts.d, "decoded_mma_d")?,
                self.decoded_register_values(&parts.a, "decoded_mma_a", f64)?,
                self.decoded_register_values(&parts.b, "decoded_mma_b", f64)?,
            ];
            if parts.has_c {
                operands.push(self.decoded_register_values(&parts.c, "decoded_mma_c", f64)?);
            }
            let site = self.v2_site(Some(source_op_id));
            let call = abi::warp_call(
                MMA_SYNC,
                &site,
                &[format!("({})", operands.join(", "))],
                Some(&parts.variant),
                None,
                false,
                true,
            );
            self.emit_line(&format!("{call};"));
            return Ok(());
        }

        let MmaParts::Sparse(parts) = parts else {
            unreachable!("dense MMA parts are lowered above");
        };
        let metadata_mask = self.control_name("decoded_sparse_mma_metadata_mask");
        self.emit_line(&format!(
            "let {metadata_mask} = WarpMask::from_bits(<{}>::metadata_source_mask({}_usize)?.bits()) & ctx.active_mask();",
            parts.variant, parts.selector
        ));
        let metadata_value = self.decoded_register_value(
            &parts.metadata,
            "decoded_sparse_mma_metadata",
            false,
            Some(&metadata_mask),
        )?;
        let d = self.decoded_register_addresses(&parts.d, "decoded_sparse_mma_d")?;
        let a = self.decoded_register_values(&parts.a, "decoded_sparse_mma_a", false)?;
        let b = self.decoded_register_values(&parts.b, "decoded_sparse_mma_b", false)?;
        let c = self.decoded_register_values(&parts.c, "decoded_sparse_mma_c", false)?;
        let site = self.v2_site(Some(source_op_id));
        let call = abi::warp_call(
            MMA_SP_SYNC,
            &site,
            &[format!(
                "({d}, {a}, {b}, {c}, {metadata_value}, {}_usize)",
                parts.selector
            )],
            Some(&parts.variant),
            None,
            false,
            true,
        );
        self.emit_line(&format!("{call};"));
        Ok(())
    }

    fn legacy_register_group(
        &mut self,
        pointer: &str,
        offsets: &str,
        count: i64,
        dtype: &str,
        label: &str,
        space: &str,
    ) -> AResult<String> {
        let (register_bytes, logical_itemsize) = matrix_register_storage(dtype)?;
        let base = self.control_name(&format!("{label}_base"));
        self.emit_line(&format!(
            "let {base} = {}.element_offset(&v2_register(({offsets}).clone()), {logical_itemsize}_usize, v2_mask(ctx.active_mask()))?;",
            abi::address(space, &abi::cloned(pointer), None)
        ));
        let registers: Vec<String> = (0..count)
            .map(|index| {
                format!(
                    "{base}.byte_offset(&{}, {logical_itemsize}_usize, v2_mask(ctx.active_mask()))?",
                    abi::splat(&format!("{}_i64", index * register_bytes))
                )
            })
            .collect();
        Ok(format!("[{}]", registers.join(", ")))
    }

    fn emit_legacy_mma(&mut self, call: &LegacyMatrixCall, source_op_id: i64) -> AResult<()> {
        let form = legacy_dense_variant(call)?;
        let op_name = call.op_name.as_str();
        let args = &call.args;
        let a = self.matrix_named_pointer(&args[6], &format!("{op_name}.A"))?;
        let a_offset = self.matrix_warp_i64(&args[7])?;
        let b = self.matrix_named_pointer(&args[8], &format!("{op_name}.B"))?;
        let b_offset = self.matrix_warp_i64(&args[9])?;
        let c = self.matrix_named_pointer(&args[10], &format!("{op_name}.accumulator"))?;
        let c_offset = self.matrix_warp_i64(&args[11])?;
        let a_dtype = call.a_dtype.as_deref().expect("legacy a_dtype");
        let c_dtype = call.c_dtype.as_deref().expect("legacy c_dtype");
        let a_registers = self.legacy_register_group(
            &a,
            &a_offset,
            form.a_count,
            a_dtype,
            "legacy_mma_a_registers",
            "v2::Generic",
        )?;
        let b_registers = self.legacy_register_group(
            &b,
            &b_offset,
            form.b_count,
            call.b_dtype.as_deref().expect("legacy b_dtype"),
            "legacy_mma_b_registers",
            "v2::Generic",
        )?;
        let c_registers = self.legacy_register_group(
            &c,
            &c_offset,
            form.c_count,
            c_dtype,
            "legacy_mma_accumulator_registers",
            "v2::Register",
        )?;
        let c_sources = self.legacy_register_group(
            &c,
            &c_offset,
            form.c_count,
            c_dtype,
            "legacy_mma_c_sources",
            "v2::Generic",
        )?;
        let load_variant = format!(
            "v2::mem::variant::Ld<v2::reg::variant::{}, v2::Generic>",
            if a_dtype == "float64" { "F64" } else { "U32" }
        );
        let mut sources = Vec::new();
        for (addresses, count) in [
            (a_registers, form.a_count),
            (b_registers, form.b_count),
            (c_sources, form.c_count),
        ] {
            let name = self.control_name("legacy_mma_sources");
            self.emit_line(&format!("let {name} = {addresses};"));
            let mut loads = Vec::new();
            for index in 0..count {
                let site = self.v2_site(Some(source_op_id));
                loads.push(abi::warp_call(
                    "mem::ld",
                    &site,
                    &[format!("{name}[{index}].clone()")],
                    Some(&load_variant),
                    None,
                    false,
                    true,
                ));
            }
            let values = self.control_name("legacy_mma_values");
            self.emit_line(&format!("let {values} = [{}];", loads.join(", ")));
            sources.push(values);
        }
        let variant = v2_dense_variant(call, &form)?;
        let site = self.v2_site(Some(source_op_id));
        let invocation = abi::warp_call(
            MMA_SYNC,
            &site,
            &[format!("({c_registers}, {})", sources.join(", "))],
            Some(&variant),
            None,
            false,
            true,
        );
        self.emit_line(&format!("{invocation};"));
        Ok(())
    }

    fn emit_mma_fill(&mut self, call: &LegacyMatrixCall, source_op_id: i64) -> AResult<()> {
        let op_name = call.op_name.as_str();
        let d_dtype = call.d_dtype.as_deref().expect("fill dtype");
        let local_size = call.local_size.expect("fill local_size");
        let destination =
            self.matrix_named_pointer(&call.args[1], &format!("{op_name}.destination"))?;
        let offsets = self.matrix_warp_i64(&call.args[2])?;
        let itemsize = dtype_byte_len(self.ctx.schema, d_dtype)?;
        let base = self.control_name("mma_fill_base");
        let slot = self.control_name("mma_fill_slot");
        let address = self.control_name("mma_fill_address");
        let zero = match d_dtype {
            "float16" => "0.0_f32",
            "float32" => "0.0_f32",
            "float64" => "0.0_f64",
            "int32" => "0_i32",
            _ => {
                return Err(crate::analyze::util::Failure::Ffi(
                    crate::analyze::util::ffi_error("mma_fill without a validated dtype"),
                ))
            }
        };
        self.emit_line(&format!(
            "let {base} = {}.element_offset(&v2_register({offsets}), {itemsize}_usize, v2_mask(ctx.active_mask()))?;",
            abi::address("v2::Generic", &destination, None)
        ));
        self.emit_line(&format!("for {slot} in 0..{local_size}_usize {{"));
        self.emit_line(&format!(
            "    let {address} = {base}.byte_offset(&{}, {itemsize}_usize, v2_mask(ctx.active_mask()))?;",
            abi::splat(&format!("({slot} * {itemsize}_usize) as i64"))
        ));
        let site = self.v2_site(Some(source_op_id));
        let store = abi::warp_call(
            MEM_ST,
            &site,
            &[format!("({address}, {})", abi::splat(zero))],
            Some(&format!(
                "v2::mem::variant::St<{}, v2::Generic>",
                v2_memory_type_rust(self.ctx.schema, d_dtype)?
            )),
            None,
            false,
            true,
        );
        self.emit_line(&format!("    {store};"));
        self.emit_line("}");
        Ok(())
    }

    fn emit_mma_store(&mut self, call: &LegacyMatrixCall, source_op_id: i64) -> AResult<()> {
        let op_name = call.op_name.as_str();
        let d_dtype = call.d_dtype.as_deref().expect("store dtype");
        let destination =
            self.matrix_named_pointer(&call.args[2], &format!("{op_name}.destination"))?;
        let source = self.matrix_named_pointer(&call.args[3], &format!("{op_name}.source"))?;
        let source_offsets = self.matrix_warp_i64(&call.args[4])?;
        let destination_strides = self.matrix_warp_i64(&call.args[5])?;
        let itemsize = dtype_byte_len(self.ctx.schema, d_dtype)?;
        let local_id = self.control_name("mma_store_local_id");
        let source_elements = self.control_name("mma_store_source_elements");
        let destination_elements = self.control_name("mma_store_destination_elements");
        let source_address = self.control_name("mma_store_source_address");
        let destination_address = self.control_name("mma_store_destination_address");
        let value = self.control_name("mma_store_value");
        self.emit_line(&format!("for {local_id} in 0..8_usize {{"));
        self.emit_line(&format!(
            "    let {source_elements} = WarpValue::from_fn(|lane| {source_offsets}[lane].wrapping_add({local_id} as i64));"
        ));
        self.emit_line(&format!(
            "    let {destination_elements} = WarpValue::from_fn(|lane| {{"
        ));
        self.emit_line(&format!(
            "        let row = 8_usize * (({local_id} % 4_usize) / 2_usize) + lane / 4_usize;"
        ));
        self.emit_line(&format!(
            "        let col = 8_usize * ({local_id} / 4_usize) + 2_usize * (lane % 4_usize) + {local_id} % 2_usize;"
        ));
        self.emit_line(&format!(
            "        (row as i64).wrapping_mul({destination_strides}[lane]).wrapping_add(col as i64)"
        ));
        self.emit_line("    });");
        self.emit_line(&format!(
            "    let {source_address} = {}.element_offset(&v2_register({source_elements}), {itemsize}_usize, v2_mask(ctx.active_mask()))?;",
            abi::address("v2::Generic", &abi::cloned(&source), None)
        ));
        self.emit_line(&format!(
            "    let {destination_address} = {}.element_offset(&v2_register({destination_elements}), {itemsize}_usize, v2_mask(ctx.active_mask()))?;",
            abi::address("v2::Generic", &abi::cloned(&destination), None)
        ));
        let marker = v2_memory_type_rust(self.ctx.schema, d_dtype)?;
        let site = self.v2_site(Some(source_op_id));
        let load = abi::warp_call(
            MEM_LD,
            &site,
            &[source_address.clone()],
            Some(&format!("v2::mem::variant::Ld<{marker}, v2::Generic>")),
            None,
            false,
            true,
        );
        self.emit_line(&format!("    let {value} = {load};"));
        let site = self.v2_site(Some(source_op_id));
        let store = abi::warp_call(
            MEM_ST,
            &site,
            &[format!("({destination_address}, {value})")],
            Some(&format!("v2::mem::variant::St<{marker}, v2::Generic>")),
            None,
            false,
            true,
        );
        self.emit_line(&format!("    {store};"));
        self.emit_line("}");
        Ok(())
    }

    /// Statement lowering of one legacy matrix call.
    pub fn emit_legacy_matrix(
        &mut self,
        call: &LegacyMatrixCall,
        source_op_id: i64,
    ) -> AResult<()> {
        match call.op_name.as_str() {
            "tirx.ptx_legacy.mma" => self.emit_legacy_mma(call, source_op_id),
            "tirx.mma_fill" | "tirx.mma_fill_legacy" => self.emit_mma_fill(call, source_op_id),
            "tirx.mma_store" | "tirx.mma_store_legacy" => self.emit_mma_store(call, source_op_id),
            other => {
                crate::analyze::util::not_covered(format!("statement call {other} has no lowering"))
            }
        }
    }
}

pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = mma_parts(emitter.ctx, decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_warp_mma(decoded, &parts, source_op_id)?;
    Ok(None)
}

pub fn emit_legacy(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    emitter.written_global_buffers = None;
    let parts = resolve_legacy_matrix_call(emitter.ctx, call.node, &call.op_name)?;
    if call.op_name == "tirx.ptx_legacy.mma" {
        v2_dense_variant(&parts, &legacy_dense_variant(&parts)?)?;
    }
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_legacy_matrix(&parts, source_op_id)?;
    Ok(None)
}
