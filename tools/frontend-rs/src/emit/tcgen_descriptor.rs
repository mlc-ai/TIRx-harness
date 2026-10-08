//! Descriptor encoding lookups and bit assembly for TCGEN helper calls.

use crate::analyze::buffers::call_op_name;
use crate::analyze::cuda_arch::{cuda_arch, CUDA_ARCH_ATTR};
use crate::analyze::util::{
    dtype_of, oref, prim, repr_text, simplify, static_string, unsupported, AResult, Failure,
};
use crate::analyze::Ctx;
use crate::tvm_compat::int_value;
use tvm::ir::{CallObj, IntImmObj};
use tvm::tirx::PrimFunc;
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

pub const DESCRIPTOR_CALLS: [&str; 3] = [
    "tirx.cuda.tcgen05_encode_matrix_descriptor",
    "tirx.cuda.tcgen05_encode_instr_descriptor",
    "tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled",
];

/// `TcgenDescriptorCall`.
pub struct TcgenDescriptorCall {
    pub args: Vec<ObjectRef>,
    pub matrix_source_is_null: bool,
    pub encoded_u32: Option<i64>,
}

// dtype, format bits, dense transpose, block transpose, narrow-N transpose key
#[rustfmt::skip]
const FORMAT_ENCODINGS: &[(&str, i64, bool, bool, bool)] = &[
    ("float16", 0, true, false, false),
    ("bfloat16", 1, true, false, false),
    ("tf32", 2, true, false, false),
    ("float8_e4m3fn", 0, true, true, true),
    ("float8_e4m3fnuz", 0, true, true, true),
    ("float8_e5m2", 1, true, true, true),
    ("float6_e2m3fn", 3, false, false, false),
    ("float6_e3m2fn", 4, false, false, false),
    ("float4_e2m1fn", 5, false, false, false),
    ("uint8", 0, true, false, true),
    ("int8", 1, true, false, true),
    ("float32", 1, false, false, false),
    ("int32", 2, false, false, false),
];

fn format_encoding(dtype: &str) -> &'static (&'static str, i64, bool, bool, bool) {
    FORMAT_ENCODINGS
        .iter()
        .find(|row| row.0 == dtype)
        .expect("mapped descriptor dtype")
}

// Each row maps descriptor operand spellings to the engine family that
// interprets their format bits. These are encoding keys, not inferred dtypes.
#[rustfmt::skip]
const DENSE_ENCODINGS: &[(&str, &str, &str, &str)] = &[
    ("float16", "float16", "float16", "f16"),
    ("float16", "float16", "bfloat16", "f16"),
    ("float16", "bfloat16", "float16", "f16"),
    ("float16", "bfloat16", "bfloat16", "f16"),
    ("float32", "float16", "float16", "f16"),
    ("float32", "float16", "bfloat16", "f16"),
    ("float32", "bfloat16", "float16", "f16"),
    ("float32", "bfloat16", "bfloat16", "f16"),
    ("float32", "tf32", "tf32", "tf32"),
    ("int32", "int8", "int8", "i8"),
    ("int32", "int8", "uint8", "i8"),
    ("int32", "uint8", "int8", "i8"),
    ("int32", "uint8", "uint8", "i8"),
    ("float16", "float8_e4m3fn", "float8_e4m3fn", "f8f6f4"),
    ("float16", "float8_e4m3fn", "float8_e4m3fnuz", "f8f6f4"),
    ("float16", "float8_e4m3fn", "float8_e5m2", "f8f6f4"),
    ("float16", "float8_e4m3fn", "float6_e2m3fn", "f8f6f4"),
    ("float16", "float8_e4m3fn", "float6_e3m2fn", "f8f6f4"),
    ("float16", "float8_e4m3fn", "float4_e2m1fn", "f8f6f4"),
    ("float16", "float8_e4m3fnuz", "float8_e4m3fn", "f8f6f4"),
    ("float16", "float8_e4m3fnuz", "float8_e4m3fnuz", "f8f6f4"),
    ("float16", "float8_e4m3fnuz", "float8_e5m2", "f8f6f4"),
    ("float16", "float8_e4m3fnuz", "float6_e2m3fn", "f8f6f4"),
    ("float16", "float8_e4m3fnuz", "float6_e3m2fn", "f8f6f4"),
    ("float16", "float8_e4m3fnuz", "float4_e2m1fn", "f8f6f4"),
    ("float16", "float8_e5m2", "float8_e4m3fn", "f8f6f4"),
    ("float16", "float8_e5m2", "float8_e4m3fnuz", "f8f6f4"),
    ("float16", "float8_e5m2", "float8_e5m2", "f8f6f4"),
    ("float16", "float8_e5m2", "float6_e2m3fn", "f8f6f4"),
    ("float16", "float8_e5m2", "float6_e3m2fn", "f8f6f4"),
    ("float16", "float8_e5m2", "float4_e2m1fn", "f8f6f4"),
    ("float16", "float6_e2m3fn", "float8_e4m3fn", "f8f6f4"),
    ("float16", "float6_e2m3fn", "float8_e4m3fnuz", "f8f6f4"),
    ("float16", "float6_e2m3fn", "float8_e5m2", "f8f6f4"),
    ("float16", "float6_e2m3fn", "float6_e2m3fn", "f8f6f4"),
    ("float16", "float6_e2m3fn", "float6_e3m2fn", "f8f6f4"),
    ("float16", "float6_e2m3fn", "float4_e2m1fn", "f8f6f4"),
    ("float16", "float6_e3m2fn", "float8_e4m3fn", "f8f6f4"),
    ("float16", "float6_e3m2fn", "float8_e4m3fnuz", "f8f6f4"),
    ("float16", "float6_e3m2fn", "float8_e5m2", "f8f6f4"),
    ("float16", "float6_e3m2fn", "float6_e2m3fn", "f8f6f4"),
    ("float16", "float6_e3m2fn", "float6_e3m2fn", "f8f6f4"),
    ("float16", "float6_e3m2fn", "float4_e2m1fn", "f8f6f4"),
    ("float16", "float4_e2m1fn", "float8_e4m3fn", "f8f6f4"),
    ("float16", "float4_e2m1fn", "float8_e4m3fnuz", "f8f6f4"),
    ("float16", "float4_e2m1fn", "float8_e5m2", "f8f6f4"),
    ("float16", "float4_e2m1fn", "float6_e2m3fn", "f8f6f4"),
    ("float16", "float4_e2m1fn", "float6_e3m2fn", "f8f6f4"),
    ("float16", "float4_e2m1fn", "float4_e2m1fn", "f8f6f4"),
    ("float32", "float8_e4m3fn", "float8_e4m3fn", "f8f6f4"),
    ("float32", "float8_e4m3fn", "float8_e4m3fnuz", "f8f6f4"),
    ("float32", "float8_e4m3fn", "float8_e5m2", "f8f6f4"),
    ("float32", "float8_e4m3fn", "float6_e2m3fn", "f8f6f4"),
    ("float32", "float8_e4m3fn", "float6_e3m2fn", "f8f6f4"),
    ("float32", "float8_e4m3fn", "float4_e2m1fn", "f8f6f4"),
    ("float32", "float8_e4m3fnuz", "float8_e4m3fn", "f8f6f4"),
    ("float32", "float8_e4m3fnuz", "float8_e4m3fnuz", "f8f6f4"),
    ("float32", "float8_e4m3fnuz", "float8_e5m2", "f8f6f4"),
    ("float32", "float8_e4m3fnuz", "float6_e2m3fn", "f8f6f4"),
    ("float32", "float8_e4m3fnuz", "float6_e3m2fn", "f8f6f4"),
    ("float32", "float8_e4m3fnuz", "float4_e2m1fn", "f8f6f4"),
    ("float32", "float8_e5m2", "float8_e4m3fn", "f8f6f4"),
    ("float32", "float8_e5m2", "float8_e4m3fnuz", "f8f6f4"),
    ("float32", "float8_e5m2", "float8_e5m2", "f8f6f4"),
    ("float32", "float8_e5m2", "float6_e2m3fn", "f8f6f4"),
    ("float32", "float8_e5m2", "float6_e3m2fn", "f8f6f4"),
    ("float32", "float8_e5m2", "float4_e2m1fn", "f8f6f4"),
    ("float32", "float6_e2m3fn", "float8_e4m3fn", "f8f6f4"),
    ("float32", "float6_e2m3fn", "float8_e4m3fnuz", "f8f6f4"),
    ("float32", "float6_e2m3fn", "float8_e5m2", "f8f6f4"),
    ("float32", "float6_e2m3fn", "float6_e2m3fn", "f8f6f4"),
    ("float32", "float6_e2m3fn", "float6_e3m2fn", "f8f6f4"),
    ("float32", "float6_e2m3fn", "float4_e2m1fn", "f8f6f4"),
    ("float32", "float6_e3m2fn", "float8_e4m3fn", "f8f6f4"),
    ("float32", "float6_e3m2fn", "float8_e4m3fnuz", "f8f6f4"),
    ("float32", "float6_e3m2fn", "float8_e5m2", "f8f6f4"),
    ("float32", "float6_e3m2fn", "float6_e2m3fn", "f8f6f4"),
    ("float32", "float6_e3m2fn", "float6_e3m2fn", "f8f6f4"),
    ("float32", "float6_e3m2fn", "float4_e2m1fn", "f8f6f4"),
    ("float32", "float4_e2m1fn", "float8_e4m3fn", "f8f6f4"),
    ("float32", "float4_e2m1fn", "float8_e4m3fnuz", "f8f6f4"),
    ("float32", "float4_e2m1fn", "float8_e5m2", "f8f6f4"),
    ("float32", "float4_e2m1fn", "float6_e2m3fn", "f8f6f4"),
    ("float32", "float4_e2m1fn", "float6_e3m2fn", "f8f6f4"),
    ("float32", "float4_e2m1fn", "float4_e2m1fn", "f8f6f4"),
];

// A/B type, scale type, engine kind, A/B format bits, scale format bit.
type BlockEncoding = (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    i64,
    i64,
    i64,
);

#[rustfmt::skip]
const BLOCK_ENCODINGS: &[BlockEncoding] = &[
    ("float8_e4m3fn", "float8_e4m3fn", "float8_e8m0fnu", "mxf8f6f4", 0, 0, 1),
    ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e8m0fnu", "mxf8f6f4", 0, 0, 1),
    ("float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu", "mxf8f6f4", 0, 1, 1),
    ("float8_e4m3fn", "float6_e2m3fn", "float8_e8m0fnu", "mxf8f6f4", 0, 3, 1),
    ("float8_e4m3fn", "float6_e3m2fn", "float8_e8m0fnu", "mxf8f6f4", 0, 4, 1),
    ("float8_e4m3fn", "float4_e2m1fn", "float8_e8m0fnu", "mxf8f6f4", 0, 5, 1),
    ("float8_e4m3fnuz", "float8_e4m3fn", "float8_e8m0fnu", "mxf8f6f4", 0, 0, 1),
    ("float8_e4m3fnuz", "float8_e4m3fnuz", "float8_e8m0fnu", "mxf8f6f4", 0, 0, 1),
    ("float8_e4m3fnuz", "float8_e5m2", "float8_e8m0fnu", "mxf8f6f4", 0, 1, 1),
    ("float8_e4m3fnuz", "float6_e2m3fn", "float8_e8m0fnu", "mxf8f6f4", 0, 3, 1),
    ("float8_e4m3fnuz", "float6_e3m2fn", "float8_e8m0fnu", "mxf8f6f4", 0, 4, 1),
    ("float8_e4m3fnuz", "float4_e2m1fn", "float8_e8m0fnu", "mxf8f6f4", 0, 5, 1),
    ("float8_e5m2", "float8_e4m3fn", "float8_e8m0fnu", "mxf8f6f4", 1, 0, 1),
    ("float8_e5m2", "float8_e4m3fnuz", "float8_e8m0fnu", "mxf8f6f4", 1, 0, 1),
    ("float8_e5m2", "float8_e5m2", "float8_e8m0fnu", "mxf8f6f4", 1, 1, 1),
    ("float8_e5m2", "float6_e2m3fn", "float8_e8m0fnu", "mxf8f6f4", 1, 3, 1),
    ("float8_e5m2", "float6_e3m2fn", "float8_e8m0fnu", "mxf8f6f4", 1, 4, 1),
    ("float8_e5m2", "float4_e2m1fn", "float8_e8m0fnu", "mxf8f6f4", 1, 5, 1),
    ("float6_e2m3fn", "float8_e4m3fn", "float8_e8m0fnu", "mxf8f6f4", 3, 0, 1),
    ("float6_e2m3fn", "float8_e4m3fnuz", "float8_e8m0fnu", "mxf8f6f4", 3, 0, 1),
    ("float6_e2m3fn", "float8_e5m2", "float8_e8m0fnu", "mxf8f6f4", 3, 1, 1),
    ("float6_e2m3fn", "float6_e2m3fn", "float8_e8m0fnu", "mxf8f6f4", 3, 3, 1),
    ("float6_e2m3fn", "float6_e3m2fn", "float8_e8m0fnu", "mxf8f6f4", 3, 4, 1),
    ("float6_e2m3fn", "float4_e2m1fn", "float8_e8m0fnu", "mxf8f6f4", 3, 5, 1),
    ("float6_e3m2fn", "float8_e4m3fn", "float8_e8m0fnu", "mxf8f6f4", 4, 0, 1),
    ("float6_e3m2fn", "float8_e4m3fnuz", "float8_e8m0fnu", "mxf8f6f4", 4, 0, 1),
    ("float6_e3m2fn", "float8_e5m2", "float8_e8m0fnu", "mxf8f6f4", 4, 1, 1),
    ("float6_e3m2fn", "float6_e2m3fn", "float8_e8m0fnu", "mxf8f6f4", 4, 3, 1),
    ("float6_e3m2fn", "float6_e3m2fn", "float8_e8m0fnu", "mxf8f6f4", 4, 4, 1),
    ("float6_e3m2fn", "float4_e2m1fn", "float8_e8m0fnu", "mxf8f6f4", 4, 5, 1),
    ("float4_e2m1fn", "float8_e4m3fn", "float8_e8m0fnu", "mxf8f6f4", 5, 0, 1),
    ("float4_e2m1fn", "float8_e4m3fnuz", "float8_e8m0fnu", "mxf8f6f4", 5, 0, 1),
    ("float4_e2m1fn", "float8_e5m2", "float8_e8m0fnu", "mxf8f6f4", 5, 1, 1),
    ("float4_e2m1fn", "float6_e2m3fn", "float8_e8m0fnu", "mxf8f6f4", 5, 3, 1),
    ("float4_e2m1fn", "float6_e3m2fn", "float8_e8m0fnu", "mxf8f6f4", 5, 4, 1),
    ("float4_e2m1fn", "float4_e2m1fn", "float8_e8m0fnu", "mxf4", 1, 1, 1),
    ("float4_e2m1fn", "float4_e2m1fn", "float8_e4m3fn", "mxf4nvf4", 1, 1, 0),
];

fn dense_kind(d_dtype: &str, a_dtype: &str, b_dtype: &str) -> AResult<&'static str> {
    match DENSE_ENCODINGS.iter().find(|row| (row.0, row.1, row.2) == (d_dtype, a_dtype, b_dtype)) {
        Some(row) => Ok(row.3),
        None => unsupported(format!(
            "tcgen05 dense instruction descriptor has invalid dtype combination D={d_dtype}, A={a_dtype}, B={b_dtype}"
        )),
    }
}

fn block_encoding(
    d_dtype: &str,
    a_dtype: &str,
    b_dtype: &str,
    sfa_dtype: &str,
    sfb_dtype: &str,
) -> AResult<&'static BlockEncoding> {
    if d_dtype != "float32" {
        return unsupported("tcgen05 block-scaled instruction descriptor requires float32 D");
    }
    match BLOCK_ENCODINGS.iter().find(|row| (row.0, row.1, row.2, row.2) == (a_dtype, b_dtype, sfa_dtype, sfb_dtype)) {
        Some(row) => Ok(row),
        None => unsupported(format!(
            "tcgen05 block-scaled instruction descriptor has invalid dtype combination D={d_dtype}, A={a_dtype}, B={b_dtype}, SFA={sfa_dtype}, SFB={sfb_dtype}"
        )),
    }
}

// Exact shape keys accepted by the descriptor helpers. N lists include the
// narrow integer encodings and the dense FP4 K96 helper encodings explicitly.
const N8: &[i64] = &[
    8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168,
    176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256,
];
const N16: &[i64] = &[
    16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256,
];
const N32: &[i64] = &[32, 64, 96, 128, 160, 192, 224, 256];
const N_I8_CTA1: &[i64] = &[
    8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256,
];

// kind, CTA group, M, K, sparse, N encodings
#[rustfmt::skip]
const SHAPE_ENCODINGS: &[(&str, i64, i64, i64, bool, &[i64])] = &[
    ("f16", 1, 64, 16, false, N8),
    ("f16", 1, 128, 16, false, N8),
    ("f16", 1, 64, 32, true, N8),
    ("f16", 1, 128, 32, true, N8),
    ("f16", 2, 128, 16, false, N16),
    ("f16", 2, 256, 16, false, N16),
    ("f16", 2, 128, 32, true, N16),
    ("f16", 2, 256, 32, true, N16),
    ("tf32", 1, 64, 8, false, N8),
    ("tf32", 1, 128, 8, false, N8),
    ("tf32", 1, 64, 16, true, N8),
    ("tf32", 1, 128, 16, true, N8),
    ("tf32", 2, 128, 8, false, N16),
    ("tf32", 2, 256, 8, false, N16),
    ("tf32", 2, 128, 16, true, N16),
    ("tf32", 2, 256, 16, true, N16),
    ("f8f6f4", 1, 64, 32, false, N8),
    ("f8f6f4", 1, 128, 32, false, N8),
    ("f8f6f4", 1, 64, 64, true, N8),
    ("f8f6f4", 1, 128, 64, true, N8),
    ("f8f6f4", 2, 128, 32, false, N16),
    ("f8f6f4", 2, 256, 32, false, N16),
    ("f8f6f4", 2, 128, 64, true, N16),
    ("f8f6f4", 2, 256, 64, true, N16),
    ("i8", 1, 64, 32, false, N_I8_CTA1),
    ("i8", 1, 128, 32, false, N_I8_CTA1),
    ("i8", 1, 64, 64, true, N_I8_CTA1),
    ("i8", 1, 128, 64, true, N_I8_CTA1),
    ("i8", 2, 128, 32, false, N32),
    ("i8", 2, 256, 32, false, N32),
    ("i8", 2, 128, 64, true, N32),
    ("i8", 2, 256, 64, true, N32),
    ("mxf8f6f4", 1, 128, 32, false, N8),
    ("mxf8f6f4", 1, 128, 64, true, N8),
    ("mxf8f6f4", 2, 128, 32, false, N16),
    ("mxf8f6f4", 2, 256, 32, false, N16),
    ("mxf8f6f4", 2, 256, 64, true, N16),
    ("mxf4", 1, 128, 64, false, N8),
    ("mxf4", 1, 128, 96, false, N8),
    ("mxf4", 1, 128, 128, true, N8),
    ("mxf4", 2, 128, 64, false, N16),
    ("mxf4", 2, 256, 64, false, N16),
    ("mxf4", 2, 256, 96, false, N16),
    ("mxf4", 2, 256, 128, true, N16),
    ("mxf4nvf4", 1, 128, 64, false, N8),
    ("mxf4nvf4", 1, 128, 96, false, N8),
    ("mxf4nvf4", 1, 128, 128, true, N8),
    ("mxf4nvf4", 2, 128, 64, false, N16),
    ("mxf4nvf4", 2, 256, 64, false, N16),
    ("mxf4nvf4", 2, 256, 96, false, N16),
    ("mxf4nvf4", 2, 256, 128, true, N16),
];

pub fn validate_tcgen05_instruction_shape(
    kind: &str,
    cta_group: i64,
    m: i64,
    n: i64,
    k: i64,
    sparse: bool,
) -> AResult<()> {
    if SHAPE_ENCODINGS.iter().any(|row| {
        (row.0, row.1, row.2, row.3, row.4) == (kind, cta_group, m, k, sparse) && row.5.contains(&n)
    }) {
        return Ok(());
    }
    // Keep actionable rejection reasons after the single encoding lookup.
    if cta_group != 1 && cta_group != 2 {
        return unsupported(format!(
            "tcgen05 instruction descriptor cta_group must be 1 or 2, got {cta_group}"
        ));
    }
    if !SHAPE_ENCODINGS.iter().any(|row| row.0 == kind) {
        return unsupported(format!(
            "unknown tcgen05 instruction descriptor kind {:?}",
            kind
        ));
    }
    if sparse && matches!(kind, "mxf8f6f4" | "mxf4" | "mxf4nvf4") && cta_group == 2 && m != 256 {
        return unsupported(format!(
            "invalid sparse tcgen05 block-scaled descriptor shape kind={kind}, cta_group={cta_group}, M={m}, N={n}, K={k}; CTA group 2 requires M=256"
        ));
    }
    unsupported(format!(
        "invalid tcgen05 descriptor shape kind={kind}, cta_group={cta_group}, M={m}, N={n}, K={k}"
    ))
}

fn validate_8bit_transpose_b_shape(
    b_dtype: &str,
    trans_b: bool,
    cta_group: i64,
    n: i64,
) -> AResult<()> {
    if !trans_b || !format_encoding(b_dtype).4 {
        return Ok(());
    }
    let (step, encodings) = if cta_group == 1 { (16, N16) } else { (32, N32) };
    if !encodings.contains(&n) {
        return unsupported(format!(
            "tcgen05 8-bit transpose B requires cta_group={cta_group} N in [{step}, 256] with step {step}, got N={n}"
        ));
    }
    Ok(())
}

// Engine kind -> descriptor flag bits it encodes (negate A/B, saturate D).
const DENSE_FLAG_ENCODINGS: &[(&str, i64)] = &[
    ("f16", (1 << 13) | (1 << 14)),
    ("tf32", (1 << 13) | (1 << 14)),
    ("f8f6f4", (1 << 13) | (1 << 14)),
    ("i8", 1 << 3),
];

/// Assemble a dense descriptor using its dtype, shape and flag encoding rows.
#[allow(clippy::too_many_arguments)]
pub fn encode_dense_instr_descriptor_fields(
    d_dtype: &str,
    a_dtype: &str,
    b_dtype: &str,
    m: i64,
    n: i64,
    k: i64,
    trans_a: bool,
    trans_b: bool,
    cta_group: i64,
    neg_a: bool,
    neg_b: bool,
    sat_d: bool,
    sparse: bool,
) -> AResult<i64> {
    let kind = dense_kind(d_dtype, a_dtype, b_dtype)?;
    validate_tcgen05_instruction_shape(kind, cta_group, m, n, k, sparse)?;
    if trans_a && !format_encoding(a_dtype).2 {
        return unsupported(format!("tcgen05 transpose A is invalid for {a_dtype}"));
    }
    if trans_b && !format_encoding(b_dtype).2 {
        return unsupported(format!("tcgen05 transpose B is invalid for {b_dtype}"));
    }
    validate_8bit_transpose_b_shape(b_dtype, trans_b, cta_group, n)?;
    let flag_bits = DENSE_FLAG_ENCODINGS
        .iter()
        .find(|row| row.0 == kind)
        .expect("mapped dense kind")
        .1;
    if (neg_a || neg_b) && flag_bits & ((1 << 13) | (1 << 14)) == 0 {
        return unsupported(format!("tcgen05 negate is invalid for kind {kind}"));
    }
    if sat_d && flag_bits & (1 << 3) == 0 {
        return unsupported(format!("tcgen05 saturation is invalid for kind {kind}"));
    }
    let mut value: i64 = i64::from(sparse) << 2;
    value |= i64::from(sat_d) << 3;
    value |= (format_encoding(d_dtype).1 & 0x3) << 4;
    value |= (format_encoding(a_dtype).1 & 0x7) << 7;
    value |= (format_encoding(b_dtype).1 & 0x7) << 10;
    value |= i64::from(neg_a) << 13;
    value |= i64::from(neg_b) << 14;
    value |= i64::from(trans_a) << 15;
    value |= i64::from(trans_b) << 16;
    value |= ((n >> 3) & 0x3F) << 17;
    value |= ((m >> 4) & 0x1F) << 24;
    Ok(value & 0xFFFF_FFFF)
}

/// `encode_block_scaled_instr_descriptor_fields`.
#[allow(clippy::too_many_arguments)]
pub fn encode_block_scaled_instr_descriptor_fields(
    d_dtype: &str,
    a_dtype: &str,
    b_dtype: &str,
    sfa_dtype: &str,
    sfb_dtype: &str,
    m: i64,
    n: i64,
    k: i64,
    trans_a: bool,
    trans_b: bool,
    cta_group: i64,
    neg_a: bool,
    neg_b: bool,
    sparse: bool,
) -> AResult<i64> {
    let encoding = block_encoding(d_dtype, a_dtype, b_dtype, sfa_dtype, sfb_dtype)?;
    let kind = encoding.3;
    validate_tcgen05_instruction_shape(kind, cta_group, m, n, k, sparse)?;
    if trans_a && !format_encoding(a_dtype).3 {
        return unsupported(format!(
            "tcgen05 block transpose A is invalid for {a_dtype}"
        ));
    }
    if trans_b && !format_encoding(b_dtype).3 {
        return unsupported(format!(
            "tcgen05 block transpose B is invalid for {b_dtype}"
        ));
    }
    validate_8bit_transpose_b_shape(b_dtype, trans_b, cta_group, n)?;
    let (a_format, b_format, scale_format) = (encoding.4, encoding.5, encoding.6);
    let mut value: i64 = i64::from(sparse) << 2;
    value |= (a_format & 0x7) << 7;
    value |= (b_format & 0x7) << 10;
    value |= i64::from(neg_a) << 13;
    value |= i64::from(neg_b) << 14;
    value |= i64::from(trans_a) << 15;
    value |= i64::from(trans_b) << 16;
    value |= ((n >> 3) & 0x3F) << 17;
    value |= (scale_format & 0x1) << 23;
    value |= ((m >> 4) & 0x1F) << 24;
    value |= i64::from(k == 96) << 31;
    Ok(value & 0xFFFF_FFFF)
}

/// `util::static_int` with the descriptor requirement wording.
fn static_int(ctx: &Ctx, value: &ObjectRef, field: &str, requirement: &str) -> AResult<i64> {
    crate::analyze::util::static_int(&ctx.analyzer, &prim(value)?, field, requirement)
}

fn static_bool(ctx: &Ctx, value: &ObjectRef, field: &str) -> AResult<bool> {
    let simplified = simplify(&ctx.analyzer, &prim(value)?)?;
    let node = oref(simplified);
    let Some(imm) = node.as_node::<IntImmObj>() else {
        return unsupported(format!(
            "{field} must specialize to a static bool, got {}",
            repr_text(value)?
        ));
    };
    if dtype_of(&node)? != "bool" {
        return unsupported(format!(
            "{field} must specialize to a static bool, got {}",
            repr_text(value)?
        ));
    }
    Ok(int_value(imm)? != 0)
}

fn null_handle(ctx: &Ctx, expression: &ObjectRef) -> AResult<bool> {
    let Some(call) = expression.as_node::<CallObj>() else {
        return Ok(false);
    };
    if call_op_name(call)?.as_deref() != Some("tirx.reinterpret") {
        return Ok(false);
    }
    if call.args.len() != 1 {
        return Ok(false);
    }
    match static_int(
        ctx,
        &oref(call.args.get(0)?),
        "null handle",
        "must be a static integer",
    ) {
        Ok(value) => Ok(value == 0),
        Err(Failure::Unsupported { .. }) => Ok(false),
        Err(error) => Err(error),
    }
}

fn require_signature(op_name: &str, args: &[ObjectRef], expected: &[&str]) -> AResult<()> {
    let mut actual: Vec<String> = Vec::new();
    for argument in args {
        actual.push(dtype_of(argument)?);
    }
    if actual.len() != expected.len() || actual.iter().zip(expected.iter()).any(|(a, e)| a != e) {
        return unsupported(format!(
            "{op_name} expects exact arguments {:?}, got {:?}",
            &expected, &actual
        ));
    }
    Ok(())
}

/// `parse_tcgen_descriptor_call`: `None` for any other call.
pub fn parse_tcgen_descriptor_call(
    ctx: &Ctx,
    node: &ObjectRef,
) -> AResult<Option<TcgenDescriptorCall>> {
    let Some(call) = node.as_node::<CallObj>() else {
        return Ok(None);
    };
    let Some(op_name) = call_op_name(call)? else {
        return Ok(None);
    };
    if !DESCRIPTOR_CALLS.contains(&op_name.as_str()) {
        return Ok(None);
    }
    let result_dtype = dtype_of(node)?;
    if !result_dtype.is_empty() {
        return unsupported(format!("{op_name} must return void, got {result_dtype}"));
    }
    let args: Vec<ObjectRef> = call.args.iter().map(oref).collect();
    if op_name == "tirx.cuda.tcgen05_encode_matrix_descriptor" {
        require_signature(
            &op_name,
            &args,
            &["handle", "handle", "int32", "int32", "int32"],
        )?;
        let matrix_source_is_null = null_handle(ctx, &args[1])?;
        return Ok(Some(TcgenDescriptorCall {
            args,
            matrix_source_is_null,
            encoded_u32: None,
        }));
    }
    let requirement = "must specialize to a static integer";
    if op_name == "tirx.cuda.tcgen05_encode_instr_descriptor" {
        require_signature(
            &op_name,
            &args,
            &[
                "handle", "", "", "", "int32", "int32", "int32", "bool", "bool", "int32", "bool",
                "bool", "bool", "bool",
            ],
        )?;
        let encoded = encode_dense_instr_descriptor_fields(
            &static_string(&args[1], "dense.d_dtype")?,
            &static_string(&args[2], "dense.a_dtype")?,
            &static_string(&args[3], "dense.b_dtype")?,
            static_int(ctx, &args[4], "dense.M", requirement)?,
            static_int(ctx, &args[5], "dense.N", requirement)?,
            static_int(ctx, &args[6], "dense.K", requirement)?,
            static_bool(ctx, &args[7], "dense.trans_a")?,
            static_bool(ctx, &args[8], "dense.trans_b")?,
            static_int(ctx, &args[9], "dense.n_cta_groups", requirement)?,
            static_bool(ctx, &args[10], "dense.neg_a")?,
            static_bool(ctx, &args[11], "dense.neg_b")?,
            static_bool(ctx, &args[12], "dense.sat_d")?,
            static_bool(ctx, &args[13], "dense.is_sparse")?,
        )?;
        return Ok(Some(TcgenDescriptorCall {
            args,
            matrix_source_is_null: false,
            encoded_u32: Some(encoded),
        }));
    }
    require_signature(
        &op_name,
        &args,
        &[
            "handle", "", "", "", "", "", "int32", "int32", "int32", "int32", "int32", "bool",
            "bool", "int32", "bool", "bool", "bool",
        ],
    )?;
    let encoded = encode_block_scaled_instr_descriptor_fields(
        &static_string(&args[1], "block.d_dtype")?,
        &static_string(&args[2], "block.a_dtype")?,
        &static_string(&args[3], "block.b_dtype")?,
        &static_string(&args[4], "block.sfa_dtype")?,
        &static_string(&args[5], "block.sfb_dtype")?,
        static_int(ctx, &args[8], "block.M", requirement)?,
        static_int(ctx, &args[9], "block.N", requirement)?,
        static_int(ctx, &args[10], "block.K", requirement)?,
        static_bool(ctx, &args[11], "block.trans_a")?,
        static_bool(ctx, &args[12], "block.trans_b")?,
        static_int(ctx, &args[13], "block.n_cta_groups", requirement)?,
        static_bool(ctx, &args[14], "block.neg_a")?,
        static_bool(ctx, &args[15], "block.neg_b")?,
        static_bool(ctx, &args[16], "block.is_sparse")?,
    )?;
    Ok(Some(TcgenDescriptorCall {
        args,
        matrix_source_is_null: false,
        encoded_u32: Some(encoded),
    }))
}

pub const TCGEN_DESCRIPTOR_LAYOUT: &str = "<artifact-tcgen-descriptor-layout>";

const TCGEN_DESCRIPTOR_LAYOUT_BY_ARCH: &[(&str, &str)] = &[
    ("sm_100a", "sm100"),
    ("sm_100f", "sm100"),
    ("sm_103a", "sm103"),
    ("sm_103f", "sm103"),
    ("sm_107a", "sm107"),
    ("sm_107f", "sm107"),
];

/// `require_tcgen_descriptor_layout`.
pub fn require_tcgen_descriptor_layout(func: &PrimFunc) -> AResult<&'static str> {
    let Some(arch) = cuda_arch(func)? else {
        return unsupported(format!(
            "TCGEN descriptors require PrimFunc attribute {:?} to select exact SM100/SM103/SM107 descriptor semantics",
            CUDA_ARCH_ATTR));
    };
    match TCGEN_DESCRIPTOR_LAYOUT_BY_ARCH
        .iter()
        .find(|(candidate, _)| *candidate == arch)
    {
        Some((_, layout)) => Ok(layout),
        None => {
            let mut supported: Vec<&str> = TCGEN_DESCRIPTOR_LAYOUT_BY_ARCH
                .iter()
                .map(|(candidate, _)| *candidate)
                .collect();
            supported.sort_unstable();
            unsupported(format!(
                "TCGEN descriptors have unsupported CUDA architecture {:?}; expected one of {}",
                &arch,
                supported.join(", ")
            ))
        }
    }
}

/// `tcgen_descriptor_variant`: legacy non-F8-CTA2 calls retain SM100 when no
/// architecture is authored.
pub fn tcgen_descriptor_variant(func: &PrimFunc) -> AResult<String> {
    let layout = if cuda_arch(func)?.is_some() {
        require_tcgen_descriptor_layout(func)?
    } else {
        "sm100"
    };
    Ok(format!(
        "v2::tcgen05::variant::MatrixDescriptorSm{}",
        &layout[2..]
    ))
}
