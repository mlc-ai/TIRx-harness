//! Validation and emission of scalar calls and collectives.

use crate::analyze::buffers::call_op_name;
use crate::analyze::memory::MemorySpace;
use crate::analyze::util::{
    as_buffer, buffer_name, buffer_scope, dtype_of, ffi_text, not_covered, oref, unmodeled,
    unsupported, AResult,
};
use crate::analyze::Ctx;
use crate::decode::Decoded;
use crate::emit::abi;
use crate::emit::calls::call_args;
use crate::emit::stmt::pointer_pointee_dtype;
use crate::emit::{join_control_provenance, ControlProvenance, Emitter, RustValue, Uniformity};
use crate::tables::{
    dtype_byte_len, expr_rust_type, is_integer_dtype, is_integer_rust_type, render_bitwise,
    stmt_rust_scalar_by_dtype,
};
use crate::tvm_compat::int_value;
use tvm::ir::StringImmObj;
use tvm::ir::{CallObj, IntImmObj, PrimExpr, TensorLoadObj};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

const REDUCTION_FLOATS: [&str; 4] = ["float16", "bfloat16", "float32", "float64"];
const FETCH_REGISTERS_32: [&str; 39] = [
    "smid",
    "tid.x",
    "tid.y",
    "tid.z",
    "ntid.x",
    "ntid.y",
    "ntid.z",
    "laneid",
    "warpid",
    "nwarpid",
    "ctaid.x",
    "ctaid.y",
    "ctaid.z",
    "nctaid.x",
    "nctaid.y",
    "nctaid.z",
    "clusterid.x",
    "clusterid.y",
    "clusterid.z",
    "nclusterid.x",
    "nclusterid.y",
    "nclusterid.z",
    "cluster_ctaid.x",
    "cluster_ctaid.y",
    "cluster_ctaid.z",
    "cluster_nctaid.x",
    "cluster_nctaid.y",
    "cluster_nctaid.z",
    "cluster_ctarank",
    "cluster_nctarank",
    "lanemask_eq",
    "lanemask_le",
    "lanemask_lt",
    "lanemask_ge",
    "lanemask_gt",
    "clock",
    "clock_hi",
    "globaltimer_lo",
    "globaltimer_hi",
];
const FETCH_REGISTERS_64: [&str; 3] = ["gridid", "clock64", "globaltimer"];

#[derive(Clone, Debug)]
pub struct CallSignature {
    pub op_name: String,
    pub result_dtype: String,
    pub arg_dtypes: Vec<String>,
}

impl CallSignature {
    pub fn render(&self) -> String {
        format!(
            "{}({})->{}",
            self.op_name,
            self.arg_dtypes.join(", "),
            self.result_dtype
        )
    }
}

fn signature(node: &ObjectRef, call: &CallObj) -> AResult<CallSignature> {
    let op_name = call_op_name(call)?.unwrap_or_else(|| "<missing-op>".to_owned());
    let mut arg_dtypes = Vec::new();
    for argument in call.args.iter() {
        arg_dtypes.push(dtype_of(&oref(argument))?);
    }
    Ok(CallSignature {
        op_name,
        result_dtype: dtype_of(node)?,
        arg_dtypes,
    })
}

fn exact(signature: &CallSignature, result: &str, args: &[&str]) -> Result<bool, String> {
    if signature.result_dtype == result
        && signature.arg_dtypes.len() == args.len()
        && signature
            .arg_dtypes
            .iter()
            .zip(args.iter())
            .all(|(actual, expected)| actual == expected)
    {
        return Ok(true);
    }
    let expected = CallSignature {
        op_name: signature.op_name.clone(),
        result_dtype: result.to_owned(),
        arg_dtypes: args.iter().map(|arg| (*arg).to_owned()).collect(),
    };
    Err(format!("expected {}", expected.render()))
}

fn is_scalar(ctx: &Ctx, dtype: &str) -> bool {
    ctx.schema.call_scalars.contains(dtype)
}

fn is_reduction_scalar(dtype: &str) -> bool {
    is_integer_dtype(dtype) || REDUCTION_FLOATS.contains(&dtype)
}

/// A pure call's signature mismatch, or whether the signature alone fixes its
/// form (`false` for the calls `resolve_pure_call` forms from static operands).

fn validate_prim_if_then_else(ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    let args = &s.arg_dtypes;
    let result = s.result_dtype.as_str();
    {
        if args.len() != 3 {
            return Err("expected condition, true value, and false value".into());
        }
        if args[0] != "bool" {
            return Err("condition must be bool".into());
        }
        if args[1] != args[2] || result != args[1] {
            return Err("branches and result must have the same dtype".into());
        }
        if !(is_scalar(ctx, &args[1]) || args[1] == "handle") {
            return Err("branch dtype is not a supported scalar or handle".into());
        }
        Ok(true)
    }
}

fn validate_reinterpret(ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    let args = &s.arg_dtypes;
    let result = s.result_dtype.as_str();
    {
        if args.len() != 1 {
            return Err("expected one operand".into());
        }
        let source = args[0].as_str();
        let target = result;
        if matches!(
            (source, target),
            ("handle", "handle") | ("handle", "uint64") | ("uint64", "handle")
        ) {
            return Ok(true);
        }
        let (Some(source_bits), Some(target_bits)) = (
            ctx.schema.call_dtype_bits(source),
            ctx.schema.call_dtype_bits(target),
        ) else {
            return Err("source and result must be equal-width scalars or handle<->uint64".into());
        };
        if source_bits != target_bits {
            return Err("source and result must have identical bit widths".into());
        }
        if matches!(
            (source, target),
            ("uint16", "float16")
                | ("float16", "uint16")
                | ("uint16", "bfloat16")
                | ("bfloat16", "uint16")
        ) {
            return Ok(true);
        }
        let source_vector = ctx.schema.vector_dtype_abi(source).is_some();
        let target_vector = ctx.schema.vector_dtype_abi(target).is_some();
        if source == target
            && (ctx.schema.reinterpret_identity_scalars.contains(source) || source_vector)
        {
            return Ok(true);
        }
        if ctx
            .schema
            .reinterpret_numeric_decode_pairs
            .contains(&(source.to_owned(), target.to_owned()))
        {
            return Ok(true);
        }
        if (ctx.schema.reinterpret_raw_scalars.contains(source) || source_vector)
            && (ctx.schema.reinterpret_raw_scalars.contains(target) || target_vector)
        {
            return Ok(true);
        }
        Err(
                "raw payload reinterpret is not modeled for scalar low-precision/storage-only dtype; use an explicitly supported packed storage dtype"
                    .into(),
            )
    }
}

fn validate_address_of(ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    let args = &s.arg_dtypes;
    let result = s.result_dtype.as_str();
    {
        if args.len() != 1 {
            return Err("expected one addressable operand".into());
        }
        let source = args[0].as_str();
        if source != "handle"
            && source != "float4_e2m1fn"
            && ctx.schema.call_dtype_bits(source).is_none()
        {
            return Err("operand dtype is not addressable".into());
        }
        if result != "handle" && result != "uint64" {
            return Err("result must be handle or uint64".into());
        }
        Ok(false)
    }
}

fn validate_cuda_mov_sreg(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    let args = &s.arg_dtypes;
    let result = s.result_dtype.as_str();
    {
        if args.len() != 2 {
            return Err("expected bit width and static register name".into());
        }
        if !is_integer_dtype(&args[0]) || !args[1].is_empty() {
            return Err("bit width must be integer and register name must be static".into());
        }
        if result != "int32" && result != "int64" {
            return Err("result must be int32 or int64".into());
        }
        Ok(false)
    }
}

fn validate_cuda_smem_addr_from_uint64(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    let args = &s.arg_dtypes;
    let result = s.result_dtype.as_str();
    {
        if result != "uint32" || args.len() != 1 {
            return Err("expected one address token returning uint32".into());
        }
        if args[0] != "handle" && args[0] != "uint64" {
            return Err("source address token must be handle or uint64".into());
        }
        Ok(true)
    }
}

fn validate_cuda_warp_reduce(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    let args = &s.arg_dtypes;
    let result = s.result_dtype.as_str();
    {
        if args.len() != 3 {
            return Err("expected value, operation, and width".into());
        }
        if !is_reduction_scalar(&args[0]) || result != args[0] {
            return Err("value and result must use the same supported numeric scalar dtype".into());
        }
        if !args[1].is_empty() {
            return Err("operation must be static string".into());
        }
        if !is_integer_dtype(&args[2]) {
            return Err("width must be an integer".into());
        }
        Ok(false)
    }
}

fn validate_cuda_cta_reduce(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    let args = &s.arg_dtypes;
    let result = s.result_dtype.as_str();
    {
        if args.len() != 4 {
            return Err("expected value, operation, num_warps, and scratch pointer".into());
        }
        if !is_reduction_scalar(&args[0]) || result != args[0] {
            return Err("value and result must use the same supported numeric scalar dtype".into());
        }
        if !args[1].is_empty() {
            return Err("operation must be static string".into());
        }
        if !is_integer_dtype(&args[2]) {
            return Err("num_warps must be an integer".into());
        }
        if args[3] != "handle" {
            return Err("scratch must be a pointer handle".into());
        }
        Ok(false)
    }
}

fn validate_tvm_warp_shuffle(ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    let args = &s.arg_dtypes;
    let result = s.result_dtype.as_str();
    {
        if args.len() != 5 {
            return Err("expected mask, value, source selector, width, and warp size".into());
        }
        if !is_integer_dtype(&args[0]) {
            return Err("mask must be an integer".into());
        }
        if !is_scalar(ctx, &args[1]) || result != args[1] {
            return Err("value and result must use the same supported scalar dtype".into());
        }
        if args[2..5].iter().any(|dtype| !is_integer_dtype(dtype)) {
            return Err("selector, width, and warp size must be integers".into());
        }
        Ok(true)
    }
}

fn validate_cuda_shfl_sync(ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    let args = &s.arg_dtypes;
    let result = s.result_dtype.as_str();
    {
        if args.len() != 4 {
            return Err("expected mask, value, source selector, and width".into());
        }
        if !is_integer_dtype(&args[0]) {
            return Err("mask must be an integer".into());
        }
        let value = args[1].as_str();
        if !(is_scalar(ctx, value) || value == "float16x2" || value == "bfloat16x2")
            || result != value
        {
            return Err("value and result must use the same supported scalar dtype".into());
        }
        if !is_integer_dtype(&args[2]) || !is_integer_dtype(&args[3]) {
            return Err("selector and width must be integers".into());
        }
        Ok(true)
    }
}

fn validate_cuda_syncthreads_and(ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    let args = &s.arg_dtypes;
    let result = s.result_dtype.as_str();
    {
        if result != "int64" || args.len() != 1 {
            return Err("expected one predicate returning int64".into());
        }
        if !is_scalar(ctx, &args[0]) {
            return Err("predicate must be a scalar".into());
        }
        Ok(true)
    }
}

fn validate_cuda_any_sync(ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    let args = &s.arg_dtypes;
    let result = s.result_dtype.as_str();
    {
        if args.len() != 2 || result != "int32" {
            return Err("expected mask and predicate returning int32".into());
        }
        if !is_integer_dtype(&args[0]) || !is_scalar(ctx, &args[1]) {
            return Err("mask must be integer and predicate must be scalar".into());
        }
        Ok(true)
    }
}

fn validate_cuda_ballot_sync(ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    let args = &s.arg_dtypes;
    let result = s.result_dtype.as_str();
    {
        if args.len() != 2 || result != "uint32" {
            return Err("expected mask and predicate returning uint32".into());
        }
        if !is_integer_dtype(&args[0]) || !is_scalar(ctx, &args[1]) {
            return Err("mask must be integer and predicate must be scalar".into());
        }
        Ok(true)
    }
}

fn validate_cuda_get_tmem_addr(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    let args = &s.arg_dtypes;
    let result = s.result_dtype.as_str();
    {
        if result != "uint32" || args.len() != 3 {
            return Err("expected encoded address and row/column offsets returning uint32".into());
        }
        let int32_or_uint32 = |dtype: &str| dtype == "int32" || dtype == "uint32";
        if !int32_or_uint32(&args[0]) || !int32_or_uint32(&args[1]) {
            return Err("encoded address and row offset must be int32 or uint32".into());
        }
        if !int32_or_uint32(&args[2]) {
            return Err("column offset must be int32 or uint32".into());
        }
        Ok(true)
    }
}

fn validate_isnullptr(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "bool", &["handle"])
}

fn validate_cuda_float_as_uint(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "uint32", &["float32"])
}

fn validate_cuda_uint_as_float(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "float32", &["uint32"])
}

fn validate_cuda_make_float2(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "uint64", &["float32", "float32"])
}

fn validate_cuda_float2_x(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "float32", &["uint64"])
}

fn validate_fma(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "float32", &["float32", "float32", "float32"])
}

fn validate_cuda_fdividef(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "float32", &["float32", "float32"])
}

fn validate_fabs(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "float32", &["float32"])
}

fn validate_cuda_thread_rank(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "int32", &[])
}

fn validate_cuda_elect_sync(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "uint32", &[])
}

fn validate_cuda_float22bfloat162_rn(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "uint32", &["float32", "float32"])
}

fn validate_cuda_fmul2_rn(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    {
        exact(s, "uint64", &["uint64", "uint64"])
    }
}

fn validate_cuda_reduce_add_sync_u32(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    {
        exact(s, "uint32", &["uint32", "uint32"])
    }
}

fn validate_tvm_warp_activemask(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "uint32", &[])
}

fn validate_popcount(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "uint32", &["uint32"])
}

fn validate_cuda_ffs_u32(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "int32", &["uint32"])
}

fn validate_cuda_bfloat1622float2(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "uint64", &["uint32"])
}

fn validate_cuda_hmin2(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "uint32", &["uint32", "uint32"])
}

fn validate_cuda_fp8x4_e4m3_from_float4(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    {
        exact(s, "uint32", &["float32", "float32", "float32", "float32"])
    }
}

fn validate_cuda_half2float(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "float32", &["float16"])
}

fn validate_cuda_bfloat162float(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "float32", &["bfloat16"])
}

fn validate_cuda_float22bfloat162_rn_from_float2(
    _ctx: &Ctx,
    s: &CallSignature,
) -> Result<bool, String> {
    exact(s, "uint32", &["uint64"])
}

fn validate_cuda_clock64(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "uint64", &[])
}

fn validate_cuda_iket_mark(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "", &[""])
}

fn validate_cuda_iket_official_event(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "uint32", &["int32", ""])
}

fn validate_cuda_iket_range_end(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "", &["uint32"])
}

fn validate_cuda_iket_range_pop(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "", &[])
}

fn validate_cuda_iket_range_push(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "", &[""])
}

fn validate_cuda_iket_range_start(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "uint32", &[""])
}

fn validate_cuda_iket_sentinel_token(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(s, "uint32", &[""])
}

fn validate_timer_init_cuda(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    {
        exact(s, "", &["handle", "handle", "handle", "int32", "int32"])
    }
}

fn validate_timer_start_cuda(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    exact(
        s,
        "",
        &["int32", "handle", "handle", "handle", "int32", "bool"],
    )
}

fn validate_timer_finalize_cuda(_ctx: &Ctx, s: &CallSignature) -> Result<bool, String> {
    {
        exact(s, "", &["handle", "handle", "handle", "int32", "bool"])
    }
}

/// `(signature, semantics, whether the signature alone fixes the form)`.

fn static_string(argument: &ObjectRef, signature: &CallSignature, field: &str) -> AResult<String> {
    match argument.as_node::<StringImmObj>() {
        Some(imm) => Ok(ffi_text(&imm.value)),
        None => unsupported(format!(
            "Call({}): {field} must be static",
            signature.render()
        )),
    }
}

fn address_of_parts(signature: &CallSignature, args: &[ObjectRef]) -> AResult<()> {
    if args[0].as_node::<TensorLoadObj>().is_none() {
        return unsupported(format!(
            "Call({}): address_of requires a concrete TensorLoad",
            signature.render()
        ));
    }

    Ok(())
}

fn cta_reduce_parts(signature: &CallSignature, args: &[ObjectRef]) -> AResult<(String, i64)> {
    let operation = static_string(&args[1], &signature, "operation")?;
    if !matches!(operation.as_str(), "sum" | "max" | "min") {
        return unsupported(format!(
            "Call({}): unsupported CTA reduction operation {:?}",
            signature.render(),
            &operation
        ));
    }
    let Some(count) = args[2]
        .as_node::<IntImmObj>()
        .and_then(|imm| int_value(imm).ok())
    else {
        return unsupported(format!(
            "Call({}): num_warps must be static",
            signature.render()
        ));
    };
    if !(1..=32).contains(&count) || count & (count - 1) != 0 {
        return unsupported(format!(
            "Call({}): num_warps must be a power of two in [1, 32]",
            signature.render()
        ));
    }

    Ok((operation, count))
}

fn warp_reduce_parts(signature: &CallSignature, args: &[ObjectRef]) -> AResult<(String, i64)> {
    let operation = static_string(&args[1], &signature, "operation")?;
    if !matches!(operation.as_str(), "sum" | "max" | "min") {
        return unsupported(format!(
            "Call({}): operation must be sum, max, or min",
            signature.render()
        ));
    }
    let Some(count) = args[2]
        .as_node::<IntImmObj>()
        .and_then(|imm| int_value(imm).ok())
    else {
        return unsupported(format!(
            "Call({}): width must be static",
            signature.render()
        ));
    };
    if !(2..=32).contains(&count) || count & (count - 1) != 0 {
        return unsupported(format!(
            "Call({}): width must be a power of two in [2, 32]",
            signature.render()
        ));
    }

    Ok((operation, count))
}

fn fetch_register_parts(signature: &CallSignature, args: &[ObjectRef]) -> AResult<String> {
    let width = args[0]
        .as_node::<IntImmObj>()
        .and_then(|imm| int_value(imm).ok());
    let Some(bits) = width.filter(|bits| *bits == 32 || *bits == 64) else {
        return unsupported(format!(
            "Call({}): register width must be static 32 or 64",
            signature.render()
        ));
    };
    let expected_result = format!("int{bits}");
    if signature.result_dtype != expected_result {
        return unsupported(format!(
            "Call({}): {bits}-bit fetch requires {expected_result} result",
            signature.render()
        ));
    }
    let register_name = static_string(&args[1], &signature, "register name")?;
    let register_name = register_name.trim_start_matches('%').to_owned();
    let allowed: &[&str] = if bits == 32 {
        &FETCH_REGISTERS_32
    } else {
        &FETCH_REGISTERS_64
    };
    if !allowed.contains(&register_name.as_str()) {
        return unmodeled(
            format!("call:{}", signature.op_name),
            format!(
                "Call({}): register {:?} is not modeled at {bits} bits",
                signature.render(),
                &register_name
            ),
        );
    }

    Ok(register_name)
}

const SM100_MAX_WARP_IDS: i64 = 64;
const F32X2_VARIANT: &str =
    "v2::reg::variant::F32x2Arithmetic<v2::reg::variant::Rn, v2::reg::variant::Ftz>";

impl<'a> Emitter<'a> {
    // ------------------------------------------------------------------
    // Pure scalar calls.
    // ------------------------------------------------------------------

    fn pure_result_type(&self, signature: &CallSignature) -> AResult<String> {
        let dtype = &signature.result_dtype;
        if dtype == "handle" {
            return Ok("u64".to_owned());
        }
        expr_rust_type(self.ctx.schema, dtype)
    }

    fn emit_arguments(&mut self, expr: &ObjectRef) -> AResult<Vec<RustValue>> {
        let mut values = Vec::new();
        for argument in call_args(expr) {
            values.push(self.emit_expr(&argument)?);
        }
        Ok(values)
    }

    fn emit_unary_atom(
        &mut self,
        expr: &ObjectRef,
        name: &str,
        result_type: &str,
        method: &str,
    ) -> AResult<RustValue> {
        let arguments = self.emit_arguments(expr)?;
        self.emit_call_atom(name, &arguments, result_type, |codes| {
            format!("({}){method}", codes[0])
        })
    }

    /// Instrument a call result with the call node itself as the context.
    fn instrument_call(
        &mut self,
        expr: &ObjectRef,
        inputs: &[&RustValue],
        output: RustValue,
    ) -> AResult<RustValue> {
        self.instrument(expr, "Call:if_then_else", inputs, output)
    }

    fn emit_address_of(&mut self, args: &[ObjectRef]) -> AResult<RustValue> {
        let Some(load) = args[0].as_node::<TensorLoadObj>() else {
            return unsupported("address_of currently requires a concrete TensorLoad operand");
        };
        let Some(source) = as_buffer(&oref(load.source.clone())) else {
            return not_covered("address_of source is not a typed buffer");
        };
        let indices: Vec<PrimExpr> = load.indices.iter().collect();
        let pointer = self.emit_buffer_address(&source, &indices)?;
        let scope = buffer_scope(&source);
        if matches!(
            scope.as_str(),
            "local" | "local_scalar" | "register" | "reg"
        ) {
            return Ok(pointer);
        }
        let address = self.temp("address_of");
        self.emit_line(&format!(
            "let {address} = ({}).generic_addresses_u64(&ctx, ctx.active_mask())?;",
            pointer.code
        ));
        Ok(RustValue::new(address, "u64", Uniformity::Varying))
    }

    /// `emit_buffer_address`.
    pub fn emit_buffer_address(
        &mut self,
        buffer: &tvm::tirx::BufferVar,
        indices: &[PrimExpr],
    ) -> AResult<RustValue> {
        if self.memory_plan.resolve(buffer)?.space == MemorySpace::Tmem {
            return unsupported(format!(
                "buffer:{}:address_of on TMEM is not implemented",
                buffer_name(buffer)
            ));
        }
        let info = self.ctx.inspect_layout(buffer, &self.bindings)?;
        let mut owner_axes: Vec<String> = info
            .physical_axes
            .iter()
            .filter(|axis| axis.as_str() != "m")
            .cloned()
            .collect();
        owner_axes.sort();
        owner_axes.dedup();
        if !owner_axes.is_empty() {
            return unsupported(format!(
                "buffer:{}:address_of with owner axes {:?} is not implemented",
                buffer_name(buffer),
                &owner_axes
            ));
        }
        let index = self.physical_index(buffer, indices)?;
        let buffer_ref = self.buffer_ref(buffer)?;
        let mut pointer_index = index.code.clone();
        if info.dtype == "float4_e2m1fn" {
            let byte_index = self.control_name("float4_address_byte_index");
            self.emit_line(&format!(
                "let {byte_index} = WarpValue::from_fn(|lane| {}[lane].div_euclid(2_i64));",
                index.code
            ));
            pointer_index = byte_index;
        }
        Ok(RustValue::new(
            abi::physical_ptr(&buffer_ref, &pointer_index, info.itemsize),
            "PhysicalPtr",
            Uniformity::Varying,
        ))
    }

    fn emit_reinterpret(
        &mut self,
        expr: &ObjectRef,
        signature: &CallSignature,
    ) -> AResult<RustValue> {
        let args = call_args(expr);
        let has_handle = signature.result_dtype == "handle"
            || signature.arg_dtypes.iter().any(|dtype| dtype == "handle");
        if has_handle {
            let argument = self.emit_expr(&args[0])?;
            if argument.rust_type == "PhysicalPtr"
                && signature.arg_dtypes[0] == "handle"
                && signature.result_dtype == "handle"
            {
                let materialized = self.temp("reinterpret_pointer");
                let mut pointer_code = format!("({}).clone()", argument.code);
                let target_dtype = pointer_pointee_dtype(expr);
                if let Some(dtype) =
                    target_dtype.filter(|dtype| !matches!(dtype.as_str(), "" | "handle" | "void"))
                {
                    pointer_code = format!(
                        "{pointer_code}.with_pointee_itemsize({}_usize)",
                        dtype_byte_len(self.ctx.schema, &dtype)?
                    );
                }
                self.emit_line(&format!("let {materialized} = {pointer_code};"));
                return Ok(RustValue::new(
                    materialized,
                    "PhysicalPtr",
                    argument.uniformity,
                ));
            }
            let (materialized, uniformity) = if argument.rust_type == "PhysicalPtr" {
                let materialized = self.temp("reinterpret_address");
                self.emit_line(&format!(
                    "let {materialized} = ({}).generic_addresses_u64(&ctx, ctx.active_mask())?;",
                    argument.code
                ));
                (materialized, Uniformity::Varying)
            } else {
                let argument = self.as_warp_value(argument);
                if argument.rust_type != "u64" {
                    return unsupported(format!(
                        "handle<->uint64 reinterpret lowered to {}, expected u64",
                        argument.rust_type
                    ));
                }
                let materialized = self.temp("reinterpret_address");
                self.emit_line(&format!(
                    "let {materialized} = ({}).clone();",
                    argument.code
                ));
                (materialized, argument.uniformity)
            };
            let output = RustValue::new(materialized, "u64", uniformity);
            return self.instrument(expr, "Call:reinterpret", &[&output.clone()], output);
        }
        let result_type = self.pure_result_type(signature)?;
        let source_dtype = signature.arg_dtypes[0].clone();
        let target_dtype = signature.result_dtype.clone();
        let mut arguments = Vec::new();
        for argument in self.emit_arguments(expr)? {
            arguments.push(self.observe_pointer_bits(argument));
        }
        let source_type = arguments[0].rust_type.clone();
        let codec = match (source_dtype.as_str(), target_dtype.as_str()) {
            ("uint16", "float16") => Some("fp16_bits_to_f32"),
            ("uint16", "bfloat16") => Some("bf16_bits_to_f32"),
            ("float16", "uint16") => Some("decoded_fp16_to_bits"),
            ("bfloat16", "uint16") => Some("decoded_bf16_to_bits"),
            _ => None,
        };
        let result_type_for_atom = result_type.clone();
        self.emit_quantized_call_atom(
            codec.unwrap_or("reinterpret"),
            &arguments,
            &result_type,
            |codes| match codec {
                Some(codec) => format!("{codec}({} as _)", codes[0]),
                None => reinterpret_atom(&codes[0], &source_type, &result_type_for_atom),
            },
            // Decoded storage bits are already in the target format; rounding again
            // would quiet signaling NaNs and corrupt a bit-preserving reinterpret.
            matches!(target_dtype.as_str(), "float16" | "bfloat16")
                .then_some(target_dtype.as_str()),
        )
    }

    fn emit_thread_rank(&mut self, expr: &ObjectRef) -> AResult<RustValue> {
        if self.selected_lane.is_some() {
            return Ok(self.emit_varying_value(
                "thread_rank",
                "(ctx.warp_id_in_cta() * WARP_SIZE + lane) as i32",
                "i32",
                ControlProvenance::None,
            ));
        }
        let site = self.expr_site(expr);
        let warp_id = self.temp("thread_rank_warp_id");
        let scaled = self.temp("thread_rank_scaled_warp");
        let lane_id = self.temp("thread_rank_lane_id");
        let rank_u32 = self.temp("thread_rank_u32");
        let name = self.temp("thread_rank");
        let mov_warp = abi::lane_call(
            "reg::mov",
            &site,
            &["()".to_owned()],
            Some("v2::reg::variant::WarpIdInCta"),
        );
        self.emit_line(&format!("let {warp_id} = {mov_warp};"));
        let mul = abi::lane_call(
            "reg::mul",
            &site,
            &[format!("({warp_id}, {})", abi::splat("WARP_SIZE as u32"))],
            Some("v2::reg::variant::U32"),
        );
        self.emit_line(&format!("let {scaled} = {mul};"));
        let mov_lane = abi::lane_call(
            "reg::mov",
            &site,
            &["()".to_owned()],
            Some("v2::reg::variant::LaneId"),
        );
        self.emit_line(&format!("let {lane_id} = {mov_lane};"));
        let add = abi::lane_call(
            "reg::add",
            &site,
            &[format!("({scaled}, {lane_id})")],
            Some("v2::reg::variant::U32"),
        );
        self.emit_line(&format!("let {rank_u32} = {add};"));
        let cvt = abi::lane_call(
            "reg::cvt",
            &site,
            &[rank_u32.clone()],
            Some("v2::reg::variant::Cvt<v2::reg::variant::U32, v2::reg::variant::I32>"),
        );
        self.emit_line(&format!("let {name} = {cvt};"));
        self.emit_line(&format!("let {name} = v2_register_out({name});"));
        Ok(RustValue::new(name, "i32", Uniformity::Varying))
    }

    fn emit_fetch_register(
        &mut self,
        signature: &CallSignature,
        register_name: &str,
    ) -> AResult<RustValue> {
        let result_type = self.pure_result_type(signature)?;
        const ZERO: [&str; 15] = [
            "tid.y",
            "tid.z",
            "ctaid.y",
            "ctaid.z",
            "clusterid.y",
            "clusterid.z",
            "cluster_ctaid.y",
            "cluster_ctaid.z",
            "clock",
            "clock_hi",
            "globaltimer_lo",
            "globaltimer_hi",
            "gridid",
            "clock64",
            "globaltimer",
        ];
        const ONE: [&str; 8] = [
            "ntid.y",
            "ntid.z",
            "nctaid.y",
            "nctaid.z",
            "nclusterid.y",
            "nclusterid.z",
            "cluster_nctaid.y",
            "cluster_nctaid.z",
        ];
        let uniform = match register_name {
            "smid" => Some("0_u32".to_owned()),
            "warpid" => Some("ctx.warp_id_in_cta()".to_owned()),
            "nwarpid" => Some(format!("{SM100_MAX_WARP_IDS}_usize")),
            "ctaid.x" => Some("ctx.global_cta_id()".to_owned()),
            "nctaid.x" => Some("ctx.topology().cta_count()".to_owned()),
            "clusterid.x" => Some("ctx.cluster_id()".to_owned()),
            "nclusterid.x" => Some("ctx.topology().clusters()".to_owned()),
            "cluster_ctaid.x" => Some("ctx.cta_id_in_cluster()".to_owned()),
            "cluster_nctaid.x" => Some("ctx.topology().ctas_per_cluster()".to_owned()),
            "cluster_ctarank" => Some("ctx.cta_id_in_cluster()".to_owned()),
            "cluster_nctarank" => Some("ctx.topology().ctas_per_cluster()".to_owned()),
            "ntid.x" => Some("ctx.topology().warps_per_cta() * WARP_SIZE".to_owned()),
            _ => None,
        };
        if ZERO.contains(&register_name) || ONE.contains(&register_name) {
            let value = if ONE.contains(&register_name) { 1 } else { 0 };
            return Ok(RustValue::new(
                format!("{value}_{result_type}"),
                result_type,
                Uniformity::Uniform,
            ));
        }
        if let Some(code) = uniform {
            return Ok(RustValue::new(
                format!("({code}) as {result_type}"),
                result_type,
                Uniformity::Uniform,
            ));
        }
        if register_name == "laneid" {
            return Ok(self.emit_varying_value(
                "laneid",
                &format!("lane as {result_type}"),
                &result_type,
                ControlProvenance::None,
            ));
        }
        if register_name == "tid.x" {
            return Ok(self.emit_varying_value(
                "tid_x",
                &format!("(ctx.warp_id_in_cta() * WARP_SIZE + lane) as {result_type}"),
                &result_type,
                ControlProvenance::None,
            ));
        }
        let lanemask = match register_name {
            "lanemask_eq" => Some("1_u32 << lane"),
            "lanemask_le" => Some("u32::MAX >> (31_usize - lane)"),
            "lanemask_lt" => Some("(1_u32 << lane).wrapping_sub(1_u32)"),
            "lanemask_ge" => Some("u32::MAX << lane"),
            "lanemask_gt" => Some("(u32::MAX << lane) & !(1_u32 << lane)"),
            _ => None,
        };
        if let Some(expression) = lanemask {
            return Ok(self.emit_varying_value(
                &register_name,
                &format!("({expression}) as {result_type}"),
                &result_type,
                ControlProvenance::None,
            ));
        }
        unsupported(format!(
            "fetch_register lowering missed validated register {:?}",
            &register_name
        ))
    }

    fn emit_shared_address(
        &mut self,
        expr: &ObjectRef,
        name: &str,
        function: &str,
    ) -> AResult<RustValue> {
        let arguments = self.emit_arguments(expr)?;
        let argument = &arguments[0];
        if argument.rust_type == "PhysicalPtr" {
            return Ok(RustValue::new(
                argument.code.clone(),
                "PhysicalPtr",
                argument.uniformity,
            ));
        }
        self.emit_call_atom(name, &arguments, "u32", |codes| {
            format!("{function}(({}) as u64)", codes[0])
        })
    }

    pub fn emit_bitwise_node(
        &mut self,
        expr: &ObjectRef,
        name: &str,
        operands: &[ObjectRef],
    ) -> AResult<RustValue> {
        let result_type = expr_rust_type(self.ctx.schema, &dtype_of(expr)?)?;
        if result_type != "bool" && !is_integer_rust_type(&result_type) {
            return unsupported("Bitwise operations require a scalar boolean or integer dtype");
        }
        let mut arguments = Vec::new();
        for (index, operand) in operands.iter().enumerate() {
            let value = self.emit_expr(operand)?;
            let value = self.observe_pointer_bits(value);
            // Shared warp helpers require a typed u32 shift-count operand.
            let target = if matches!(name, "shift_left" | "shift_right") && index == 1 {
                "u32"
            } else {
                &result_type
            };
            arguments.push(self.coerce_value(value, target, name)?);
        }
        self.with_call_expr(expr, |emitter| {
            emitter.emit_call_atom(name, &arguments, &result_type, |codes| {
                render_bitwise(name, codes)
            })
        })
    }

    fn emit_v2_register_call_expr(
        &mut self,
        expr: &ObjectRef,
        mnemonic: &str,
        variant: &str,
        argument_types: &[&str],
        result_type: &str,
    ) -> AResult<RustValue> {
        let arguments = self.emit_arguments(expr)?;
        self.emit_v2_register_call(
            expr,
            mnemonic,
            variant,
            &arguments,
            argument_types,
            result_type,
        )
    }

    pub fn emit_v2_register_call(
        &mut self,
        expr: &ObjectRef,
        mnemonic: &str,
        variant: &str,
        arguments: &[RustValue],
        argument_types: &[&str],
        result_type: &str,
    ) -> AResult<RustValue> {
        if arguments.len() != argument_types.len() {
            return Err(crate::analyze::util::Failure::Ffi(
                crate::analyze::util::ffi_error(&format!(
                    "{mnemonic} argument contract is inconsistent"
                )),
            ));
        }
        let mut typed = Vec::new();
        for (argument, dtype) in arguments.iter().zip(argument_types.iter()) {
            let value = self.coerce_value(argument.clone(), dtype, mnemonic)?;
            typed.push(self.as_warp_value(value));
        }
        let operands: Vec<String> = typed
            .iter()
            .map(|argument| format!("v2_register(({}).clone())", argument.code))
            .collect();
        let args = if operands.len() == 1 {
            operands[0].clone()
        } else {
            format!("({})", operands.join(", "))
        };
        let site = self.expr_site(expr);
        let raw_result = self.temp(&format!("{}_raw", mnemonic.replace('.', "_")));
        let result = self.temp(&mnemonic.replace('.', "_"));
        let family = mnemonic.split('.').next().unwrap_or(mnemonic);
        let call = abi::lane_call(&format!("reg::{family}"), &site, &[args], Some(variant));
        self.emit_line(&format!("let {raw_result} = {call};"));
        self.emit_line(&format!("let {result} = v2_register_out({raw_result});"));
        let output = RustValue {
            control_provenance: join_control_provenance(typed.iter()),
            ..RustValue::new(result, result_type, Uniformity::Varying)
        };
        let Some(call_expr) = self.call_expr_stack.last().cloned() else {
            return Ok(output);
        };
        let inputs: Vec<&RustValue> = typed.iter().collect();
        self.instrument(&call_expr, &format!("Call:{mnemonic}"), &inputs, output)
    }

    // ------------------------------------------------------------------
    // Warp collectives.
    // ------------------------------------------------------------------

    fn emit_warp_shuffle_call(
        &mut self,
        expr: &ObjectRef,
        result_type: &str,
        mode: &str,
    ) -> AResult<RustValue> {
        let args = call_args(expr);
        if args.len() != 4 && args.len() != 5 {
            return unsupported(format!(
                "Warp shuffle lowering expected four or five operands, got {}",
                args.len()
            ));
        }
        let masks = self.emit_expr(&args[0])?;
        let masks = self.coerce_value(masks, "u32", "shuffle_participants")?;
        let participant_masks = self.as_warp_value(masks);
        let values = self.emit_expr(&args[1])?;
        let values = self.as_warp_value(values);
        let selectors = self.emit_expr(&args[2])?;
        let selectors = self.as_i64(selectors)?;
        let selectors = self.as_warp_value(selectors);
        let widths = self.emit_expr(&args[3])?;
        let widths = self.as_i64(widths)?;
        let widths = self.as_warp_value(widths);
        let warp_sizes = if args.len() == 5 {
            let value = self.emit_expr(&args[4])?;
            let value = self.as_i64(value)?;
            Some(self.as_warp_value(value))
        } else {
            None
        };
        let source_op_id = self.lowered_instruction_site(expr);
        let selectors_u32 = self.temp("shuffle_selector");
        let controls = self.temp("shuffle_control");
        self.emit_line(&format!(
            "let mut {selectors_u32} = WarpValue::splat(0_u32);"
        ));
        self.emit_line(&format!("let mut {controls} = WarpValue::splat(0_u32);"));
        self.emit_line("for lane in ctx.active_mask() {");
        self.emit_line(&format!("    let selector = {}[lane];", selectors.code));
        self.emit_line(&format!("    let width = {}[lane];", widths.code));
        let warp_size = match &warp_sizes {
            None => "WARP_SIZE as i64".to_owned(),
            Some(value) => format!("{}[lane]", value.code),
        };
        self.emit_line(&format!("    let warp_size = {warp_size};"));
        self.emit_line("    if selector < 0 || warp_size != WARP_SIZE as i64 || width <= 0 || width > WARP_SIZE as i64 || (WARP_SIZE as i64) % width != 0_i64 {");
        self.emit_line(
            "        return Err(EngineError::message(\"invalid warp shuffle selector/width\"));",
        );
        self.emit_line("    }");
        self.emit_line("    let width = width as u32;");
        self.emit_line("    let segment_mask = !(width - 1_u32) & 0x1f_u32;");
        let clamp = if mode == "up" {
            "0_u32"
        } else {
            "width - 1_u32"
        };
        self.emit_line(&format!("    let clamp = {clamp};"));
        self.emit_line(&format!("    {selectors_u32}[lane] = selector as u32;"));
        self.emit_line(&format!(
            "    {controls}[lane] = (segment_mask << 8_u32) | clamp;"
        ));
        self.emit_line("}");
        let mode_marker = match mode {
            "index" => "Index",
            "up" => "Up",
            "down" => "Down",
            _ => "Butterfly",
        };
        let all_inputs: Vec<&RustValue> = [&participant_masks, &values, &selectors, &widths]
            .into_iter()
            .chain(warp_sizes.iter())
            .collect();
        let provenance = join_control_provenance(all_inputs.iter().copied());
        let carrier: Option<(&str, &str, &str, &str)> = match result_type {
            "i8" => Some(("I32", "i32", "as i32", "as i8")),
            "i16" => Some(("I32", "i32", "as i32", "as i16")),
            "i32" => Some(("I32", "i32", "", "")),
            "u8" => Some(("U32", "u32", "as u32", "as u8")),
            "u16" => Some(("U32", "u32", "as u32", "as u16")),
            "u32" => Some(("U32", "u32", "", "")),
            "f32" => Some(("F32", "f32", "", "")),
            "bool" => Some(("U32", "u32", "bool", "bool")),
            _ => None,
        };
        if carrier.is_none() && matches!(result_type, "i64" | "u64" | "f64") {
            let bits = self.temp("shuffle_b64_bits");
            let low = self.temp("shuffle_b64_low");
            let high = self.temp("shuffle_b64_high");
            let to_bits = match result_type {
                "i64" => format!("{}[lane] as u64", values.code),
                "u64" => format!("{}[lane]", values.code),
                _ => format!("{}[lane].to_bits()", values.code),
            };
            self.emit_line(&format!(
                "let {bits} = WarpValue::from_fn(|lane| {to_bits});"
            ));
            self.emit_line(&format!(
                "let {low} = WarpValue::from_fn(|lane| {bits}[lane] as u32);"
            ));
            self.emit_line(&format!(
                "let {high} = WarpValue::from_fn(|lane| ({bits}[lane] >> 32_u32) as u32);"
            ));
            let mut outputs: Vec<String> = Vec::new();
            for (piece, source) in [low, high].iter().enumerate() {
                let raw_piece = self.temp(&format!("shuffle_{mode}_b64_raw"));
                let output_piece = self.temp(&format!("shuffle_{mode}_b64"));
                let site = if piece == 0 {
                    format!("{source_op_id}_u64")
                } else {
                    format!("(0x8000_0000_0000_0000_u64 | {source_op_id}_u64)")
                };
                let call = abi::warp_call(
                    "warp::shfl_sync",
                    &abi::site_expr(&site),
                    &[format!(
                        "(v2_register(({}).clone()), v2_register(({source}).clone()), v2_register(({selectors_u32}).clone()), v2_register(({controls}).clone()))",
                        participant_masks.code
                    )],
                    Some(&format!(
                        "v2::warp::variant::Shfl<v2::reg::variant::U32, v2::warp::variant::{mode_marker}>"
                    )),
                    None,
                    false,
                    true,
                );
                self.emit_line(&format!("let {raw_piece} = {call};"));
                self.emit_line(&format!(
                    "let {output_piece} = v2_register_out({raw_piece});"
                ));
                outputs.push(output_piece);
            }
            let result = self.temp(&format!("shuffle_{mode}"));
            let mut reconstructed = format!(
                "({}[lane] as u64) | (({}[lane] as u64) << 32_u32)",
                outputs[0], outputs[1]
            );
            if result_type == "i64" {
                reconstructed = format!("({reconstructed}) as i64");
            } else if result_type == "f64" {
                reconstructed = format!("f64::from_bits({reconstructed})");
            }
            self.emit_line(&format!(
                "let {result} = WarpValue::from_fn(|lane| {reconstructed});"
            ));
            return Ok(RustValue {
                control_provenance: provenance,
                ..RustValue::new(result, result_type, Uniformity::Varying)
            });
        }
        let Some((type_marker, carrier_type, encode, decode)) = carrier else {
            return unsupported(format!(
                "shfl.sync result type {:?} has no exact PTX b32 carrier",
                result_type
            ));
        };
        let mut shuffle_values = values.clone();
        if !encode.is_empty() {
            let encoded = self.temp("shuffle_carrier");
            let expression = if encode == "bool" {
                format!("u32::from({}[lane])", values.code)
            } else {
                format!("{}[lane] {encode}", values.code)
            };
            self.emit_line(&format!(
                "let {encoded} = WarpValue::from_fn(|lane| {expression});"
            ));
            shuffle_values = RustValue::new(encoded, carrier_type, Uniformity::Varying);
        }
        let raw_result = self.temp(&format!("shuffle_{mode}_raw"));
        let call = abi::warp_call(
            "warp::shfl_sync",
            &abi::site(source_op_id as u64),
            &[format!(
                "(v2_register(({}).clone()), v2_register(({}).clone()), v2_register(({selectors_u32}).clone()), v2_register(({controls}).clone()))",
                participant_masks.code, shuffle_values.code
            )],
            Some(&format!(
                "v2::warp::variant::Shfl<v2::reg::variant::{type_marker}, v2::warp::variant::{mode_marker}>"
            )),
            None,
            false,
            true,
        );
        self.emit_line(&format!("let {raw_result} = {call};"));
        let carrier_result = self.temp(&format!("shuffle_{mode}_carrier"));
        self.emit_line(&format!(
            "let {carrier_result} = v2_register_out({raw_result});"
        ));
        let mut result = carrier_result.clone();
        if !decode.is_empty() && decode != "bool" {
            result = self.temp(&format!("shuffle_{mode}"));
            self.emit_line(&format!(
                "let {result} = WarpValue::from_fn(|lane| {carrier_result}[lane] {decode});"
            ));
        }
        if result_type == "bool" {
            let mask = self.temp("shuffle_mask");
            self.emit_line(&format!(
                "let {mask} = {carrier_result}.to_mask(|_, value| *value != 0_u32) & ctx.active_mask();"
            ));
            return Ok(RustValue {
                control_provenance: provenance,
                ..RustValue::mask(mask)
            });
        }
        Ok(RustValue {
            control_provenance: provenance,
            ..RustValue::new(result, result_type, Uniformity::Varying)
        })
    }

    fn emit_redux(
        &mut self,
        expr: &ObjectRef,
        prefix: &str,
        operation: &str,
    ) -> AResult<RustValue> {
        let args = call_args(expr);
        let mask_value = self.emit_expr(&args[0])?;
        let mask_value = self.coerce_value(
            mask_value,
            "u32",
            if prefix == "reduce_add_u32" {
                "reduce_participants"
            } else {
                "min_participants"
            },
        )?;
        let mask_value = self.as_warp_value(mask_value);
        let values = self.emit_expr(&args[1])?;
        let values = self.coerce_value(
            values,
            "u32",
            if prefix == "reduce_add_u32" {
                "reduce_values"
            } else {
                "min_values"
            },
        )?;
        let values = self.as_warp_value(values);
        let result = self.temp(prefix);
        let site = self.expr_site(expr);
        let call = abi::warp_call(
            "warp::redux_sync",
            &site,
            &[format!(
                "(v2_register(({}).clone()), v2_register(({}).clone()))",
                mask_value.code, values.code
            )],
            Some(&format!(
                "v2::warp::variant::Redux<v2::reg::variant::U32, v2::warp::variant::{operation}>"
            )),
            None,
            false,
            true,
        );
        self.emit_line(&format!("let {result} = {call};"));
        self.emit_line(&format!("let {result} = v2_register_out({result});"));
        Ok(RustValue {
            control_provenance: join_control_provenance([&mask_value, &values]),
            ..RustValue::new(result, "u32", Uniformity::Varying)
        })
    }

    fn emit_any_sync(&mut self, expr: &ObjectRef) -> AResult<RustValue> {
        let args = call_args(expr);
        let mask_value = self.emit_expr(&args[0])?;
        let mask_value = self.coerce_value(mask_value, "u32", "any_participants")?;
        let mask_value = self.as_warp_value(mask_value);
        let predicate = self.emit_expr(&args[1])?;
        let predicate = self.coerce_value(predicate, "bool", "any_predicate")?;
        let predicate = self.as_warp_value(predicate);
        let raw_result = self.temp("any_sync_raw");
        let result = self.temp("any_sync");
        let site = self.expr_site(expr);
        let call = abi::warp_call(
            "warp::vote_sync",
            &site,
            &[format!(
                "(v2_register(({}).clone()), v2_register(({}).clone()))",
                mask_value.code, predicate.code
            )],
            Some("v2::warp::variant::Any"),
            None,
            false,
            true,
        );
        self.emit_line(&format!("let {raw_result} = {call};"));
        self.emit_line(&format!(
            "let {raw_result} = v2_register_out({raw_result});"
        ));
        self.emit_line(&format!(
            "let {result} = WarpValue::from_fn(|lane| i32::from({raw_result}[lane]));"
        ));
        Ok(RustValue {
            control_provenance: join_control_provenance([&mask_value, &predicate]),
            ..RustValue::new(result, "i32", Uniformity::Varying)
        })
    }

    fn emit_ballot_sync(&mut self, expr: &ObjectRef) -> AResult<RustValue> {
        let args = call_args(expr);
        let mask_value = self.emit_expr(&args[0])?;
        let mask_value = self.coerce_value(mask_value, "u32", "ballot_participants")?;
        let mask_value = self.as_warp_value(mask_value);
        let predicate = self.emit_expr(&args[1])?;
        let predicate = self.coerce_value(predicate, "bool", "ballot_predicate")?;
        let predicate = self.as_warp_value(predicate);
        let result = self.temp("ballot_sync");
        let site = self.expr_site(expr);
        let call = abi::warp_call(
            "warp::vote_sync",
            &site,
            &[format!(
                "(v2_register(({}).clone()), v2_register(({}).clone()))",
                mask_value.code, predicate.code
            )],
            Some("v2::warp::variant::Ballot"),
            None,
            false,
            true,
        );
        self.emit_line(&format!("let {result} = {call};"));
        self.emit_line(&format!("let {result} = v2_register_out({result});"));
        Ok(RustValue {
            control_provenance: join_control_provenance([&mask_value, &predicate]),
            ..RustValue::new(result, "u32", Uniformity::Varying)
        })
    }

    fn emit_warp_reduce(
        &mut self,
        expr: &ObjectRef,
        result_type: &str,
        operation: &str,
        width: i64,
    ) -> AResult<RustValue> {
        let args = call_args(expr);
        let value = self.emit_expr(&args[0])?;
        let value = self.as_warp_value(value);
        if value.rust_type != result_type {
            return unsupported("warp_reduce value/result Rust types disagree");
        }
        let result = self.temp(&format!("warp_reduce_{operation}"));
        let logical_dtype = dtype_of(expr)?;
        let site = self.expr_site(expr);
        let participants = self.temp("warp_reduce_participants");
        let controls = self.temp("warp_reduce_controls");
        let delta = self.temp("warp_reduce_delta");
        self.emit_line(&format!("let {participants} = WarpValue::splat(u32::MAX);"));
        let segment_mask = (!(width - 1)) & 0x1F;
        let control = (segment_mask << 8) | (width - 1);
        self.emit_line(&format!(
            "let {controls} = WarpValue::splat({control}_u32);"
        ));
        self.emit_line(&format!("let mut {result} = ({}).clone();", value.code));
        self.emit_line(&format!("let mut {delta}: u32 = {}_u32;", width / 2));
        self.emit_line(&format!("while {delta} > 0_u32 {{"));
        let selectors = self.temp("warp_reduce_selectors");
        self.emit_line(&format!("    let {selectors} = WarpValue::splat({delta});"));
        let shuffle_operands = |carrier: &str| -> String {
            format!(
                "(v2_register(({participants}).clone()), v2_register(({carrier}).clone()), v2_register(({selectors}).clone()), v2_register(({controls}).clone()))"
            )
        };
        let shuffled: String;
        if matches!(logical_dtype.as_str(), "int64" | "uint64" | "float64") {
            let bits = self.temp("warp_reduce_bits");
            let low = self.temp("warp_reduce_low");
            let high = self.temp("warp_reduce_high");
            let to_bits = match logical_dtype.as_str() {
                "int64" => format!("{result}[lane] as u64"),
                "uint64" => format!("{result}[lane]"),
                _ => format!("{result}[lane].to_bits()"),
            };
            self.emit_line(&format!(
                "    let {bits} = WarpValue::from_fn(|lane| {to_bits});"
            ));
            self.emit_line(&format!(
                "    let {low} = WarpValue::from_fn(|lane| {bits}[lane] as u32);"
            ));
            self.emit_line(&format!(
                "    let {high} = WarpValue::from_fn(|lane| ({bits}[lane] >> 32_u32) as u32);"
            ));
            let mut pieces = Vec::new();
            for (carrier, prefix) in [
                (low, "warp_reduce_low_shuffled"),
                (high, "warp_reduce_high_shuffled"),
            ] {
                let raw = self.temp(&format!("{prefix}_raw"));
                let piece = self.temp(prefix);
                let call = abi::warp_call(
                    "warp::shfl_sync",
                    &site,
                    &[shuffle_operands(&carrier)],
                    Some("v2::warp::variant::Shfl<v2::reg::variant::U32, v2::warp::variant::Butterfly>"),
                    None,
                    false,
                    true,
                );
                self.emit_line(&format!("    let {raw} = {call};"));
                self.emit_line(&format!("    let {piece} = v2_register_out({raw});"));
                pieces.push(piece);
            }
            let joined_name = self.temp("warp_reduce_shuffled");
            let mut joined = format!(
                "({}[lane] as u64) | (({}[lane] as u64) << 32_u32)",
                pieces[0], pieces[1]
            );
            if logical_dtype == "int64" {
                joined = format!("({joined}) as i64");
            } else if logical_dtype == "float64" {
                joined = format!("f64::from_bits({joined})");
            }
            self.emit_line(&format!(
                "    let {joined_name} = WarpValue::from_fn(|lane| {joined});"
            ));
            shuffled = joined_name;
        } else {
            let mut carrier = self.temp("warp_reduce_carrier");
            let mut carrier_marker = "U32";
            let mut decode: Option<String> = None;
            match logical_dtype.as_str() {
                "float32" => {
                    carrier = result.clone();
                    carrier_marker = "F32";
                }
                "float16" | "bfloat16" => {
                    let encoder = if logical_dtype == "float16" {
                        "f32_to_fp16_bits"
                    } else {
                        "f32_to_bf16_bits"
                    };
                    let decoder = if logical_dtype == "float16" {
                        "fp16_bits_to_f32"
                    } else {
                        "bf16_bits_to_f32"
                    };
                    self.emit_line(&format!(
                        "    let {carrier} = WarpValue::from_fn(|lane| u32::from({encoder}({result}[lane])));"
                    ));
                    decode = Some(format!("{decoder}({{value}} as u16)"));
                }
                "int8" | "int16" | "int32" => {
                    self.emit_line(&format!(
                        "    let {carrier} = WarpValue::from_fn(|lane| {result}[lane] as i32);"
                    ));
                    carrier_marker = "I32";
                    decode = Some(format!("{{value}} as {result_type}"));
                }
                "uint8" | "uint16" | "uint32" => {
                    self.emit_line(&format!(
                        "    let {carrier} = WarpValue::from_fn(|lane| {result}[lane] as u32);"
                    ));
                    decode = Some(format!("{{value}} as {result_type}"));
                }
                other => {
                    return unsupported(format!(
                        "warp_reduce has no exact shuffle carrier for {other}"
                    ))
                }
            }
            let raw = self.temp("warp_reduce_shuffle_raw");
            let carrier_result = self.temp("warp_reduce_shuffle_carrier");
            let call = abi::warp_call(
                "warp::shfl_sync",
                &site,
                &[shuffle_operands(&carrier)],
                Some(&format!(
                    "v2::warp::variant::Shfl<v2::reg::variant::{carrier_marker}, v2::warp::variant::Butterfly>"
                )),
                None,
                false,
                true,
            );
            self.emit_line(&format!("    let {raw} = {call};"));
            self.emit_line(&format!(
                "    let {carrier_result} = v2_register_out({raw});"
            ));
            match decode {
                None => shuffled = carrier_result,
                Some(decode) => {
                    let name = self.temp("warp_reduce_shuffled");
                    let decoded = decode.replace("{value}", &format!("{carrier_result}[lane]"));
                    self.emit_line(&format!(
                        "    let {name} = WarpValue::from_fn(|lane| {decoded});"
                    ));
                    shuffled = name;
                }
            }
        }
        let combine_function = match operation {
            "sum" => "add",
            "max" => "max",
            _ => "min",
        };
        if logical_dtype == "float16" || logical_dtype == "bfloat16" {
            let encode = if logical_dtype == "float16" {
                "f32_to_fp16_bits"
            } else {
                "f32_to_bf16_bits"
            };
            let decode = if logical_dtype == "float16" {
                "fp16_bits_to_f32"
            } else {
                "bf16_bits_to_f32"
            };
            let lhs_bits = self.temp("warp_reduce_lhs_bits");
            let rhs_bits = self.temp("warp_reduce_rhs_bits");
            let raw_combined = self.temp("warp_reduce_combined_raw");
            let combined = self.temp("warp_reduce_combined");
            let marker = if logical_dtype == "float16" && operation == "sum" {
                "F16Rn"
            } else if logical_dtype == "bfloat16" && operation == "sum" {
                "Bf16Rn"
            } else if logical_dtype == "float16" {
                "F16"
            } else {
                "Bf16"
            };
            self.emit_line(&format!(
                "    let {lhs_bits} = WarpValue::from_fn(|lane| {encode}({result}[lane]));"
            ));
            self.emit_line(&format!(
                "    let {rhs_bits} = WarpValue::from_fn(|lane| {encode}({shuffled}[lane]));"
            ));
            if operation == "sum" {
                let call = abi::lane_call(
                    "reg::add",
                    &site,
                    &[format!(
                        "(v2_register({lhs_bits}), v2_register({rhs_bits}))"
                    )],
                    Some(&format!("v2::reg::variant::{marker}")),
                );
                self.emit_line(&format!("    let {raw_combined} = {call};"));
            } else {
                let compare = if operation == "max" { "Gt" } else { "Lt" };
                let predicate_raw = self.temp("warp_reduce_predicate_raw");
                let predicate = self.temp("warp_reduce_predicate");
                let select_marker = if logical_dtype == "float16" {
                    "F16"
                } else {
                    "Bf16"
                };
                let setp = abi::lane_call(
                    "reg::setp",
                    &site,
                    &[format!("(v2_register(({lhs_bits}).clone()), v2_register(({rhs_bits}).clone()))")],
                    Some(&format!(
                        "v2::reg::variant::Setp<v2::reg::variant::{select_marker}, v2::reg::variant::{compare}>"
                    )),
                );
                self.emit_line(&format!("    let {predicate_raw} = {setp};"));
                self.emit_line(&format!(
                    "    let {predicate} = v2_register_out({predicate_raw});"
                ));
                let selp = abi::lane_call(
                    "reg::selp",
                    &site,
                    &[format!("(v2_register({predicate}), v2_register({lhs_bits}), v2_register({rhs_bits}))")],
                    Some(&format!("v2::reg::variant::{select_marker}")),
                );
                self.emit_line(&format!("    let {raw_combined} = {selp};"));
            }
            self.emit_line(&format!(
                "    let {combined} = v2_register_out({raw_combined});"
            ));
            self.emit_line(&format!(
                "    {result} = WarpValue::from_fn(|lane| {decode}({combined}[lane]));"
            ));
        } else {
            let promoted_type = match logical_dtype.as_str() {
                "int8" => "i16",
                "uint8" => "u16",
                _ => result_type,
            };
            let marker = match logical_dtype.as_str() {
                "int8" | "int16" => "I16",
                "int32" => "I32",
                "int64" => "I64",
                "uint8" | "uint16" => "U16",
                "uint32" => "U32",
                "uint64" => "U64",
                "float32" => {
                    if operation == "sum" {
                        "F32Rn"
                    } else {
                        "F32"
                    }
                }
                "float64" => {
                    if operation == "sum" {
                        "F64Rn"
                    } else {
                        "F64"
                    }
                }
                other => return not_covered(format!("warp_reduce marker for {other}")),
            };
            let mut lhs = result.clone();
            let mut rhs = shuffled.clone();
            if promoted_type != result_type {
                lhs = self.temp("warp_reduce_promoted_lhs");
                rhs = self.temp("warp_reduce_promoted_rhs");
                self.emit_line(&format!(
                    "    let {lhs} = WarpValue::from_fn(|lane| {result}[lane] as {promoted_type});"
                ));
                self.emit_line(&format!(
                    "    let {rhs} = WarpValue::from_fn(|lane| {shuffled}[lane] as {promoted_type});"
                ));
            }
            let raw_combined = self.temp("warp_reduce_combined_raw");
            let combined = self.temp("warp_reduce_combined");
            let call = abi::lane_call(
                &format!("reg::{combine_function}"),
                &site,
                &[format!(
                    "(v2_register(({lhs}).clone()), v2_register(({rhs}).clone()))"
                )],
                Some(&format!("v2::reg::variant::{marker}")),
            );
            self.emit_line(&format!("    let {raw_combined} = {call};"));
            self.emit_line(&format!(
                "    let {combined} = v2_register_out({raw_combined});"
            ));
            if promoted_type == result_type {
                self.emit_line(&format!("    {result} = {combined};"));
            } else {
                self.emit_line(&format!(
                    "    {result} = WarpValue::from_fn(|lane| {combined}[lane] as {result_type});"
                ));
            }
        }
        self.emit_line(&format!("    {delta} >>= 1_u32;"));
        self.emit_line("}");
        Ok(RustValue {
            control_provenance: value.control_provenance,
            ..RustValue::new(result, result_type, Uniformity::Varying)
        })
    }

    // ------------------------------------------------------------------
    // CTA collectives.
    // ------------------------------------------------------------------

    fn emit_cta_vote(&mut self, expr: &ObjectRef, operation: &str) -> AResult<RustValue> {
        let source_op_id = self.static_op_id(expr)?;
        let args = call_args(expr);
        let predicate = self.emit_expr(&args[0])?;
        let predicate = self.coerce_value(predicate, "bool", "cta_vote_predicate")?;
        let predicate = self.as_warp_value(predicate);
        let result = self.control_name("cta_vote_values");
        let site = self.v2_site(Some(source_op_id));
        let call = abi::warp_call(
            "collective::cta_vote",
            &site,
            &[format!("v2_register(({}).clone())", predicate.code)],
            Some(&format!(
                "v2::collective::variant::{}",
                if operation == "and" { "All" } else { "Any" }
            )),
            None,
            true,
            true,
        );
        self.emit_suspend_line(&format!("let {result} = v2_register_out({call});"));
        Ok(RustValue::new(result, "i64", Uniformity::Varying))
    }

    fn emit_cta_reduce(
        &mut self,
        expr: &ObjectRef,
        operation: &str,
        num_warps: i64,
    ) -> AResult<RustValue> {
        let source_op_id = self.static_op_id(expr)?;
        let args = call_args(expr);
        let value = self.emit_expr(&args[0])?;
        let value = self.as_warp_value(value);
        let result_dtype = dtype_of(expr)?;
        let rust_type = stmt_rust_scalar_by_dtype(&result_dtype);
        if rust_type.is_none()
            || matches!(
                result_dtype.as_str(),
                "bool" | "float8_e4m3fn" | "float8_e8m0fnu" | "float4_e2m1fn"
            )
        {
            return unsupported(format!(
                "CTA reduction requires an integer, float16, bfloat16, float32, or float64 scalar, got {result_dtype}"
            ));
        }
        let rust_type = rust_type.unwrap();
        if value.rust_type != rust_type {
            return unsupported(format!(
                "CTA reduction value lowered to {}, expected {rust_type}",
                value.rust_type
            ));
        }
        let operation_variant = match operation {
            "sum" => "Sum",
            "max" => "Max",
            _ => "Min",
        };
        let scalar_marker = match result_dtype.as_str() {
            "int8" => "v2::reg::variant::I8",
            "int16" => "v2::reg::variant::I16",
            "int32" => "v2::reg::variant::I32",
            "int64" => "v2::reg::variant::I64",
            "uint8" => "v2::reg::variant::U8",
            "uint16" => "v2::reg::variant::U16",
            "uint32" => "v2::reg::variant::U32",
            "uint64" => "v2::reg::variant::U64",
            "float16" => "v2::collective::variant::F16",
            "bfloat16" => "v2::collective::variant::Bf16",
            "float32" => "v2::reg::variant::F32",
            "float64" => "v2::reg::variant::F64",
            other => return not_covered(format!("CTA reduction marker for {other}")),
        };
        let scratch = self.pointer(&args[3], "CTA reduction scratch pointer")?;
        let result = self.control_name("cta_reduce_values");
        let site = self.v2_site(Some(source_op_id));
        let call = abi::warp_call(
            "collective::cta_reduce",
            &site,
            &[format!(
                "(v2_register(({}).clone()), {}, {num_warps}_usize)",
                value.code,
                abi::address("v2::Shared", &abi::cloned(&scratch.code), None)
            )],
            Some(&format!(
                "v2::collective::variant::Reduce<{scalar_marker}, v2::collective::variant::{operation_variant}>"
            )),
            None,
            true,
            true,
        );
        self.emit_suspend_line(&format!("let {result} = v2_register_out({call});"));
        Ok(RustValue::new(result, rust_type, Uniformity::Varying))
    }
}

pub(crate) fn reinterpret_atom(code: &str, source_type: &str, target_type: &str) -> String {
    if source_type == target_type {
        return code.to_owned();
    }
    if source_type == "F32x4" && target_type == "U64x2" {
        return format!(
            "[u64::from(({code})[0].to_bits()) | (u64::from(({code})[1].to_bits()) << 32), u64::from(({code})[2].to_bits()) | (u64::from(({code})[3].to_bits()) << 32)]"
        );
    }
    if source_type == "U64x2" && target_type == "F32x4" {
        return format!(
            "[f32::from_bits(({code})[0] as u32), f32::from_bits((({code})[0] >> 32) as u32), f32::from_bits(({code})[1] as u32), f32::from_bits((({code})[1] >> 32) as u32)]"
        );
    }
    if source_type == "f32" {
        let bits = format!("({code}).to_bits()");
        return if target_type == "u32" {
            bits
        } else {
            format!("({bits}) as {target_type}")
        };
    }
    if target_type == "f32" {
        return format!("f32::from_bits(({code}) as u32)");
    }
    if source_type == "f64" {
        let bits = format!("({code}).to_bits()");
        return if target_type == "u64" {
            bits
        } else {
            format!("({bits}) as {target_type}")
        };
    }
    if target_type == "f64" {
        return format!("f64::from_bits(({code}) as u64)");
    }
    format!("({code}) as {target_type}")
}

fn check_signature(signature: &CallSignature, result: Result<bool, String>) -> AResult<()> {
    match result {
        Ok(_) => Ok(()),
        Err(mismatch)
            if signature.op_name == "tirx.reinterpret"
                && mismatch.starts_with("raw payload reinterpret is not modeled") =>
        {
            unmodeled(format!("call:{}", signature.op_name), mismatch)
        }
        Err(mismatch) => unsupported(format!("Call({}): {mismatch}", signature.render())),
    }
}
pub fn prim_if_then_else(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = prim_if_then_else_signature(emitter.ctx, call.node, call.call)?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            let args = &call.args;
            let signature = &signature;
            {
                let result_type = emitter.pure_result_type(signature)?;
                let (result, condition, selected) =
                    emitter.emit_conditional(&args[0], &args[1], &args[2], &result_type, "if")?;
                emitter.instrument_call(expr, &[&condition, &selected], result)
            }
        })
        .map(Some)
}
pub fn address_of(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = address_of_signature(emitter.ctx, call.node, call.call)?;
    address_of_parts(&signature, &call.args)?;
    emitter
        .with_call_expr(call.node, |emitter| {
            let args = &call.args;
            emitter.emit_address_of(&args)
        })
        .map(Some)
}
pub fn cuda_activemask(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_tvm_warp_activemask(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let site = emitter.expr_site(expr);
                let result = emitter.temp("active_mask");
                let call = abi::warp_call("warp::activemask", &site, &[], None, None, false, true);
                emitter.emit_line(&format!("let {result} = {call};"));
                Ok(RustValue {
                    requires_statement: true,
                    ..RustValue::new(format!("{result}[0]"), "u32", Uniformity::Uniform)
                })
            }
        })
        .map(Some)
}
pub fn cuda_shfl_down_sync(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_cuda_shfl_sync(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            let signature = &signature;
            {
                let result_type = emitter.pure_result_type(signature)?;
                emitter.emit_warp_shuffle_call(expr, &result_type, "down")
            }
        })
        .map(Some)
}
pub fn cuda_shfl_sync(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_cuda_shfl_sync(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            let signature = &signature;
            {
                let result_type = emitter.pure_result_type(signature)?;
                emitter.emit_warp_shuffle_call(expr, &result_type, "index")
            }
        })
        .map(Some)
}
pub fn cuda_shfl_up_sync(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_cuda_shfl_sync(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            let signature = &signature;
            {
                let result_type = emitter.pure_result_type(signature)?;
                emitter.emit_warp_shuffle_call(expr, &result_type, "up")
            }
        })
        .map(Some)
}
pub fn cuda_shfl_xor_sync(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_cuda_shfl_sync(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            let signature = &signature;
            {
                let result_type = emitter.pure_result_type(signature)?;
                emitter.emit_warp_shuffle_call(expr, &result_type, "xor")
            }
        })
        .map(Some)
}
pub fn cuda_any_sync(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_cuda_any_sync(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_any_sync(expr)
        })
        .map(Some)
}
pub fn cuda_ballot_sync(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_ballot_sync(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_ballot_sync(expr)
        })
        .map(Some)
}
pub fn cuda_bfloat1622float2(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_bfloat1622float2(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let arguments = emitter.emit_arguments(expr)?;
                emitter.emit_call_atom("unpack_bf16x2", &arguments, "u64", |codes| {
                    format!("unpack_bf16x2({})", codes[0])
                })
            }
        })
        .map(Some)
}
pub fn cuda_bfloat162float(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_bfloat162float(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let arguments = emitter.emit_arguments(expr)?;
                emitter.emit_call_atom("low_precision_to_f32", &arguments, "f32", |codes| {
                    codes[0].clone()
                })
            }
        })
        .map(Some)
}
pub fn cuda_clock64(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_cuda_clock64(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |_emitter| {
            Ok(RustValue::new("0_u64", "u64", Uniformity::Uniform))
        })
        .map(Some)
}
pub fn cuda_cta_reduce(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_cta_reduce(emitter.ctx, &signature),
    )?;
    let (operation, num_warps) = cta_reduce_parts(&signature, &call.args)?;
    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_cta_reduce(expr, &operation, num_warps)
        })
        .map(Some)
}
pub fn cuda_elect_sync(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_elect_sync(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let site = emitter.expr_site(expr);
                let raw_result = emitter.temp("elect_sync_raw");
                let result = emitter.temp("elect_sync");
                let call = abi::warp_call(
                    "warp::elect_sync",
                    &site,
                    &[abi::splat("0xffff_ffff_u32")],
                    None,
                    None,
                    false,
                    true,
                );
                emitter.emit_line(&format!("let (_, {raw_result}) = {call};"));
                emitter.emit_line(&format!(
                    "let {raw_result} = v2_register_out({raw_result});"
                ));
                emitter.emit_line(&format!(
                    "let {result} = WarpValue::from_fn(|lane| u32::from({raw_result}[lane]));"
                ));
                Ok(RustValue {
                    control_provenance: ControlProvenance::ElectSync,
                    ..RustValue::new(result, "u32", Uniformity::Varying)
                })
            }
        })
        .map(Some)
}
pub fn cuda_fadd2_rn(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_cuda_fmul2_rn(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                emitter.emit_v2_register_call_expr(
                    expr,
                    "add",
                    F32X2_VARIANT,
                    &["u64", "u64"],
                    "u64",
                )
            }
        })
        .map(Some)
}
pub fn cuda_fdividef(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_cuda_fdividef(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let arguments = emitter.emit_arguments(expr)?;
                emitter.emit_call_atom("fdividef", &arguments, "f32", |codes| {
                    format!("({}) / ({})", codes[0], codes[1])
                })
            }
        })
        .map(Some)
}
pub fn cuda_ffs_u32(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_cuda_ffs_u32(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let arguments = emitter.emit_arguments(expr)?;
                emitter.emit_call_atom("ffs_u32", &arguments, "i32", |codes| {
                    format!(
                        "if {} == 0 {{ 0_i32 }} else {{ ({}).trailing_zeros() as i32 + 1_i32 }}",
                        codes[0], codes[0]
                    )
                })
            }
        })
        .map(Some)
}
pub fn cuda_float22bfloat162_rn(
    emitter: &mut Emitter,
    call: &Decoded,
) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_float22bfloat162_rn(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let arguments = emitter.emit_arguments(expr)?;
                emitter.emit_call_atom("pack_bf16x2", &arguments, "u32", |codes| {
                    format!("pack_bf16x2({}, {})", codes[0], codes[1])
                })
            }
        })
        .map(Some)
}
pub fn cuda_float22bfloat162_rn_from_float2(
    emitter: &mut Emitter,
    call: &Decoded,
) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_float22bfloat162_rn_from_float2(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let arguments = emitter.emit_arguments(expr)?;
                emitter.emit_call_atom("pack_bf16x2_from_float2", &arguments, "u32", |codes| {
                    format!(
                        "pack_bf16x2(float2_x({}), float2_y({}))",
                        codes[0], codes[0]
                    )
                })
            }
        })
        .map(Some)
}
pub fn cuda_float2_x(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_cuda_float2_x(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let arguments = emitter.emit_arguments(expr)?;
                emitter.emit_call_atom("float2_x", &arguments, "f32", |codes| {
                    format!("float2_x({})", codes[0])
                })
            }
        })
        .map(Some)
}
pub fn cuda_float2_y(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_cuda_float2_x(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let arguments = emitter.emit_arguments(expr)?;
                emitter.emit_call_atom("float2_y", &arguments, "f32", |codes| {
                    format!("float2_y({})", codes[0])
                })
            }
        })
        .map(Some)
}
pub fn cuda_float_as_uint(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_float_as_uint(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let arguments = emitter.emit_arguments(expr)?;
                emitter.emit_call_atom("float_as_uint", &arguments, "u32", |codes| {
                    format!("({}).to_bits()", codes[0])
                })
            }
        })
        .map(Some)
}
pub fn cuda_fmul2_rn(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_cuda_fmul2_rn(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                emitter.emit_v2_register_call_expr(
                    expr,
                    "mul",
                    F32X2_VARIANT,
                    &["u64", "u64"],
                    "u64",
                )
            }
        })
        .map(Some)
}
pub fn cuda_fp8x4_e4m3_from_float4(
    emitter: &mut Emitter,
    call: &Decoded,
) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_fp8x4_e4m3_from_float4(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let arguments = emitter.emit_arguments(expr)?;
                emitter.emit_call_atom("fp8x4_e4m3_from_float4", &arguments, "u32", |codes| {
                    format!("fp8x4_e4m3_from_float4({})", codes.join(", "))
                })
            }
        })
        .map(Some)
}
pub fn cuda_get_tmem_addr(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_get_tmem_addr(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let arguments = emitter.emit_arguments(expr)?;
                let typed = vec![
                    emitter.coerce_value(arguments[0].clone(), "u32", "get_tmem_addr_encoded")?,
                    emitter.coerce_value(arguments[1].clone(), "i32", "get_tmem_addr_row")?,
                    emitter.coerce_value(arguments[2].clone(), "u32", "get_tmem_addr_column")?,
                ];
                emitter.emit_call_atom("get_tmem_addr", &typed, "u32", |codes| {
                    format!("get_tmem_addr({}, {}, {})", codes[0], codes[1], codes[2])
                })
            }
        })
        .map(Some)
}
pub fn cuda_half2float(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_half2float(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let arguments = emitter.emit_arguments(expr)?;
                emitter.emit_call_atom("low_precision_to_f32", &arguments, "f32", |codes| {
                    format!("cuda_canonicalize_nan_f32({})", codes[0])
                })
            }
        })
        .map(Some)
}
pub fn cuda_hmax2(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_cuda_hmin2(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_v2_register_call_expr(
                expr,
                "max",
                "v2::reg::variant::Bf16x2",
                &["u32", "u32"],
                "u32",
            )
        })
        .map(Some)
}
pub fn cuda_hmin2(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_cuda_hmin2(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_v2_register_call_expr(
                expr,
                "min",
                "v2::reg::variant::Bf16x2",
                &["u32", "u32"],
                "u32",
            )
        })
        .map(Some)
}
pub fn cuda_iket_mark(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_cuda_iket_mark(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |_emitter| {
            Ok(RustValue::new("0_u32", "u32", Uniformity::Uniform))
        })
        .map(Some)
}
pub fn cuda_iket_official_event(
    emitter: &mut Emitter,
    call: &Decoded,
) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_iket_official_event(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |_emitter| {
            Ok(RustValue::new("0_u32", "u32", Uniformity::Uniform))
        })
        .map(Some)
}
pub fn cuda_iket_range_end(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_iket_range_end(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |_emitter| {
            Ok(RustValue::new("0_u32", "u32", Uniformity::Uniform))
        })
        .map(Some)
}
pub fn cuda_iket_range_pop(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_iket_range_pop(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |_emitter| {
            Ok(RustValue::new("0_u32", "u32", Uniformity::Uniform))
        })
        .map(Some)
}
pub fn cuda_iket_range_push(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_iket_range_push(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |_emitter| {
            Ok(RustValue::new("0_u32", "u32", Uniformity::Uniform))
        })
        .map(Some)
}
pub fn cuda_iket_range_start(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_iket_range_start(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |_emitter| {
            Ok(RustValue::new("0_u32", "u32", Uniformity::Uniform))
        })
        .map(Some)
}
pub fn cuda_iket_sentinel_token(
    emitter: &mut Emitter,
    call: &Decoded,
) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_iket_sentinel_token(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |_emitter| {
            Ok(RustValue::new("0_u32", "u32", Uniformity::Uniform))
        })
        .map(Some)
}
pub fn cuda_make_float2(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_make_float2(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let arguments = emitter.emit_arguments(expr)?;
                emitter.emit_call_atom("make_float2", &arguments, "u64", |codes| {
                    format!("make_float2({}, {})", codes[0], codes[1])
                })
            }
        })
        .map(Some)
}
pub fn cuda_mov_sreg(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_cuda_mov_sreg(emitter.ctx, &signature))?;
    let register_name = fetch_register_parts(&signature, &call.args)?;
    emitter
        .with_call_expr(call.node, |emitter| {
            let signature = &signature;
            emitter.emit_fetch_register(signature, &register_name)
        })
        .map(Some)
}
pub fn cuda_reduce_add_sync_u32(
    emitter: &mut Emitter,
    call: &Decoded,
) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_reduce_add_sync_u32(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_redux(expr, "reduce_add_u32", "Add")
        })
        .map(Some)
}
pub fn cuda_reduce_min_sync_u32(
    emitter: &mut Emitter,
    call: &Decoded,
) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_reduce_add_sync_u32(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_redux(expr, "reduce_min_u32", "Min")
        })
        .map(Some)
}
pub fn cuda_sm100_2sm_leader_smem_addr(
    emitter: &mut Emitter,
    call: &Decoded,
) -> AResult<Option<RustValue>> {
    shared_address_signature(emitter.ctx, call.node, call.call)?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_shared_address(
                expr,
                "sm100_mbar_addr",
                "sm100_tma_2sm_mbarrier_address_from_u64",
            )
        })
        .map(Some)
}
pub fn cuda_smem_addr_from_uint64(
    emitter: &mut Emitter,
    call: &Decoded,
) -> AResult<Option<RustValue>> {
    shared_address_signature(emitter.ctx, call.node, call.call)?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                emitter.emit_shared_address(expr, "smem_addr_from_u64", "shared_address_from_u64")
            }
        })
        .map(Some)
}
pub fn cuda_syncthreads_and(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_syncthreads_and(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_cta_vote(expr, "and")
        })
        .map(Some)
}
pub fn cuda_syncthreads_or(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_syncthreads_and(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_cta_vote(expr, "or")
        })
        .map(Some)
}
pub fn cuda_thread_rank(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_thread_rank(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_thread_rank(expr)
        })
        .map(Some)
}
pub fn cuda_uint_as_float(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_uint_as_float(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let arguments = emitter.emit_arguments(expr)?;
                emitter.emit_call_atom("uint_as_float", &arguments, "f32", |codes| {
                    format!("f32::from_bits({})", codes[0])
                })
            }
        })
        .map(Some)
}
pub fn cuda_warp_reduce(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_cuda_warp_reduce(emitter.ctx, &signature),
    )?;
    let (operation, width) = warp_reduce_parts(&signature, &call.args)?;
    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            let signature = &signature;
            {
                let result_type = emitter.pure_result_type(signature)?;
                emitter.emit_warp_reduce(expr, &result_type, &operation, width)
            }
        })
        .map(Some)
}
pub fn exp(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_fabs(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_unary_atom(expr, "exp", "f32", ".exp()")
        })
        .map(Some)
}
pub fn fabs(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_fabs(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_unary_atom(expr, "fabs", "f32", ".abs()")
        })
        .map(Some)
}
pub fn fma(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_fma(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_v2_register_call_expr(
                expr,
                "fma",
                "v2::reg::variant::F32Rn",
                &["f32", "f32", "f32"],
                "f32",
            )
        })
        .map(Some)
}
pub fn isnullptr(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_isnullptr(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let args = &call.args;
            {
                let pointer = emitter.emit_expr(&args[0])?;
                emitter.emit_pointer_is_null(&pointer)
            }
        })
        .map(Some)
}
pub fn log(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_fabs(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_unary_atom(expr, "log", "f32", ".ln()")
        })
        .map(Some)
}
pub fn log1p(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_fabs(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_unary_atom(expr, "log1p", "f32", ".ln_1p()")
        })
        .map(Some)
}
pub fn log2(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_fabs(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_unary_atom(expr, "log2", "f32", ".log2()")
        })
        .map(Some)
}
pub fn popcount(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_popcount(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            emitter.emit_unary_atom(expr, "popcount", "u32", ".count_ones()")
        })
        .map(Some)
}
pub fn reinterpret(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = reinterpret_signature(emitter.ctx, call.node, call.call)?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            let signature = &signature;
            emitter.emit_reinterpret(expr, signature)
        })
        .map(Some)
}
pub fn rsqrt(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_fabs(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let arguments = emitter.emit_arguments(expr)?;
                emitter.emit_call_atom("rsqrt", &arguments, "f32", |codes| {
                    format!("1.0_f32 / ({}).sqrt()", codes[0])
                })
            }
        })
        .map(Some)
}
pub fn sigmoid(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(&signature, validate_fabs(emitter.ctx, &signature))?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let arguments = emitter.emit_arguments(expr)?;
                emitter.emit_call_atom("sigmoid", &arguments, "f32", |codes| {
                    format!("1.0_f32 / (1.0_f32 + (-({})).exp())", codes[0])
                })
            }
        })
        .map(Some)
}
pub fn timer_end_cuda(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_timer_start_cuda(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |_emitter| {
            Ok(RustValue::new("0_u64", "u64", Uniformity::Uniform))
        })
        .map(Some)
}
pub fn timer_finalize_cuda(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_timer_finalize_cuda(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |_emitter| {
            Ok(RustValue::new("0_u64", "u64", Uniformity::Uniform))
        })
        .map(Some)
}
pub fn timer_init_cuda(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_timer_init_cuda(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |_emitter| {
            Ok(RustValue::new("0_u64", "u64", Uniformity::Uniform))
        })
        .map(Some)
}
pub fn timer_start_cuda(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_timer_start_cuda(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |_emitter| {
            Ok(RustValue::new("0_u64", "u64", Uniformity::Uniform))
        })
        .map(Some)
}
pub fn tvm_warp_activemask(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_tvm_warp_activemask(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            {
                let site = emitter.expr_site(expr);
                let result = emitter.temp("active_mask");
                let call = abi::warp_call("warp::activemask", &site, &[], None, None, false, true);
                emitter.emit_line(&format!("let {result} = {call};"));
                Ok(RustValue {
                    requires_statement: true,
                    ..RustValue::new(format!("{result}[0]"), "u32", Uniformity::Uniform)
                })
            }
        })
        .map(Some)
}
pub fn tvm_warp_shuffle(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_tvm_warp_shuffle(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            let signature = &signature;
            {
                let result_type = emitter.pure_result_type(signature)?;
                emitter.emit_warp_shuffle_call(expr, &result_type, "index")
            }
        })
        .map(Some)
}
pub fn tvm_warp_shuffle_down(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_tvm_warp_shuffle(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            let signature = &signature;
            {
                let result_type = emitter.pure_result_type(signature)?;
                emitter.emit_warp_shuffle_call(expr, &result_type, "down")
            }
        })
        .map(Some)
}
pub fn tvm_warp_shuffle_up(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_tvm_warp_shuffle(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            let signature = &signature;
            {
                let result_type = emitter.pure_result_type(signature)?;
                emitter.emit_warp_shuffle_call(expr, &result_type, "up")
            }
        })
        .map(Some)
}
pub fn tvm_warp_shuffle_xor(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let signature = signature(call.node, call.call)?;
    check_signature(
        &signature,
        validate_tvm_warp_shuffle(emitter.ctx, &signature),
    )?;

    emitter
        .with_call_expr(call.node, |emitter| {
            let expr = call.node;
            let signature = &signature;
            {
                let result_type = emitter.pure_result_type(signature)?;
                emitter.emit_warp_shuffle_call(expr, &result_type, "xor")
            }
        })
        .map(Some)
}

pub fn address_of_signature(ctx: &Ctx, node: &ObjectRef, call: &CallObj) -> AResult<CallSignature> {
    let signature = signature(node, call)?;
    check_signature(&signature, validate_address_of(ctx, &signature))?;
    Ok(signature)
}

pub fn reinterpret_signature(
    ctx: &Ctx,
    node: &ObjectRef,
    call: &CallObj,
) -> AResult<CallSignature> {
    let signature = signature(node, call)?;
    check_signature(&signature, validate_reinterpret(ctx, &signature))?;
    Ok(signature)
}

pub fn prim_if_then_else_signature(
    ctx: &Ctx,
    node: &ObjectRef,
    call: &CallObj,
) -> AResult<CallSignature> {
    let signature = signature(node, call)?;
    check_signature(&signature, validate_prim_if_then_else(ctx, &signature))?;
    Ok(signature)
}

pub fn shared_address_signature(
    ctx: &Ctx,
    node: &ObjectRef,
    call: &CallObj,
) -> AResult<CallSignature> {
    let signature = signature(node, call)?;
    check_signature(
        &signature,
        validate_cuda_smem_addr_from_uint64(ctx, &signature),
    )?;
    Ok(signature)
}

pub fn source_text(node: &ObjectRef) -> AResult<Option<String>> {
    let Some(call) = node.as_node::<CallObj>() else {
        return Ok(None);
    };
    if call.args.len() != 1 || call_op_name(call)?.as_deref() != Some("tirx.address_of") {
        return Ok(None);
    }
    let argument = oref(call.args.get(0)?);
    if argument.as_node::<TensorLoadObj>().is_some() {
        return Ok(Some(format!(
            "T.address_of({})",
            crate::analyze::util::repr_text(&argument)?
        )));
    }
    Ok(None)
}

/// The lazy branches of the scalar conditional used in static launch extents.
pub fn conditional_operands(
    node: &ObjectRef,
) -> AResult<Option<(ObjectRef, ObjectRef, ObjectRef)>> {
    let Some(call) = node.as_node::<CallObj>() else {
        return Ok(None);
    };
    if call_op_name(call)?.as_deref() != Some("prim.if_then_else") {
        return Ok(None);
    }
    if call.args.len() != 3 {
        return crate::analyze::util::not_covered(
            "malformed if_then_else in a static launch extent",
        );
    }
    Ok(Some((
        oref(call.args.get(0)?),
        oref(call.args.get(1)?),
        oref(call.args.get(2)?),
    )))
}
