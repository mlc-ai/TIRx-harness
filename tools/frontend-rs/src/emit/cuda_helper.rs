//! Validation and emission of the cuda_helper instruction family.

use crate::analyze::buffers::call_op_name;
use crate::analyze::util::{
    as_buffer, buffer_dtype, buffer_scope, dtype_of, dtype_text, expr_type, ffi_text, int_imm,
    not_covered, oref, prim, prim_dtype, unsupported, AResult,
};
use crate::decode::Decoded;
use crate::emit::async_copy::ST_ASYNC;
use crate::emit::ptx_warp::SHFL_SYNC;
use crate::emit::{abi, Emitter, RustValue, Uniformity};
use crate::tables::{json_string, MEM_LD, MEM_ST};
use crate::tvm_compat::int_value;
use tvm::ir::StringImmObj;
use tvm::ir::{CallObj, IntImmObj, PointerTypeObj, PrimType, TensorLoadObj};
use tvm::prim::Cast;
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KnownCudaFuncKind {
    CombineIntFracEx2,
    FlashkdaFmafRn,
    FlashkdaRsqrtf,
    FlashkdaTanhApprox,
    TensorMapAcquire,
    TensorMapRelease,
    TensorMapReplaceGlobalAddress,
    TensorMapReplaceGlobalDim,
    TensorMapReplaceGlobalStride,
    FmaScaleSubF32x2,
    GdnCpPrefillPredicatedGamma,
    GdnLg2ApproxFtz,
    OpaqueSmIdxU32,
    OpaqueWarpId,
    Tcgen05MmaMxf4Block32Ss,
    ShlU32Clamp,
    MqaFp4WreluReduce64,
    MqaFp8WreluReduce64,
    PrefetchL2,
    Relu2FmaF32x2,
    SmemDescAdd16bOffset,
    SmemDescMakeLoUniform,
    StAsyncClusterTaskInfo,
}

const KIND_VALUES: &[(KnownCudaFuncKind, &str)] = &[
    (KnownCudaFuncKind::CombineIntFracEx2, "combine_int_frac_ex2"),
    (KnownCudaFuncKind::FlashkdaFmafRn, "flashkda_fmaf_rn"),
    (KnownCudaFuncKind::FlashkdaRsqrtf, "flashkda_rsqrtf"),
    (
        KnownCudaFuncKind::FlashkdaTanhApprox,
        "flashkda_tanh_approx",
    ),
    (KnownCudaFuncKind::TensorMapAcquire, "tensor_map_acquire"),
    (KnownCudaFuncKind::TensorMapRelease, "tensor_map_release"),
    (
        KnownCudaFuncKind::TensorMapReplaceGlobalAddress,
        "tensor_map_replace_global_address",
    ),
    (
        KnownCudaFuncKind::TensorMapReplaceGlobalDim,
        "tensor_map_replace_global_dim",
    ),
    (
        KnownCudaFuncKind::TensorMapReplaceGlobalStride,
        "tensor_map_replace_global_stride",
    ),
    (
        KnownCudaFuncKind::FmaScaleSubF32x2,
        "tvm_builtin_fma_scale_sub_f32x2",
    ),
    (
        KnownCudaFuncKind::GdnCpPrefillPredicatedGamma,
        "gdn_cp_prefill_predicated_gamma",
    ),
    (KnownCudaFuncKind::GdnLg2ApproxFtz, "gdn_lg2_approx_ftz"),
    (
        KnownCudaFuncKind::OpaqueSmIdxU32,
        "tvm_builtin_opaque_sm_idx_u32",
    ),
    (
        KnownCudaFuncKind::OpaqueWarpId,
        "tvm_builtin_opaque_warp_id",
    ),
    (
        KnownCudaFuncKind::Tcgen05MmaMxf4Block32Ss,
        "tvm_builtin_tcgen05_mma_mxf4_block32_ss",
    ),
    (KnownCudaFuncKind::ShlU32Clamp, "shl_u32_clamp"),
    (
        KnownCudaFuncKind::MqaFp4WreluReduce64,
        "tvm_builtin_mqa_fp4_wrelu_reduce_64",
    ),
    (
        KnownCudaFuncKind::MqaFp8WreluReduce64,
        "tvm_builtin_mqa_fp8_wrelu_reduce_64",
    ),
    (KnownCudaFuncKind::PrefetchL2, "tirx_prefetch_l2"),
    (KnownCudaFuncKind::Relu2FmaF32x2, "tirx_relu2_fma_f32x2"),
    (
        KnownCudaFuncKind::SmemDescAdd16bOffset,
        "tvm_builtin_smem_desc_add_16B_offset",
    ),
    (
        KnownCudaFuncKind::SmemDescMakeLoUniform,
        "smem_desc_make_lo_uniform",
    ),
    (
        KnownCudaFuncKind::StAsyncClusterTaskInfo,
        "tvm_builtin_st_async_cluster_task_info",
    ),
];

impl KnownCudaFuncKind {
    pub fn value(self) -> &'static str {
        KIND_VALUES
            .iter()
            .find(|(kind, _)| *kind == self)
            .map(|(_, value)| *value)
            .expect("registered helper kind")
    }

    fn from_value(name: &str) -> Option<Self> {
        KIND_VALUES
            .iter()
            .find(|(_, value)| *value == name)
            .map(|(kind, _)| *kind)
    }

    pub fn is_tensor_map(self) -> bool {
        matches!(
            self,
            KnownCudaFuncKind::TensorMapAcquire
                | KnownCudaFuncKind::TensorMapRelease
                | KnownCudaFuncKind::TensorMapReplaceGlobalAddress
                | KnownCudaFuncKind::TensorMapReplaceGlobalDim
                | KnownCudaFuncKind::TensorMapReplaceGlobalStride
        )
    }
}

#[derive(Clone)]
pub struct KnownCudaFuncCall {
    pub kind: KnownCudaFuncKind,
    pub arguments: Vec<ObjectRef>,
    pub index: Option<i64>,
}

const COMBINE_INT_FRAC_EX2_SOURCE: &str = r#"
__device__ __forceinline__ float combine_int_frac_ex2(float x_rounded, float frac_ex2) {
  float out;
  asm volatile(
    "{\n\t"
    ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\n\t"
    "mov.b32 x_rounded_i, %1;\n\t"
    "mov.b32 frac_ex_i, %2;\n\t"
    "shl.b32 x_rounded_e, x_rounded_i, 23;\n\t"
    "add.s32 out_i, x_rounded_e, frac_ex_i;\n\t"
    "mov.b32 %0, out_i;\n\t"
    "}\n"
    : "=f"(out) : "f"(x_rounded), "f"(frac_ex2));
  return out;
}
"#;

const SHL_U32_CLAMP_SOURCE: &str = r#"
__device__ __forceinline__ unsigned int shl_u32_clamp(unsigned int val, unsigned int shift) {
  unsigned int r;
  asm("shl.b32 %0, %1, %2;" : "=r"(r) : "r"(val), "r"(shift));
  return r;
}
"#;

const GDN_LG2_APPROX_FTZ_SOURCE: &str = r#"
__device__ __forceinline__ float gdn_lg2_approx_ftz(float value) {
    float out;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(value));
    return out;
}
"#;

const GDN_CP_PREFILL_PREDICATED_GAMMA_SOURCE: &str = r#"
__forceinline__ __device__ float gdn_cp_prefill_predicated_gamma(
        uint32_t s_addr, uint32_t t_addr, uint32_t pred) {
    float gamma;
    asm volatile(
        "{ .reg .pred p; .reg .f32 s_log; .reg .f32 t_log; "
        "mov.f32 %0, 0f00000000; "
        "setp.ne.b32 p, %1, 0; "
        "@p ld.shared.f32 s_log, [%2]; "
        "@p ld.shared.f32 t_log, [%3]; "
        "@p sub.f32 %0, s_log, t_log; "
        "@p ex2.approx.ftz.f32 %0, %0; }"
        : "=f"(gamma)
        : "r"(pred), "r"(s_addr), "r"(t_addr)
        : "memory");
    return gamma;
}
"#;

const FLASHKDA_TANH_APPROX_SOURCE: &str = r#"// tanh.approx.f32 (sigmoid via tanh; used throughout the prep role)
__device__ __forceinline__ float flashkda_tanh_approx(float x) {
    float y;
    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
"#;

const FLASHKDA_FMAF_RN_SOURCE: &str = r#"// __fmaf_rn (value form of fma.rn.f32)
__device__ __forceinline__ float flashkda_fmaf_rn(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}
"#;

const FLASHKDA_RSQRTF_SOURCE: &str = r#"// rsqrtf
__device__ __forceinline__ float flashkda_rsqrtf(float x) {
    return rsqrtf(x);
}
"#;

const FLASHKDA_TENSOR_MAP_ACQUIRE_SOURCE: &str = r#"__device__ __forceinline__ void flashkda_tensormap_acquire(const void *tmap_ptr) {
    asm volatile(
        "fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n"
        :: "l"(tmap_ptr) : "memory");
}
"#;

const GDN_TENSOR_MAP_ACQUIRE_SOURCE: &str = r#"__device__ __forceinline__ void gdn_tensormap_acquire(const void *desc) {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
                 :: "l"(desc) : "memory");
}
"#;

const GDN_TENSOR_MAP_RELEASE_SOURCE: &str = r#"__device__ __forceinline__ void gdn_tensormap_release() {
    asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
}
"#;

const GDN_TENSOR_MAP_REPLACE_GLOBAL_ADDRESS_SOURCE: &str = r#"__device__ __forceinline__ void gdn_tensormap_replace_global_address(void *desc, const void *addr) {
    asm volatile("tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;"
                 :: "l"(desc), "l"(addr) : "memory");
}
"#;

fn gdn_tensor_map_replace_global_dim_source(index: i64) -> String {
    format!(
        r#"__device__ __forceinline__ void gdn_tensormap_replace_global_dim_{index}(void *desc, unsigned int value) {{
    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], {index}, %1;"
                 :: "l"(desc), "r"(value) : "memory");
}}
"#
    )
}

fn gdn_tensor_map_replace_global_stride_source(index: i64) -> String {
    format!(
        r#"__device__ __forceinline__ void gdn_tensormap_replace_global_stride_{index}(void *desc, unsigned long long value) {{
    asm volatile("tensormap.replace.tile.global_stride.global.b1024.b64 [%0], {index}, %1;"
                 :: "l"(desc), "l"(value) : "memory");
}}
"#
    )
}

const FMA_SCALE_SUB_F32X2_SOURCE: &str = r#"
__forceinline__ __device__ unsigned long long tvm_builtin_fma_scale_sub_f32x2(
    unsigned long long scores,
    unsigned long long scale,
    unsigned long long lse) {
    float2 score_pair = *reinterpret_cast<float2*>(&scores);
    float2 scale_pair = *reinterpret_cast<float2*>(&scale);
    float2 lse_pair = *reinterpret_cast<float2*>(&lse);
    float2 result = make_float2(
        fmaf(score_pair.x, scale_pair.x, -lse_pair.x),
        fmaf(score_pair.y, scale_pair.y, -lse_pair.y));
    return *reinterpret_cast<unsigned long long*>(&result);
}
"#;

const OPAQUE_SM_IDX_U32_SOURCE: &str = r#"
__forceinline__ __device__ unsigned int tvm_builtin_opaque_sm_idx_u32(unsigned int x) {
    unsigned int y;
    asm volatile("mov.u32 %0, %1;" : "=r"(y) : "r"(x));
    return y;
}
"#;

const OPAQUE_WARP_ID_SOURCE: &str = r#"
__forceinline__ __device__ int tvm_builtin_opaque_warp_id(int x) {
    int y;
    asm volatile("mov.u32 %0, %1;" : "=r"(y) : "r"(x));
    return y;
}
"#;

const TCGEN05_MMA_MXF4_BLOCK32_SS_SOURCE: &str = r#"
__forceinline__ __device__ void tvm_builtin_tcgen05_mma_mxf4_block32_ss(
    uint32_t tmem_c,
    uint64_t desc_a,
    uint64_t desc_b,
    uint32_t i_desc,
    uint32_t scale_c,
    uint32_t tmem_sfa,
    uint32_t tmem_sfb) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 "
        "[%0], %1, %2, %3, [%5], [%6], p;\n"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(i_desc), "r"(scale_c),
          "r"(tmem_sfa), "r"(tmem_sfb));
}
"#;

const PREFETCH_L2_SOURCE: &str = r#"
__forceinline__ __device__ void tirx_prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
"#;

const RELU2_FMA_F32X2_SOURCE: &str = r#"
__forceinline__ __device__ void tirx_relu2_fma_f32x2(uint64_t* d, unsigned long long a, unsigned long long w, unsigned long long c) {
    asm volatile(
        "{\n"
        ".reg .f32 al, ah, rl, rh;\n"
        ".reg .b64 rp;\n"
        "mov.b64 {al, ah}, %1;\n"
        "abs.f32 rl, al;\n"
        "abs.f32 rh, ah;\n"
        "mov.b64 rp, {rl, rh};\n"
        "add.rn.f32x2 rp, %1, rp;\n"
        "fma.rn.f32x2 %0, rp, %2, %3;\n"
        "}\n"
        : "=l"(*reinterpret_cast<uint64_t*>(d))
        : "l"(a), "l"(w), "l"(c));
}
"#;

const ST_ASYNC_CLUSTER_TASK_INFO_SOURCE: &str = r#"
__forceinline__ __device__ void tvm_builtin_st_async_cluster_task_info(
    void* dst, void* bar, uint32_t dst_cta_idx,
    uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3,
    uint32_t v4, uint32_t v5, uint32_t v6, uint32_t v7) {
    const uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    const uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t mapped_bar, mapped_dst;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(mapped_bar) : "r"(bar_addr), "r"(dst_cta_idx));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(mapped_dst) : "r"(dst_addr), "r"(dst_cta_idx));
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4 [%0], {%1, %2, %3, %4}, [%5];" ::
        "r"(mapped_dst), "r"(v0), "r"(v1), "r"(v2), "r"(v3), "r"(mapped_bar));
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4 [%0], {%1, %2, %3, %4}, [%5];" ::
        "r"(mapped_dst + 16), "r"(v4), "r"(v5), "r"(v6), "r"(v7), "r"(mapped_bar));
}
"#;

const SMEM_DESC_ADD_16B_OFFSET_SOURCE: &str = r#"
__forceinline__ __device__ uint64_t tvm_builtin_smem_desc_add_16B_offset(
    uint64_t desc_base, int32_t offset) {
    SmemDescriptor desc;
    desc.desc_ = desc_base;
    desc.lo += static_cast<uint32_t>(offset);
    return desc.desc_;
}
"#;

const SMEM_DESC_MAKE_LO_UNIFORM_SOURCE: &str = r#"
__forceinline__ __device__ void smem_desc_make_lo_uniform(uint64_t* desc) {
    SmemDescriptor* d = reinterpret_cast<SmemDescriptor*>(desc);
    d->lo = __shfl_sync(0xffffffff, d->lo, 0);
}
"#;

fn mqa_wrelu_reduce_64_source(function_name: &str) -> String {
    format!(
        r#"
__forceinline__ __device__ float {function_name}(
const float* __restrict__ accum, const float* __restrict__ weights) {{
    float2 sum_0 = make_float2(0.0f, 0.0f);
    float2 sum_1 = make_float2(0.0f, 0.0f);
    #pragma unroll
    for (int j = 0; j < 64; j += 4) {{
        float2 a_0 = make_float2(accum[j], accum[j + 1]);
        float2 a_1 = make_float2(fabsf(accum[j]), fabsf(accum[j + 1]));
        sum_0 = __ffma2_rn(__fadd2_rn(a_0, a_1), make_float2(weights[j], weights[j + 1]), sum_0);
        float2 a_2 = make_float2(accum[j + 2], accum[j + 3]);
        float2 a_3 = make_float2(fabsf(accum[j + 2]), fabsf(accum[j + 3]));
        sum_1 = __ffma2_rn(__fadd2_rn(a_2, a_3), make_float2(weights[j + 2], weights[j + 3]), sum_1);
    }}
    float2 sum = __fadd2_rn(sum_0, sum_1);
    return (sum.x + sum.y) / 2.0f;
}}
"#
    )
}

fn string_imm(value: &ObjectRef, field: &str) -> AResult<String> {
    match value.as_node::<StringImmObj>() {
        Some(imm) => Ok(ffi_text(&imm.value)),
        None => unsupported(format!("{field} must be a static string")),
    }
}

fn canonical_source(value: &str) -> String {
    value.chars().filter(|c| !c.is_whitespace()).collect()
}

/// `"handle"` for pointers, else the dtype text or `""`.
fn helper_dtype(node: &ObjectRef) -> String {
    let Some(ty) = expr_type(node) else {
        return String::new();
    };
    if ty.as_node::<PointerTypeObj>().is_some() {
        return "handle".to_owned();
    }
    prim_dtype(&ty).map(dtype_text).unwrap_or_default()
}

/// `address_of(TensorLoad)` with exactly one argument: the load node.
fn address_of_load(value: &ObjectRef) -> AResult<Option<ObjectRef>> {
    let Some(call) = value.as_node::<CallObj>() else {
        return Ok(None);
    };
    if call_op_name(call)?.as_deref() != Some("tirx.address_of") || call.args.len() != 1 {
        return Ok(None);
    }
    let argument = oref(call.args.get(0)?);
    if argument.as_node::<TensorLoadObj>().is_none() {
        return Ok(None);
    }
    Ok(Some(argument))
}

fn pointer_scope_and_dtype(value: &ObjectRef) -> Option<(String, String)> {
    let ty = expr_type(value)?;
    let pointer = ty.as_node::<PointerTypeObj>()?;
    let dtype = prim_dtype(&pointer.element_type)
        .map(dtype_text)
        .unwrap_or_default();
    Some((ffi_text(&pointer.storage_scope), dtype))
}

fn require_local_f32_array_address(
    value: &ObjectRef,
    field: &str,
    element_count: i64,
) -> AResult<()> {
    let Some(load_node) = address_of_load(value)? else {
        return unsupported(format!(
            "{field} must be address_of a local float32 array element"
        ));
    };
    match pointer_scope_and_dtype(value) {
        Some((scope, dtype)) if scope == "local" && dtype == "float32" => {}
        _ => return unsupported(format!("{field} must be a local float32 pointer")),
    }
    let load = load_node.as_node::<TensorLoadObj>().expect("tensor load");
    let Some(source) = as_buffer(&oref(load.source.clone())) else {
        return unsupported(format!("{field} must address local float32 storage"));
    };
    if buffer_scope(&source) != "local" || buffer_dtype(&source) != "float32" {
        return unsupported(format!("{field} must address local float32 storage"));
    }
    let indices: Vec<ObjectRef> = load.indices.iter().map(oref).collect();
    let Some(last) = indices.last() else {
        return unsupported(format!("{field} final array index must be a static zero"));
    };
    let Some(base) = last.as_node::<IntImmObj>().and_then(|imm| int_value(imm).ok()) else {
        return unsupported(format!("{field} final array index must be a static zero"));
    };
    let shape: Vec<ObjectRef> = source.buffer_type().shape.iter().map(oref).collect();
    let extent = match shape.last() {
        Some(extent) if base == 0 => int_imm(extent),
        _ => None,
    };
    let Some(extent) = extent else {
        return unsupported(format!("{field} final array index must be a static zero"));
    };
    if extent < element_count {
        return unsupported(format!(
            "{field} requires at least {element_count} contiguous elements, got {extent}"
        ));
    }
    Ok(())
}

fn require_local_u64_address(value: &ObjectRef, field: &str) -> AResult<()> {
    let Some(load_node) = address_of_load(value)? else {
        return unsupported(format!("{field} must be address_of local uint64 storage"));
    };
    let load = load_node.as_node::<TensorLoadObj>().expect("tensor load");
    let source = as_buffer(&oref(load.source.clone()));
    let pointer_ok = matches!(
        pointer_scope_and_dtype(value),
        Some((scope, dtype)) if scope == "local" && dtype == "uint64"
    );
    let source_ok = source
        .as_ref()
        .is_some_and(|source| buffer_scope(source) == "local" && buffer_dtype(source) == "uint64");
    if !pointer_ok || !source_ok {
        return unsupported(format!("{field} must be a local uint64 pointer"));
    }
    Ok(())
}

fn require_concrete_buffer_pointer(value: &ObjectRef, field: &str) -> AResult<()> {
    if let Some((scope, _)) = pointer_scope_and_dtype(value) {
        if scope.starts_with("shared") {
            return Ok(());
        }
    }
    if address_of_load(value)?.is_some() {
        return Ok(());
    }
    unsupported(format!("{field} must be a concrete buffer address"))
}

fn parse_indexed(name: &str, prefix: &str, max: i64) -> Option<i64> {
    let rest = name.strip_prefix(prefix)?;
    let mut chars = rest.chars();
    let digit = chars.next()?;
    if chars.next().is_some() || !digit.is_ascii_digit() {
        return None;
    }
    let index = i64::from(digit as u8 - b'0');
    (index <= max).then_some(index)
}

/// `classify_known_cuda_func_call`: `None` for every other call.
pub fn classify_known_cuda_func_call(node: &ObjectRef) -> AResult<Option<KnownCudaFuncCall>> {
    let Some(call) = node.as_node::<CallObj>() else {
        return Ok(None);
    };
    if call_op_name(call)?.as_deref() != Some("tirx.cuda.func_call") {
        return Ok(None);
    }
    let arguments: Vec<ObjectRef> = call.args.iter().map(oref).collect();
    let function_name = match arguments
        .first()
        .and_then(|first| first.as_node::<StringImmObj>())
    {
        Some(imm) => ffi_text(&imm.value),
        None => "<non-static>".to_owned(),
    };
    let quoted = format!("{:?}", &function_name);
    let mut index: Option<i64> = None;
    let kind = if function_name == "flashkda_tensormap_acquire"
        || function_name == "gdn_tensormap_acquire"
    {
        KnownCudaFuncKind::TensorMapAcquire
    } else if function_name == "gdn_tensormap_release" {
        KnownCudaFuncKind::TensorMapRelease
    } else if function_name == "gdn_tensormap_replace_global_address" {
        KnownCudaFuncKind::TensorMapReplaceGlobalAddress
    } else if let Some(found) =
        parse_indexed(&function_name, "gdn_tensormap_replace_global_dim_", 4)
    {
        index = Some(found);
        KnownCudaFuncKind::TensorMapReplaceGlobalDim
    } else if let Some(found) =
        parse_indexed(&function_name, "gdn_tensormap_replace_global_stride_", 3)
    {
        index = Some(found);
        KnownCudaFuncKind::TensorMapReplaceGlobalStride
    } else {
        match KnownCudaFuncKind::from_value(&function_name) {
            Some(kind) => kind,
            None => {
                return unsupported(format!(
                    "tirx.cuda.func_call helper {quoted} is unsupported; opaque CUDA helper bodies are not part of TIRx semantics"
                ))
            }
        }
    };
    if arguments.len() < 2 {
        return unsupported(format!(
            "tirx.cuda.func_call helper {quoted} is missing its source body"
        ));
    }
    let source = string_imm(
        arguments.last().expect("source body"),
        &format!("tirx.cuda.func_call helper {quoted} body"),
    )?;
    let operands: Vec<ObjectRef> = arguments[1..arguments.len() - 1].to_vec();
    let expected_dtype: &str;
    let expected_argument_dtypes: Vec<&str>;
    let expected_source: String;
    let source_description: String;
    match kind {
        KnownCudaFuncKind::CombineIntFracEx2 => {
            expected_dtype = "float32";
            expected_argument_dtypes = vec!["float32", "float32"];
            expected_source = COMBINE_INT_FRAC_EX2_SOURCE.to_owned();
            source_description = "validated bit-composition implementation".to_owned();
        }
        KnownCudaFuncKind::FlashkdaFmafRn => {
            expected_dtype = "float32";
            expected_argument_dtypes = vec!["float32", "float32", "float32"];
            expected_source = FLASHKDA_FMAF_RN_SOURCE.to_owned();
            source_description =
                "validated fused round-to-nearest float32 implementation".to_owned();
        }
        KnownCudaFuncKind::FlashkdaRsqrtf => {
            expected_dtype = "float32";
            expected_argument_dtypes = vec!["float32"];
            expected_source = FLASHKDA_RSQRTF_SOURCE.to_owned();
            source_description =
                "validated float32 reciprocal-square-root implementation".to_owned();
        }
        KnownCudaFuncKind::FlashkdaTanhApprox => {
            expected_dtype = "float32";
            expected_argument_dtypes = vec!["float32"];
            expected_source = FLASHKDA_TANH_APPROX_SOURCE.to_owned();
            source_description = "validated tanh.approx.f32 implementation".to_owned();
        }
        KnownCudaFuncKind::TensorMapAcquire => {
            expected_dtype = "";
            expected_argument_dtypes = vec!["handle"];
            expected_source = if function_name == "flashkda_tensormap_acquire" {
                FLASHKDA_TENSOR_MAP_ACQUIRE_SOURCE.to_owned()
            } else {
                GDN_TENSOR_MAP_ACQUIRE_SOURCE.to_owned()
            };
            source_description = "validated TensorMap acquire fence".to_owned();
        }
        KnownCudaFuncKind::TensorMapRelease => {
            expected_dtype = "";
            expected_argument_dtypes = vec![];
            expected_source = GDN_TENSOR_MAP_RELEASE_SOURCE.to_owned();
            source_description = "validated TensorMap release fence".to_owned();
        }
        KnownCudaFuncKind::TensorMapReplaceGlobalAddress => {
            expected_dtype = "";
            expected_argument_dtypes = vec!["handle", "handle"];
            expected_source = GDN_TENSOR_MAP_REPLACE_GLOBAL_ADDRESS_SOURCE.to_owned();
            source_description = "validated TensorMap global-address replacement".to_owned();
        }
        KnownCudaFuncKind::TensorMapReplaceGlobalDim => {
            let index = index.expect("dim index");
            expected_dtype = "";
            expected_argument_dtypes = vec!["handle", "uint32"];
            expected_source = gdn_tensor_map_replace_global_dim_source(index);
            source_description =
                format!("validated TensorMap global-dimension {index} replacement");
        }
        KnownCudaFuncKind::TensorMapReplaceGlobalStride => {
            let index = index.expect("stride index");
            expected_dtype = "";
            expected_argument_dtypes = vec!["handle", "uint64"];
            expected_source = gdn_tensor_map_replace_global_stride_source(index);
            source_description = format!("validated TensorMap global-stride {index} replacement");
        }
        KnownCudaFuncKind::FmaScaleSubF32x2 => {
            expected_dtype = "uint64";
            expected_argument_dtypes = vec!["uint64", "uint64", "uint64"];
            expected_source = FMA_SCALE_SUB_F32X2_SOURCE.to_owned();
            source_description = "validated packed float32 FMA implementation".to_owned();
        }
        KnownCudaFuncKind::GdnCpPrefillPredicatedGamma => {
            expected_dtype = "float32";
            expected_argument_dtypes = vec!["uint32", "uint32", "uint32"];
            expected_source = GDN_CP_PREFILL_PREDICATED_GAMMA_SOURCE.to_owned();
            source_description = "validated predicated shared-load exp2 implementation".to_owned();
        }
        KnownCudaFuncKind::GdnLg2ApproxFtz => {
            expected_dtype = "float32";
            expected_argument_dtypes = vec!["float32"];
            expected_source = GDN_LG2_APPROX_FTZ_SOURCE.to_owned();
            source_description = "validated lg2.approx.ftz.f32 implementation".to_owned();
        }
        KnownCudaFuncKind::OpaqueSmIdxU32 => {
            expected_dtype = "uint32";
            expected_argument_dtypes = vec!["uint32"];
            expected_source = OPAQUE_SM_IDX_U32_SOURCE.to_owned();
            source_description = "validated register-move identity implementation".to_owned();
        }
        KnownCudaFuncKind::OpaqueWarpId => {
            expected_dtype = "int32";
            expected_argument_dtypes = vec!["int32"];
            expected_source = OPAQUE_WARP_ID_SOURCE.to_owned();
            source_description = "validated register-move identity implementation".to_owned();
        }
        KnownCudaFuncKind::Tcgen05MmaMxf4Block32Ss => {
            expected_dtype = "";
            expected_argument_dtypes = vec![
                "uint32", "uint64", "uint64", "uint32", "uint32", "uint32", "uint32",
            ];
            expected_source = TCGEN05_MMA_MXF4_BLOCK32_SS_SOURCE.to_owned();
            source_description =
                "validated CTA1 MXF4 block32 TCGEN05 MMA implementation".to_owned();
        }
        KnownCudaFuncKind::ShlU32Clamp => {
            expected_dtype = "uint32";
            expected_argument_dtypes = vec!["uint32", "uint32"];
            expected_source = SHL_U32_CLAMP_SOURCE.to_owned();
            source_description = "validated PTX clamping-shift implementation".to_owned();
        }
        KnownCudaFuncKind::MqaFp4WreluReduce64 | KnownCudaFuncKind::MqaFp8WreluReduce64 => {
            expected_dtype = "float32";
            expected_argument_dtypes = vec!["handle", "handle"];
            expected_source = mqa_wrelu_reduce_64_source(kind.value());
            source_description = "validated 64-element packed weighted-ReLU reduction".to_owned();
            if operands.len() != 2 {
                return unsupported(format!(
                    "tirx.cuda.func_call helper {quoted} expects two pointer arguments"
                ));
            }
            require_local_f32_array_address(
                &operands[0],
                &format!("tirx.cuda.func_call helper {quoted} accum"),
                64,
            )?;
            require_local_f32_array_address(
                &operands[1],
                &format!("tirx.cuda.func_call helper {quoted} weights"),
                64,
            )?;
        }
        KnownCudaFuncKind::PrefetchL2 => {
            expected_dtype = "";
            expected_argument_dtypes = vec!["handle"];
            expected_source = PREFETCH_L2_SOURCE.to_owned();
            source_description = "validated L2 prefetch implementation".to_owned();
        }
        KnownCudaFuncKind::Relu2FmaF32x2 => {
            expected_dtype = "";
            expected_argument_dtypes = vec!["handle", "uint64", "uint64", "uint64"];
            expected_source = RELU2_FMA_F32X2_SOURCE.to_owned();
            source_description =
                "validated packed two-lane weighted-ReLU FMA implementation".to_owned();
            if operands.len() != 4 {
                return unsupported(format!(
                    "tirx.cuda.func_call helper {quoted} expects a destination and three operands"
                ));
            }
            require_local_u64_address(
                &operands[0],
                &format!("tirx.cuda.func_call helper {quoted} destination"),
            )?;
        }
        KnownCudaFuncKind::SmemDescAdd16bOffset => {
            expected_dtype = "uint64";
            expected_argument_dtypes = vec!["uint64", "int32"];
            expected_source = SMEM_DESC_ADD_16B_OFFSET_SOURCE.to_owned();
            source_description = "validated low-32-bit wrapping descriptor offset".to_owned();
        }
        KnownCudaFuncKind::SmemDescMakeLoUniform => {
            expected_dtype = "";
            expected_argument_dtypes = vec!["handle"];
            expected_source = SMEM_DESC_MAKE_LO_UNIFORM_SOURCE.to_owned();
            source_description = "validated lane-zero low-32-bit descriptor broadcast".to_owned();
            if operands.len() != 1 {
                return unsupported(format!(
                    "tirx.cuda.func_call helper {quoted} expects one destination"
                ));
            }
            require_local_u64_address(
                &operands[0],
                &format!("tirx.cuda.func_call helper {quoted} destination"),
            )?;
        }
        KnownCudaFuncKind::StAsyncClusterTaskInfo => {
            expected_dtype = "";
            let mut dtypes = vec!["handle", "handle"];
            dtypes.extend(std::iter::repeat("uint32").take(9));
            expected_argument_dtypes = dtypes;
            expected_source = ST_ASYNC_CLUSTER_TASK_INFO_SOURCE.to_owned();
            source_description =
                "validated remote 32-byte task-info store implementation".to_owned();
            if operands.len() != 11 {
                return unsupported(format!(
                    "tirx.cuda.func_call helper {quoted} expects two pointers, a CTA rank, and eight uint32 values"
                ));
            }
            require_concrete_buffer_pointer(
                &operands[0],
                &format!("tirx.cuda.func_call helper {quoted} destination"),
            )?;
            require_concrete_buffer_pointer(
                &operands[1],
                &format!("tirx.cuda.func_call helper {quoted} barrier"),
            )?;
        }
    }
    let actual: Vec<String> = operands.iter().map(helper_dtype).collect();
    let expected: Vec<String> = expected_argument_dtypes
        .iter()
        .map(|dtype| (*dtype).to_owned())
        .collect();
    if helper_dtype(node) != expected_dtype || actual != expected {
        return unsupported(format!(
            "tirx.cuda.func_call helper {quoted} requires {:?} -> {}",
            &expected,
            if expected_dtype.is_empty() {
                "void"
            } else {
                expected_dtype
            }
        ));
    }
    if canonical_source(&source) != canonical_source(&expected_source) {
        return unsupported(format!(
            "tirx.cuda.func_call helper {quoted} body does not match the {source_description}"
        ));
    }
    Ok(Some(KnownCudaFuncCall {
        kind,
        arguments: operands,
        index,
    }))
}

/// The registered lowerer only sees helper calls.
pub fn parse_known_cuda_func_call(node: &ObjectRef) -> AResult<KnownCudaFuncCall> {
    match classify_known_cuda_func_call(node)? {
        Some(parsed) => Ok(parsed),
        None => Err(crate::analyze::util::Failure::Ffi(
            crate::analyze::util::ffi_error("CUDA-helper lowerer received a non-helper call"),
        )),
    }
}

/// The engine variants of the helper bodies.
pub const MMA_MXF4_BLOCK32_SS_VARIANT: &str =
    "v2::tcgen05::variant::MmaBlockMxf4E8m0SsCta1<<artifact-tmem-mode>>";
pub const ST_ASYNC_TASK_INFO_VARIANT: &str =
    "v2::async_copy::variant::StAsyncClusterCompleteTxBytesU32x4";
pub const SMEM_DESC_LD_VARIANT: &str = "v2::mem::variant::Ld<v2::reg::variant::U64, v2::Generic>";
pub const SMEM_DESC_SHFL_VARIANT: &str =
    "v2::warp::variant::Shfl<v2::reg::variant::U32, v2::warp::variant::Index>";
pub const SMEM_DESC_ST_VARIANT: &str = "v2::mem::variant::St<v2::reg::variant::U64, v2::Generic>";

/// The parsed helper call.

// ----------------------------------------------------------------------
// Destination-passing CUDA helpers.
// ----------------------------------------------------------------------

pub const DPS_HELPERS: &[(&str, &[&str], usize)] = &[
    ("tirx.cuda.float22half2", &["handle", "handle"], 0),
    ("tirx.cuda.half8tofloat8", &["handle", "handle"], 1),
    ("tirx.cuda.float8tohalf8", &["handle", "handle"], 1),
    ("tirx.cuda.runtime_instr_desc", &["handle", "uint32"], 0),
];

pub fn validate_dps_helper(node: &ObjectRef, op_name: &str) -> AResult<()> {
    let Some(call) = node.as_node::<CallObj>() else {
        return Err(crate::analyze::util::Failure::Ffi(
            crate::analyze::util::ffi_error("destination-passing lowering expects a TIRx Call"),
        ));
    };
    let expected: &[&str] = DPS_HELPERS
        .iter()
        .find(|(name, _, _)| *name == op_name)
        .map(|(_, dtypes, _)| *dtypes)
        .expect("registered destination-passing helper");
    let mut actual: Vec<String> = Vec::new();
    for argument in call.args.iter() {
        actual.push(dtype_of(&oref(argument))?);
    }
    if !dtype_of(node)?.is_empty() {
        return unsupported(format!(
            "Call({op_name}): destination-passing call must be void"
        ));
    }
    if op_name == "tirx.cuda.runtime_instr_desc" {
        if actual.len() != 2
            || actual[0] != "handle"
            || !crate::tables::is_integer_dtype(&actual[1])
        {
            return unsupported(format!(
                "Call({op_name}): expected handle descriptor and integer sf_id, got {:?}",
                &actual
            ));
        }
    } else if actual != expected {
        return unsupported(format!(
            "Call({op_name}): expected args {:?}, got {:?}",
            &expected, &actual
        ));
    }
    Ok(())
}

/// A validated destination-passing CUDA helper call.

fn unit() -> RustValue {
    RustValue::new("()", "()", Uniformity::Uniform)
}

/// Destination-passing operands are plain pointers, so every load and store
/// this family issues is generic-space.
const GENERIC: &str = "v2::Generic";

fn memory_call(function: &str, variant: &str, source_op_id: i64, arguments: &str) -> String {
    abi::warp_call(
        function,
        &abi::site(source_op_id as u64),
        &[arguments.to_owned()],
        Some(variant),
        None,
        false,
        true,
    )
}

fn generic_address(pointer: &str) -> String {
    abi::address(GENERIC, &abi::cloned(pointer), None)
}

impl<'a> Emitter<'a> {
    /// A helper pointer operand with a caller-chosen binding label and pointer
    /// recovery.
    fn helper_pointer(
        &mut self,
        argument: &ObjectRef,
        label: &str,
        binding: &str,
        message: &str,
    ) -> AResult<String> {
        let mut value = self.emit_expr(argument)?;
        if value.rust_type != "PhysicalPtr" {
            value = self.emit_raw_generic_pointer(value, "ctx.active_mask()")?;
        }
        if value.rust_type != "PhysicalPtr" {
            return unsupported(format!("{label} {message}"));
        }
        let name = self.control_name(binding);
        self.emit_line(&format!("let {name} = ({}).clone();", value.code));
        Ok(name)
    }

    /// One `sync::tensormap_replace` call.
    fn emit_tensor_map_replace(&mut self, source_op_id: i64, arguments: String, variant: &str) {
        let site = self.v2_site(Some(source_op_id));
        let call = abi::warp_call(
            "sync::tensormap_replace",
            &site,
            &[arguments],
            Some(variant),
            None,
            false,
            true,
        );
        self.emit_line(&format!("{call};"));
    }

    fn emit_tensor_map_helper(
        &mut self,
        parsed: &KnownCudaFuncCall,
        source_op_id: i64,
    ) -> AResult<()> {
        let registry = self.tensor_map_registry_ref()?;
        match parsed.kind {
            KnownCudaFuncKind::TensorMapAcquire => {
                let descriptor = self.helper_pointer(
                    &parsed.arguments[0],
                    "tensor_map_acquire_descriptor",
                    "tensor_map_acquire_descriptor",
                    "did not resolve to a physical address",
                )?;
                self.emit_line(&format!(
                    "{registry}.acquire(warp, v2_context(ctx), {}, &{descriptor}, MemoryScope::Gpu)?;",
                    abi::site(source_op_id as u64)
                ));
                Ok(())
            }
            KnownCudaFuncKind::TensorMapRelease => {
                self.emit_line(&format!(
                    "{registry}.release(warp, v2_context(ctx), {}, MemoryScope::Gpu)?;",
                    abi::site(source_op_id as u64)
                ));
                Ok(())
            }
            KnownCudaFuncKind::TensorMapReplaceGlobalAddress => {
                let descriptor = self.helper_pointer(
                    &parsed.arguments[0],
                    "tensor_map_replace_descriptor",
                    "tensor_map_replace_descriptor",
                    "did not resolve to a physical address",
                )?;
                let address = self.helper_pointer(
                    &parsed.arguments[1],
                    "tensor_map_replace_global_address",
                    "tensor_map_replace_global_address",
                    "did not resolve to a physical address",
                )?;
                self.emit_tensor_map_replace(
                    source_op_id,
                    format!(
                        "({}, {registry}.clone(), {})",
                        abi::address("v2::Global", &descriptor, None),
                        abi::address("v2::Global", &address, None)
                    ),
                    "v2::sync::variant::TensorMapAddress<v2::Global>",
                );
                Ok(())
            }
            KnownCudaFuncKind::TensorMapReplaceGlobalDim
            | KnownCudaFuncKind::TensorMapReplaceGlobalStride => {
                let index = parsed.index.expect("replacement index");
                let descriptor = self.helper_pointer(
                    &parsed.arguments[0],
                    "tensor_map_replace_descriptor",
                    "tensor_map_replace_descriptor",
                    "did not resolve to a physical address",
                )?;
                let field = if parsed.kind == KnownCudaFuncKind::TensorMapReplaceGlobalDim {
                    "global_dim"
                } else {
                    "global_stride"
                };
                // These known C++ helpers take unsigned 32-/64-bit parameters before
                // passing their register bits to PTX.
                let operand = Cast::new(
                    PrimType::new(if field == "global_dim" {
                        "uint32"
                    } else {
                        "uint64"
                    })?,
                    prim(&parsed.arguments[1])?,
                )?;
                let value =
                    self.v2_register_operand(&oref(operand), "u64", "tensor_map_replace_value")?;
                self.emit_tensor_map_replace(
                    source_op_id,
                    format!(
                        "({}, {registry}.clone(), {}, Some({index}_usize), {value})",
                        abi::address("v2::Global", &descriptor, None),
                        json_string(field)
                    ),
                    "v2::sync::variant::TensorMapField<v2::Global>",
                );
                Ok(())
            }
            _ => unreachable!("TensorMap helper kinds only"),
        }
    }

    pub fn emit_cuda_helper(
        &mut self,
        expr: &ObjectRef,
        parsed: &KnownCudaFuncCall,
    ) -> AResult<RustValue> {
        if parsed.kind.is_tensor_map() {
            // The exact source occurrence is resolved before the family
            // lowering runs.
            let source_op_id = self.static_op_id(expr)?;
            self.emit_tensor_map_helper(&parsed, source_op_id)?;
            return Ok(unit());
        }
        if parsed.kind == KnownCudaFuncKind::Tcgen05MmaMxf4Block32Ss {
            let source_op_id = self.static_op_id(expr)?;
            self.emit_known_cuda_tcgen05_mxf4_block32_ss(&parsed.arguments, source_op_id)?;
            return Ok(unit());
        }
        if matches!(
            parsed.kind,
            KnownCudaFuncKind::MqaFp4WreluReduce64 | KnownCudaFuncKind::MqaFp8WreluReduce64
        ) {
            return self.emit_mqa_wrelu_reduce_64(&parsed, expr);
        }
        if parsed.kind == KnownCudaFuncKind::StAsyncClusterTaskInfo {
            let destination =
                self.emit_raw_shared_pointer(&parsed.arguments[0], None, "ctx.active_mask()")?;
            let barrier =
                self.emit_raw_shared_pointer(&parsed.arguments[1], None, "ctx.active_mask()")?;
            if destination.rust_type != "PhysicalPtr" || barrier.rust_type != "PhysicalPtr" {
                return unsupported(
                    "tvm_builtin_st_async_cluster_task_info pointers did not resolve to physical addresses",
                );
            }
            let destination_pointer = self.temp("st_async_destination");
            let barrier_pointer = self.temp("st_async_barrier");
            self.emit_line(&format!(
                "let {destination_pointer} = ({}).clone();",
                destination.code
            ));
            self.emit_line(&format!(
                "let {barrier_pointer} = ({}).clone();",
                barrier.code
            ));
            let rank = self.emit_expr(&parsed.arguments[2])?;
            let rank = self.coerce_value(rank, "i64", "st_async_cluster_rank")?;
            let rank = self.as_warp_value(rank);
            let mut values = Vec::new();
            for argument in &parsed.arguments[3..] {
                let value = self.emit_expr(argument)?;
                let value = self.coerce_value(value, "u32", "st_async_cluster_value")?;
                values.push(self.as_warp_value(value));
            }
            let site = self.expr_site(expr);
            let second_destination = self.temp("st_async_second_destination");
            self.emit_line(&format!(
                "let {second_destination} = ({destination_pointer}).with_byte_offset(&WarpValue::splat(16_i64), 1_usize, ctx.active_mask())?;"
            ));
            let variant = ST_ASYNC_TASK_INFO_VARIANT;
            for (pointer, chunk) in [
                (destination_pointer.as_str(), &values[..4]),
                (second_destination.as_str(), &values[4..]),
            ] {
                let value_args: Vec<String> = chunk
                    .iter()
                    .map(|value| abi::register(&value.code))
                    .collect();
                let arguments = format!(
                    "({}, {}, {}, [{}])",
                    abi::address("v2::Shared", &abi::cloned(pointer), None),
                    abi::address("v2::Shared", &abi::cloned(&barrier_pointer), None),
                    abi::register(&rank.code),
                    value_args.join(", ")
                );
                let call = abi::warp_call(
                    ST_ASYNC,
                    &site,
                    &[arguments],
                    Some(variant),
                    None,
                    false,
                    true,
                );
                self.emit_line(&format!("{call};"));
            }
            return Ok(unit());
        }
        if parsed.kind == KnownCudaFuncKind::SmemDescMakeLoUniform {
            let pointer = self.emit_expr(&parsed.arguments[0])?;
            if pointer.rust_type != "PhysicalPtr" {
                return unsupported(
                    "smem_desc_make_lo_uniform destination did not resolve to a physical address",
                );
            }
            let materialized_pointer = self.temp("smem_desc_uniform_pointer");
            let loaded = self.temp("smem_desc_uniform_loaded");
            let low = self.temp("smem_desc_uniform_low");
            let member_mask = self.temp("smem_desc_uniform_member_mask");
            let selector = self.temp("smem_desc_uniform_selector");
            let control = self.temp("smem_desc_uniform_control");
            let shuffled_raw = self.temp("smem_desc_uniform_shuffled_raw");
            let shuffled = self.temp("smem_desc_uniform_shuffled");
            let result = self.temp("smem_desc_uniform_result");
            self.emit_line(&format!(
                "let {materialized_pointer} = ({}).clone();",
                pointer.code
            ));
            let load_site = self.expr_site(expr);
            let load = abi::warp_call(
                MEM_LD,
                &load_site,
                &[abi::address(
                    "v2::Generic",
                    &abi::cloned(&materialized_pointer),
                    None,
                )],
                Some(SMEM_DESC_LD_VARIANT),
                None,
                false,
                true,
            );
            self.emit_line(&format!("let {loaded} = v2_register_out({load});"));
            self.emit_line(&format!(
                "let {low} = WarpValue::from_fn(|lane| {loaded}[lane] as u32);"
            ));
            self.emit_line(&format!(
                "let {member_mask} = WarpValue::splat(0xffff_ffff_u32);"
            ));
            self.emit_line(&format!("let {selector} = WarpValue::splat(0_u32);"));
            self.emit_line(&format!("let {control} = WarpValue::splat(31_u32);"));
            let shuffle_site = self.expr_site(expr);
            let shuffle = abi::warp_call(
                SHFL_SYNC,
                &shuffle_site,
                &[format!(
                    "({}, {}, {}, {})",
                    abi::register(&member_mask),
                    abi::register(&low),
                    abi::register(&selector),
                    abi::register(&control)
                )],
                Some(SMEM_DESC_SHFL_VARIANT),
                None,
                false,
                true,
            );
            self.emit_line(&format!("let {shuffled_raw} = {shuffle};"));
            self.emit_line(&format!(
                "let {shuffled} = v2_register_out({shuffled_raw});"
            ));
            self.emit_line(&format!(
                "let {result} = WarpValue::from_fn(|lane| ({loaded}[lane] & 0xffff_ffff_0000_0000_u64) | {shuffled}[lane] as u64);"
            ));
            let store_site = self.expr_site(expr);
            let store = abi::warp_call(
                MEM_ST,
                &store_site,
                &[format!(
                    "({}, {})",
                    abi::address("v2::Generic", &abi::cloned(&materialized_pointer), None),
                    abi::register(&result)
                )],
                Some(SMEM_DESC_ST_VARIANT),
                None,
                false,
                true,
            );
            self.emit_line(&format!("{store};"));
            return Ok(unit());
        }
        if parsed.kind == KnownCudaFuncKind::GdnCpPrefillPredicatedGamma {
            let predicate = self.emit_expr(&parsed.arguments[2])?;
            let predicate =
                self.coerce_value(predicate, "u32", "gdn_cp_prefill_gamma_predicate")?;
            let predicate = self.as_warp_value(predicate);
            let mask = self.temp("gdn_cp_prefill_gamma_mask");
            let context = self.temp("gdn_cp_prefill_gamma_context");
            self.emit_line(&format!(
                "let {mask} = {}.to_mask(|_, value| *value != 0_u32) & ctx.active_mask();",
                predicate.code
            ));
            self.emit_line(&format!("let {context} = ctx.with_active_mask({mask});"));
            let site = self.expr_site(expr);
            let mut loaded = Vec::new();
            for (label, argument) in [("s", &parsed.arguments[0]), ("t", &parsed.arguments[1])] {
                let pointer = self.emit_raw_shared_pointer(argument, None, "ctx.active_mask()")?;
                if pointer.rust_type != "PhysicalPtr" {
                    return unsupported(
                        "gdn_cp_prefill_predicated_gamma address did not resolve to a physical address",
                    );
                }
                let raw = self.temp(&format!("gdn_cp_prefill_gamma_{label}_raw"));
                let value = self.temp(&format!("gdn_cp_prefill_gamma_{label}"));
                let load = abi::warp_call(
                    "mem::ld",
                    &site,
                    &[abi::address(
                        "v2::Shared",
                        &abi::cloned(&pointer.code),
                        None,
                    )],
                    Some("v2::mem::variant::Ld<v2::reg::variant::F32, v2::Shared>"),
                    Some(&abi::context(&context)),
                    false,
                    true,
                );
                self.emit_line(&format!("let {raw} = {load};"));
                self.emit_line(&format!("let {value} = v2_register_out({raw});"));
                loaded.push(value);
            }
            let result = self.temp("gdn_cp_prefill_gamma");
            self.emit_line(&format!(
                "let {result} = WarpValue::from_fn(|lane| if {mask}.contains(lane) {{ ptx_exp2_approx_ftz_f32({}[lane] - {}[lane]) }} else {{ 0.0_f32 }});",
                loaded[0], loaded[1]
            ));
            return Ok(RustValue::new(result, "f32", Uniformity::Varying));
        }
        let mut arguments = Vec::new();
        for argument in &parsed.arguments {
            arguments.push(self.emit_expr(argument)?);
        }
        let prefix = parsed.kind.value();
        match parsed.kind {
            KnownCudaFuncKind::GdnLg2ApproxFtz => {
                self.emit_call_atom(prefix, &arguments, "f32", |codes| {
                    format!("ptx_lg2_approx_ftz_f32({})", codes[0])
                })
            }
            KnownCudaFuncKind::FlashkdaFmafRn => {
                self.emit_call_atom(prefix, &arguments, "f32", |codes| {
                    format!("({}).mul_add({}, {})", codes[0], codes[1], codes[2])
                })
            }
            KnownCudaFuncKind::FlashkdaRsqrtf => {
                self.emit_call_atom(prefix, &arguments, "f32", |codes| {
                    format!("1.0_f32 / ({}).sqrt()", codes[0])
                })
            }
            KnownCudaFuncKind::FlashkdaTanhApprox => {
                self.emit_call_atom(prefix, &arguments, "f32", |codes| {
                    format!("ptx_tanh_approx_f32({})", codes[0])
                })
            }
            KnownCudaFuncKind::CombineIntFracEx2 => {
                self.emit_call_atom(prefix, &arguments, "f32", |codes| {
                    format!(
                        "f32::from_bits((({}).to_bits().wrapping_shl(23_u32).wrapping_add(({}).to_bits())))",
                        codes[0], codes[1]
                    )
                })
            }
            KnownCudaFuncKind::FmaScaleSubF32x2 => {
                self.emit_call_atom(prefix, &arguments, "u64", |codes| {
                    format!(
                        "make_float2(float2_x({0}).mul_add(float2_x({1}), -float2_x({2})), float2_y({0}).mul_add(float2_y({1}), -float2_y({2})))",
                        codes[0], codes[1], codes[2]
                    )
                })
            }
            KnownCudaFuncKind::OpaqueSmIdxU32 | KnownCudaFuncKind::OpaqueWarpId => {
                Ok(arguments[0].clone())
            }
            KnownCudaFuncKind::PrefetchL2 => {
                let mut pointer = arguments[0].clone();
                if pointer.rust_type != "PhysicalPtr" {
                    pointer = self.emit_raw_generic_pointer(pointer, "ctx.active_mask()")?;
                }
                let materialized = self.temp("prefetch_l2_pointer");
                self.emit_line(&format!("let {materialized} = ({}).clone();", pointer.code));
                self.emit_line(&format!("let _ = &{materialized};"));
                Ok(unit())
            }
            KnownCudaFuncKind::Relu2FmaF32x2 => {
                let destination = arguments[0].clone();
                if destination.rust_type != "PhysicalPtr" {
                    return unsupported(
                        "tirx_relu2_fma_f32x2 destination requires a resolved physical address",
                    );
                }
                let operands: Vec<RustValue> = arguments[1..]
                    .iter()
                    .map(|value| self.as_warp_value(value.clone()))
                    .collect();
                if operands.iter().any(|value| value.rust_type != "u64") {
                    return unsupported(
                        "tirx_relu2_fma_f32x2 operands must lower to packed uint64 values",
                    );
                }
                let (values, weights, addends) = (&operands[0], &operands[1], &operands[2]);
                let magnitudes = self.temp("relu2_magnitudes");
                let relu_raw = self.temp("relu2_add_raw");
                let relu = self.temp("relu2_add");
                let result_raw = self.temp("relu2_fma_raw");
                let result = self.temp("relu2_fma");
                self.emit_line(&format!(
                    "let {magnitudes} = WarpValue::from_fn(|lane| make_float2(float2_x({0}[lane]).abs(), float2_y({0}[lane]).abs()));",
                    values.code
                ));
                let variant = "v2::reg::variant::F32x2Arithmetic<v2::reg::variant::Rn>";
                let add_site = self.expr_site(expr);
                let add = abi::lane_call(
                    "reg::add",
                    &add_site,
                    &[format!(
                        "({}, {})",
                        abi::register(&values.code),
                        abi::register(&magnitudes)
                    )],
                    Some(variant),
                );
                self.emit_line(&format!("let {relu_raw} = {add};"));
                self.emit_line(&format!("let {relu} = v2_register_out({relu_raw});"));
                let fma_site = self.expr_site(expr);
                let fma = abi::lane_call(
                    "reg::fma",
                    &fma_site,
                    &[format!(
                        "({}, {}, {})",
                        abi::register(&relu),
                        abi::register(&weights.code),
                        abi::register(&addends.code)
                    )],
                    Some(variant),
                );
                self.emit_line(&format!("let {result_raw} = {fma};"));
                self.emit_line(&format!("let {result} = v2_register_out({result_raw});"));
                let store_site = self.expr_site(expr);
                let store = abi::warp_call(
                    "mem::st",
                    &store_site,
                    &[format!(
                        "({}, {})",
                        abi::address("v2::Generic", &abi::cloned(&destination.code), None),
                        abi::register(&result)
                    )],
                    Some("v2::mem::variant::St<v2::reg::variant::U64, v2::Generic>"),
                    None,
                    false,
                    true,
                );
                self.emit_line(&format!("{store};"));
                Ok(unit())
            }
            KnownCudaFuncKind::SmemDescAdd16bOffset => {
                self.emit_call_atom(prefix, &arguments, "u64", |codes| {
                    format!(
                        "(({0}) & 0xffff_ffff_0000_0000_u64) | ((({0}) as u32).wrapping_add(({1}) as u32) as u64)",
                        codes[0], codes[1]
                    )
                })
            }
            KnownCudaFuncKind::ShlU32Clamp => {
                self.emit_call_atom(prefix, &arguments, "u32", |codes| {
                    format!("({}).checked_shl({}).unwrap_or(0_u32)", codes[0], codes[1])
                })
            }
            other => unsupported(format!(
                "Unhandled known CUDA helper {:?}",
                other.value())),
        }
    }

    fn emit_mqa_wrelu_reduce_64(
        &mut self,
        call: &KnownCudaFuncCall,
        source_node: &ObjectRef,
    ) -> AResult<RustValue> {
        let mut pointers = Vec::new();
        for argument in &call.arguments {
            pointers.push(self.emit_expr(argument)?);
        }
        if pointers
            .iter()
            .any(|pointer| pointer.rust_type != "PhysicalPtr")
        {
            return unsupported(format!("{} requires physical pointers", call.kind.value()));
        }
        let mut materialized_pointers = Vec::new();
        for pointer in &pointers {
            let name = self.temp("mqa_pointer");
            self.emit_line(&format!("let {name} = ({}).clone();", pointer.code));
            materialized_pointers.push(RustValue::new(name, "PhysicalPtr", pointer.uniformity));
        }
        let pointers = materialized_pointers;
        let result = self.temp("mqa_wrelu_reduce");
        let accumulators = [self.temp("mqa_sum_0"), self.temp("mqa_sum_1")];
        for accumulator in &accumulators {
            self.emit_line(&format!(
                "let mut {accumulator} = WarpValue::splat(make_float2(0.0_f32, 0.0_f32));"
            ));
        }
        let loop_index = self.temp("mqa_head");
        self.emit_line(&format!(
            "for {loop_index} in (0_i64..64_i64).step_by(4_usize) {{"
        ));
        let mut loaded: Vec<(String, String)> = Vec::new();
        for offset in 0..4 {
            let mut values = Vec::new();
            for (pointer_index, pointer) in pointers.iter().enumerate() {
                let byte_offset = self.temp("mqa_byte_offset");
                let element_pointer = self.temp("mqa_element_pointer");
                let value = self.temp(if pointer_index == 0 {
                    "mqa_value"
                } else {
                    "mqa_weight"
                });
                self.emit_line(&format!(
                    "let {byte_offset} = WarpValue::splat(({loop_index} + {offset}_i64) * 4_i64);"
                ));
                self.emit_line(&format!(
                    "let {element_pointer} = physical_ptr_byte_offset(&{}, &{byte_offset}, 4_usize, ctx.active_mask())?;",
                    pointer.code
                ));
                let site = self.expr_site(source_node);
                let load = abi::warp_call(
                    "mem::ld",
                    &site,
                    &[abi::address(
                        "v2::Generic",
                        &abi::cloned(&element_pointer),
                        None,
                    )],
                    Some("v2::mem::variant::Ld<v2::reg::variant::F32, v2::Generic>"),
                    None,
                    false,
                    true,
                );
                self.emit_line(&format!("let {value} = v2_register_out({load});"));
                values.push(value);
            }
            loaded.push((values[0].clone(), values[1].clone()));
        }
        let ftz_variant =
            "v2::reg::variant::F32x2Arithmetic<v2::reg::variant::Rn, v2::reg::variant::Ftz>";
        for (pair_index, accumulator) in accumulators.iter().enumerate() {
            let first = loaded[2 * pair_index].clone();
            let second = loaded[2 * pair_index + 1].clone();
            let relu_pair = self.temp("mqa_relu_pair");
            let weight_pair = self.temp("mqa_weight_pair");
            let values = self.temp("mqa_values_pair");
            let magnitudes = self.temp("mqa_magnitudes_pair");
            self.emit_line(&format!(
                "let {values} = WarpValue::from_fn(|lane| make_float2({}[lane], {}[lane]));",
                first.0, second.0
            ));
            self.emit_line(&format!(
                "let {magnitudes} = WarpValue::from_fn(|lane| make_float2({}[lane].abs(), {}[lane].abs()));",
                first.0, second.0
            ));
            let add_site = self.expr_site(source_node);
            let add = abi::lane_call(
                "reg::add",
                &add_site,
                &[format!(
                    "(v2_register({values}), v2_register({magnitudes}))"
                )],
                Some(ftz_variant),
            );
            self.emit_line(&format!("let {relu_pair} = v2_register_out({add});"));
            self.emit_line(&format!(
                "let {weight_pair} = WarpValue::from_fn(|lane| make_float2({}[lane], {}[lane]));",
                first.1, second.1
            ));
            let fma_site = self.expr_site(source_node);
            let fma = abi::lane_call(
                "reg::fma",
                &fma_site,
                &[format!(
                    "(v2_register({relu_pair}), v2_register({weight_pair}), v2_register(({accumulator}).clone()))"
                )],
                Some("v2::reg::variant::F32x2Arithmetic<v2::reg::variant::Rn>"),
            );
            self.emit_line(&format!("{accumulator} = v2_register_out({fma});"));
        }
        self.emit_line("}");

        let packed_sum = self.temp("mqa_packed_sum");
        let packed_site = self.expr_site(source_node);
        let packed = abi::lane_call(
            "reg::add",
            &packed_site,
            &[format!(
                "(v2_register({}), v2_register({}))",
                accumulators[0], accumulators[1]
            )],
            Some(ftz_variant),
        );
        self.emit_line(&format!("let {packed_sum} = v2_register_out({packed});"));
        let sum_lhs = self.temp("mqa_sum_lhs");
        let sum_rhs = self.temp("mqa_sum_rhs");
        let sum_raw = self.temp("mqa_sum_raw");
        let sum_value = self.temp("mqa_sum");
        self.emit_line(&format!(
            "let {sum_lhs} = WarpValue::from_fn(|lane| float2_x({packed_sum}[lane]));"
        ));
        self.emit_line(&format!(
            "let {sum_rhs} = WarpValue::from_fn(|lane| float2_y({packed_sum}[lane]));"
        ));
        let sum_site = self.expr_site(source_node);
        let sum = abi::lane_call(
            "reg::add",
            &sum_site,
            &[format!("(v2_register({sum_lhs}), v2_register({sum_rhs}))")],
            Some("v2::reg::variant::F32Rn"),
        );
        self.emit_line(&format!("let {sum_raw} = {sum};"));
        self.emit_line(&format!("let {sum_value} = v2_register_out({sum_raw});"));
        let div_site = self.expr_site(source_node);
        let div = abi::lane_call(
            "reg::div",
            &div_site,
            &[format!(
                "(v2_register({sum_value}), v2_register(WarpValue::splat(2.0_f32)))"
            )],
            Some("v2::reg::variant::F32Rn"),
        );
        self.emit_line(&format!("let {result} = v2_register_out({div});"));
        Ok(RustValue::new(result, "f32", Uniformity::Varying))
    }

    // ------------------------------------------------------------------
    // Destination-passing helpers.
    // ------------------------------------------------------------------

    fn dps_pointer(&mut self, expression: &ObjectRef, label: &str) -> AResult<String> {
        self.helper_pointer(
            expression,
            label,
            "dps_pointer",
            "did not resolve to a physical address",
        )
    }

    fn dps_offset_pointer(
        &mut self,
        pointer: &str,
        byte_offset: i64,
        itemsize: i64,
        prefix: &str,
    ) -> String {
        let offsets = self.control_name(&format!("{prefix}_offsets"));
        let result = self.control_name(&format!("{prefix}_pointer"));
        self.emit_line(&format!(
            "let {offsets} = WarpValue::splat({byte_offset}_i64);"
        ));
        self.emit_line(&format!(
            "let {result} = {pointer}.with_byte_offset(&{offsets}, {itemsize}_usize, ctx.active_mask())?;"
        ));
        result
    }

    fn emit_float22half2(&mut self, args: &[ObjectRef], source_op_id: i64) -> AResult<()> {
        let destination = self.dps_pointer(&args[0], "float22half2 destination")?;
        let source = self.dps_pointer(&args[1], "float22half2 source")?;
        let loaded = self.control_name("float2_source");
        let converted = self.control_name("half2_result");
        self.emit_line(&format!(
            "let {loaded} = {};",
            memory_call(
                MEM_LD,
                "v2::mem::variant::Ld<v2::reg::variant::U64, v2::Generic>",
                source_op_id,
                &generic_address(&source),
            )
        ));
        self.emit_line(&format!(
            "let {converted} = {};",
            abi::per_lane(&format!(
                "u32::from(cuda_f32_to_fp16_bits(float2_x({loaded}[lane]))) | (u32::from(cuda_f32_to_fp16_bits(float2_y({loaded}[lane]))) << 16)"
            ))
        ));
        self.emit_line(&format!(
            "{};",
            memory_call(
                MEM_ST,
                "v2::mem::variant::St<v2::reg::variant::U32, v2::Generic>",
                source_op_id,
                &format!("({}, {converted})", generic_address(&destination)),
            )
        ));
        Ok(())
    }

    fn emit_packed_conversion(
        &mut self,
        args: &[ObjectRef],
        source_op_id: i64,
        op_name: &str,
        half_to_float: bool,
    ) -> AResult<()> {
        let source = self.dps_pointer(&args[0], &format!("{op_name} source"))?;
        let destination = self.dps_pointer(&args[1], &format!("{op_name} destination"))?;
        for index in 0..4 {
            let source_itemsize = if half_to_float { 4 } else { 8 };
            let destination_itemsize = if half_to_float { 8 } else { 4 };
            let source_pointer = self.dps_offset_pointer(
                &source,
                index * source_itemsize,
                source_itemsize,
                "convert_source",
            );
            let destination_pointer = self.dps_offset_pointer(
                &destination,
                index * destination_itemsize,
                destination_itemsize,
                "convert_destination",
            );
            let loaded = self.control_name("convert_loaded");
            let converted = self.control_name("convert_result");
            let load_type = if half_to_float { "U32" } else { "U64" };
            self.emit_line(&format!(
                "let {loaded} = {};",
                memory_call(
                    MEM_LD,
                    &format!("v2::mem::variant::Ld<v2::reg::variant::{load_type}, v2::Generic>"),
                    source_op_id,
                    &generic_address(&source_pointer),
                )
            ));
            let store_type = if half_to_float {
                self.emit_line(&format!(
                    "let {converted} = {};",
                    abi::per_lane(&format!(
                        "make_float2(cuda_fp16_bits_to_f32({loaded}[lane] as u16), cuda_fp16_bits_to_f32(({loaded}[lane] >> 16) as u16))"
                    ))
                ));
                "U64"
            } else {
                self.emit_line(&format!(
                    "let {converted} = {};",
                    abi::per_lane(&format!(
                        "u32::from(cuda_f32_to_fp16_bits(float2_x({loaded}[lane]))) | (u32::from(cuda_f32_to_fp16_bits(float2_y({loaded}[lane]))) << 16)"
                    ))
                ));
                "U32"
            };
            self.emit_line(&format!(
                "{};",
                memory_call(
                    MEM_ST,
                    &format!("v2::mem::variant::St<v2::reg::variant::{store_type}, v2::Generic>"),
                    source_op_id,
                    &format!("({}, {converted})", generic_address(&destination_pointer)),
                )
            ));
        }
        Ok(())
    }

    fn emit_runtime_instr_desc(&mut self, args: &[ObjectRef], source_op_id: i64) -> AResult<()> {
        let descriptor = self.dps_pointer(&args[0], "runtime_instr_desc descriptor")?;
        let sf_id = self.emit_expr(&args[1])?;
        let sf_id = self.coerce_value(sf_id, "u32", "runtime_instr_desc_sf_id")?;
        let sf_id = self.as_warp_value(sf_id);
        let loaded = self.control_name("runtime_instr_desc_loaded");
        let patched = self.control_name("runtime_instr_desc_patched");
        self.emit_line(&format!(
            "let {loaded} = {};",
            memory_call(
                MEM_LD,
                "v2::mem::variant::Ld<v2::reg::variant::U32, v2::Generic>",
                source_op_id,
                &generic_address(&descriptor),
            )
        ));
        self.emit_line(&format!(
            "let {patched} = {};",
            abi::per_lane(&format!(
                "tcgen_runtime_instruction_descriptor({loaded}[lane], {}[lane])",
                sf_id.code
            ))
        ));
        self.emit_line(&format!(
            "{};",
            memory_call(
                MEM_ST,
                "v2::mem::variant::St<v2::reg::variant::U32, v2::Generic>",
                source_op_id,
                &format!("({}, {patched})", generic_address(&descriptor)),
            )
        ));
        Ok(())
    }

    /// The `Evaluate` statement form of a destination-passing helper.
    pub fn emit_dps_statement(
        &mut self,
        expr: &ObjectRef,
        op_name: &str,
        source_op_id: i64,
    ) -> AResult<()> {
        let args: Vec<ObjectRef> = expr
            .as_node::<tvm::ir::CallObj>()
            .map(|call| call.args.iter().map(crate::analyze::util::oref).collect())
            .unwrap_or_default();
        match op_name {
            "tirx.cuda.float22half2" => self.emit_float22half2(&args, source_op_id),
            "tirx.cuda.half8tofloat8" => {
                self.emit_packed_conversion(&args, source_op_id, "tirx.cuda.half8tofloat8", true)
            }
            "tirx.cuda.float8tohalf8" => {
                self.emit_packed_conversion(&args, source_op_id, "tirx.cuda.float8tohalf8", false)
            }
            "tirx.cuda.runtime_instr_desc" => self.emit_runtime_instr_desc(&args, source_op_id),
            other => not_covered(format!("statement call {other} has no lowering")),
        }
    }
}

pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let parts = parse_known_cuda_func_call(call.node)?;
    use KnownCudaFuncKind::*;
    match parts.kind {
        TensorMapReplaceGlobalAddress | TensorMapReplaceGlobalDim | TensorMapReplaceGlobalStride => {
            emitter.record_pointer_write(&parts.arguments[0], "global")?;
        }
        SmemDescMakeLoUniform | Relu2FmaF32x2 => {
            emitter.record_pointer_write(&parts.arguments[0], "")?;
        }
        CombineIntFracEx2 | FlashkdaFmafRn | FlashkdaRsqrtf | FlashkdaTanhApprox
        | TensorMapAcquire | TensorMapRelease | FmaScaleSubF32x2
        | GdnCpPrefillPredicatedGamma | GdnLg2ApproxFtz | OpaqueSmIdxU32 | OpaqueWarpId
        | Tcgen05MmaMxf4Block32Ss | ShlU32Clamp | MqaFp4WreluReduce64 | MqaFp8WreluReduce64
        | PrefetchL2 | SmemDescAdd16bOffset | StAsyncClusterTaskInfo => {}
    }
    emitter
        .with_call_expr(call.node, |emitter| {
            emitter.emit_cuda_helper(call.node, &parts)
        })
        .map(Some)
}

pub fn emit_dps(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    validate_dps_helper(call.node, &call.op_name)?;
    let destination = DPS_HELPERS.iter()
        .find(|(name, _, _)| *name == call.op_name)
        .expect("validated DPS helper").2;
    let node = call.node.as_node::<CallObj>().expect("validated call");
    emitter.record_pointer_write(&oref(node.args.get(destination)?), "")?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_dps_statement(call.node, &call.op_name, source_op_id)?;
    Ok(None)
}

pub fn requires_tensor_map_registry(nodes: &[ObjectRef]) -> AResult<bool> {
    for node in nodes {
        if crate::decode::call_name(node)?.as_deref() != Some("tirx.cuda.func_call") {
            continue;
        }
        match parse_known_cuda_func_call(node) {
            Ok(helper) if helper.kind.is_tensor_map() => return Ok(true),
            Ok(_) | Err(crate::analyze::util::Failure::Unsupported { .. }) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(false)
}
