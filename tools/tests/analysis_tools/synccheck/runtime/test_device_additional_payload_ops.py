"""Additional exact-op runtime evidence for native Synccheck payload calls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.tirx.layout import S, TCol, TileLayout, TLane

from tests.numsim.integration.test_direct_memory_artifact import (
    pass_emitted_cp_async_raw,
    raw_cp_async_zero_fill,
)
from tests.numsim.integration.test_packed_ptx_cvt_and_return import packed_ptx_cvt
from tests.numsim.microtests.cases.ptx_cvt_fp8 import packed_fp8_cvt_sm100_forms
from tests.numsim.microtests.cases.ptx_cvt_narrow import (
    narrow_cvt_bf16x2_forms,
    narrow_cvt_sm100_forms,
)
from tests.numsim.microtests.cases.ptx_cvt_scalar import scalar_cvt_narrowing
from tests.numsim.runtime.test_approximate_f32_contract import (
    approximate_f32_calls,
    bf16x2_exp2_calls,
)
from tests.numsim.runtime.test_dense_mma_forms import (
    ptx_mma_f16_accumulator_m16n8k8,
    ptx_mma_f64_m8n8k4,
    ptx_mma_s8_u8_m16n8k32_no_c,
)
from tests.numsim.runtime.test_packed_f16x2_arithmetic import (
    packed_f16x2_multiply,
    packed_f16x2_subtract,
    scalar_f16_add,
)
from tests.numsim.runtime.test_ptx_compare_semantics import (
    ptx_compare_semantics,
    ptx_slct_semantics,
)
from tests.numsim.runtime.test_ptx_new_register_semantics import ptx_new_register_semantics
from tests.numsim.runtime.test_ptx_register_bits import (
    ptx_latest_canonical_scalar_forms,
    ptx_register_bits,
)
from tests.numsim.runtime.test_ptx_warp_collectives import (
    raw_ptx_f32_reductions,
    raw_ptx_movmatrix_b16,
    raw_ptx_warp_collectives,
)
from tests.numsim.runtime.test_raw_tcgen_codegen import (
    raw_tcgen_ld_missing_shape_mappings,
    raw_tcgen_mma_tf32_ts_predicated,
)
from tests.numsim.runtime.test_scalar_control import (
    current_scalar_device_intrinsics,
    explicit_cvta_shared_cluster_u64,
    explicit_mapa_forms,
    packed_f16x2_conversion,
    scalar_warp_intrinsics,
)
from tests.numsim.runtime.test_scalar_helpers import mega_scalar_helpers
from tests.numsim.runtime.test_sparse_mma_forms import (
    sparse_float8_m16n8k64_zero,
    sparse_s8_u8_m16n8k64,
    sparse_tf32_m16n8k8,
)
from tests.numsim.support.kernels import tcgen_commit_runtime_multicast
from tests.analysis_tools.synccheck.runtime.test_device_payload_ops import _assert_payload_runtime
from tirx_harness import numsim

_TMEM_D_16 = TileLayout(S[(128, 16) : (1 @ TLane, 1 @ TCol)])


@T.prim_func
def additional_scalar_payload_ops(output: T.Buffer((32, 3), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value: T.let = T.cast(lane + 1, "float32")

    T.ptx.max.f32(output[lane, 0], value, T.float32(17))
    T.ptx.min.f32(output[lane, 1], value, T.float32(17))
    T.ptx.neg.f32(output[lane, 2], value)


@T.prim_func
def direct_lg2_payload(
    source: T.Buffer((32,), "float32"),
    output: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    result = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.lg2.approx.ftz.f32(result[0], source[lane]))
    output[lane] = result[0]


@T.prim_func
def direct_tensor_map_payload_ops(
    source_map: T.TensorMap(),
    replacement: T.Buffer((3, 4), "float32"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage: T.Buffer((128,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    descriptor = descriptor_storage.ptr_to([0])
    if lane == 0:
        payload = T.decl_buffer(
            (8,),
            "uint64",
            data=T.reinterpret("handle", T.address_of(source_map)),
            scope="param",
            align=16,
        )
        T.ptx.st.global_.v4.b64(descriptor, payload[0], payload[1], payload[2], payload[3])
        T.ptx.st.global_.v4.b64(
            T.reinterpret("handle", T.reinterpret("uint64", descriptor) + T.uint64(32)),
            payload[4],
            payload[5],
            payload[6],
            payload[7],
        )
        T.ptx.tensormap_replace.tile.global_address.global_.b1024.b64(
            descriptor, T.reinterpret("uint64", replacement.data)
        )
        T.ptx.tensormap_replace.tile.global_dim.global_.b1024.b32(descriptor, 0, T.uint32(4))
        T.ptx.tensormap_replace.tile.global_dim.global_.b1024.b32(descriptor, 1, T.uint32(3))
        T.ptx.tensormap_replace.tile.global_stride.global_.b1024.b64(descriptor, 0, T.uint64(16))
        T.ptx.fence.proxy.tensormap__generic.release.gpu()
        T.ptx.fence.proxy.tensormap__generic.acquire.gpu(descriptor)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](T.address_of(shared[0, 0]), descriptor, 0, 0, T.address_of(barriers[0]))
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 48)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        for row in T.serial(3):
            for column in T.serial(4):
                output[row, column] = shared[row, column]


@T.prim_func
def mma_f16c_f32d_zero(output: T.Buffer((32, 8), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    a_values = T.alloc_local((4,), "float16")
    b_values = T.alloc_local((4,), "float16")
    c_values = T.alloc_local((8,), "float16")
    d_values = T.alloc_local((8,), "float32")
    for index in T.unroll(4):
        a_values[index] = T.float16(0)
        b_values[index] = T.float16(0)
    for index in T.unroll(8):
        c_values[index] = T.float16(0)
    a_words = a_values.view("uint32")
    b_words = b_values.view("uint32")
    c_words = c_values.view("uint32")

    T.ptx.mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f16(
        *[d_values[index] for index in range(8)],
        a_words[0],
        a_words[1],
        b_words[0],
        b_words[1],
        *[c_words[index] for index in range(4)],
    )
    for index in T.unroll(8):
        output[lane, index] = d_values[index]


@T.prim_func
def additional_sparse_mma_payload_ops(
    output_f16: T.Buffer((2, 32, 4), "float16"),
    output_i32: T.Buffer((32, 4), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    a_f16 = T.alloc_local((8,), "float16")
    b_f16 = T.alloc_local((8,), "float16")
    acc_f16 = T.alloc_local((4,), "float16")
    a_i8 = T.alloc_local((8,), "int8")
    b_u8 = T.alloc_local((8,), "uint8")
    acc_i32 = T.alloc_local((4,), "int32")
    metadata = T.alloc_local((1,), "uint32")
    for index in T.unroll(8):
        a_f16[index] = T.float16(0)
        b_f16[index] = T.float16(0)
        a_i8[index] = T.int8(0)
        b_u8[index] = T.uint8(0)
    for index in T.unroll(4):
        acc_f16[index] = T.float16(0)
        acc_i32[index] = T.int32(0)
    a_f16_words = a_f16.view("uint32")
    b_f16_words = b_f16.view("uint32")
    acc_f16_words = acc_f16.view("uint32")
    a_i8_words = a_i8.view("uint32")
    b_u8_words = b_u8.view("uint32")
    acc_i32_words = acc_i32.view("uint32")
    metadata[0] = T.uint32(0x44444444)

    T.ptx.mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16(
        acc_f16_words[0],
        acc_f16_words[1],
        a_f16_words[0],
        a_f16_words[1],
        b_f16_words[0],
        b_f16_words[1],
        acc_f16_words[0],
        acc_f16_words[1],
        metadata[0],
        0,
    )
    for index in T.unroll(4):
        output_f16[0, lane, index] = acc_f16[index]

    T.ptx.mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16(
        acc_f16_words[0],
        acc_f16_words[1],
        *[a_f16_words[index] for index in range(4)],
        *[b_f16_words[index] for index in range(4)],
        acc_f16_words[0],
        acc_f16_words[1],
        metadata[0],
        0,
    )
    for index in T.unroll(4):
        output_f16[1, lane, index] = acc_f16[index]

    T.ptx.mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32(
        *[acc_i32_words[index] for index in range(4)],
        a_i8_words[0],
        a_i8_words[1],
        b_u8_words[0],
        b_u8_words[1],
        *[acc_i32_words[index] for index in range(4)],
        metadata[0],
        0,
    )
    for index in T.unroll(4):
        output_i32[lane, index] = acc_i32[index]


@T.prim_func
def tcgen05_st_split_roundtrip(
    source: T.Buffer((4, 32, 2), "uint32"),
    output: T.Buffer((128, 16), "uint32"),
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    physical_row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer(
        (128, 16),
        "uint32",
        scope="tmem",
        layout=_TMEM_D_16,
        allocated_addr=0,
    )
    registers = T.alloc_local((2,), "uint32")
    for column in T.unroll(16):
        tmem[physical_row, column] = T.uint32(0)
    registers[0] = source[warp, lane, 0]
    registers[1] = source[warp, lane, 1]
    T.ptx["tcgen05.st.sync.aligned.16x32bx2.x2.b32"](
        T.uint32(0),
        2,
        registers[0],
        registers[1],
    )
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.cuda.cta_sync()
    for column in T.unroll(16):
        output[physical_row, column] = tmem[physical_row, column]


_KERNELS = (
    pass_emitted_cp_async_raw,
    raw_cp_async_zero_fill,
    current_scalar_device_intrinsics,
    packed_f16x2_conversion,
    packed_ptx_cvt,
    packed_fp8_cvt_sm100_forms,
    narrow_cvt_bf16x2_forms,
    narrow_cvt_sm100_forms,
    scalar_cvt_narrowing,
    scalar_f16_add,
    packed_f16x2_multiply,
    packed_f16x2_subtract,
    ptx_register_bits,
    ptx_new_register_semantics,
    ptx_compare_semantics,
    ptx_slct_semantics,
    ptx_latest_canonical_scalar_forms,
    raw_ptx_movmatrix_b16,
    raw_ptx_warp_collectives,
    raw_ptx_f32_reductions,
    approximate_f32_calls,
    bf16x2_exp2_calls,
    direct_lg2_payload,
    mega_scalar_helpers,
    explicit_mapa_forms,
    explicit_cvta_shared_cluster_u64,
    scalar_warp_intrinsics,
    additional_scalar_payload_ops,
    ptx_mma_f16_accumulator_m16n8k8,
    mma_f16c_f32d_zero,
    ptx_mma_f64_m8n8k4,
    ptx_mma_s8_u8_m16n8k32_no_c,
    sparse_tf32_m16n8k8,
    sparse_float8_m16n8k64_zero,
    sparse_s8_u8_m16n8k64,
    additional_sparse_mma_payload_ops,
    tcgen_commit_runtime_multicast,
    raw_tcgen_ld_missing_shape_mappings,
    raw_tcgen_mma_tf32_ts_predicated,
    tcgen05_st_split_roundtrip,
    direct_tensor_map_payload_ops,
)


def _zero_argument(parameter: Any) -> Any:
    if parameter.is_scalar():
        return 0
    return np.zeros(
        tuple(int(extent) for extent in parameter.shape),
        dtype=np.dtype(str(parameter.dtype)),
    )


def _kernel_name(kernel: Any) -> str:
    """Return the global name for decorated and directly constructed PrimFuncs."""
    return getattr(kernel, "__name__", None) or str(kernel.attrs["global_symbol"])


def _arguments_by_kernel() -> dict[str, dict[str, Any]]:
    arguments = {}
    for kernel in _KERNELS:
        kernel_name = _kernel_name(kernel)
        if kernel_name == "direct_tensor_map_payload_ops":
            source = np.zeros((3, 4), dtype=np.float32)
            arguments[kernel_name] = {
                "source_map": numsim.TensorMap(
                    base=source,
                    global_shape=(4, 3),
                    global_strides=(16,),
                    box_shape=(4, 3),
                    element_strides=(1, 1),
                ).numpy(),
                "replacement": np.zeros((3, 4), dtype=np.float32),
                "output": np.zeros((3, 4), dtype=np.float32),
                "descriptor_storage": np.zeros(128, dtype=np.uint8),
            }
        else:
            arguments[kernel_name] = {
                parameter.name: _zero_argument(parameter) for parameter in kernel.params
            }
    for kernel_name in (
        "sparse_tf32_m16n8k8",
        "sparse_s8_u8_m16n8k64",
    ):
        arguments[kernel_name]["metadata_words"] = np.full(
            32,
            np.uint32(0x44444444),
            dtype=np.uint32,
        )
    return arguments


@pytest.fixture(scope="module")
def native_additional_payload_artifact(tmp_path_factory):
    return numsim.transpile(
        _KERNELS, cache_dir=tmp_path_factory.mktemp("synccheck-payload"), _analysis_capable=True
    )


@pytest.mark.parametrize("kernel_name", [_kernel_name(kernel) for kernel in _KERNELS])
def test_payload_runtime(native_additional_payload_artifact, kernel_name):
    _assert_payload_runtime(native_additional_payload_artifact, _arguments_by_kernel(), kernel_name)
