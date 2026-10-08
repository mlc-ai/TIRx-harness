from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tests.numsim.support.kernels import (
    cta_copy_roundtrip,
    cta_ldgsts_copy_roundtrip,
    dsmem_copy_remote_cta,
    ldgsts_copy_roundtrip,
    owner_driven_dynamic_col_min_fallback,
    owner_driven_fp8_register_to_shared,
    owner_driven_nonexact_zero_row_min_fallback,
    owner_driven_nonexact_zero_scalar_index_fallback,
    owner_driven_scalar_load_fallback,
    owner_driven_shifted_alias_fallback,
    owner_driven_tile_pointwise,
    tile_copy_cast_mul,
    tile_directed_rounding,
    tile_reductions,
    tma_copy_cluster_multicast,
    tma_copy_nan_fill_boundary,
    tma_copy_roundtrip,
    tma_copy_transaction_mismatch,
    tma_copy_zero_fill_boundary,
    tma_scalar_hoist_fallbacks,
    tma_scalar_hoist_respects_boundary,
)
from tirx_harness.numsim.transpiler.frontend import analyze, verify
from tirx_harness.numsim.transpiler.artifact_template import emit_rust_module
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TileLayout

def _typed_tma_reduce_kernel(dtype: str, reduction: str):
    @T.prim_func
    def kernel(source: T.Buffer((4,), dtype), output: T.Buffer((4,), dtype)):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        shared = T.alloc_buffer((4,), dtype, scope="shared")
        if lane < 4:
            shared[lane] = source[lane]
        T.cuda.warp_sync()
        T.ptx.fence.proxy.async_.shared__cta()
        if lane == 0:
            Tx.copy_async(output[:], shared[:], dispatch="tma_auto", use_tma_reduce=reduction)
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group.read(0)

    return kernel


@T.prim_func
def tma_proxy_source_canonical(output: T.Buffer((4,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    if lane < 4:
        shared[lane] = T.cast(lane + 3, "float32")
    T.cuda.warp_sync()
    if lane == 0:
        Tx.copy_async(output[:], shared[:], dispatch="tma_auto")
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def tma_unknown_cache_hint(source: T.Buffer((4,), "float32"), output: T.Buffer((4,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
        Tx.copy_async(
            shared[:],
            source[:],
            dispatch="tma_auto",
            mbar=T.address_of(barrier[0]),
            cache_hint="unknown",
        )
    output[lane % 4] = shared[lane % 4]


@T.prim_func
def tma_reduce_wrong_direction(source: T.Buffer((4,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
        Tx.copy_async(
            shared[:],
            source[:],
            dispatch="tma_auto",
            mbar=T.address_of(barrier[0]),
            use_tma_reduce="add",
        )


@T.prim_func
def forced_smem_elementwise_on_local(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    lhs: T.f32[1]
    rhs: T.f32[1]
    result: T.f32[1]
    lhs[0] = T.float32(1)
    rhs[0] = T.float32(2)
    Tx.add(result, lhs, rhs, dispatch="smem")
    output[lane] = result[0]


def test_frontend_keeps_tile_calls_as_tirx_nodes_until_direct_lowering():
    spec = analyze(tile_copy_cast_mul).kernels[0]
    tile_sources = [entry for entry in spec.source_map if entry.kind == "TilePrimitiveCall"]

    assert tile_sources
    assert all(entry.resolved is None for entry in tile_sources)
    assert all(type(entry.node).__name__ == "TilePrimitiveCall" for entry in tile_sources)


@T.prim_func
def auto_packed_max_is_not_scalarized(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source: T.f32[8]
    result: T.f32[1]
    for index in T.serial(8):
        source[index] = T.cast(index, "float32")
    Tx.max(result, source)
    output[lane] = result[0]


@T.prim_func
def packed_f32_default_contract(
    source: T.Buffer((32, 18), "float32"), output: T.Buffer((32, 8), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    lhs: T.f32[2]
    rhs: T.f32[2]
    addend: T.f32[2]
    result: T.f32[2]

    for index in T.serial(2):
        lhs[index] = source[lane, index]
        rhs[index] = source[lane, 2 + index]
    Tx.add(result, lhs, rhs)
    output[lane, 0] = result[0]
    output[lane, 1] = result[1]

    for index in T.serial(2):
        lhs[index] = source[lane, 4 + index]
        rhs[index] = source[lane, 6 + index]
    Tx.sub(result, lhs, rhs)
    output[lane, 2] = result[0]
    output[lane, 3] = result[1]

    for index in T.serial(2):
        lhs[index] = source[lane, 8 + index]
        rhs[index] = source[lane, 10 + index]
    Tx.mul(result, lhs, rhs)
    output[lane, 4] = result[0]
    output[lane, 5] = result[1]

    for index in T.serial(2):
        lhs[index] = source[lane, 12 + index]
        rhs[index] = source[lane, 14 + index]
        addend[index] = source[lane, 16 + index]
    Tx.fma(result, lhs, rhs, addend)
    output[lane, 6] = result[0]
    output[lane, 7] = result[1]


@T.prim_func
def packed_f32_explicit_binary_rounding(
    source: T.Buffer((32, 8), "float32"), output: T.Buffer((32, 16), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    lhs: T.f32[2]
    rhs: T.f32[2]
    result: T.f32[2]

    for index in T.serial(2):
        lhs[index] = source[lane, index]
        rhs[index] = source[lane, 2 + index]
    Tx.add(result, lhs, rhs, rounding_mode="rn")
    output[lane, 0] = result[0]
    output[lane, 1] = result[1]
    Tx.add(result, lhs, rhs, rounding_mode="rm")
    output[lane, 2] = result[0]
    output[lane, 3] = result[1]
    Tx.add(result, lhs, rhs, rounding_mode="rp")
    output[lane, 4] = result[0]
    output[lane, 5] = result[1]
    Tx.add(result, lhs, rhs, rounding_mode="rz")
    output[lane, 6] = result[0]
    output[lane, 7] = result[1]

    lhs[0] = source[lane, 4]
    lhs[1] = -source[lane, 4]
    rhs[0] = source[lane, 5]
    rhs[1] = -source[lane, 5]
    Tx.sub(result, lhs, rhs, rounding_mode="rn")
    output[lane, 8] = result[0]
    output[lane, 9] = result[1]
    Tx.sub(result, lhs, rhs, rounding_mode="rz")
    output[lane, 10] = result[0]
    output[lane, 11] = result[1]

    lhs[0] = source[lane, 6]
    lhs[1] = -source[lane, 6]
    rhs[0] = source[lane, 7]
    rhs[1] = source[lane, 7]
    Tx.mul(result, lhs, rhs, rounding_mode="rn")
    output[lane, 12] = result[0]
    output[lane, 13] = result[1]
    Tx.mul(result, lhs, rhs, rounding_mode="rz")
    output[lane, 14] = result[0]
    output[lane, 15] = result[1]


@T.prim_func
def strided_f32_tile_uses_scalar_fallback(
    source: T.Buffer((32, 4), "float32"), output: T.Buffer((32, 2), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    lhs = T.alloc_buffer((2,), "float32", scope="local", layout=TileLayout(S[2:2]))
    rhs = T.alloc_buffer((2,), "float32", scope="local", layout=TileLayout(S[2:2]))
    result = T.alloc_buffer((2,), "float32", scope="local", layout=TileLayout(S[2:2]))
    lhs[0] = source[lane, 0]
    lhs[1] = source[lane, 1]
    rhs[0] = source[lane, 2]
    rhs[1] = source[lane, 3]
    Tx.add(result, lhs, rhs)
    output[lane, 0] = result[0]
    output[lane, 1] = result[1]


@T.prim_func
def typed_tma_static_single_bit_mask_is_unicast(
    source: T.Buffer((4,), "float32"), output: T.Buffer((2, 4), "float32")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane < 4:
        shared[lane] = T.float32(-7)
    if (cta == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        Tx.copy_async(
            shared[:],
            source[:],
            dispatch="tma_auto",
            mbar=T.address_of(barriers[0]),
            cta_group=2,
            cta_mask=T.int32(2),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cluster_sync()
    if lane < 4:
        output[cta, lane] = shared[lane]


@T.prim_func
def typed_tma_dynamic_single_bit_mask_is_multicast(
    source: T.Buffer((4,), "float32"), cta_mask: T.int32, output: T.Buffer((2, 4), "float32")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane < 4:
        shared[lane] = T.float32(-7)
    if (cta == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        Tx.copy_async(
            shared[:],
            source[:],
            dispatch="tma_auto",
            mbar=T.address_of(barriers[0]),
            cta_group=2,
            cta_mask=cta_mask,
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cluster_sync()
    if lane < 4:
        output[cta, lane] = shared[lane]


@T.prim_func
def typed_tma_multicast_cta_group1_completes_each_target_barrier(
    source: T.Buffer((4,), "float32"), output: T.Buffer((2, 4), "float32")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        Tx.copy_async(
            shared[:],
            source[:],
            dispatch="tma_auto",
            mbar=T.address_of(barriers[0]),
            cta_group=1,
            cta_mask=T.int32(3),
        )
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cluster_sync()
    if lane < 4:
        output[cta, lane] = shared[lane]


@T.prim_func
def typed_tma_multicast_cta_group2_completes_each_target_pair_barrier(
    source: T.Buffer((4,), "float32"), output: T.Buffer((4, 4), "float32")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([4])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta % 2 == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        Tx.copy_async(
            shared[:],
            source[:],
            dispatch="tma_auto",
            mbar=T.address_of(barriers[0]),
            cta_group=2,
            cta_mask=T.int32(15),
        )
    if (cta % 2 == 0) and (lane == 0):
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 32)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cluster_sync()
    if lane < 4:
        output[cta, lane] = shared[lane]


def _sequential_sum(values: np.ndarray, initial: np.float32 = np.float32(0)) -> np.float32:
    result = np.float32(initial)
    for value in values:
        result = np.float32(result + np.float32(value))
    return result


def test_copy_cast_and_mul_emit_direct_native_rust(tmp_path):
    source = np.linspace(-3, 3, 128, dtype=np.float32).reshape(32, 4).astype(np.float16)
    weight = np.linspace(0.25, 1.75, 128, dtype=np.float32).reshape(32, 4).astype(np.float16)
    output = np.zeros((32, 4), dtype=np.float16)

    module = numsim.transpile(tile_copy_cast_mul, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "weight": weight, "output": output})

    expected = (source.astype(np.float32) * weight.astype(np.float32) * 0.5).astype(np.float16)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_explicit_elementwise_rounding_is_layout_independent_and_not_ftz(tmp_path):
    output = np.zeros((32, 4), dtype=np.float32)

    module = numsim.transpile(tile_directed_rounding, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    one_up = np.asarray([0x3F80_0001], dtype=np.uint32).view(np.float32)[0]
    expected_row = np.asarray([1.0, 1.0, one_up, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(result.outputs["output"], np.tile(expected_row, (32, 1)))


def test_even_but_noncontiguous_f32_layout_uses_scalar_fallback(tmp_path):
    smallest_subnormal = np.asarray([1], dtype=np.uint32).view(np.float32)[0]
    smallest_normal = np.asarray([0x0080_0000], dtype=np.uint32).view(np.float32)[0]
    one_up = np.asarray([0x3F80_0001], dtype=np.uint32).view(np.float32)[0]
    normal_plus_subnormal = np.asarray([0x0080_0001], dtype=np.uint32).view(np.float32)[0]
    row = np.asarray(
        [1.0, smallest_normal, np.float32(3 * 2**-25), smallest_subnormal], dtype=np.float32
    )
    source = np.tile(row, (32, 1))
    output = np.zeros((32, 2), dtype=np.float32)

    module = numsim.transpile(strided_f32_tile_uses_scalar_fallback, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.asarray([one_up, normal_plus_subnormal], dtype=np.float32)
    np.testing.assert_array_equal(result.outputs["output"], np.tile(expected, (32, 1)))


def test_contiguous_f32_uses_canonical_rn_without_ftz(tmp_path):
    smallest_subnormal = np.asarray([1], dtype=np.uint32).view(np.float32)[0]
    smallest_normal = np.asarray([0x0080_0000], dtype=np.uint32).view(np.float32)[0]
    one_up = np.asarray([0x3F80_0001], dtype=np.uint32).view(np.float32)[0]
    mul_lhs = np.asarray([0x3FD0_0281], dtype=np.uint32).view(np.float32)[0]
    mul_rhs = np.asarray([0x3FE6_CB52], dtype=np.uint32).view(np.float32)[0]
    three_quarter_ulp = np.float32(3 * 2**-25)
    quarter_ulp = np.float32(2**-25)
    row = np.asarray(
        [
            1.0,
            smallest_normal,
            three_quarter_ulp,
            smallest_subnormal,
            one_up,
            smallest_normal,
            quarter_ulp,
            -smallest_subnormal,
            mul_lhs,
            smallest_normal,
            mul_rhs,
            0.5,
            1.0,
            smallest_subnormal,
            1.0,
            np.ldexp(np.float32(1), 126),
            three_quarter_ulp,
            0.0,
        ],
        dtype=np.float32,
    )
    source = np.tile(row, (32, 1))
    output = np.zeros((32, 8), dtype=np.float32)

    module = numsim.transpile(packed_f32_default_contract, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.asarray(
        [
            np.float32(row[0] + row[2]),
            np.float32(row[1] + row[3]),
            np.float32(row[4] - row[6]),
            np.float32(row[5] - row[7]),
            np.float32(row[8] * row[10]),
            np.float32(row[9] * row[11]),
            np.float32(np.float64(row[12]) * np.float64(row[14]) + np.float64(row[16])),
            np.float32(np.float64(row[13]) * np.float64(row[15]) + np.float64(row[17])),
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result.outputs["output"], np.tile(expected, (32, 1)))


def test_contiguous_binary_ops_honor_explicit_rounding_modes(tmp_path):
    half_ulp = np.float32(2**-24)
    quarter_ulp = np.float32(2**-25)
    one_up = np.asarray([0x3F80_0001], dtype=np.uint32).view(np.float32)[0]
    minus_one_down = np.asarray([0xBF80_0001], dtype=np.uint32).view(np.float32)[0]
    mul_lhs = np.asarray([0x3FD0_0281], dtype=np.uint32).view(np.float32)[0]
    mul_rhs = np.asarray([0x3FE6_CB52], dtype=np.uint32).view(np.float32)[0]
    mul_rn = np.asarray([0x403B_8775], dtype=np.uint32).view(np.float32)[0]
    mul_rz = np.asarray([0x403B_8774], dtype=np.uint32).view(np.float32)[0]
    row = np.asarray(
        [1.0, -1.0, half_ulp, -half_ulp, one_up, quarter_ulp, mul_lhs, mul_rhs], dtype=np.float32
    )
    source = np.tile(row, (32, 1))
    output = np.zeros((32, 16), dtype=np.float32)

    module = numsim.transpile(packed_f32_explicit_binary_rounding, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.asarray(
        [
            1.0,
            -1.0,
            1.0,
            minus_one_down,
            one_up,
            -1.0,
            1.0,
            -1.0,
            one_up,
            -one_up,
            1.0,
            -1.0,
            mul_rn,
            -mul_rn,
            mul_rz,
            -mul_rz,
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result.outputs["output"], np.tile(expected, (32, 1)))


def test_thread_owned_pointwise_uses_physical_layout_owners(tmp_path):
    source = np.linspace(-3, 3, 128 * 16, dtype=np.float32).reshape(128, 16).astype(np.float16)
    weight = np.linspace(0.25, 1.75, 128 * 16, dtype=np.float32).reshape(128, 16).astype(np.float16)
    scale = np.linspace(0.25, 1.25, 128, dtype=np.float32)
    output = np.zeros((128, 16), dtype=np.float16)

    module = numsim.transpile(owner_driven_tile_pointwise, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "weight": weight, "scale_input": scale, "output": output}
    )

    expected = (
        (source.astype(np.float32) * weight.astype(np.float32) + np.float32(0.25)) * scale[:, None]
    ).astype(np.float16)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_fp8_register_to_shared_owner_copy_runs_in_numsim(tmp_path):
    source = np.resize(
        np.array([0x00, 0x30, 0x38, 0x40, 0xB8, 0xC0, 0x48, 0x50], dtype=np.uint8),
        (128, 8),
    )
    module = numsim.transpile(owner_driven_fp8_register_to_shared, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source.copy(),
            "output": np.zeros_like(source),
        },
    )
    np.testing.assert_array_equal(result.outputs["output"], source)


def test_thread_owned_scalar_buffer_load_uses_canonical_rn(tmp_path):
    source = np.arange(128 * 16, dtype=np.float32).reshape(128, 16) / np.float32(32)
    scale = np.linspace(0.5, 1.5, 128, dtype=np.float32)
    output = np.zeros_like(source)

    module = numsim.transpile(owner_driven_scalar_load_fallback, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "scale": scale, "output": output})

    np.testing.assert_array_equal(
        result.outputs["output"], (source * scale[:, None]).astype(np.float32)
    )


def test_owner_driven_shifted_alias_falls_back(tmp_path):
    module = numsim.transpile(owner_driven_shifted_alias_fallback, cache_dir=tmp_path)
    output = np.zeros((128, 17), dtype=np.float32)
    result = numsim.Engine().run(module, {"output": output})

    rows = np.arange(128, dtype=np.float32)[:, None] * np.float32(100)
    columns = np.maximum(np.arange(17, dtype=np.float32) - np.float32(1), np.float32(0))
    np.testing.assert_array_equal(result.outputs["output"], rows + columns)


def test_owner_driven_nonexact_zero_row_min_falls_back(tmp_path):
    module = numsim.transpile(owner_driven_nonexact_zero_row_min_fallback, cache_dir=tmp_path)
    output = np.zeros((128, 16), dtype=np.float32)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(
        result.outputs["output"], np.arange(128 * 16, dtype=np.float32).reshape(128, 16)
    )


def test_owner_driven_dynamic_col_min_falls_back(tmp_path):
    module = numsim.transpile(owner_driven_dynamic_col_min_fallback, cache_dir=tmp_path)
    output = np.zeros((128, 16), dtype=np.float32)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(
        result.outputs["output"], np.arange(128 * 16, dtype=np.float32).reshape(128, 16)
    )


def test_owner_driven_nonexact_zero_scalar_index_falls_back(tmp_path):
    module = numsim.transpile(owner_driven_nonexact_zero_scalar_index_fallback, cache_dir=tmp_path)
    output = np.zeros((128, 16), dtype=np.float32)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(
        result.outputs["output"], np.arange(128 * 16, dtype=np.float32).reshape(128, 16) * 2
    )


def test_sum_max_min_and_accum_reduce_each_lane_local_tile(tmp_path):
    source = np.arange(32 * 8, dtype=np.float32).reshape(32, 8) / np.float32(16) - np.float32(5)
    output = np.zeros((32, 4), dtype=np.float32)

    module = numsim.transpile(tile_reductions, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.stack(
        [
            np.asarray([_sequential_sum(row) for row in source], dtype=np.float32),
            np.max(source, axis=1),
            np.min(source, axis=1),
            np.asarray([_sequential_sum(row, np.float32(10)) for row in source], dtype=np.float32),
        ],
        axis=1,
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_reduction_preserves_subnormal_inputs_in_lexicographic_order(tmp_path):
    source = np.zeros((32, 8), dtype=np.float32)
    source[:, 0] = np.float32(np.finfo(np.float32).tiny)
    source[:, 4] = np.asarray([0x807F_FFFF], dtype=np.uint32).view(np.float32)[0]
    output = np.zeros((32, 4), dtype=np.float32)

    result = numsim.Engine().run(
        numsim.transpile(tile_reductions, cache_dir=tmp_path), {"source": source, "output": output}
    )

    expected = np.asarray([_sequential_sum(row) for row in source], dtype=np.float32)
    np.testing.assert_array_equal(result.outputs["output"][:, 0], expected)
    assert np.all(expected.view(np.uint32) == np.uint32(0x0000_0001))


def test_lexicographic_sum_preserves_subnormal_result(tmp_path):
    source = np.zeros((32, 8), dtype=np.float32)
    source[:, 0] = np.asarray([0x0100_0000], dtype=np.uint32).view(np.float32)[0]
    source[:, 1] = np.asarray([0x80FF_FFFF], dtype=np.uint32).view(np.float32)[0]
    output = np.zeros((32, 4), dtype=np.float32)

    module = numsim.transpile(tile_reductions, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    actual_bits = result.outputs["output"][:, 0].view(np.uint32)
    assert np.all(actual_bits == np.uint32(0x0000_0001))


def test_forced_elementwise_dispatch_does_not_change_local_semantics(tmp_path):
    output = np.zeros((32,), dtype=np.float32)
    result = numsim.Engine().run(
        numsim.transpile(forced_smem_elementwise_on_local, cache_dir=tmp_path), {"output": output}
    )
    np.testing.assert_array_equal(result.outputs["output"], np.full(32, 3, dtype=np.float32))


def test_max_reduction_uses_canonical_lowering(tmp_path):
    output = np.zeros(32, dtype=np.float32)
    module = numsim.transpile(auto_packed_max_is_not_scalarized, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.full(32, 7, dtype=np.float32))


def test_tma_gmem_smem_roundtrip_delivers_physical_bytes_before_wait(tmp_path):
    source = (np.arange(32, dtype=np.float32).reshape(4, 8) - 11).astype(np.float16)
    output = np.zeros_like(source)

    module = numsim.transpile(tma_copy_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_tma_shared_source_copy_reads_current_shared_bytes(tmp_path):
    module = numsim.transpile(tma_proxy_source_canonical, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(4, dtype=np.float32)})

    np.testing.assert_array_equal(result.outputs["output"], np.arange(3, 7, dtype=np.float32))


@pytest.mark.parametrize(
    ("reduction", "dtype", "source", "initial", "expected"),
    [
        (
            "add",
            "float32",
            np.asarray([1.5, -2.0, 4.25, 3.0], dtype=np.float32),
            np.asarray([10.0, 20.0, -1.0, 8.0], dtype=np.float32),
            np.asarray([11.5, 18.0, 3.25, 11.0], dtype=np.float32),
        ),
        (
            "min",
            "float16",
            np.asarray([3.0, -10.0, 20.0, 7.0], dtype=np.float16),
            np.asarray([5.0, -5.0, 9.0, 7.0], dtype=np.float16),
            np.asarray([3.0, -10.0, 9.0, 7.0], dtype=np.float16),
        ),
        (
            "max",
            "int64",
            np.asarray([-3, 10, 20, 7], dtype=np.int64),
            np.asarray([5, -5, 9, 8], dtype=np.int64),
            np.asarray([5, 10, 20, 8], dtype=np.int64),
        ),
        (
            "inc",
            "uint32",
            np.asarray([4, 2, 4, 9], dtype=np.uint32),
            np.asarray([0, 2, 5, 8], dtype=np.uint32),
            np.asarray([1, 0, 0, 9], dtype=np.uint32),
        ),
        (
            "dec",
            "uint32",
            np.asarray([4, 2, 4, 9], dtype=np.uint32),
            np.asarray([0, 2, 5, 8], dtype=np.uint32),
            np.asarray([4, 1, 4, 7], dtype=np.uint32),
        ),
        (
            "and",
            "uint32",
            np.asarray([0x0F0F, 0xFF00, 0xAAAA, 0x1234], dtype=np.uint32),
            np.asarray([0xFFFF, 0x0FF0, 0x5555, 0xFFFF], dtype=np.uint32),
            np.asarray([0x0F0F, 0x0F00, 0x0000, 0x1234], dtype=np.uint32),
        ),
        (
            "or",
            "uint64",
            np.asarray([0x0F, 0xF0, 0xAA, 0x1234], dtype=np.uint64),
            np.asarray([0xF0, 0x0F, 0x55, 0xAB00], dtype=np.uint64),
            np.asarray([0xFF, 0xFF, 0xFF, 0xBB34], dtype=np.uint64),
        ),
        (
            "xor",
            "uint32",
            np.asarray([0x0F, 0xF0, 0xAA, 0x1234], dtype=np.uint32),
            np.asarray([0xF0, 0x0F, 0x55, 0xAB00], dtype=np.uint32),
            np.asarray([0xFF, 0xFF, 0xFF, 0xB934], dtype=np.uint32),
        ),
    ],
)
def test_typed_tma_reduce_reuses_raw_tensor_map_reduction_abi(
    tmp_path, reduction, dtype, source, initial, expected
):
    module = numsim.transpile(
        _typed_tma_reduce_kernel(dtype, reduction), cache_dir=tmp_path / f"{reduction}_{dtype}"
    )
    result = numsim.Engine().run(module, {"source": source, "output": initial.copy()})

    np.testing.assert_array_equal(result.outputs["output"], expected)


@pytest.mark.parametrize(
    ("reduction", "dtype", "message"),
    [
        ("add", "int64", "invalid for dtype"),
        ("add", "float64", "invalid for dtype"),
        ("min", "float32", "invalid for dtype"),
        ("inc", "int32", "invalid for dtype"),
        ("unknown", "uint32", "unsupported TMA reduction"),
    ],
)
def test_typed_tma_reduce_rejects_invalid_operation_dtype_pairs(
    tmp_path, reduction, dtype, message
):
    with pytest.raises(numsim.UnsupportedTIRxError, match=message):
        numsim.transpile(
            _typed_tma_reduce_kernel(dtype, reduction), cache_dir=tmp_path / f"{reduction}_{dtype}"
        )


def test_typed_tma_reduce_rejects_non_store_direction(tmp_path):
    with pytest.raises(numsim.UnsupportedTIRxError, match="only valid for shared->global"):
        numsim.transpile(tma_reduce_wrong_direction, cache_dir=tmp_path)


def test_typed_tma_rejects_unknown_cache_hint(tmp_path):
    with pytest.raises(numsim.UnsupportedTIRxError, match="unsupported TMA cache_hint"):
        numsim.transpile(tma_unknown_cache_hint, cache_dir=tmp_path)


def test_ldgsts_copy_keeps_cooperative_warp_lowering(tmp_path):
    source = np.linspace(-3, 4, 8, dtype=np.float32)
    output = np.zeros_like(source)

    module = numsim.transpile(ldgsts_copy_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_cta_copy_uses_canonical_semantics_independent_of_dispatch(tmp_path):
    source = np.linspace(-7, 9, 64, dtype=np.float32)
    output = np.zeros_like(source)

    module = numsim.transpile(cta_copy_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_cta_ldgsts_uses_ordinary_copy_semantics(tmp_path):
    source = np.linspace(-5, 11, 64, dtype=np.float32)
    output = np.zeros_like(source)

    module = numsim.transpile(cta_ldgsts_copy_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_tma_global_boundary_uses_zero_fill_and_counts_full_delivery(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4) + np.float32(0.25)
    output = np.full((4, 4), np.float32(-99))

    module = numsim.transpile(tma_copy_zero_fill_boundary, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.concatenate([np.zeros((1, 4), dtype=np.float32), source], axis=0)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_tma_global_boundary_uses_hardware_oob_nan_pattern(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4) + np.float32(0.25)
    output = np.zeros((4, 4), dtype=np.float32)

    module = numsim.transpile(tma_copy_nan_fill_boundary, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    actual_bits = result.outputs["output"].view(np.uint32)
    np.testing.assert_array_equal(actual_bits[0], np.full(4, 0x7FF7_7FF7, dtype=np.uint32))
    np.testing.assert_array_equal(result.outputs["output"][1:], source)


def test_tma_snapshots_global_dynamic_aliased_and_selected_issue_operands(tmp_path):
    source = np.linspace(-4, 3, 8, dtype=np.float32)
    global_start = np.zeros(1, dtype=np.int32)
    output = np.zeros((4, 8), dtype=np.float32)

    module = numsim.transpile(tma_scalar_hoist_fallbacks, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "global_start": global_start, "output": output}
    )

    np.testing.assert_array_equal(result.outputs["output"], np.stack([source] * 4))


def test_tma_scalar_hoist_does_not_cross_write_or_await_boundaries(tmp_path):
    source = np.linspace(-5, 2, 8, dtype=np.float32)
    output = np.zeros(16, dtype=np.float32)

    module = numsim.transpile(tma_scalar_hoist_respects_boundary, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.concatenate([source, source]))


def test_tma_multicast_writes_each_target_cta_and_credits_each_copy(tmp_path):
    source = np.linspace(-2, 3, 8, dtype=np.float32)
    output = np.zeros((2, 8), dtype=np.float32)

    module = numsim.transpile(tma_copy_cluster_multicast, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.stack([source, source]))
    assert result.stats["task_count"] == 2


def test_typed_tma_static_single_bit_mask_is_unicast_from_the_issuing_cta(tmp_path):
    source = np.arange(4, dtype=np.float32) + np.float32(2.5)
    output = np.zeros((2, 4), dtype=np.float32)

    module = numsim.transpile(typed_tma_static_single_bit_mask_is_unicast, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.stack((source, np.full(4, -7, dtype=np.float32)))
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_typed_tma_dynamic_single_bit_mask_retains_multicast_semantics(tmp_path):
    source = np.arange(4, dtype=np.float32) + np.float32(2.5)
    output = np.zeros((2, 4), dtype=np.float32)

    module = numsim.transpile(typed_tma_dynamic_single_bit_mask_is_multicast, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "cta_mask": 2, "output": output})

    expected = np.stack((np.full(4, -7, dtype=np.float32), source))
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_typed_tma_cta_group1_multicast_completes_each_target_barrier(tmp_path):
    source = np.arange(4, dtype=np.float32) - np.float32(1.5)
    output = np.zeros((2, 4), dtype=np.float32)

    module = numsim.transpile(
        typed_tma_multicast_cta_group1_completes_each_target_barrier, cache_dir=tmp_path
    )
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.broadcast_to(source, (2, 4)))


def test_typed_tma_cta_group2_multicast_completes_each_target_pair_barrier(tmp_path):
    source = np.arange(4, dtype=np.float32) - np.float32(1.5)
    output = np.zeros((4, 4), dtype=np.float32)

    module = numsim.transpile(
        typed_tma_multicast_cta_group2_completes_each_target_pair_barrier, cache_dir=tmp_path
    )
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.broadcast_to(source, (4, 4)))


def test_dsmem_copy_writes_the_named_remote_cta(tmp_path):
    output = np.zeros(8, dtype=np.float32)

    module = numsim.transpile(dsmem_copy_remote_cta, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.arange(5, 13, dtype=np.float32))


def test_tma_transaction_under_delivery_reports_deterministic_deadlock(tmp_path):
    source = np.arange(32, dtype=np.float16).reshape(4, 8)
    output = np.zeros_like(source)
    module = numsim.transpile(tma_copy_transaction_mismatch, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="transactions=64/68"):
        numsim.Engine().run(module, {"source": source, "output": output})


_VALID_TMA_REDUCTION_DTYPES = {
    "add": ("bfloat16", "float16", "float32", "int32", "uint32", "uint64"),
    "min": ("bfloat16", "float16", "int32", "int64", "uint32", "uint64"),
    "max": ("bfloat16", "float16", "int32", "int64", "uint32", "uint64"),
    "inc": ("uint32",),
    "dec": ("uint32",),
    "and": ("float32", "float64", "int32", "int64", "uint32", "uint64"),
    "or": ("float32", "float64", "int32", "int64", "uint32", "uint64"),
    "xor": ("float32", "float64", "int32", "int64", "uint32", "uint64"),
}


@pytest.mark.parametrize(
    ("reduction", "dtype"),
    [
        (reduction, dtype)
        for reduction, dtypes in _VALID_TMA_REDUCTION_DTYPES.items()
        for dtype in dtypes
    ],
)
def test_typed_tma_reduce_accepts_every_ptx_operation_dtype_pair(reduction, dtype):
    kernel = _typed_tma_reduce_kernel(dtype, reduction)
    spec = analyze(kernel)
    verify(spec)
    emit_rust_module(spec, kernel)
