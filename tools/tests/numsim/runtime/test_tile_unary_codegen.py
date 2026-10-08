from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TileLayout


@T.prim_func
def tile_unary_local(source: T.Buffer((32,), "float32"), output: T.Buffer((8, 32), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value: T.f32[1]
    bias: T.f32[1]
    ignored: T.f32[1]
    result: T.f32[1]

    value[0] = source[lane]
    bias[0] = source[lane] * T.float32(0.125)

    Tx.zero(result, ignored, dispatch="reg")
    output[0, lane] = result[0]
    Tx.sqrt(result, value, T.float32(0.25), T.float32(2.0), dispatch="reg")
    output[1, lane] = result[0]
    Tx.fill(result, T.float32(1.75), dispatch="reg")
    output[2, lane] = result[0]
    Tx.reciprocal(result, value, dispatch="reg")
    output[3, lane] = result[0]
    Tx.silu(result, value, dispatch="reg")
    output[4, lane] = result[0]
    Tx.exp(result, value, bias, T.float32(0.5), dispatch="reg")
    output[5, lane] = result[0]
    Tx.exp2(result, value, dispatch="reg")
    output[6, lane] = result[0]
    Tx.log2(result, value, dispatch="reg")
    output[7, lane] = result[0]


@T.prim_func
def tile_unary_shared_cta(source: T.Buffer((64,), "float32"), output: T.Buffer((8, 64), "float32")):
    T.device_entry()
    _cta = T.cta_id([1])
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    thread = warp * 32 + lane
    layout = TileLayout(S[64])
    value = T.alloc_buffer((64,), "float32", scope="shared", layout=layout)
    bias = T.alloc_buffer((64,), "float32", scope="shared", layout=layout)
    ignored = T.alloc_buffer((64,), "float32", scope="shared", layout=layout)
    result = T.alloc_buffer((64,), "float32", scope="shared", layout=layout)

    value[thread] = source[thread]
    bias[thread] = source[thread] * T.float32(0.125)
    T.cuda.cta_sync()

    Tx.cta.zero(result, ignored, dispatch="smem")
    output[0, thread] = result[thread]
    T.cuda.cta_sync()
    Tx.cta.sqrt(result, value, T.float32(0.25), T.float32(2.0), dispatch="smem")
    output[1, thread] = result[thread]
    T.cuda.cta_sync()
    Tx.cta.fill(result, T.float32(1.75), dispatch="smem")
    output[2, thread] = result[thread]
    T.cuda.cta_sync()
    Tx.cta.reciprocal(result, value, dispatch="smem")
    output[3, thread] = result[thread]
    T.cuda.cta_sync()
    Tx.cta.silu(result, value, dispatch="smem")
    output[4, thread] = result[thread]
    T.cuda.cta_sync()
    Tx.cta.exp(result, value, bias, T.float32(0.5), dispatch="smem")
    output[5, thread] = result[thread]
    T.cuda.cta_sync()
    Tx.cta.exp2(result, value, dispatch="smem")
    output[6, thread] = result[thread]
    T.cuda.cta_sync()
    Tx.cta.log2(result, value, dispatch="smem")
    output[7, thread] = result[thread]


@T.prim_func
def tile_in_place_shared_cta(
    source: T.Buffer((64,), "float32"), output: T.Buffer((2, 64), "float32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    thread = warp * 32 + lane
    layout = TileLayout(S[64])
    value = T.alloc_buffer((64,), "float32", scope="shared", layout=layout)
    addend = T.alloc_buffer((64,), "float32", scope="shared", layout=layout)
    value[thread] = source[thread]
    addend[thread] = source[thread] * T.float32(0.25)
    T.cuda.cta_sync()

    Tx.cta.add(value, value, addend, dispatch="smem")
    output[0, thread] = value[thread]
    T.cuda.cta_sync()
    Tx.cta.exp(value, value, dispatch="smem")
    output[1, thread] = value[thread]


@T.prim_func
def tile_warp_shared_varying_scalar(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value = T.alloc_buffer((32,), "float32", scope="shared", layout=TileLayout(S[32]))
    value[lane] = T.float32(1)
    T.cuda.warp_sync()

    Tx.warp.add(value, value, T.cast(lane, "float32"), dispatch="smem")
    output[lane] = value[lane]


@T.prim_func
def tile_warpgroup_in_place_shared(
    source: T.Buffer((128,), "float32"), output: T.Buffer((128,), "float32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    thread = T.thread_id_in_wg([128])
    value = T.alloc_buffer((128,), "float32", scope="shared", layout=TileLayout(S[128]))
    value[thread] = source[thread]
    T.cuda.cta_sync()

    Tx.wg.add(value, value, T.float32(4), dispatch="smem")
    output[thread] = value[thread]


@T.prim_func
def tile_shared_cta_cast(source: T.Buffer((256,), "float32"), output: T.Buffer((256,), "float16")):
    T.device_entry()
    _cta = T.cta_id([1])
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    thread = warp * 32 + lane
    source_shared = T.alloc_buffer((256,), "float32", scope="shared", layout=TileLayout(S[256]))
    result_shared = T.alloc_buffer((256,), "float16", scope="shared", layout=TileLayout(S[256]))
    for index in T.serial(4):
        linear = thread + index * 64
        source_shared[linear] = source[linear]
    T.cuda.cta_sync()

    Tx.cta.cast(result_shared, source_shared, dispatch="smem")
    for index in T.serial(4):
        linear = thread + index * 64
        output[linear] = result_shared[linear]


@T.prim_func
def tile_warpgroup_shared_vector_owners(
    source: T.Buffer((512,), "float32"), output: T.Buffer((2, 512), "float32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    thread = T.thread_id_in_wg([128])
    packed = T.alloc_buffer((512,), "float32", scope="shared", layout=TileLayout(S[512]))
    scalar = T.alloc_buffer((512,), "float32", scope="shared", layout=TileLayout(S[512]))
    for index in T.serial(4):
        linear = thread + index * 128
        packed[linear] = source[linear]
        scalar[linear] = source[linear]
    T.cuda.cta_sync()

    Tx.wg.add(packed, packed, T.cast(thread, "float32"), dispatch="smem")
    Tx.wg.fdiv(
        scalar,
        scalar,
        T.cast(thread + 1, "float32"),
        dispatch="smem",
    )
    for index in T.serial(4):
        linear = thread + index * 128
        output[0, linear] = packed[linear]
        output[1, linear] = scalar[linear]


@T.prim_func
def tile_unary_unknown_config(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value: T.f32[1]
    result: T.f32[1]
    value[0] = T.float32(1)
    Tx.exp(result, value, undocumented_mode=True)
    output[lane] = result[0]


def _expected(source: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    bias = (source * np.float32(0.125)).astype(np.float32)
    scaled = (source * np.float32(2.0)).astype(np.float32)
    sqrt_input = (scaled + np.float32(0.25)).astype(np.float32)
    exp_input = (source * np.float32(0.5)).astype(np.float32)
    exp_input = (exp_input + bias).astype(np.float32)
    return np.stack(
        [
            np.zeros_like(source),
            np.sqrt(sqrt_input).astype(np.float32),
            np.full_like(source, np.float32(1.75)),
            (np.float32(1.0) / source).astype(np.float32),
            (source / (np.float32(1.0) + np.exp(-source))).astype(np.float32),
            np.exp(exp_input).astype(np.float32),
            np.exp2(source).astype(np.float32),
            np.log2(source).astype(np.float32),
        ]
    )


def test_unary_tile_ops_run_through_local_reg_lowering(tmp_path):
    source = np.linspace(0.125, 1.5, 32, dtype=np.float32)
    output = np.zeros((8, 32), dtype=np.float32)

    module = numsim.transpile(tile_unary_local, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_allclose(result.outputs["output"], _expected(source), rtol=2e-06, atol=2e-06)


def test_unary_tile_ops_run_through_two_warp_shared_cta_lowering(tmp_path):
    source = np.linspace(0.125, 1.5, 64, dtype=np.float32)
    output = np.zeros((8, 64), dtype=np.float32)

    module = numsim.transpile(tile_unary_shared_cta, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_allclose(result.outputs["output"], _expected(source), rtol=2e-6, atol=2e-6)


def test_shared_cta_in_place_ops_have_single_writer_per_element(tmp_path):
    source = np.linspace(-0.5, 0.5, 64, dtype=np.float32)
    output = np.zeros((2, 64), dtype=np.float32)

    module = numsim.transpile(tile_in_place_shared_cta, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    added = (source + source * np.float32(0.25)).astype(np.float32)
    expected = np.stack([added, np.exp(added).astype(np.float32)])
    np.testing.assert_allclose(result.outputs["output"], expected, rtol=2e-6, atol=2e-6)


def test_warp_shared_element_owner_selects_lane_varying_scalar(tmp_path):
    output = np.zeros((32,), dtype=np.float32)

    module = numsim.transpile(tile_warp_shared_varying_scalar, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    expected = np.arange(32, dtype=np.float32) + np.float32(1)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_warpgroup_shared_in_place_op_writes_once_across_four_warps(tmp_path):
    source = np.linspace(-3.0, 3.0, 128, dtype=np.float32)
    output = np.zeros((128,), dtype=np.float32)

    module = numsim.transpile(tile_warpgroup_in_place_shared, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_allclose(
        result.outputs["output"], source + np.float32(4), rtol=1e-7, atol=1e-7
    )


def test_shared_cta_cast_uses_shared_ownership_and_completion(tmp_path):
    source = np.linspace(-3.0, 3.0, 256, dtype=np.float32)
    output = np.zeros((256,), dtype=np.float16)

    module = numsim.transpile(tile_shared_cta_cast, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source.astype(np.float16))


def test_warpgroup_shared_owner_is_independent_of_vector_chunk(tmp_path):
    source = np.linspace(1.0, 2.0, 512, dtype=np.float32)
    output = np.zeros((2, 512), dtype=np.float32)

    module = numsim.transpile(tile_warpgroup_shared_vector_owners, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    indices = np.arange(512, dtype=np.int64)
    owner = (indices % 128).astype(np.float32)
    expected = np.stack([source + owner, source / (owner + np.float32(1))])
    np.testing.assert_allclose(result.outputs["output"], expected, rtol=2e-6, atol=2e-6)


def test_unary_tile_ops_fail_closed_on_unknown_config(tmp_path):
    with pytest.raises(UnsupportedTIRxError, match="unsupported config keys"):
        numsim.transpile(tile_unary_unknown_config, cache_dir=tmp_path)


@T.prim_func
def tile_fill_untyped_literals(output: T.Buffer((3, 32), "float32")):
    """TIRx leaves unannotated Python literals as raw int/float in the call args."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value: T.f32[1]
    counter = T.alloc_local((1,), "int32")

    Tx.fill(value, 0, dispatch="reg")
    output[0, lane] = value[0]
    Tx.fill(value, 1.5, dispatch="reg")
    output[1, lane] = value[0]
    Tx.fill(counter, 3, dispatch="reg")
    output[2, lane] = T.cast(counter[0], "float32")


def test_fill_accepts_untyped_python_literals(tmp_path):
    output = np.zeros((3, 32), dtype=np.float32)

    module = numsim.transpile(tile_fill_untyped_literals, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    expected = np.stack(
        [
            np.zeros(32, dtype=np.float32),
            np.full(32, 1.5, dtype=np.float32),
            np.full(32, 3.0, dtype=np.float32),
        ]
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


@T.prim_func
def tile_fill_wide_integer_literal(output: T.Buffer((32,), "uint32")):
    """A mask literal does not fit int32; the frontend must not reject it."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    mask = T.alloc_local((1,), "uint32")

    Tx.fill(mask, 0xFFFFFFFF, dispatch="reg")
    output[lane] = mask[0]


def test_fill_accepts_an_integer_literal_wider_than_int32(tmp_path):
    output = np.zeros(32, dtype=np.uint32)

    module = numsim.transpile(tile_fill_wide_integer_literal, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(
        result.outputs["output"], np.full(32, 0xFFFFFFFF, dtype=np.uint32)
    )


_E4M3_FILL_LAYOUT = TileLayout(S[16])


@T.prim_func
def tile_fill_e4m3(output: T.Buffer((32, 16), "float32")):
    T.device_entry()
    thread = T.thread_id([32])
    values = T.alloc_local((16,), "float8_e4m3fn", layout=_E4M3_FILL_LAYOUT)
    Tx.fill(values, T.cast(T.float32(1.25), "float8_e4m3fn"), dispatch="reg")
    for index in T.unroll(16):
        output[thread, index] = T.cast(values[index], "float32")


def test_fill_accepts_e4m3_register_tile(tmp_path):
    module = numsim.transpile(tile_fill_e4m3, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros((32, 16), dtype=np.float32)})

    np.testing.assert_array_equal(
        result.outputs["output"], np.full((32, 16), 1.25, dtype=np.float32)
    )
