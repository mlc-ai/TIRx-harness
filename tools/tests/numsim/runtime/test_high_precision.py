from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TileLayout

from tirx_harness import numsim
from tirx_harness.numsim.errors import NumSimExecutionError, UnsupportedTIRxError


@T.prim_func
def cancellation(
    source: T.Buffer((32, 3), "float32"),
    scratch: T.Buffer((32,), "float32"),
    output: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    scratch[lane] = source[lane, 0] + source[lane, 1]
    output[lane] = scratch[lane] + source[lane, 2]


@T.prim_func
def half_spill(
    source: T.Buffer((32,), "float32"),
    scratch: T.Buffer((32,), "float16"),
    output: T.Buffer((32,), "float16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    scratch[lane] = T.cast(source[lane], "float16")
    output[lane] = scratch[lane] - T.float16(1)


def test_high_precision_retains_cancellation_residual_across_global_store(tmp_path):
    source = np.tile(np.array([2**24, 1, -(2**24)], dtype=np.float32), (32, 1))
    results = {}
    modules = {}
    for precision in ("native", "high"):
        module = numsim.transpile(cancellation, precision=precision, cache_dir=tmp_path)
        result = numsim.Engine().run(
            module,
            {
                "source": source,
                "scratch": np.zeros(32, np.float32),
                "output": np.zeros(32, np.float32),
            },
            outputs=("output",),
        )
        results[precision] = result.outputs["output"]
        modules[precision] = module
    assert modules["native"].cache_key != modules["high"].cache_key
    np.testing.assert_array_equal(results["native"], 0)
    np.testing.assert_array_equal(results["high"], source.astype(np.float64).sum(axis=1))
    assert results["high"].dtype == np.float64


def test_high_precision_preserves_half_addresses_and_unrounded_outputs(tmp_path):
    source = 1 + np.arange(1, 33, dtype=np.float32) / 65536
    scratch = np.zeros(32, np.float16)
    output = np.zeros(32, np.float16)
    module = numsim.transpile(half_spill, precision="high", cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "scratch": scratch, "output": output})
    np.testing.assert_array_equal(result.outputs["output"], source.astype(np.float64) - 1)
    assert scratch.nbytes == output.nbytes == 64


@T.prim_func
def scalar_inputs(
    half: T.float16,
    brain: T.bfloat16,
    single: T.float32,
    double: T.float64,
    output: T.Buffer((5, 32), "float64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[0, lane] = T.cast(half + T.float16(0.1), "float64")
    output[1, lane] = T.cast(brain + T.bfloat16(0.1), "float64")
    output[2, lane] = T.cast(single + T.float32(0.1), "float64")
    output[3, lane] = double + T.float64(0.1)
    output[4, lane] = T.cast(lane < 16, "float32")


def test_high_precision_preserves_scalar_and_literal_quantization(tmp_path):
    import ml_dtypes

    value = 1.0012
    module = numsim.transpile(scalar_inputs, precision="high", cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "half": value,
            "brain": value,
            "single": value,
            "double": value,
            "output": np.zeros((5, 32), np.float64),
        },
    )
    expected = np.empty((5, 32), np.float64)
    for row, dtype in enumerate((np.float16, ml_dtypes.bfloat16, np.float32, np.float64)):
        expected[row] = float(dtype(value)) + float(dtype(0.1))
    expected[4] = np.arange(32) < 16
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_precision_configuration_rejects_unknown_modes():
    with pytest.raises(ValueError, match="precision"):
        numsim.transpile(cancellation, precision="fp128")
    with pytest.raises(ValueError, match="checkers"):
        numsim.transpile(cancellation, precision="high", _analysis_capable=True)


@T.prim_func
def partial_byte_read(
    source: T.Buffer((32,), "float32"),
    halves: T.Buffer((32,), "float16"),
    bytes_view: T.Buffer((64,), "uint8"),
    output: T.Buffer((32,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    halves[lane] = T.cast(source[lane], "float16")
    output[lane] = bytes_view[lane * 2]


@T.prim_func
def bit_reinterpret(source: T.Buffer((32,), "float32"), output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.reinterpret("uint32", source[lane])


def test_high_precision_rejects_partial_bytes_after_float_store(tmp_path):
    halves = np.zeros(32, np.float16)
    module = numsim.transpile(partial_byte_read, precision="high", cache_dir=tmp_path)
    with pytest.raises(NumSimExecutionError, match="partial or differently typed"):
        numsim.Engine().run(
            module,
            {
                "source": np.full(32, 1.0001, np.float32),
                "halves": halves,
                "bytes_view": halves.view(np.uint8),
                "output": np.zeros(32, np.uint8),
            },
        )


def test_high_precision_rejects_float_bit_reinterpretation(tmp_path):
    with pytest.raises(UnsupportedTIRxError, match="high precision.*reinterpret"):
        numsim.transpile(bit_reinterpret, precision="high", cache_dir=tmp_path)


@pytest.mark.parametrize("operation", ["raw_math", "async_copy", "atomic", "tcgen"])
def test_high_precision_rejects_unported_native_operations(operation, tmp_path):
    from tests.numsim.runtime.test_approximate_f32_contract import approximate_f32_calls
    from tests.numsim.runtime.test_memory_ops import cp_async_plain_4, cuda_atomic_add_float32
    from tests.numsim.support.kernels import dense_gemm_async_cta1

    kernels = {
        "raw_math": approximate_f32_calls,
        "async_copy": cp_async_plain_4,
        "atomic": cuda_atomic_add_float32,
        "tcgen": dense_gemm_async_cta1,
    }
    with pytest.raises(UnsupportedTIRxError, match="high precision"):
        numsim.transpile(kernels[operation], precision="high", cache_dir=tmp_path)


@T.prim_func
def shared_copy_spill(
    source: T.Buffer((32,), "float32"),
    replacement: T.Buffer((32,), "float16"),
    output: T.Buffer((2, 32), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    first = T.alloc_buffer((32,), "float16", scope="shared", layout=TileLayout(S[32]))
    second = T.alloc_buffer((32,), "float16", scope="shared", layout=TileLayout(S[32]))
    first[lane] = T.cast(source[lane], "float16")
    T.cuda.warp_sync()
    Tx.warp.copy(second[:], first[:])
    output[0, lane] = T.cast(second[lane], "float32")
    Tx.warp.copy(second[:], replacement[:])
    output[1, lane] = T.cast(second[lane], "float32")


def test_tile_copy_snapshots_shadow_values_and_clears_overwritten_values(tmp_path):
    source = 1 + np.arange(32, dtype=np.float32) / 65536
    replacement = np.linspace(-2, 2, 32, dtype=np.float16)
    module = numsim.transpile(shared_copy_spill, precision="high", cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "replacement": replacement,
            "output": np.zeros((2, 32), np.float32),
        },
    )
    np.testing.assert_array_equal(
        result.outputs["output"], np.stack([source, replacement]).astype(np.float64)
    )


def test_high_precision_tile_reductions_and_cross_warp_transport(tmp_path):
    from tests.numsim.runtime.test_tile_owner_transport import (
        _cast_unary_binary_cross_warp_owner_remap,
    )
    from tests.numsim.runtime.test_tile_reduction_variants import shared_cta_f16_reductions

    source = np.tile(np.array([2048, 1, -2048, 0.5], np.float16), (64, 1))
    module = numsim.transpile(shared_cta_f16_reductions, precision="high", cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "output": np.zeros((3, 64), np.float16)}
    )
    expected = np.stack(
        [source.astype(np.float64).sum(axis=1), source.max(axis=1), source.min(axis=1)]
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)

    source = 1 + np.arange(256, dtype=np.float32).reshape(128, 2) / 65536
    module = numsim.transpile(
        _cast_unary_binary_cross_warp_owner_remap, precision="high", cache_dir=tmp_path
    )
    result = numsim.Engine().run(module, {"source": source, "output": np.zeros_like(source)})
    np.testing.assert_allclose(
        result.outputs["output"], np.sqrt(source.astype(np.float64)) + source, rtol=1e-14
    )


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e8m0fnu", "bfloat16"])
def test_high_precision_low_width_outputs_keep_unrounded_values(dtype, tmp_path):
    @T.prim_func
    def low_width_output(source: T.Buffer((32,), "float32"), output: T.Buffer((32,), dtype)):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        output[lane] = T.cast(source[lane], dtype)

    source = 1 + np.arange(1, 33, dtype=np.float32) / 65536
    storage_dtype = np.uint16 if dtype == "bfloat16" else np.uint8
    output = np.zeros(32, storage_dtype)
    module = numsim.transpile(low_width_output, precision="high", cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})
    np.testing.assert_array_equal(result.outputs["output"], source.astype(np.float64))
    assert output.nbytes == 32 * np.dtype(storage_dtype).itemsize


@T.prim_func
def strided_half_spill(
    source: T.Buffer((32,), "float32"),
    scratch: T.Buffer((96,), "float16"),
    output: T.Buffer((96,), "float16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    scratch[lane * 3 + 1] = T.cast(source[lane], "float16")
    output[lane * 3 + 1] = scratch[lane * 3 + 1] - T.float16(1)


def test_high_precision_preserves_strided_addresses_and_same_type_aliases(tmp_path):
    source = 1 + np.arange(1, 33, dtype=np.float32) / 65536
    backing = np.zeros(96, np.float16)
    scratch = backing[:]
    output = backing[:]
    module = numsim.transpile(strided_half_spill, precision="high", cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "scratch": scratch, "output": output})
    np.testing.assert_array_equal(result.outputs["output"][1::3], source.astype(np.float64) - 1)
    np.testing.assert_array_equal(backing[::3], 0)
    np.testing.assert_array_equal(backing[2::3], 0)


@pytest.mark.parametrize("dtype,transpose", [("float16", False), ("bfloat16", True)])
def test_warp_gemm_accumulates_in_fp64_with_original_fragment_layout(dtype, transpose, tmp_path):
    import ml_dtypes
    from tests.numsim.integration.test_warp_gemm_artifact import _build_warp_gemm_variant

    kernel, (rows, columns, reduction), _, _ = _build_warp_gemm_variant(
        dtype=dtype,
        mma_k=16,
        m_tiles=2,
        n_tiles=2,
        k_tiles=2,
        transpose_a=transpose,
        transpose_b=transpose,
        beta=1,
    )
    carrier = np.float16 if dtype == "float16" else ml_dtypes.bfloat16
    left = np.zeros((rows, reduction), dtype=carrier)
    right = np.zeros((reduction, columns), dtype=carrier)
    left[:, :3] = [4096, 1, -4096]
    right[:3, :] = np.array([4096, 1, 4096])[:, None]
    initial = ((np.arange(rows * columns, dtype=np.float32) % 257 - 128) / 512).reshape(
        rows, columns
    )
    expected = left.astype(np.float64) @ right.astype(np.float64) + initial.astype(np.float64)
    module = numsim.transpile(kernel, precision="high", cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_global": np.ascontiguousarray(left.T if transpose else left),
            "b_global": np.ascontiguousarray(right.T if transpose else right),
            "c_global": initial,
            "output": np.zeros((rows, columns), np.float32),
        },
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_high_precision_shadow_survives_kernel_phases_but_not_separate_runs(tmp_path):
    source = np.tile(np.array([1, 2**-25, 0], np.float32), (32, 1))
    intermediate = np.zeros(32, np.float32)
    module = numsim.transpile([cancellation, half_spill], precision="high", cache_dir=tmp_path)
    inputs = {
        "k0:source": source,
        "k0:scratch": np.zeros(32, np.float32),
        "k0:output": intermediate,
        "k1:source": intermediate,
        "k1:scratch": np.zeros(32, np.float16),
        "k1:output": np.zeros(32, np.float16),
    }
    result = numsim.Engine(max_workers=4).run(module, inputs, outputs=("k1:output",))
    np.testing.assert_array_equal(result.outputs["k1:output"], 2**-25)
    np.testing.assert_array_equal(intermediate, 1)
    source[:, 1] = 2**-26
    result = numsim.Engine(max_workers=4).run(module, inputs, outputs=("k1:output",))
    np.testing.assert_array_equal(result.outputs["k1:output"], 2**-26)


@T.prim_func
def warp_sum_spill(source: T.Buffer((32,), "float32"), output: T.Buffer((32,), "float16")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value: T.float16 = T.cast(source[lane], "float16")
    output[lane] = T.cuda.warp_sum(value, width=8)


def test_high_precision_promotes_warp_shuffle_reduction(tmp_path):
    source = np.tile(np.array([2048, 1, -2048, 0.5, 0, 0, 0, 0], np.float32), 4)
    module = numsim.transpile(warp_sum_spill, precision="high", cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": np.zeros(32, np.float16)})
    np.testing.assert_array_equal(result.outputs["output"], 1.5)


@T.prim_func
def transcendental(source: T.Buffer((32,), "float32"), output: T.Buffer((3, 32), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[0, lane] = T.exp(source[lane])
    output[1, lane] = T.log(source[lane])
    output[2, lane] = T.rsqrt(source[lane])


def test_high_precision_transcendentals_use_fp64_math(tmp_path):
    source = np.linspace(0.125, 4, 32, dtype=np.float32)
    module = numsim.transpile(transcendental, precision="high", cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "output": np.zeros((3, 32), np.float32)}
    )
    values = source.astype(np.longdouble)
    expected = np.stack([np.exp(values), np.log(values), 1 / np.sqrt(values)]).astype(np.float64)
    np.testing.assert_allclose(result.outputs["output"], expected, rtol=3e-16, atol=1e-16)
