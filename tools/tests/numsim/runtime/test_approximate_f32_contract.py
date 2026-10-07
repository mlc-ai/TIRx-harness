from __future__ import annotations

import ctypes
import math
import sys

import numpy as np
import pytest

from tirx_harness import numsim
from tests.numsim.support.manifest import emitted_calls
from tvm.script import tirx as T


@T.prim_func
def approximate_f32_calls(
    source: T.Buffer((32,), "float32"),
    exponentials: T.Buffer((32,), "float32"),
    reciprocals: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    exponential = T.local_scalar("float32")
    reciprocal = T.local_scalar("float32")
    T.ptx.ex2.approx.ftz.f32(exponential, source[lane])
    T.ptx.rcp.approx.ftz.f32(reciprocal, source[lane])
    exponentials[lane] = exponential
    reciprocals[lane] = reciprocal


@T.prim_func
def non_ftz_exp2_calls(
    source: T.Buffer((32,), "float32"),
    exponentials: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    exponential = T.local_scalar("float32")
    T.ptx.ex2.approx.f32(exponential, source[lane])
    exponentials[lane] = exponential


@T.prim_func
def non_ftz_rcp_calls(
    source: T.Buffer((32,), "float32"),
    reciprocals: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    reciprocal = T.local_scalar("float32")
    T.ptx.rcp.approx.f32(reciprocal, source[lane])
    reciprocals[lane] = reciprocal


@T.prim_func
def bf16x2_exp2_calls(
    source: T.Buffer((32,), "uint32"),
    exponentials: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    exponential = T.local_scalar("uint32")
    T.ptx["ex2.approx.ftz.bf16x2"](exponential, source[lane])
    exponentials[lane] = exponential


@T.prim_func
def f16x2_exp2_calls(
    source: T.Buffer((32,), "uint32"),
    exponentials: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    exponential = T.local_scalar("uint32")
    T.ptx["ex2.approx.f16x2"](exponential, source[lane])
    exponentials[lane] = exponential


@T.prim_func
def flashkda_ptx_unary_calls(
    source: T.Buffer((32,), "float32"),
    tanh_output: T.Buffer((32,), "float32"),
    rsqrt_output: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    tanh_value = T.local_scalar("float32")
    rsqrt_value = T.local_scalar("float32")
    T.ptx.tanh.approx.f32(tanh_value, source[lane])
    T.ptx.rsqrt.approx.ftz.f32(rsqrt_value, source[lane])
    tanh_output[lane] = tanh_value
    rsqrt_output[lane] = rsqrt_value


@T.prim_func
def f16x2_tanh_calls(
    source: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value = T.local_scalar("uint32")
    T.ptx["tanh.approx.f16x2"](value, source[lane])
    output[lane] = value


def _assert_nan_aware_bitwise_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    expected_nan = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), expected_nan)
    np.testing.assert_array_equal(
        actual[~expected_nan].view(np.uint32), expected[~expected_nan].view(np.uint32)
    )


@pytest.fixture(scope="module")
def flashkda_ptx_unary_result(tmp_path_factory):
    source_bits = np.resize(
        np.array(
            [
                0x00000001,
                0x80000001,
                0x00000000,
                0x80000000,
                0x3E800000,
                0x3F800000,
                0x40800000,
                0x41800000,
                0x7F800000,
                0xFF800000,
                0xBF800000,
                0x7FC12345,
            ],
            dtype=np.uint32,
        ),
        32,
    )
    source = source_bits.view(np.float32)
    module = numsim.transpile(
        flashkda_ptx_unary_calls,
        cache_dir=tmp_path_factory.mktemp("flashkda-ptx-unary"),
    )
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "tanh_output": np.zeros(32, dtype=np.float32),
            "rsqrt_output": np.zeros(32, dtype=np.float32),
        },
    )
    return source_bits, result


def test_tanh_approx_representative_executes(flashkda_ptx_unary_result):
    source_bits, result = flashkda_ptx_unary_result
    source = source_bits.view(np.float32)
    expected_tanh = np.array([math.tanh(float(value)) for value in source], dtype=np.float32)
    _assert_nan_aware_bitwise_equal(result.outputs["tanh_output"], expected_tanh)


def test_f16x2_tanh_approx_representative_executes(tmp_path):
    values = np.resize(
        np.array(
            [
                [0.0, -0.0],
                [0.25, -0.25],
                [1.0, -1.0],
                [4.0, -4.0],
                [math.inf, -math.inf],
            ],
            dtype=np.float16,
        ),
        (32, 2),
    )
    source = values.view(np.uint32).reshape(32)
    module = numsim.transpile(f16x2_tanh_calls, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source": source, "output": np.zeros(32, dtype=np.uint32)},
    )
    expected = np.array(
        [[math.tanh(float(value)) for value in pair] for pair in values],
        dtype=np.float16,
    )

    np.testing.assert_array_equal(
        result.outputs["output"], expected.view(np.uint32).reshape(32)
    )


def test_rsqrt_approx_ftz_representative_executes(flashkda_ptx_unary_result):
    source_bits, result = flashkda_ptx_unary_result
    flushed = source_bits.copy()
    subnormal = (flushed & np.uint32(0x7F800000)) == 0
    nonzero = (flushed & np.uint32(0x007FFFFF)) != 0
    flushed[subnormal & nonzero] &= np.uint32(0x80000000)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_rsqrt = np.float32(1.0) / np.sqrt(flushed.view(np.float32), dtype=np.float32)

    _assert_nan_aware_bitwise_equal(result.outputs["rsqrt_output"], expected_rsqrt)


@pytest.mark.parametrize(
    ("kernel", "op_name", "instruction", "variant"),
    (
        (approximate_f32_calls, "tirx.ptx.ex2", "exp2", "F32RnFtz"),
        (approximate_f32_calls, "tirx.ptx.rcp", "rcp", "F32RnFtz"),
        (non_ftz_exp2_calls, "tirx.ptx.ex2", "exp2", "F32"),
        (non_ftz_rcp_calls, "tirx.ptx.rcp", "rcp", "F32"),
    ),
)
def test_approximate_f32_emission_selects_the_subnormal_policy(
    kernel, op_name, instruction, variant
):
    calls = emitted_calls(kernel, op_name)
    assert [(call.function, call.generics) for call in calls] == [
        (f"v2::reg::{instruction}", f"v2::reg::variant::{variant}"),
    ]


def test_non_ftz_exp2_executes_and_preserves_subnormal_outputs(tmp_path):
    source = np.resize(
        np.array([0.0, 1.0, -149.0, -148.0, -150.0, 0.5], dtype=np.float32),
        32,
    )
    module = numsim.transpile(non_ftz_exp2_calls, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "exponentials": np.zeros(32, dtype=np.float32),
        },
    )

    expected = np.exp2(source.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(
        result.outputs["exponentials"].view(np.uint32), expected.view(np.uint32)
    )
    assert result.outputs["exponentials"][2].view(np.uint32) == np.uint32(1)


def test_bf16x2_exp2_executes_both_lanes_and_flushes_bf16_subnormals(tmp_path):
    source = np.resize(
        np.array(
            [
                0x3F80_0000,  # (0.0, 1.0) -> (1.0, 2.0)
                0x0001_FC00,  # (-Inf, +subnormal) -> (+0.0, 1.0)
                0xC2FE_C2FC,  # (-126.0, -127.0) -> (min normal, FTZ)
            ],
            dtype=np.uint32,
        ),
        32,
    )
    module = numsim.transpile(bf16x2_exp2_calls, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "exponentials": np.zeros(32, dtype=np.uint32),
        },
    )

    expected = np.resize(
        np.array([0x4000_3F80, 0x3F80_0000, 0x0000_0080], dtype=np.uint32),
        32,
    )
    np.testing.assert_array_equal(result.outputs["exponentials"], expected)


def test_f16x2_exp2_executes_both_lanes_and_preserves_subnormal_results(tmp_path):
    source = np.resize(
        np.array(
            [
                0x3C00_0000,  # (0.0, 1.0) -> (1.0, 2.0)
                0x0001_FC00,  # (-Inf, +subnormal) -> (+0.0, 1.0)
                0xCE00_CB00,  # (-14.0, -24.0) -> (min normal, min subnormal)
                0x7C00_CE40,  # (-25.0, +Inf) -> (+0.0, +Inf)
            ],
            dtype=np.uint32,
        ),
        32,
    )
    module = numsim.transpile(f16x2_exp2_calls, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "exponentials": np.zeros(32, dtype=np.uint32),
        },
    )

    expected = np.resize(
        np.array(
            [0x4000_3C00, 0x3C00_0000, 0x0001_0400, 0x7C00_0000],
            dtype=np.uint32,
        ),
        32,
    )
    np.testing.assert_array_equal(result.outputs["exponentials"], expected)


def test_non_ftz_rcp_executes_and_preserves_subnormal_outputs(tmp_path):
    source = np.resize(
        np.array([3.0, np.finfo(np.float32).max, np.inf, -np.inf], dtype=np.float32),
        32,
    )
    module = numsim.transpile(non_ftz_rcp_calls, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "reciprocals": np.zeros(32, dtype=np.float32),
        },
    )

    expected_bits = np.resize(
        np.array([0x3EAA_AAAB, 0x0020_0000, 0x0000_0000, 0x8000_0000], dtype=np.uint32),
        32,
    )
    np.testing.assert_array_equal(result.outputs["reciprocals"].view(np.uint32), expected_bits)


def _linux_fenv():
    if not sys.platform.startswith("linux"):
        pytest.skip("the fenv regression currently requires the Linux C runtime")
    libc = ctypes.CDLL(None)
    try:
        fegetround = libc.fegetround
        fesetround = libc.fesetround
    except AttributeError:
        pytest.skip("the process C runtime does not expose fegetround/fesetround")
    fegetround.argtypes = []
    fegetround.restype = ctypes.c_int
    fesetround.argtypes = [ctypes.c_int]
    fesetround.restype = ctypes.c_int
    return fegetround, fesetround


def test_worker_normalizes_rounding_without_polluting_the_caller(tmp_path):
    source = np.ones(32, dtype=np.float32)
    source[0] = np.float32(1.234567)
    source[1] = np.float32(3.0)
    arguments = {
        "source": source,
        "exponentials": np.zeros(32, dtype=np.float32),
        "reciprocals": np.zeros(32, dtype=np.float32),
    }
    module = numsim.transpile(approximate_f32_calls, cache_dir=tmp_path)
    engine = numsim.Engine(max_workers=1)
    fegetround, fesetround = _linux_fenv()
    original_rounding = fegetround()
    fe_downward = 0x400
    if fesetround(fe_downward) != 0:
        pytest.skip("the process C runtime rejected FE_DOWNWARD")
    try:
        result = engine.run(module, arguments)
        assert fegetround() == fe_downward
    finally:
        assert fesetround(original_rounding) == 0

    exp_bits = result.outputs["exponentials"].view(np.uint32)
    reciprocal_bits = result.outputs["reciprocals"].view(np.uint32)
    assert exp_bits[0] == np.uint32(0x4016994F)
    assert reciprocal_bits[1] == np.uint32(0x3EAAAAAB)


def test_output_comparison_does_not_guess_how_a_local_ulp_budget_propagates():
    representative = np.array([1.0], dtype=np.float32)
    adjacent = np.nextafter(representative, np.float32(2.0))
    report = numsim.compare(
        numsim.NumSimResult({"output": representative}),
        {"output": adjacent},
        tolerances={"output": numsim.ComparisonSpec(rtol=0.0, atol=0.0)},
    )

    assert not report.ok
