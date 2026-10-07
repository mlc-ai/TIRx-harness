from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tvm.script import tirx as T


_TANH_APPROX_SOURCE = r"""// tanh.approx.f32 (sigmoid via tanh; used throughout the prep role)
__device__ __forceinline__ float flashkda_tanh_approx(float x) {
    float y;
    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
"""


_FMAF_RN_SOURCE = r"""// __fmaf_rn (value form of fma.rn.f32)
__device__ __forceinline__ float flashkda_fmaf_rn(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}
"""


_RSQRTF_SOURCE = r"""// rsqrtf
__device__ __forceinline__ float flashkda_rsqrtf(float x) {
    return rsqrtf(x);
}
"""


@T.prim_func
def flashkda_math_helpers(
    a: T.Buffer((32,), "float32"),
    b: T.Buffer((32,), "float32"),
    c: T.Buffer((32,), "float32"),
    rsqrt_input: T.Buffer((32,), "float32"),
    tanh_input: T.Buffer((32,), "float32"),
    output: T.Buffer((32, 3), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane, 0] = T.cuda.func_call(
        "flashkda_fmaf_rn",
        a[lane],
        b[lane],
        c[lane],
        source_code=_FMAF_RN_SOURCE,
        return_type="float32",
    )
    output[lane, 1] = T.cuda.func_call(
        "flashkda_rsqrtf",
        rsqrt_input[lane],
        source_code=_RSQRTF_SOURCE,
        return_type="float32",
    )
    output[lane, 2] = T.cuda.func_call(
        "flashkda_tanh_approx",
        tanh_input[lane],
        source_code=_TANH_APPROX_SOURCE,
        return_type="float32",
    )


@T.prim_func
def modified_flashkda_fmaf_rn(output: T.Buffer((1,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[0] = T.cuda.func_call(
            "flashkda_fmaf_rn",
            T.float32(1),
            T.float32(2),
            T.float32(3),
            source_code=_FMAF_RN_SOURCE.replace("__fmaf_rn(a, b, c)", "a * b + c"),
            return_type="float32",
        )


@T.prim_func
def modified_flashkda_rsqrtf(output: T.Buffer((1,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[0] = T.cuda.func_call(
            "flashkda_rsqrtf",
            T.float32(4),
            source_code=_RSQRTF_SOURCE.replace("rsqrtf(x)", "sqrtf(x)"),
            return_type="float32",
        )


@T.prim_func
def modified_flashkda_tanh_approx(output: T.Buffer((1,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[0] = T.cuda.func_call(
            "flashkda_tanh_approx",
            T.float32(1),
            source_code=_TANH_APPROX_SOURCE.replace("tanh.approx.f32", "ex2.approx.f32"),
            return_type="float32",
        )


@T.prim_func
def flashkda_rsqrtf_wrong_dtype(output: T.Buffer((1,), "float64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[0] = T.cuda.func_call(
            "flashkda_rsqrtf",
            T.float64(4),
            source_code=_RSQRTF_SOURCE,
            return_type="float64",
        )


def test_flashkda_math_helpers_have_reviewed_observable_semantics(tmp_path):
    lane = np.arange(32, dtype=np.float32)
    a = (1.0 + lane / 64.0).astype(np.float32)
    b = (1.0 - lane / 96.0).astype(np.float32)
    c = (-0.75 + lane / 128.0).astype(np.float32)
    a[0] = np.float32(1.0 + 2.0**-12)
    b[0] = np.float32(1.0 - 2.0**-12)
    c[0] = np.float32(-1.0)
    rsqrt_input = np.linspace(0.25, 16.0, 32, dtype=np.float32)
    tanh_input = np.linspace(-4.0, 4.0, 32, dtype=np.float32)

    module = numsim.transpile(flashkda_math_helpers, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a": a,
            "b": b,
            "c": c,
            "rsqrt_input": rsqrt_input,
            "tanh_input": tanh_input,
            "output": np.zeros((32, 3), dtype=np.float32),
        },
    )

    expected_fma = (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(
        np.float32
    )
    expected_rsqrt = np.divide(
        np.float32(1), np.sqrt(rsqrt_input, dtype=np.float32), dtype=np.float32
    )
    expected_tanh = np.tanh(tanh_input.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(result.outputs["output"][:, 0], expected_fma)
    np.testing.assert_array_equal(result.outputs["output"][:, 1], expected_rsqrt)
    np.testing.assert_allclose(
        result.outputs["output"][:, 2], expected_tanh, rtol=0, atol=np.float32(2.0**-23)
    )
    assert result.outputs["output"][0, 0] == np.float32(-(2.0**-24))


@pytest.mark.parametrize(
    "kernel",
    (
        modified_flashkda_fmaf_rn,
        modified_flashkda_rsqrtf,
        modified_flashkda_tanh_approx,
    ),
)
def test_flashkda_math_helper_semantic_mutations_fail_closed(kernel, tmp_path):
    with pytest.raises(UnsupportedTIRxError, match="does not match"):
        numsim.transpile(kernel, cache_dir=tmp_path)


def test_flashkda_math_helper_dtype_mismatch_fails_closed(tmp_path):
    with pytest.raises(UnsupportedTIRxError, match="requires"):
        numsim.transpile(flashkda_rsqrtf_wrong_dtype, cache_dir=tmp_path)
