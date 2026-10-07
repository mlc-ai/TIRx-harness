from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tvm.script import tirx as T


_COMBINE_SOURCE = r"""
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
"""


_SHL_U32_CLAMP_SOURCE = r"""
__device__ __forceinline__ unsigned int shl_u32_clamp(unsigned int val, unsigned int shift) {
  unsigned int r;
  asm("shl.b32 %0, %1, %2;" : "=r"(r) : "r"(val), "r"(shift));
  return r;
}
"""


_GDN_LG2_APPROX_FTZ_SOURCE = r"""
__device__ __forceinline__ float gdn_lg2_approx_ftz(float value) {
    float out;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(value));
    return out;
}
"""


_FMA_SCALE_SUB_F32X2_SOURCE = r"""
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
"""


def _pack_float2(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return low.view(np.uint32).astype(np.uint64) | (high.view(np.uint32).astype(np.uint64) << 32)


def _unpack_float2(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = (values & np.uint64(0xFFFF_FFFF)).astype(np.uint32).view(np.float32)
    high = (values >> np.uint64(32)).astype(np.uint32).view(np.float32)
    return low, high


@T.prim_func
def known_combine_int_frac_ex2(
    rounded: T.Buffer((32,), "float32"),
    fraction: T.Buffer((32,), "float32"),
    output: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.cuda.func_call(
        "combine_int_frac_ex2",
        rounded[lane],
        fraction[lane],
        source_code=_COMBINE_SOURCE,
        return_type="float32",
    )


@T.prim_func
def modified_combine_int_frac_ex2(output: T.Buffer((1,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[0] = T.cuda.func_call(
            "combine_int_frac_ex2",
            T.float32(0),
            T.float32(1),
            source_code=_COMBINE_SOURCE.replace("return out;", "out += 1.0f; return out;"),
            return_type="float32",
        )


@T.prim_func
def known_shl_u32_clamp(
    values: T.Buffer((8,), "uint32"),
    shifts: T.Buffer((8,), "uint32"),
    output: T.Buffer((8,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 8:
        output[lane] = T.cuda.func_call(
            "shl_u32_clamp",
            values[lane],
            shifts[lane],
            source_code=_SHL_U32_CLAMP_SOURCE,
            return_type="uint32",
        )


@T.prim_func
def known_gdn_lg2_approx_ftz(values: T.Buffer((8,), "float32"), output: T.Buffer((8,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 8:
        output[lane] = T.cuda.func_call(
            "gdn_lg2_approx_ftz",
            values[lane],
            source_code=_GDN_LG2_APPROX_FTZ_SOURCE,
            return_type="float32",
        )


@T.prim_func
def modified_gdn_lg2_approx_ftz(output: T.Buffer((1,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[0] = T.cuda.func_call(
            "gdn_lg2_approx_ftz",
            T.float32(1),
            source_code=_GDN_LG2_APPROX_FTZ_SOURCE.replace("approx.ftz", "approx"),
            return_type="float32",
        )


@T.prim_func
def known_fma_scale_sub_f32x2(
    scores: T.Buffer((8,), "uint64"),
    scales: T.Buffer((8,), "uint64"),
    lse: T.Buffer((8,), "uint64"),
    output: T.Buffer((8,), "uint64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 8:
        output[lane] = T.cuda.func_call(
            "tvm_builtin_fma_scale_sub_f32x2",
            scores[lane],
            scales[lane],
            lse[lane],
            source_code=_FMA_SCALE_SUB_F32X2_SOURCE,
            return_type="uint64",
        )


@T.prim_func
def modified_fma_scale_sub_f32x2(output: T.Buffer((1,), "uint64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[0] = T.cuda.func_call(
            "tvm_builtin_fma_scale_sub_f32x2",
            T.uint64(0),
            T.uint64(0),
            T.uint64(0),
            source_code=_FMA_SCALE_SUB_F32X2_SOURCE.replace("-lse_pair.x", "+lse_pair.x"),
            return_type="uint64",
        )


def test_known_combine_int_frac_ex2_is_bit_exact(tmp_path):
    rounded_bits = np.arange(32, dtype=np.uint32) + np.uint32(0x4B40_0000)
    fraction_bits = np.arange(32, dtype=np.uint32) * np.uint32(0x0001_0203) + np.uint32(0x3F80_0000)
    rounded = rounded_bits.view(np.float32)
    fraction = fraction_bits.view(np.float32)
    output = np.zeros(32, dtype=np.float32)

    module = numsim.transpile(known_combine_int_frac_ex2, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"rounded": rounded, "fraction": fraction, "output": output},
    )

    expected_bits = (rounded_bits << np.uint32(23)) + fraction_bits
    np.testing.assert_array_equal(result.outputs["output"].view(np.uint32), expected_bits)
    assert "wrapping_shl(23_u32)" in module.rust_source
    assert "wrapping_add" in module.rust_source


def test_known_shl_u32_clamp_is_bit_exact(tmp_path):
    values = np.array(
        [0xFFFF_FFFF, 1, 3, 0x8000_0001, 0xDEAD_BEEF, 7, 9, 11],
        dtype=np.uint32,
    )
    shifts = np.array([0, 1, 7, 31, 32, 33, 63, 255], dtype=np.uint32)
    output = np.zeros(8, dtype=np.uint32)

    module = numsim.transpile(known_shl_u32_clamp, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"values": values, "shifts": shifts, "output": output},
    )

    expected = np.array(
        [
            (int(value) << int(shift)) & 0xFFFF_FFFF if shift < 32 else 0
            for value, shift in zip(values, shifts, strict=True)
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert "checked_shl" in module.rust_source
    assert "unwrap_or(0_u32)" in module.rust_source


def test_known_gdn_lg2_approx_ftz_flushes_subnormal_input(tmp_path):
    values = np.array(
        [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 10.0, np.nextafter(np.float32(0), np.float32(1))],
        dtype=np.float32,
    )
    output = np.zeros(8, dtype=np.float32)

    module = numsim.transpile(known_gdn_lg2_approx_ftz, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"values": values, "output": output})

    expected = np.log2(values[:-1].astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(result.outputs["output"][:-1], expected, rtol=0, atol=0)
    assert np.isneginf(result.outputs["output"][-1])


def test_known_fma_scale_sub_f32x2_computes_both_packed_lanes(tmp_path):
    scores_low = np.linspace(-3.0, 4.0, 8, dtype=np.float32)
    scores_high = np.linspace(2.0, -1.5, 8, dtype=np.float32)
    scales_low = np.full(8, np.float32(0.5))
    scales_high = np.full(8, np.float32(2.0))
    lse_low = np.linspace(-0.25, 0.625, 8, dtype=np.float32)
    lse_high = np.linspace(1.0, -0.75, 8, dtype=np.float32)
    inputs = {
        "scores": _pack_float2(scores_low, scores_high),
        "scales": _pack_float2(scales_low, scales_high),
        "lse": _pack_float2(lse_low, lse_high),
        "output": np.zeros(8, dtype=np.uint64),
    }

    module = numsim.transpile(known_fma_scale_sub_f32x2, cache_dir=tmp_path)
    result = numsim.Engine().run(module, inputs)

    actual_low, actual_high = _unpack_float2(result.outputs["output"])
    expected_low = scores_low * scales_low - lse_low
    expected_high = scores_high * scales_high - lse_high
    np.testing.assert_array_equal(actual_low, expected_low)
    np.testing.assert_array_equal(actual_high, expected_high)


def test_modified_gdn_lg2_helper_remains_fail_closed(tmp_path):
    with pytest.raises(numsim.UnsupportedTIRxError, match="does not match the validated"):
        numsim.transpile(modified_gdn_lg2_approx_ftz, cache_dir=tmp_path)


def test_modified_known_helper_remains_fail_closed(tmp_path):
    with pytest.raises(numsim.UnsupportedTIRxError, match="does not match the validated"):
        numsim.transpile(modified_combine_int_frac_ex2, cache_dir=tmp_path)


def test_modified_packed_fma_helper_remains_fail_closed(tmp_path):
    with pytest.raises(numsim.UnsupportedTIRxError, match="does not match the validated"):
        numsim.transpile(modified_fma_scale_sub_f32x2, cache_dir=tmp_path)
