from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import racecheck, synccheck
from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def packed_ptx_cvt(
    high: T.Buffer((8,), "float32"),
    low: T.Buffer((8,), "float32"),
    e4m3x2: T.Buffer((8,), "uint16"),
    packed_bf16: T.Buffer((32,), "uint32"),
    packed_e8m0: T.Buffer((32,), "uint16"),
    unpacked_e8m0: T.Buffer((32,), "uint32"),
    unpacked_e4m3_bf16: T.Buffer((32,), "uint32"),
    unpacked_e4m3_f16: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    index: T.let = lane % 8
    packed = T.local_scalar("uint16")
    T.ptx.cvt.rn.bf16x2.f32(packed_bf16[lane], high[index], low[index])
    T.ptx.cvt.rz.ue8m0x2.f32(packed, high[index], low[index])
    packed_e8m0[lane] = packed
    T.ptx.cvt.rn.bf16x2.ue8m0x2(unpacked_e8m0[lane], packed)
    T.ptx.cvt.rn.bf16x2.e4m3x2(unpacked_e4m3_bf16[lane], e4m3x2[index])
    T.ptx.cvt.rn.f16x2.e4m3x2(unpacked_e4m3_f16[lane], e4m3x2[index])


@T.prim_func
def canonical_cvt_modifiers(
    source: T.Buffer((8,), "float32"),
    high: T.Buffer((8,), "float32"),
    low: T.Buffer((8,), "float32"),
    saturated: T.Buffer((8,), "float32"),
    packed: T.Buffer((8,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 8:
        T.ptx["cvt.sat.f32.f32"](saturated[lane], source[lane])
        T.ptx["cvt.rn.satfinite.bf16x2.f32"](packed[lane], high[lane], low[lane])


@T.prim_func
def warp_early_return(output: T.Buffer((64,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    if warp == 0:
        return 0
    output[warp * 32 + lane] = warp * 100 + lane


@T.prim_func
def low_precision_scalar_reinterpret(
    bf16_bits: T.Buffer((32,), "uint16"),
    fp16_bits: T.Buffer((32,), "uint16"),
    decoded: T.Buffer((32, 2), "float32"),
    roundtrip: T.Buffer((32, 2), "uint16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    bf16: T.let = T.reinterpret("bfloat16", bf16_bits[lane])
    fp16: T.let = T.reinterpret("float16", fp16_bits[lane])
    decoded[lane, 0] = T.cast(bf16, "float32")
    decoded[lane, 1] = T.cast(fp16, "float32")
    roundtrip[lane, 0] = T.reinterpret("uint16", bf16)
    roundtrip[lane, 1] = T.reinterpret("uint16", fp16)


def _packed_cvt_inputs() -> dict[str, np.ndarray]:
    return {
        "high": np.asarray([1.0, 1.99, 3.99, 0.75, 0.0, -1.0, np.inf, np.nan], np.float32),
        "low": np.asarray([2.0, 3.99, 7.99, 1.25, -0.0, -2.0, -np.inf, -np.nan], np.float32),
        "e4m3x2": np.asarray(
            [0x4038, 0xC038, 0xB040, 0x0038, 0x7F38, 0xFF38, 0x3838, 0x4040],
            np.uint16,
        ),
        "packed_bf16": np.zeros(32, np.uint32),
        "packed_e8m0": np.zeros(32, np.uint16),
        "unpacked_e8m0": np.zeros(32, np.uint32),
        "unpacked_e4m3_bf16": np.zeros(32, np.uint32),
        "unpacked_e4m3_f16": np.zeros(32, np.uint32),
    }


def test_packed_ptx_cvt_matches_instruction_lane_and_rounding_contract(tmp_path):
    inputs = _packed_cvt_inputs()
    result = numsim.Engine().run(numsim.transpile(packed_ptx_cvt, cache_dir=tmp_path), inputs)

    expected_packed = np.asarray(
        [0x7F80, 0x7F80, 0x8081, 0x7E7F, 0x0000, 0x7F80, 0xFFFF, 0xFFFF],
        np.uint16,
    )
    expected_e8m0 = np.asarray(
        [
            0x3F804000,
            0x3F804000,
            0x40004080,
            0x3F003F80,
            0x00400040,
            0x3F804000,
            0x7FFF7FFF,
            0x7FFF7FFF,
        ],
        np.uint32,
    )
    np.testing.assert_array_equal(result.outputs["packed_e8m0"], np.tile(expected_packed, 4))
    expected_bf16 = np.asarray(
        [
            0x3F804000,
            0x3FFF407F,
            0x407F4100,
            0x3F403FA0,
            0x00008000,
            0xBF80C000,
            0x7F80FF80,
            0x7FFF7FFF,
        ],
        np.uint32,
    )
    np.testing.assert_array_equal(result.outputs["packed_bf16"], np.tile(expected_bf16, 4))
    np.testing.assert_array_equal(result.outputs["unpacked_e8m0"], np.tile(expected_e8m0, 4))
    expected_e4m3_bf16 = np.asarray(
        [
            0x40003F80,
            0xC0003F80,
            0xBF004000,
            0x00003F80,
            0x7FFF3F80,
            0x7FFF3F80,
            0x3F803F80,
            0x40004000,
        ],
        np.uint32,
    )
    np.testing.assert_array_equal(
        result.outputs["unpacked_e4m3_bf16"], np.tile(expected_e4m3_bf16, 4)
    )
    # Lanes 4 and 5 read the e4m3 NaN encodings 0x7f and 0xff.  Both widen to
    # the canonical 0x7fff payload with the sign dropped, matching
    # `cvt.rn.f16x2.e4m3x2` on an NVIDIA B200 (driver 595.58.03, CUDA 13.2);
    # the earlier 0x7e00 came from routing the NaN through an `f32` NaN and
    # re-narrowing it.  `tests/numsim/microtests/cases/ptx_cvt_fp8_goldens.py`
    # holds the exhaustive per-byte table this row is a slice of.
    expected_e4m3_f16 = np.asarray(
        [
            0x40003C00,
            0xC0003C00,
            0xB8004000,
            0x00003C00,
            0x7FFF3C00,
            0x7FFF3C00,
            0x3C003C00,
            0x40004000,
        ],
        np.uint32,
    )
    np.testing.assert_array_equal(
        result.outputs["unpacked_e4m3_f16"], np.tile(expected_e4m3_f16, 4)
    )


@pytest.mark.parametrize("backend", ["numsim", pytest.param("gpu", marks=pytest.mark.numsim_gpu)])
def test_canonical_cvt_modifiers_match_ptx_semantics(tmp_path, pytestconfig, backend):
    if backend == "gpu":
        from tests.numsim.microtests.harness import require_numsim_gpu, run_gpu_primfunc

        require_numsim_gpu(pytestconfig)
    source = np.asarray([-2.0, -0.0, 0.25, 1.0, 2.0, np.nan, np.inf, -np.inf], np.float32)
    maximum = np.finfo(np.float32).max
    high = np.asarray([maximum, -maximum, np.inf, -np.inf, 1.0, -1.0, np.nan, 0.0], np.float32)
    low = np.asarray([-maximum, maximum, -np.inf, np.inf, -1.0, 1.0, 0.0, np.nan], np.float32)
    arguments = {
        "source": source,
        "high": high,
        "low": low,
        "saturated": np.zeros(8, np.float32),
        "packed": np.zeros(8, np.uint32),
    }
    for checker in (synccheck, racecheck):
        checker(canonical_cvt_modifiers, arguments).require_clean()
    result = numsim.Engine().run(
        numsim.transpile(canonical_cvt_modifiers, cache_dir=tmp_path), arguments
    )
    outputs = result.outputs
    if backend == "gpu":
        outputs = run_gpu_primfunc(
            canonical_cvt_modifiers, arguments, outputs=("saturated", "packed"), arch="sm_100a"
        )
        for name in ("saturated", "packed"):
            np.testing.assert_array_equal(
                outputs[name].view(np.uint32), result.outputs[name].view(np.uint32)
            )

    # B200 CVT saturation canonicalizes negative zero too, matching the
    # shared saturation helper; this is a bit-level assertion, not a tolerance.
    expected_saturated = np.asarray([0.0, 0.0, 0.25, 1.0, 1.0, 0.0, 1.0, 0.0], np.float32)
    np.testing.assert_array_equal(
        outputs["saturated"].view(np.uint32), expected_saturated.view(np.uint32)
    )
    expected_packed = np.asarray(
        [
            0x7F7FFF7F,
            0xFF7F7F7F,
            0x7F7FFF7F,
            0xFF7F7F7F,
            0x3F80BF80,
            0xBF803F80,
            0x7FFF0000,
            0x00007FFF,
        ],
        np.uint32,
    )
    np.testing.assert_array_equal(outputs["packed"], expected_packed)


def test_warp_uniform_early_return_removes_only_returning_warp(tmp_path):
    inputs = {"output": np.full(64, -7, np.int32)}
    result = numsim.Engine().run(numsim.transpile(warp_early_return, cache_dir=tmp_path), inputs)
    expected = np.full(64, -7, np.int32)
    expected[32:] = np.arange(100, 132, dtype=np.int32)
    np.testing.assert_array_equal(result.outputs["output"], expected)

    synccheck(warp_early_return, inputs=inputs).require_clean()
    racecheck(warp_early_return, inputs=inputs).require_clean()


def test_low_precision_scalar_reinterpret_preserves_bits_and_decodes_values(tmp_path):
    bf16_bits = np.arange(0x3F70, 0x3F90, dtype=np.uint16)
    fp16_bits = np.arange(0x3BF0, 0x3C10, dtype=np.uint16)
    inputs = {
        "bf16_bits": bf16_bits,
        "fp16_bits": fp16_bits,
        "decoded": np.zeros((32, 2), np.float32),
        "roundtrip": np.zeros((32, 2), np.uint16),
    }
    result = numsim.Engine().run(
        numsim.transpile(low_precision_scalar_reinterpret, cache_dir=tmp_path), inputs
    )

    np.testing.assert_array_equal(result.outputs["roundtrip"][:, 0], bf16_bits)
    np.testing.assert_array_equal(result.outputs["roundtrip"][:, 1], fp16_bits)
    np.testing.assert_array_equal(
        result.outputs["decoded"][:, 0], (bf16_bits.astype(np.uint32) << 16).view(np.float32)
    )
    np.testing.assert_array_equal(
        result.outputs["decoded"][:, 1], fp16_bits.view(np.float16).astype(np.float32)
    )
