from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def ptx_clmad_register_semantics(
    lhs: T.Buffer((32,), "uint64"),
    rhs: T.Buffer((32,), "uint64"),
    addend: T.Buffer((32,), "uint64"),
    output: T.Buffer((2, 32), "uint64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.clmad.lo.u64(output[0, lane], lhs[lane], rhs[lane], addend[lane])
    T.ptx.clmad.hi.u64(output[1, lane], lhs[lane], rhs[lane], addend[lane])


def _carryless_product(lhs: int, rhs: int) -> int:
    """Independent GF(2) product used by the PTX ``clmad`` oracle."""

    product = 0
    for bit in range(64):
        if lhs & (1 << bit):
            product ^= rhs << bit
    return product


def _clmad_reference(
    lhs: np.ndarray, rhs: np.ndarray, addend: np.ndarray, *, high: bool
) -> np.ndarray:
    mask = (1 << 64) - 1
    shift = 64 if high else 0
    return np.asarray(
        [
            ((_carryless_product(int(a), int(b)) >> shift) ^ int(c)) & mask
            for a, b, c in zip(lhs, rhs, addend, strict=True)
        ],
        dtype=np.uint64,
    )


def _clmad_arguments() -> dict[str, np.ndarray]:
    mask = (1 << 64) - 1
    lhs = np.asarray(
        [
            0,
            1,
            1 << 63,
            mask,
            0x8000_0000_0000_0001,
            0x0123_4567_89AB_CDEF,
            *(
                ((0x9E37_79B9_7F4A_7C15 * lane) ^ (1 << (lane % 64))) & mask
                for lane in range(6, 32)
            ),
        ],
        dtype=np.uint64,
    )
    rhs = np.asarray(
        [
            mask,
            mask,
            1 << 63,
            mask,
            0xC96C_5795_D787_0F42,
            0xFEDC_BA98_7654_3210,
            *(
                ((0xD6E8_FEB8_6659_FD93 * (lane + 1)) ^ (mask >> (lane % 17))) & mask
                for lane in range(6, 32)
            ),
        ],
        dtype=np.uint64,
    )
    addend = np.asarray(
        [((0xA5A5_5A5A_DEAD_BEEF + 0x0102_0408_1020_4081 * lane) & mask) for lane in range(32)],
        dtype=np.uint64,
    )
    return {
        "lhs": lhs,
        "rhs": rhs,
        "addend": addend,
        "output": np.zeros((2, 32), dtype=np.uint64),
    }


def test_ptx_clmad_hi_and_lo_match_independent_gf2_oracle(tmp_path):
    arguments = _clmad_arguments()
    result = numsim.Engine().run(
        numsim.transpile(ptx_clmad_register_semantics, cache_dir=tmp_path),
        arguments,
        outputs=("output",),
    )

    expected = np.stack(
        (
            _clmad_reference(arguments["lhs"], arguments["rhs"], arguments["addend"], high=False),
            _clmad_reference(arguments["lhs"], arguments["rhs"], arguments["addend"], high=True),
        )
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


__all__ = ["ptx_clmad_register_semantics"]
