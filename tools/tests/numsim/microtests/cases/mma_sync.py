from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from tvm.script import tirx as T


@T.prim_func
def raw_dense_f16_f32_k16(
    a_words: T.Buffer((32, 4), "uint32"),
    b_words: T.Buffer((32, 2), "uint32"),
    c_values: T.Buffer((32, 4), "float32"),
    output: T.Buffer((32, 4), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    a = T.alloc_local((4,), "uint32")
    b = T.alloc_local((2,), "uint32")
    c = T.alloc_local((4,), "float32")
    d = T.alloc_local((4,), "float32")
    for register in T.unroll(4):
        a[register] = a_words[lane, register]
        c[register] = c_values[lane, register]
    for register in T.unroll(2):
        b[register] = b_words[lane, register]
    T.ptx.mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32(
        d[0],
        d[1],
        d[2],
        d[3],
        a[0],
        a[1],
        a[2],
        a[3],
        b[0],
        b[1],
        c[0],
        c[1],
        c[2],
        c[3],
    )
    for register in T.unroll(4):
        output[lane, register] = T.reinterpret("uint32", d[register])


@T.prim_func
def raw_dense_fp8_f32_k32(
    a_words: T.Buffer((32, 4), "uint32"),
    b_words: T.Buffer((32, 2), "uint32"),
    c_values: T.Buffer((32, 4), "float32"),
    output: T.Buffer((32, 4), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    a = T.alloc_local((4,), "uint32")
    b = T.alloc_local((2,), "uint32")
    c = T.alloc_local((4,), "float32")
    d = T.alloc_local((4,), "float32")
    for register in T.unroll(4):
        a[register] = a_words[lane, register]
        c[register] = c_values[lane, register]
    for register in T.unroll(2):
        b[register] = b_words[lane, register]
    T.ptx.mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32(
        d[0],
        d[1],
        d[2],
        d[3],
        a[0],
        a[1],
        a[2],
        a[3],
        b[0],
        b[1],
        c[0],
        c[1],
        c[2],
        c[3],
    )
    for register in T.unroll(4):
        output[lane, register] = T.reinterpret("uint32", d[register])


@T.prim_func
def raw_sparse_f16_f32_k32(
    a_words: T.Buffer((32, 4), "uint32"),
    b_words: T.Buffer((32, 4), "uint32"),
    c_values: T.Buffer((32, 4), "float32"),
    metadata_words: T.Buffer((32,), "uint32"),
    output: T.Buffer((32, 4), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    a = T.alloc_local((4,), "uint32")
    b = T.alloc_local((4,), "uint32")
    accumulator = T.alloc_local((4,), "float32")
    metadata = T.alloc_local((1,), "uint32")
    for register in T.unroll(4):
        a[register] = a_words[lane, register]
        b[register] = b_words[lane, register]
        accumulator[register] = c_values[lane, register]
    metadata[0] = metadata_words[lane]
    T.ptx.mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32(
        accumulator[0],
        accumulator[1],
        accumulator[2],
        accumulator[3],
        a[0],
        a[1],
        a[2],
        a[3],
        b[0],
        b[1],
        b[2],
        b[3],
        accumulator[0],
        accumulator[1],
        accumulator[2],
        accumulator[3],
        metadata[0],
        1,
    )
    for register in T.unroll(4):
        output[lane, register] = T.reinterpret("uint32", accumulator[register])


@T.prim_func
def raw_dense_u8_wrap_and_sat(
    wrap_output: T.Buffer((32, 4), "uint32"),
    sat_output: T.Buffer((32, 4), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    a = T.alloc_local((4,), "uint32")
    b = T.alloc_local((2,), "uint32")
    c = T.alloc_local((4,), "uint32")
    wrap = T.alloc_local((4,), "uint32")
    saturated = T.alloc_local((4,), "uint32")
    for register in T.unroll(4):
        a[register] = T.uint32(0x7F7F7F7F)
        c[register] = T.uint32(0x7FFFFFF0)
    for register in T.unroll(2):
        b[register] = T.uint32(0x7F7F7F7F)
    T.ptx.mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32(
        wrap[0],
        wrap[1],
        wrap[2],
        wrap[3],
        a[0],
        a[1],
        a[2],
        a[3],
        b[0],
        b[1],
        c[0],
        c[1],
        c[2],
        c[3],
    )
    T.ptx.mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.u8.s32(
        saturated[0],
        saturated[1],
        saturated[2],
        saturated[3],
        a[0],
        a[1],
        a[2],
        a[3],
        b[0],
        b[1],
        c[0],
        c[1],
        c[2],
        c[3],
    )
    for register in T.unroll(4):
        wrap_output[lane, register] = wrap[register]
        sat_output[lane, register] = saturated[register]


def _exact_value(lane: int, register: int, element: int, seed: int, *, sparse: bool) -> float:
    if sparse:
        code = (lane * 11 + register * 7 + element * 13 + seed * 5) % 23
        return (code - 11) * 0.25
    code = (lane * 7 + register * 11 + element * 13 + seed * 5) % 19
    return (code - 9) * 0.25


def _b16_words(
    a_count: int, b_count: int, seed: int, *, sparse: bool
) -> tuple[np.ndarray, np.ndarray]:
    def word(lane: int, register: int, local_seed: int) -> np.uint32:
        low = np.float16(_exact_value(lane, register, 0, local_seed, sparse=sparse)).view(np.uint16)
        high = np.float16(_exact_value(lane, register, 1, local_seed + 3, sparse=sparse)).view(
            np.uint16
        )
        return np.uint32(int(low) | (int(high) << 16))

    a = np.array(
        [[word(lane, register, seed) for register in range(a_count)] for lane in range(32)],
        dtype=np.uint32,
    )
    b = np.array(
        [[word(lane, register, seed + 7) for register in range(b_count)] for lane in range(32)],
        dtype=np.uint32,
    )
    return a, b


def _c_values(seed: int, *, sparse: bool) -> np.ndarray:
    final_seed = 31 if sparse else 23
    return np.array(
        [
            [
                _exact_value(lane, register, seed, final_seed, sparse=sparse) * 0.125
                for register in range(4)
            ]
            for lane in range(32)
        ],
        dtype=np.float32,
    )


def _fp8_words(a_count: int, b_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    codes = (0x30, 0x34, 0x38, 0x3C, 0x40, 0xB4, 0xB8, 0xC0)

    def word(lane: int, register: int, local_seed: int) -> np.uint32:
        value = 0
        for byte in range(4):
            index = (lane * 5 + register * 3 + byte * 7 + local_seed) & 7
            value |= codes[index] << (8 * byte)
        return np.uint32(value)

    return (
        np.array(
            [[word(lane, register, seed) for register in range(a_count)] for lane in range(32)],
            dtype=np.uint32,
        ),
        np.array(
            [[word(lane, register, seed + 7) for register in range(b_count)] for lane in range(32)],
            dtype=np.uint32,
        ),
    )


def _metadata_2of4(seed: int) -> np.ndarray:
    codes = (0x4, 0x8, 0xC, 0x9, 0xD, 0xE)
    words = np.zeros(32, dtype=np.uint32)
    for lane in range(32):
        value = 0
        for nibble in range(8):
            value |= codes[(lane * 5 + nibble * 7 + seed) % 6] << (4 * nibble)
        words[lane] = np.uint32(value)
    return words


def make_dense_f16_case() -> dict[str, np.ndarray]:
    a, b = _b16_words(4, 2, 2, sparse=False)
    return {
        "a_words": a,
        "b_words": b,
        "c_values": _c_values(2, sparse=False),
        "output": np.zeros((32, 4), dtype=np.uint32),
    }


def make_dense_fp8_case() -> dict[str, np.ndarray]:
    a, b = _fp8_words(4, 2, 28)
    return {
        "a_words": a,
        "b_words": b,
        "c_values": _c_values(28, sparse=False),
        "output": np.zeros((32, 4), dtype=np.uint32),
    }


def make_sparse_f16_case() -> dict[str, np.ndarray]:
    a, b = _b16_words(4, 4, 4, sparse=True)
    return {
        "a_words": a,
        "b_words": b,
        "c_values": _c_values(4, sparse=True),
        "metadata_words": _metadata_2of4(4),
        "output": np.zeros((32, 4), dtype=np.uint32),
    }


def make_dense_u8_wrap_and_sat_case() -> dict[str, np.ndarray]:
    return {
        "wrap_output": np.zeros((32, 4), dtype=np.uint32),
        "sat_output": np.zeros((32, 4), dtype=np.uint32),
    }


@dataclass(frozen=True)
class MmaSyncCase:
    name: str
    prim_func: Any
    make_arguments: Callable[[], Mapping[str, Any]]
    outputs: tuple[str, ...]


MMA_SYNC_CASES = (
    MmaSyncCase(
        name="dense_f16_f32_k16",
        prim_func=raw_dense_f16_f32_k16,
        make_arguments=make_dense_f16_case,
        outputs=("output",),
    ),
    MmaSyncCase(
        name="dense_fp8_f32_k32",
        prim_func=raw_dense_fp8_f32_k32,
        make_arguments=make_dense_fp8_case,
        outputs=("output",),
    ),
    MmaSyncCase(
        name="sparse_f16_f32_k32",
        prim_func=raw_sparse_f16_f32_k32,
        make_arguments=make_sparse_f16_case,
        outputs=("output",),
    ),
    MmaSyncCase(
        name="dense_u8_wrap_and_sat",
        prim_func=raw_dense_u8_wrap_and_sat,
        make_arguments=make_dense_u8_wrap_and_sat_case,
        outputs=("wrap_output", "sat_output"),
    ),
)
