from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def sparse_f16_m16n8k32(
    packed_a: T.Buffer((16, 16), "float16"),
    b: T.Buffer((32, 8), "float16"),
    c: T.Buffer((16, 8), "float32"),
    metadata_words: T.Buffer((32,), "uint32"),
    output: T.Buffer((16, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((8,), "float16")
    b_regs = T.alloc_local((8,), "float16")
    acc = T.alloc_local((4,), "float32")
    metadata = T.alloc_local((1,), "uint32")
    a_packed = a_regs.view("uint32")
    b_packed = b_regs.view("uint32")
    for slot in T.unroll(8):
        half = T.meta_var(slot // 4)
        within = T.meta_var(slot % 4)
        row = T.meta_var(group + (within // 2) * 8)
        a_regs[slot] = packed_a[row, (thread + 4 * half) * 2 + within % 2]
        inner = T.meta_var(8 * (slot // 2) + 2 * thread + slot % 2)
        b_regs[slot] = b[inner, group]
    acc[0] = c[group, thread * 2]
    acc[1] = c[group, thread * 2 + 1]
    acc[2] = c[group + 8, thread * 2]
    acc[3] = c[group + 8, thread * 2 + 1]
    metadata[0] = metadata_words[lane]
    T.ptx["mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"](
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        a_packed[0],
        a_packed[1],
        a_packed[2],
        a_packed[3],
        b_packed[0],
        b_packed[1],
        b_packed[2],
        b_packed[3],
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        metadata[0],
        1,
    )
    output[group, thread * 2] = acc[0]
    output[group, thread * 2 + 1] = acc[1]
    output[group + 8, thread * 2] = acc[2]
    output[group + 8, thread * 2 + 1] = acc[3]


@T.prim_func
def sparse_tf32_m16n8k8(
    packed_a: T.Buffer((16, 4), "float32"),
    b: T.Buffer((8, 8), "float32"),
    c: T.Buffer((16, 8), "float32"),
    metadata_words: T.Buffer((32,), "uint32"),
    output: T.Buffer((16, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((2,), "float32")
    b_regs = T.alloc_local((2,), "float32")
    acc = T.alloc_local((4,), "float32")
    metadata = T.alloc_local((1,), "uint32")
    a_packed = a_regs.view("uint32")
    b_packed = b_regs.view("uint32")
    a_regs[0] = packed_a[group, thread]
    a_regs[1] = packed_a[group + 8, thread]
    b_regs[0] = b[thread, group]
    b_regs[1] = b[thread + 4, group]
    acc[0] = c[group, thread * 2]
    acc[1] = c[group, thread * 2 + 1]
    acc[2] = c[group + 8, thread * 2]
    acc[3] = c[group + 8, thread * 2 + 1]
    metadata[0] = metadata_words[lane]
    T.ptx["mma.sp.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"](
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        a_packed[0],
        a_packed[1],
        b_packed[0],
        b_packed[1],
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        metadata[0],
        2,
    )
    output[group, thread * 2] = acc[0]
    output[group, thread * 2 + 1] = acc[1]
    output[group + 8, thread * 2] = acc[2]
    output[group + 8, thread * 2 + 1] = acc[3]


@T.prim_func
def sparse_s8_u8_m16n8k64(
    packed_a: T.Buffer((16, 32), "int8"),
    b: T.Buffer((64, 8), "uint8"),
    c: T.Buffer((16, 8), "int32"),
    metadata_words: T.Buffer((32,), "uint32"),
    output: T.Buffer((16, 8), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((16,), "int8")
    b_regs = T.alloc_local((16,), "uint8")
    acc = T.alloc_local((4,), "int32")
    metadata = T.alloc_local((1,), "uint32")
    a_packed = a_regs.view("uint32")
    b_packed = b_regs.view("uint32")
    acc_packed = acc.view("uint32")
    for slot in T.unroll(16):
        half = T.meta_var(slot // 8)
        within = T.meta_var(slot % 8)
        row = T.meta_var(group + (within // 4) * 8)
        a_regs[slot] = packed_a[row, half * 16 + thread * 4 + within % 4]
        inner = T.meta_var(16 * (slot // 4) + 4 * thread + slot % 4)
        b_regs[slot] = b[inner, group]
    acc[0] = c[group, thread * 2]
    acc[1] = c[group, thread * 2 + 1]
    acc[2] = c[group + 8, thread * 2]
    acc[3] = c[group + 8, thread * 2 + 1]
    metadata[0] = metadata_words[lane]
    T.ptx["mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.u8.s32"](
        acc_packed[0],
        acc_packed[1],
        acc_packed[2],
        acc_packed[3],
        a_packed[0],
        a_packed[1],
        a_packed[2],
        a_packed[3],
        b_packed[0],
        b_packed[1],
        b_packed[2],
        b_packed[3],
        acc_packed[0],
        acc_packed[1],
        acc_packed[2],
        acc_packed[3],
        metadata[0],
        0,
    )
    output[group, thread * 2] = acc[0]
    output[group, thread * 2 + 1] = acc[1]
    output[group + 8, thread * 2] = acc[2]
    output[group + 8, thread * 2 + 1] = acc[3]


@T.prim_func
def sparse_float8_m16n8k64_zero(output: T.Buffer((32, 4), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    a_regs = T.alloc_local((16,), "float8_e4m3fn")
    b_regs = T.alloc_local((16,), "float8_e4m3fn")
    acc = T.alloc_local((4,), "float32")
    metadata = T.alloc_local((1,), "uint32")
    a_packed = a_regs.view("uint32")
    b_packed = b_regs.view("uint32")
    for index in T.unroll(16):
        a_regs[index] = T.cast(T.float32(0), "float8_e4m3fn")
        b_regs[index] = T.cast(T.float32(0), "float8_e4m3fn")
    for index in T.unroll(4):
        acc[index] = T.float32(0)
    metadata[0] = T.uint32(0x44444444)
    T.ptx["mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32"](
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        a_packed[0],
        a_packed[1],
        a_packed[2],
        a_packed[3],
        b_packed[0],
        b_packed[1],
        b_packed[2],
        b_packed[3],
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        metadata[0],
        0,
    )
    for index in T.unroll(4):
        output[lane, index] = acc[index]


def _metadata_words(codes: np.ndarray, selector: int) -> np.ndarray:
    rows, chunks = codes.shape
    assert rows == 16 and chunks in {4, 8, 16}
    words = np.full((32,), np.uint32(0x44444444), dtype=np.uint32)
    for group in range(8):
        if chunks == 4:
            word = 0
            for row_high in range(2):
                for chunk in range(4):
                    word |= int(codes[group + 8 * row_high, chunk]) << (4 * (4 * row_high + chunk))
            words[4 * group + selector] = np.uint32(word)
        elif chunks == 8:
            for half in range(2):
                word = sum(
                    int(codes[group + 8 * row_high, 4 * half + chunk])
                    << (4 * (4 * row_high + chunk))
                    for row_high in range(2)
                    for chunk in range(4)
                )
                words[4 * group + 2 * selector + half] = np.uint32(word)
        else:
            for quarter in range(4):
                word = sum(
                    int(codes[group + 8 * row_high, 4 * quarter + chunk])
                    << (4 * (4 * row_high + chunk))
                    for row_high in range(2)
                    for chunk in range(4)
                )
                words[4 * group + quarter] = np.uint32(word)
    return words


def _expand_2of4(packed: np.ndarray, codes: np.ndarray) -> np.ndarray:
    dense = np.zeros((packed.shape[0], packed.shape[1] * 2), dtype=packed.dtype)
    for row in range(packed.shape[0]):
        for chunk in range(codes.shape[1]):
            code = int(codes[row, chunk])
            dense[row, 4 * chunk + (code & 3)] = packed[row, 2 * chunk]
            dense[row, 4 * chunk + ((code >> 2) & 3)] = packed[row, 2 * chunk + 1]
    return dense


def _expand_tf32(packed: np.ndarray, codes: np.ndarray) -> np.ndarray:
    dense = np.zeros((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
    for row in range(packed.shape[0]):
        for chunk in range(codes.shape[1]):
            position = 0 if int(codes[row, chunk]) == 0x4 else 1
            dense[row, 2 * chunk + position] = packed[row, chunk]
    return dense


def test_sparse_float8_k64_matches_reference(tmp_path):
    module = numsim.transpile(sparse_float8_m16n8k64_zero, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.ones((32, 4), dtype=np.float32)})

    np.testing.assert_array_equal(result.outputs["output"], np.zeros((32, 4), dtype=np.float32))


def test_sparse_f16_k32_uses_thread_pair_metadata(tmp_path):
    packed_a = (np.arange(16 * 16, dtype=np.float16).reshape(16, 16) % 5 - 2).astype(np.float16)
    b = (np.arange(32 * 8, dtype=np.float16).reshape(32, 8) % 7 - 3).astype(np.float16)
    c = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 16
    valid = np.array([0x4, 0x8, 0xC, 0x9, 0xD, 0xE], dtype=np.uint8)
    codes = valid[(np.arange(16)[:, None] * 3 + np.arange(8)[None, :]) % len(valid)]
    module = numsim.transpile(sparse_f16_m16n8k32, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "packed_a": packed_a,
            "b": b,
            "c": c,
            "metadata_words": _metadata_words(codes, 1),
            "output": np.zeros_like(c),
        },
    )
    expected = _expand_2of4(packed_a, codes).astype(np.float32) @ b.astype(np.float32) + c
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_sparse_tf32_k8_uses_one_of_two_metadata(tmp_path):
    packed_a = (np.arange(16 * 4, dtype=np.float32).reshape(16, 4) % 5 - 2).astype(np.float32)
    b = (np.arange(8 * 8, dtype=np.float32).reshape(8, 8) % 5 - 2).astype(np.float32)
    c = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 16
    codes = np.where((np.arange(16)[:, None] + np.arange(4)[None, :]) % 2, 0xE, 0x4).astype(
        np.uint8
    )
    module = numsim.transpile(sparse_tf32_m16n8k8, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "packed_a": packed_a,
            "b": b,
            "c": c,
            "metadata_words": _metadata_words(codes, 2),
            "output": np.zeros_like(c),
        },
    )
    expected = _expand_tf32(packed_a, codes) @ b + c
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_sparse_int8_k64_uses_all_thread_metadata(tmp_path):
    packed_a = (np.arange(16 * 32, dtype=np.int16).reshape(16, 32) % 15 - 7).astype(np.int8)
    b = (np.arange(64 * 8, dtype=np.uint16).reshape(64, 8) % 11).astype(np.uint8)
    c = (np.arange(16 * 8, dtype=np.int64).reshape(16, 8) - 200).astype(np.int32)
    valid = np.array([0x4, 0x8, 0xC, 0x9, 0xD, 0xE], dtype=np.uint8)
    codes = valid[(np.arange(16)[:, None] * 5 + np.arange(16)[None, :]) % len(valid)]
    module = numsim.transpile(sparse_s8_u8_m16n8k64, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "packed_a": packed_a,
            "b": b,
            "c": c,
            "metadata_words": _metadata_words(codes, 0),
            "output": np.zeros_like(c),
        },
    )
    expected = (
        _expand_2of4(packed_a, codes).astype(np.int64) @ b.astype(np.int64) + c.astype(np.int64)
    ).astype(np.int32)
    np.testing.assert_array_equal(result.outputs["output"], expected)
