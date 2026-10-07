from __future__ import annotations

from tirx_harness.numsim.transpiler.frontend import analyze

import numpy as np

from tirx_harness import numsim
from tvm.script import tirx as T
from tvm.tirx.layout import S, TCol, TileLayout, TLane

_TMEM_8 = TileLayout(S[(128, 8) : (1 @ TLane, 1 @ TCol)])
_TMEM_16 = TileLayout(S[(128, 16) : (1 @ TLane, 1 @ TCol)])
_TMEM_32 = TileLayout(S[(128, 32) : (1 @ TLane, 1 @ TCol)])


@T.prim_func
def ptx_mma_tf32_m16n8k8(
    a: T.Buffer((16, 8), "float32"),
    b: T.Buffer((8, 8), "float32"),
    c: T.Buffer((16, 8), "float32"),
    output: T.Buffer((16, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((4,), "float32")
    b_regs = T.alloc_local((2,), "float32")
    c_regs = T.alloc_local((4,), "float32")
    d_regs = T.alloc_local((4,), "float32")

    a_regs[0] = a[group, thread]
    a_regs[1] = a[group + 8, thread]
    a_regs[2] = a[group, thread + 4]
    a_regs[3] = a[group + 8, thread + 4]
    b_regs[0] = b[thread, group]
    b_regs[1] = b[thread + 4, group]
    c_regs[0] = c[group, thread * 2]
    c_regs[1] = c[group, thread * 2 + 1]
    c_regs[2] = c[group + 8, thread * 2]
    c_regs[3] = c[group + 8, thread * 2 + 1]
    a_words = a_regs.view("uint32")
    b_words = b_regs.view("uint32")

    T.ptx.mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32(
        d_regs[0],
        d_regs[1],
        d_regs[2],
        d_regs[3],
        a_words[0],
        a_words[1],
        a_words[2],
        a_words[3],
        b_words[0],
        b_words[1],
        c_regs[0],
        c_regs[1],
        c_regs[2],
        c_regs[3],
    )

    output[group, thread * 2] = d_regs[0]
    output[group, thread * 2 + 1] = d_regs[1]
    output[group + 8, thread * 2] = d_regs[2]
    output[group + 8, thread * 2 + 1] = d_regs[3]


@T.prim_func
def ptx_mma_f16_m16n8k16(
    a: T.Buffer((16, 16), "float16"),
    b: T.Buffer((16, 8), "float16"),
    c: T.Buffer((16, 8), "float32"),
    output: T.Buffer((16, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((8,), "float16")
    b_regs = T.alloc_local((4,), "float16")
    c_regs = T.alloc_local((4,), "float32")
    d_regs = T.alloc_local((4,), "float32")

    for register in T.unroll(4):
        row = T.meta_var(group + (register % 2) * 8)
        col = T.meta_var(thread * 2 + (register // 2) * 8)
        a_regs[register * 2] = a[row, col]
        a_regs[register * 2 + 1] = a[row, col + 1]
    for register in T.unroll(2):
        row = T.meta_var(thread * 2 + register * 8)
        b_regs[register * 2] = b[row, group]
        b_regs[register * 2 + 1] = b[row + 1, group]
    c_regs[0] = c[group, thread * 2]
    c_regs[1] = c[group, thread * 2 + 1]
    c_regs[2] = c[group + 8, thread * 2]
    c_regs[3] = c[group + 8, thread * 2 + 1]
    a_words = a_regs.view("uint32")
    b_words = b_regs.view("uint32")

    T.ptx.mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32(
        d_regs[0],
        d_regs[1],
        d_regs[2],
        d_regs[3],
        a_words[0],
        a_words[1],
        a_words[2],
        a_words[3],
        b_words[0],
        b_words[1],
        c_regs[0],
        c_regs[1],
        c_regs[2],
        c_regs[3],
    )

    output[group, thread * 2] = d_regs[0]
    output[group, thread * 2 + 1] = d_regs[1]
    output[group + 8, thread * 2] = d_regs[2]
    output[group + 8, thread * 2 + 1] = d_regs[3]


@T.prim_func
def ptx_mma_f16_m16n8k16_zero_c(
    a: T.Buffer((16, 16), "float16"),
    b: T.Buffer((16, 8), "float16"),
    output: T.Buffer((16, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((8,), "float16")
    b_regs = T.alloc_local((4,), "float16")
    d_regs = T.alloc_local((4,), "float32")

    for register in T.unroll(4):
        row = T.meta_var(group + (register % 2) * 8)
        col = T.meta_var(thread * 2 + (register // 2) * 8)
        a_regs[register * 2] = a[row, col]
        a_regs[register * 2 + 1] = a[row, col + 1]
    for register in T.unroll(2):
        row = T.meta_var(thread * 2 + register * 8)
        b_regs[register * 2] = b[row, group]
        b_regs[register * 2 + 1] = b[row + 1, group]
    a_words = a_regs.view("uint32")
    b_words = b_regs.view("uint32")

    T.ptx.mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32(
        d_regs[0],
        d_regs[1],
        d_regs[2],
        d_regs[3],
        a_words[0],
        a_words[1],
        a_words[2],
        a_words[3],
        b_words[0],
        b_words[1],
        T.float32(0.0),
        T.float32(0.0),
        T.float32(0.0),
        T.float32(0.0),
    )

    output[group, thread * 2] = d_regs[0]
    output[group, thread * 2 + 1] = d_regs[1]
    output[group + 8, thread * 2] = d_regs[2]
    output[group + 8, thread * 2 + 1] = d_regs[3]


@T.prim_func
def ptx_mma_f16_m16n8k16_mixed_c(output: T.Buffer((4, 32), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    d = T.alloc_local((4,), "float32")
    a = T.alloc_local((4,), "uint32")
    b = T.alloc_local((2,), "uint32")
    c = T.alloc_local((1,), "float32")
    for i in T.unroll(4):
        a[i] = T.uint32(0x3C003C00)
    for i in T.unroll(2):
        b[i] = T.uint32(0x40004000)
    c[0] = 5
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
        T.float32(0.0),
        T.float32(0.0),
        T.float32(0.0),
    )
    for i in T.unroll(4):
        output[i, lane] = d[i]


@T.prim_func
def ptx_mma_legacy_f16_m16n8k16(
    a: T.Buffer((16, 16), "float16"),
    b: T.Buffer((16, 8), "float16"),
    c: T.Buffer((16, 8), "float32"),
    output: T.Buffer((16, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((8,), "float16")
    b_regs = T.alloc_local((4,), "float16")
    accumulator = T.alloc_local((4,), "float32")

    for register in T.unroll(4):
        row = T.meta_var(group + (register % 2) * 8)
        col = T.meta_var(thread * 2 + (register // 2) * 8)
        a_regs[register * 2] = a[row, col]
        a_regs[register * 2 + 1] = a[row, col + 1]
    for register in T.unroll(2):
        row = T.meta_var(thread * 2 + register * 8)
        b_regs[register * 2] = b[row, group]
        b_regs[register * 2 + 1] = b[row + 1, group]
    accumulator[0] = c[group, thread * 2]
    accumulator[1] = c[group, thread * 2 + 1]
    accumulator[2] = c[group + 8, thread * 2]
    accumulator[3] = c[group + 8, thread * 2 + 1]

    T.ptx_legacy.mma(
        "m16n8k16",
        "row",
        "col",
        "float16",
        "float16",
        "float32",
        a_regs.data,
        0,
        b_regs.data,
        0,
        accumulator.data,
        0,
        False,
        dtype="float32",
    )

    output[group, thread * 2] = accumulator[0]
    output[group, thread * 2 + 1] = accumulator[1]
    output[group + 8, thread * 2] = accumulator[2]
    output[group + 8, thread * 2 + 1] = accumulator[3]


@T.prim_func
def mma_fragment_fill_and_store(
    filled: T.Buffer((2, 32, 8), "float32"),
    stored: T.Buffer((2, 16, 16), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    current_fragment = T.alloc_local((8,), "float32")
    legacy_fragment = T.alloc_local((8,), "float32")
    current_fill = T.alloc_local((8,), "float32")
    legacy_fill = T.alloc_local((8,), "float32")
    for local_id in T.serial(8):
        current_fragment[local_id] = T.cast(lane * 100 + local_id, "float32")
        legacy_fragment[local_id] = T.cast(10000 + lane * 100 + local_id, "float32")
        current_fill[local_id] = T.float32(1)
        legacy_fill[local_id] = T.float32(2)

    T.evaluate(T.cuda.mma_fill(8, current_fill.data, 0, dtype="float32"))
    T.evaluate(T.cuda.mma_fill_legacy(8, legacy_fill.data, 0, dtype="float32"))
    T.evaluate(
        T.cuda.mma_store(
            16,
            16,
            stored.ptr_to([0, 0, 0]),
            current_fragment.data,
            0,
            16,
            dtype="float32",
        )
    )
    T.evaluate(
        T.cuda.mma_store_legacy(
            16,
            16,
            stored.ptr_to([1, 0, 0]),
            legacy_fragment.data,
            0,
            16,
            dtype="float32",
        )
    )
    for local_id in T.serial(8):
        filled[0, lane, local_id] = current_fill[local_id]
        filled[1, lane, local_id] = legacy_fill[local_id]


@T.prim_func
def ptx_mma_sp_f16_m16n8k16(
    packed_a: T.Buffer((16, 8), "float16"),
    b: T.Buffer((16, 8), "float16"),
    c: T.Buffer((16, 8), "float32"),
    metadata_words: T.Buffer((32,), "uint32"),
    output: T.Buffer((16, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((4,), "float16")
    b_regs = T.alloc_local((4,), "float16")
    acc = T.alloc_local((4,), "float32")
    metadata = T.alloc_local((1,), "uint32")

    a_regs[0] = packed_a[group, thread * 2]
    a_regs[1] = packed_a[group, thread * 2 + 1]
    a_regs[2] = packed_a[group + 8, thread * 2]
    a_regs[3] = packed_a[group + 8, thread * 2 + 1]
    b_regs[0] = b[thread * 2, group]
    b_regs[1] = b[thread * 2 + 1, group]
    b_regs[2] = b[thread * 2 + 8, group]
    b_regs[3] = b[thread * 2 + 9, group]
    acc[0] = c[group, thread * 2]
    acc[1] = c[group, thread * 2 + 1]
    acc[2] = c[group + 8, thread * 2]
    acc[3] = c[group + 8, thread * 2 + 1]
    metadata[0] = metadata_words[lane]
    a_words = a_regs.view("uint32")
    b_words = b_regs.view("uint32")

    T.ptx.mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32(
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        a_words[0],
        a_words[1],
        b_words[0],
        b_words[1],
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


def _tf32(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    finite = (bits & np.uint32(0x7F800000)) != np.uint32(0x7F800000)
    rounded = bits.copy()
    bias = np.uint32(0xFFF) + ((bits >> np.uint32(13)) & np.uint32(1))
    rounded[finite] = (bits[finite] + bias[finite]) & np.uint32(0xFFFFE000)
    return rounded.view(np.float32)


def _sparse_metadata_words(codes: np.ndarray, selector: int) -> np.ndarray:
    words = np.full((32,), np.uint32(0x44444444), dtype=np.uint32)
    for group in range(8):
        word = 0
        for row in (group, group + 8):
            for chunk in range(4):
                code_index = 4 * (row // 8) + chunk
                word |= int(codes[row, chunk]) << (4 * code_index)
        words[4 * group + selector] = np.uint32(word)
    return words


def _expand_sparse_m16n8k16(packed: np.ndarray, codes: np.ndarray) -> np.ndarray:
    dense = np.zeros((16, 16), dtype=np.float32)
    for row in range(16):
        for chunk in range(4):
            code = int(codes[row, chunk])
            first = code & 0x3
            second = (code >> 2) & 0x3
            dense[row, chunk * 4 + first] = packed[row, chunk * 2]
            dense[row, chunk * 4 + second] = packed[row, chunk * 2 + 1]
    return dense


def test_matrix_family_supports_dense_and_sparse_ptx():
    assert analyze(ptx_mma_tf32_m16n8k8).unsupported == ()
    assert analyze(ptx_mma_sp_f16_m16n8k16).unsupported == ()


def test_mma_fragment_fill_and_store_observe_lane_register_layout(tmp_path):
    module = numsim.transpile(mma_fragment_fill_and_store, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "filled": np.ones((2, 32, 8), dtype=np.float32),
            "stored": np.zeros((2, 16, 16), dtype=np.float32),
        },
    )

    np.testing.assert_array_equal(result.outputs["filled"], np.zeros((2, 32, 8), np.float32))
    expected = np.empty((2, 16, 16), dtype=np.float32)
    for row in range(16):
        for column in range(16):
            lane = 4 * (row % 8) + (column % 8) // 2
            local_id = 4 * (column // 8) + 2 * (row // 8) + column % 2
            expected[0, row, column] = lane * 100 + local_id
            expected[1, row, column] = 10000 + lane * 100 + local_id
    np.testing.assert_array_equal(result.outputs["stored"], expected)


def test_ptx_mma_tf32_observes_lane_register_abi(tmp_path):
    a = (np.arange(16 * 8, dtype=np.float32).reshape(16, 8) % 7 - 3) / 4
    b = (np.arange(8 * 8, dtype=np.float32).reshape(8, 8) % 5 - 2) / 2
    c = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 16
    module = numsim.transpile(ptx_mma_tf32_m16n8k8, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"a": a, "b": b, "c": c, "output": np.zeros((16, 8), dtype=np.float32)}
    )
    expected = _tf32(a) @ _tf32(b) + c
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_ptx_mma_f16_observes_packed_b16_register_abi(tmp_path):
    a = (np.arange(16 * 16, dtype=np.float16).reshape(16, 16) % 5) / np.float16(2)
    b = (np.arange(16 * 8, dtype=np.float16).reshape(16, 8) % 7) / np.float16(4)
    c = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 32
    module = numsim.transpile(ptx_mma_f16_m16n8k16, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"a": a, "b": b, "c": c, "output": np.zeros((16, 8), dtype=np.float32)}
    )
    expected = a.astype(np.float32) @ b.astype(np.float32) + c
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_ptx_mma_f16_literal_zero_c_uses_no_c_engine_abi(tmp_path):
    a = (np.arange(16 * 16, dtype=np.float16).reshape(16, 16) % 5) / np.float16(2)
    b = (np.arange(16 * 8, dtype=np.float16).reshape(16, 8) % 7) / np.float16(4)
    module = numsim.transpile(ptx_mma_f16_m16n8k16_zero_c, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"a": a, "b": b, "output": np.zeros((16, 8), dtype=np.float32)}
    )
    expected = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_ptx_mma_mixed_accumulator_and_literal_zero_c(tmp_path):
    module = numsim.transpile(ptx_mma_f16_m16n8k16_mixed_c, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros((4, 32), np.float32)})
    expected = np.full((4, 32), 32, np.float32)  # 16 products of 1 * 2.
    expected[0] += 5
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_ptx_mma_legacy_executes_actual_pointer_offset_abi(tmp_path):
    a = (np.arange(16 * 16, dtype=np.float16).reshape(16, 16) % 5) / np.float16(2)
    b = (np.arange(16 * 8, dtype=np.float16).reshape(16, 8) % 7) / np.float16(4)
    c = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 32
    module = numsim.transpile(ptx_mma_legacy_f16_m16n8k16, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"a": a, "b": b, "c": c, "output": np.zeros((16, 8), dtype=np.float32)}
    )
    expected = a.astype(np.float32) @ b.astype(np.float32) + c
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_ptx_mma_sp_selector1_expands_nonuniform_metadata(tmp_path):
    packed_a = (np.arange(16 * 8, dtype=np.float16).reshape(16, 8) % 5) / np.float16(2)
    b = (np.arange(16 * 8, dtype=np.float16).reshape(16, 8) % 7) / np.float16(4)
    c = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 32
    valid_codes = np.array([0x4, 0x8, 0xC, 0x9, 0xD, 0xE], dtype=np.uint8)
    codes = valid_codes[(np.arange(16)[:, None] * 5 + np.arange(4)[None, :]) % 6]
    metadata_words = _sparse_metadata_words(codes, selector=1)
    dense_a = _expand_sparse_m16n8k16(packed_a, codes)
    module = numsim.transpile(ptx_mma_sp_f16_m16n8k16, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "packed_a": packed_a,
            "b": b,
            "c": c,
            "metadata_words": metadata_words,
            "output": np.zeros((16, 8), dtype=np.float32),
        },
    )
    expected = dense_a @ b.astype(np.float32) + c
    np.testing.assert_array_equal(result.outputs["output"], expected)
