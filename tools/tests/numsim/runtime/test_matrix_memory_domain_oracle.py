from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def ptx_ldmatrix_x1_x2_domain(output: T.Buffer((96,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    x1 = T.alloc_local((1,), "uint32")
    x2 = T.alloc_local((2,), "uint32")
    for byte in T.unroll(16):
        shared[lane * 16 + byte] = T.cast(lane * 16 + byte, "uint8")
    T.cuda.warp_sync()
    T.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(x1[0], shared.ptr_to([lane * 16]))
    T.ptx.ldmatrix.sync.aligned.m8n8.x2.shared.b16(x2[0], x2[1], shared.ptr_to([lane * 16]))
    output[lane] = x1[0]
    for matrix in T.unroll(2):
        output[32 + matrix * 32 + lane] = x2[matrix]


@T.prim_func
def legacy_ldmatrix_x1_domain(output: T.Buffer((64,), "uint16")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((256,), "uint16", scope="shared")
    local = T.alloc_buffer((2,), "uint16", scope="local")
    for element in T.unroll(8):
        shared[lane * 8 + element] = T.cast(lane * 8 + element + 1, "uint16")
    T.cuda.warp_sync()
    T.evaluate(
        T.ptx_legacy.ldmatrix(
            False,
            1,
            ".b16",
            local.data,
            0,
            shared.data,
            0,
            dtype="uint16",
        )
    )
    output[lane * 2] = local[0]
    output[lane * 2 + 1] = local[1]


@T.prim_func
def stmatrix_b16_x1_domain(output: T.Buffer((64,), "uint16")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8, 8), "uint16", scope="shared")
    source = T.alloc_buffer((1,), "uint32", scope="local")
    low = T.cast(lane * 2 + 1, "uint32")
    high = T.cast(lane * 2 + 2, "uint32")
    source[0] = low | T.shift_left(high, T.uint32(16))
    T.ptx.stmatrix.sync.aligned.m8n8.x1.shared__cta.b16(
        T.address_of(shared[lane % 8, 0]),
        source[0],
    )
    T.cuda.warp_sync()
    for index in T.unroll(2):
        linear = lane * 2 + index
        output[linear] = shared[linear // 8, linear % 8]


@T.prim_func
def stmatrix_b8_x4_domain(output: T.Buffer((512,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32, 16), "uint8", scope="shared")
    source = T.alloc_buffer((4,), "uint32", scope="local")
    for matrix in T.unroll(4):
        value0 = T.cast((matrix * 73 + lane * 5 + 3) % 256, "uint32")
        value1 = T.cast((matrix * 73 + lane * 5 + 20) % 256, "uint32")
        value2 = T.cast((matrix * 73 + lane * 5 + 37) % 256, "uint32")
        value3 = T.cast((matrix * 73 + lane * 5 + 54) % 256, "uint32")
        source[matrix] = (
            value0
            | T.shift_left(value1, T.uint32(8))
            | T.shift_left(value2, T.uint32(16))
            | T.shift_left(value3, T.uint32(24))
        )
    T.ptx.stmatrix.sync.aligned.m16n8.x4.trans.shared.b8(
        T.address_of(shared[lane, 0]),
        source[0],
        source[1],
        source[2],
        source[3],
    )
    T.cuda.warp_sync()
    for index in T.unroll(16):
        linear = lane * 16 + index
        output[linear] = shared[linear // 16, linear % 16]


def _packed_shared_bytes(*offsets: int) -> np.uint32:
    return np.uint32(int.from_bytes(bytes(offset & 0xFF for offset in offsets), "little"))


def test_ptx_ldmatrix_x1_x2_domain_matches_independent_lane_byte_mapping(tmp_path):
    module = numsim.transpile(ptx_ldmatrix_x1_x2_domain, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(96, dtype=np.uint32)})

    expected = np.empty(96, dtype=np.uint32)
    for lane in range(32):
        row = lane // 4
        fragment = lane % 4
        source_lane0 = fragment * 2
        source_lane1 = source_lane0 + 1
        low = source_lane0 * 16 + row * 2
        high = source_lane1 * 16 + row * 2
        expected[lane] = _packed_shared_bytes(low, low + 1, high, high + 1)
        for matrix in range(2):
            provider_lane = matrix * 8 + row
            start = provider_lane * 16 + fragment * 4
            expected[32 + matrix * 32 + lane] = _packed_shared_bytes(
                start, start + 1, start + 2, start + 3
            )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_legacy_ldmatrix_x1_domain_matches_independent_fragment_mapping(tmp_path):
    module = numsim.transpile(legacy_ldmatrix_x1_domain, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(64, dtype=np.uint16)})

    expected = np.empty((32, 2), dtype=np.uint16)
    for lane in range(32):
        fragment = lane % 4
        expected[lane, 0] = fragment * 2 + 1
        expected[lane, 1] = fragment * 2 + 2
    np.testing.assert_array_equal(result.outputs["output"], expected.reshape(-1))


def test_stmatrix_b16_x1_domain_matches_independent_lane_half_mapping(tmp_path):
    module = numsim.transpile(stmatrix_b16_x1_domain, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(64, dtype=np.uint16)})

    expected = np.empty((8, 8), dtype=np.uint16)
    for source_lane in range(32):
        for half_index in range(2):
            row = source_lane // 4
            column = 2 * (source_lane % 4) + half_index
            expected[row, column] = source_lane * 2 + half_index + 1
    np.testing.assert_array_equal(result.outputs["output"], expected.reshape(-1))


def test_stmatrix_b8_x4_domain_matches_independent_lane_byte_mapping(tmp_path):
    module = numsim.transpile(stmatrix_b8_x4_domain, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(512, dtype=np.uint8)})

    expected = np.empty((4, 8, 16), dtype=np.uint8)
    for matrix in range(4):
        for source_lane in range(32):
            for byte_index in range(4):
                row = 2 * (source_lane % 4) + byte_index % 2
                column = source_lane // 4 + 8 * (byte_index // 2)
                expected[matrix, row, column] = (
                    matrix * 73 + source_lane * 5 + byte_index * 17 + 3
                ) % 256
    np.testing.assert_array_equal(result.outputs["output"], expected.reshape(-1))
