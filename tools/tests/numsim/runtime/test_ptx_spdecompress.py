from __future__ import annotations

import numpy as np
import pytest
import tvm

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def ptx_spdecompress_b8_b4_2_4_x2(
    metadata: T.Buffer((32,), "uint32"),
    compressed: T.Buffer((32,), "uint32"),
    output: T.Buffer((2, 32), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["spdecompress.b8.b4.sp::2:4.x2"](
        output[0, lane], output[1, lane], metadata[lane], compressed[lane]
    )


@T.prim_func
def ptx_spdecompress_invalid_index(
    metadata: T.Buffer((32,), "uint32"),
    compressed: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["spdecompress.b8.b4.sp::1:2.x2"](output[lane], metadata[lane], compressed[lane])


def _reference(
    metadata: np.ndarray,
    compressed: np.ndarray,
    *,
    elem_bits: int,
    index_bits: int,
    src: int,
    dst: int,
    num: int,
) -> np.ndarray:
    output_registers = (dst * elem_bits * num + 31) // 32
    output = np.zeros((output_registers, 32), dtype=np.uint32)
    element_mask = (1 << elem_bits) - 1
    index_mask = (1 << index_bits) - 1
    for lane in range(32):
        words = [0] * output_registers
        for repetition in range(num):
            for source in range(src):
                packed_source = repetition * src + source
                metadata_bit = packed_source * index_bits
                destination = (
                    int(metadata[metadata_bit // 32, lane]) >> (metadata_bit % 32)
                ) & index_mask
                assert destination < dst
                compressed_bit = packed_source * elem_bits
                value = (
                    int(compressed[compressed_bit // 32, lane]) >> (compressed_bit % 32)
                ) & element_mask
                data_bit = (repetition * dst + destination) * elem_bits
                shift = data_bit % 32
                words[data_bit // 32] &= ~(element_mask << shift)
                words[data_bit // 32] |= value << shift
        output[:, lane] = words
    return output


def test_ptx_spdecompress_matches_low_bit_first_sparse_scatter(tmp_path):
    metadata = np.empty((1, 32), dtype=np.uint32)
    compressed = np.empty((1, 32), dtype=np.uint32)
    for lane in range(32):
        indices = (lane % 4, (lane + 2) % 4, (lane + 1) % 4, (lane + 3) % 4)
        if lane == 0:
            indices = (1, 1, 2, 0)  # Duplicate indices are overwritten by the later source.
        metadata[0, lane] = sum(index << (4 * slot) for slot, index in enumerate(indices))
        values = tuple((0x31 + 17 * lane + 29 * slot) & 0xFF for slot in range(4))
        compressed[0, lane] = sum(value << (8 * slot) for slot, value in enumerate(values))

    result = numsim.Engine().run(
        numsim.transpile(ptx_spdecompress_b8_b4_2_4_x2, cache_dir=tmp_path),
        {
            "metadata": metadata[0],
            "compressed": compressed[0],
            "output": np.full((2, 32), np.uint32(0xDEAD_BEEF), dtype=np.uint32),
        },
        outputs=("output",),
    )
    expected = _reference(
        metadata,
        compressed,
        elem_bits=8,
        index_bits=4,
        src=2,
        dst=4,
        num=2,
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert result.outputs["output"][0, 0] == 0x0000_4E00


def _maximum_output_form():
    outputs = ", ".join(f"output[{index}, lane]" for index in range(128))
    metadata = ", ".join(f"metadata[{index}, lane]" for index in range(4))
    compressed = ", ".join(f"compressed[{index}, lane]" for index in range(16))
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(
    metadata: T.Buffer((4, 32), "uint32"),
    compressed: T.Buffer((16, 32), "uint32"),
    output: T.Buffer((128, 32), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["spdecompress.b16.b4.sp::2:16.x16"]({outputs}, {metadata}, {compressed})
""",
        {"T": T},
    )


def test_ptx_spdecompress_executes_the_128_register_output_boundary(tmp_path):
    metadata = np.zeros((4, 32), dtype=np.uint32)
    compressed = np.empty((16, 32), dtype=np.uint32)
    for repetition in range(16):
        low = (0x1000 + repetition) & 0xFFFF
        high = (0xA000 + repetition) & 0xFFFF
        compressed[repetition, :] = low | (high << 16)
    result = numsim.Engine().run(
        numsim.transpile(_maximum_output_form(), cache_dir=tmp_path),
        {
            "metadata": metadata,
            "compressed": compressed,
            "output": np.full((128, 32), np.uint32(0xDEAD_BEEF), dtype=np.uint32),
        },
        outputs=("output",),
    )
    expected = _reference(
        metadata,
        compressed,
        elem_bits=16,
        index_bits=4,
        src=2,
        dst=16,
        num=16,
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert result.outputs["output"][120, 0] == 0x0000_A00F
    assert result.outputs["output"][127, 0] == 0


def test_ptx_spdecompress_rejects_metadata_index_at_or_above_dst(tmp_path):
    with pytest.raises(numsim.NumSimExecutionError, match=r"metadata index 2 is outside 0\.\.2"):
        numsim.Engine().run(
            numsim.transpile(ptx_spdecompress_invalid_index, cache_dir=tmp_path),
            {
                "metadata": np.full(32, np.uint32(2), dtype=np.uint32),
                "compressed": np.full(32, np.uint32(0x2211), dtype=np.uint32),
                "output": np.zeros(32, dtype=np.uint32),
            },
            outputs=("output",),
        )


__all__ = ["ptx_spdecompress_b8_b4_2_4_x2", "ptx_spdecompress_invalid_index"]
