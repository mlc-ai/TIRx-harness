from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm_tirx.script import tirx as T


@T.function
def vector_ir_roundtrip(
    source_u32: T.Tensor((64,), "uint32"),
    source_f32: T.Tensor((128,), "float32"),
    output_u32: T.Tensor((64,), "uint32"),
    output_f32: T.Tensor((128,), "float32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((128,), "float32")
    for slot in T.unroll(4):
        shared[lane * 4 + slot] = source_f32[lane * 4 + slot]
    T.cuda.warp_sync()
    packed = source_u32[T.Ramp(lane * 2, 1, 2)]
    values = shared[T.Ramp(lane * 4, 1, 4)]
    output_u32[lane * 2] = T.Shuffle([packed], [0])
    output_u32[lane * 2 + 1] = T.Shuffle([packed], [1])
    output_f32[lane * 4] = T.Shuffle([values], [0])
    output_f32[lane * 4 + 1] = T.Shuffle([values], [1])
    output_f32[lane * 4 + 2] = T.Shuffle([values], [2])
    output_f32[lane * 4 + 3] = T.Shuffle([values], [3])


@T.function
def invalid_vector_shuffle_extract(
    source: T.Tensor((64,), "uint32"), output: T.Tensor((32,), "uint32")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    packed = source[T.Ramp(lane * 2, 1, 2)]
    output[lane] = T.Shuffle([packed], [2])


@T.function
def packed_vector_arithmetic_is_not_scalar_arithmetic(
    lhs: T.Tensor((32,), "int8x4"),
    rhs: T.Tensor((32,), "int8x4"),
    output: T.Tensor((32,), "int8x4"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    output[lane] = lhs[lane] + rhs[lane]


@T.function
def unsupported_generic_vector_extract(
    source: T.Tensor((128,), "int8"), output: T.Tensor((32,), "int8")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    packed = source[T.Ramp(lane * 4, 1, 4)]
    output[lane] = T.Shuffle([packed], [0])


@T.function
def reinterpret_128bit_vector_roundtrip(
    source: T.Tensor((32,), "uint64x2"), output: T.Tensor((32,), "uint64x2")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    as_float: T.let = T.reinterpret(source[lane], ty="float32x4")
    output[lane] = T.reinterpret(as_float, ty="uint64x2")


@T.function
def construct_uint32x2_bits(
    low: T.Tensor((32,), "uint32"),
    high: T.Tensor((32,), "uint32"),
    output: T.Tensor((32,), "uint64"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    output[lane] = T.reinterpret(T.Shuffle([low[lane], high[lane]], [0, 1]), ty="uint64")


def test_vector_ir_roundtrip_executes_packed_and_f32x4_semantics(tmp_path):
    source_u32 = (np.arange(64, dtype=np.uint32) * np.uint32(0x01020305)) ^ np.uint32(0xA55AA55A)
    source_f32 = np.arange(128, dtype=np.float32) * np.float32(0.125) - np.float32(3)
    output_u32 = np.zeros_like(source_u32)
    output_f32 = np.zeros_like(source_f32)

    spec = analyze(vector_ir_roundtrip)
    assert spec.unsupported == ()
    module = numsim.transpile(vector_ir_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source_u32": source_u32,
            "source_f32": source_f32,
            "output_u32": output_u32,
            "output_f32": output_f32,
        },
    )

    np.testing.assert_array_equal(result.outputs["output_u32"], source_u32)
    np.testing.assert_array_equal(result.outputs["output_f32"], source_f32)
    # A Bind keeps the float32x4 value as one four-lane register value.
    assert "loaded_f32x4" in module.rust_source
    assert "packed_u32x2" in module.rust_source


def test_vector_shuffle_classifier_and_frontend_reject_out_of_range_extract():
    spec = analyze(invalid_vector_shuffle_extract)
    assert any("outside uint32x2 lane range" in item for item in spec.unsupported)


def test_packed_storage_vectors_fail_closed_for_ordinary_arithmetic():
    spec = analyze(packed_vector_arithmetic_is_not_scalar_arithmetic)

    assert any("packed vector storage" in item for item in spec.unsupported)


def test_generic_vector_extract_fails_during_analysis():
    spec = analyze(unsupported_generic_vector_extract)

    assert any("only a packed storage ABI" in item for item in spec.unsupported)


def test_128bit_vector_reinterpret_is_a_bitwise_roundtrip(tmp_path):
    source = np.arange(64, dtype=np.uint64).reshape(32, 2).view(np.dtype("V16")).reshape(32)
    output = np.zeros(32, dtype=np.dtype("V16"))

    module = numsim.transpile(reinterpret_128bit_vector_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "output": output,
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], source)
    assert "f32::from_bits" in module.rust_source
    assert ".to_bits()" in module.rust_source
    assert " as [f32; 4]" not in module.rust_source


def test_uint32x2_construction_preserves_low_and_high_words(tmp_path):
    low = np.arange(32, dtype=np.uint32) * np.uint32(0x1020304) + np.uint32(0xF0000001)
    high = np.arange(32, dtype=np.uint32) * np.uint32(0xA0B0C0D) ^ np.uint32(0x89ABCDEF)
    result = numsim.Engine().run(
        numsim.transpile(construct_uint32x2_bits, cache_dir=tmp_path),
        {
            "low": low,
            "high": high,
            "output": np.zeros(32, dtype=np.uint64),
        },
    )

    expected = low.astype(np.uint64) | (high.astype(np.uint64) << np.uint64(32))
    np.testing.assert_array_equal(result.outputs["output"], expected)
