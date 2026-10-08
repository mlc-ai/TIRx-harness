from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.transpiler.frontend import analyze, verify
import tvm
from tvm import tirx
from tvm.script import tirx as T


@T.prim_func
def raw_stmatrix_layout(
    output_b8: T.Buffer((128,), "uint8"), output_b16: T.Buffer((256,), "uint16")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_b8 = T.alloc_buffer((8, 16), "uint8", scope="shared")
    shared_b16 = T.alloc_buffer((4, 8, 8), "uint16", scope="shared")
    source_b8 = T.alloc_buffer((1,), "uint32", scope="local")
    source_b16 = T.alloc_buffer((4,), "uint32", scope="local")
    source_b8[0] = (
        T.cast(lane * 4, "uint32")
        | T.shift_left(T.cast(lane * 4 + 1, "uint32"), T.uint32(8))
        | T.shift_left(T.cast(lane * 4 + 2, "uint32"), T.uint32(16))
        | T.shift_left(T.cast(lane * 4 + 3, "uint32"), T.uint32(24))
    )
    for matrix in T.unroll(0, 4):
        low = T.cast(matrix * 1000 + lane * 2, "uint32")
        high = T.cast(matrix * 1000 + lane * 2 + 1, "uint32")
        source_b16[matrix] = low | T.shift_left(high, T.uint32(16))
    T.ptx.stmatrix.sync.aligned.m16n8.x1.trans.shared.b8(
        T.address_of(shared_b8[lane % 8, 0]),
        source_b8[0],
    )
    T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
        T.address_of(shared_b16[lane // 8, lane % 8, 0]),
        source_b16[0],
        source_b16[1],
        source_b16[2],
        source_b16[3],
    )
    T.cuda.warp_sync()
    for index in T.unroll(0, 4):
        linear = lane * 4 + index
        output_b8[linear] = shared_b8[linear // 16, linear % 16]
    for index in T.unroll(0, 8):
        linear = lane * 8 + index
        output_b16[linear] = shared_b16[linear // 64, linear % 64 // 8, linear % 8]


@T.prim_func
def raw_stmatrix_x2_forms(
    output_b8: T.Buffer((256,), "uint8"), output_b16: T.Buffer((128,), "uint16")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_b8 = T.alloc_buffer((2, 8, 16), "uint8", scope="shared")
    shared_b16 = T.alloc_buffer((2, 8, 8), "uint16", scope="shared")
    source_b8 = T.alloc_buffer((2,), "uint32", scope="local")
    source_b16 = T.alloc_buffer((2,), "uint32", scope="local")
    for matrix in T.unroll(0, 2):
        source_b8[matrix] = (
            T.cast(matrix * 128 + lane * 4, "uint32")
            | T.shift_left(T.cast(matrix * 128 + lane * 4 + 1, "uint32"), T.uint32(8))
            | T.shift_left(T.cast(matrix * 128 + lane * 4 + 2, "uint32"), T.uint32(16))
            | T.shift_left(T.cast(matrix * 128 + lane * 4 + 3, "uint32"), T.uint32(24))
        )
        low = T.cast(matrix * 1000 + lane * 2, "uint32")
        high = T.cast(matrix * 1000 + lane * 2 + 1, "uint32")
        source_b16[matrix] = low | T.shift_left(high, T.uint32(16))
    T.ptx.stmatrix.sync.aligned.m16n8.x2.trans.shared__cta.b8(
        T.address_of(shared_b8[lane // 8, lane % 8, 0]),
        source_b8[0],
        source_b8[1],
    )
    T.ptx.stmatrix.sync.aligned.m8n8.x2.shared__cta.b16(
        T.address_of(shared_b16[lane // 8, lane % 8, 0]),
        source_b16[0],
        source_b16[1],
    )
    T.cuda.warp_sync()
    for index in T.unroll(0, 8):
        linear = lane * 8 + index
        output_b8[linear] = shared_b8[linear // 128, linear % 128 // 16, linear % 16]
    for index in T.unroll(0, 4):
        linear = lane * 4 + index
        output_b16[linear] = shared_b16[linear // 64, linear % 64 // 8, linear % 8]


def _stmatrix_chain(count, space, shape, dtype, transpose):
    chain = f"stmatrix.sync.aligned.{shape}.x{count}"
    if transpose:
        chain += ".trans"
    if space:
        chain += f".{space}"
    chain += dtype
    return chain


def _analyze_stmatrix_form(count, space, shape, dtype, transpose):
    chain = _stmatrix_chain(count, space, shape, dtype, transpose)
    parameters = ", ".join(f"value_{index}: T.uint32" for index in range(count))
    operands = ", ".join(f"value_{index}" for index in range(count))
    source = f'''
@T.prim_func
def store_matrix({parameters}):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32, 16), "uint8", scope="shared")
    T.ptx["{chain}"](T.address_of(shared[lane, 0]), {operands})
'''
    return analyze(tvm.script.from_source(source, extra_vars={"T": T}))


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("space", ["shared", "shared::cta"])
@pytest.mark.parametrize(
    ("shape", "dtype", "transpose"),
    [("m8n8", ".b16", False), ("m8n8", ".b16", True), ("m16n8", ".b8", True)],
)
def test_stmatrix_registry_accepts_reviewed_shared_forms(count, space, shape, dtype, transpose):
    assert _analyze_stmatrix_form(count, space, shape, dtype, transpose).unsupported == ()


def test_stmatrix_analyze_rejects_a_pointer_without_backing():
    pointer = tirx.Var("pointer", "handle")
    value = tirx.Var("value", "uint32")
    call = T.ptx.stmatrix.sync.aligned.m8n8.x1.shared.b16(pointer, value)
    spec = analyze(tirx.PrimFunc([pointer, value], tirx.Evaluate(call)))
    assert any("Unbound TIRx variable" in reason for reason in spec.unsupported)
    with pytest.raises(numsim.UnsupportedTIRxError, match="Unbound TIRx variable"):
        verify(spec)


@pytest.mark.parametrize(
    ("shape", "dtype", "transpose"),
    [("m8n8", ".b8", False), ("m8n8", ".b8", True), ("m16n8", ".b16", True)],
)
def test_stmatrix_parser_rejects_invalid_ptx_shape_type_pairs(shape, dtype, transpose):
    pointer = tirx.Var("pointer", "handle")
    value = tirx.Var("value", "uint32")
    chain = _stmatrix_chain(1, "shared", shape, dtype, transpose)
    with pytest.raises(KeyError, match="not a valid modifier"):
        T.ptx[chain](pointer, value)


def test_stmatrix_registry_accepts_repository_and_x2_forms():
    assert analyze(raw_stmatrix_layout).unsupported == ()
    assert analyze(raw_stmatrix_x2_forms).unsupported == ()


def test_stmatrix_native_layout_matches_blackwell_gpu_microtest(tmp_path):
    output_b8 = np.zeros(128, dtype=np.uint8)
    output_b16 = np.zeros(256, dtype=np.uint16)
    module = numsim.transpile(raw_stmatrix_layout, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output_b8": output_b8, "output_b16": output_b16})

    expected_b8 = np.empty((8, 16), dtype=np.uint8)
    for source_lane in range(32):
        for byte_index in range(4):
            row = 2 * (source_lane % 4) + byte_index % 2
            column = source_lane // 4 + 8 * (byte_index // 2)
            expected_b8[row, column] = source_lane * 4 + byte_index
    expected_b16 = np.empty((4, 8, 8), dtype=np.uint16)
    for matrix in range(4):
        for source_lane in range(32):
            for half_index in range(2):
                row = 2 * (source_lane % 4) + half_index
                column = source_lane // 4
                expected_b16[matrix, row, column] = matrix * 1000 + source_lane * 2 + half_index
    np.testing.assert_array_equal(result.outputs["output_b8"], expected_b8.reshape(-1))
    np.testing.assert_array_equal(result.outputs["output_b16"], expected_b16.reshape(-1))


def test_stmatrix_x2_layouts_match_ptx_fragment_mapping(tmp_path):
    output_b8 = np.zeros(256, dtype=np.uint8)
    output_b16 = np.zeros(128, dtype=np.uint16)
    module = numsim.transpile(raw_stmatrix_x2_forms, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output_b8": output_b8, "output_b16": output_b16})

    expected_b8 = np.empty((2, 8, 16), dtype=np.uint8)
    expected_b16 = np.empty((2, 8, 8), dtype=np.uint16)
    for matrix in range(2):
        for source_lane in range(32):
            for byte_index in range(4):
                row = 2 * (source_lane % 4) + byte_index % 2
                column = source_lane // 4 + 8 * (byte_index // 2)
                expected_b8[matrix, row, column] = matrix * 128 + source_lane * 4 + byte_index
            for half_index in range(2):
                row = source_lane // 4
                column = 2 * (source_lane % 4) + half_index
                expected_b16[matrix, row, column] = matrix * 1000 + source_lane * 2 + half_index
    np.testing.assert_array_equal(result.outputs["output_b8"], expected_b8.reshape(-1))
    np.testing.assert_array_equal(result.outputs["output_b16"], expected_b16.reshape(-1))
