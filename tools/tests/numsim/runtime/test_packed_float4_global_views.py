from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

from tirx_harness import numsim
from tvm.script import tirx as T
from tvm.tirx.layout import ComposeLayout, S, TileLayout

_PADDED_LAYOUT = ComposeLayout(0, 0, 0, TileLayout(S[(2, 2) : (4, 1)]))


@T.prim_func
def read_four_float4(source: T.Buffer((4,), "float4_e2m1fn"), output: T.Buffer((4,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 4:
        output[lane] = T.cast(source[lane], "float32")


@T.prim_func
def read_three_float4(source: T.Buffer((3,), "float4_e2m1fn"), output: T.Buffer((3,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 3:
        output[lane] = T.cast(source[lane], "float32")


@T.prim_func
def padded_global_alias(storage: T.Buffer((6,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    padded = T.decl_buffer(
        (2, 2), "uint32", data=storage.data, scope="global", layout=_PADDED_LAYOUT
    )
    if lane == 0:
        storage[2] = 90
        storage[3] = 91
        padded[0, 0] = 10
        padded[0, 1] = 11
        padded[1, 0] = 12
        padded[1, 1] = 13


@pytest.fixture(scope="module")
def four_float4_module(tmp_path_factory):
    return numsim.transpile(
        read_four_float4, cache_dir=tmp_path_factory.mktemp("packed_float4_four")
    )


def test_explicit_uint8_backing_supplies_two_float4_values_per_byte(four_float4_module):
    packed = np.array([0x42, 0x65], dtype=np.uint8)
    output = np.zeros(4, dtype=np.float32)

    result = numsim.Engine().run(
        four_float4_module,
        {
            "source": packed,
            "output": output,
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], [1.0, 2.0, 3.0, 4.0])


def test_direct_one_byte_per_value_float4_array_is_rejected(four_float4_module):
    unpacked = np.array([1.0, 2.0, 3.0, 4.0], dtype=ml_dtypes.float4_e2m1fn)

    with pytest.raises(numsim.NumSimExecutionError, match="contiguous uint8 array"):
        numsim.Engine().run(
            four_float4_module,
            {"source": unpacked, "output": np.zeros(4, dtype=np.float32)},
        )


def test_odd_float4_logical_count_uses_a_ceiling_byte_span(tmp_path):
    packed = np.array([0x42, 0xF5], dtype=np.uint8)
    output = np.zeros(3, dtype=np.float32)
    module = numsim.transpile(read_three_float4, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {
            "source": packed,
            "output": output,
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], [1.0, 2.0, 3.0])


def test_global_alias_view_uses_layout_physical_span(tmp_path):
    storage = np.zeros(6, dtype=np.uint32)
    module = numsim.transpile(padded_global_alias, cache_dir=tmp_path)

    result = numsim.Engine().run(module, {"storage": storage})

    np.testing.assert_array_equal(
        result.outputs["storage"], np.array([10, 11, 90, 91, 12, 13], dtype=np.uint32)
    )
