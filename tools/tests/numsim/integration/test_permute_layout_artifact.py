from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tests.numsim.support.kernels import (
    global_permute_layout_roundtrip,
    permuted_global_layout_read,
    shared_permute_layout_roundtrip,
    shared_permute_layout_zero_fills_bf16_padding,
    shared_permute_layout_zero_fills_fp8_padding,
    shared_permute_layout_zero_fills_fp16_padding,
    shared_permute_layout_zero_fills_uninitialized_padding,
)
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TileLayout

_EXPLICIT_PERMUTED_LAYOUT = TileLayout(S[(4, 32) : (1, 4)])


@T.prim_func
def _explicit_permute_layout_dispatch(
    source: T.Buffer((128,), "uint32"), output: T.Buffer((128,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source_shared = T.alloc_buffer((128,), "uint32", scope="shared")
    permuted_shared = T.alloc_buffer(
        (128,), "uint32", scope="shared", layout=_EXPLICIT_PERMUTED_LAYOUT
    )
    if lane == 0:
        Tx.copy(source_shared[:], source[:])
    T.cuda.warp_sync()
    Tx.warp.permute_layout(permuted_shared[:], source_shared[:], dispatch="warp_xor_swizzle")
    T.cuda.warp_sync()
    if lane == 0:
        Tx.copy(output[:], permuted_shared[:])


def test_shared_permute_layout_copies_logical_values_between_physical_layouts(tmp_path):
    source = np.arange(128, dtype=np.uint32) * np.uint32(17) + np.uint32(3)
    output = np.zeros_like(source)

    module = numsim.transpile(shared_permute_layout_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_explicit_permute_layout_dispatch_uses_canonical_semantics(tmp_path):
    source = np.arange(128, dtype=np.uint32) * np.uint32(23) + np.uint32(7)
    output = np.zeros_like(source)

    module = numsim.transpile(_explicit_permute_layout_dispatch, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_global_permute_layout_writes_destination_physical_order(tmp_path):
    source = np.arange(128, dtype=np.uint32) * np.uint32(29) + np.uint32(11)
    output = np.zeros_like(source)

    module = numsim.transpile(global_permute_layout_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.empty_like(source)
    logical_index = np.arange(128)
    physical_index = (logical_index // 32) + (logical_index % 32) * 4
    expected[physical_index] = source
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_shared_permute_layout_zero_fills_only_uninitialized_padding(tmp_path):
    source = np.arange(16, dtype=np.uint32) * np.uint32(19) + np.uint32(5)
    output = np.full(128, np.uint32(0xDEADBEEF), dtype=np.uint32)

    module = numsim.transpile(
        shared_permute_layout_zero_fills_uninitialized_padding, cache_dir=tmp_path
    )
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.zeros(128, dtype=np.uint32)
    expected[:16] = source
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_shared_permute_layout_zero_fills_fp16_padding(tmp_path):
    source = np.arange(1, 17, dtype=np.float16)
    output = np.full(128, np.float16(-1), dtype=np.float16)

    module = numsim.transpile(shared_permute_layout_zero_fills_fp16_padding, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.zeros(128, dtype=np.float16)
    expected[:16] = source
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_shared_permute_layout_zero_fills_bf16_padding(tmp_path):
    source_values = np.arange(1, 17, dtype=np.float32)
    source = (source_values.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    output = np.full(128, np.uint16(0xFFFF), dtype=np.uint16)

    module = numsim.transpile(shared_permute_layout_zero_fills_bf16_padding, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "output": output,
        },
    )

    expected = np.zeros(128, dtype=np.uint16)
    expected[:16] = source
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_shared_permute_layout_zero_fills_fp8_padding(tmp_path):
    source = np.arange(0x20, 0x30, dtype=np.uint8)
    output = np.full(128, np.uint8(0xFF), dtype=np.uint8)

    module = numsim.transpile(shared_permute_layout_zero_fills_fp8_padding, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "output": output,
        },
    )

    expected = np.zeros(128, dtype=np.uint8)
    expected[:16] = source
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_global_nondefault_layout_consumes_physical_host_order(tmp_path):
    logical = np.arange(128, dtype=np.uint32) * np.uint32(13) + np.uint32(5)
    physical = np.empty_like(logical)
    logical_index = np.arange(128)
    physical_index = (logical_index // 32) + (logical_index % 32) * 4
    physical[physical_index] = logical
    output = np.zeros_like(logical)

    module = numsim.transpile(permuted_global_layout_read, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": physical, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], logical)
