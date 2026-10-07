import numpy as np
import pytest
import tirx_kernels.tirx_lite as txl
from tvm.script import tirx as T

from tirx_harness import numsim
from tirx_harness.numsim.errors import NumSimExecutionError


@txl.kernel(warps=1, arch="sm_100a", grid=1, check_ir=False)
def runtime_sized_global_alias(
    storage: txl.gptr[txl.u8],
    output: txl.gptr[txl.u32, (1,)],
):
    lane = txl.lane_id()
    words = txl.decl_buffer((8,), txl.u32, data=storage.data, scope="global")
    with txl.If(lane == 0), txl.Then():
        txl.buffer_store(output, words[7], [0])


@T.prim_func
def runtime_sized_global_shifted_alias(
    storage_ptr: T.handle,
    word_offset: T.int64,
    output: T.Buffer((1,), "uint32"),
):
    storage_dim0 = T.int64()
    storage = T.match_buffer(storage_ptr, (storage_dim0,), "uint8")
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    words = T.decl_buffer(
        (1,),
        "uint32",
        data=storage.data,
        elem_offset=word_offset,
        scope="global",
    )
    if lane == 0:
        output[0] = words[0]


@T.prim_func
def runtime_sized_global_dynamic_reinterpret_alias(
    storage_ptr: T.handle,
    byte_count: T.int64,
    output: T.Buffer((1,), "uint32"),
):
    storage_dim0 = T.int64()
    storage = T.match_buffer(storage_ptr, (storage_dim0,), "uint8")
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    words = T.decl_buffer(
        (byte_count // 4,),
        "uint32",
        data=storage.data,
        scope="global",
    )
    if lane == 0:
        output[0] = words[byte_count // 4 - 1]


def test_runtime_sized_global_owner_defers_static_alias_bound_to_launch(tmp_path):
    storage = np.arange(32, dtype=np.uint8)
    result = numsim.Engine().run(
        numsim.transpile(runtime_sized_global_alias.func, cache_dir=tmp_path),
        {
            "storage": storage,
            "output": np.zeros(1, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], storage[28:32].view(np.uint32))


def test_runtime_sized_global_owner_rejects_alias_larger_than_launch_backing(tmp_path):
    module = numsim.transpile(runtime_sized_global_alias.func, cache_dir=tmp_path)

    with pytest.raises(NumSimExecutionError, match="view .* exceeds allocation"):
        numsim.Engine().run(
            module,
            {
                "storage": np.zeros(31, dtype=np.uint8),
                "output": np.zeros(1, dtype=np.uint32),
            },
        )


def test_runtime_sized_global_owner_bounds_runtime_shifted_alias_at_launch(tmp_path):
    storage = np.arange(32, dtype=np.uint8)
    module = numsim.transpile(runtime_sized_global_shifted_alias, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "storage": storage,
            "word_offset": np.int64(7),
            "output": np.zeros(1, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], storage[28:32].view(np.uint32))

    with pytest.raises(NumSimExecutionError, match="out-of-bounds|outside"):
        numsim.Engine().run(
            module,
            {
                "storage": storage,
                "word_offset": np.int64(8),
                "output": np.zeros(1, dtype=np.uint32),
            },
        )


def test_runtime_sized_global_owner_bounds_dynamic_reinterpret_alias_at_launch(tmp_path):
    storage = np.arange(32, dtype=np.uint8)
    module = numsim.transpile(
        runtime_sized_global_dynamic_reinterpret_alias,
        cache_dir=tmp_path,
    )
    result = numsim.Engine().run(
        module,
        {
            "storage": storage,
            "byte_count": np.int64(32),
            "output": np.zeros(1, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], storage[28:32].view(np.uint32))

    with pytest.raises(NumSimExecutionError, match="view .* exceeds allocation"):
        numsim.Engine().run(
            module,
            {
                "storage": storage,
                "byte_count": np.int64(36),
                "output": np.zeros(1, dtype=np.uint32),
            },
        )
