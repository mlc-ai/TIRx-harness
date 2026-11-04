import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.errors import NumSimExecutionError
from tvm_tirx.script import tirx as T

_STORAGE_BYTES = T.dynamic("storage_bytes", "int64")


@T.function
def runtime_sized_global_alias(
    storage: T.Tensor((_STORAGE_BYTES,), "uint8"),
    output: T.Tensor((1,), "uint32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    words = T.decl_tensor(storage.data, ty_args=[T.Tensor((8,), "uint32", scope="global")])
    if lane == 0:
        output[0] = words[7]


@T.function
def runtime_sized_global_shifted_alias(
    storage: T.Tensor((_STORAGE_BYTES,), "uint8"),
    word_offset: T.int64,
    output: T.Tensor((1,), "uint32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    words = T.decl_tensor(
        T.ptr_byte_offset(storage.data, word_offset * 4, ty=T.Ptr("uint32", "global")),
        ty_args=[T.Tensor((1,), "uint32", scope="global")],
    )
    if lane == 0:
        output[0] = words[0]


@T.function
def runtime_sized_global_dynamic_reinterpret_alias(
    storage: T.Tensor((_STORAGE_BYTES,), "uint8"),
    byte_count: T.int64,
    output: T.Tensor((1,), "uint32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    words = T.decl_tensor(
        storage.data, ty_args=[T.Tensor((byte_count // 4,), "uint32", scope="global")]
    )
    if lane == 0:
        output[0] = words[byte_count // 4 - 1]


def test_runtime_sized_global_owner_defers_static_alias_bound_to_launch(tmp_path):
    storage = np.arange(32, dtype=np.uint8)
    result = numsim.Engine().run(
        numsim.transpile(runtime_sized_global_alias, cache_dir=tmp_path),
        {
            "storage": storage,
            "output": np.zeros(1, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], storage[28:32].view(np.uint32))


def test_runtime_sized_global_owner_rejects_alias_larger_than_launch_backing(tmp_path):
    module = numsim.transpile(runtime_sized_global_alias, cache_dir=tmp_path)

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
