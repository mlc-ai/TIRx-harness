"""tvm-tirx tensor allocations and pointer-offset views."""

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tvm_tirx.ir import PtrType
from tvm_tirx.script import tirx as T


@T.function
def pool_views(source: T.Tensor((64,), "float32"), output: T.Tensor((64,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64, dynamic_smem_bytes=512))
    tid = T.cuda.thread_idx("x")
    pool = T.alloc_tensor(ty_args=[T.Tensor((0,), "uint8", scope="shared.dyn")])
    first = T.decl_tensor(
        T.ptr_byte_offset(pool.data, 0, ty=PtrType("float32", "shared.dyn")),
        ty_args=[T.Tensor((64,), "float32", scope="shared.dyn")],
    )
    second = T.decl_tensor(
        T.ptr_byte_offset(pool.data, 256, ty=PtrType("float32", "shared.dyn")),
        ty_args=[T.Tensor((64,), "float32", scope="shared.dyn")],
    )
    first[tid] = source[tid]
    second[tid] = source[tid] * T.float32(2)
    T.cuda.cta_sync()
    output[tid] = first[63 - tid] + second[tid]


def test_pool_views_start_at_their_byte_offsets(tmp_path):
    source = np.arange(64, dtype=np.float32)
    module = numsim.transpile(pool_views, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source": source, "output": np.zeros(64, dtype=np.float32)},
        outputs=("output",),
    )
    np.testing.assert_array_equal(result.outputs["output"], source[::-1] + 2 * source)


@T.function
def misaligned_view(output: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32, dynamic_smem_bytes=256))
    lane = T.cuda.lane_id()
    pool = T.alloc_tensor(ty_args=[T.Tensor((0,), "uint8", scope="shared.dyn")])
    view = T.decl_tensor(
        T.ptr_byte_offset(pool.data, 2, ty=PtrType("float32", "shared.dyn")),
        ty_args=[T.Tensor((32,), "float32", scope="shared.dyn")],
    )
    view[lane] = T.float32(1)
    output[lane] = view[lane]


def test_views_between_elements_are_rejected(tmp_path):
    with pytest.raises(UnsupportedTIRxError, match="not a multiple of the 4-byte element"):
        numsim.transpile(misaligned_view, cache_dir=tmp_path)
