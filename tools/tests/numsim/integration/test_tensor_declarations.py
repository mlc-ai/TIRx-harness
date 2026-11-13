"""tvm-tirx tensor allocations and pointer-offset views."""

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tvm_tirx.ir import PtrType
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import Axis, S, TileLayout


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


_TMEM_LAYOUT = TileLayout(S[(128, 4) : (1 @ Axis.get("TLane"), 1 @ Axis.get("TCol"))])


@T.function
def tmem_from_alloc_tensor(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    tmem = T.alloc_tensor(ty_args=[T.Tensor((128, 4), "uint32", scope="tmem", layout=_TMEM_LAYOUT)])
    output[lane] = tmem[lane, 0]


@T.function
def tmem_from_tmem_helper(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    tmem = T.tmem((128, 4), "uint32", allocated_addr=0, layout=_TMEM_LAYOUT)
    output[lane] = tmem[lane, 0]


@T.function
def tmem_from_decl_tmem(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    tmem = T.cuda.decl_tmem(
        T.uint32(0),
        ty_args=[T.Tensor((128, 4), "uint32", scope="tmem", layout=_TMEM_LAYOUT)],
    )
    output[lane] = tmem[lane, 0]


@pytest.mark.parametrize(
    "kernel", [tmem_from_alloc_tensor, tmem_from_tmem_helper, tmem_from_decl_tmem]
)
def test_declared_tmem_buffers_are_rejected(kernel, tmp_path):
    with pytest.raises(UnsupportedTIRxError, match=r'does not support scope="tmem" buffers'):
        numsim.transpile(kernel, cache_dir=tmp_path)
