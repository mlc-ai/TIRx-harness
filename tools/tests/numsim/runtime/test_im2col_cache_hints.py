"""Im2col cache hints retain operands and applypriority's bulk completion."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked
from tirx_harness import numsim


def hint_kernel(override, mode="im2col"):
    suffix = ".override::global_address" if override else ""
    address = ', T.reinterpret("uint64", source.ptr_to([0]))' if override else ""
    extra = ", T.uint16(0)" if mode == "im2col" else ", T.uint16(0), T.uint16(0)"
    operands = f"T.address_of(tmap){address}, T.int32(0), T.int32(0), T.int32(0){extra}"
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(tmap: T.TensorMap(), source: T.Buffer((32768,), "float32"),
           out: T.Buffer((1,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx["cp.async.bulk.prefetch.tensor.3d.L2.global.{mode}{suffix}"]({operands})
        T.ptx["cp.async.bulk.prefetch.tensor.3d.L2.global.{mode}.L2::evict_last{suffix}"]({operands})
        T.ptx["applypriority.async.bulk.tensor.3d.global.bulk_group.{mode}.L2::evict_normal{suffix}"]({operands})
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)
        out[0] = source[0]
""",
        {"T": T},
    )


@pytest.mark.parametrize(
    "override,mode",
    [(False, "im2col"), (True, "im2col"), (False, "im2col::w"), (True, "im2col::w::128")],
)
def test_im2col_cache_hints(override, mode, tmp_path):
    source = np.arange(32768, dtype=np.float32)
    descriptor = numsim.TensorMap(
        base=source,
        global_shape=(8, 8, 8),
        global_strides=(32, 256),
        box_shape=(4, 1, 1),
        element_strides=(1, 1, 1),
    ).numpy()
    args = {"source": source, "tmap": descriptor, "out": np.zeros(1, np.float32)}
    kernel = hint_kernel(override, mode)
    result = run_checked(kernel, args, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"], source[:1])
