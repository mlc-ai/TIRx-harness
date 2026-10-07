"""Policy encodings are opaque: test their use, not hardware-specific bits."""

import numpy as np
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked


@T.prim_func
def policy_kernel(
    source: T.Buffer((32,), "uint32"),
    out: T.Buffer((32,), "uint32"),
    fraction: T.float32,
    primary: T.uint32,
    total: T.uint32,
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    policy = T.alloc_local((1,), "uint64")
    inherited = T.alloc_local((1,), "uint64")
    value = T.alloc_local((1,), "uint32")
    T.ptx["createpolicy.fractional.L2::evict_last.b64"](policy[0])
    T.ptx["createpolicy.fractional.L2::evict_last.b64"](policy[0], fraction)
    T.ptx["createpolicy.cvt.L2.b64"](policy[0], policy[0])
    T.ptx["createpolicy.range.global.L2::evict_last.L2::evict_first.b64"](
        policy[0], source.ptr_to([0]), primary, total
    )
    if lane % 2 == 0:
        inherited[0] = policy[0]
    T.ptx["createpolicy.cvt.L2.b64"](
        policy[0], inherited[0], pred=lane % 2 == 0, preserve_dst=True
    )
    T.ptx["ld.global.L2::cache_hint.u32"](value[0], source.ptr_to([lane]), policy[0])
    out[lane] = value[0]


def inputs(fraction=0.5, primary=64, total=128):
    return {
        "source": np.arange(32, dtype=np.uint32),
        "out": np.zeros(32, np.uint32),
        "fraction": fraction,
        "primary": primary,
        "total": total,
    }


def test_cache_policy(tmp_path):
    args = inputs()
    result = run_checked(policy_kernel, args, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"], args["source"])
