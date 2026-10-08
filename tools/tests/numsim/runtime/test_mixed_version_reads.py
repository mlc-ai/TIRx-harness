"""Whole b128 reads can observe the bytes left by multiple masked-write versions."""

import numpy as np
import tvm
from tvm.script import tirx as T

from tirx_harness import racecheck, synccheck


_MODES = ("volatile", "acquire", "relaxed", "cas")


def mixed_version_case(mode="cas", *, late_data=False):
    read = (
        'T.ptx["atom.acquire.gpu.global.cas.b128"](value[0], flag.ptr_to([0]), expected[0], expected[0])'
        if mode == "cas"
        else f'T.ptx["ld.{mode}{"" if mode == "volatile" else ".gpu"}.global.b128"](value[0], flag.ptr_to([0]))'
    )
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(flag: T.Buffer((2,), "uint64"), data: T.Buffer((2,), "uint32"), out: T.Buffer((2,), "uint32")):
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    shared = T.alloc_shared((32,), "uint8", align=16)
    value = T.alloc_local((1,), "uint128")
    expected = T.alloc_local((1,), "uint128")
    if warp < 2:
        if lane < 16:
            shared[warp * 16 + lane] = T.uint8(1)
        T.cuda.warp_sync()
        T.ptx.fence.proxy.async_.shared__cta()
        if lane == 0:
            {"T.evaluate(0)" if late_data else 'data[warp] = T.Cast("uint32", 42 + warp)'}
            T.ptx.fence.release.gpu()
            T.ptx["cp.async.bulk.relaxed.gpu.global.shared::cta.bulk_group.cp_mask.b128"](
                flag.ptr_to([0]), shared.ptr_to([warp * 16]), T.uint32(16),
                T.Cast("uint16", T.if_then_else(warp == 0, 255, 65280)))
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group(0)
    T.cuda.cta_sync()
    {"if warp < 2 and lane == 0:" if late_data else ""}
        {'data[warp] = T.Cast("uint32", 42 + warp)' if late_data else ""}
    if warp == 2 and lane == 0:
        T.ptx.mov.b128(value[0], T.uint64(0), T.uint64(0))
        T.ptx.mov.b128(expected[0], T.uint64(0x0101010101010101), T.uint64(0x0101010101010101))
        while value[0] != expected[0]:
            T.ptx["fence.proxy.async.global"]()
            {read}
            {"T.ptx.fence.acquire.gpu()" if mode == "relaxed" else "T.evaluate(0)"}
        out[0] = data[0]
        out[1] = data[1]
""",
        {"T": T},
    )


def inputs():
    return {
        "flag": np.zeros(2, np.uint64),
        "data": np.zeros(2, np.uint32),
        "out": np.zeros(2, np.uint32),
    }


def test_mixed_version_does_not_publish_later_data():
    # Neither the prior release heads nor the barrier publish subsequent stores.
    for mode in _MODES:
        kernel = mixed_version_case(mode, late_data=True)
        synccheck(kernel, inputs()).require_clean()
        report = racecheck(kernel, inputs())
        assert report.verdict == "error", report.format()
        assert not any(f.status == "incomplete" for f in report.findings), report.format()
        assert any(f.details["access_pair"] in {"write_read", "read_write"} for f in report.findings), report.format()
