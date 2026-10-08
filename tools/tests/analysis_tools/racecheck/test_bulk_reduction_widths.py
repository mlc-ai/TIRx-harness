"""Bulk reductions are atomic per PTX element, not per four-byte word."""

import numpy as np
import tvm
from tvm.script import tirx as T

from tirx_harness import racecheck, synccheck


def reduction_and_peer(ptx_type, peer):
    half = ptx_type == "f16"
    reduction = f"add{'.noftz' if half else ''}.{ptx_type}"
    if peer == "matching_atomic":
        spelling = f"cp.reduce.async.bulk.global.shared::cta.bulk_group.{reduction}"
    elif peer == "overlapping_atomic":
        spelling = "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32"
    else:
        spelling = "cp.async.bulk.global.shared::cta.bulk_group"
    offset = 16 if peer == "outside" else 0
    return tvm.script.from_source(
        f'''
@T.prim_func
def kernel(destination: T.Buffer((32,), "uint8")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_shared((32,), "uint8", align=16)
    if lane == 0:
        for index in T.serial(16):
            shared[warp * 16 + index] = T.uint8(0)
        T.ptx.fence.proxy.async_.shared__cta()
        if warp == 0:
            T.ptx["cp.reduce.async.bulk.global.shared::cta.bulk_group.{reduction}"](
                destination.ptr_to([0]), shared.ptr_to([0]), T.uint32(16))
        else:
            T.ptx["{spelling}"](
                destination.ptr_to([{offset}]), shared.ptr_to([16]), T.uint32(16))
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)
''',
        {"T": T},
    )


def test_bulk_reduction_element_width_and_boundary():
    for ptx_type in ("f16", "u64"):
        for peer in ("matching_atomic", "overlapping_atomic", "inside", "outside"):
            kernel = reduction_and_peer(ptx_type, peer)
            inputs = {"destination": np.zeros(32, np.uint8)}
            synccheck(kernel, inputs).require_clean()
            report = racecheck(kernel, inputs)
            if peer in {"matching_atomic", "outside"}:
                report.require_clean()
            else:
                assert report.verdict == "error", (ptx_type, peer, report.format())
                assert any(f.status == "error" for f in report.findings), report.format()
                assert all(f.status != "incomplete" for f in report.findings), report.format()
