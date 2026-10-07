"""Racecheck contracts for the mandatory .sync on PTX matrix load/store ops."""

from __future__ import annotations

import numpy as np

from tirx_harness import racecheck
from tvm.script import tirx as T


@T.prim_func
def ldmatrix_orders_cross_lane_shared_writes():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    fragment = T.alloc_local((1,), "uint32")
    for byte in T.unroll(16):
        shared[lane * 16 + byte] = T.cast(lane * 16 + byte, "uint8")
    T.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(fragment[0], shared.ptr_to([lane * 16]))


@T.prim_func
def stmatrix_orders_cross_lane_shared_reads(output: T.Buffer((64,), "uint16")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8, 8), "uint16", scope="shared")
    fragment = T.alloc_local((1,), "uint32")
    low = T.cast(lane * 2 + 1, "uint32")
    high = T.cast(lane * 2 + 2, "uint32")
    fragment[0] = low | T.shift_left(high, T.uint32(16))
    T.ptx.stmatrix.sync.aligned.m8n8.x1.shared__cta.b16(
        T.address_of(shared[lane % 8, 0]),
        fragment[0],
    )
    for element in T.unroll(2):
        linear = lane * 2 + element
        output[linear] = shared[linear // 8, linear % 8]


def _assert_complete_clean(report):
    report.require_clean()
    native = report.to_dict()["native"]
    assert native["findings"] == []
    assert native["incomplete"] == []


def test_ldmatrix_mandatory_sync_orders_provider_lane_writes():
    _assert_complete_clean(racecheck(ldmatrix_orders_cross_lane_shared_writes, {}))


def test_stmatrix_mandatory_sync_orders_consumer_lane_reads():
    _assert_complete_clean(
        racecheck(
            stmatrix_orders_cross_lane_shared_reads,
            {"output": np.zeros(64, dtype=np.uint16)},
        )
    )
