"""Release-semantics same-address multi-lane RMW around a divergent-lane race.

The lane_order tests cover relaxed ``T.cuda.atomic_add``; this file covers the
release-flavored form. One same-address 32-lane release RMW leaves the lane
serialization order -- and with it every lane's read-from version --
unconstrained, so the checker withholds those edges. Withholding must stay
exactly that: the divergent-lane race around the RMW is still reported with
one precise finding and no ``incomplete``, and an explicit ``warp_sync``
still orders the same pattern into ``clean``.
"""

from __future__ import annotations

import numpy as np
import pytest
from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tvm.script import tirx as T


@T.prim_func
def native_lane_race_bridged_only_by_same_address_rmw(
    data: T.Buffer((1,), "int32"),
    flag: T.Buffer((1,), "int32"),
    output: T.Buffer((1,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    observed = T.local_scalar("int32")
    if lane == 0:
        data[0] = 37
    # All 32 lanes release-RMW one address. Each lane's version would carry its
    # release head, so a fabricated intra-instruction chain would hand lane 0's
    # head to whichever lane's version ends up last.
    T.ptx.atom.release.gpu.global_.add.s32(observed, flag.ptr_to([0]), T.int32(1))
    if lane == 1:
        T.ptx.ld.acquire.gpu.global_.s32(observed, flag.ptr_to([0]))
        output[0] = data[0] + observed


@T.prim_func
def native_lane_handoff_ordered_by_warp_sync(
    data: T.Buffer((1,), "int32"),
    flag: T.Buffer((1,), "int32"),
    output: T.Buffer((1,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    observed = T.local_scalar("int32")
    if lane == 0:
        data[0] = 37
    T.ptx.atom.release.gpu.global_.add.s32(observed, flag.ptr_to([0]), T.int32(1))
    T.cuda.warp_sync()
    if lane == 1:
        T.ptx.ld.acquire.gpu.global_.s32(observed, flag.ptr_to([0]))
        output[0] = data[0] + observed


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-racecheck-lane-serialization-guard")


def _inputs() -> dict[str, np.ndarray]:
    return {
        "data": np.zeros(1, dtype=np.int32),
        "flag": np.zeros(1, dtype=np.int32),
        "output": np.zeros(1, dtype=np.int32),
    }


def test_same_address_rmw_serialization_is_not_happens_before(native_cache_dir):
    report = racecheck(
        native_lane_race_bridged_only_by_same_address_rmw,
        inputs=_inputs(),
        cache_dir=native_cache_dir,
        max_workers=1,
    )

    native = report.to_dict()["native"]
    # The race verdict itself stays decidable: no incomplete, one exact race.
    assert report.verdict == "error"
    assert native["incomplete"] == []
    races = [f for f in native["findings"] if f["access_pair"] == "write_read"]
    assert len(races) == 1
    finding = races[0]
    assert {finding["prior"]["lane"], finding["current"]["lane"]} == {0, 1}
    assert finding["prior"]["space"] == finding["current"]["space"] == "global"
    # Report the proven race, not an additional declaration-policy review.
    assert {f["access_pair"] for f in native["findings"]} == {"write_read"}
    assert "undeclared_protocol_words" not in native


def test_warp_sync_orders_the_same_pattern(native_cache_dir):
    report = racecheck(
        native_lane_handoff_ordered_by_warp_sync,
        inputs=_inputs(),
        cache_dir=native_cache_dir,
        max_workers=1,
    )

    report.require_clean()
    assert report.to_dict()["native"]["incomplete"] == []
