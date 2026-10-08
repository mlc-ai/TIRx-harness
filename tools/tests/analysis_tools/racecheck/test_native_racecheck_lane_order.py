"""Native Racecheck integration coverage."""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm.script import tirx as T


@T.prim_func
def native_same_warp_lane_handoff(use_sync: T.int32, output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "int32", scope="shared")

    if use_sync >= 3:
        if lane == 0:
            shared[0] = 0
        T.cuda.warp_sync()
        if lane < 2:
            T.cuda.atomic_add(shared.ptr_to([0]), T.int32(1))
    else:
        if lane == 0:
            shared[0] = 41
    if (use_sync == 1) | (use_sync == 4):
        T.cuda.warp_sync()
    elif (use_sync == 2) | (use_sync == 5):
        T.cuda.cta_sync()
    if lane == 1:
        output[0] = shared[0]


@T.prim_func
def native_same_lane_program_order(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "int32", scope="shared")

    if lane == 0:
        shared[0] = 17
        output[0] = shared[0]


@T.prim_func
def native_same_address_warp_atomic(
    counter: T.Buffer((1,), "int32"), output: T.Buffer((32,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.cuda.atomic_add(counter.ptr_to([0]), T.int32(1))


@T.prim_func
def native_disjoint_warp_atomics(
    counters: T.Buffer((32,), "int32"), output: T.Buffer((32,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.cuda.atomic_add(counters.ptr_to([lane]), T.int32(1))


@T.prim_func
def native_same_address_atomic_controls_warp_sync(
    counter: T.Buffer((1,), "int32"), output: T.Buffer((32,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    ticket: T.let = T.cuda.atomic_add(counter.ptr_to([0]), T.int32(1))

    # The native numeric order returns ticket == lane, so this execution is a
    # full-warp sync. Another legal lane serialization can change participation.
    if ticket == lane:
        T.cuda.warp_sync()
    output[lane] = ticket


@T.prim_func
def native_same_address_atomic_selects_output_address(
    counter: T.Buffer((1,), "int32"), output: T.Buffer((32,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    ticket: T.let = T.cuda.atomic_add(counter.ptr_to([0]), T.int32(1))
    output[ticket] = lane


@T.prim_func
def native_atomic_poll_does_not_publish_shared_write(
    counter: T.Buffer((1,), "int32"), output: T.Buffer((1,), "int32")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "int32", scope="shared")
    observed = T.local_scalar("int32")

    if warp == 0:
        if lane == 0:
            observed = T.int32(0)
            # Relaxed on purpose: this wait is the subject of the test. The
            # counter's modification order says nothing about `shared`, and a
            # relaxed wait states exactly that, so the read below stays the
            # race it is. An acquiring wait here would order `shared` and
            # leave the test with nothing to catch.
            T.cuda.wait_until(
                observed, counter.ptr_to([0]), observed >= T.int32(1), "gpu", "global",
            )
            output[0] = shared[0]
    elif lane == 0:
        shared[0] = 37
        T.ptx.red.relaxed.gpu.global_.add.s32(counter.ptr_to([0]), T.int32(1))


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-racecheck-lane-order")


def _atomic_inputs() -> dict[str, np.ndarray]:
    return {
        "counter": np.zeros(1, dtype=np.int32),
        "output": np.zeros(32, dtype=np.int32),
    }


@pytest.mark.parametrize("atomic", [False, True])
def test_public_racecheck_requires_sync_between_different_lanes_in_one_warp(native_cache_dir, atomic):
    mode = 3 if atomic else 0
    unsynchronized = racecheck(
        native_same_warp_lane_handoff,
        inputs={"use_sync": np.int32(mode), "output": np.zeros(1, dtype=np.int32)},
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    warp_synchronized = racecheck(
        native_same_warp_lane_handoff,
        inputs={"use_sync": np.int32(mode + 1), "output": np.zeros(1, dtype=np.int32)},
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    cta_synchronized = racecheck(
        native_same_warp_lane_handoff,
        inputs={"use_sync": np.int32(mode + 2), "output": np.zeros(1, dtype=np.int32)},
        cache_dir=native_cache_dir,
        max_workers=1,
    )

    assert unsynchronized.verdict == "error"
    native = unsynchronized.to_dict()["native"]
    assert native["incomplete"] == []
    assert len(native["findings"]) == 1
    finding = native["findings"][0]
    assert finding["access_pair"] == "write_read"
    assert finding["ordering_domain"] == "execution"
    assert finding["ordering_failure"] == "missing_same_warp_lane_order"
    assert finding["prior"]["operation"]["global_warp_id"] == 0
    assert finding["current"]["operation"]["global_warp_id"] == 0
    assert finding["prior"]["lane"] == 0
    assert finding["current"]["lane"] == 1
    assert finding["prior"]["operation"] != finding["current"]["operation"]
    assert finding["prior"]["space"] == finding["current"]["space"] == "shared"
    assert finding["overlap"]["byte_len"] == 4

    for synchronized in (warp_synchronized, cta_synchronized):
        synchronized.require_clean()
        assert synchronized.to_dict()["native"]["incomplete"] == []


def test_public_racecheck_preserves_same_lane_program_order(native_cache_dir):
    report = racecheck(
        native_same_lane_program_order,
        inputs={"output": np.zeros(1, dtype=np.int32)},
        cache_dir=native_cache_dir,
        max_workers=1,
    )

    report.require_clean()
    assert report.to_dict()["native"]["incomplete"] == []


def test_direct_atomic_modification_order_does_not_publish_shared_write(native_cache_dir):
    report = racecheck(
        native_atomic_poll_does_not_publish_shared_write,
        inputs={
            "counter": np.zeros(1, dtype=np.int32),
            "output": np.zeros(1, dtype=np.int32),
        },
        cache_dir=native_cache_dir,
        max_workers=2,
    )

    assert report.verdict == "error"
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert len(native["findings"]) == 1
    finding = native["findings"][0]
    assert finding["access_pair"] == "write_read"
    assert finding["prior"]["space"] == finding["current"]["space"] == "shared"
    assert finding["prior"]["operation"]["global_warp_id"] == 1
    assert finding["current"]["operation"]["global_warp_id"] == 0


def test_output_only_same_address_multi_lane_atomic_is_clean(native_cache_dir):
    numeric_module = numsim.transpile(
        native_same_address_warp_atomic,
        cache_dir=native_cache_dir,
        _analysis_checker=None,
    )
    synccheck_module = numsim.transpile(
        native_same_address_warp_atomic,
        cache_dir=native_cache_dir,
        _analysis_checker="synccheck",
    )
    racecheck_module = numsim.transpile(
        native_same_address_warp_atomic,
        cache_dir=native_cache_dir,
    )
    numeric = numsim.Engine().run(numeric_module, _atomic_inputs())
    np.testing.assert_array_equal(numeric.outputs["counter"], np.array([32], dtype=np.int32))
    np.testing.assert_array_equal(numeric.outputs["output"], np.arange(32, dtype=np.int32))

    inputs = _atomic_inputs()
    sync_phase = numsim.Engine().run_synccheck_phase(synccheck_module, inputs)
    race_phase = numsim.Engine().run_racecheck_phase(racecheck_module, inputs)
    assert sync_phase.verdict == "clean"
    assert race_phase.verdict == "clean"
    assert sync_phase.incomplete == []
    assert race_phase.incomplete == []

    public_sync = synccheck(
        native_same_address_warp_atomic,
        inputs=_atomic_inputs(),
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    public_race = racecheck(
        native_same_address_warp_atomic,
        inputs=_atomic_inputs(),
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    public_sync.require_clean()
    public_race.require_clean()
    assert public_sync.findings == []
    assert public_race.findings == []


def test_synccheck_uses_one_concrete_same_instruction_atomic_lane_order(native_cache_dir):
    sync = synccheck(
        native_same_address_atomic_controls_warp_sync,
        inputs=_atomic_inputs(),
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    race = racecheck(
        native_same_address_atomic_controls_warp_sync,
        inputs=_atomic_inputs(),
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    sync.require_clean()
    assert sync.findings == []
    native_sync = sync.to_dict()["native"]
    assert native_sync["incomplete"] == []

    race.require_clean()
    assert race.findings == []


def test_address_only_same_instruction_atomic_return_is_clean(native_cache_dir):
    sync = synccheck(
        native_same_address_atomic_selects_output_address,
        inputs=_atomic_inputs(),
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    race = racecheck(
        native_same_address_atomic_selects_output_address,
        inputs=_atomic_inputs(),
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    sync.require_clean()
    race.require_clean()
    assert sync.findings == []
    assert race.findings == []


def test_disjoint_multi_lane_atomics_remain_exactly_clean(native_cache_dir):
    inputs = {"counters": np.zeros(32, dtype=np.int32), "output": np.zeros(32, dtype=np.int32)}
    sync = synccheck(
        native_disjoint_warp_atomics,
        inputs=inputs,
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    race = racecheck(
        native_disjoint_warp_atomics,
        inputs=inputs,
        cache_dir=native_cache_dir,
        max_workers=1,
    )

    sync.require_clean()
    race.require_clean()
    assert sync.to_dict()["native"]["incomplete"] == []
    assert race.to_dict()["native"]["incomplete"] == []
