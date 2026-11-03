"""Racecheck contracts for asynchronous-memory lifetimes."""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import racecheck
from tvm_tirx.script import tirx as T


@T.function
def native_classic_cp_async_lifetime(
    mode: T.int32,
    source: T.Tensor((128,), "float32"),
    output: T.Tensor((128,), "float32"),
):
    """Keep classic cp.async source/destination accesses live until wait_group."""

    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((128,), "float32")

    T.ptx["cp.async.cg.shared.global"](
        T.address_of(shared[lane * 4]),
        T.address_of(source[lane * 4]),
        16,
    )
    T.ptx.cp.async_.commit_group()
    if (mode == 1) or (mode == 3):
        T.ptx.cp.async_.wait_group(0)
    if mode <= 1:
        source[lane * 4] = T.cast(lane, "float32")
    else:
        output[lane * 4] = shared[lane * 4]
    T.ptx.cp.async_.wait_group(0)


@T.function
def native_classic_cp_async_orders_prior_lane_write(
    source: T.Tensor((128,), "float32"),
):
    """A cp.async operation starts after prior instructions in its issuing lane."""

    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((128,), "float32")

    shared[lane * 4] = T.float32(0)
    T.ptx["cp.async.cg.shared.global"](
        T.address_of(shared[lane * 4]),
        T.address_of(source[lane * 4]),
        16,
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)


@T.function
def native_classic_cp_async_cross_warp_source_race(
    source: T.Tensor((128,), "float32"),
):
    """Keep completion from masking a missing cross-warp source ordering edge."""

    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((128,), "float32")

    if warp == 0:
        T.ptx["cp.async.cg.shared.global"](
            T.address_of(shared[lane * 4]),
            T.address_of(source[lane * 4]),
            16,
        )
        T.ptx.cp.async_.commit_group()
        T.ptx.cp.async_.wait_group(0)
    else:
        source[lane * 4] = T.cast(lane, "float32")


def _assert_clean(report) -> None:
    report.require_clean()
    native = report.to_dict()["native"]
    assert native["findings"] == []
    assert native["incomplete"] == []


def _assert_physical_finding(
    report,
    expected_kind: str | set[str],
    *,
    expected_status: str = "error",
) -> dict:
    assert report.verdict == expected_status
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert len(native["findings"]) == 1
    finding = native["findings"][0]
    assert finding["status"] == expected_status
    expected_kinds = {expected_kind} if isinstance(expected_kind, str) else expected_kind
    assert finding["kind"] == (
        "tmem_lifetime_review" if expected_status == "review" else "data_race"
    )
    assert finding["access_pair"] in expected_kinds
    assert finding["overlap"]["byte_len"] > 0
    for witness in (finding["prior"], finding["current"]):
        assert witness["operation"]["source_op_id"] is not None
        assert witness["operation"]["source"]["source_text"].strip()
    return finding


# Only a prior TMEM load can carry the unmodeled register dependency;
# conflicts behind a store frontier are exact errors.


@pytest.mark.parametrize(
    ("mode", "space"),
    [
        (0, "global"),
        (2, "shared"),
    ],
)
def test_classic_cp_async_keeps_exact_lane_accesses_live_until_wait(mode, space):
    report = racecheck(
        native_classic_cp_async_lifetime,
        inputs={
            "mode": np.int32(mode),
            "source": np.zeros(128, dtype=np.float32),
            "output": np.zeros(128, dtype=np.float32),
        },
    )

    assert report.verdict == "error"
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert native["findings"]
    assert {finding["access_pair"] for finding in native["findings"]} <= {
        "read_write",
        "write_read",
    }
    assert {
        endpoint["space"]
        for finding in native["findings"]
        for endpoint in (finding["prior"], finding["current"])
    } == {space}


@pytest.mark.parametrize("mode", [1, 3])
def test_classic_cp_async_wait_group_completes_source_and_destination_accesses(mode):
    _assert_clean(
        racecheck(
            native_classic_cp_async_lifetime,
            inputs={
                "mode": np.int32(mode),
                "source": np.zeros(128, dtype=np.float32),
                "output": np.zeros(128, dtype=np.float32),
            },
        )
    )


def test_classic_cp_async_inherits_prior_issuing_lane_clock():
    _assert_clean(
        racecheck(
            native_classic_cp_async_orders_prior_lane_write,
            inputs={"source": np.zeros(128, dtype=np.float32)},
        )
    )


def test_classic_cp_async_cross_warp_source_race_reports_missing_ordering():
    report = racecheck(
        native_classic_cp_async_cross_warp_source_race,
        inputs={"source": np.zeros(128, dtype=np.float32)},
    )
    assert report.verdict == "error"
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert native["findings"]
    assert {
        (finding["ordering_domain"], finding["ordering_failure"]) for finding in native["findings"]
    } == {("execution", "missing_inter_actor_sync")}
    assert {
        endpoint["space"]
        for finding in native["findings"]
        for endpoint in (finding["prior"], finding["current"])
    } == {"global"}
