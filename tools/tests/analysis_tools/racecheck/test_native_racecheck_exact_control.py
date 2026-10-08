"""Native Racecheck integration coverage."""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tests.analysis_tools.racecheck._native_race_trace import full_direct_race_run
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx


@T.prim_func
def native_racecheck_exact_data_dependent_control(
    mode: T.int32,
    if_limit: T.int32,
    for_extents: T.Buffer((32,), "int32"),
    while_extents: T.Buffer((32,), "int32"),
    source: T.Buffer((4,), "int32"),
    output: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    iteration = T.local_scalar("int32")
    iteration = T.int32(0)

    if mode == 0:
        if lane < if_limit:
            output[lane] = source[lane]
    elif mode == 1:
        for step in T.serial(for_extents[lane]):
            if lane < 4:
                output[lane] = source[lane + step]
    else:
        while iteration < while_extents[lane]:
            if lane < 4:
                output[lane] = source[lane + iteration]
            iteration = iteration + 1


@T.prim_func
def native_racecheck_data_guarded_cross_lane(
    enabled: T.Buffer((1,), "int32"), output: T.Buffer((32,), "int32")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "int32", scope="shared")

    if warp == 0:
        shared[lane] = 0
    T.ptx.bar.sync(T.uint32(6), T.uint32(64))
    if (warp == 0) and (enabled[0] != 0):
        shared[lane] = lane + 1
    if warp == 1:
        output[lane] = shared[(lane + 1) % 32]


@T.prim_func
def native_racecheck_tile_address_from_local_scalar(output: T.Buffer((8,), "float32")):
    T.device_entry()
    cluster = T.cluster_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    destination_start = T.local_scalar("int32")

    if lane == 0:
        destination_start = cluster * 4
    if lane < 4:
        shared[lane] = T.cast(cluster * 4 + lane, "float32")
    T.ptx.fence.proxy.async_.shared__cta()
    T.cuda.cta_sync()
    if lane == 0:
        Tx.copy_async(
            output[destination_start : destination_start + 4],
            shared[:],
            dispatch="tma_auto",
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-racecheck-exact-control")


def _inputs(
    *,
    mode: int,
    if_limit: int = 0,
    for_extents: np.ndarray | None = None,
    while_extents: np.ndarray | None = None,
):
    if for_extents is None:
        for_extents = np.zeros(32, dtype=np.int32)
    if while_extents is None:
        while_extents = np.zeros(32, dtype=np.int32)
    return {
        "mode": np.int32(mode),
        "if_limit": np.int32(if_limit),
        "for_extents": for_extents,
        "while_extents": while_extents,
        "source": np.arange(4, dtype=np.int32),
        "output": np.zeros(32, dtype=np.int32),
    }


def _run(inputs, cache_dir):
    return racecheck(
        native_racecheck_exact_data_dependent_control,
        inputs=inputs,
        cache_dir=cache_dir,
    )


def _run_cross_lane(enabled: int, cache_dir):
    return racecheck(
        native_racecheck_data_guarded_cross_lane,
        inputs={
            "enabled": np.array([enabled], dtype=np.int32),
            "output": np.zeros(32, dtype=np.int32),
        },
        cache_dir=cache_dir,
    )


def _assert_exact_clean_error_pair(clean, error):
    assert clean.verdict == "clean"
    assert clean.findings == []
    clean_native = clean.to_dict()["native"]
    assert clean_native["verdict"] == "clean"
    assert clean_native["findings"] == []
    assert clean_native["incomplete"] == []

    assert error.verdict == "error"
    assert [finding.kind for finding in error.findings] == ["oob"]
    assert {finding.status for finding in error.findings} == {"error"}
    error_native = error.to_dict()["native"]
    assert error_native["verdict"] == "error"
    assert error_native["incomplete"] == []
    replay = error_native
    assert replay["verdict"] == "error"
    assert replay["incomplete"] == []
    assert replay["execution_error"]["kind"] == "oob"
    assert replay["accesses_complete"] is False
    assert replay["accesses"] == []

    assert error_native["engine"]["artifact_key"] == clean_native["engine"]["artifact_key"]
    assert error_native["input"]["digest"] != clean_native["input"]["digest"]
    return clean_native, error_native, replay


def _loop_payload_accesses(replay):
    return [
        access
        for access in replay["accesses"]
        if access["space"] == "global" and access["operation"]["loop_frames"]
    ]


def test_public_native_racecheck_resolves_scalar_lane_guard_exactly(native_cache_dir):
    clean = _run(_inputs(mode=0, if_limit=4), native_cache_dir)
    error = _run(_inputs(mode=0, if_limit=5), native_cache_dir)

    clean_native, error_native, replay = _assert_exact_clean_error_pair(clean, error)

    assert clean_native["input"]["scalars"]["if_limit"]["value"]["decimal"] == "4"
    assert error_native["input"]["scalars"]["if_limit"]["value"]["decimal"] == "5"
    assert [access for access in replay["accesses"] if access["space"] == "global"] == []


def test_public_native_racecheck_executes_lane_varying_dynamic_for_exactly(native_cache_dir):
    clean_extents = np.zeros(32, dtype=np.int32)
    clean_extents[:4] = 1
    error_extents = clean_extents.copy()
    error_extents[3] = 2

    clean_inputs = _inputs(mode=1, for_extents=clean_extents)
    error_inputs = _inputs(mode=1, for_extents=error_extents)
    clean = _run(clean_inputs, native_cache_dir)
    error = _run(error_inputs, native_cache_dir)

    _clean_native, error_native, _compact_replay = _assert_exact_clean_error_pair(clean, error)
    replay = full_direct_race_run(
        native_racecheck_exact_data_dependent_control,
        error_inputs,
        native_cache_dir,
        error_native,
    )

    payload_accesses = _loop_payload_accesses(replay)
    assert [access["access_kind"] for access in payload_accesses] == ["read", "write"]
    assert [access["active_lane_count"] for access in payload_accesses] == [4, 4]
    assert [
        access["operation"]["loop_frames"][0]["iteration_ordinal"] for access in payload_accesses
    ] == [0, 0]


def test_public_native_racecheck_executes_lane_varying_dynamic_while_exactly(native_cache_dir):
    clean_extents = np.zeros(32, dtype=np.int32)
    clean_extents[:4] = 1
    error_extents = clean_extents.copy()
    error_extents[3] = 2

    clean_inputs = _inputs(mode=2, while_extents=clean_extents)
    error_inputs = _inputs(mode=2, while_extents=error_extents)
    clean = _run(clean_inputs, native_cache_dir)
    error = _run(error_inputs, native_cache_dir)

    _clean_native, error_native, _compact_replay = _assert_exact_clean_error_pair(clean, error)
    replay = full_direct_race_run(
        native_racecheck_exact_data_dependent_control,
        error_inputs,
        native_cache_dir,
        error_native,
    )

    payload_accesses = _loop_payload_accesses(replay)
    assert [access["access_kind"] for access in payload_accesses] == ["read", "write"]
    assert [access["active_lane_count"] for access in payload_accesses] == [4, 4]
    assert [
        access["operation"]["loop_frames"][0]["iteration_ordinal"] for access in payload_accesses
    ] == [0, 0]


def test_public_native_racecheck_resolves_data_guarded_cross_lane_hazard_exactly(native_cache_dir):
    clean = _run_cross_lane(0, native_cache_dir)
    error = _run_cross_lane(1, native_cache_dir)

    clean.require_clean()
    assert clean.findings == []
    clean_native = clean.to_dict()["native"]
    assert clean_native["incomplete"] == []

    assert error.verdict == "error"
    error_native = error.to_dict()["native"]
    assert [(finding.status, finding.kind) for finding in error.findings] == [
        ("error", error_native["findings"][0]["kind"])
    ]
    assert error_native["incomplete"] == []
    replay = error_native
    assert replay["verdict"] == "error"
    assert replay["incomplete"] == []
    assert len(replay["findings"]) == 1
    finding = replay["findings"][0]
    assert finding["access_pair"] in {"write_read", "read_write"}
    assert finding["prior"]["space"] == "shared"
    assert finding["current"]["space"] == "shared"
    assert finding["overlap"]["byte_len"] == 4

    assert error_native["engine"]["artifact_key"] == clean_native["engine"]["artifact_key"]
    assert error_native["input"]["digest"] != clean_native["input"]["digest"]


def test_public_native_racecheck_preserves_tile_address_local_scalars(native_cache_dir):
    inputs = {"output": np.zeros(8, dtype=np.float32)}
    report = racecheck(
        native_racecheck_tile_address_from_local_scalar,
        inputs=inputs,
        cache_dir=native_cache_dir,
    )

    report.require_clean()
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    replay = full_direct_race_run(
        native_racecheck_tile_address_from_local_scalar,
        inputs,
        native_cache_dir,
        native,
    )
    global_writes = [
        access
        for access in replay["accesses"]
        if access["space"] == "global" and access["access_kind"] == "write"
    ]
    assert len(global_writes) == 8
    assert sorted(
        access["lanes"][0]["spans"][0]["byte_offset"] for access in global_writes
    ) == list(range(0, 32, 4))
