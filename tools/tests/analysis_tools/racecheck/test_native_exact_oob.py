from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm.script import tirx as T


@T.prim_func
def native_buffer_selected_lane_oob(
    selected_lane: T.Buffer((32,), "int32"),
    source: T.Buffer((1,), "int32"),
    output: T.Buffer((1,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == selected_lane[lane]:
        output[0] = source[lane]


@T.prim_func
def native_concrete_lane_oob_variants(
    mode: T.int32, bound: T.int32, data_bound: T.Buffer((1,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    direct = T.alloc_buffer((8,), "int32", scope="shared")
    then_buffer = T.alloc_buffer((32,), "int32", scope="shared")
    else_buffer = T.alloc_buffer((8,), "int32", scope="shared")

    if mode == 0:
        if lane < bound:
            direct[lane + 4] = 1
    elif mode == 1:
        if lane < bound:
            direct[7 - lane] = 2
    elif mode == 2:
        nonlinear_bound: T.let = data_bound[0] * data_bound[0]
        if lane < nonlinear_bound:
            direct[lane] = 3
    elif mode == 3:
        if lane < bound:
            then_buffer[lane] = 4
        else:
            else_buffer[lane] = 5
    else:
        if lane < bound:
            direct[lane] = 6


@T.prim_func
def native_warp_tiled_oob(warp_limit: T.int32):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((40,), "int32", scope="shared")
    if (warp < warp_limit) and (lane < 8):
        shared[warp * 16 + lane] = 7


@T.prim_func
def native_boolean_guard_oob(mode: T.int32, bound: T.int32, flag: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "int32", scope="shared")

    if mode == 0:
        if (flag[0] != 0) and (lane < bound):
            shared[lane] = 1
    elif mode == 1:
        if (flag[0] != 0) & (lane < bound):
            shared[lane] = 2
    elif mode == 2:
        if (flag[0] != 0) | (lane < bound):
            shared[lane] = 3
    else:
        if not ((flag[0] == 0) | (lane >= bound)):
            shared[lane] = 4


@T.prim_func
def native_data_guarded_oob(mode: T.int32, enabled: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "int32", scope="shared")

    if mode == 0:
        if enabled[0] != 0:
            shared[lane] = 1
    else:
        if lane < enabled[0]:
            shared[50] = 2


@T.prim_func
def native_numeric_loop_relay_oob(
    limit: T.int32,
    base: T.Buffer((1,), "int32"),
    output: T.Buffer((2,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    index = T.local_scalar("int32")

    for iteration in T.serial(limit):
        if lane == 0:
            index = base[0] + iteration
            output[index] = iteration


def _resource_limits() -> numsim.ResourceLimits:
    return numsim.ResourceLimits(
        max_schedules=16,
        max_backtrack_nodes=100,
        max_events_per_run=1_000,
        max_total_events=10_000,
        max_loop_steps=1_000,
        max_wall_time_ms=30_000,
        max_diagnostic_bytes=1_000_000,
    )


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-exact-oob")


def _selected_lane_inputs(selected_lane: int):
    lane_selectors = np.full(32, -1, dtype=np.int32)
    lane_selectors[selected_lane] = selected_lane
    return {
        "selected_lane": lane_selectors,
        "source": np.array([17], dtype=np.int32),
        "output": np.zeros(1, dtype=np.int32),
    }


def _run(selected_lane: int, cache_dir):
    return racecheck(
        native_buffer_selected_lane_oob,
        inputs=_selected_lane_inputs(selected_lane),
        cache_dir=cache_dir,
    )


def _run_variant(*, mode: int, bound: int, data_bound: int, cache_dir):
    return racecheck(
        native_concrete_lane_oob_variants,
        inputs={
            "mode": np.int32(mode),
            "bound": np.int32(bound),
            "data_bound": np.array([data_bound], dtype=np.int32),
        },
        cache_dir=cache_dir,
    )


def _run_warp_tiled(warp_limit: int, cache_dir):
    return racecheck(
        native_warp_tiled_oob,
        inputs={"warp_limit": np.int32(warp_limit)},
        cache_dir=cache_dir,
    )


def _run_boolean_guard(*, mode: int, flag: int, bound: int, cache_dir):
    return racecheck(
        native_boolean_guard_oob,
        inputs={
            "mode": np.int32(mode),
            "bound": np.int32(bound),
            "flag": np.array([flag], dtype=np.int32),
        },
        cache_dir=cache_dir,
    )


def _run_data_guarded_oob(*, mode: int, enabled: int, cache_dir):
    return racecheck(
        native_data_guarded_oob,
        inputs={
            "mode": np.int32(mode),
            "enabled": np.array([enabled], dtype=np.int32),
        },
        cache_dir=cache_dir,
    )


def _direct_clean_payload(report):
    report.require_clean()
    assert report.verdict == "clean"
    assert report.findings == []
    payload = report.to_dict()["native"]
    assert payload["incomplete"] == []
    assert payload["execution_error"] is None
    assert "search" not in payload
    assert "counterexample" not in payload
    assert payload["stats"]["available"] is True
    return payload


def _direct_oob_payload(report):
    assert report.verdict == "error"
    assert [(finding.status, finding.kind) for finding in report.findings] == [("error", "oob")]
    payload = report.to_dict()["native"]
    assert payload["findings"] == []
    assert payload["incomplete"] == []
    assert payload["execution_error"]["kind"] == "oob"
    assert "search" not in payload
    assert "counterexample" not in payload
    return payload


def test_public_native_racecheck_resolves_buffer_selected_oob_exactly(native_cache_dir):
    active_oob = _run(1, native_cache_dir)
    masked_clean = _run(0, native_cache_dir)

    active_native = _direct_oob_payload(active_oob)
    assert "lane 1" in active_native["execution_error"]["message"]

    masked_native = _direct_clean_payload(masked_clean)

    expected_bindings = ["output", "selected_lane", "source"]
    assert active_native["input"]["bindings"] == expected_bindings
    assert masked_native["input"]["bindings"] == expected_bindings
    assert active_native["engine"]["mode"] == "native"
    assert masked_native["engine"]["mode"] == "native"
    assert active_native["engine"]["artifact_key"]
    assert masked_native["engine"]["artifact_key"]
    assert active_native["engine"]["artifact_key"] == masked_native["engine"]["artifact_key"]
    assert active_native["input"]["digest"] != masked_native["input"]["digest"]


def test_native_racecheck_and_synccheck_scope_typed_lane_predicate_oob(native_cache_dir):
    clean_sync = synccheck(
        native_buffer_selected_lane_oob,
        inputs=_selected_lane_inputs(0),
        cache_dir=native_cache_dir,
        coverage_bounds=numsim.CoverageBounds(0, 0),
        resource_limits=_resource_limits(),
        max_workers=1,
    )
    error_sync = synccheck(
        native_buffer_selected_lane_oob,
        inputs=_selected_lane_inputs(1),
        cache_dir=native_cache_dir,
        coverage_bounds=numsim.CoverageBounds(0, 0),
        resource_limits=_resource_limits(),
        max_workers=1,
    )
    clean_sync.require_clean()
    assert clean_sync.findings == []
    assert error_sync.verdict == "error"
    assert [(finding.status, finding.kind) for finding in error_sync.findings] == [("error", "oob")]
    error_sync_native = error_sync.to_dict()["native"]
    assert error_sync_native["findings"] == []
    assert error_sync_native["incomplete"] == []
    assert error_sync_native["execution_error"]["kind"] == "oob"
    assert "lane 1" in error_sync_native["execution_error"]["message"]


def test_public_native_synccheck_executes_numeric_loop_address_relay(native_cache_dir):
    def run(base: int):
        return synccheck(
            native_numeric_loop_relay_oob,
            inputs={
                "limit": np.int32(2),
                "base": np.array([base], dtype=np.int32),
                "output": np.zeros(2, dtype=np.int32),
            },
            cache_dir=native_cache_dir,
            coverage_bounds=numsim.CoverageBounds(0, 0),
            resource_limits=_resource_limits(),
            max_workers=1,
        )

    run(0).require_clean()
    error = run(1)
    assert error.verdict == "error"
    assert [(finding.status, finding.kind) for finding in error.findings] == [("error", "oob")]
    native = error.to_dict()["native"]
    assert native["incomplete"] == []
    assert native["execution_error"]["kind"] == "oob"
    assert "lane 0" in native["execution_error"]["message"]


@pytest.mark.parametrize(
    ("mode", "clean_bound", "error_bound", "clean_data_bound", "error_data_bound", "error_lane"),
    [
        pytest.param(0, 4, 5, 0, 0, 4, id="offset"),
        pytest.param(1, 8, 9, 0, 0, 8, id="decreasing"),
        pytest.param(2, 0, 0, 2, 3, 8, id="nonlinear-data-bound"),
        pytest.param(3, 32, 8, 0, 0, 8, id="then-else-complement"),
        pytest.param(4, 8, 9, 0, 0, 8, id="single-warp-lane-guard"),
    ],
)
def test_public_native_racecheck_resolves_concrete_lane_oob_variants_exactly(
    mode: int,
    clean_bound: int,
    error_bound: int,
    clean_data_bound: int,
    error_data_bound: int,
    error_lane: int,
    native_cache_dir,
):
    clean = _run_variant(
        mode=mode, bound=clean_bound, data_bound=clean_data_bound, cache_dir=native_cache_dir
    )
    error = _run_variant(
        mode=mode, bound=error_bound, data_bound=error_data_bound, cache_dir=native_cache_dir
    )

    clean_native = _direct_clean_payload(clean)
    error_native = _direct_oob_payload(error)
    assert f"lane {error_lane}" in error_native["execution_error"]["message"]

    assert error_native["engine"]["artifact_key"] == clean_native["engine"]["artifact_key"]
    assert error_native["input"]["digest"] != clean_native["input"]["digest"]


def test_public_native_racecheck_resolves_warp_tiled_oob_exactly(native_cache_dir):
    clean = _run_warp_tiled(3, native_cache_dir)
    error = _run_warp_tiled(4, native_cache_dir)

    clean_native = _direct_clean_payload(clean)
    error_native = _direct_oob_payload(error)
    assert "warp 3 failed" in error_native["execution_error"]["message"]
    assert "lane 0" in error_native["execution_error"]["message"]
    assert error_native["execution_model"] == "direct_online_vc"
    assert error_native["stats"]["task_count"] == 4
    assert error_native["stats"]["completed_task_count"] == 3

    assert error_native["engine"]["artifact_key"] == clean_native["engine"]["artifact_key"]
    assert error_native["input"]["digest"] != clean_native["input"]["digest"]
    assert clean_native["input"]["scalars"]["warp_limit"]["value"]["decimal"] == "3"
    assert error_native["input"]["scalars"]["warp_limit"]["value"]["decimal"] == "4"


@pytest.mark.parametrize(
    ("mode", "clean_flag", "clean_bound", "error_flag", "error_bound"),
    [
        pytest.param(0, 1, 8, 1, 9, id="logical-and"),
        pytest.param(1, 1, 8, 1, 9, id="boolean-bitwise-and"),
        pytest.param(2, 0, 8, 1, 0, id="boolean-bitwise-or"),
        pytest.param(3, 1, 8, 1, 9, id="not-of-or"),
    ],
)
def test_public_native_racecheck_resolves_boolean_guard_spellings_exactly(
    mode: int,
    clean_flag: int,
    clean_bound: int,
    error_flag: int,
    error_bound: int,
    native_cache_dir,
):
    clean = _run_boolean_guard(
        mode=mode, flag=clean_flag, bound=clean_bound, cache_dir=native_cache_dir
    )
    error = _run_boolean_guard(
        mode=mode, flag=error_flag, bound=error_bound, cache_dir=native_cache_dir
    )

    clean_native = _direct_clean_payload(clean)
    error_native = _direct_oob_payload(error)
    assert "lane 8" in error_native["execution_error"]["message"]

    assert error_native["engine"]["artifact_key"] == clean_native["engine"]["artifact_key"]
    assert error_native["input"]["digest"] != clean_native["input"]["digest"]


@pytest.mark.parametrize(
    ("mode", "error_lane"),
    [
        pytest.param(0, 8, id="pure-data-guard"),
        pytest.param(1, 0, id="constant-index-under-data-derived-lane-guard"),
    ],
)
def test_public_native_racecheck_resolves_data_guarded_oob_exactly(
    mode: int, error_lane: int, native_cache_dir
):
    clean = _run_data_guarded_oob(mode=mode, enabled=0, cache_dir=native_cache_dir)
    error = _run_data_guarded_oob(mode=mode, enabled=1, cache_dir=native_cache_dir)

    clean_native = _direct_clean_payload(clean)
    error_native = _direct_oob_payload(error)
    assert f"lane {error_lane}" in error_native["execution_error"]["message"]

    assert error_native["engine"]["artifact_key"] == clean_native["engine"]["artifact_key"]
    assert error_native["input"]["digest"] != clean_native["input"]["digest"]


def test_public_native_racecheck_missing_required_binding_is_typed_incomplete(native_cache_dir):
    complete = _run_variant(mode=2, bound=0, data_bound=2, cache_dir=native_cache_dir)
    missing = racecheck(
        native_concrete_lane_oob_variants,
        inputs={"mode": np.int32(2), "bound": np.int32(0)},
        cache_dir=native_cache_dir,
    )

    complete.require_clean()
    complete_native = complete.to_dict()["native"]

    assert missing.verdict == "incomplete"
    assert [(finding.status, finding.kind) for finding in missing.findings] == [
        ("incomplete", "analysis_incomplete")
    ]
    missing_native = missing.to_dict()["native"]
    assert missing_native["verdict"] == "incomplete"
    assert missing_native["findings"] == []
    [incomplete] = missing_native["incomplete"]
    source_anchor = incomplete["source_anchor"]
    assert {key: value for key, value in incomplete.items() if key != "source_anchor"} == {
        "kind": "analysis_incomplete", "reason": "missing_input_bindings",
        "message": "native analysis requires complete concrete bindings before execution",
        "bindings": ["data_bound"],
    }
    assert source_anchor["scope"] == "kernel"
    assert source_anchor["kernel_index"] == 0
    assert source_anchor["source_text"] == "kernel native_concrete_lane_oob_variants"
    source_span = source_anchor["source_span"]
    source_path = Path(source_span["source_name"]).resolve()
    assert source_path == Path(__file__).resolve()
    assert (
        "def native_concrete_lane_oob_variants("
        in source_path.read_text().splitlines()[source_span["line"] - 1]
    )
    assert missing_native["access_count"] == 0
    assert missing_native["accesses"] == []
    assert missing_native["execution_error"] is None
    assert missing_native.get("counterexample") is None
    assert missing_native["coverage"] == {
        "status": "not_started",
        "eligible_for_clean": False,
        "termination": {"kind": "missing_input_bindings"},
    }
    assert missing_native["input"]["digest"] is None
    assert missing_native["input"]["bindings"] == ["bound", "mode"]
    assert missing_native["engine"]["artifact_key"] == complete_native["engine"]["artifact_key"]
