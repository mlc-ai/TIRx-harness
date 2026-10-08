from __future__ import annotations

import re

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.transpiler import suspend_scaffold
from tvm.script import tirx as T


@T.prim_func
def two_direct_output_writers(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = lane
    output[lane] = lane + 100


@T.prim_func
def root_uniform_output_writers(output: T.Buffer((64,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    if warp == 0:
        output[lane] = lane
    else:
        output[32 + lane] = lane + 100


@T.prim_func
def root_sync_output_writers(output: T.Buffer((64,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = lane
    output[32 + lane] = lane + 100


@T.prim_func
def repeated_split_output_writers(output: T.Buffer((96,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[32 + lane] = lane + 7
    output[64 + lane] = lane + 11


def test_shared_split_preserves_each_writer_and_cached_module(tmp_path, monkeypatch):
    monkeypatch.setattr(suspend_scaffold, "_ROOT_SYNC_SPLIT_MIN_LINES", 0)
    monkeypatch.setattr(suspend_scaffold, "_ROOT_SYNC_SPLIT_TARGET_LINES", 1)
    module = numsim.transpile(repeated_split_output_writers, cache_dir=tmp_path)
    expected = np.concatenate((np.zeros(32, dtype=np.int32), np.arange(7, 39), np.arange(11, 43)))
    for compiled in (module, numsim.transpile(repeated_split_output_writers, cache_dir=tmp_path)):
        result = numsim.Engine().run(compiled, {"output": np.zeros(96, dtype=np.int32)})
        np.testing.assert_array_equal(result.outputs["output"], expected)


@T.prim_func
def repeated_async_exchange(output: T.Buffer((96,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_shared((96,), "int32")
    if warp == 0:
        shared[32 + lane] = lane + 7
        T.cuda.cta_sync()
        output[32 + lane] = shared[64 + lane] + 5
    else:
        shared[64 + lane] = lane + 11
        T.cuda.cta_sync()
        output[64 + lane] = shared[32 + lane] + 5


def test_shared_async_split_preserves_exchange_and_writers(tmp_path, monkeypatch):
    monkeypatch.setattr(suspend_scaffold, "_ROOT_UNIFORM_IF_SPLIT_MIN_LINES", 0)
    monkeypatch.setattr(suspend_scaffold, "_INNER_ASYNC_SPLIT_MIN_LINES", 10**9)
    compiled = numsim.transpile(repeated_async_exchange, cache_dir=tmp_path)
    expected = np.concatenate((np.zeros(32, dtype=np.int32), np.arange(16, 48), np.arange(12, 44)))
    for workers in (1, 2):
        result = numsim.Engine(max_workers=workers).run(
            compiled, {"output": np.zeros(96, dtype=np.int32)}
        )
        np.testing.assert_array_equal(result.outputs["output"], expected)


@T.prim_func
def inner_split_control_output_writers(output: T.Buffer((64,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    uniform_live_in: T.let = (warp // 2) * 2
    if warp == 0:
        live_in: T.let = lane + 7
        output[lane] = live_in + uniform_live_in
        for _outer in T.serial(1):
            for _inner in T.serial(1):
                T.cuda.warp_sync()
                if lane < 16:
                    continue
                output[lane] = live_in + 1
        for _outer in T.serial(1):
            if warp == 0:
                output[lane] = live_in + 1
            else:
                output[lane] = -1
        for _outer in T.serial(1):
            for step in T.serial(2):
                T.cuda.warp_sync()
                if step == 1:
                    break
                output[lane] = output[lane] + 1
    else:
        output[32 + lane] = lane + 100


@T.prim_func
def depth_two_inner_sequence_output_writers(output: T.Buffer((64,), "int32"), lane_limit: T.int32):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    if warp == 0:
        if lane < lane_limit:
            for _outer in T.serial(1):
                for _inner in T.serial(1):
                    for _prefix in T.serial(1):
                        output[lane] = lane
                    T.cuda.warp_sync()
                    output[lane] = output[lane] + 1
        else:
            output[lane] = lane + 50
    else:
        output[32 + lane] = lane + 100


@T.prim_func
def inner_while_sync_output_writers(output: T.Buffer((64,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    if warp == 0:
        for _outer in T.serial(1):
            while output[lane] == 0:
                T.cuda.warp_sync()
                output[lane] = lane + 1
    else:
        output[32 + lane] = lane + 100


@T.prim_func
def nested_live_in_output_writers(output: T.Buffer((64,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    outer_live_in: T.let = lane + 7
    if warp == 0:
        for _outer in T.serial(1):
            while output[lane] == 0:
                T.cuda.warp_sync()
                if warp == 0:
                    if warp < 1:
                        output[lane] = outer_live_in + 1
                    else:
                        output[lane] = -2
                else:
                    output[lane] = -1
    else:
        output[32 + lane] = lane + 100


def _rust_function(source: str, name: str) -> str:
    lines = source.splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line.removeprefix("pub(super) ").startswith((f"fn {name}(", f"async fn {name}("))
    )
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if re.match(r"^(?:pub\(super\) )?(?:async )?fn [A-Za-z0-9_]+\(", lines[index])
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_injected_mismatch_reports_buffer_index_and_values(tmp_path):
    module = numsim.transpile(two_direct_output_writers, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.int32)})
    expected = result.outputs["output"].copy()
    expected[7] += 1

    report = numsim.compare(result, {"output": expected})

    assert not report.ok
    assert len(report.mismatches) == 1
    mismatch = report.mismatches[0]
    assert mismatch.output == "output"
    assert mismatch.index == (7,)
    assert mismatch.actual == 107
    assert mismatch.expected == 108

    with pytest.raises(AssertionError) as captured:
        report.require_ok()
    message = str(captured.value)
    assert "first mismatch" in message
    assert "output='output' index=(7,) actual=107 expected=108" in message


def test_split_root_uniform_arms_keep_helper_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(suspend_scaffold, "_ROOT_UNIFORM_IF_SPLIT_MIN_LINES", 0)
    module = numsim.transpile(root_uniform_output_writers, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(64, dtype=np.int32)})
    expected = np.concatenate((np.arange(32, dtype=np.int32), np.arange(100, 132, dtype=np.int32)))
    np.testing.assert_array_equal(result.outputs["output"], expected)

    assert "kernel_0_root_if_0_then" in module.rust_source
    assert "kernel_0_root_if_0_else" in module.rust_source


def test_small_root_uniform_if_rolls_back_partition_probe(tmp_path):
    module = numsim.transpile(root_uniform_output_writers, cache_dir=tmp_path)
    numsim.Engine().run(module, {"output": np.zeros(64, dtype=np.int32)})

    assert "kernel_0_root_if_0_then" not in module.rust_source


def test_split_root_sync_run_keeps_helper_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(suspend_scaffold, "_ROOT_SYNC_SPLIT_MIN_LINES", 0)
    module = numsim.transpile(root_sync_output_writers, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(64, dtype=np.int32)})
    expected = np.concatenate((np.arange(32, dtype=np.int32), np.arange(100, 132, dtype=np.int32)))
    np.testing.assert_array_equal(result.outputs["output"], expected)

    assert "kernel_0_root_sync_0" in module.rust_source


def test_large_root_sync_run_splits_at_statement_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(suspend_scaffold, "_ROOT_SYNC_SPLIT_MIN_LINES", 0)
    monkeypatch.setattr(suspend_scaffold, "_ROOT_SYNC_SPLIT_TARGET_LINES", 1)
    module = numsim.transpile(root_sync_output_writers, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(64, dtype=np.int32)})

    expected = np.concatenate((np.arange(32, dtype=np.int32), np.arange(100, 132)))
    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert "kernel_0_root_sync_0" in module.rust_source
    assert "kernel_0_root_sync_1" in module.rust_source


def test_small_root_sync_run_rolls_back_partition_probe(tmp_path):
    module = numsim.transpile(root_sync_output_writers, cache_dir=tmp_path)
    numsim.Engine().run(module, {"output": np.zeros(64, dtype=np.int32)})

    assert "kernel_0_root_sync_0" not in module.rust_source


def test_inner_splits_preserve_output(tmp_path, monkeypatch):
    monkeypatch.setattr(suspend_scaffold, "_ROOT_UNIFORM_IF_SPLIT_MIN_LINES", 0)
    monkeypatch.setattr(suspend_scaffold, "_INNER_ASYNC_SPLIT_MIN_LINES", 0)
    monkeypatch.setattr(suspend_scaffold, "_INNER_UNIFORM_IF_SPLIT_MIN_LINES", 0)
    module = numsim.transpile(inner_split_control_output_writers, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(64, dtype=np.int32)})
    expected = np.concatenate((np.arange(9, 41, dtype=np.int32), np.arange(100, 132)))
    np.testing.assert_array_equal(result.outputs["output"], expected)

    assert "kernel_0_inner_async_0" in module.rust_source
    assert "kernel_0_inner_if_0_then" in module.rust_source

    cached_module = numsim.transpile(inner_split_control_output_writers, cache_dir=tmp_path)
    assert cached_module.cache_key == module.cache_key


def test_oversized_inner_helper_splits_subthreshold_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(suspend_scaffold, "_ROOT_UNIFORM_IF_SPLIT_MIN_LINES", 0)
    monkeypatch.setattr(suspend_scaffold, "_INNER_VARYING_IF_SPLIT_MIN_LINES", 0)
    monkeypatch.setattr(suspend_scaffold, "_INNER_ASYNC_SPLIT_MIN_LINES", 10**9)
    monkeypatch.setattr(suspend_scaffold, "_INNER_ASYNC_SPLIT_TARGET_LINES", 1)

    module = numsim.transpile(depth_two_inner_sequence_output_writers, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"output": np.zeros(64, dtype=np.int32), "lane_limit": 32},
    )

    expected = np.concatenate(
        (
            np.arange(1, 33, dtype=np.int32),
            np.arange(100, 132, dtype=np.int32),
        )
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


@pytest.mark.parametrize("checker", ("synccheck", "racecheck"))
def test_analysis_inner_split_without_suspend_is_synchronous(
    checker, tmp_path, monkeypatch
):
    monkeypatch.setattr(suspend_scaffold, "_ROOT_UNIFORM_IF_SPLIT_MIN_LINES", 0)
    monkeypatch.setattr(suspend_scaffold, "_INNER_VARYING_IF_SPLIT_MIN_LINES", 0)
    monkeypatch.setattr(suspend_scaffold, "_INNER_ASYNC_SPLIT_MIN_LINES", 10**9)
    monkeypatch.setattr(suspend_scaffold, "_ANALYSIS_INNER_ASYNC_SPLIT_TARGET_LINES", 1)

    module = numsim.transpile(
        depth_two_inner_sequence_output_writers,
        cache_dir=tmp_path,
        _analysis_checker=checker,
        _default_generated_opt_level=0,
    )
    inputs = {"output": np.zeros(64, dtype=np.int32), "lane_limit": 32}
    engine = numsim.Engine()
    if checker == "synccheck":
        result = engine.run_synccheck_phase(
            module,
            inputs,
            coverage_bounds=numsim.CoverageBounds(0, 0),
        )
    else:
        result = engine.run_racecheck_phase(module, inputs)

    payload = result.to_dict()
    assert result.verdict == "clean", payload
    assert result.findings == [], payload
    assert result.advisories == [], payload
    assert payload["incomplete"] == [], payload
    assert payload["execution_error"] is None, payload
    if checker == "synccheck":
        assert payload["coverage"]["status"] == "complete_within_bounds", payload
    assert payload["stats"]["completed_task_count"] == payload["stats"]["task_count"], payload
    sync_helpers = re.findall(
        r"(?m)^pub\(super\) fn (kernel_0_inner_async_\d+)\(", module.rust_source
    )
    assert sync_helpers
    for helper in sync_helpers:
        assert f"mod numsim_split_{helper} {{" in module.rust_source
        assert f"use numsim_split_{helper}::{helper};" in module.rust_source
        call = re.search(
            rf"ctx = {re.escape(helper)}\([\s\S]*?\n\s*(\)\?;|\)\.await\?;)",
            module.rust_source,
        )
        assert call is not None
        assert call.group(1) == ")?;"
    async_helpers = re.findall(
        r"(?m)^pub\(super\) async fn (kernel_0_(?:inner|root)[A-Za-z0-9_]+)\(",
        module.rust_source,
    )
    assert async_helpers
    for helper in async_helpers:
        assert f"mod numsim_split_{helper} {{" in module.rust_source
        assert f"use numsim_split_{helper}::{helper};" in module.rust_source
        assert re.search(
            rf"ctx = NumSimFuture_{re.escape(helper)}\({re.escape(helper)}\([\s\S]*?\)\)\.await\?;",
            module.rust_source,
        )
    assert module.rust_source.startswith("#![allow(warnings)]\n#![feature(optimize_attribute)]")
    assert "#[inline(never)]\n#[optimize(size)]\nasync fn warp_main(" in module.rust_source
    assert "NumSimModuleFuture(warp_main(" in module.rust_source


def test_forced_splits_compile_and_run(tmp_path, monkeypatch):
    thresholds = (
        "_ROOT_UNIFORM_IF_SPLIT_MIN_LINES",
        "_INNER_ASYNC_SPLIT_MIN_LINES",
        "_INNER_UNIFORM_IF_SPLIT_MIN_LINES",
    )
    expected = np.concatenate((np.arange(8, 40, dtype=np.int32), np.arange(100, 132)))

    for name in thresholds:
        monkeypatch.setattr(suspend_scaffold, name, 10**9)
    baseline = numsim.transpile(nested_live_in_output_writers, cache_dir=tmp_path)
    baseline_run = numsim.Engine().run(baseline, {"output": np.zeros(64, dtype=np.int32)})

    for name in thresholds:
        monkeypatch.setattr(suspend_scaffold, name, 0)
    split = numsim.transpile(nested_live_in_output_writers, cache_dir=tmp_path)
    split_run = numsim.Engine().run(split, {"output": np.zeros(64, dtype=np.int32)})

    np.testing.assert_array_equal(baseline_run.outputs["output"], expected)
    np.testing.assert_array_equal(split_run.outputs["output"], expected)


def test_inner_split_preserves_while_execution(monkeypatch, tmp_path):
    baseline = numsim.transpile(inner_while_sync_output_writers, cache_dir=tmp_path / "baseline")
    baseline_run = numsim.Engine().run(baseline, {"output": np.zeros(64, dtype=np.int32)})

    monkeypatch.setattr(suspend_scaffold, "_ROOT_UNIFORM_IF_SPLIT_MIN_LINES", 0)
    monkeypatch.setattr(suspend_scaffold, "_INNER_ASYNC_SPLIT_MIN_LINES", 0)
    split = numsim.transpile(inner_while_sync_output_writers, cache_dir=tmp_path / "split")
    split_run = numsim.Engine().run(split, {"output": np.zeros(64, dtype=np.int32)})

    expected = np.concatenate(
        (np.arange(1, 33, dtype=np.int32), np.arange(100, 132, dtype=np.int32))
    )
    np.testing.assert_array_equal(baseline_run.outputs["output"], expected)
    np.testing.assert_array_equal(split_run.outputs["output"], expected)


@T.prim_func
def varying_while_transfers(output: T.Buffer((32,), "int32")):
    T.device_entry()
    warp = T.warp_id([1])
    lane = T.lane_id([32])
    counter = T.alloc_local((1,), "int32")
    if warp == 0:
        counter[0] = 0
        output[lane] = 0
        while counter[0] < 5:
            counter[0] = counter[0] + 1
            if counter[0] > lane % 3 + 1:
                break
            if counter[0] == 2:
                continue
            output[lane] = output[lane] + counter[0]
        output[lane] = output[lane] + counter[0] * 10


def test_while_split_preserves_varying_break_and_continue(monkeypatch, tmp_path):
    from tirx_harness import racecheck, synccheck

    monkeypatch.setattr(suspend_scaffold, "_ROOT_UNIFORM_IF_SPLIT_MIN_LINES", 0)
    monkeypatch.setattr(suspend_scaffold, "_INNER_ASYNC_SPLIT_MIN_LINES", 0)
    monkeypatch.setattr(suspend_scaffold, "_INNER_ASYNC_SPLIT_TARGET_LINES", 1)
    monkeypatch.setattr(suspend_scaffold, "_ANALYSIS_INNER_ASYNC_SPLIT_TARGET_LINES", 1)
    module = numsim.transpile(varying_while_transfers, cache_dir=tmp_path)
    assert "&mut WarpMask" in module.rust_source
    inputs = {"output": np.zeros(32, dtype=np.int32)}
    result = numsim.Engine().run(module, inputs)
    expected = np.take(np.array([21, 31, 44], dtype=np.int32), np.arange(32) % 3)
    np.testing.assert_array_equal(result.outputs["output"], expected)
    for checker in (synccheck, racecheck):
        checker(varying_while_transfers, inputs).require_clean()
