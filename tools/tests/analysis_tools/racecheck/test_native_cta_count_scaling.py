"""Racecheck's cost must stay linear in the number of CTAs a launch has."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pytest

from tirx_harness import numsim
from tvm_tirx.script import tirx as T

_CTAS = 16384
_THREADS = 32
_WORKERS = 4
# 2026-11-25 UTC, fs-mbz-gpu-645 (shared CPU node, load average about 30;
# the mega_moe racecheck budgets run 2-4x over there): 0.92s alone (-n 1),
# 2.42s inside the full suite (-n 24, worksteal). The cap is about 2x that
# maximum for the node's load. main 4586c36 takes 478s on this case.
_LIMIT_SECONDS = 5.0


@T.function
def many_cta_store(output: T.Tensor((_CTAS * _THREADS,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=_CTAS, block=_THREADS))
    idx = T.cuda.block_idx("x") * _THREADS + T.cuda.thread_idx("x")
    output[idx] = idx


@pytest.mark.performance
def test_racecheck_phase_is_linear_in_cta_count(tmp_path_factory, monkeypatch, record_property):
    # One racecheck shard per cluster used to build a launch-wide actor table
    # (dropped right away), and the end-of-launch register-pool check scanned
    # every warpgroup of the launch per CTA: both quadratic in the CTA count.
    # At 16384 single-warp CTAs that was more than 300s of setup; linear, the
    # phase takes a few seconds.
    monkeypatch.delenv("NUMSIM_PROFILE", raising=False)
    compiled = numsim.transpile(
        many_cta_store,
        cache_dir=tmp_path_factory.mktemp("racecheck-cta-count-scaling"),
        _analysis_capable=True,
        _analysis_checker="racecheck",
    )
    output = np.zeros(_CTAS * _THREADS, dtype=np.int32)

    started = perf_counter()
    result = numsim.Engine(max_workers=_WORKERS).run_racecheck_phase(
        compiled, {"output": output}, phase_index=0
    )
    elapsed = perf_counter() - started
    payload = result.to_dict()
    record_property("racecheck_elapsed_seconds", elapsed)

    assert result.verdict == "clean", payload
    assert payload["incomplete"] == [], payload
    assert payload["execution_error"] is None, payload
    assert payload["stats"]["completed_task_count"] == _CTAS, payload
    assert elapsed < _LIMIT_SECONDS, (elapsed, payload["timing"])
