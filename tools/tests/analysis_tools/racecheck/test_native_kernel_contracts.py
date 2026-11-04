"""Racecheck integration coverage for retained kernel contracts."""

from __future__ import annotations

import numpy as np

from tests.analysis_tools.racecheck.test_native_racecheck_artifact import (
    native_racecheck_write_write,
)
from tirx_harness import racecheck
from tvm_tirx.script import tirx as T


@T.function
def native_strided_local_index_overlap():
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((64,), "float32")
    for iteration in T.serial(2):
        index: T.let = lane + iteration * 32
        shared[index] = T.cast(warp, "float32")


@T.function
def native_nested_local_index_disjoint(output: T.Tensor((128,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((128,), "float32")
    for iteration in T.serial(2):
        for inner in T.serial(1):
            index: T.let = warp * 64 + lane + iteration * 32 + inner
            shared[index] = T.cast(index, "float32")
    T.cuda.cta_sync()
    output[warp * 64 + lane] = shared[warp * 64 + lane]


@T.function
def native_uint32_local_scalar(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    value = T.alloc_local((1,), "uint32")
    if lane == 0:
        value[0] = T.uint32(17)
        output[0] = value[0]


def test_local_scalar_indices_preserve_strided_race_and_nested_clean_case():
    racing = racecheck(native_strided_local_index_overlap, inputs={})
    clean = racecheck(
        native_nested_local_index_disjoint,
        inputs={"output": np.zeros(128, dtype=np.float32)},
    )

    assert racing.verdict == "error"
    native = racing.to_dict()["native"]
    assert native["incomplete"] == []
    assert [finding["access_pair"] for finding in native["findings"]] == ["write_write"]
    assert native["findings"][0]["ordering_domain"] == "execution"
    assert native["findings"][0]["ordering_failure"] == "missing_inter_actor_sync"
    source_texts = {
        witness["operation"]["source"]["source_text"]
        for witness in (native["findings"][0]["prior"], native["findings"][0]["current"])
    }
    assert source_texts and all("shared[index]" in text for text in source_texts)
    clean.require_clean()
    assert clean.to_dict()["native"]["incomplete"] == []


def test_racecheck_executes_non_int32_local_storage():
    report = racecheck(
        native_uint32_local_scalar,
        inputs={"output": np.zeros(1, dtype=np.uint32)},
    )
    report.require_clean()
    assert report.to_dict()["native"]["incomplete"] == []


def test_race_report_prints_both_conflicting_source_operations():
    report = racecheck(native_racecheck_write_write, inputs={})

    assert report.verdict == "error"
    rendered = report.format()
    assert "Prior: warp 0, source op" in rendered
    assert "Current: warp 1, source op" in rendered
    assert "shared[0] = warp + 1" in rendered
    assert "allocation#" in rendered
