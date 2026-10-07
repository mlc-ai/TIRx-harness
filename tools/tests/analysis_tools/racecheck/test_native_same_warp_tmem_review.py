"""Native Racecheck policy for unmodeled tcgen05.ld completion."""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import tmem_datapath_layout, wg_local_layout

PTX_SAME_THREAD_REGISTER_DEPENDENCY_URL = (
    "https://docs.nvidia.com/cuda/parallel-thread-execution/"
    "#tcgen05-memory-consistency-model-canonical-sync-patterns-reg-dependency-same-thread"
)


@T.prim_func
def same_warp_tmem_dependency(needs_wait: T.int32):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer(
        (128, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 32),
        allocated_addr=0,
    )
    registers = T.alloc_local((32,), "float32")
    register_tile = registers.view(128, 32, layout=wg_local_layout(32))

    for col in T.serial(32):
        tmem[row, col] = T.cast(row * 32 + col, "float32")
    T.cuda.cta_sync()

    Tx.wg.copy_async(register_tile[:, :], tmem[:, :])
    if needs_wait != 0:
        T.ptx.tcgen05.wait__ld.sync.aligned()
    Tx.wg.copy_async(tmem[:, :], register_tile[:, :])
    T.ptx.tcgen05.wait__ld.sync.aligned()
    T.ptx.tcgen05.wait__st.sync.aligned()


@T.prim_func
def unrelated_review_must_not_hide_cross_warp_race():
    """Keep an unrelated pending load live after a same-warp review."""

    T.device_entry()
    warpgroup = T.warpgroup_id([2])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer(
        (128, 64),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 64),
        allocated_addr=0,
    )
    registers_a = T.alloc_local((32,), "float32")
    registers_b = T.alloc_local((32,), "float32")
    register_tile_a = registers_a.view(128, 32, layout=wg_local_layout(32))
    register_tile_b = registers_b.view(128, 32, layout=wg_local_layout(32))

    if warpgroup == 0:
        for col in T.serial(64):
            tmem[row, col] = T.cast(row * 64 + col, "float32")
    else:
        for col in T.serial(32):
            registers_b[col] = T.cast(row * 32 + col, "float32")
    T.cuda.cta_sync()

    if warpgroup == 0:
        Tx.wg.copy_async(register_tile_a[:, :], tmem[:, 0:32])
        Tx.wg.copy_async(register_tile_b[:, :], tmem[:, 32:64])
        Tx.wg.copy_async(tmem[:, 0:32], register_tile_a[:, :])
        T.ptx.tcgen05.wait__st.sync.aligned()
        T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    if warpgroup == 1:
        T.ptx.tcgen05.fence__after_thread_sync()
        Tx.wg.copy_async(tmem[:, 32:64], register_tile_b[:, :])
        T.ptx.tcgen05.wait__st.sync.aligned()
    if warpgroup == 0:
        T.ptx.tcgen05.wait__ld.sync.aligned()


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-racecheck-same-warp-tmem-review")


def _run(needs_wait: int, cache_dir):
    return racecheck(
        same_warp_tmem_dependency,
        inputs={"needs_wait": np.int32(needs_wait)},
        cache_dir=cache_dir,
        max_workers=1,
    )


def test_unmodeled_same_warp_tmem_dependency_is_review(native_cache_dir) -> None:
    report = _run(0, native_cache_dir)

    assert report.verdict == "review", report.format()
    assert len(report.findings) == 1
    public_finding = report.findings[0]
    assert public_finding.status == "review"
    assert public_finding.details["access_pair"] == "read_write"
    assert public_finding.kind == "tmem_lifetime_review"
    assert (
        "cannot determine whether the earlier tcgen05.ld completed through "
        "a true register dependency"
    ) in public_finding.message
    assert PTX_SAME_THREAD_REGISTER_DEPENDENCY_URL in public_finding.message

    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert native["execution_error"] is None
    assert native["stats"]["completed_task_count"] == native["stats"]["task_count"]
    assert len(native["findings"]) == 1
    finding = native["findings"][0]
    assert finding["status"] == "review"
    assert finding["prior"]["space"] == finding["current"]["space"] == "tmem"
    assert (
        finding["prior"]["operation"]["global_warp_id"]
        == finding["current"]["operation"]["global_warp_id"]
    )
    for witness in (finding["prior"], finding["current"]):
        assert witness["operation"]["source"]["source_text"].strip()


def test_explicit_wait_resolves_same_warp_tmem_dependency(native_cache_dir) -> None:
    report = _run(1, native_cache_dir)
    report.require_clean()
    assert report.to_dict()["native"]["incomplete"] == []


def test_unwaited_tmem_load_conflicts_remain_distinct_reviews(native_cache_dir) -> None:
    report = racecheck(
        unrelated_review_must_not_hide_cross_warp_race,
        cache_dir=native_cache_dir,
        max_workers=1,
    )

    assert report.verdict == "review", report.format()
    native = report.to_dict()["native"]
    assert any(
        finding["status"] == "review"
        and finding["prior"]["operation"]["global_warp_id"]
        == finding["current"]["operation"]["global_warp_id"]
        for finding in native["findings"]
    )
    assert any(
        finding["status"] == "review"
        and finding["prior"]["operation"]["global_warp_id"]
        != finding["current"]["operation"]["global_warp_id"]
        for finding in native["findings"]
    )
    assert native["incomplete"] == []
