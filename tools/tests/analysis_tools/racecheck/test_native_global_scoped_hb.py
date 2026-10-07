from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def global_scoped_message_passing(
    mode: T.int32,
    data: T.Buffer((2,), "int32"),
    flag: T.Buffer((1,), "int32"),
    output: T.Buffer((1,), "int32"),
):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    observed = T.local_scalar("int32")
    # Isolate release/acquire halves: an SC fence can independently order the
    # misplaced-publication case. CUDA thread_fence has separate SC coverage.
    if (cta == 0) and (lane == 0):
        data[0] = T.int32(41)
        if mode == 0:
            T.ptx.st.release.gpu.global_.s32(flag.ptr_to([0]), T.int32(1))
        elif mode == 1:
            T.ptx.st.release.cluster.global_.s32(flag.ptr_to([0]), T.int32(1))
        elif mode == 2:
            T.ptx.st.relaxed.gpu.global_.s32(flag.ptr_to([0]), T.int32(1))
        elif mode == 3:
            T.ptx.st.release.gpu.global_.s32(flag.ptr_to([0]), T.int32(1))
        elif mode == 4:
            flag[0] = T.int32(1)
        elif mode == 5:
            T.ptx.fence.acq_rel.gpu()
            T.ptx.st.volatile.global_.s32(flag.ptr_to([0]), T.int32(1))
        elif mode == 6:
            T.ptx.st.volatile.global_.s32(flag.ptr_to([0]), T.int32(1))
            T.ptx.fence.acq_rel.gpu()
        elif mode == 7:
            T.ptx.fence.acq_rel.gpu()
            T.ptx.st.volatile.global_.s32(flag.ptr_to([0]), T.int32(1))
        else:
            T.ptx.fence.acq_rel.gpu()
            data[1] = T.int32(73)
            T.ptx.st.relaxed.sys.global_.s32(flag.ptr_to([0]), T.int32(1))
    elif (cta == 1) and (lane == 0):
        observed = T.int32(0)
        if (mode == 0) or (mode == 2):
            T.cuda.wait_until(
                observed, flag.ptr_to([0]), observed != T.int32(0), "gpu", "global"
            )
        elif mode == 1:
            T.cuda.wait_until(
                observed, flag.ptr_to([0]), observed != T.int32(0), "cluster", "global"
            )
        elif mode == 3:
            # A reader that takes no edge, spelled the way a kernel writes one
            # without the primitive: a declared wait always acquires, so it
            # cannot express this half of the contrast.
            while observed == T.int32(0):
                T.ptx.ld.relaxed.gpu.global_.s32(observed, flag.ptr_to([0]))
        elif mode == 4:
            while observed == T.int32(0):
                observed = flag[0]
        elif mode == 7:
            # The fence is before the load rather than after it, so it completes
            # no acquire pattern. The poll has to be non-acquiring for that to
            # be the deciding fact.
            T.ptx.fence.acq_rel.gpu()
            while observed == T.int32(0):
                T.ptx.ld.volatile.global_.s32(observed, flag.ptr_to([0]))
        else:
            T.cuda.wait_until(
                observed, flag.ptr_to([0]), observed != T.int32(0), "sys", "global",
            )
            T.ptx.fence.acq_rel.gpu()
        output[0] = data[1] if mode == 8 else data[0]


@T.prim_func
def typed_cluster_fence_message_passing(
    data: T.Buffer((1,), "int32"),
    flag: T.Buffer((1,), "int32"),
    output: T.Buffer((1,), "int32"),
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    observed = T.local_scalar("int32")
    if (warp == 0) and (lane == 0):
        data[0] = T.int32(41)
        T.ptx.fence.acq_rel.cluster()
        T.ptx.st.relaxed.sys.global_.s32(flag.ptr_to([0]), T.int32(1))
    elif (warp == 1) and (lane == 0):
        observed = T.int32(0)
        T.cuda.wait_until(
            observed, flag.ptr_to([0]), observed != T.int32(0), "sys", "global",
        )
        T.ptx.fence.acq_rel.cluster()
        output[0] = data[0]


def _run_message_passing(tmp_path, mode: int, *, max_workers: int = 2):
    module = numsim.transpile(
        global_scoped_message_passing,
        cache_dir=tmp_path,
        _default_generated_opt_level=0,
        _analysis_capable=True,
        _analysis_checker="racecheck",
    )
    result = numsim.Engine(max_workers=max_workers).run_racecheck_phase(
        module,
        {
            "mode": mode,
            "data": np.zeros(2, dtype=np.int32),
            "flag": np.zeros(1, dtype=np.int32),
            "output": np.zeros(1, dtype=np.int32),
        },
        inspect_accesses=True,
    )
    return result.to_dict()


@pytest.mark.parametrize("mode", [0, 5])
def test_scoped_global_message_passing_and_fence_halves_are_clean(tmp_path, mode):
    result = _run_message_passing(tmp_path, mode)

    assert result["verdict"] == "clean"
    assert result["findings"] == []
    assert result["incomplete"] == []
    assert result["checked_memory_spaces"] == ["global", "shared", "tmem"]


@pytest.mark.parametrize(
    ("mode", "expected_offset"),
    [(1, 0), (2, 0), (3, 0), (6, 0), (7, 0), (8, 4)],
)
def test_scoped_global_missing_edge_or_misplaced_fence_reports_data_race(
    tmp_path, mode, expected_offset
):
    result = _run_message_passing(tmp_path, mode)

    assert result["verdict"] == "error"
    assert result["incomplete"] == []
    assert any(
        finding["overlap"]["byte_offset"] == expected_offset for finding in result["findings"]
    )
    if mode == 1:
        diagnostics = [
            finding for finding in result["findings"] if finding["kind"] == "scope_mismatch"
        ]
        assert len(diagnostics) == 1
        assert diagnostics[0]["release_scope"] == "cluster"
        assert diagnostics[0]["acquire_scope"] == "cluster"
        assert diagnostics[0]["actor_relation"] == "cross_cluster"


def test_plain_flag_and_data_conflicts_are_both_reported(tmp_path):
    result = _run_message_passing(tmp_path, 4)

    assert result["verdict"] == "error"
    assert result["incomplete"] == []
    offsets = {finding["overlap"]["byte_offset"] for finding in result["findings"]}
    assert 0 in offsets
    assert len(result["findings"]) >= 2


def test_access_journal_retains_exact_scoped_metadata(tmp_path):
    result = _run_message_passing(tmp_path, 0)
    scoped = [
        access for access in result["accesses"] if access["memory_order"] in {"release", "acquire"}
    ]

    # Only the publication is here. The reader's half is a declared wait, and
    # a wait adjudicates no access of its own -- the polling is the engine's --
    # so the journal holds the protocol's store and no matching acquire.
    assert {access["memory_order"] for access in scoped} == {"release"}
    assert {access["memory_scope"] for access in scoped} == {"gpu"}
    assert {access["memory_proxy"] for access in scoped} == {"generic"}
    # A scoped single copy is `strong`, not `atomic`: it is not a
    # read-modify-write. See MemoryAccessClass in physical_access.rs.
    assert {access["memory_access_class"] for access in scoped} == {"atomic"}


@pytest.mark.parametrize(
    ("mode", "order", "scope", "access_class"),
    [
        (2, "relaxed", "gpu", "atomic"),
        (3, "release", "gpu", "atomic"),
        (4, "weak", None, "plain"),
        (5, "relaxed", "sys", "atomic"),
    ],
)
def test_access_journal_preserves_memory_semantic_variants(
    tmp_path, mode, order, scope, access_class
):
    """Each spelling reaches the journal as the semantics it was written with.

    Mode 3's relaxed half is the reader, and a declared wait adjudicates no
    access, so what the journal holds for it is the publishing store. Mode 4
    is plain on both sides.

    The journal no longer carries a `declared` flag: `wait_until` is the
    only declared operation and it adjudicates no access, so a word is claimed
    by the address the wait names rather than by each access carrying a mark.
    """
    result = _run_message_passing(tmp_path, mode)

    assert any(
        access["memory_order"] == order
        and access["memory_scope"] == scope
        and access["memory_proxy"] == "generic"
        and access["memory_access_class"] == access_class
        for access in result["accesses"]
    )


def test_ptx_cluster_fence_orders_same_cta_message_passing(tmp_path):
    module = numsim.transpile(
        typed_cluster_fence_message_passing,
        cache_dir=tmp_path,
        _default_generated_opt_level=0,
        _analysis_capable=True,
        _analysis_checker="racecheck",
    )
    data = np.zeros(1, dtype=np.int32)
    flag = np.zeros(1, dtype=np.int32)
    output = np.zeros(1, dtype=np.int32)
    result = numsim.Engine(max_workers=2).run_racecheck_phase(
        module,
        {"data": data, "flag": flag, "output": output},
    )

    assert result.verdict == "clean"
    assert result.findings == []
    assert result.incomplete == []


def test_scoped_global_findings_are_worker_count_deterministic(tmp_path):
    payloads = [_run_message_passing(tmp_path, 1, max_workers=count) for count in (1, 2, 4, 8)]
    baseline = payloads[0]

    for payload in payloads[1:]:
        assert payload["verdict"] == baseline["verdict"]
        assert payload["findings"] == baseline["findings"]
        assert payload["incomplete"] == baseline["incomplete"]
