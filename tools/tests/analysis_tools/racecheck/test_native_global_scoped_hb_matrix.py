from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def same_cluster_message_passing(
    data: T.Buffer((1,), "int32"),
    flag: T.Buffer((1,), "int32"),
    output: T.Buffer((1,), "int32"),
):
    T.device_entry()
    cta = T.cta_id_in_cluster([2])
    lane = T.lane_id([32])
    observed = T.local_scalar("int32")
    if (cta == 0) and (lane == 0):
        data[0] = T.int32(17)
        T.ptx.st.release.cluster.global_.s32(flag.ptr_to([0]), T.int32(1))
    elif (cta == 1) and (lane == 0):
        observed = T.int32(0)
        T.cuda.wait_until(
            observed, flag.ptr_to([0]), observed != T.int32(0), "cluster", "global"
        )
        output[0] = data[0]


@T.prim_func
def host_initialized_global_reads(
    source: T.Buffer((1,), "int32"),
    output: T.Buffer((2,), "int32"),
):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx.ld.acquire.gpu.global_.s32(output[cta], source.ptr_to([0]))


@T.prim_func
def exact_read_from_same_value_or_aba(
    aba: T.int32,
    data: T.Buffer((2,), "int32"),
    flag: T.Buffer((1,), "uint32"),
    turn: T.Buffer((1,), "uint32"),
    ready: T.Buffer((1,), "uint32"),
    output: T.Buffer((2,), "int32"),
):
    T.device_entry()
    cta = T.cta_id([3])
    lane = T.lane_id([32])
    observed = T.local_scalar("uint32")
    if (cta == 0) and (lane == 0):
        data[0] = T.int32(11)
        T.ptx.st.release.gpu.global_.u32(flag.ptr_to([0]), T.uint32(1))
        T.ptx.st.relaxed.gpu.global_.u32(turn.ptr_to([0]), T.uint32(1))
    elif (cta == 1) and (lane == 0):
        observed = T.uint32(0)
        while observed == T.uint32(0):
            T.ptx.ld.relaxed.gpu.global_.u32(observed, turn.ptr_to([0]))
        data[1] = T.int32(22)
        if aba != 0:
            T.ptx.st.relaxed.gpu.global_.u32(flag.ptr_to([0]), T.uint32(0))
        T.ptx.st.release.gpu.global_.u32(flag.ptr_to([0]), T.uint32(1))
        T.ptx.st.relaxed.gpu.global_.u32(ready.ptr_to([0]), T.uint32(1))
    elif (cta == 2) and (lane == 0):
        observed = T.uint32(0)
        while observed == T.uint32(0):
            T.ptx.ld.relaxed.gpu.global_.u32(observed, ready.ptr_to([0]))
        T.ptx.ld.acquire.gpu.global_.u32(observed, flag.ptr_to([0]))
        output[0] = data[0]
        output[1] = data[1]


@T.prim_func
def lane_precise_publication(
    mode: T.int32,
    data: T.Buffer((2,), "int32"),
    flag: T.Buffer((1,), "int32"),
    output: T.Buffer((2,), "int32"),
):
    T.device_entry()
    cta = T.cta_id_in_cluster([2])
    lane = T.lane_id([32])
    observed = T.local_scalar("int32")
    if cta == 0:
        if mode <= 1:
            if lane == 1:
                data[0] = T.int32(31)
        elif mode <= 3:
            if lane == 0:
                data[0] = T.int32(31)
        else:
            if lane == 1:
                data[0] = T.int32(31)
            if lane == 2:
                data[1] = T.int32(37)
        if mode == 1:
            T.cuda.warp_sync()
        elif mode == 4:
            T.cuda.cta_sync()
        if lane == 0:
            T.ptx.st.release.cluster.global_.s32(flag.ptr_to([0]), T.int32(1))
    else:
        if lane == 0:
            observed = T.int32(0)
            T.cuda.wait_until(
                observed, flag.ptr_to([0]), observed != T.int32(0), "cluster", "global"
            )
        if mode == 3:
            T.cuda.warp_sync()
        if mode <= 1:
            if lane == 0:
                output[0] = data[0]
        elif mode <= 3:
            if lane == 1:
                output[0] = data[0]
        elif lane == 0:
            output[0] = data[0]
            output[1] = data[1]


@T.prim_func
def rmw_release_sequence_matrix(
    mode: T.int32,
    data: T.Buffer((1,), "int32"),
    flag: T.Buffer((1,), "uint32"),
    turn: T.Buffer((1,), "uint32"),
    ready: T.Buffer((1,), "uint32"),
    output: T.Buffer((3,), "int32"),
):
    T.device_entry()
    cta = T.cta_id([3])
    lane = T.lane_id([32])
    observed = T.local_scalar("uint32")
    old = T.local_scalar("uint32")
    if mode == 2:
        if (cta <= 1) and (lane == 0):
            T.ptx.atom.relaxed.cluster.global_.add.u32(old, flag.ptr_to([0]), T.uint32(1))
            output[cta] = T.cast(old, "int32")
    elif (cta == 0) and (lane == 0):
        data[0] = T.int32(53)
        if mode == 1:
            T.ptx.atom.relaxed.gpu.global_.add.u32(old, flag.ptr_to([0]), T.uint32(1))
        else:
            T.ptx.st.release.gpu.global_.u32(flag.ptr_to([0]), T.uint32(1))
        T.ptx.st.relaxed.gpu.global_.u32(turn.ptr_to([0]), T.uint32(1))
    elif (cta == 1) and (lane == 0):
        observed = T.uint32(0)
        T.cuda.wait_until(
            observed, turn.ptr_to([0]), observed != T.uint32(0), "gpu", "global"
        )
        if mode == 3:
            T.ptx.red.relaxed.gpu.global_.add.u32(flag.ptr_to([0]), T.uint32(1))
            output[0] = data[0]
        else:
            T.ptx.atom.relaxed.gpu.global_.add.u32(old, flag.ptr_to([0]), T.uint32(1))
            T.ptx.st.relaxed.gpu.global_.u32(ready.ptr_to([0]), T.uint32(1))
    elif (cta == 2) and (lane == 0) and (mode <= 1):
        observed = T.uint32(0)
        T.cuda.wait_until(
            observed, ready.ptr_to([0]), observed != T.uint32(0), "gpu", "global"
        )
        if mode == 0:
            T.cuda.wait_until(
                observed, flag.ptr_to([0]), observed >= T.uint32(1), "gpu", "global"
            )
        else:
            T.ptx.ld.relaxed.gpu.global_.u32(observed, flag.ptr_to([0]))
        output[0] = data[0]


@T.prim_func
def torn_scoped_observation(
    data: T.Buffer((1,), "int32"),
    flag: T.Buffer((1,), "uint64"),
    turn: T.Buffer((1,), "uint32"),
    ready: T.Buffer((1,), "uint32"),
    output: T.Buffer((1,), "int32"),
):
    T.device_entry()
    cta = T.cta_id([3])
    lane = T.lane_id([32])
    observed32 = T.local_scalar("uint32")
    observed64 = T.local_scalar("uint64")
    if (cta == 0) and (lane == 0):
        data[0] = T.int32(71)
        T.ptx.st.release.gpu.global_.u64(flag.ptr_to([0]), T.uint64(1))
        T.ptx.st.relaxed.gpu.global_.u32(turn.ptr_to([0]), T.uint32(1))
    elif (cta == 1) and (lane == 0):
        observed32 = T.uint32(0)
        while observed32 == T.uint32(0):
            T.ptx.ld.relaxed.gpu.global_.u32(observed32, turn.ptr_to([0]))
        high_half = T.ptr_byte_offset(flag.ptr_to([0]), T.uint32(4), "uint32")
        T.ptx.st.relaxed.gpu.global_.u32(high_half, T.uint32(0))
        T.ptx.st.relaxed.gpu.global_.u32(ready.ptr_to([0]), T.uint32(1))
    elif (cta == 2) and (lane == 0):
        observed32 = T.uint32(0)
        while observed32 == T.uint32(0):
            T.ptx.ld.relaxed.gpu.global_.u32(observed32, ready.ptr_to([0]))
        T.ptx.ld.acquire.gpu.global_.u64(observed64, flag.ptr_to([0]))
        output[0] = data[0] + T.cast(observed64 & T.uint64(0), "int32")


@T.prim_func
def async_store_global_publication(
    wait_for_completion: T.int32,
    source: T.Buffer((16,), "uint8"),
    destination: T.Buffer((16,), "uint8"),
    flag: T.Buffer((1,), "uint32"),
    output: T.Buffer((1,), "uint8"),
):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    observed = T.local_scalar("uint32")
    if (cta == 0) and (lane == 0):
        for index in T.serial(16):
            shared[index] = source[index]
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint"](
            destination.ptr_to([0]),
            shared.ptr_to([0]),
            T.cast(16, "uint32"),
            T.uint64(0x1000000000000000),
        )
        T.ptx.cp.async_.bulk.commit_group()
        if wait_for_completion != 0:
            T.ptx.cp.async_.bulk.wait_group(0)
        T.ptx.st.release.gpu.global_.u32(flag.ptr_to([0]), T.uint32(1))
    elif (cta == 1) and (lane == 0):
        observed = T.uint32(0)
        T.cuda.wait_until(
            observed, flag.ptr_to([0]), observed != T.uint32(0), "gpu", "global"
        )
        output[0] = destination[0]


@T.prim_func
def tcgen_commit_global_publication(
    use_acquire: T.int32,
    data: T.Buffer((1,), "int32"),
    flag: T.Buffer((1,), "uint32"),
    output: T.Buffer((1,), "int32"),
):
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    observed = T.local_scalar("uint32")

    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if (warp == 0) and (lane == 0):
        data[0] = T.int32(73)
        T.ptx.st.release.gpu.global_.u32(flag.ptr_to([0]), T.uint32(1))
    elif (warp == 1) and (lane == 0):
        observed = T.uint32(0)
        if use_acquire != 0:
            T.cuda.wait_until(
                observed, flag.ptr_to([0]), observed != T.uint32(0), "gpu", "global"
            )
        else:
            # The contrast is a reader that takes no edge. A declared wait
            # always acquires, so the only way to spell one is the loop a
            # kernel writes without the primitive.
            while observed == T.uint32(0):
                T.ptx.ld.relaxed.gpu.global_.u32(observed, flag.ptr_to([0]))
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    elif warp == 2:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        if lane == 0:
            output[0] = data[0]


@T.prim_func
def million_failed_polls_then_release_acquire(
    data: T.Buffer((1,), "int32"),
    flag: T.Buffer((1,), "uint32"),
    ready: T.Buffer((1,), "uint32"),
    output: T.Buffer((1,), "int32"),
):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    observed = T.local_scalar("uint32")
    if (cta == 0) and (lane == 0):
        observed = T.uint32(0)
        for _step in T.serial(1_000_000):
            T.ptx.ld.relaxed.gpu.global_.u32(observed, flag.ptr_to([0]))
        T.ptx.st.release.gpu.global_.u32(ready.ptr_to([0]), T.uint32(1))
        T.cuda.wait_until(
            observed, flag.ptr_to([0]), observed != T.uint32(0), "gpu", "global"
        )
        output[0] = data[0]
    elif (cta == 1) and (lane == 0):
        observed = T.uint32(0)
        T.cuda.wait_until(
            observed, ready.ptr_to([0]), observed != T.uint32(0), "gpu", "global"
        )
        data[0] = T.int32(89)
        T.ptx.st.release.gpu.global_.u32(flag.ptr_to([0]), T.uint32(1))


def _transpile(kernel, tmp_path):
    return numsim.transpile(
        kernel,
        cache_dir=tmp_path,
        _default_generated_opt_level=0,
        _analysis_capable=True,
        _analysis_checker="racecheck",
    )


def _run(module, inputs, *, max_workers: int = 3, inspect_accesses: bool = False):
    return (
        numsim.Engine(max_workers=max_workers)
        .run_racecheck_phase(
            module,
            inputs,
            inspect_accesses=inspect_accesses,
        )
        .to_dict()
    )


def _zeros(dtype, count=1):
    return np.zeros(count, dtype=dtype)


def test_same_cluster_release_acquire_is_clean(tmp_path):
    result = _run(
        _transpile(same_cluster_message_passing, tmp_path),
        {
            "data": _zeros(np.int32),
            "flag": _zeros(np.int32),
            "output": _zeros(np.int32),
        },
    )

    assert result["verdict"] == "clean"
    assert result["findings"] == []
    assert result["incomplete"] == []


def test_host_initialized_version_is_before_all_kernel_reads(tmp_path):
    result = _run(
        _transpile(host_initialized_global_reads, tmp_path),
        {
            "source": np.array([101], dtype=np.int32),
            "output": _zeros(np.int32, 2),
        },
    )

    assert result["verdict"] == "clean"
    assert result["findings"] == []
    assert result["incomplete"] == []


@pytest.mark.parametrize("aba", [0, 1])
def test_same_value_and_aba_observations_use_only_the_exact_writer(tmp_path, aba):
    result = _run(
        _transpile(exact_read_from_same_value_or_aba, tmp_path),
        {
            "aba": aba,
            "data": _zeros(np.int32, 2),
            "flag": _zeros(np.uint32),
            "turn": _zeros(np.uint32),
            "ready": _zeros(np.uint32),
            "output": _zeros(np.int32, 2),
        },
    )

    assert result["verdict"] == "error"
    assert result["incomplete"] == []
    data_offsets = {
        finding["overlap"]["byte_offset"]
        for finding in result["findings"]
        if finding["overlap"]["byte_len"] == 4
    }
    assert 0 in data_offsets
    assert 4 not in data_offsets


@pytest.mark.parametrize(
    ("mode", "expected_verdict"),
    [(0, "error"), (1, "clean"), (2, "error"), (3, "clean"), (4, "clean")],
)
def test_publication_and_acquisition_are_lane_precise(tmp_path, mode, expected_verdict):
    result = _run(
        _transpile(lane_precise_publication, tmp_path),
        {
            "mode": mode,
            "data": _zeros(np.int32, 2),
            "flag": _zeros(np.int32),
            "output": _zeros(np.int32, 2),
        },
    )

    assert result["verdict"] == expected_verdict
    assert result["incomplete"] == []
    if expected_verdict == "clean":
        assert result["findings"] == []
    else:
        assert any(finding["overlap"]["byte_offset"] == 0 for finding in result["findings"])


@pytest.mark.parametrize(
    ("mode", "expected_verdict"),
    [(0, "clean"), (1, "error"), (2, "error"), (3, "error")],
)
def test_rmw_release_sequence_and_morally_strong_rules(tmp_path, mode, expected_verdict):
    result = _run(
        _transpile(rmw_release_sequence_matrix, tmp_path),
        {
            "mode": mode,
            "data": _zeros(np.int32),
            "flag": _zeros(np.uint32),
            "turn": _zeros(np.uint32),
            "ready": _zeros(np.uint32),
            "output": _zeros(np.int32, 3),
        },
    )

    assert result["verdict"] == expected_verdict
    assert result["incomplete"] == []
    if mode == 0:
        assert result["findings"] == []
    elif mode == 2:
        scope_mismatch = next(
            finding for finding in result["findings"] if finding["kind"] == "scope_mismatch"
        )
        assert scope_mismatch["ordering_domain"] == "memory"
        assert scope_mismatch["ordering_failure"] == "scope_mismatch"
        assert not any(
            finding.get("access_pair") in {"write_read", "read_write", "write_write"}
            for finding in result["findings"]
        )
    else:
        assert any(finding["overlap"]["byte_offset"] == 0 for finding in result["findings"])


def test_torn_scoped_read_reports_the_unordered_mixed_size_writes(tmp_path):
    result = _run(
        _transpile(torn_scoped_observation, tmp_path),
        {
            "data": _zeros(np.int32),
            "flag": _zeros(np.uint64),
            "turn": _zeros(np.uint32),
            "ready": _zeros(np.uint32),
            "output": _zeros(np.int32),
        },
    )

    # This original kernel already races before its acquire: CTA 1's u32
    # overwrite overlaps CTA 0's u64 release, with only relaxed polling between.
    # Mixed-size observations are modeled now, not a generic coverage gap.
    assert result["verdict"] == "error"
    assert result["incomplete"] == []
    finding = next(item for item in result["findings"] if item["access_pair"] == "write_write")
    assert finding["ordering_failure"] == "missing_release_acquire"
    assert (finding["overlap"]["byte_offset"], finding["overlap"]["byte_len"]) == (4, 4)
    assert (finding["prior"]["span"]["byte_len"], finding["current"]["span"]["byte_len"]) == (8, 4)
    assert finding["prior"]["operation"]["global_warp_id"] == 0
    assert finding["current"]["operation"]["global_warp_id"] == 1


@pytest.mark.parametrize(
    ("wait_for_completion", "expected_verdict"),
    [(1, "clean"), (0, "error")],
)
def test_async_store_requires_completion_handoff_before_publication(
    tmp_path, wait_for_completion, expected_verdict
):
    """Positive control for retired-token reclamation.

    The clean arm waits for the async global destination, retires that token,
    and only then publishes a generic release flag consumed by another CTA.
    Clearing the token clock at retirement would erase the completion edge and
    turn this ordered destination read into a false race.
    """
    source = np.arange(16, dtype=np.uint8) + np.uint8(1)
    result = _run(
        _transpile(async_store_global_publication, tmp_path),
        {
            "wait_for_completion": wait_for_completion,
            "source": source,
            "destination": _zeros(np.uint8, 16),
            "flag": _zeros(np.uint32),
            "output": _zeros(np.uint8),
        },
    )

    assert result["verdict"] == expected_verdict
    if wait_for_completion:
        assert result["findings"] == []
        assert result["incomplete"] == []
    else:
        assert result["findings"] or result["incomplete"]


@pytest.mark.parametrize(
    ("use_acquire", "expected_verdict"),
    [(1, "clean"), (0, "error")],
)
def test_tcgen_commit_mbarrier_forwards_the_issuing_thread_publication(
    tmp_path, use_acquire, expected_verdict
):
    result = _run(
        _transpile(tcgen_commit_global_publication, tmp_path),
        {
            "use_acquire": use_acquire,
            "data": _zeros(np.int32),
            "flag": _zeros(np.uint32),
            "output": _zeros(np.int32),
        },
    )

    assert result["verdict"] == expected_verdict
    assert result["incomplete"] == []
    if expected_verdict == "clean":
        assert result["findings"] == []
    else:
        assert any(finding["overlap"]["byte_offset"] == 0 for finding in result["findings"])


def test_million_failed_polls_keep_the_online_report_bounded(tmp_path):
    module = _transpile(million_failed_polls_then_release_acquire, tmp_path)
    result = (
        numsim.Engine(
            max_workers=2,
            native_loop_iteration_budget=1_000_010,
        )
        .run_racecheck_phase(
            module,
            {
                "data": _zeros(np.int32),
                "flag": _zeros(np.uint32),
                "ready": _zeros(np.uint32),
                "output": _zeros(np.int32),
            },
            inspect_accesses=False,
        )
        .to_dict()
    )

    assert result["verdict"] == "clean"
    assert result["findings"] == []
    assert result["incomplete"] == []
    assert result["access_count"] >= 1_000_005
    assert result["accesses_complete"] is False
    assert result["accesses"] == []
