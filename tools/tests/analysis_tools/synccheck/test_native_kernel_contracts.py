"""Public native Synccheck kernel contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_synccheck as internal_synccheck
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.lang.pipeline import TMABar


@T.prim_func
def native_trailing_warp_mbarrier(expected_arrivals: T.uint32):
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    warp = T.warp_id([9])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")

    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), expected_arrivals)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0:
        if lane == 0:
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    else:
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))


@T.prim_func
def native_lane_and_elect_mbarrier(mode: T.int32, epoch: T.int32, expected_arrivals: T.uint32):
    T.device_entry()
    wg = T.warpgroup_id([2])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    tid_in_wg = T.thread_id_in_wg([128])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")

    if (wg == 0) and (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), expected_arrivals)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if wg == 0:
        if mode == 0:
            if tid_in_wg == 0:
                T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
        elif mode == 1:
            if (epoch == 0) & (tid_in_wg == 0):
                T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
        elif mode == 2:
            if T.cuda.elect_sync():
                T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
        else:
            leader = T.alloc_local((1,), "bool")
            leader[0] = (warp == 1) & T.cuda.elect_sync()
            if leader[0]:
                T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
    elif (warp == 0) and (lane == 0):
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)


@T.prim_func
def native_collective_in_while(mode: T.int32, bound: T.int32):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    iteration = T.alloc_local((1,), "int32")
    iteration[0] = 0
    while iteration[0] < 1:
        if mode == 0:
            _uniform = T.cuda.warp_sum(T.float32(1.0))
        elif mode == 1:
            if lane == 0:
                _lane_zero = T.cuda.warp_sum(T.float32(1.0))
        elif mode == 2:
            if warp == 0:
                _whole_warp = T.cuda.warp_sum(T.float32(1.0))
        elif mode == 3:
            if T.cuda.elect_sync():
                _elected = T.cuda.warp_sum(T.float32(1.0))
        else:
            if lane < bound:
                _width_sixteen = T.cuda.warp_sum(T.float32(1.0), width=16)
        iteration[0] = iteration[0] + 1


@T.prim_func
def native_setmaxnreg_inside_loop():
    T.device_entry()
    _wg = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    for _iteration in T.serial(2):
        T.ptx.setmaxnreg.dec.sync.aligned.u32(88)


@T.prim_func
def native_setmaxnreg_with_trailing_warps():
    T.device_entry()
    warp = T.warp_id([6])
    _lane = T.lane_id([32])
    if warp < 4:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(88)


@T.prim_func
def native_mbarrier_depth_two_pipeline():
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((4,), "uint64", scope="shared")

    if (warp == 0) and (lane == 0):
        for barrier in T.serial(4):
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[barrier]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if (warp == 0) and (lane == 0):
        for iteration in T.serial(4):
            slot: T.let = iteration % 2
            if iteration >= 2:
                T.cuda.mbarrier_wait(T.address_of(barriers[2 + slot]), 0)
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[slot]))

    if (warp == 1) and (lane == 0):
        phase: T.int32 = 0
        for iteration in T.serial(4):
            slot: T.let = iteration % 2
            T.cuda.mbarrier_wait(T.address_of(barriers[slot]), phase)
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[2 + slot]))
            if slot == 1:
                phase = phase ^ 1


@T.prim_func
def tma_completion_many_waiters(
    source: T.Buffer((8, 64), "float16"),
    output: T.Buffer((1,), "float16"),
):
    T.device_entry()
    warpgroup = T.warpgroup_id([4])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])

    pool = T.SMEMPool()
    barrier = TMABar(pool, 1)
    shared = pool.alloc_tcgen05_mma_AB((8, 64), "float16")
    pool.commit()
    barrier.init(1)

    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if (warpgroup == 0) & (warp == 0):
        if T.cuda.elect_sync():
            Tx.copy_async(
                shared[:, :],
                source[:, :],
                dispatch="tma_auto",
                mbar=barrier.ptr_to([0]),
            )
            barrier.arrive(0, tx_count=8 * 64 * 2)

    barrier.wait(0, 0)
    if (warpgroup == 0) & (warp == 0) & (lane == 0):
        output[0] = shared[0, 0]


@T.prim_func
def native_conditional_tcgen_alloc():
    T.device_entry()
    warp = T.warp_id([9])
    _lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")

    if warp == 8:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def native_cross_warp_tmem_dealloc(ordered: T.int32):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    gate = T.alloc_buffer((1,), "uint64", scope="shared")
    value = T.alloc_local((1,), "uint32")

    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(gate[0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.cuda.cta_sync()

    if ordered != 0:
        if warp == 1:
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](value[0], address[0])
            T.ptx.tcgen05.wait__ld.sync.aligned()
            if lane == 0:
                T.ptx.mbarrier.arrive.shared.b64(T.address_of(gate[0]))
        elif warp == 0:
            if lane == 0:
                T.cuda.mbarrier_wait(T.address_of(gate[0]), 0)
            T.cuda.warp_sync()
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
    else:
        if warp == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
            if lane == 0:
                T.ptx.mbarrier.arrive.shared.b64(T.address_of(gate[0]))
        elif warp == 1:
            if lane == 0:
                T.cuda.mbarrier_wait(T.address_of(gate[0]), 0)
            T.cuda.warp_sync()
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](value[0], address[0])
            T.ptx.tcgen05.wait__ld.sync.aligned()


@T.prim_func
def native_rank_conditional_tmem_pool():
    T.device_entry()
    T.cta_id([2])
    cluster_rank = T.cta_id_in_cluster([2])
    T.warpgroup_id([1])

    pool = T.SMEMPool()
    tmem_address = pool.alloc((1,), "uint32", align=4)
    pool.commit()

    rank0_pool = T.TMEMPool(
        pool,
        total_cols=64,
        cta_group=1,
        alloc_warp=0,
        dealloc_warp=0,
        tmem_addr=tmem_address,
    )
    rank0_pool.alloc_tcgen05_mma_D((64, 64), "float32", M=64, cta_group=1)
    rank1_pool = T.TMEMPool(
        pool,
        total_cols=512,
        cta_group=1,
        alloc_warp=0,
        dealloc_warp=0,
        tmem_addr=tmem_address,
    )
    rank1_pool.alloc_tcgen05_mma_D((128, 128), "float32", M=128, cta_group=1)

    if cluster_rank == 0:
        rank0_pool.commit()
    else:
        rank1_pool.commit()
    T.cuda.cluster_sync()
    if cluster_rank == 0:
        rank0_pool.dealloc()
    else:
        rank1_pool.dealloc()


@T.prim_func
def native_thread_topology_conditional_tmem_pool():
    T.device_entry()
    T.cta_id([2])
    cluster_rank = T.cta_id_in_cluster([2])
    T.thread_id([64])

    pool = T.SMEMPool()
    tmem_address = pool.alloc((1,), "uint32", align=4)
    pool.commit()

    rank0_pool = T.TMEMPool(
        pool,
        total_cols=64,
        cta_group=1,
        alloc_warp=0,
        dealloc_warp=0,
        tmem_addr=tmem_address,
    )
    rank0_pool.alloc_tcgen05_mma_D((64, 64), "float32", M=64, cta_group=1)
    rank1_pool = T.TMEMPool(
        pool,
        total_cols=128,
        cta_group=1,
        alloc_warp=0,
        dealloc_warp=0,
        tmem_addr=tmem_address,
    )
    rank1_pool.alloc_tcgen05_mma_D((128, 128), "float32", M=128, cta_group=1)

    if cluster_rank == 0:
        rank0_pool.commit()
    else:
        rank1_pool.commit()
    T.cuda.cluster_sync()
    if cluster_rank == 0:
        rank0_pool.dealloc()
    else:
        rank1_pool.dealloc()


def native_setmaxnreg_budget_kernel(settings):
    """Build representative four-WG register-budget shapes."""

    setting0, setting1, setting2, setting3 = settings

    @T.prim_func
    def kernel():
        T.device_entry()
        wg = T.warpgroup_id([4])
        _warp = T.warp_id_in_wg([4])
        _lane = T.lane_id([32])
        if wg == 0:
            if setting0 is not None:
                T.ptx[f"setmaxnreg.{'inc' if setting0[0] else 'dec'}.sync.aligned.u32"](setting0[1])
        elif wg == 1:
            if setting1 is not None:
                T.ptx[f"setmaxnreg.{'inc' if setting1[0] else 'dec'}.sync.aligned.u32"](setting1[1])
        elif wg == 2:
            if setting2 is not None:
                T.ptx[f"setmaxnreg.{'inc' if setting2[0] else 'dec'}.sync.aligned.u32"](setting2[1])
        else:
            if setting3 is not None:
                T.ptx[f"setmaxnreg.{'inc' if setting3[0] else 'dec'}.sync.aligned.u32"](setting3[1])

    return kernel


def native_six_warpgroup_setmaxnreg_budget(cg1_target: int):
    """Build the working and hanging six-WG register splits from the KDA artifacts."""

    @T.prim_func
    def kernel():
        T.device_entry()
        wg = T.warpgroup_id([6])
        _warp = T.warp_id_in_wg([4])
        _lane = T.lane_id([32])
        if wg == 0:
            T.ptx.setmaxnreg.inc.sync.aligned.u32(104)
        elif wg == 1:
            T.ptx.setmaxnreg.inc.sync.aligned.u32(104)
        elif wg == 2:
            T.ptx.setmaxnreg.inc.sync.aligned.u32(104)
        elif wg == 3:
            T.ptx.setmaxnreg.inc.sync.aligned.u32(cg1_target)
        elif wg == 4:
            T.ptx.setmaxnreg.dec.sync.aligned.u32(40)
        else:
            T.ptx.setmaxnreg.dec.sync.aligned.u32(48)

    return kernel


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-synccheck-kernel-contracts")


def _run(kernel, *, inputs: dict[str, object], cache_dir):
    return internal_synccheck(
        kernel,
        inputs=inputs,
        cache_dir=cache_dir,
        max_workers=1,
        native_loop_iteration_budget=1_000,
        native_loop_reschedule_quantum=16,
    )


def _assert_exact_error(report, kind: str) -> None:
    assert report.verdict == "error"
    assert [(finding.status, finding.kind) for finding in report.findings] == [("error", kind)]
    native = report.to_dict()["native"]
    assert native["verdict"] == "error"
    assert native["incomplete"] == []


def test_native_synccheck_models_the_trailing_ninth_warp_and_launch_attr(native_cache_dir):
    report = _run(
        native_trailing_warp_mbarrier,
        inputs={"expected_arrivals": np.uint32(8 * 32)},
        cache_dir=native_cache_dir,
    )

    report.require_clean()
    native = report.to_dict()["native"]
    assert native["stats"]["task_count"] == 9
    assert native["stats"]["completed_task_count"] == 9


def test_native_synccheck_trailing_warp_does_not_hide_real_under_arrival(native_cache_dir):
    report = _run(
        native_trailing_warp_mbarrier,
        inputs={"expected_arrivals": np.uint32(9 * 32)},
        cache_dir=native_cache_dir,
    )

    _assert_exact_error(report, "deadlock")
    assert "arrival_count=256/288" in report.findings[0].message


@pytest.mark.parametrize(
    ("mode", "epoch", "expected_arrivals"),
    [
        pytest.param(0, 0, 1, id="thread-zero"),
        pytest.param(1, 0, 1, id="bitwise-data-and-thread-zero"),
        pytest.param(2, 0, 4, id="one-elected-lane-per-warp"),
        pytest.param(3, 0, 1, id="local-leader-relay"),
    ],
)
def test_native_synccheck_counts_exact_lane_and_elect_participants(
    native_cache_dir, mode: int, epoch: int, expected_arrivals: int
):
    report = _run(
        native_lane_and_elect_mbarrier,
        inputs={
            "mode": np.int32(mode),
            "epoch": np.int32(epoch),
            "expected_arrivals": np.uint32(expected_arrivals),
        },
        cache_dir=native_cache_dir,
    )

    report.require_clean()
    assert report.findings == []


def test_native_synccheck_resolves_data_guard_instead_of_guessing_participation(
    native_cache_dir,
):
    report = _run(
        native_lane_and_elect_mbarrier,
        inputs={"mode": np.int32(1), "epoch": np.int32(1), "expected_arrivals": np.uint32(1)},
        cache_dir=native_cache_dir,
    )

    _assert_exact_error(report, "deadlock")
    assert all(finding.kind != "control_flow_unresolved" for finding in report.findings)


@pytest.mark.parametrize(
    ("mode", "bound"),
    [
        pytest.param(0, 32, id="uniform"),
        pytest.param(2, 32, id="warp-uniform-guard"),
        pytest.param(4, 32, id="width-sixteen-full-warp-participation"),
    ],
)
def test_native_synccheck_accepts_full_warp_collectives_inside_while(
    native_cache_dir, mode: int, bound: int
):
    report = _run(
        native_collective_in_while,
        inputs={"mode": np.int32(mode), "bound": np.int32(bound)},
        cache_dir=native_cache_dir,
    )

    report.require_clean()


@pytest.mark.parametrize(
    ("mode", "bound", "mask"),
    [
        pytest.param(1, 32, "0x00000001", id="lane-zero"),
        pytest.param(3, 32, "0x00000001", id="elect-sync"),
        pytest.param(4, 16, "0x0000ffff", id="width-sixteen-half-warp-branch"),
    ],
)
def test_native_synccheck_reports_divergent_collective_inside_while_at_source(
    native_cache_dir, mode: int, bound: int, mask: str
):
    report = _run(
        native_collective_in_while,
        inputs={"mode": np.int32(mode), "bound": np.int32(bound)},
        cache_dir=native_cache_dir,
    )

    _assert_exact_error(report, "warp_collective_divergence")
    assert mask in report.findings[0].message
    assert "shfl.sync" in report.findings[0].message
    rendered = report.format()
    assert "Source: warp" in rendered
    expected_width = 16 if mode == 4 else 32
    assert f'T.cuda.warp_reduce(T.float32(1.0), "sum", {expected_width})' in rendered


def test_native_synccheck_does_not_drop_setmaxnreg_inside_loop(native_cache_dir):
    report = _run(native_setmaxnreg_inside_loop, inputs={}, cache_dir=native_cache_dir)

    _assert_exact_error(report, "setmaxnreg_missing_warpgroup_sync")
    assert (
        "all warps in the warpgroup must synchronize explicitly before a subsequent "
        "setmaxnreg instruction"
    ) in report.findings[0].message
    rendered = report.format()
    assert "Source: warp" in rendered
    assert "T.ptx.setmaxnreg.dec.sync.aligned.u32(88)" in rendered


def test_native_synccheck_accepts_setmaxnreg_with_trailing_warps(native_cache_dir):
    report = _run(native_setmaxnreg_with_trailing_warps, inputs={}, cache_dir=native_cache_dir)

    report.require_clean()
    assert report.findings == []


def test_native_synccheck_preserves_depth_two_multi_generation_pipeline(native_cache_dir):
    report = _run(native_mbarrier_depth_two_pipeline, inputs={}, cache_dir=native_cache_dir)

    report.require_clean()
    native = report.to_dict()["native"]
    effects = [effect["effect"] for effect in native["effects"]]
    assert effects.count("mbarrier.init") == 4
    assert effects.count("mbarrier.arrive") == 8
    assert effects.count("mbarrier.wait") == 6


def test_native_synccheck_reduces_tma_completion_waiter_interleavings(native_cache_dir):
    report = internal_synccheck(
        tma_completion_many_waiters,
        inputs={
            "source": np.full((8, 64), 7, dtype=np.float16),
            "output": np.zeros((1,), dtype=np.float16),
        },
        cache_dir=native_cache_dir,
        max_workers=1,
        resource_limits=numsim.ResourceLimits(
            max_schedules=100,
            max_backtrack_nodes=64,
            max_events_per_run=100_000,
            max_total_events=1_000_000,
            max_loop_steps=1_000_000,
            max_wall_time_ms=30_000,
            max_diagnostic_bytes=1_000_000,
        ),
    )

    report.require_clean()
    native = report.to_dict()["native"]
    assert native["coverage"]["termination"]["kind"] == "worklist_exhausted"
    assert native["search"]["visited_state_count"] <= 32
    assert native["search"]["explored_transition_count"] <= 64


def test_native_synccheck_keeps_conditional_tcgen_alloc_at_one_warp(native_cache_dir):
    report = _run(native_conditional_tcgen_alloc, inputs={}, cache_dir=native_cache_dir)

    report.require_clean()
    native = report.to_dict()["native"]
    assert native["stats"]["task_count"] == 9
    assert native["stats"]["completed_task_count"] == 9


def test_native_synccheck_requires_cross_warp_tmem_quiescence_before_dealloc(
    native_cache_dir,
):
    ordered = _run(
        native_cross_warp_tmem_dealloc,
        inputs={"ordered": np.int32(1)},
        cache_dir=native_cache_dir,
    )

    ordered.require_clean()

    unordered = _run(
        native_cross_warp_tmem_dealloc,
        inputs={"ordered": np.int32(0)},
        cache_dir=native_cache_dir,
    )

    assert unordered.verdict == "error"
    assert unordered.findings[0].kind == "synchronization_collective_publication"
    assert "not covered by any live allocation" in unordered.findings[0].message


@pytest.mark.parametrize(
    ("kernel", "expected_tasks"),
    [
        pytest.param(native_rank_conditional_tmem_pool, 8, id="rank-conditional-two-cta"),
        pytest.param(
            native_thread_topology_conditional_tmem_pool,
            4,
            id="thread-topology-two-warps",
        ),
    ],
)
def test_native_synccheck_executes_conditional_tmem_pool_with_exact_topology(
    native_cache_dir, kernel, expected_tasks: int
):
    report = _run(kernel, inputs={}, cache_dir=native_cache_dir)

    report.require_clean()
    native = report.to_dict()["native"]
    assert native["stats"]["task_count"] == expected_tasks
    assert native["stats"]["completed_task_count"] == expected_tasks


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param([(True, 128)] * 4, id="full-register-file-budget"),
        pytest.param(
            [(True, 200), (False, 120), (False, 96), (False, 96)],
            id="conserved-asymmetric-split",
        ),
        pytest.param([None, None, None, None], id="no-setmaxnreg"),
        pytest.param(
            [(True, 160), (False, 80), (False, 80), None],
            id="missing-wg-default-within-budget",
        ),
    ],
)
def test_native_synccheck_accepts_native_setmaxnreg_budget_shapes(native_cache_dir, settings):
    report = _run(native_setmaxnreg_budget_kernel(settings), inputs={}, cache_dir=native_cache_dir)

    report.require_clean()
    assert report.findings == []


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param(
            [(True, 224), (True, 232), (False, 48), (False, 64)],
            id="oversubscribed-final-allocation",
        ),
        pytest.param(
            [(True, 232), (False, 120), (False, 120), None],
            id="missing-wg-default-tips-budget",
        ),
    ],
)
def test_native_synccheck_rejects_native_setmaxnreg_oversubscription(native_cache_dir, settings):
    report = _run(native_setmaxnreg_budget_kernel(settings), inputs={}, cache_dir=native_cache_dir)

    assert report.verdict == "error"
    assert report.to_dict()["native"]["incomplete"] == []
    assert any(
        finding.kind in {"setmaxnreg_pool_deadlock", "deadlock"}
        for finding in report.findings
    )
    rendered = report.format()
    assert "T.ptx.setmaxnreg" in rendered
    assert "execution cannot make progress" in rendered


def test_native_synccheck_accepts_six_warpgroup_launch_allocation(native_cache_dir):
    report = _run(
        native_six_warpgroup_setmaxnreg_budget(80),
        inputs={},
        cache_dir=native_cache_dir,
    )

    report.require_clean()
    assert report.findings == []


def test_native_synccheck_rejects_six_warpgroup_rounding_residual_as_pool(
    native_cache_dir,
):
    report = _run(
        native_six_warpgroup_setmaxnreg_budget(112),
        inputs={},
        cache_dir=native_cache_dir,
    )

    assert report.verdict == "error"
    assert any(
        finding.kind in {"setmaxnreg_pool_deadlock", "deadlock"}
        for finding in report.findings
    )


def test_native_synccheck_report_has_machine_readable_clean_contract(native_cache_dir):
    report = _run(
        native_collective_in_while,
        inputs={"mode": np.int32(0), "bound": np.int32(32)},
        cache_dir=native_cache_dir,
    )

    assert report.verdict == "clean"
    assert report.findings == []
    report.require_clean()
    payload = report.to_dict()
    assert payload["checker"] == "synccheck"
    assert payload["verdict"] == "clean"
    assert payload["findings"] == []
    json.dumps(payload)
    for native_name in ("ok", "errors", "skipped", "has_errors"):
        assert not hasattr(report, native_name)


def test_native_synccheck_report_has_machine_readable_error_contract(native_cache_dir):
    report = _run(
        native_collective_in_while,
        inputs={"mode": np.int32(1), "bound": np.int32(32)},
        cache_dir=native_cache_dir,
    )

    _assert_exact_error(report, "warp_collective_divergence")
    with pytest.raises(RuntimeError, match="synccheck error"):
        report.require_clean()
    payload = report.to_dict()
    assert payload["checker"] == "synccheck"
    assert payload["verdict"] == "error"
    assert payload["findings"]
    json.dumps(payload)
