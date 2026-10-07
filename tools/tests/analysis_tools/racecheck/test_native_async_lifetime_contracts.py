"""Racecheck contracts for asynchronous-memory lifetimes."""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import racecheck
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import tmem_datapath_layout, wg_local_layout


@T.prim_func
def native_tcgen_transfer_lifetime(mode: T.int32):
    """Exercise LD/ST hazards and the matching directional waits."""

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
        registers[col] = T.cast(row * 32 + col, "float32")
    T.cuda.cta_sync()

    if mode == 0:
        Tx.wg.copy_async(register_tile[:, :], tmem[:, :])
        Tx.wg.copy_async(tmem[:, :], register_tile[:, :])
    elif mode == 1:
        Tx.wg.copy_async(register_tile[:, :], tmem[:, :])
        T.ptx.tcgen05.wait__ld.sync.aligned()
        Tx.wg.copy_async(tmem[:, :], register_tile[:, :])
    elif mode == 2:
        Tx.wg.copy_async(tmem[:, :], register_tile[:, :])
        Tx.wg.copy_async(register_tile[:, :], tmem[:, :])
    elif mode == 3:
        Tx.wg.copy_async(tmem[:, :], register_tile[:, :])
        T.ptx.tcgen05.wait__st.sync.aligned()
        Tx.wg.copy_async(register_tile[:, :], tmem[:, :])
    else:
        Tx.wg.copy_async(tmem[:, :], register_tile[:, :])
        Tx.wg.copy_async(tmem[:, :], register_tile[:, :])

    # Drain surviving work so clean cases can release the TMEM allocation.
    T.ptx.tcgen05.wait__ld.sync.aligned()
    T.ptx.tcgen05.wait__st.sync.aligned()


@T.prim_func
def native_tcgen_cross_warpgroup_race_after_waits():
    """Completion waits do not order conflicting work from two warpgroups."""

    T.device_entry()
    warpgroup = T.warpgroup_id([2])
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

    if warpgroup == 0:
        for col in T.serial(32):
            tmem[row, col] = T.cast(row * 32 + col, "float32")
    else:
        for col in T.serial(32):
            registers[col] = T.cast(row * 32 + col, "float32")
    T.cuda.cta_sync()

    if warpgroup == 0:
        Tx.wg.copy_async(register_tile[:, :], tmem[:, :])
        T.ptx.tcgen05.wait__ld.sync.aligned()
    else:
        Tx.wg.copy_async(tmem[:, :], register_tile[:, :])
        T.ptx.tcgen05.wait__st.sync.aligned()


def _make_native_tma_store_fifo_lifetime(wait_n: int):
    @T.prim_func
    def native_tma_store_fifo_lifetime(
        shape: T.int32,
        overwrite_source: T.int32,
        output: T.Buffer((12,), "float32"),
    ):
        """Exercise FIFO groups, including empty and predicated-off commits."""

        T.device_entry()
        warp = T.warp_id([2])
        lane = T.lane_id([32])
        source0 = T.alloc_buffer((4,), "float32", scope="shared", align=128)
        source1 = T.alloc_buffer((4,), "float32", scope="shared", align=128)
        source2 = T.alloc_buffer((4,), "float32", scope="shared", align=128)
        barrier = T.alloc_buffer((1,), "uint64", scope="shared")

        if (warp == 0) and (lane < 4):
            source0[lane] = T.cast(lane, "float32")
            source1[lane] = T.cast(lane + 4, "float32")
            source2[lane] = T.cast(lane + 8, "float32")
        if (warp == 0) and (lane == 0):
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cta_sync()
        T.ptx.fence.proxy.async_.shared__cta()

        if (warp == 0) and (lane == 0):
            Tx.copy_async(output[0:4], source0[:], dispatch="tma_auto")
            T.ptx.cp.async_.bulk.commit_group(pred=shape != 3)
            if shape == 0:
                Tx.copy_async(output[4:8], source1[:], dispatch="tma_auto")
                T.ptx.cp.async_.bulk.commit_group()
            elif shape == 1:
                # An empty commit is still a FIFO entry.
                T.ptx.cp.async_.bulk.commit_group()
            elif shape == 2:
                Tx.copy_async(output[4:8], source1[:], dispatch="tma_auto")
                T.ptx.cp.async_.bulk.commit_group()
                Tx.copy_async(output[8:12], source2[:], dispatch="tma_auto")
                T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group.read(wait_n)
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(barrier[0]))

        if warp == 1:
            if lane == 0:
                T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
            # The mbarrier acquire belongs only to lane 0. Publish it to the lanes
            # that overwrite the source so this test isolates FIFO lifetime rather
            # than relying on an invalid warp-wide acquire.
            T.cuda.warp_sync()
            if lane < 4:
                if overwrite_source == 0:
                    source0[lane] = T.cast(lane, "float32")
                elif overwrite_source == 1:
                    source1[lane] = T.cast(lane, "float32")
                else:
                    source2[lane] = T.cast(lane, "float32")

    return native_tma_store_fifo_lifetime


@T.prim_func
def native_tma_store_cross_warp_race_after_wait(output: T.Buffer((4,), "float32")):
    """A bulk wait completes each store but does not order the two issuers."""

    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    source = T.alloc_buffer((4,), "float32", scope="shared", align=128)

    if (warp == 0) and (lane < 4):
        source[lane] = T.cast(lane, "float32")
    T.cuda.cta_sync()
    T.ptx.fence.proxy.async_.shared__cta()

    if lane == 0:
        Tx.copy_async(output[:], source[:], dispatch="tma_auto")
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def native_tma_store_uncommitted_exit(
    overwrite_source: T.int32,
    output: T.Buffer((4,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source = T.alloc_buffer((4,), "float32", scope="shared", align=128)

    if lane < 4:
        source[lane] = T.cast(lane, "float32")
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        Tx.copy_async(output[:], source[:], dispatch="tma_auto")
        if overwrite_source != 0:
            source[0] = T.float32(100)


@T.prim_func
def native_classic_cp_async_lifetime(
    mode: T.int32,
    source: T.Buffer((128,), "float32"),
    output: T.Buffer((128,), "float32"),
):
    """Keep classic cp.async source/destination accesses live until wait_group."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "float32", scope="shared")

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


@T.prim_func
def native_classic_cp_async_orders_prior_lane_write(
    source: T.Buffer((128,), "float32"),
):
    """A cp.async operation starts after prior instructions in its issuing lane."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "float32", scope="shared")

    shared[lane * 4] = T.float32(0)
    T.ptx["cp.async.cg.shared.global"](
        T.address_of(shared[lane * 4]),
        T.address_of(source[lane * 4]),
        16,
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)


@T.prim_func
def native_classic_cp_async_cross_warp_source_race(
    source: T.Buffer((128,), "float32"),
):
    """Keep completion from masking a missing cross-warp source ordering edge."""

    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "float32", scope="shared")

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


@T.prim_func
def native_tcgen_copy_completion_shared_reuse(with_post_completion_fence: T.int32):
    """A TCGEN completion alone does not bridge its async read to generic reuse."""

    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128, 16), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer(
        (128, 16),
        "uint8",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 16),
        allocated_addr=0,
    )

    for offset in T.unroll(32):
        shared[warp * 32 + lane + (offset // 16) * 64, offset % 16] = T.cast(offset, "uint8")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    T.ptx.fence.proxy.async_.shared__cta()

    if (warp == 0) and (lane == 0):
        Tx.copy_async(tmem[:, :], shared[:, :], shape="128x128b", cta_group=1)
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    if warp == 1:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        if with_post_completion_fence != 0:
            T.ptx.fence.proxy.async_.shared__cta()
        shared[lane, 0] = T.cast(lane, "uint8")


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
    assert finding["kind"] == ("tmem_lifetime_review" if expected_status == "review" else "data_race")
    assert finding["access_pair"] in expected_kinds
    assert finding["overlap"]["byte_len"] > 0
    for witness in (finding["prior"], finding["current"]):
        assert witness["operation"]["source_op_id"] is not None
        assert witness["operation"]["source"]["source_text"].strip()
    return finding


# Only a prior TMEM load can carry the unmodeled register dependency;
# conflicts behind a store frontier are exact errors.
@pytest.mark.parametrize(
    ("mode", "kind", "status"),
    [
        (0, "read_write", "review"),
        (2, "write_read", "error"),
        (4, "write_write", "error"),
    ],
)
def test_tcgen_transfer_requires_the_directional_wait(mode, kind, status):
    report = racecheck(
        native_tcgen_transfer_lifetime,
        inputs={"mode": np.int32(mode)},
    )
    finding = _assert_physical_finding(report, kind, expected_status=status)
    assert finding["ordering_domain"] == "completion"
    assert finding["ordering_failure"] == "async_lifetime_not_drained"
    assert finding["prior"]["space"] == finding["current"]["space"] == "tmem"
    assert (
        finding["prior"]["operation"]["global_warp_id"]
        == finding["current"]["operation"]["global_warp_id"]
    )


@pytest.mark.parametrize("mode", [1, 3])
def test_tcgen_directional_wait_drains_prior_work(mode):
    _assert_clean(
        racecheck(
            native_tcgen_transfer_lifetime,
            inputs={"mode": np.int32(mode)},
        )
    )


def test_tcgen_waits_do_not_mask_missing_cross_warpgroup_ordering():
    report = racecheck(native_tcgen_cross_warpgroup_race_after_waits)
    assert report.verdict == "error"
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert native["findings"]
    assert {
        (finding["ordering_domain"], finding["ordering_failure"])
        for finding in native["findings"]
    } == {("execution", "missing_inter_actor_sync")}
    assert {
        endpoint["space"]
        for finding in native["findings"]
        for endpoint in (finding["prior"], finding["current"])
    } == {"tmem"}


def test_tcgen_completion_without_a_post_completion_proxy_fence_reports_reuse_race():
    finding = _assert_physical_finding(
        racecheck(
            native_tcgen_copy_completion_shared_reuse,
            inputs={"with_post_completion_fence": np.int32(0)},
        ),
        {"read_write", "write_read"},
    )
    assert finding["ordering_domain"] == "memory"
    assert finding["ordering_failure"] == "missing_proxy_bridge"
    assert finding["proxy_bridge"] == {
        "prior_proxy": "async",
        "current_proxy": "generic",
        "prior_domain": "shared_cta",
        "current_domain": "shared_cta",
    }
    assert finding["prior"]["space"] == finding["current"]["space"] == "shared"


def test_tcgen_post_completion_proxy_fence_orders_generic_shared_reuse():
    _assert_clean(
        racecheck(
            native_tcgen_copy_completion_shared_reuse,
            inputs={"with_post_completion_fence": np.int32(1)},
        )
    )


@pytest.mark.parametrize(
    ("shape", "wait_n", "overwrite_source", "should_race"),
    [
        (0, 0, 1, False),
        (0, 1, 0, False),
        (0, 1, 1, True),
        (1, 1, 0, False),
        (1, 2, 0, True),
        (2, 2, 0, False),
        (2, 2, 1, True),
        (2, 2, 2, True),
        (3, 0, 0, True),
    ],
)
def test_tma_store_source_lifetime_uses_fifo_groups(shape, wait_n, overwrite_source, should_race):
    report = racecheck(
        _make_native_tma_store_fifo_lifetime(wait_n),
        inputs={
            "shape": np.int32(shape),
            "overwrite_source": np.int32(overwrite_source),
            "output": np.zeros(12, dtype=np.float32),
        },
    )
    if not should_race:
        _assert_clean(report)
        return
    finding = _assert_physical_finding(report, {"read_write", "write_read"})
    assert finding["prior"]["space"] == finding["current"]["space"] == "shared"


def test_tma_store_wait_does_not_mask_missing_cross_warp_ordering():
    report = racecheck(
        native_tma_store_cross_warp_race_after_wait,
        inputs={"output": np.zeros(4, dtype=np.float32)},
    )
    assert report.verdict == "error"
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert native["findings"]
    assert {
        (finding["ordering_domain"], finding["ordering_failure"])
        for finding in native["findings"]
    } == {("execution", "missing_inter_actor_sync")}
    assert {
        endpoint["space"]
        for finding in native["findings"]
        for endpoint in (finding["prior"], finding["current"])
    } == {"global"}


@pytest.mark.parametrize("overwrite_source", [0, 1], ids=["no-reuse", "premature-reuse"])
def test_uncommitted_tma_store_still_owns_its_source_until_kernel_exit(overwrite_source):
    report = racecheck(
        native_tma_store_uncommitted_exit,
        inputs={
            "overwrite_source": np.int32(overwrite_source),
            "output": np.zeros(4, dtype=np.float32),
        },
    )
    if overwrite_source == 0:
        _assert_clean(report)
        return
    finding = _assert_physical_finding(report, {"read_write", "write_read"})
    assert finding["prior"]["space"] == finding["current"]["space"] == "shared"


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
        (finding["ordering_domain"], finding["ordering_failure"])
        for finding in native["findings"]
    } == {("execution", "missing_inter_actor_sync")}
    assert {
        endpoint["space"]
        for finding in native["findings"]
        for endpoint in (finding["prior"], finding["current"])
    } == {"global"}
