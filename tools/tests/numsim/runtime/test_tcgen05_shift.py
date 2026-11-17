"""Standalone tcgen05.shift: asynchronous TMEM row shift completed by tcgen05.commit."""

import numpy as np
import pytest

from tests.numsim.support.execution import assert_rejected, run_checked
from tirx_harness import racecheck, synccheck
from tvm_tirx.script import tirx as T

COLUMNS = 32


def shift_kernel(cta_group=1, *, column=0, lane=0, wait=True, predicate=None):
    """Fill 32 TMEM columns, shift the eight columns at ``column``, read all back."""
    alloc = f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32"
    dealloc = f"tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32"
    relinquish = f"tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned"
    commit = f"tcgen05.commit.cta_group::{cta_group}.mbarrier::arrive::one.shared::cluster.b64"
    shift = f"tcgen05.shift.cta_group::{cta_group}.down"
    st = f"tcgen05.st.sync.aligned.32x32b.x{COLUMNS}.b32"
    ld = f"tcgen05.ld.sync.aligned.32x32b.x{COLUMNS}.b32"
    offset = (lane << 16) | column

    @T.function
    def kernel(
        seed: T.Tensor((cta_group, 128, COLUMNS), "uint32"),
        issue: T.int32,
        out: T.Tensor((cta_group, 128, COLUMNS), "uint32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=cta_group, block=128, cluster=cta_group))
        cta = T.cuda.cluster_cta_id("x")
        warp = T.cuda.warp_in_warpgroup()
        lane_id = T.cuda.lane_id()
        row = T.meta_var(warp * 32 + lane_id)
        address = T.alloc_shared((1,), "uint32")
        barrier = T.alloc_shared((1,), "uint64")
        regs = T.alloc_local((COLUMNS,), "uint32")
        if warp == 0:
            T.ptx[alloc](address.ptr_to([0]), COLUMNS)
            if lane_id == 0:
                T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cta_sync()
        for i in T.unroll(COLUMNS):
            regs[i] = seed[cta, row, i]
        # Each warp writes its own 32-lane partition.
        T.ptx[st](address[0] + T.uint32(warp * 32 << 16), *[regs[i] for i in range(COLUMNS)])
        T.ptx.tcgen05.wait__st.sync.aligned()
        T.cuda.cluster_sync()
        if cta == 0 and warp == 0 and lane_id == 0:
            if T.constexpr(predicate is None):
                T.ptx[shift](address[0] + T.uint32(offset))
            else:
                T.ptx[shift](address[0] + T.uint32(offset), pred=issue != 0)
            T.ptx[commit](barrier.ptr_to([0]))
        if T.constexpr(wait):
            if cta == 0 and warp == 0:
                T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.cuda.cluster_sync()
        T.ptx[ld](*[regs[i] for i in range(COLUMNS)], address[0] + T.uint32(warp * 32 << 16))
        T.ptx.tcgen05.wait__ld.sync.aligned()
        for i in T.unroll(COLUMNS):
            out[cta, row, i] = regs[i]
        T.cuda.cluster_sync()
        if warp == 0:
            T.ptx[dealloc](address[0], COLUMNS)
            T.ptx[relinquish]()

    return kernel


def shift_inputs(cta_group, issue=1):
    seed = np.arange(cta_group * 128 * COLUMNS, dtype=np.uint32).reshape(cta_group, 128, COLUMNS)
    seed = seed * np.uint32(2654435761) + np.uint32(17)
    return {
        "seed": seed,
        "issue": np.int32(issue),
        "out": np.zeros_like(seed),
    }


def shift_oracle(seed, column):
    """In every 32-lane partition of every CTA, lane r of the eight shifted
    columns takes lane r + 1's old value; lane 31 keeps its own."""
    expected = seed.copy()
    for start in range(0, 128, 32):
        expected[:, start : start + 31, column : column + 8] = seed[
            :, start + 1 : start + 32, column : column + 8
        ]
    return expected


@pytest.mark.parametrize("cta_group", (1, 2))
@pytest.mark.parametrize("column", (0, 4, 24))
def test_tcgen05_shift_moves_each_partition_down_one_lane(cta_group, column, tmp_path):
    inputs = shift_inputs(cta_group)
    outputs = run_checked(shift_kernel(cta_group, column=column), inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(outputs.outputs["out"], shift_oracle(inputs["seed"], column))


def test_tcgen05_shift_predicate_gates_issue(tmp_path):
    kernel = shift_kernel(predicate=True)
    for issue in (0, 1):
        inputs = shift_inputs(1, issue)
        outputs = run_checked(kernel, inputs, cache_dir=tmp_path)
        expected = shift_oracle(inputs["seed"], 0) if issue else inputs["seed"]
        np.testing.assert_array_equal(outputs.outputs["out"], expected)


@pytest.mark.parametrize("cta_group", (1, 2))
def test_tcgen05_shift_read_before_commit_completion_races(cta_group):
    kernel = shift_kernel(cta_group, wait=False)
    report = racecheck(kernel, shift_inputs(cta_group))
    assert report.verdict == "error", report.format()
    assert "data_race" in report.format() and "tirx.ptx.tcgen05_shift" in report.format()
    synccheck(kernel, shift_inputs(cta_group)).require_clean()


@pytest.mark.parametrize("cta_group", (1, 2))
@pytest.mark.parametrize("lane", (32, 64, 96))
def test_tcgen05_shift_ignores_aligned_address_lane(cta_group, lane, tmp_path):
    # SM107 hardware: every 32-aligned lane shifts all four partitions, as lane 0.
    inputs = shift_inputs(cta_group)
    outputs = run_checked(shift_kernel(cta_group, column=8, lane=lane), inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(outputs.outputs["out"], shift_oracle(inputs["seed"], 8))


@pytest.mark.parametrize(
    "column,lane,reason",
    [
        (0, 16, "row address must name one aligned warp"),
        (28, 0, "not covered by any live allocation"),
        *((column, 0, "misaligned") for column in (1, 2, 3, 5, 6)),
    ],
)
def test_tcgen05_shift_invalid_addresses_are_errors(column, lane, reason):
    assert_rejected(shift_kernel(column=column, lane=lane), shift_inputs(1), reason)
