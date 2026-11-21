"""CTAs whose thread count is not a multiple of 32.

The last warp is partial: its upper lanes do not exist and never execute. PTX
counts `bar` arrivals per warp, so a converged partial warp completes its
warp's 32 arrivals ([VERIFY], see tests/numsim/microtests/test_partial_warp_cta.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm_tirx.script import tirx as T


@T.function
def cta_sync_with_partial_warp(output: T.Tensor((64,), "int32")):
    # 33 threads; output[33:] must stay untouched by the absent lanes.
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=33))
    tid = T.cuda.thread_idx("x")
    shared = T.alloc_shared((33,), "int32")
    shared[tid] = tid * 3 + 1
    T.cuda.cta_sync()
    output[tid] = shared[32 - tid]


@T.function
def named_barrier_with_partial_warp(output: T.Tensor((33,), "int32")):
    # `bar.sync 1, 64`: warp 0 and the one-lane warp 1 each count 32.
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=33))
    tid = T.cuda.thread_idx("x")
    shared = T.alloc_shared((33,), "int32")
    shared[tid] = tid + 1
    T.ptx.bar.sync(T.uint32(1), T.uint32(64))
    output[tid] = shared[32 - tid]


@T.function
def producer_consumer_with_partial_warp(output: T.Tensor((16,), "int32")):
    # 48 threads: the 16-lane warp 1 produces and arrives, warp 0 waits.
    # Warp and lane come from thread_idx: TIRx's warp-index helpers require
    # whole-warp CTAs when compiled for the GPU.
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=48))
    warp = T.cuda.thread_idx("x") // 32
    lane = T.cuda.thread_idx("x") % 32
    shared = T.alloc_shared((16,), "int32")
    if warp == 1:
        shared[lane] = lane * 7
        T.ptx.bar.arrive(T.uint32(2), T.uint32(64))
    else:
        T.ptx.bar.sync(T.uint32(2), T.uint32(64))
        if lane < 16:
            output[lane] = shared[15 - lane]


@T.function
def barrier_count_too_large_for_partial_warp(output: T.Tensor((33,), "int32")):
    # Two warps contribute at most 64 arrivals; 96 never completes.
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=33))
    tid = T.cuda.thread_idx("x")
    T.ptx.bar.sync(T.uint32(1), T.uint32(96))
    output[tid] = tid


@pytest.mark.parametrize(
    "kernel, expected",
    [
        (
            cta_sync_with_partial_warp,
            np.concatenate([np.arange(32, -1, -1) * 3 + 1, np.full(31, -1)]),
        ),
        (named_barrier_with_partial_warp, np.arange(33, 0, -1)),
        (producer_consumer_with_partial_warp, np.arange(15, -1, -1) * 7),
    ],
)
def test_barriers_complete_with_a_partial_warp(tmp_path, kernel, expected):
    inputs = {"output": np.full(expected.size, -1, dtype=np.int32)}
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(module, dict(inputs), outputs=("output",))
    np.testing.assert_array_equal(result.outputs["output"], expected)
    for check in (synccheck, racecheck):
        report = check(kernel, inputs=dict(inputs), cache_dir=tmp_path, max_workers=2)
        assert report.verdict == "clean", report.format()


def test_partial_warp_is_credited_only_up_to_its_warp(tmp_path):
    report = synccheck(
        barrier_count_too_large_for_partial_warp,
        inputs={"output": np.zeros(33, dtype=np.int32)},
        cache_dir=tmp_path,
        max_workers=2,
    )
    assert report.verdict == "error", report.format()
