"""Warp collectives in the partial last warp of a 48-thread CTA.

Absent lanes behave as exited threads ("all non-exited threads named in mask
must execute"): a full member mask is valid and absent lanes contribute
nothing. Reading an absent lane's value is reported, and warp-wide matrix
instructions, which need all 32 hardware lanes, are reported as divergence.
"""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm_tirx.script import tirx as T


@T.function
def full_mask_collectives(output: T.Tensor((48, 3), "int32")):
    # Warp 1 has lanes 0..15; xor offsets up to 8 read only existing lanes.
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=48))
    full: T.let = T.uint32(0xFFFFFFFF)
    tid = T.cuda.thread_idx("x")
    value = T.local_scalar("int32")
    value = tid
    value = value + T.cuda.__shfl_xor_sync(full, value, 8, 32)
    value = value + T.cuda.__shfl_xor_sync(full, value, 4, 32)
    value = value + T.cuda.__shfl_xor_sync(full, value, 2, 32)
    value = value + T.cuda.__shfl_xor_sync(full, value, 1, 32)
    output[tid, 0] = value
    output[tid, 1] = T.Cast("int32", T.cuda.ballot_sync(full, tid >= 0))
    output[tid, 2] = T.Cast("int32", T.cuda.reduce_add_sync_u32(full, T.uint32(1)))


@T.function
def absent_lane_shuffle(output: T.Tensor((48,), "int32")):
    # Offset 16 makes warp 1's lanes 0..15 read lanes 16..31, which do not exist.
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=48))
    full: T.let = T.uint32(0xFFFFFFFF)
    tid = T.cuda.thread_idx("x")
    output[tid] = T.cuda.__shfl_xor_sync(full, tid, 16, 32)


@T.function
def partial_warp_mma(output: T.Tensor((48, 8), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=48))
    tid = T.cuda.thread_idx("x")
    halves = T.alloc_local((16,), "float16")
    d_values = T.alloc_local((8,), "float32")
    for index in T.unroll(16):
        halves[index] = T.float16(0)
    words = halves.view("uint32")
    T.ptx.mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f16(
        *[d_values[index] for index in range(8)],
        *[words[index] for index in range(8)],
    )
    for index in T.unroll(8):
        output[tid, index] = d_values[index]


def test_full_mask_collectives_fold_over_existing_lanes(tmp_path):
    inputs = {"output": np.zeros((48, 3), np.int32)}
    module = numsim.transpile(full_mask_collectives, cache_dir=tmp_path)
    output = numsim.Engine().run(module, dict(inputs), outputs=("output",)).outputs["output"]
    tid = np.arange(48)
    np.testing.assert_array_equal(output[:, 0], [tid[t // 16 * 16 :][:16].sum() for t in tid])
    np.testing.assert_array_equal(output[:, 1], np.where(tid < 32, -1, 0xFFFF))
    np.testing.assert_array_equal(output[:, 2], np.where(tid < 32, 32, 16))
    for check in (synccheck, racecheck):
        report = check(full_mask_collectives, inputs=dict(inputs), cache_dir=tmp_path)
        assert report.verdict == "clean", report.format()


@pytest.mark.parametrize(
    "kernel, shape, message",
    [
        (absent_lane_shuffle, (48,), "does not exist in this partial warp"),
        (partial_warp_mma, (48, 8), "do not exist"),
    ],
)
def test_reading_absent_lanes_is_an_error(tmp_path, kernel, shape, message):
    output = np.zeros(shape, np.float32 if len(shape) == 2 else np.int32)
    report = synccheck(kernel, inputs={"output": output}, cache_dir=tmp_path, max_workers=2)
    assert report.verdict == "error", report.format()
    assert message in report.format()
