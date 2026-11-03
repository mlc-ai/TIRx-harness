"""Launch coordinates and dimensions of a tvm-tirx device entry."""

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tvm_tirx.script import tirx as T

GRID, BLOCK, CLUSTER = 4, 256, 2


@T.function
def launch_indices(output: T.Tensor((GRID, BLOCK, 17), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=GRID, block=BLOCK, cluster=CLUSTER))
    cta = T.cuda.block_idx("x")
    tid = T.cuda.thread_idx("x")
    output[cta, tid, 0] = cta
    output[cta, tid, 1] = tid
    output[cta, tid, 2] = T.cuda.block_idx("y") + T.cuda.thread_idx("z")
    output[cta, tid, 3] = T.cuda.cluster_id("x")
    output[cta, tid, 4] = T.cuda.cluster_cta_id("x")
    output[cta, tid, 5] = T.cuda.cta_pair_id()
    output[cta, tid, 6] = T.cuda.grid_dim("x")
    output[cta, tid, 7] = T.cuda.block_dim("x")
    output[cta, tid, 8] = T.cuda.cluster_dim("x")
    output[cta, tid, 9] = T.cuda.grid_dim("y") * T.cuda.cluster_dim("z")
    output[cta, tid, 10] = T.cuda.lane_id()
    output[cta, tid, 11] = T.cuda.warp_id()
    output[cta, tid, 12] = T.cuda.linear_thread_id()
    output[cta, tid, 13] = T.cuda.warpgroup_id()
    output[cta, tid, 14] = T.cuda.warp_in_warpgroup()
    output[cta, tid, 15] = T.cuda.thread_in_warpgroup()
    output[cta, tid, 16] = T.cuda.block_dim("z")


def _expected() -> np.ndarray:
    cta = np.arange(GRID)[:, None]
    tid = np.arange(BLOCK)[None, :]
    columns = [
        cta,
        tid,
        0,
        cta // CLUSTER,
        cta % CLUSTER,
        cta % 2,
        GRID,
        BLOCK,
        CLUSTER,
        1,
        tid % 32,
        tid // 32,
        tid,
        tid // 128,
        (tid // 32) % 4,
        tid % 128,
        1,
    ]
    return np.stack([np.broadcast_to(column, (GRID, BLOCK)) for column in columns], axis=-1).astype(
        np.int32
    )


def test_launch_indices_follow_the_device_entry_launch(tmp_path):
    module = numsim.transpile(launch_indices, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"output": np.full((GRID, BLOCK, 17), -1, dtype=np.int32)},
        outputs=("output",),
    )
    np.testing.assert_array_equal(result.outputs["output"], _expected())


@T.function
def two_dimensional_grid(output: T.Tensor((2, 2), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=(2, 2), block=32))
    if T.cuda.lane_id() == 0:
        output[T.cuda.block_idx("x"), T.cuda.block_idx("y")] = 1


@T.function
def preferred_cluster_differs(output: T.Tensor((4,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=4, block=32, cluster=1, preferred_cluster=2))
    if T.cuda.lane_id() == 0:
        output[T.cuda.block_idx("x")] = T.cuda.cluster_dim("x")


@pytest.mark.parametrize(
    "kernel, message",
    [
        (two_dimensional_grid, "one-dimensional, but grid.y is 2"),
        (preferred_cluster_differs, "preferred_cluster must equal cluster"),
    ],
)
def test_launches_numsim_cannot_execute_exactly_are_rejected(tmp_path, kernel, message):
    with pytest.raises(UnsupportedTIRxError, match=message):
        numsim.transpile(kernel, cache_dir=tmp_path)
