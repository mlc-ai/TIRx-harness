"""Launch coordinates and dimensions of a tvm-tirx device entry."""

import numpy as np
import pytest

import tvm_tirx
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


GRID_3D, CLUSTER_3D = (4, 3, 2), (2, 3, 1)


@T.function
def three_dimensional_launch(output: T.Tensor((2, 3, 4, 16), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=GRID_3D, block=32, cluster=CLUSTER_3D))
    if T.cuda.lane_id() == 0:
        x = T.cuda.block_idx("x")
        y = T.cuda.block_idx("y")
        z = T.cuda.block_idx("z")
        output[z, y, x, 0] = T.cuda.cluster_id("x")
        output[z, y, x, 1] = T.cuda.cluster_id("y")
        output[z, y, x, 2] = T.cuda.cluster_id("z")
        output[z, y, x, 3] = T.cuda.cluster_cta_id("x")
        output[z, y, x, 4] = T.cuda.cluster_cta_id("y")
        output[z, y, x, 5] = T.cuda.cluster_cta_id("z")
        output[z, y, x, 6] = T.cuda.grid_dim("y")
        output[z, y, x, 7] = T.cuda.grid_dim("z")
        output[z, y, x, 8] = T.cuda.cluster_dim("y")
        output[z, y, x, 9] = T.cuda.cluster_dim("z")
        output[z, y, x, 10] = T.Cast("int32", T.cuda.mov_sreg(32, "cluster_ctarank"))
        output[z, y, x, 11] = T.Cast("int32", T.cuda.mov_sreg(32, "ctaid.y"))
        output[z, y, x, 12] = T.Cast("int32", T.cuda.mov_sreg(32, "ctaid.z"))
        output[z, y, x, 13] = T.Cast("int32", T.cuda.mov_sreg(32, "nclusterid.y"))
        output[z, y, x, 14] = T.Cast("int32", T.cuda.mov_sreg(32, "cluster_ctaid.y"))
        output[z, y, x, 15] = T.Cast("int32", T.cuda.mov_sreg(32, "nctaid.x"))


def three_dimensional_expected() -> np.ndarray:
    """Each CTA's coordinates by the CUDA definitions, indexed by its blockIdx."""

    grid, cluster = np.array(GRID_3D), np.array(CLUSTER_3D)
    expected = np.empty((*GRID_3D[::-1], 16), dtype=np.int32)
    for z, y, x in np.ndindex(*GRID_3D[::-1]):
        block = np.array([x, y, z])
        cluster_id, cluster_cta = block // cluster, block % cluster
        rank = cluster_cta[0] + cluster[0] * (cluster_cta[1] + cluster[1] * cluster_cta[2])
        expected[z, y, x] = [
            *cluster_id,
            *cluster_cta,
            grid[1],
            grid[2],
            cluster[1],
            cluster[2],
            rank,
            y,
            z,
            grid[1] // cluster[1],
            cluster_cta[1],
            grid[0],
        ]
    return expected


def test_three_dimensional_grid_and_cluster_coordinates_follow_cuda(tmp_path):
    module = numsim.transpile(three_dimensional_launch, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"output": np.full((*GRID_3D[::-1], 16), -1, dtype=np.int32)},
        outputs=("output",),
    )
    np.testing.assert_array_equal(result.outputs["output"], three_dimensional_expected())


def _identity_kernel(block: tuple[int, int, int]):
    threads = int(np.prod(block))
    sregs = ("tid.x", "tid.y", "tid.z", "ntid.x", "ntid.y", "ntid.z", "laneid", "warpid")
    columns = [
        *(f'T.cuda.thread_idx("{axis}")' for axis in "xyz"),
        *(f'T.cuda.block_dim("{axis}")' for axis in "xyz"),
        "T.cuda.linear_thread_id()",
        "T.cuda.warp_id()",
        "T.cuda.lane_id()",
        *(f'T.Cast("int32", T.cuda.mov_sreg(32, "{register}"))' for register in sregs),
    ]
    stores = "\n".join(f"    output[linear, {k}] = {value}" for k, value in enumerate(columns))
    return tvm_tirx.script.from_source(
        f"""
@T.function
def identities(output: T.Tensor(({threads}, {len(columns)}), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block={block}))
    linear = T.cuda.linear_thread_id()
{stores}
""",
        {"T": T},
    )


@pytest.mark.parametrize("block", [(16, 16, 1), (7, 5, 3), (1, 1, 1)])
def test_multidimensional_and_partial_warp_thread_identities(tmp_path, block):
    # CUDA linearizes x + y*bx + z*bx*by and forms warps from the linear id.
    linear = np.arange(int(np.prod(block)))
    z, y, x = np.unravel_index(linear, block[::-1])
    bx, by, bz = (np.full_like(linear, extent) for extent in block)
    warp, lane = linear // 32, linear % 32
    expected = np.stack([x, y, z, bx, by, bz, linear, warp, lane, x, y, z, bx, by, bz, lane, warp])
    module = numsim.transpile(_identity_kernel(block), cache_dir=tmp_path)
    output = np.full(expected.T.shape, -1, dtype=np.int32)
    result = numsim.Engine().run(module, {"output": output}, outputs=("output",))
    np.testing.assert_array_equal(result.outputs["output"], expected.T)


@T.function
def too_many_threads(output: T.Tensor((1,), "int32")):
    # `LaunchConfig` rejects this shape itself; direct launch threads reach NumSim.
    y = T.launch_thread("threadIdx.y", 33)
    x = T.launch_thread("threadIdx.x", 32)
    output[0] = x + y


@T.function
def preferred_cluster_differs(output: T.Tensor((4,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=4, block=32, cluster=1, preferred_cluster=2))
    if T.cuda.lane_id() == 0:
        output[T.cuda.block_idx("x")] = T.cuda.cluster_dim("x")


@pytest.mark.parametrize(
    "kernel, message",
    [
        (too_many_threads, r"CTA has 1056 threads \(block 32x33x1\); CUDA allows at most 1024"),
        (preferred_cluster_differs, "preferred_cluster must equal cluster"),
    ],
)
def test_launches_numsim_cannot_execute_exactly_are_rejected(tmp_path, kernel, message):
    with pytest.raises(UnsupportedTIRxError, match=message):
        numsim.transpile(kernel, cache_dir=tmp_path)
