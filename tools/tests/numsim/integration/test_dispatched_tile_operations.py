"""CUDA tile operations run as the instructions TIRx's TileDispatch selects."""

import numpy as np

from tirx_harness import numsim
from tvm_tirx.backend.cuda.axis import Axis
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import S, TileLayout

_REGISTERS = TileLayout(S[(4, 32, 3, 4) : (1 @ Axis.wid_in_wg, 1 @ Axis.laneid, 4, 1)])


@T.function(private=True)
def plus_one(value: T.float32) -> T.float32:
    return value + T.float32(1)


@T.function
def tile_add_one(
    source: T.Tensor((4, 32, 3, 4), "float32", align=16),
    output: T.Tensor((4, 32, 3, 4), "float32", align=16),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
    registers = T.alloc_local((4, 32, 3, 4), "float32", layout=_REGISTERS)
    T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], registers, source, scope="warpgroup")
    T.cuda.tile.ewise(plus_one, registers, registers, scope="warpgroup")
    T.cuda.tile.apply(T.ptx["st.global.v4.b32"], output, registers, scope="warpgroup")


def test_register_tile_operations_run_as_their_dispatched_instructions(tmp_path):
    source = np.random.default_rng(0).random((4, 32, 3, 4), dtype=np.float32)
    module = numsim.transpile(tile_add_one, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source": source, "output": np.zeros_like(source)},
        outputs=("output",),
    )
    np.testing.assert_array_equal(result.outputs["output"], source + np.float32(1))


@T.function
def tma_add_one(
    source: T.Tensor((8, 32), "float32", align=16),
    output: T.Tensor((8, 32), "float32", align=16),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((8, 32), "float32", layout="default", align=128)
    mbar = T.alloc_shared((1,), "uint64", align=8)
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(1024))
    T.ptx.fence.proxy.async_.shared__cta()
    T.cuda.cta_sync()
    if lane == 0:
        T.cuda.tile.apply(
            T.ptx["cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"],
            shared,
            source,
            box=(8, 32),
            coords=(0, 0),
            mbar=mbar.ptr_to([0]),
        )
    T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
    for row in range(8):
        shared[row, lane] = shared[row, lane] + 1
    T.ptx.fence.proxy.async_.shared__cta()
    T.cuda.cta_sync()
    if lane == 0:
        T.cuda.tile.apply(
            T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"],
            output,
            shared,
            box=(8, 32),
            coords=(0, 0),
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)
    T.cuda.cta_sync()


def test_tma_tile_operations_use_their_host_encoded_tensor_maps(tmp_path):
    source = np.arange(256, dtype=np.float32).reshape(8, 32)
    module = numsim.transpile(tma_add_one, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source": source, "output": np.zeros_like(source)},
        outputs=("output",),
    )
    np.testing.assert_array_equal(result.outputs["output"], source + 1)
