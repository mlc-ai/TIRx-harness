"""CUDA tile operations run as the instructions TIRx's TileDispatch selects."""

import numpy as np

from tirx_harness import numsim
from tvm_tirx.backend.cuda.axis import Axis
from tvm_tirx.backend.cuda.layout import tmem_mma_operand_layout
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import ComposeLayout, S, TileLayout

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


# K-major float16 operands in 32-byte swizzle atoms, one 16-element row per atom.
_M, _N, _K, _K_BLOCKS = 128, 64, 16, 3
_A_SHARED = ComposeLayout(
    3, 1, 3, TileLayout(S[(1, _K_BLOCKS, _M, 1, _K) : (0, _M * _K, _K, _M * _K, 1)])
)
_B_SHARED = ComposeLayout(
    3, 1, 3, TileLayout(S[(1, _K_BLOCKS, _N, 1, _K) : (0, _N * _K, _K, _N * _K, 1)])
)
_D_TMEM = tmem_mma_operand_layout("D", (_M, _N), "float32", M=_M, cta_group=1, ws=False)


@T.function
def tcgen05_matmul(
    A: T.Tensor((_M, _K_BLOCKS * _K), "float16"),
    B: T.Tensor((_N, _K_BLOCKS * _K), "float16"),
    output: T.Tensor((128, _N), "uint32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
    tid = T.cuda.thread_idx("x")
    warp = T.cuda.warp_id()
    a = T.alloc_tensor(
        ty_args=[
            T.Tensor(
                (1, _K_BLOCKS, _M, _K), "float16", scope="shared", layout=_A_SHARED, align=1024
            )
        ]
    )
    b = T.alloc_tensor(
        ty_args=[
            T.Tensor(
                (1, _K_BLOCKS, _N, _K), "float16", scope="shared", layout=_B_SHARED, align=1024
            )
        ]
    )
    address = T.alloc_shared((1,), "uint32", layout="default")
    barrier = T.alloc_shared((1,), "uint64", layout="default")
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
            T.address_of(address), T.uint32(_N)
        )
    if tid == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), T.uint32(1))
        for ki, i, j in T.grid(_K_BLOCKS, _M, _K):
            a[0, ki, i, j] = A[i, ki * _K + j]
        for ki, i, j in T.grid(_K_BLOCKS, _N, _K):
            b[0, ki, i, j] = B[i, ki * _K + j]
    T.cuda.cta_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    d = T.cuda.decl_tmem(
        address[0], ty_args=[T.Tensor((1, 1, _M, _N), "float32", scope="tmem", layout=_D_TMEM)]
    )
    if tid == 0:
        T.cuda.tile.apply(
            T.ptx["tcgen05.mma.cta_group::1.kind::f16"],
            d,
            a,
            b,
            instruction_shape=(_M, _N, _K),
            accum=False,
            a_major="k",
            b_major="k",
            a_format="float16",
            b_format="float16",
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            barrier.ptr_to([0])
        )
    T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    word = T.alloc_local((1,), "uint32", layout="default")
    for j in range(_N):
        T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](word[0], address[0] + T.uint32(j))
        T.ptx.tcgen05.wait__ld.sync.aligned()
        output[tid, j] = word[0]
    T.cuda.cta_sync()
    if warp == 0:
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], T.uint32(_N))


def test_tcgen05_tile_operations_address_their_declared_tmem(tmp_path):
    rng = np.random.default_rng(4)
    A = (rng.integers(-4, 5, (_M, _K_BLOCKS * _K)) / 4).astype("float16")
    B = (rng.integers(-4, 5, (_N, _K_BLOCKS * _K)) / 4).astype("float16")
    module = numsim.transpile(tcgen05_matmul, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"A": A, "B": B, "output": np.zeros((128, _N), "uint32")},
        outputs=("output",),
    )
    np.testing.assert_array_equal(
        result.outputs["output"].view("float32"), A.astype("float32") @ B.astype("float32").T
    )
