"""TIRx ports of the single-node NVLS multimem kernels in CUTLASS and FlashInfer.

``two_shot_all_reduce`` is CUTLASS's
``examples/python/CuTeDSL/cute/blackwell/kernel/distributed/all_reduce_two_shot_multimem.py``:
each CTA reduces one 128x128 f32 tile of its rank's chunk with
``multimem.ld_reduce`` and broadcasts it with ``multimem.st``, then joins the
other ranks through an SM-wise ``multimem.red.release.sys`` flag barrier.

``gemm_all_reduce_two_shot`` is the GEMM + two-shot all-reduce of FlashInfer's
``flashinfer/cute_dsl/gemm_allreduce_two_shot.py`` and CUTLASS's
``distributed_gemm_all_reduce_blackwell.py`` (LDMCxSTMC), with their warp
roles. Warp 5 TMA-loads 128x64 bfloat16 blocks of A and B into a two-stage
128B-swizzled shared ring; warp 4 issues ``tcgen05.mma.kind::f16`` into one of
two TMEM accumulators and commits the stage and the accumulator. Warps 0-3 are
the epilogue: they ``tcgen05.ld`` the accumulator, convert it to C's dtype in
shared memory and TMA-store the tile to the rank's unicast C, then one thread
arrives on the tile's multicast flag. Warps 6-9 are the all-reduce warps: one
thread spins on the unicast flag until every rank arrived, and the warpgroup
``ld_reduce``/``st``-s the rank's ``128 / world`` rows of the tile through the
multicast C, ``.acc::f32`` for half-precision C. FlashInfer's fp8 C has no
port: ptxas rejects ``multimem.ld_reduce`` on ``e4m3``/``e5m2`` for sm_100a,
sm_100f and sm_103a. Scalar switches choose the signalling, and
``PROTOCOLS`` names the settings each source writes:

* ``flashinfer``: ``fence.acq_rel.gpu; multimem.red.relaxed.gpu;
  fence.proxy.alias`` to arrive, a relaxed ``.gpu`` CAS spin to wait, and an
  alias fence after the final SM-wise release.
* ``cutlass``: ``multimem.red.release.gpu`` to arrive, an acquire ``.gpu`` CAS
  spin to wait, and no alias fence anywhere.
* ``sys_fenced``: ``fence.proxy.async.global; fence.proxy.alias;
  multimem.red.release.sys`` to arrive, an acquire ``.sys`` CAS spin followed
  by ``fence.proxy.alias`` to wait; the ``sys_fenced_*`` entries drop one
  ingredient each, three of them in the mainloop: the staging
  ``fence.proxy.async.shared::cta`` before the TMA store, the MMA warp's wait
  for the TMA loads to land, and a full ``cp.async.bulk.wait_group`` (rather
  than ``.read``, which only waits for the store to read shared memory)
  before the arrival.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tirx_harness.numsim import MulticastWindow

TILE = 128
# (tiles, ctas, k) of the GEMM + all-reduce the tests run: two laps per CTA,
# two K blocks per tile, so both pipeline stages and both TMEM buffers turn over.
GEMM_SHAPE = (4, 2, 128)
BLOCK_K = 64
STAGES = 2
GEMM_WARPS = 10
C_DTYPES = {"float32": np.float32, "bfloat16": np.uint16, "float16": np.uint16}

_SYS_FENCED = {
    "sys_scope": 1, "release_arrive": 1, "acquire_wait": 1, "async_fence": 1,
    "producer_alias": 1, "consumer_alias": 1, "final_alias": 1,
}
_MAINLOOP = {"staging_fence": 1, "load_wait": 1, "store_wait": 1}
PROTOCOLS = {name: {**_MAINLOOP, **switches} for name, switches in {
    "flashinfer": {
        "sys_scope": 0, "release_arrive": 0, "acquire_wait": 0, "async_fence": 0,
        "producer_alias": 2, "consumer_alias": 0, "final_alias": 1,
    },
    "cutlass": {
        "sys_scope": 0, "release_arrive": 1, "acquire_wait": 1, "async_fence": 0,
        "producer_alias": 0, "consumer_alias": 0, "final_alias": 0,
    },
    "sys_fenced": _SYS_FENCED,
    "sys_fenced_relaxed_arrive": {**_SYS_FENCED, "release_arrive": 0},
    "sys_fenced_producer_alias_only": {**_SYS_FENCED, "consumer_alias": 0},
    "sys_fenced_consumer_alias_only": {**_SYS_FENCED, "producer_alias": 0},
    "sys_fenced_alias_after_arrive": {**_SYS_FENCED, "producer_alias": 2, "consumer_alias": 0},
    "sys_fenced_no_alias": {**_SYS_FENCED, "producer_alias": 0, "consumer_alias": 0},
    "sys_fenced_no_async_fence": {**_SYS_FENCED, "async_fence": 0},
    "sys_fenced_relaxed_wait": {**_SYS_FENCED, "acquire_wait": 0},
    "sys_fenced_gpu_scope": {**_SYS_FENCED, "sys_scope": 0},
    "sys_fenced_no_staging_fence": {**_SYS_FENCED, "staging_fence": 0},
    "sys_fenced_unwaited_load": {**_SYS_FENCED, "load_wait": 0},
    "sys_fenced_read_only_store_wait": {**_SYS_FENCED, "store_wait": 0},
}.items()}


def two_shot_all_reduce(tiles: int, world: int, n_tiles: int = 1):
    """CUTLASS's kernel on a ``(TILE * tiles, TILE * n_tiles)`` f32 tensor:
    ``tiles * n_tiles // world`` CTAs of 128 threads. Tiles are numbered
    column-major over the tile grid, as the source's ``zipped_divide`` does, and
    rank ``r`` owns tiles ``r * ctas .. (r + 1) * ctas - 1``. Thread
    ``(m, n) = divmod(tid, 32)`` owns rows ``m + 4 * i`` and columns
    ``4 * n .. 4 * n + 3`` of the tile, the ``(4, 32) x (32, 4)`` raked TV
    layout of the source."""

    assert tiles * n_tiles % world == 0
    ctas = tiles * n_tiles // world
    rows, cols = TILE * tiles, TILE * n_tiles

    @T.prim_func
    def all_reduce_multimem(
        rank: T.int32,
        data_in: T.Buffer((rows, cols), "float32"),
        data_out: T.Buffer((rows, cols), "float32"),
        in_mc: T.Buffer((rows, cols), "float32"),
        out_mc: T.Buffer((rows, cols), "float32"),
        flag: T.Buffer((ctas,), "int32"),
        flag_mc: T.Buffer((ctas,), "int32"),
    ):
        T.device_entry()
        cta = T.cta_id([ctas])
        warp = T.warp_id([4])
        lane = T.lane_id([32])
        v = T.alloc_local((4,), "float32")
        seen = T.alloc_local((1,), "int32")
        tile = rank * ctas + cta
        col = tile // tiles * TILE + lane * 4
        for i in T.serial(32):
            row = tile % tiles * TILE + warp + 4 * i
            T.ptx.multimem_ld_reduce.weak.global_.add.v4.f32(
                v[0], v[1], v[2], v[3], in_mc.ptr_to([row, col]))
            T.ptx.multimem_st.weak.global_.v4.f32(
                out_mc.ptr_to([row, col]), v[0], v[1], v[2], v[3])
        T.cuda.cta_sync()
        if warp == 0 and lane == 0:
            T.ptx.multimem_red.release.sys.global_.add.s32(flag_mc.ptr_to([cta]), T.int32(1))
            seen[0] = 0
            while seen[0] != world:
                T.ptx.atom.relaxed.sys.global_.cas.b32(
                    seen[0], flag.ptr_to([cta]), T.int32(world), T.int32(0))

    return all_reduce_multimem


def two_shot_inputs(tiles: int, world: int, seed: int = 0,
                    ranks: int | None = None, n_tiles: int = 1) -> list[dict]:
    """Bindings for the kernel built for `world`, launched on `ranks` ranks
    (default `world`)."""

    ranks = world if ranks is None else ranks
    shape = (TILE * tiles, TILE * n_tiles)
    rng = np.random.default_rng(seed)
    sources = [rng.standard_normal(shape).astype(np.float32) for _ in range(ranks)]
    outs = [np.zeros(shape, np.float32) for _ in range(ranks)]
    flags = [np.zeros(tiles * n_tiles // world, np.int32) for _ in range(ranks)]
    in_mc, out_mc, flag_mc = (MulticastWindow(x) for x in (sources, outs, flags))
    return [
        {"rank": np.int32(rank), "data_in": sources[rank], "data_out": outs[rank],
         "in_mc": in_mc, "out_mc": out_mc, "flag": flags[rank], "flag_mc": flag_mc}
        for rank in range(ranks)
    ]


def gemm_all_reduce_two_shot(tiles: int, ctas: int, k: int, world: int,
                             dtype: str = "float32"):
    """C = A @ B^T with a two-shot all-reduce of C over ``(TILE * tiles, TILE)``,
    a persistent grid of ``ctas`` CTAs walking tiles ``cta, cta + ctas, ...``.

    A (``rows x k``) and B (``TILE x k``) are K-major bfloat16 behind TensorMaps;
    C is ``dtype`` (float32, bfloat16 or float16; halves travel as uint16).
    Warps 0-3 are the epilogue, warp 4 issues tcgen05 MMAs, warp 5 the TMA
    loads, and warps 6-9 the all-reduce, as in FlashInfer's kernel.
    """

    assert tiles % ctas == 0 and k % BLOCK_K == 0 and dtype in C_DTYPES
    rows, laps, share, kblocks = TILE * tiles, tiles // ctas, TILE // world, k // BLOCK_K
    itemsize = np.dtype(C_DTYPES[dtype]).itemsize
    vec = 16 // itemsize
    thr_n = TILE // vec
    thr_m = 128 // thr_n
    assert share % thr_m == 0
    half = dtype != "float32"
    is_bf16 = dtype == "bfloat16"
    carrier = "uint16" if half else "float32"
    operand_bytes = TILE * BLOCK_K * 2
    stage_bytes = 2 * operand_bytes
    staging_offset = STAGES * stage_bytes
    arena_bytes = staging_offset + TILE * TILE * itemsize

    @T.prim_func
    def gemm_all_reduce(
        sys_scope: T.int32,
        release_arrive: T.int32,
        acquire_wait: T.int32,
        async_fence: T.int32,
        producer_alias: T.int32,
        consumer_alias: T.int32,
        final_alias: T.int32,
        staging_fence: T.int32,
        load_wait: T.int32,
        store_wait: T.int32,
        rank: T.int32,
        a_map: T.TensorMap(),
        b_map: T.TensorMap(),
        c_map: T.TensorMap(),
        c: T.Buffer((rows, TILE), carrier),
        c_mc: T.Buffer((rows, TILE), carrier),
        flag: T.Buffer((tiles + ctas,), "int32"),
        flag_mc: T.Buffer((tiles + ctas,), "int32"),
    ):
        T.device_entry()
        cta = T.cta_id([ctas])
        warp = T.warp_id([GEMM_WARPS])
        lane = T.lane_id([32])
        arena = T.alloc_buffer((arena_bytes,), "uint8", scope="shared.dyn", align=1024)
        staging = T.decl_buffer((TILE, TILE), dtype, data=arena.data,
                                elem_offset=staging_offset // itemsize, scope="shared.dyn")
        # full[0:2], empty[2:4], acc_full[4:6], acc_empty[6:8]
        bars = T.alloc_buffer((8,), "uint64", scope="shared")
        tmem = T.alloc_buffer((1,), "uint32", scope="shared")
        regs = T.alloc_local((16,), "uint32")
        v = T.alloc_local((4,), "uint32")
        vf = T.alloc_local((4,), "float32")
        seen = T.alloc_local((1,), "int32")
        idesc: T.uint32
        da: T.uint64
        db: T.uint64

        if warp == 4:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem[0]), 256)
        if warp == 5 and lane == 0:
            for i in T.unroll(2):
                T.ptx.mbarrier.init.shared.b64(T.address_of(bars[i]), 1)
                T.ptx.mbarrier.init.shared.b64(T.address_of(bars[2 + i]), 1)
                T.ptx.mbarrier.init.shared.b64(T.address_of(bars[4 + i]), 1)
                T.ptx.mbarrier.init.shared.b64(T.address_of(bars[6 + i]), 4)
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx.tcgen05.fence__before_thread_sync()
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence__after_thread_sync()

        if warp == 5:
            if lane == 0:
                for lap in T.serial(laps):
                    tile = cta + lap * ctas
                    for kb in T.serial(kblocks):
                        step = lap * kblocks + kb
                        s = step % STAGES
                        T.cuda.mbarrier_wait(T.address_of(bars[2 + s]), (step // STAGES) % 2 ^ 1)
                        T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                            T.address_of(bars[s]), stage_bytes)
                        T.evaluate(T.ptx[
                            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
                        ](T.address_of(arena[s * stage_bytes]), T.address_of(a_map),
                          kb * BLOCK_K, tile * TILE, T.address_of(bars[s])))
                        T.evaluate(T.ptx[
                            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
                        ](T.address_of(arena[s * stage_bytes + operand_bytes]), T.address_of(b_map),
                          kb * BLOCK_K, 0, T.address_of(bars[s])))
        elif warp == 4:
            if lane == 0:
                T.cuda.tcgen05.encode_instr_descriptor(
                    T.address_of(idesc), d_dtype="float32", a_dtype="bfloat16",
                    b_dtype="bfloat16", M=TILE, N=TILE, K=16, trans_a=False, trans_b=False,
                    n_cta_groups=1,
                )
                for lap in T.serial(laps):
                    acc = lap % 2
                    T.cuda.mbarrier_wait(T.address_of(bars[6 + acc]), (lap // 2) % 2 ^ 1)
                    T.ptx.tcgen05.fence__after_thread_sync()
                    for kb in T.serial(kblocks):
                        step = lap * kblocks + kb
                        s = step % STAGES
                        if load_wait == 1:
                            T.cuda.mbarrier_wait(T.address_of(bars[s]), (step // STAGES) % 2)
                        T.ptx.tcgen05.fence__after_thread_sync()
                        T.cuda.tcgen05.encode_matrix_descriptor(
                            T.address_of(da), T.address_of(arena[s * stage_bytes]),
                            ldo=1, sdo=64, swizzle=3)
                        T.cuda.tcgen05.encode_matrix_descriptor(
                            T.address_of(db), T.address_of(arena[s * stage_bytes + operand_bytes]),
                            ldo=1, sdo=64, swizzle=3)
                        for kk in T.unroll(BLOCK_K // 16):
                            T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
                                tmem[0] + T.uint32(acc * TILE),
                                da + T.uint64(2 * kk), db + T.uint64(2 * kk), idesc,
                                0, 0, 0, 0,
                                T.ptx.pred(T.Select(kb + kk > 0, T.uint32(1), T.uint32(0))),
                            )
                        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                            T.address_of(bars[2 + s]))
                    T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                        T.address_of(bars[4 + acc]))
        elif warp < 4:
            for lap in T.serial(laps):
                tile = cta + lap * ctas
                acc = lap % 2
                T.cuda.mbarrier_wait(T.address_of(bars[4 + acc]), (lap // 2) % 2)
                T.ptx.tcgen05.fence__after_thread_sync()
                row = warp * 32 + lane
                for chunk in T.serial(TILE // 16):
                    T.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                        regs[0], regs[1], regs[2], regs[3], regs[4], regs[5], regs[6], regs[7],
                        regs[8], regs[9], regs[10], regs[11], regs[12], regs[13], regs[14],
                        regs[15],
                        tmem[0] + T.shift_left(T.uint32(warp * 32), T.uint32(16))
                        + T.uint32(acc * TILE + chunk * 16),
                    )
                    T.ptx.tcgen05.wait__ld.sync.aligned()
                    for j in T.unroll(16):
                        staging[row, chunk * 16 + j] = T.cast(
                            T.reinterpret("float32", regs[j]), dtype)
                T.ptx.tcgen05.fence__before_thread_sync()
                if lane == 0:
                    T.ptx.mbarrier.arrive.shared.b64(T.address_of(bars[6 + acc]))
                if staging_fence == 1:
                    T.ptx.fence.proxy.async_.shared__cta()
                T.ptx.bar.sync(T.uint32(1), T.uint32(128))
                if warp == 0 and lane == 0:
                    T.evaluate(T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
                        T.address_of(c_map), 0, tile * TILE, T.address_of(staging[0, 0])))
                    T.ptx.cp.async_.bulk.commit_group()
                    if store_wait == 1:
                        T.ptx.cp.async_.bulk.wait_group(0)
                    else:
                        T.ptx.cp.async_.bulk.wait_group.read(0)
                    if async_fence == 1:
                        T.ptx.fence.proxy.async_.global_()
                    if producer_alias == 1:
                        T.ptx.fence.proxy.alias()
                    if release_arrive == 1:
                        if sys_scope == 1:
                            T.ptx.multimem_red.release.sys.global_.add.s32(
                                flag_mc.ptr_to([tile]), T.int32(1))
                        else:
                            T.ptx.multimem_red.release.gpu.global_.add.s32(
                                flag_mc.ptr_to([tile]), T.int32(1))
                    elif sys_scope == 1:
                        T.ptx.fence.acq_rel.sys()
                        T.ptx.multimem_red.relaxed.sys.global_.add.s32(
                            flag_mc.ptr_to([tile]), T.int32(1))
                    else:
                        T.ptx.fence.acq_rel.gpu()
                        T.ptx.multimem_red.relaxed.gpu.global_.add.s32(
                            flag_mc.ptr_to([tile]), T.int32(1))
                    if producer_alias == 2:
                        T.ptx.fence.proxy.alias()
                T.ptx.bar.sync(T.uint32(1), T.uint32(128))
        else:
            t = (warp - 6) * 32 + lane
            for lap in T.serial(laps):
                tile = cta + lap * ctas
                if warp == 6 and lane == 0:
                    seen[0] = 0
                    while seen[0] != world:
                        if acquire_wait == 1:
                            if sys_scope == 1:
                                T.ptx.atom.acquire.sys.global_.cas.b32(
                                    seen[0], flag.ptr_to([tile]), T.int32(world), T.int32(0))
                            else:
                                T.ptx.atom.acquire.gpu.global_.cas.b32(
                                    seen[0], flag.ptr_to([tile]), T.int32(world), T.int32(0))
                        elif sys_scope == 1:
                            T.ptx.atom.relaxed.sys.global_.cas.b32(
                                seen[0], flag.ptr_to([tile]), T.int32(world), T.int32(0))
                        else:
                            T.ptx.atom.relaxed.gpu.global_.cas.b32(
                                seen[0], flag.ptr_to([tile]), T.int32(world), T.int32(0))
                    if consumer_alias == 1:
                        T.ptx.fence.proxy.alias()
                T.ptx.bar.sync(T.uint32(3), T.uint32(128))
                for i in T.serial(share // thr_m):
                    row = tile * TILE + rank * share + t // thr_n + thr_m * i
                    col = (t % thr_n) * vec
                    if half:
                        if is_bf16:
                            T.ptx.multimem_ld_reduce.weak.global_.add.acc__f32.v4.bf16x2(
                                v[0], v[1], v[2], v[3], c_mc.ptr_to([row, col]))
                        else:
                            T.ptx.multimem_ld_reduce.weak.global_.add.acc__f32.v4.f16x2(
                                v[0], v[1], v[2], v[3], c_mc.ptr_to([row, col]))
                        T.ptx.multimem_st.weak.global_.v4.f32(
                            c_mc.ptr_to([row, col]), T.reinterpret("float32", v[0]),
                            T.reinterpret("float32", v[1]), T.reinterpret("float32", v[2]),
                            T.reinterpret("float32", v[3]))
                    else:
                        T.ptx.multimem_ld_reduce.weak.global_.add.v4.f32(
                            vf[0], vf[1], vf[2], vf[3], c_mc.ptr_to([row, col]))
                        T.ptx.multimem_st.weak.global_.v4.f32(
                            c_mc.ptr_to([row, col]), vf[0], vf[1], vf[2], vf[3])
            T.ptx.bar.sync(T.uint32(3), T.uint32(128))
            if warp == 6 and lane == 0:
                T.ptx.multimem_red.release.sys.global_.add.s32(
                    flag_mc.ptr_to([tiles + cta]), T.int32(1))
                if final_alias == 1:
                    T.ptx.fence.proxy.alias()
                seen[0] = 0
                while seen[0] != world:
                    T.ptx.atom.acquire.sys.global_.cas.b32(
                        seen[0], flag.ptr_to([tiles + cta]), T.int32(world), T.int32(0))
        T.cuda.cta_sync()
        if warp == 4:
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem[0], 256)
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()

    return gemm_all_reduce


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_values(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def c_values(c: np.ndarray, dtype: str) -> np.ndarray:
    """The float32 values of a C replica held in its carrier."""
    if dtype == "bfloat16":
        return bf16_values(c)
    if dtype == "float16":
        return c.view(np.float16).astype(np.float32)
    return c


def gemm_inputs(tiles: int, ctas: int, k: int, world: int, protocol: str,
                dtype: str = "float32", seed: int = 0, integer: bool = False,
                paired: bool = False) -> list[dict]:
    """Per-rank bindings; A, B and C sit behind TensorMaps over the rank's
    operands and its C replica, as NumSim descriptor images or, with
    ``paired``, as PairedTensorMaps the GPU launcher encodes. ``integer`` draws
    small integers, so every product and sum is exact in any reduction order."""

    from tests.numsim.microtests.harness import PairedTensorMap

    rows = TILE * tiles
    rng = np.random.default_rng(seed)
    draw = (lambda shape: rng.integers(-2, 3, shape)) if integer else rng.standard_normal
    a = [_bf16_bits(draw((rows, k))) for _ in range(world)]
    b = [_bf16_bits(draw((TILE, k))) for _ in range(world)]
    c = [np.zeros((rows, TILE), C_DTYPES[dtype]) for _ in range(world)]
    flags = [np.zeros(tiles + ctas, np.int32) for _ in range(world)]
    c_mc, flag_mc = MulticastWindow(c), MulticastWindow(flags)
    modes = {name: np.int32(value) for name, value in PROTOCOLS[protocol].items()}
    itemsize = np.dtype(C_DTYPES[dtype]).itemsize

    def operand(array, outer):
        return PairedTensorMap(array, (k, outer), (k * 2,), (BLOCK_K, TILE), (1, 1),
                               logical_dtype="bfloat16", swizzle="128B")

    bindings = [
        {**modes, "rank": np.int32(rank),
         "a_map": operand(a[rank], rows), "b_map": operand(b[rank], TILE),
         "c_map": PairedTensorMap(c[rank], (TILE, rows), (TILE * itemsize,), (TILE, TILE),
                                  (1, 1), logical_dtype=dtype),
         "c": c[rank], "c_mc": c_mc, "flag": flags[rank], "flag_mc": flag_mc}
        for rank in range(world)
    ]
    return bindings if paired else numsim_bindings(bindings)


def numsim_bindings(bindings: list[dict]) -> list[dict]:
    """NumSim's view of `gemm_inputs`: each PairedTensorMap becomes its
    descriptor image over the same array."""

    from tests.numsim.microtests.harness import PairedTensorMap, _numsim_tensor_map

    return [
        {name: _numsim_tensor_map(value) if isinstance(value, PairedTensorMap) else value
         for name, value in rank.items()}
        for rank in bindings
    ]


def gemm_reference(bindings: list[dict]) -> np.ndarray:
    """The float64 all-reduced product of the operands of paired bindings."""
    return sum(
        bf16_values(r["a_map"].array).astype(np.float64)
        @ bf16_values(r["b_map"].array).astype(np.float64).T
        for r in bindings
    )
