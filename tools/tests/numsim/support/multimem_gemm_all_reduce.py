"""SM100 GEMM fused with a two-shot NVLS multimem all-reduce, at full performance.

TIRx port of the persistent, warp-specialized GEMM + all-reduce kernels in

* CUTLASS ``examples/python/CuTeDSL/cute/blackwell/kernel/distributed/
  distributed_gemm_all_reduce_blackwell.py``
  (``Sm100PersistentDenseGemmAllReduceLDMCxSTMCKernel``), and
* FlashInfer ``flashinfer/cute_dsl/gemm_allreduce_two_shot.py``
  (``PersistentDenseGemmKernel`` with ``all_reduce="two_shot"``).

Both origins compute ``C = A @ B^T`` (A ``m x k``, B ``n x k``, both K-major, C
row-major) with the same warp roles: epilogue warps 0-3, the MMA warp 4, the
TMA warp 5 and all-reduce warps from 6 up. The kernel takes the origins'
configuration space -- 1- or 2-CTA ``tcgen05.mma``, the MMA tile, the cluster
shape (with TMA multicast of A along the cluster's N and of B along its M), the
static persistent scheduler's raster order and swizzle, and the TMA-store or
direct-store epilogue -- and derives every dependent parameter (stage counts,
epilogue subtile, TMEM columns, all-reduce warps) with the origins' formulas, so
one configuration gives the same schedule as the origin built with it.

Per output tile the epilogue stores C to this rank and bumps the tile's flag on
every rank through ``flag_mc``; this rank's all-reduce warps wait for the flag to
reach the world size, ``multimem.ld_reduce`` their ``1/world`` of the tile's
rows from ``c_mc`` and ``multimem.st`` the sum to every rank. ``protocol``
selects the origin's flag protocol and data placement:

* ``cutlass``: ``multimem.red.release.gpu`` after the epilogue's stores
  complete (``cp.async.bulk.wait_group 0`` or a join of the epilogue warps) to
  arrive, an acquire ``.gpu`` CAS spin to wait, the reduced tile written to
  ``out_mc`` (out of place), ``W`` all-reduce warps with ``W`` the largest of
  4..1 giving every thread whole 128-bit accesses, and four ``ld_reduce`` in
  flight per thread.
* ``flashinfer``: ``fence.acq_rel.gpu; multimem.red.relaxed.gpu;
  fence.proxy.alias`` to arrive, a relaxed ``.gpu`` CAS spin to wait, the
  reduced tile written back to ``c_mc`` (in place), four all-reduce warps, and
  each ``ld_reduce`` followed by its ``st``. FlashInfer's direct-store epilogue
  arrives from warp 0 without first synchronizing the other epilogue warps'
  stores; this port joins the four warps before the arrive.

Both end with an SM-wise ``multimem.red.release.sys`` / acquire ``.sys`` CAS
barrier so no rank exits while another still reads its memory.
"""

import functools
import math
from dataclasses import dataclass

import tirx_kernels.tirx_lite as txl

SMEM_CAPACITY = 232_448
AB_BYTES = {"tf32": 4, "f16": 2, "bf16": 2}
C_BYTES = {"f32": 4, "f16": 2, "bf16": 2}
STORAGE = {"tf32": "float32", "f16": "float16", "bf16": "bfloat16", "f32": "float32"}
TRY_WAIT_TICKS = 10_000_000


@dataclass(frozen=True)
class Plan:
    """Static schedule of one configuration; field names follow the origins."""

    m: int
    n: int
    k: int
    ab: str
    c: str
    world: int
    use_2cta: bool
    mma_tiler: tuple
    cluster: tuple
    raster: str
    swizzle: int
    use_tma_store: bool
    protocol: str
    max_active_clusters: int

    @property
    def cta_group(self):
        return 2 if self.use_2cta else 1

    @property
    def cta_m(self):
        return self.mma_tiler[0] // self.cta_group

    @property
    def cta_n(self):
        return self.mma_tiler[1]

    @property
    def instruction_k(self):
        return 8 if self.ab == "tf32" else 16

    @property
    def k_tile(self):
        return 4 * self.instruction_k

    @property
    def k_tiles(self):
        return self.k // self.k_tile

    @property
    def b_rows(self):
        return self.cta_n // self.cta_group

    @property
    def cluster_size(self):
        return self.cluster[0] * self.cluster[1]

    @property
    def cluster_m_groups(self):
        return self.cluster[0] // self.cta_group

    @property
    def m_tiles(self):
        return self.m // self.cta_m

    @property
    def n_tiles(self):
        return self.n // self.cta_n

    @property
    def cluster_tiles(self):
        return self.m_tiles // self.cluster[0], self.n_tiles // self.cluster[1]

    @property
    def cluster_work(self):
        return self.cluster_tiles[0] * self.cluster_tiles[1]

    @property
    def clusters(self):
        """Persistent clusters launched (``StaticPersistentTileScheduler.get_grid_shape``)."""
        return min(self.cluster_work, self.max_active_clusters)

    @property
    def a_stage_bytes(self):
        return self.cta_m * self.k_tile * AB_BYTES[self.ab]

    @property
    def b_stage_bytes(self):
        return self.b_rows * self.k_tile * AB_BYTES[self.ab]

    @property
    def epi_tile(self):
        """``compute_epilogue_tile_shape`` (no source C, row-major D)."""
        if not self.use_tma_store:
            return self.cta_m, self.cta_n
        warp_n = 2 if self.cta_m == 64 and self.use_2cta else 1
        warp_m = 4 // warp_n
        tile_m = min(self.cta_m, 32 * warp_m)
        n_perf = 4096 // tile_m
        while self.cta_n % n_perf:
            n_perf //= 2
        tile_n = min(self.cta_n, max(n_perf, 8 * warp_n, 128 // (8 * C_BYTES[self.c]) * warp_n))
        return tile_m, tile_n if self.cta_n % tile_n == 0 else self.cta_n

    @property
    def c_stage_bytes(self):
        tile_m, tile_n = self.epi_tile
        return tile_m * tile_n * C_BYTES[self.c] if self.use_tma_store else 0

    @property
    def ab_stages(self):
        """``_compute_stages``: A/B stages fill what the mbarriers and two C stages leave."""
        reserved = 1024 + 2 * self.c_stage_bytes
        return (SMEM_CAPACITY - reserved) // (self.a_stage_bytes + self.b_stage_bytes)

    @property
    def c_stages(self):
        if not self.use_tma_store:
            return 0
        ab = (self.a_stage_bytes + self.b_stage_bytes) * self.ab_stages
        return 2 + (SMEM_CAPACITY - ab - 1024 - 2 * self.c_stage_bytes) // self.c_stage_bytes

    @property
    def acc_columns(self):
        """TMEM columns of one accumulator stage (``compute_acc_tmem_cols_per_stage``)."""
        return self.cta_n // 2 if self.use_2cta and self.cta_m == 64 else self.cta_n

    @property
    def tmem_columns(self):
        return max(32, 1 << (2 * self.acc_columns - 1).bit_length())

    @property
    def comm_warps(self):
        """``_pick_num_comm_warp_for_128b`` for CUTLASS; FlashInfer uses four."""
        if self.protocol == "flashinfer":
            return 4
        slab = self.cta_m * self.cta_n // self.world
        atom = 16 // C_BYTES[self.c]
        for warps in (4, 3, 2, 1):
            if slab % (32 * warps) == 0 and (slab // (32 * warps)) % atom == 0:
                return warps
        raise ValueError(f"no all-reduce warp count gives 128-bit accesses for {self}")

    @property
    def shared_bytes(self):
        return 1024 + self.c_stages * self.c_stage_bytes + self.ab_stages * (
            self.a_stage_bytes + self.b_stage_bytes)


def plan(m, n, k, ab, c, world, *, use_2cta, mma_tiler, cluster, use_tma_store, raster="m",
         swizzle=1, protocol="cutlass", max_active_clusters):
    p = Plan(m, n, k, ab, c, world, use_2cta, tuple(mma_tiler), tuple(cluster), raster, swizzle,
             use_tma_store, protocol, max_active_clusters)
    if p.cta_m not in (64, 128) or p.cta_n % 32 or p.cluster[0] % p.cta_group:
        raise ValueError(f"unsupported MMA tile / cluster {p.mma_tiler} {p.cluster}")
    if p.m % (p.cta_m * p.cluster[0]) or p.n % (p.cta_n * p.cluster[1]) or p.k % p.k_tile:
        raise ValueError("the problem must be a whole number of cluster tiles and K tiles")
    if p.cta_m % world:
        raise ValueError("every rank needs whole rows of each tile")
    if swizzle > 1 and p.cluster_tiles[1 if raster == "m" else 0] % swizzle:
        raise ValueError("the swizzle must divide the swizzled cluster count")
    if protocol not in ("cutlass", "flashinfer"):
        raise ValueError(protocol)
    if protocol == "flashinfer" and p.cta_n * C_BYTES[c] // 16 > 32:
        raise ValueError("FlashInfer's all-reduce lays one tile row over at most a warp")
    if p.shared_bytes > SMEM_CAPACITY or p.tmem_columns > 512:
        raise ValueError("configuration exceeds shared or tensor memory")
    return p


def _elected():
    lane = txl.local_scalar("uint32")
    pred = txl.local_scalar("uint32")
    txl.ptx.elect_sync(lane, pred, txl.uint32(0xFFFFFFFF))
    return pred == txl.uint32(1)


def _try_wait(dst, barrier, phase):
    txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
        dst, barrier, txl.cast(phase, "uint32"))


def _wait(barrier, phase):
    ready = txl.local_scalar("uint32", init=txl.uint32(0))
    with txl.While(ready == txl.uint32(0)):
        txl.ptx.mbarrier.try_wait.parity.shared.b64(
            ready, barrier, txl.cast(phase, "uint32"), txl.uint32(TRY_WAIT_TICKS))


def _wait_unless(barrier, phase, ready):
    with txl.If(ready == txl.uint32(0)), txl.Then():
        _wait(barrier, phase)


# K-major SW128 shared-memory matrix descriptor (LBO 16 B, SBO 1024 B) without its
# start address.
_SW128_K_DESCRIPTOR = (1 << 16) | (64 << 32) | (1 << 46) | (2 << 61)


def _descriptor_lo(address):
    """Low word of the ``_SW128_K_DESCRIPTOR`` descriptor of ``address``."""
    return txl.bitwise_or(
        txl.bitwise_and(txl.shift_right(address, txl.uint32(4)), txl.uint32(0x3FFF)),
        txl.uint32(_SW128_K_DESCRIPTOR & 0xFFFFFFFF))


@functools.cache
def gemm_all_reduce(m, n, k, ab, c, rank, world, *, use_2cta, mma_tiler, cluster,
                    use_tma_store, raster="m", swizzle=1, protocol="cutlass",
                    max_active_clusters, flag_len):
    """Build the kernel for one rank; returns a ``txl.Kernel``.

    Parameters, all byte buffers but the flags: ``a`` (m x k), ``b`` (n x k),
    ``c`` (m x n, this rank's GEMM output), ``c_mc`` / ``out_mc`` (multicast
    views of C and of the out-of-place result), ``flag`` / ``flag_mc``
    (``flag_len`` int32 flags, zero between launches).
    """
    p = plan(m, n, k, ab, c, world, use_2cta=use_2cta, mma_tiler=mma_tiler, cluster=cluster,
             use_tma_store=use_tma_store, raster=raster, swizzle=swizzle, protocol=protocol,
             max_active_clusters=max_active_clusters)
    g, cta_m, cta_n = p.cta_group, p.cta_m, p.cta_n
    cm, cn = p.cluster
    ncm, ncn = p.cluster_tiles
    cmg = p.cluster_m_groups
    stages, c_stages = p.ab_stages, p.c_stages
    abb, cb = AB_BYTES[ab], C_BYTES[c]
    c_type = c
    warps = 6 + p.comm_warps
    final_flags = p.m_tiles * p.n_tiles
    if flag_len < final_flags + cm * cn * p.clusters:
        raise ValueError("flag buffer too small")

    ab_empty_arrivals = cn + cmg - 1
    a_piece = cta_m // cn
    b_piece = p.b_rows // cmg
    a_piece_bytes = p.a_stage_bytes // cn
    b_piece_bytes = p.b_stage_bytes // cmg
    tma_bytes = (p.a_stage_bytes + p.b_stage_bytes) * g

    # Epilogue subtiles. Every epilogue thread reads 32x32b TMEM lanes: its row of
    # the accumulator, ``sub_cols`` columns at a time.
    #   cta_m 128 (1-CTA M=128, 2-CTA M=256): row 32w+lane, 32 columns per subtile.
    #   1-CTA M=64: rows sit in lanes 32w+[0,16); lanes 16-31 idle; 64 columns.
    #   2-CTA M=128: warps 0-1 hold columns [0, n/2), warps 2-3 [n/2, n) of rows
    #     32(w%2)+lane; a subtile is 32 columns of each half.
    if cta_m == 128:
        layout, sub_cols = "full", 32
    elif g == 1:
        layout, sub_cols = "half_lanes", 64
    else:
        layout, sub_cols = "split_n", 32
    subtiles = cta_n // 64 if layout == "split_n" else cta_n // sub_cols
    block_cols = min(sub_cols, 128 // cb)
    blocks = 2 if layout == "split_n" else sub_cols // block_cols
    block_rows = cta_m
    row_bytes = block_cols * cb
    block_bytes = block_rows * row_bytes
    swizzle_mode = {128: 3, 64: 2, 32: 1}[row_bytes]
    swizzle_mask = {128: 0x70, 64: 0x30, 32: 0x10}[row_bytes]
    if use_tma_store:
        assert blocks * block_bytes == p.c_stage_bytes, (blocks, block_bytes, p)
    chunk_elems = 16 // cb
    chunks = sub_cols * cb // 16

    c_offset = 1024
    a_offset = c_offset + c_stages * p.c_stage_bytes
    b_offset = a_offset + stages * p.a_stage_bytes

    # All-reduce slab: this rank's m_local rows of the CTA tile.
    m_local = cta_m // world
    atom = 16 // cb
    threads = 32 * p.comm_warps
    if protocol == "cutlass":
        thr_n = math.gcd(cta_n // atom, threads)
    else:
        thr_n = cta_n // atom
    thr_m = threads // thr_n
    loop_m = m_local // thr_m
    loop_n = cta_n // (thr_n * atom)
    if m_local % thr_m or cta_n % (thr_n * atom) or loop_m * loop_n == 0:
        raise ValueError("all-reduce slab does not tile the all-reduce threads")

    def host_prelude(params):
        a_map = txl.stack_alloca("tensormap", 1)
        b_map = txl.stack_alloca("tensormap", 1)
        c_map = txl.stack_alloca("tensormap", 1)

        def encode(descriptor, dtype, data, inner, outer, box_inner, box_outer, swizzle_mode,
                   element_bytes):
            txl.call_packed("runtime.cuTensorMapEncodeTiled", descriptor, dtype, 3, data,
                            inner, outer, 1, inner * element_bytes, inner * outer * element_bytes,
                            box_inner, box_outer, 1, 1, 1, 1, 0, swizzle_mode, 2, 0)

        encode(a_map, STORAGE[ab], params["a"].data, k, m, p.k_tile, a_piece, 3, abb)
        encode(b_map, STORAGE[ab], params["b"].data, k, n, p.k_tile, b_piece, 3, abb)
        if use_tma_store:
            encode(c_map, STORAGE[c], params["c"].data, n, m, block_cols, block_rows,
                   swizzle_mode, cb)
        return a_map, b_map, c_map

    def kernel(a, b, c, c_mc, out_mc, flag, flag_mc, *, host):
        del a, b
        a_map, b_map, c_map = host
        block_x, block_y, first_work = txl.cta_id()
        scope_x, scope_y = txl.cta_id_in_cluster([cm, cn], preferred=[cm, cn])
        del scope_x, scope_y
        cluster_rank = txl.local_scalar("int32", init=txl.cuda.mov_sreg(32, "cluster_ctarank"))
        cluster_x = txl.local_scalar("int32", init=cluster_rank % cm)
        cluster_y = txl.local_scalar("int32", init=cluster_rank // cm)
        cta_v = cluster_x % g
        leader_cta = cta_v == 0
        cluster_m_group = cluster_x // g
        pair_leader_x = cluster_m_group * g
        leader_rank = pair_leader_x + cm * cluster_y
        warp = txl.warp_id()
        lane = txl.lane_id()

        roles = txl.specialize(chain_dispatch=True)
        epilogue_role = roles.role("epilogue", warps=[0, 1, 2, 3])
        mma_role = roles.role("mma", warps=[4])
        tma_role = roles.role("tma", warps=[5])
        comm_role = roles.role("all_reduce", warps=list(range(6, warps)))

        smem = txl.alloc_buffer((p.shared_bytes,), txl.u8, scope="shared.dyn", align=1024)
        pool = txl.smem_pool(base=smem)
        ab_pipe = txl.Pipeline(pool, stages, full="tma", empty="tcgen05",
                               init_empty=ab_empty_arrivals, leader=txl.bool(False))
        acc_pipe = txl.Pipeline(pool, 2, full="tcgen05", empty="mbar", init_empty=4 * g,
                                leader=txl.bool(False))
        tmem_dealloc = pool.alloc((1,), txl.u64, align=8)
        tmem_slot = pool.alloc((1,), txl.u32, align=4)
        if pool.bytes > 1024:
            raise ValueError("mbarriers overflow their reserved kilobyte")

        with tma_role:
            txl.ptx.prefetch.tensormap(txl.address_of(a_map))
            txl.ptx.prefetch.tensormap(txl.address_of(b_map))
            if use_tma_store:
                txl.ptx.prefetch.tensormap(txl.address_of(c_map))

        with txl.If(warp == 0), txl.Then():
            with txl.If(_elected()), txl.Then():
                with txl.unroll(0, stages) as stage:
                    txl.ptx.mbarrier.init.shared.b64(ab_pipe.full.ptr_to([stage]), txl.uint32(1))
                    txl.ptx.mbarrier.init.shared.b64(
                        ab_pipe.empty.ptr_to([stage]), txl.uint32(ab_empty_arrivals))
                with txl.unroll(0, 2) as stage:
                    txl.ptx.mbarrier.init.shared.b64(acc_pipe.full.ptr_to([stage]), txl.uint32(1))
                    txl.ptx.mbarrier.init.shared.b64(
                        acc_pipe.empty.ptr_to([stage]), txl.uint32(4 * g))
                if g == 2:
                    txl.ptx.mbarrier.init.shared.b64(tmem_dealloc.ptr_to([0]), txl.uint32(32))
        txl.ptx.fence.mbarrier_init.release.cluster()
        if p.cluster_size > 1:
            txl.ptx.barrier.cluster.arrive.relaxed()

        smem_base = txl.local_scalar("uint32")
        txl.assign(smem_base, txl.cuda.cvta_generic_to_shared(smem.ptr_to([0])))
        cluster_smem_u64 = txl.local_scalar("uint64")
        txl.ptx.cvta.to.shared__cluster.u64(cluster_smem_u64, smem.ptr_to([0]))
        cluster_smem = txl.local_scalar("uint32", init=txl.cast(cluster_smem_u64, "uint32"))

        a_mask = txl.local_scalar("uint32", init=txl.uint32(0))
        for peer_n in range(cn):
            txl.assign(a_mask, txl.bitwise_or(
                a_mask, txl.uint32(1) << txl.cast(cluster_x + cm * peer_n, "uint32")))
        b_mask = txl.local_scalar("uint32", init=txl.uint32(0))
        for group in range(cmg):
            txl.assign(b_mask, txl.bitwise_or(
                b_mask, txl.uint32(1) << txl.cast(cta_v + g * group + cm * cluster_y, "uint32")))
        consumer_mask = txl.local_scalar("uint32", init=txl.uint32(0))
        for pair_v in range(g):
            for peer_n in range(cn):
                txl.assign(consumer_mask, txl.bitwise_or(
                    consumer_mask,
                    txl.uint32(1) << txl.cast(pair_leader_x + pair_v + cm * peer_n, "uint32")))
            for group in range(cmg):
                txl.assign(consumer_mask, txl.bitwise_or(
                    consumer_mask,
                    txl.uint32(1) << txl.cast(pair_v + g * group + cm * cluster_y, "uint32")))
        acc_mask = txl.local_scalar(
            "uint32", init=txl.uint32((1 << g) - 1) << txl.cast(leader_rank, "uint32"))
        ab_full_leader = ab_pipe.full.remote_view(leader_rank)
        acc_empty_leader = acc_pipe.empty.remote_view(leader_rank)

        if p.cluster_size > 1:
            txl.ptx.barrier.cluster.wait()
        else:
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(32 * warps))

        def tile_of(work):
            """CTA tile (m, n) of a cluster work index (``StaticPersistentTileScheduler``)."""
            s = swizzle
            if s == 1 and raster == "m":
                cl_m, cl_n = work % ncm, work // ncm
            elif s == 1:
                cl_m, cl_n = work // ncn, work % ncn
            elif raster == "m":
                cl_m = (work // s) % ncm
                cl_n = work % s + s * (work // (s * ncm))
            else:
                cl_m = work % s + s * (work // (s * ncn))
                cl_n = (work // s) % ncn
            return cl_m * cm + cluster_x, cl_n * cn + cluster_y

        def flag_index(work, tile_m, tile_n):
            if protocol == "cutlass":
                return tile_m + tile_n * p.m_tiles
            return work * p.cluster_size + cluster_rank

        def next_work(work):
            txl.assign(work, work + p.clusters)

        with tma_role:
            state = txl.PipelineState(stages, phase=1)
            work = txl.local_scalar("int32", init=first_work)
            count = txl.local_scalar("int32")
            ready = txl.local_scalar("uint32")
            # As in the MMA warp, one elected thread runs the whole load stream.
            with txl.If(_elected()), txl.Then():
                with txl.While(work < p.cluster_work):
                    tile_m, tile_n = tile_of(work)
                    txl.assign(count, 0)
                    txl.assign(ready, txl.uint32(1))
                    _try_wait(ready, ab_pipe.empty.ptr_to([state.stage]), state.phase)
                    with txl.While(count < p.k_tiles):
                        _wait_unless(ab_pipe.empty.ptr_to([state.stage]), state.phase, ready)
                        with txl.If(leader_cta), txl.Then():
                            txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                ab_pipe.full.ptr_to([state.stage]), txl.uint32(tma_bytes))
                        loads = (
                            (a_map, a_offset + state.stage * p.a_stage_bytes
                             + cluster_y * a_piece_bytes,
                             tile_m * cta_m + cluster_y * a_piece, a_mask, cn),
                            (b_map, b_offset + state.stage * p.b_stage_bytes
                             + cluster_m_group * b_piece_bytes,
                             tile_n * cta_n + cta_v * p.b_rows + cluster_m_group * b_piece,
                             b_mask, cmg),
                        )
                        for tensor_map, offset, row, mask, peers in loads:
                            coords = (txl.cast(count * p.k_tile, "int32"), txl.cast(row, "int32"),
                                      txl.int32(0))
                            if g == 1 and peers == 1:
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                                    ".mbarrier::complete_tx::bytes"
                                ](smem.ptr_to([offset]), txl.address_of(tensor_map), *coords,
                                  ab_pipe.full.ptr_to([state.stage]))
                            elif g == 1:
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                ](cluster_smem + offset, txl.address_of(tensor_map), *coords,
                                  ab_pipe.full.ptr_to([state.stage]), txl.cast(mask, "uint16"))
                            elif peers == 1:
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.cta_group::2"
                                ](cluster_smem + offset, txl.address_of(tensor_map), *coords,
                                  ab_full_leader.ptr_to([state.stage]))
                            else:
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".cta_group::2"
                                ](cluster_smem + offset, txl.address_of(tensor_map), *coords,
                                  ab_full_leader.ptr_to([state.stage]),
                                  txl.cast(mask, "uint16"))
                        state.advance()
                        txl.assign(count, count + 1)
                        txl.assign(ready, txl.uint32(1))
                        with txl.If(count < p.k_tiles), txl.Then():
                            _try_wait(ready, ab_pipe.empty.ptr_to([state.stage]), state.phase)
                    next_work(work)
                with txl.unroll(0, stages) as _:
                    _wait(ab_pipe.empty.ptr_to([state.stage]), state.phase)
                    state.advance()

        with mma_role:
            txl.ptx.bar.sync(txl.uint32(2), txl.uint32(160))
            tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))
            idesc = txl.alloc_local((1,), "uint32")
            txl.cuda.tcgen05.encode_instr_descriptor(
                txl.address_of(idesc[0]), d_dtype="float32",
                a_dtype="tf32" if ab == "tf32" else STORAGE[ab],
                b_dtype="tf32" if ab == "tf32" else STORAGE[ab],
                M=p.mma_tiler[0], N=cta_n, K=p.instruction_k, trans_a=False, trans_b=False,
                n_cta_groups=g)
            # Descriptors move only in their low word (start address >> 4, which stays
            # below 2^14), so they are rebuilt from 32-bit arithmetic as CUTLASS does.
            a_lo = txl.local_scalar("uint32", init=_descriptor_lo(smem_base + a_offset))
            b_lo = txl.local_scalar("uint32", init=_descriptor_lo(smem_base + b_offset))
            k_step = p.instruction_k * abb // 16

            def descriptor(lo_base, stage_bytes, stage, kphase):
                lo = lo_base + txl.cast(stage, "uint32") * txl.uint32(stage_bytes // 16) \
                    + txl.uint32(kphase * k_step)
                return txl.bitwise_or(txl.uint64(_SW128_K_DESCRIPTOR >> 32 << 32),
                                      txl.cast(lo, "uint64"))

            full_bars = txl.local_scalar(
                "uint32", init=txl.cuda.cvta_generic_to_shared(ab_pipe.full.ptr_to([0])))
            empty_bars = txl.local_scalar("uint32", init=full_bars + (
                txl.cuda.cvta_generic_to_shared(ab_pipe.empty.ptr_to([0]))
                - txl.cuda.cvta_generic_to_shared(ab_pipe.full.ptr_to([0]))))

            def full_bar(stage):
                return full_bars + txl.cast(stage, "uint32") * txl.uint32(8)

            mma_state = txl.PipelineState(stages, phase=0)
            acc_state = txl.PipelineState(2, phase=1)
            work = txl.local_scalar("int32", init=first_work)
            count = txl.local_scalar("int32")
            ready = txl.local_scalar("uint32")
            accumulate = txl.local_scalar("uint32")
            # One elected thread of the leader CTA issues the whole MMA stream, so the
            # k-loop carries no per-iteration elect and warp reconvergence.
            with txl.If(leader_cta), txl.Then(), txl.If(_elected()), txl.Then():
                with txl.While(work < p.cluster_work):
                    txl.assign(count, 0)
                    txl.assign(ready, txl.uint32(1))
                    _try_wait(ready, full_bar(mma_state.stage), mma_state.phase)
                    _wait(acc_pipe.empty.ptr_to([acc_state.stage]), acc_state.phase)
                    txl.assign(accumulate, txl.uint32(0))
                    with txl.While(count < p.k_tiles):
                        _wait_unless(full_bar(mma_state.stage), mma_state.phase, ready)
                        for kphase in range(p.k_tile // p.instruction_k):
                            operands = (
                                txl.cast(tmem_base + acc_state.stage * p.acc_columns, "uint32"),
                                descriptor(a_lo, p.a_stage_bytes, mma_state.stage, kphase),
                                descriptor(b_lo, p.b_stage_bytes, mma_state.stage, kphase),
                                idesc[0],
                            )
                            kind = "tf32" if ab == "tf32" else "f16"
                            txl.ptx[f"tcgen05.mma.cta_group::{g}.kind::{kind}"](
                                *operands, *[txl.uint32(0) for _ in range(4 * g)],
                                txl.ptx.pred(txl.cast(
                                    accumulate if kphase == 0 else txl.uint32(1), "bool")))
                        empty_bar = empty_bars + txl.cast(mma_state.stage, "uint32") * txl.uint32(8)
                        if p.cluster_size > 1:
                            txl.ptx[
                                f"tcgen05.commit.cta_group::{g}.mbarrier::arrive::one"
                                ".shared::cluster.multicast::cluster.b64"
                            ](empty_bar, txl.cast(consumer_mask, "uint16"))
                        else:
                            txl.ptx[
                                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                                ".shared::cluster.b64"
                            ](empty_bar)
                        txl.assign(accumulate, txl.uint32(1))
                        mma_state.advance()
                        txl.assign(count, count + 1)
                        txl.assign(ready, txl.uint32(1))
                        with txl.If(count < p.k_tiles), txl.Then():
                            _try_wait(ready, full_bar(mma_state.stage), mma_state.phase)
                    if g == 2:
                        txl.ptx[
                            "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                            ".shared::cluster.multicast::cluster.b64"
                        ](acc_pipe.full.ptr_to([acc_state.stage]), txl.cast(acc_mask, "uint16"))
                    else:
                        txl.ptx[
                            "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                            ".shared::cluster.b64"
                        ](acc_pipe.full.ptr_to([acc_state.stage]))
                    acc_state.advance()
                    next_work(work)
                acc_state.advance()
                _wait(acc_pipe.empty.ptr_to([acc_state.stage]), acc_state.phase)

        with epilogue_role:
            with txl.If(warp == 0), txl.Then():
                txl.ptx[f"tcgen05.alloc.cta_group::{g}.sync.aligned.shared::cta.b32"](
                    tmem_slot.ptr_to([0]), txl.uint32(p.tmem_columns))
            txl.ptx.bar.sync(txl.uint32(2), txl.uint32(160))
            tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))
            acc_state = txl.PipelineState(2, phase=0)
            work = txl.local_scalar("int32", init=first_work)
            c_buffer = txl.local_scalar("int32", init=0)
            values = txl.alloc_local((sub_cols,), "float32")
            words = txl.alloc_local((sub_cols * cb // 4,), "uint32")

            if layout == "full":
                row = warp * 32 + lane
                column_half = 0
                active = None
            elif layout == "half_lanes":
                row = warp * 16 + lane
                column_half = 0
                active = lane < 16
            else:
                row = (warp % 2) * 32 + lane
                column_half = warp // 2
                active = None

            def stores_when_active(body):
                if active is None:
                    body()
                else:
                    with txl.If(active), txl.Then():
                        body()

            with txl.While(work < p.cluster_work):
                tile_m, tile_n = tile_of(work)
                _wait(acc_pipe.full.ptr_to([acc_state.stage]), acc_state.phase)
                with txl.unroll(0, subtiles) as subtile:
                    taddr = (tmem_base + txl.shift_left(txl.cast(warp * 32, "uint32"),
                                                        txl.uint32(16))
                             + txl.cast(acc_state.stage * p.acc_columns + subtile * sub_cols,
                                        "uint32"))
                    txl.ptx[f"tcgen05.ld.sync.aligned.32x32b.x{sub_cols}.b32"](
                        *[values[i] for i in range(sub_cols)], taddr)
                    txl.ptx["tcgen05.wait::ld.sync.aligned"]()
                    if not use_tma_store:
                        with txl.If(subtile == subtiles - 1), txl.Then():
                            txl.ptx["tcgen05.fence::before_thread_sync"]()
                            with txl.If(_elected()), txl.Then():
                                if g == 2:
                                    txl.ptx.mbarrier.arrive.shared__cluster.b64(
                                        acc_empty_leader.ptr_to([acc_state.stage]), txl.uint32(1))
                                else:
                                    txl.ptx.mbarrier.arrive.shared.b64(
                                        acc_pipe.empty.ptr_to([acc_state.stage]), txl.uint32(1))
                    if cb == 4:
                        for i in range(sub_cols):
                            txl.assign(words[i], txl.reinterpret("uint32", values[i]))
                    else:
                        convert = (txl.ptx.cvt.rn.bf16x2.f32 if c_type == "bf16"
                                   else txl.ptx.cvt.rn.f16x2.f32)
                        for i in range(sub_cols // 2):
                            convert(words[i], values[2 * i + 1], values[2 * i])

                    def column(chunk):
                        """Tile column of a thread's 16-byte chunk in this subtile."""
                        if layout == "split_n":
                            return column_half * (cta_n // 2) + subtile * 32 + chunk * chunk_elems
                        return subtile * sub_cols + chunk * chunk_elems

                    if use_tma_store:
                        def stage_chunks():
                            for chunk in range(chunks):
                                if layout == "split_n":
                                    block = column_half
                                    in_block = chunk
                                else:
                                    block = chunk * chunk_elems // block_cols
                                    in_block = chunk % (block_cols // chunk_elems)
                                address = (smem_base + c_offset + c_buffer * p.c_stage_bytes
                                           + block * block_bytes + row * row_bytes + in_block * 16)
                                swizzled = txl.bitwise_xor(address, txl.bitwise_and(
                                    txl.shift_right(address, txl.uint32(3)),
                                    txl.uint32(swizzle_mask)))
                                txl.ptx.st.shared.v4.b32(
                                    smem.ptr_to([txl.cast(swizzled - smem_base, "int32")]),
                                    *[words[chunk * 4 + i] for i in range(4)])

                        stores_when_active(stage_chunks)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(128))
                        with txl.If((warp == 0) & (lane == 0)), txl.Then():
                            for block in range(blocks):
                                if layout == "split_n":
                                    col = tile_n * cta_n + block * (cta_n // 2) + subtile * 32
                                else:
                                    col = tile_n * cta_n + subtile * sub_cols + block * block_cols
                                txl.ptx["cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"](
                                    txl.address_of(c_map), txl.cast(col, "int32"),
                                    txl.cast(tile_m * cta_m, "int32"), txl.int32(0),
                                    smem.ptr_to([c_offset + c_buffer * p.c_stage_bytes
                                                 + block * block_bytes]))
                            txl.ptx.cp.async_.bulk.commit_group()
                            txl.ptx.cp.async_.bulk.wait_group.read(c_stages - 1)
                        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(128))
                        txl.assign(c_buffer, c_buffer + 1)
                        with txl.If(c_buffer == c_stages), txl.Then():
                            txl.assign(c_buffer, 0)
                    else:
                        def store_chunks():
                            for chunk in range(chunks):
                                offset = ((tile_m * cta_m + row) * n + tile_n * cta_n
                                          + column(chunk)) * cb
                                txl.ptx.st.global_.v4.b32(
                                    c.ptr_to([offset]), *[words[chunk * 4 + i] for i in range(4)])

                        stores_when_active(store_chunks)
                if use_tma_store:
                    if protocol == "cutlass":
                        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(128))
                    txl.ptx["tcgen05.fence::before_thread_sync"]()
                    with txl.If(_elected()), txl.Then():
                        if g == 2:
                            txl.ptx.mbarrier.arrive.shared__cluster.b64(
                                acc_empty_leader.ptr_to([acc_state.stage]), txl.uint32(1))
                        else:
                            txl.ptx.mbarrier.arrive.shared.b64(
                                acc_pipe.empty.ptr_to([acc_state.stage]), txl.uint32(1))
                else:
                    txl.ptx.bar.sync(txl.uint32(1), txl.uint32(128))
                acc_state.advance()
                index = flag_index(work, tile_m, tile_n)
                with txl.If((warp == 0) & (lane == 0)), txl.Then():
                    if use_tma_store:
                        txl.ptx.cp.async_.bulk.wait_group(0)
                    if protocol == "cutlass":
                        txl.ptx.multimem_red.release.gpu.global_.add.s32(
                            flag_mc.ptr_to([index]), txl.int32(1))
                    else:
                        txl.ptx.fence.acq_rel.gpu()
                        txl.ptx.multimem_red.relaxed.gpu.global_.add.s32(
                            flag_mc.ptr_to([index]), txl.int32(1))
                        txl.ptx.fence.proxy.alias()
                next_work(work)

            with txl.If(warp == 0), txl.Then():
                txl.ptx[f"tcgen05.relinquish_alloc_permit.cta_group::{g}.sync.aligned"]()
            txl.ptx.bar.sync(txl.uint32(1), txl.uint32(128))
            with txl.If(warp == 0), txl.Then():
                if g == 2:
                    peer_dealloc = txl.local_scalar("uint32")
                    txl.ptx.mapa.shared__cluster.u32(
                        peer_dealloc, txl.cuda.cvta_generic_to_shared(tmem_dealloc.ptr_to([0])),
                        txl.cast(cluster_rank ^ 1, "uint32"))
                    txl.ptx.mbarrier.arrive.shared__cluster.b64(peer_dealloc, txl.uint32(1))
                    _wait(tmem_dealloc.ptr_to([0]), txl.uint32(0))
                txl.ptx[f"tcgen05.dealloc.cta_group::{g}.sync.aligned.b32"](
                    tmem_base, txl.uint32(p.tmem_columns))

        with comm_role:
            thread = (warp - 6) * 32 + lane
            thr_row = thread // thr_n
            thr_col = thread % thr_n
            if cb == 4:
                regs = txl.alloc_local((4 * 4,), "float32")
            else:
                regs = txl.alloc_local((4 * 4,), "uint32")
            dst = out_mc if protocol == "cutlass" else c_mc
            seen = txl.local_scalar("int32")
            work = txl.local_scalar("int32", init=first_work)

            def element_offset(tile_m, tile_n, it):
                i, j = divmod(it, loop_n)
                r = tile_m * cta_m + rank * m_local + i * thr_m + thr_row
                col = tile_n * cta_n + (j * thr_n + thr_col) * atom
                return (r * n + col) * cb

            def ld_reduce(slot, offset):
                out = [regs[4 * slot + e] for e in range(4)]
                if c_type == "f32":
                    txl.ptx.multimem_ld_reduce.weak.global_.add.v4.f32(*out, c_mc.ptr_to([offset]))
                elif c_type == "bf16":
                    txl.ptx.multimem_ld_reduce.weak.global_.add.acc__f32.v4.bf16x2(
                        *out, c_mc.ptr_to([offset]))
                else:
                    txl.ptx.multimem_ld_reduce.weak.global_.add.acc__f32.v4.f16x2(
                        *out, c_mc.ptr_to([offset]))

            def st(slot, offset):
                values = [regs[4 * slot + e] for e in range(4)]
                if cb != 4:
                    values = [txl.reinterpret("float32", v) for v in values]
                txl.ptx.multimem_st.weak.global_.v4.f32(dst.ptr_to([offset]), *values)

            with txl.While(work < p.cluster_work):
                tile_m, tile_n = tile_of(work)
                index = flag_index(work, tile_m, tile_n)
                with txl.If((warp == 6) & (lane == 0)), txl.Then():
                    txl.assign(seen, 0)
                    with txl.While(seen != world):
                        if protocol == "cutlass":
                            txl.ptx.atom.acquire.gpu.global_.cas.b32(
                                seen, flag.ptr_to([index]), txl.int32(world), txl.int32(0))
                        else:
                            txl.ptx.atom.relaxed.gpu.global_.cas.b32(
                                seen, flag.ptr_to([index]), txl.int32(world), txl.int32(0))
                txl.ptx.bar.sync(txl.uint32(3), txl.uint32(threads))
                total = loop_m * loop_n
                if protocol == "cutlass":
                    for start in range(0, total, 4):
                        group = range(start, min(start + 4, total))
                        offsets = [element_offset(tile_m, tile_n, it) for it in group]
                        for slot, offset in enumerate(offsets):
                            ld_reduce(slot, offset)
                        for slot, offset in enumerate(offsets):
                            st(slot, offset)
                else:
                    for it in range(total):
                        offset = element_offset(tile_m, tile_n, it)
                        ld_reduce(0, offset)
                        st(0, offset)
                next_work(work)
            if protocol == "cutlass":
                # Performance only (the leader's release.sys is cumulative over the
                # bar.sync): each thread drains its stores at GPU scope first, so the
                # GPU-wide MEMBAR.SYS behind the exit release finishes in one batch.
                # Under FlashInfer's protocol the fence costs more than it saves.
                txl.ptx.fence.acq_rel.gpu()
            txl.ptx.bar.sync(txl.uint32(3), txl.uint32(threads))
            with txl.If((warp == 6) & (lane == 0)), txl.Then():
                sm = block_x + block_y * cm + first_work * (cm * cn)
                txl.ptx.multimem_red.release.sys.global_.add.s32(
                    flag_mc.ptr_to([final_flags + sm]), txl.int32(1))
                if protocol != "cutlass":
                    txl.ptx.fence.proxy.alias()
                txl.assign(seen, 0)
                with txl.While(seen != world):
                    txl.ptx.atom.acquire.sys.global_.cas.b32(
                        seen, flag.ptr_to([final_flags + sm]), txl.int32(world), txl.int32(0))

    matrix_bytes = {"a": m * k * abb, "b": n * k * abb, "c": m * n * cb}
    kernel.__annotations__ = {
        "a": txl.gptr[txl.u8, (matrix_bytes["a"],)],
        "b": txl.gptr[txl.u8, (matrix_bytes["b"],)],
        "c": txl.gptr[txl.u8, (matrix_bytes["c"],)],
        "c_mc": txl.gptr[txl.u8, (matrix_bytes["c"],)],
        "out_mc": txl.gptr[txl.u8, (matrix_bytes["c"],)],
        "flag": txl.gptr[txl.i32, (flag_len,)],
        "flag_mc": txl.gptr[txl.i32, (flag_len,)],
    }
    return txl.kernel(warps=warps, arch="sm_100a", grid=[cm, cn, p.clusters],
                      host_prelude=host_prelude)(kernel)
