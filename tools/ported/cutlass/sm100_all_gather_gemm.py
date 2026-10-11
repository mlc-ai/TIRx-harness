"""SM100 all-gather GEMM: every rank computes ``out = all_gather(a) @ b.T``.

TIRx port of the persistent, warp-specialized GEMM of CUTLASS's
``distributed_all_gather_gemm_blackwell.py`` (``PersistentDenseGemmKernel``
with ``gated_a_load``): the same warp roles (epilogue warps 0-3, the MMA warp
4, the TMA warp 5), configuration space (1- or 2-CTA ``tcgen05.mma``, the MMA
tile, the cluster shape with TMA multicast of A along the cluster's N and of B
along its M, the static persistent scheduler's raster order and swizzle, and
the TMA-store or direct-store epilogue) and derived parameters (stage counts,
epilogue subtile, TMEM columns), computed with CUTLASS's formulas.

CUTLASS launches one GEMM per rank's shard, each remote one gated on a flag
its copy stream releases after pulling the shard. This port covers every shard
in one persistent launch, as FlashInfer's cake all-gather matmul does: the
scheduler walks this rank's shard first and then rank ``rank + j``'s at step
``j``, each in ``chunk_rows``-row chunks, and the TMA warp acquires a remote
chunk's flag (``ld.acquire.sys``, then ``fence.proxy.async.global`` for its
TMA loads) before loading the chunk's rows. The host schedule fills the
scratch and the flags (``benchmarks/peer/all_gather_gemm.py``): copy-engine
copies of each chunk into this rank's scratch, pulled from or pushed by the
shard's owner, each followed by a stream write of the chunk's flag;
``barrier_kernel`` clears the flags and aligns the ranks before every launch.
With ``copy_warps`` the kernel instead pushes the chunks and releases the flags
itself, from extra warps; NumSim runs, which launch one kernel per rank and no
copy engine, use that variant.

Operands, all byte buffers but the flags: ``a`` is this rank's ``m x k`` shard,
``scratch`` holds rank ``p``'s shard in rows ``[p * m, (p + 1) * m)`` (this
rank's rows unused), ``b`` is ``n x k``, all K-major; ``out`` is the row-major
``world * m x n`` result with rank ``p``'s rows at ``[p * m, (p + 1) * m)``;
``flags`` holds one ``uint32`` per (rank, chunk), nonzero once the chunk is in
``scratch``.
"""

import functools
import inspect
from dataclasses import dataclass

import tirx_kernels.tirx_lite as txl

SMEM_CAPACITY = 232_448
AB_BYTES = {"tf32": 4, "f16": 2, "bf16": 2}
C_BYTES = {"f32": 4, "f16": 2, "bf16": 2}
STORAGE = {"tf32": "float32", "f16": "float16", "bf16": "bfloat16", "f32": "float32"}
TRY_WAIT_TICKS = 10_000_000


@dataclass(frozen=True)
class Plan:
    """Static schedule of one configuration; field names follow CUTLASS."""

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
    chunk_rows: int
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
    def chunks(self):
        return self.m // self.chunk_rows

    @property
    def cluster_tiles(self):
        """Cluster tiles (M, N) of one chunk."""
        return (self.chunk_rows // (self.cta_m * self.cluster[0]),
                self.n // (self.cta_n * self.cluster[1]))

    @property
    def chunk_work(self):
        return self.cluster_tiles[0] * self.cluster_tiles[1]

    @property
    def shard_work(self):
        return self.chunks * self.chunk_work

    @property
    def cluster_work(self):
        return self.world * self.shard_work

    @property
    def clusters(self):
        """Persistent clusters launched (``StaticPersistentTileScheduler.get_grid_shape``)."""
        return min(self.cluster_work, self.max_active_clusters)

    @property
    def flags(self):
        return self.world * self.chunks

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
    def shared_bytes(self):
        return 1024 + self.c_stages * self.c_stage_bytes + self.ab_stages * (
            self.a_stage_bytes + self.b_stage_bytes)


def plan(m, n, k, ab, c, world, *, use_2cta, mma_tiler, cluster, use_tma_store, raster="m",
         swizzle=1, chunk_rows=None, max_active_clusters):
    """The schedule of one configuration; ``chunk_rows`` defaults to the whole shard."""

    p = Plan(m, n, k, ab, c, world, use_2cta, tuple(mma_tiler), tuple(cluster), raster, swizzle,
             use_tma_store, chunk_rows or m, max_active_clusters)
    if p.cta_m not in (64, 128) or p.cta_n % 32 or p.cluster[0] % p.cta_group:
        raise ValueError(f"unsupported MMA tile / cluster {p.mma_tiler} {p.cluster}")
    if p.m % p.chunk_rows or p.chunk_rows % (p.cta_m * p.cluster[0]):
        raise ValueError("a chunk must be a whole number of cluster tiles and divide the shard")
    if p.n % (p.cta_n * p.cluster[1]) or p.k % p.k_tile:
        raise ValueError("N must be a whole number of cluster tiles and K of K tiles")
    if swizzle > 1 and p.cluster_tiles[1 if raster == "m" else 0] % swizzle:
        raise ValueError("the swizzle must divide the swizzled cluster count")
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


def _with_signature(fn, params, host=False):
    """``fn`` declared with the named, annotated positional ``params`` (and the
    keyword-only ``host`` a host prelude fills)."""
    signature = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation)
        for name, annotation in params]
    if host:
        signature.append(inspect.Parameter("host", inspect.Parameter.KEYWORD_ONLY))
    fn.__signature__ = inspect.Signature(signature)
    return fn


def _peer(pointer, offset):
    """``pointer`` moved by ``offset`` bytes into a peer's replica of its buffer."""
    return txl.reinterpret("handle", txl.reinterpret("uint64", pointer) + txl.cast(offset, "uint64"))


@functools.cache
def barrier_kernel(rank, world, flag_len):
    """CUTLASS's ``SyncNvlDevices`` with release/acquire ordering: one thread
    clears ``flags``, adds one to every peer's ``counter`` (``red.release.sys``,
    after the clears), waits for its own counter to reach ``world - 1``
    (acquire ``.sys``) and takes ``world - 1`` back off. Peer ``p``'s counter
    is at the local address plus ``symm_rank_offset_p``; ``rank=None`` makes
    the rank an ``int32`` parameter ``rank`` before them."""

    def kernel(flags, counter, *scalars):
        this_rank = scalars[0] if rank is None else rank
        offsets = scalars[1:] if rank is None else scalars
        with txl.If(txl.thread_id() == 0), txl.Then():
            for index in range(flag_len):
                txl.ptx.st.global_.u32(flags.ptr_to([index]), txl.uint32(0))
            for peer in range(world):
                if rank is None:
                    with txl.If(this_rank != peer), txl.Then():
                        txl.ptx.red.release.sys.global_.add.s32(
                            _peer(counter.ptr_to([0]), offsets[peer]), txl.int32(1))
                elif peer != rank:
                    txl.ptx.red.release.sys.global_.add.s32(
                        _peer(counter.ptr_to([0]), offsets[peer]), txl.int32(1))
            arrivals = txl.local_scalar("int32")
            txl.cuda.wait_until(arrivals, counter.ptr_to([0]),
                                lambda value: value >= txl.int32(world - 1),
                                scope="sys", ptx_type="s32")
            txl.ptx.red.relaxed.sys.global_.add.s32(counter.ptr_to([0]), txl.int32(1 - world))

    params = [("flags", txl.gptr[txl.u32, (flag_len,)]), ("counter", txl.gptr[txl.i32, (1,)])]
    if rank is None:
        params.append(("rank", txl.i32))
    params += [(f"symm_rank_offset_{peer}", txl.i64) for peer in range(world)]
    return txl.kernel(warps=1, arch="sm_100a", grid=[1, 1, 1])(_with_signature(kernel, params))


@functools.cache
def all_gather_gemm(m, n, k, ab, c, rank, world, *, use_2cta, mma_tiler, cluster,
                    use_tma_store, raster="m", swizzle=1, chunk_rows=None,
                    max_active_clusters, copy_warps=0):
    """Build rank ``rank``'s GEMM; returns a ``txl.Kernel`` with parameters
    ``a, scratch, b, out, flags`` (see the module docstring).

    ``copy_warps`` > 0 adds a copy role in its own warps, which does in the
    kernel what the host schedule's pushes do: the CTAs take the (peer, chunk)
    items in turn, ``(rank - j) % world`` before ``(rank - j - 1) % world``,
    and for each one the role stores the chunk of ``a`` into that peer's
    ``scratch`` with 16-byte stores, joins its warps (``bar.sync``) and adds one
    to the peer's flag of the chunk (``red.release.sys``). ``scratch`` and
    ``flags`` must then be symmetric memory at the same offset from each peer's
    replica: parameter ``symm_rank_offset_p``, rank ``p``'s minus this rank's
    address.

    ``rank=None`` makes the rank an ``int32`` parameter ``rank`` after
    ``flags``, so every rank runs one kernel, as a NumSim multi-rank launch
    needs."""

    p = plan(m, n, k, ab, c, world, use_2cta=use_2cta, mma_tiler=mma_tiler, cluster=cluster,
             use_tma_store=use_tma_store, raster=raster, swizzle=swizzle, chunk_rows=chunk_rows,
             max_active_clusters=max_active_clusters)
    g, cta_m, cta_n = p.cta_group, p.cta_m, p.cta_n
    cm, cn = p.cluster
    ncm, ncn = p.cluster_tiles
    cmg = p.cluster_m_groups
    stages, c_stages = p.ab_stages, p.c_stages
    abb, cb = AB_BYTES[ab], C_BYTES[c]
    c_type = c
    warps = 6 + copy_warps

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

    def host_prelude(params):
        a_map = txl.stack_alloca("tensormap", 1)
        s_map = txl.stack_alloca("tensormap", 1)
        b_map = txl.stack_alloca("tensormap", 1)
        c_map = txl.stack_alloca("tensormap", 1) if use_tma_store else None

        def encode(descriptor, dtype, data, inner, outer, box_inner, box_outer, swizzle_mode,
                   element_bytes):
            txl.call_packed("runtime.cuTensorMapEncodeTiled", descriptor, dtype, 3, data,
                            inner, outer, 1, inner * element_bytes, inner * outer * element_bytes,
                            box_inner, box_outer, 1, 1, 1, 1, 0, swizzle_mode, 2, 0)

        encode(a_map, STORAGE[ab], params["a"].data, k, m, p.k_tile, a_piece, 3, abb)
        encode(s_map, STORAGE[ab], params["scratch"].data, k, world * m, p.k_tile, a_piece, 3,
               abb)
        encode(b_map, STORAGE[ab], params["b"].data, k, n, p.k_tile, b_piece, 3, abb)
        if use_tma_store:
            encode(c_map, STORAGE[c], params["out"].data, n, world * m, block_cols, block_rows,
                   swizzle_mode, cb)
        return a_map, s_map, b_map, c_map

    def kernel(a, scratch, b, out, flags, *scalars, host):
        this_rank = scalars[0] if rank is None else rank
        offsets = scalars[1:] if rank is None else scalars
        if not copy_warps:
            del a, scratch
        del b
        a_map, s_map, b_map, c_map = host
        block_x, block_y, first_work = txl.cta_id()
        cta_linear = block_x + block_y * cm + first_work * (cm * cn)
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
        del block_x, block_y

        roles = txl.specialize(chain_dispatch=True)
        epilogue_role = roles.role("epilogue", warps=[0, 1, 2, 3])
        mma_role = roles.role("mma", warps=[4])
        tma_role = roles.role("tma", warps=[5])
        if copy_warps:
            copy_role = roles.role("copy", warps=list(range(6, warps)))

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
            txl.ptx.prefetch.tensormap(txl.address_of(s_map))
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
            """(step, chunk, CTA tile m within the shard, CTA tile n) of a cluster
            work index: steps, then chunks, then ``StaticPersistentTileScheduler``'s
            order over the chunk's cluster tiles."""
            step = work // p.shard_work
            rest = work % p.shard_work
            chunk = rest // p.chunk_work
            w = rest % p.chunk_work
            s = swizzle
            if s == 1 and raster == "m":
                cl_m, cl_n = w % ncm, w // ncm
            elif s == 1:
                cl_m, cl_n = w // ncn, w % ncn
            elif raster == "m":
                cl_m = (w // s) % ncm
                cl_n = w % s + s * (w // (s * ncm))
            else:
                cl_m = w % s + s * (w // (s * ncn))
                cl_n = (w // s) % ncn
            return step, chunk, (chunk * ncm + cl_m) * cm + cluster_x, cl_n * cn + cluster_y

        def shard_of(step):
            return (step + this_rank) % world

        def next_work(work):
            txl.assign(work, work + p.clusters)

        if copy_warps:
            with copy_role:
                thread = (warp - 6) * 32 + lane
                threads = 32 * copy_warps
                chunk_bytes = p.chunk_rows * k * abb
                item = txl.local_scalar("int32", init=cta_linear)
                byte = txl.local_scalar("int32")
                words = txl.alloc_local((4,), "uint32")
                with txl.While(item < (world - 1) * p.chunks):
                    chunk = item % p.chunks
                    peer = (this_rank + world - 1 - item // p.chunks) % world
                    offset = offsets[world - 1]
                    for q in reversed(range(world - 1)):
                        offset = txl.Select(peer == q, offsets[q], offset)
                    source = chunk * chunk_bytes
                    target = (this_rank * m + chunk * p.chunk_rows) * k * abb
                    txl.assign(byte, thread * 16)
                    with txl.While(byte < chunk_bytes):
                        txl.ptx.ld.global_.v4.b32(*[words[i] for i in range(4)],
                                                  a.ptr_to([source + byte]))
                        txl.ptx.st.global_.v4.b32(_peer(scratch.ptr_to([target + byte]), offset),
                                                  *[words[i] for i in range(4)])
                        txl.assign(byte, byte + threads * 16)
                    txl.ptx.bar.sync(txl.uint32(3), txl.uint32(threads))
                    with txl.If(thread == 0), txl.Then():
                        txl.ptx.red.release.sys.global_.add.u32(
                            _peer(flags.ptr_to([this_rank * p.chunks + chunk]), offset),
                            txl.uint32(1))
                    txl.assign(item, item + cm * cn * p.clusters)

        with tma_role:
            state = txl.PipelineState(stages, phase=1)
            work = txl.local_scalar("int32", init=first_work)
            count = txl.local_scalar("int32")
            ready = txl.local_scalar("uint32")
            seen = txl.local_scalar("uint32")
            # As in the MMA warp, one elected thread runs the whole load stream.
            with txl.If(_elected()), txl.Then():
                with txl.While(work < p.cluster_work):
                    step, chunk, tile_m, tile_n = tile_of(work)
                    shard = shard_of(step)
                    remote = step != txl.int32(0)
                    with txl.If(remote), txl.Then():
                        txl.cuda.wait_until(seen, flags.ptr_to([shard * p.chunks + chunk]),
                                            lambda value: value != txl.uint32(0),
                                            scope="sys", ptx_type="u32")
                        txl.ptx.fence.proxy.async_.global_()
                    txl.assign(count, 0)
                    txl.assign(ready, txl.uint32(1))
                    _try_wait(ready, ab_pipe.empty.ptr_to([state.stage]), state.phase)
                    with txl.While(count < p.k_tiles):
                        _wait_unless(ab_pipe.empty.ptr_to([state.stage]), state.phase, ready)
                        with txl.If(leader_cta), txl.Then():
                            txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                ab_pipe.full.ptr_to([state.stage]), txl.uint32(tma_bytes))
                        a_row = tile_m * cta_m + cluster_y * a_piece
                        loads = (
                            (lambda: txl.Select(remote, txl.address_of(s_map),
                                                txl.address_of(a_map)),
                             a_offset + state.stage * p.a_stage_bytes + cluster_y * a_piece_bytes,
                             txl.Select(remote, shard * m + a_row, a_row), a_mask, cn),
                            (lambda: txl.address_of(b_map),
                             b_offset + state.stage * p.b_stage_bytes
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
                                ](smem.ptr_to([offset]), tensor_map(), *coords,
                                  ab_pipe.full.ptr_to([state.stage]))
                            elif g == 1:
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                ](cluster_smem + offset, tensor_map(), *coords,
                                  ab_pipe.full.ptr_to([state.stage]), txl.cast(mask, "uint16"))
                            elif peers == 1:
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.cta_group::2"
                                ](cluster_smem + offset, tensor_map(), *coords,
                                  ab_full_leader.ptr_to([state.stage]))
                            else:
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".cta_group::2"
                                ](cluster_smem + offset, tensor_map(), *coords,
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
                step, _, tile_m, tile_n = tile_of(work)
                tile_row = shard_of(step) * m + tile_m * cta_m
                _wait(acc_pipe.full.ptr_to([acc_state.stage]), acc_state.phase)
                with txl.unroll(0, subtiles) as subtile:
                    taddr = (tmem_base + txl.shift_left(txl.cast(warp * 32, "uint32"),
                                                        txl.uint32(16))
                             + txl.cast(acc_state.stage * p.acc_columns + subtile * sub_cols,
                                        "uint32"))
                    txl.ptx[f"tcgen05.ld.sync.aligned.32x32b.x{sub_cols}.b32"](
                        *[values[i] for i in range(sub_cols)], taddr)
                    txl.ptx["tcgen05.wait::ld.sync.aligned"]()
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
                                    txl.cast(tile_row, "int32"), txl.int32(0),
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
                                offset = ((tile_row + row) * n + tile_n * cta_n
                                          + column(chunk)) * cb
                                txl.ptx.st.global_.v4.b32(
                                    out.ptr_to([offset]), *[words[chunk * 4 + i] for i in range(4)])

                        stores_when_active(store_chunks)
                txl.ptx["tcgen05.fence::before_thread_sync"]()
                with txl.If(_elected()), txl.Then():
                    if g == 2:
                        txl.ptx.mbarrier.arrive.shared__cluster.b64(
                            acc_empty_leader.ptr_to([acc_state.stage]), txl.uint32(1))
                    else:
                        txl.ptx.mbarrier.arrive.shared.b64(
                            acc_pipe.empty.ptr_to([acc_state.stage]), txl.uint32(1))
                acc_state.advance()
                next_work(work)

            if use_tma_store:
                with txl.If((warp == 0) & (lane == 0)), txl.Then():
                    txl.ptx.cp.async_.bulk.wait_group(0)
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

    params = [
        ("a", txl.gptr[txl.u8, (m * k * abb,)]),
        ("scratch", txl.gptr[txl.u8, (world * m * k * abb,)]),
        ("b", txl.gptr[txl.u8, (n * k * abb,)]),
        ("out", txl.gptr[txl.u8, (world * m * n * cb,)]),
        ("flags", txl.gptr[txl.u32, (p.flags,)]),
    ]
    if rank is None:
        params.append(("rank", txl.i32))
    if copy_warps:
        params += [(f"symm_rank_offset_{peer}", txl.i64) for peer in range(world)]
    return txl.kernel(warps=warps, arch="sm_100a", grid=[cm, cn, p.clusters],
                      host_prelude=host_prelude)(_with_signature(kernel, params, host=True))
