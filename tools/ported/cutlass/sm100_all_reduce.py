"""CUTLASS two-shot NVLS all-reduce expressed in TIRx.

The kernel is shared by the GPU benchmark, NumSim, and native checkers.
Concrete NumPy bindings live in the test support module.
"""

from tvm.script import tirx as T

TILE = 128


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
        for iteration in T.unroll(32):
            row = tile % tiles * TILE + warp + 4 * iteration
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
