"""Remote mbarrier kernels shared by NumSim and Synccheck regressions."""

from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T


def _mapa_u64(ptr, rank):
    mapped = T.alloc_local((1,), "uint64")
    T.evaluate(T.ptx.mapa.u64(mapped[0], ptr, T.uint32(rank)))
    return mapped[0]


@T.prim_func
def mapped_remote_mbarrier_pointer_expect_tx(output: T.Buffer((2,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        if cta == 0:
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        else:
            remote_ptr: T.let[
                T.Var(
                    name="remote_barrier_expect_tx_ptr",
                    ty=PointerType(PrimType("uint64"), "shared"),
                )
            ] = T.reinterpret(
                PointerType(PrimType("uint64"), "shared"),
                _mapa_u64(barriers.ptr_to([0]), 0),
            )
            remote_barrier = T.decl_buffer((1,), "uint64", scope="shared", data=remote_ptr)
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(remote_barrier[0]), 0)
    T.cuda.cluster_sync()
    if lane == 0:
        output[cta] = 1


@T.prim_func
def mapped_remote_mbarrier_cluster_view(output: T.Buffer((2,), "int32")):
    """Arrive through a pointer-derived view of another CTA's barrier."""

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        if cta == 0:
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        else:
            remote_ptr: T.let[
                T.Var(
                    name="remote_barrier_cluster_view_ptr",
                    ty=PointerType(PrimType("uint64"), "shared"),
                )
            ] = T.reinterpret(
                PointerType(PrimType("uint64"), "shared"),
                _mapa_u64(barriers.ptr_to([0]), 0),
            )
            remote_barrier = T.decl_buffer((1,), "uint64", scope="shared", data=remote_ptr)
            T.ptx.mbarrier.arrive.shared__cluster.b64(T.address_of(remote_barrier[0]), T.uint32(1))
    T.cuda.cluster_sync()
    if lane == 0:
        output[cta] = 1
