"""Remote mbarrier kernels shared by NumSim and Synccheck regressions."""

from tvm_tirx.ir import PtrType
from tvm_tirx.ir.type import PrimType
from tvm_tirx.script import tirx as T


def _mapa_u64(ptr, rank):
    mapped = T.alloc_local((1,), "uint64").value
    T.evaluate(T.ptx.mapa.u64(mapped[0], ptr, T.uint32(rank)))
    return mapped[0]


@T.function
def mapped_remote_mbarrier_pointer_expect_tx(output: T.Tensor((2,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    barriers = T.alloc_shared((1,), "uint64")
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
                    ty=PtrType(PrimType("uint64"), "shared"),
                )
            ] = T.reinterpret(
                _mapa_u64(barriers.ptr_to([0]), 0), ty=PtrType(PrimType("uint64"), "shared")
            )
            remote_barrier = T.decl_tensor(
                remote_ptr, ty_args=[T.Tensor((1,), "uint64", scope="shared")]
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(remote_barrier[0]), 0)
    T.cuda.cluster_sync()
    if lane == 0:
        output[cta] = 1


@T.function
def mapped_remote_mbarrier_cluster_view(output: T.Tensor((2,), "int32")):
    """Arrive through a pointer-derived view of another CTA's barrier."""

    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    barriers = T.alloc_shared((1,), "uint64")
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
                    ty=PtrType(PrimType("uint64"), "shared"),
                )
            ] = T.reinterpret(
                _mapa_u64(barriers.ptr_to([0]), 0), ty=PtrType(PrimType("uint64"), "shared")
            )
            remote_barrier = T.decl_tensor(
                remote_ptr, ty_args=[T.Tensor((1,), "uint64", scope="shared")]
            )
            T.ptx.mbarrier.arrive.shared__cluster.b64(T.address_of(remote_barrier[0]), T.uint32(1))
    T.cuda.cluster_sync()
    if lane == 0:
        output[cta] = 1
