"""Host wrappers that launch the original CUTLASS and FlashInfer kernels on the
current torch stream. The kernels themselves are imported unchanged.

CuTeDSL evaluates a jit function's annotations against this module's globals,
so this module imports the DSL at the top and does not use postponed
annotations.
"""

import functools
import importlib.util
import sys

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.cute.testing as testing
import cutlass.utils as utils
import cutlass.utils.distributed as distributed
import torch

from benchmarks.multimem import common

sys.path.insert(0, str(common.CUTLASS_DISTRIBUTED))
import all_reduce_two_shot_multimem as _two_shot  # noqa: E402
import distributed_gemm_all_reduce_blackwell as _cutlass_gemm_ar  # noqa: E402

AB_TYPES = {"tf32": cutlass.TFloat32, "f16": cutlass.Float16, "bf16": cutlass.BFloat16}
C_TYPES = {"f32": cutlass.Float32, "f16": cutlass.Float16, "bf16": cutlass.BFloat16}
CantImplement = testing.CantImplementError


@functools.cache
def _flashinfer():
    """FlashInfer's module, with its final barrier's CAS wait replaced by the
    CuTeDSL helper it stands in for (CUTLASS issue 2845). FlashInfer's copy hands
    ``nvvm.atomicrmw`` the reset value as the compare operand and the expected
    count as the new value, so the barrier neither waits for the other ranks nor
    resets its flag, and a second launch on the same flags does not synchronize
    (a third hangs)."""

    path = common.FLASHINFER_DIR / "flashinfer/cute_dsl/gemm_allreduce_two_shot.py"
    spec = importlib.util.spec_from_file_location("flashinfer_gemm_allreduce_two_shot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.spin_lock_atom_cas_acquire_wait = distributed.spin_lock_atom_cas_acquire_wait
    return module


@functools.cache
def max_active_clusters(cluster_size: int) -> int:
    return utils.HardwareInfo().get_max_active_clusters(cluster_size)


def current_stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


@cute.jit
def two_shot_all_reduce(mIn: cute.Tensor, mOut: cute.Tensor, flag: cute.Tensor,
                        flag_mc: cute.Tensor, stream: cuda.CUstream,
                        local_rank: cutlass.Constexpr, world_size: cutlass.Constexpr,
                        copy_bits: cutlass.Constexpr = 128):
    """``all_reduce_two_shot_multimem.all_reduce_multimem`` with a stream."""
    vector_size = copy_bits // mIn.element_type.width
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((32, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gIn = cute.zipped_divide(mIn, tiler_mn)
    gOut = cute.zipped_divide(mOut, tiler_mn)
    _two_shot.all_reduce_multimem_kernel(
        gIn, gOut, flag, flag_mc, thr_layout, val_layout, local_rank, world_size,
    ).launch(
        grid=[cute.size(gOut, mode=[1]) // world_size, 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
        stream=stream,
    )


def two_shot_launchers(rank: int, world: int, workspace: list[dict]):
    """One launch closure per workspace of ``in_mc, out_mc, flag, flag_mc``."""

    def views(ws):
        return [from_dlpack(ws[name]) for name in ("in_mc", "out_mc", "flag", "flag_mc")]

    compiled = cute.compile(two_shot_all_reduce, *views(workspace[0]), current_stream(),
                            rank, world)
    args = [views(ws) for ws in workspace]

    def launcher(index):
        return lambda: compiled(*args[index], current_stream())

    return [launcher(index) for index in range(len(workspace))]


def _matrix(tensor: torch.Tensor, dtype):
    """A CuTe ``(rows, cols, 1)`` view of a row-major ``(rows, cols)`` tensor."""

    view = from_dlpack(tensor.unsqueeze(0).permute(1, 2, 0), assumed_align=16)
    view.element_type = dtype
    return view.mark_layout_dynamic(leading_dim=1)


def _flags(tensor: torch.Tensor):
    view = from_dlpack(tensor, assumed_align=16)
    view.element_type = cutlass.Int32
    return view.mark_layout_dynamic()


def gemm_all_reduce_launchers(origin: str, rank: int, world: int, shape: dict, config: dict,
                              workspace: list[dict], shared: dict):
    """One launch closure per workspace of the origin's GEMM + all-reduce.

    ``shape`` holds ``m, n, k, ab, c``; ``config`` the origin's knobs. Each
    workspace has its own ``a`` and ``b``; ``shared`` holds the symmetric ``c``
    and ``out`` with their multicast views and the flags. CUTLASS reduces ``c``
    into ``out``, FlashInfer reduces ``c`` in place. Raises ``CantImplement``
    for a configuration the origin rejects.
    """

    ab, c = AB_TYPES[shape["ab"]], C_TYPES[shape["c"]]
    mnkl = (shape["m"], shape["n"], shape["k"], 1)
    tiler, cluster = tuple(config["mma_tiler"]), tuple(config["cluster"])
    operands = [(_matrix(ws["a"], ab), _matrix(ws["b"], ab)) for ws in workspace]
    c_view, c_mc = _matrix(shared["c"], c), _matrix(shared["c_mc"], c)
    flag, flag_mc = _flags(shared["flag"]), _flags(shared["flag_mc"])
    clusters = max_active_clusters(cluster[0] * cluster[1])

    if origin == "cutlass":
        kernel = _cutlass_gemm_ar.Sm100PersistentDenseGemmAllReduceLDMCxSTMCKernel(
            cutlass.Float32, c, config["use_2cta"], tiler, cluster, config["use_tma_store"],
            rank_id=rank, num_ranks=world, all_reduce="LDMCxSTMC",
            swizzle_size=config["swizzle"], raster_order=config["raster"])
        if not kernel.can_implement(mnkl, ab, c, "k", "k", "n"):
            raise CantImplement(f"cutlass rejects {config}")
        fixed = dict(c=c_view, comm_in_multicast_tensor=c_mc,
                     comm_out_multicast_tensor=_matrix(shared["out_mc"], c),
                     barrier_flag_unicast=flag, barrier_flag_multicast=flag_mc)
        compiled = cute.compile(kernel, a=operands[0][0], b=operands[0][1], **fixed,
                                stream=current_stream(), max_active_clusters=clusters)

        def launcher(a, b):
            return lambda: compiled(a=a, b=b, **fixed, stream=current_stream())
    elif origin == "flashinfer":
        module = _flashinfer()
        if not module.PersistentDenseGemmKernel.can_implement(
                ab, cutlass.Float32, c, config["use_2cta"], tiler, cluster,
                config["use_tma_store"], *mnkl, "k", "k", "n", "two_shot"):
            raise CantImplement(f"flashinfer rejects {config}")
        kernel = module.PersistentDenseGemmKernel(
            cutlass.Float32, config["use_2cta"], tiler, cluster, config["use_tma_store"],
            all_reduce="two_shot", sm_version="sm_100")
        fixed = dict(c_mc=c_mc, barrier_flag=flag, barrier_flag_mc=flag_mc)
        compiled = cute.compile(kernel, operands[0][0], operands[0][1], c_view, clusters,
                                current_stream(), **fixed)

        def launcher(a, b):
            return lambda: compiled(a, b, c_view, current_stream(), **fixed)
    else:
        raise ValueError(origin)
    return [launcher(a, b) for a, b in operands]
