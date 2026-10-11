"""Host wrappers that launch the original all-gather GEMM kernels on the
current torch stream: CUTLASS's ``distributed_all_gather_gemm_blackwell.py`` and
FlashInfer's ``comm/all_gather_matmul`` (its SM100 cake backend and its default
cuTile backend). The kernels and their communication schedules are the
origins' own; the wrappers only hoist each origin's per-call allocation and
rendezvous out of the timed launch.

Every wrapper computes ``out[p * M:(p + 1) * M] = a_p @ b.T`` on each rank for
the ``M x K`` shard ``a_p`` of rank ``p`` and this rank's ``N x K`` weight ``b``
(both K-major, ``out`` row-major).
"""

import functools
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import re
import sys
import tempfile

import cuda.bindings.driver as driver
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from benchmarks.multimem import common
from benchmarks.multimem.origins import AB_TYPES, C_TYPES, CantImplement, _matrix, max_active_clusters

sys.path.insert(0, str(common.CUTLASS_DISTRIBUTED))
import distributed_all_gather_gemm_blackwell as _cutlass_ag  # noqa: E402


@functools.cache
def _flashinfer(module: str):
    """A module of FlashInfer's ``comm.all_gather_matmul`` package, imported from
    the source tree (``flashinfer.comm`` re-exports a function of the package's
    name, so the submodule is imported by its full name)."""

    if str(common.FLASHINFER_DIR) not in sys.path:
        sys.path.insert(0, str(common.FLASHINFER_DIR))
    return importlib.import_module(f"flashinfer.comm.all_gather_matmul.{module}")


_CUTILE_POLL = re.compile(
    r"ct\.load\(\s*signal_pad, index=signal_index, shape=\(\), padding_mode=zero_pad\s*\)")
_CUTILE_ATOMIC_POLL = ("ct.atomic_add(signal_pad, signal_index, 0, memory_order=ct.MemoryOrder.ACQUIRE,"
                       " memory_scope=ct.MemoryScope.SYS)")


@functools.cache
def _cutile_atomic_poll():
    """FlashInfer's ``all_gather_matmul_cutile`` module with the matmul's two
    signal-pad loads replaced by atomic reads (an acquire add of zero at system
    scope). FlashInfer polls with cuTile's default weak load; a loop of weak
    loads has no side effect, so the compiler deletes it and the matmul reads a
    peer's chunk without waiting for it. An acquire load is not enough either:
    every warp loads the flag itself, so warps leave the loop on different
    iterations and the rest wait forever at the loop's barrier. cuTile issues an
    atomic from one thread and broadcasts the result through shared memory (as
    in FlashInfer's own barrier kernel), so all warps wait and leave together."""

    original = _flashinfer("all_gather_matmul_cutile")
    source, polls = _CUTILE_POLL.subn(_CUTILE_ATOMIC_POLL, Path(original.__file__).read_text())
    if polls != 2:
        raise AssertionError("FlashInfer's cuTile signal poll changed; update the patch")
    # cuTile parses a kernel's source with ``inspect``, so the module needs a file.
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"flashinfer_cutile_atomic_poll_{digest}.py"
    if not path.exists():
        partial = path.with_suffix(f".{os.getpid()}.tmp")
        partial.write_text(source)
        os.replace(partial, path)
    name = f"{original.__package__}._cutile_atomic_poll"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def peer_pointers(tensor: torch.Tensor) -> list[int]:
    """Each rank's address of ``tensor``, a symmetric-memory tensor."""

    handle = symm_mem.rendezvous(tensor, dist.group.WORLD.group_name)
    offset = tensor.data_ptr() - handle.buffer_ptrs[handle.rank]
    return [pointer + offset for pointer in handle.buffer_ptrs]


def _stream(stream: torch.cuda.Stream):
    return driver.CUstream(stream.cuda_stream)


def cutlass_launchers(rank: int, world: int, shape: dict, config: dict,
                      workspace: list[dict], shared: dict):
    """CUTLASS's graph body per workspace: ``SyncNvlDevices`` (which also clears
    the step flags), then on a GEMM stream one persistent GEMM per rank's shard,
    local first, each remote one gated on its step flag, while a copy stream
    pulls each remote shard with ``cuMemcpyDtoDAsync`` from the peer's
    symmetric ``a`` and releases its flag with ``ReleaseFlagSet``.

    ``shared`` holds the pulled shards ``pulled[p]`` (``M x K``, local), the
    ``gate`` flags (``world`` int32, local) and the symmetric ``arrival``
    counter (one int32).
    """

    ab, c = AB_TYPES[shape["ab"]], C_TYPES[shape["c"]]
    m, n, k = shape["m"], shape["n"], shape["k"]
    tiler, cluster = tuple(config["mma_tiler"]), tuple(config["cluster"])
    kernels = {
        gated: _cutlass_ag.PersistentDenseGemmKernel(
            cutlass.Float32, config["use_2cta"], tiler, cluster, config["use_tma_store"],
            gated_a_load=gated)
        for gated in (False, True)
    }

    def a_views(ws):
        return [_matrix(ws["a"] if p == rank else shared["pulled"][p], ab) for p in range(world)]

    def c_views(ws):
        return [_matrix(ws["out"][p * m:(p + 1) * m], c) for p in range(world)]

    b_views = [_matrix(ws["b"], ab) for ws in workspace]
    a0, c0 = a_views(workspace[0]), c_views(workspace[0])
    if not kernels[False].can_implement(a0[0], b_views[0], c0[0]):
        raise CantImplement(f"cutlass rejects {config}")
    gate = from_dlpack(shared["gate"])
    clusters = max_active_clusters(cluster[0] * cluster[1])
    stream = _stream(torch.cuda.current_stream())
    # ``flag_offset`` is a Constexpr: one compiled GEMM per shard.
    compiled = [cute.compile(kernels[p != rank], a0[p], b_views[0], c0[p], clusters, stream,
                             flag_offset=p, gate_a_flags=gate) for p in range(world)]
    arrival = torch.tensor(peer_pointers(shared["arrival"]), device="cuda", dtype=torch.int64)
    arrival_view = from_dlpack(arrival)
    sync = cute.compile(_cutlass_ag.SyncNvlDevices(world), device_idx=rank,
                        device_arrival_counters=arrival_view, iteration_flags=gate,
                        stream=stream)
    release = cute.compile(_cutlass_ag.ReleaseFlagSet(), gate_a_flags=gate,
                           step=cutlass.Int32(0), stream=stream)
    # A compiled function loads its library on first call, and the load waits
    # for the device to idle: loading a later step's GEMM while an earlier gated
    # GEMM spins on its flag would deadlock. CUTLASS's example captures its
    # graph before any launch; here every library is loaded up front.
    for function in (*compiled, sync, release):
        function.to(None)
    gemm_stream, copy_stream = torch.cuda.Stream(), torch.cuda.Stream()
    shard_bytes = m * k * workspace[0]["a"].element_size()

    def launcher(ws, b_view):
        a, cs = a_views(ws), c_views(ws)
        sources = peer_pointers(ws["a"])

        def launch():
            main = torch.cuda.current_stream()
            sync(rank, arrival_view, gate, _stream(main))
            gemm_stream.wait_stream(main)
            for j in range(world):
                p = (rank + j) % world
                compiled[p](a[p], b_view, cs[p], _stream(gemm_stream), gate_a_flags=gate)
            copy_stream.wait_stream(main)
            for j in range(1, world):
                p = (rank + j) % world
                driver.cuMemcpyDtoDAsync(shared["pulled"][p].data_ptr(), sources[p], shard_bytes,
                                         copy_stream.cuda_stream)
                release(gate, cutlass.Int32(p), _stream(copy_stream))
            main.wait_stream(copy_stream)
            main.wait_stream(gemm_stream)

        return launch

    return [launcher(ws, b_view) for ws, b_view in zip(workspace, b_views)], None


def cake_launchers(rank: int, world: int, shape: dict, workspace: list[dict]):
    """FlashInfer's prepared cake launcher per workspace (``backend="cake"``,
    the weight as the ``w.t()`` view of the ``N x K`` parameter). The prologue
    clears the readiness pad, whose launch epochs a CUDA graph bakes in."""

    if shape["ab"] != shape["c"] or shape["ab"] not in ("bf16", "f16"):
        raise CantImplement("cake runs bf16 -> bf16 and f16 -> f16")
    api = _flashinfer("all_gather_matmul")
    try:
        prepared = [api.prepare_all_gather_matmul(ws["a"], ws["b"].t(), dist.group.WORLD,
                                                  backend="cake", max_rows=shape["m"])
                    for ws in workspace]
    except ValueError as error:
        raise CantImplement(f"cake rejects {shape}: {error}") from error

    def launcher(launch, ws):
        return lambda: launch(ws["a"], out=ws["out"])

    pad = prepared[0].workspace.signal_pad
    return [launcher(launch, ws) for launch, ws in zip(prepared, workspace)], pad.zero_


def cutile_launchers(rank: int, world: int, shape: dict, workspace: list[dict], shared: dict):
    """The body of FlashInfer's ``all_gather_matmul_cutile`` (the ``auto``
    backend on SM100) per workspace: the cuTile barrier, copy-engine pushes of
    each chunk into the peers' scratch with a signal write, the cuTile
    wait-signal matmul with its poll fixed (``_cutile_atomic_poll``), and the signal
    pad's reset. ``shared["scratch"]`` is the symmetric ``world x M x K``
    scratch the origin allocates per call."""

    if shape["ab"] != shape["c"] or shape["ab"] not in ("bf16", "f16"):
        raise CantImplement("the cuTile route is timed for bf16 -> bf16 and f16 -> f16")
    module = _cutile_atomic_poll()
    broadcast = _flashinfer("broadcast_input").broadcast_input
    configs = _flashinfer("configs").Configs
    configs.initialize()
    ct = module.ct
    m, n, k = shape["m"], shape["n"], shape["k"]
    tile_m, tile_n, tile_k, group_m = 128, 256, 64, 19
    if m % tile_m or n % tile_n or k % tile_k:
        raise CantImplement(f"cuTile requires M % {tile_m}, N % {tile_n}, K % {tile_k}")
    group = dist.group.WORLD.group_name
    scratch = shared["scratch"]
    scratch_handle = symm_mem.rendezvous(scratch, group)
    chunk_m = min(m, group_m * tile_m)
    num_chunks = ct.cdiv(m, chunk_m)
    signal_pad = scratch_handle.get_signal_pad(rank, (world, num_chunks), configs.SIGNAL_DTYPE, 0)
    grid = (ct.cdiv(chunk_m, tile_m) * ct.cdiv(n, tile_n),)
    comm_stream = torch.cuda.Stream()

    def launcher(ws):
        handle = symm_mem.rendezvous(ws["a"], group)
        barrier_pads = [handle.get_signal_pad(r, (world,), configs.SIGNAL_DTYPE, 0)
                        for r in range(world)]
        inputs = [ws["a"] if p == rank else scratch[p] for p in range(world)]
        w = ws["b"].t()

        def launch():
            main = torch.cuda.current_stream()
            ct.launch(main, (world,), module.barrier, (barrier_pads, rank))
            comm_stream.wait_stream(main)
            with torch.cuda.stream(comm_stream):
                broadcast(ws["a"], scratch, scratch_handle, chunk_m)
            ct.launch(main, grid, module.wait_signal_matmul_kernel,
                      (inputs, w, ws["out"], signal_pad, rank, world, tile_m, tile_n, tile_k,
                       chunk_m, num_chunks, group_m))
            signal_pad.zero_()
            main.wait_stream(comm_stream)

        return launch

    return [launcher(ws) for ws in workspace], None
