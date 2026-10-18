"""Common utilities for the multimem and peer GPU benchmarks.

Every implementation of a kernel is timed the same way, in the same processes,
on the same buffers: one CUDA graph per implementation replays ``launches``
back-to-back launches that cycle through the workspaces, and the trials of all
implementations are interleaved so drift affects each alike. A rank's time per
launch is the median over trials; the reported time is the slowest rank's.

Run the distributed two-shot kernel's CPU preflight before reserving GPUs:

    python -m benchmarks.multimem.common --world 4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import tempfile
import time
import traceback
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from tirx_harness import numsim

CUTLASS_DIR = Path(os.environ.get("CUTLASS_DIR", "~/src/cutlass")).expanduser()
FLASHINFER_DIR = Path(os.environ.get("FLASHINFER_DIR", "~/src/flashinfer")).expanduser()
CUTLASS_DISTRIBUTED = CUTLASS_DIR / "examples/python/CuTeDSL/cute/blackwell/kernel/distributed"


class CudaArray:
    """A raw device pointer, viewable by torch, for a multicast address."""

    def __init__(self, pointer: int, shape: tuple[int, ...], dtype: Any):
        self.__cuda_array_interface__ = {
            "shape": tuple(shape),
            "typestr": np.dtype(dtype).str,
            "data": (int(pointer), False),
            "version": 3,
        }


# Symmetric-memory tensors live until the worker process exits: under the
# NVSHMEM backend a free is a collective, which ranks dropping their last
# references at different points would deadlock.
_SYMMETRIC: list[Any] = []


def symmetric_unicast(shape: tuple[int, ...], dtype, device):
    """A rendezvoused symmetric-memory tensor (without a multicast view)."""

    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    tensor = symm_mem.empty(shape, dtype=dtype, device=device)
    symm_mem.rendezvous(tensor, dist.group.WORLD.group_name)
    _SYMMETRIC.append(tensor)
    return tensor


def symmetric(shape: tuple[int, ...], dtype, device):
    """A symmetric-memory tensor and the torch view of its multicast address."""

    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    tensor = symm_mem.empty(shape, dtype=dtype, device=device)
    _SYMMETRIC.append(tensor)
    handle = symm_mem.rendezvous(tensor, dist.group.WORLD.group_name)
    # The array interface has no bfloat16 typestr: view the bits as an integer.
    bits = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[tensor.element_size()]
    multicast = torch.as_tensor(CudaArray(handle.multicast_ptr, shape, bits), device=device)
    return tensor, multicast.view(dtype)


class _Eager:
    """Times an implementation that cannot be captured in a CUDA graph by
    enqueueing its launches directly, in segments of at most ``SEGMENT``
    launches. A sleep kernel ahead of each segment holds the device until the
    host has enqueued the whole segment, and only the segment is timed, so the
    host's enqueue time is not measured."""

    SEGMENT = 16

    def __init__(self, closures: list[Callable[[], None]], launches: int):
        self.closures, self.launches = closures, launches
        self.sleep_cycles = 0

    def _enqueue(self, first: int, count: int) -> None:
        for index in range(first, first + count):
            self.closures[index % len(self.closures)]()

    def calibrate(self) -> None:
        """Size the sleep at twice the slowest rank's enqueue time of a segment,
        in cycles of a clock of at most 2 GHz."""

        import torch
        import torch.distributed as dist

        torch.cuda.synchronize()
        start = time.perf_counter()
        self._enqueue(0, min(self.SEGMENT, self.launches))
        host = torch.tensor(time.perf_counter() - start, device="cuda", dtype=torch.float64)
        torch.cuda.synchronize()
        dist.all_reduce(host, op=dist.ReduceOp.MAX)
        self.sleep_cycles = int(float(host) * 2 * 2e9)

    def time(self, replays: int) -> float:
        import torch

        segments = []
        for _ in range(replays):
            for first in range(0, self.launches, self.SEGMENT):
                count = min(self.SEGMENT, self.launches - first)
                torch.cuda._sleep(self.sleep_cycles)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                self._enqueue(first, count)
                end.record()
                segments.append((start, end))
        segments[-1][1].synchronize()
        return sum(start.elapsed_time(end) for start, end in segments) * 1e3 / (replays * self.launches)


def time_launches(impls: dict[str, list[Callable[[], None]]], *, launches: int,
                  trials: int, target_ms: float = 30.0,
                  prologues: dict[str, Callable[[], None]] | None = None,
                  eager: frozenset[str] = frozenset(), idle_s: float = 0.0) -> dict[str, float]:
    """Per-launch microseconds of each implementation on this rank.

    ``impls`` maps a name to one closure per workspace; each closure enqueues
    one launch on the current torch stream. ``prologues`` optionally maps a
    name to a closure enqueued once at the start of that implementation's
    graph, for state a replay must reset (a launch epoch baked into the graph).
    The implementations in ``eager`` are not captured (``_Eager``). With
    ``idle_s``, the device idles that long before every sample, so that a
    power-capped sample's clock does not depend on what the previous sample
    drew.
    """

    import torch
    import torch.distributed as dist
    import tvm_ffi

    stream = torch.cuda.Stream()
    graphs = {}
    with torch.cuda.stream(stream), tvm_ffi.use_torch_stream():
        for closures in impls.values():
            for closure in closures:
                closure()
        torch.cuda.synchronize()
        for name, closures in impls.items():
            if name in eager:
                graphs[name] = _Eager(closures, launches)
                graphs[name].calibrate()
                continue
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                if prologues and name in prologues:
                    prologues[name]()
                for index in range(launches):
                    closures[index % len(closures)]()
            graphs[name] = graph
    torch.cuda.synchronize()

    def run(graph, replays: int) -> float:
        if idle_s:
            torch.cuda.synchronize()
            time.sleep(idle_s)
        dist.barrier()
        torch.cuda.synchronize()
        if isinstance(graph, _Eager):
            with torch.cuda.stream(stream), tvm_ffi.use_torch_stream():
                return graph.time(replays)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1e3 / (replays * launches)

    replays = {}
    for name, graph in graphs.items():
        run(graph, 1)
        estimate = torch.tensor(run(graph, 2), device="cuda", dtype=torch.float64)
        dist.all_reduce(estimate, op=dist.ReduceOp.MAX)
        replays[name] = max(1, int(target_ms * 1e3 / (float(estimate) * launches)))
    samples = {name: [] for name in graphs}
    for _ in range(trials):
        for name, graph in graphs.items():
            samples[name].append(run(graph, replays[name]))
    return {name: statistics.median(values) for name, values in samples.items()}


def slowest_rank(times: dict[str, float]) -> dict[str, float]:
    import torch
    import torch.distributed as dist

    names = sorted(times)
    values = torch.tensor([times[name] for name in names], device="cuda", dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return dict(zip(names, values.tolist()))


def _worker(rank: int, world: int, port: int, entry: str, args: tuple, out: str,
            path: list[str]) -> None:
    sys.path[:] = path
    import torch
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), LOCAL_RANK=str(rank),
                      NCCL_DEBUG="WARN")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
    code = 0
    try:
        module, name = entry.split(":")
        result = getattr(__import__(module, fromlist=[name]), name)(rank, world, *args)
        if rank == 0:
            Path(out).write_text(json.dumps(result))
        torch.cuda.synchronize()
        dist.barrier()
    except BaseException:  # noqa: BLE001 - reported, then the process exits
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Freeing NVSHMEM-backed symmetric memory is a collective barrier, which
    # hangs once the ranks' teardowns diverge; the process exits without one.
    os._exit(code)


def spawn(entry: str, world: int, *args, attempts: int = 3) -> Any:
    """Run ``entry(rank, world, *args)`` on ``world`` GPUs; return rank 0's result."""

    import torch.multiprocessing as mp

    for attempt in range(attempts):
        # The probed port can be taken again before the rendezvous binds it.
        with socket.create_server(("127.0.0.1", 0)) as probe:
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            out = os.path.join(directory, "result.json")
            try:
                mp.spawn(_worker, args=(world, port, entry, args, out, list(sys.path)),
                         nprocs=world)
            except mp.ProcessRaisedException as error:
                if "EADDRINUSE" in str(error) and attempt + 1 < attempts:
                    continue
                raise
            return json.loads(Path(out).read_text())
    raise AssertionError("unreachable")


def table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    """A Markdown table of ``rows`` with ``(key, header, format)`` columns."""

    lines = ["| " + " | ".join(header for _, header, _ in columns) + " |",
             "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(format(row[key], spec) for key, _, spec in columns) + " |"
              for row in rows]
    return "\n".join(lines)


def rank_inputs(
    world: int, *, tiles: int | None = None, seed: int = 0
) -> tuple[list[dict], np.ndarray]:
    """Build the two-shot kernel's NumSim bindings and float64 reference."""

    from ported.cutlass.sm100_all_reduce import TILE
    from tirx_harness.numsim import MulticastWindow

    tiles = world if tiles is None else tiles
    if world <= 0 or tiles <= 0 or tiles % world:
        raise ValueError("tiles must be a positive multiple of world")
    shape = (TILE * tiles, TILE)
    rng = np.random.default_rng(seed)
    sources = [rng.standard_normal(shape).astype(np.float32) for _ in range(world)]
    outputs = [np.zeros(shape, dtype=np.float32) for _ in range(world)]
    flags = [np.zeros(tiles // world, dtype=np.int32) for _ in range(world)]
    input_window = MulticastWindow(sources)
    output_window = MulticastWindow(outputs)
    flag_window = MulticastWindow(flags)
    bindings = [
        {
            "rank": np.int32(rank),
            "data_in": sources[rank],
            "data_out": outputs[rank],
            "in_mc": input_window,
            "out_mc": output_window,
            "flag": flags[rank],
            "flag_mc": flag_window,
        }
        for rank in range(world)
    ]
    reference = sum(
        (source.astype(np.float64) for source in sources), start=np.zeros(shape, np.float64)
    )
    return bindings, reference


def run_preflight(world: int, *, tiles: int | None = None, seed: int = 0) -> numsim.NumSimResult:
    """Run NumSim, Racecheck, and Synccheck for the distributed two-shot kernel."""

    from ported.cutlass.sm100_all_reduce import two_shot_all_reduce
    from tirx_harness import numsim, racecheck, synccheck

    tiles = world if tiles is None else tiles
    kernel = two_shot_all_reduce(tiles, world, 1)
    race_inputs, _ = rank_inputs(world, tiles=tiles, seed=seed)
    racecheck(kernel, race_inputs).require_clean()
    sync_inputs, _ = rank_inputs(world, tiles=tiles, seed=seed)
    synccheck(kernel, sync_inputs).require_clean()

    inputs, reference = rank_inputs(world, tiles=tiles, seed=seed)
    module = numsim.transpile(kernel)
    result = numsim.Engine().run(module, inputs, outputs=("data_out", "flag"))
    for rank in range(world):
        output = result.outputs[numsim.rank_binding_name("data_out", rank)]
        np.testing.assert_allclose(output, reference, rtol=1e-6, atol=1e-6)
        flag = result.outputs[numsim.rank_binding_name("flag", rank)]
        np.testing.assert_array_equal(flag, 0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU preflight for the distributed two-shot NVLS benchmark kernel."
    )
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--tiles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run_preflight(args.world, tiles=args.tiles, seed=args.seed)
    print(f"NumSim/Racecheck/Synccheck passed for {args.world} ranks")
    print(f"outputs: {sorted(result.outputs)}")


if __name__ == "__main__":
    main()
