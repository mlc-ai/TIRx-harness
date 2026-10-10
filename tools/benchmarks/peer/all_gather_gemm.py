"""TIRx ``sm100_all_gather_gemm`` against CUTLASS's
``distributed_all_gather_gemm_blackwell.py`` and FlashInfer's
``comm/all_gather_matmul`` (cake and the SM100 ``auto`` route, cuTile), on the
same symmetric buffers.

    python -m benchmarks.peer.all_gather_gemm tune [--impls cutlass cake cutile tirx] [--shapes NAME ...]
    python -m benchmarks.peer.all_gather_gemm compare [--shapes NAME ...]

``tune`` checks and times the configurations of each implementation on every
shape and appends one JSON line per candidate to ``--log`` (resuming from it; a
candidate that hangs or faults is recorded as failed). CUTLASS's configuration
space is what its example's kernel takes (MMA tile, cluster, 1- or 2-CTA, the
epilogue store); cake and cuTile have one configuration each (their own
heuristics'); TIRx searches CUTLASS's space with its scheduler's raster order
and swizzle and its host schedule's chunk size and copy direction (``tune``).
``compare`` times TIRx's ``TOP`` and each origin's ``ORIGIN_TOP`` fastest
correct configurations interleaved, with the device idle before every sample,
and reports TIRx's best against the fastest origin.

Every rank holds its ``M x K`` shard ``a`` (symmetric memory) and the ``N x K``
weight ``b``; every implementation leaves ``out = all_gather(a) @ b.T``
(``world * M x N``, rank ``p``'s rows at ``[p * M, (p + 1) * M)``) on each rank.
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import json
from pathlib import Path
import time
import traceback

from benchmarks.multimem import common

# The origins' own shapes (M is rows per rank): the CUTLASS example's default
# (``--mnkl 256,256,512,1``, TF32 -> f32, M split over the ranks) and its
# docstring run (8192^3 f16); FlashInfer's benchmark (``run_benchmark``: K 8192,
# N 2048, 1024 to 65536 rows per rank), the extra row count of its correctness
# run (19456) and the cake end-to-end test's 384 rows. FlashInfer's tests run
# bf16 and f16; the benchmark shapes are bf16.
ORIGIN_SHAPES = [
    {"name": "cutlass_default", "m": 64, "n": 256, "k": 512, "ab": "tf32", "c": "f32"},
    {"name": "cutlass_doc", "m": 2048, "n": 8192, "k": 8192, "ab": "f16", "c": "f16"},
    *[{"name": f"flashinfer_m{m}", "m": m, "n": 2048, "k": 8192, "ab": "bf16", "c": "bf16"}
      for m in (1024, 2048, 4096, 8192, 16384, 19456, 32768, 65536)],
    {"name": "cake_e2e_m384", "m": 384, "n": 2048, "k": 8192, "ab": "bf16", "c": "bf16"},
]
# Tensor-parallel (TP = 4) column-parallel projections, whose sequence-parallel
# inputs are all-gathered: Llama-3-70B (hidden 8192, 64 query and 8 KV heads of
# 128, intermediate 28672) QKV (N = 10240 / 4) and fused gate/up (N = 57344 / 4)
# projections at 512 to 8192 tokens per rank, bf16 -> bf16.
LLM_PROJECTIONS = [("70b_qkv", 2560), ("70b_gate_up", 14336)]
LLM_TOKENS = [512, 2048, 8192]
SWEEP_SHAPES = [
    {"name": f"llama{name}_m{m}", "m": m, "n": n, "k": 8192, "ab": "bf16", "c": "bf16"}
    for name, n in LLM_PROJECTIONS for m in LLM_TOKENS
]
SHAPES = ORIGIN_SHAPES + SWEEP_SHAPES
SHAPES_BY_NAME = {shape["name"]: shape for shape in SHAPES}

ORIGINS = ["cutlass", "cake", "cutile"]
IMPLS = [*ORIGINS, "tirx"]
# cuTile kernels take lists of tensors, which a CUDA graph cannot capture.
EAGER = frozenset({"cutile"})

# The CUTLASS example's configuration space: its kernel takes the MMA tile, the
# cluster shape, 1- or 2-CTA instructions and the epilogue's store, and builds
# its tile scheduler with the default raster order (along M) and swizzle (1).
TILERS_2CTA = [(256, 256), (256, 192), (256, 128), (256, 64),
               (128, 256), (128, 192), (128, 128), (128, 64)]
TILERS_1CTA = [(128, 256), (128, 192), (128, 128), (128, 64),
               (64, 256), (64, 192), (64, 128), (64, 64)]
CLUSTERS = [(1, 1), (1, 2), (2, 1), (2, 2)]
RASTERS = ["m", "n"]
SWIZZLES = [1, 2, 4, 8]
COPIES = ["pull", "push"]
# Geometries refined after the first pass, and schedules after the second.
REFINE_TOP, SCHEDULE_TOP = 5, 3

TORCH_TYPES = {"tf32": "float32", "f16": "float16", "bf16": "bfloat16", "f32": "float32"}
RESULTS = Path(__file__).parent / "results"
# Operand sets cycled through by the timed launches, fewer for large shapes so
# the symmetric heap holds them.
WORKSPACES, WORKSPACE_BYTES = 10, 1 << 30
MIN_CHUNK_ROWS = 64
TUNE_LAUNCHES, TUNE_TRIALS, TUNE_MS = 20, 3, 10.0
LAUNCHES, TRIALS = 40, 15
# The heaviest GEMMs reach the 1200 W power cap within a sample, so the compare
# idles the device before every sample.
IDLE_S = 0.5
# Configurations compared per implementation. The tune times without the idle,
# so an origin's fastest configuration under the compare's timer may rank
# below its third in the tune.
TOP, ORIGIN_TOP = 3, 8
CANDIDATE_TIMEOUT_S = 300


def geometries(store: bool | None = True) -> list[dict]:
    """CUTLASS's geometries; ``store=None`` gives both epilogue stores."""

    stores = [True, False] if store is None else [store]
    rows = []
    for use_tma_store in stores:
        for use_2cta, tilers in ((True, TILERS_2CTA), (False, TILERS_1CTA)):
            for cluster in CLUSTERS:
                if use_2cta and cluster[0] % 2:
                    continue
                for tiler in tilers:
                    rows.append({"use_2cta": use_2cta, "mma_tiler": list(tiler),
                                 "cluster": list(cluster), "use_tma_store": use_tma_store})
    return rows


def tiles(shape: dict, config: dict) -> bool:
    """TIRx's tiling constraint: every chunk is whole cluster tiles of a shard
    and the swizzle divides the swizzled cluster count."""

    cta_m = config["mma_tiler"][0] // (2 if config["use_2cta"] else 1)
    cta_n = config["mma_tiler"][1]
    chunk_rows = config.get("chunk_rows", shape["m"])
    cm, cn = config["cluster"]
    if shape["m"] % chunk_rows or chunk_rows % (cta_m * cm) or shape["n"] % (cta_n * cn):
        return False
    swizzled = (shape["n"] // (cta_n * cn) if config.get("raster", "m") == "m"
                else chunk_rows // (cta_m * cm))
    return swizzled % config.get("swizzle", 1) == 0


def chunk_choices(shape: dict) -> list[int]:
    """Whole shards, halves and quarters, and cake's chunk (19 row blocks of 128)."""

    m = shape["m"]
    rows = {m, m // 2, m // 4, 2432}
    return sorted(r for r in rows if r >= MIN_CHUNK_ROWS and m % r == 0)


def config_key(config: dict) -> str:
    return json.dumps(config, sort_keys=True)


def _torch_type(name: str):
    import torch

    return getattr(torch, TORCH_TYPES[name])


def workspace_count(shape: dict, world: int) -> int:
    element = 4 if shape["ab"] == "tf32" else 2
    per_set = shape["m"] * shape["k"] * element + world * shape["m"] * shape["n"] * element
    return max(2, min(WORKSPACES, WORKSPACE_BYTES // per_set))


def allocate(rank: int, world: int, shape: dict, device) -> tuple[list[dict], dict, object]:
    """Workspaces (one value set, copied) of ``a`` (symmetric), ``b`` and
    ``out``; the buffers every implementation's schedule needs in ``shared``;
    and the reference. Operands are small integers, so every product and the
    f32 accumulation are exact and each kernel's rounded result is bitwise
    the reference's."""

    import torch
    import torch.distributed as dist

    m, n, k = shape["m"], shape["n"], shape["k"]
    ab, c = _torch_type(shape["ab"]), _torch_type(shape["c"])
    generator = torch.Generator(device=device).manual_seed(1000 + rank)
    a = torch.randint(-2, 3, (m, k), generator=generator, device=device).to(ab)
    b = torch.randint(-2, 3, (n, k), generator=generator, device=device).to(ab)
    workspace = []
    for _ in range(workspace_count(shape, world)):
        ws_a = common.symmetric_unicast((m, k), ab, device)
        ws_a.copy_(a)
        workspace.append({"a": ws_a, "b": b.clone(),
                          "out": torch.empty(world * m, n, dtype=c, device=device)})
    shared = {
        "pulled": [None if p == rank else torch.empty(m, k, dtype=ab, device=device)
                   for p in range(world)],
        "gate": torch.zeros(world, dtype=torch.int32, device=device),
        "arrival": common.symmetric_unicast((1,), torch.int32, device).zero_(),
        "scratch": common.symmetric_unicast((world, m, k), ab, device),
        "flags": common.symmetric_unicast(
            (world * max(1, m // MIN_CHUNK_ROWS),), torch.uint32, device).zero_(),
    }
    gathered = torch.empty(world * m, k, dtype=ab, device=device)
    dist.all_gather_into_tensor(gathered, a)
    reference = torch.empty(world * m, n, dtype=c, device=device)
    rows = 16384
    for begin in range(0, world * m, rows):
        reference[begin:begin + rows] = (gathered[begin:begin + rows].float() @ b.float().T).to(c)
    del gathered
    torch.cuda.synchronize()
    dist.barrier()
    return workspace, shared, reference


def tirx_launchers(rank: int, world: int, shape: dict, config: dict,
                   workspace: list[dict], shared: dict):
    """One launch closure per workspace of the TIRx port built with ``config``
    (a CUTLASS GEMM configuration plus ``chunk_rows`` and ``copy``).

    The launch runs ``barrier_kernel`` (which clears this rank's flags), then
    on a copy stream every remote shard chunk by chunk -- step ``j``'s shard
    first, as the GEMM consumes them -- each chunk a copy-engine copy followed
    by a stream write of its flag: ``copy="pull"`` copies rank ``rank + j``'s
    chunk from its ``a`` into this rank's scratch and flags it here (CUTLASS's
    direction); ``copy="push"`` copies this rank's chunk into rank
    ``rank - j``'s scratch and flags it there (FlashInfer cake's). The GEMM runs
    on the main stream meanwhile."""

    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    import tvm_ffi
    from tirx_kernels.runner import compile_kernel

    from benchmarks.multimem.origins import max_active_clusters
    from benchmarks.peer.origins import peer_pointers
    from ported.cutlass.sm100_all_gather_gemm import all_gather_gemm, barrier_kernel

    m, k = shape["m"], shape["k"]
    cluster = tuple(config["cluster"])
    chunk_rows = config.get("chunk_rows", m)
    chunks = m // chunk_rows
    gemm = all_gather_gemm(
        m, shape["n"], k, shape["ab"], shape["c"], rank, world, use_2cta=config["use_2cta"],
        mma_tiler=tuple(config["mma_tiler"]), cluster=cluster,
        use_tma_store=config["use_tma_store"], raster=config.get("raster", "m"),
        swizzle=config.get("swizzle", 1), chunk_rows=chunk_rows,
        max_active_clusters=max_active_clusters(cluster[0] * cluster[1]))
    # nvcc rather than TVM's default NVRTC: NVRTC 13.0 leaves the high half of
    # the flag poll's memory descriptor unset in some configurations, and the
    # poll then faults whenever an earlier kernel left that register nonzero.
    gemm_exe = compile_kernel(gemm.func, cuda_compile_mode="nvcc")
    barrier_exe = compile_kernel(barrier_kernel(rank, world, world * chunks).func,
                                 cuda_compile_mode="nvcc")

    group = dist.group.WORLD.group_name
    flags = shared["flags"][:world * chunks]
    flag_handle = symm_mem.rendezvous(shared["flags"], group)
    peer_flags = [flag_handle.get_remote_tensor(p, shared["flags"].shape, torch.uint32)
                  for p in range(world)]
    scratch = shared["scratch"]
    scratch_handle = symm_mem.rendezvous(scratch, group)
    peer_scratch = [scratch_handle.get_remote_tensor(p, scratch.shape, scratch.dtype)
                    for p in range(world)]
    counter = shared["arrival"]
    counter_offsets = [pointer - counter.data_ptr() for pointer in peer_pointers(counter)]
    copy_stream = torch.cuda.Stream()

    def as_bytes(tensor):
        return tensor.view(torch.uint8).reshape(-1)

    def launcher(ws):
        handle = symm_mem.rendezvous(ws["a"], group)
        peer_a = [handle.get_remote_tensor(p, ws["a"].shape, ws["a"].dtype) for p in range(world)]
        args = [as_bytes(ws["a"]), as_bytes(scratch), as_bytes(ws["b"]), as_bytes(ws["out"]),
                flags]
        copies = []
        for j in range(1, world):
            for chunk in range(chunks):
                rows = slice(chunk * chunk_rows, (chunk + 1) * chunk_rows)
                if config["copy"] == "pull":
                    p = (rank + j) % world
                    copies.append((scratch[p, rows], peer_a[p][rows], flags,
                                   p * chunks + chunk))
                else:
                    q = (rank - j) % world
                    copies.append((peer_scratch[q][rank, rows], ws["a"][rows], peer_flags[q],
                                   rank * chunks + chunk))

        def launch():
            main = torch.cuda.current_stream()
            with tvm_ffi.use_torch_stream():
                barrier_exe(flags, counter, *counter_offsets)
            copy_stream.wait_stream(main)
            with torch.cuda.stream(copy_stream):
                for dst, src, flag, index in copies:
                    dst.copy_(src, non_blocking=True)
                    torch.ops.symm_mem.stream_write_value32_(flag, index, 1)
            with tvm_ffi.use_torch_stream():
                gemm_exe(*args)
            main.wait_stream(copy_stream)

        return launch

    return [launcher(ws) for ws in workspace]


def _collect_garbage() -> None:
    """Free unreachable CuTeDSL and TVM modules while the device is idle.

    The workers run with the cyclic collector disabled: a collected module
    unloads its library, and the unload waits for the device to idle. Inside a
    CUTLASS launch, while a gated GEMM spins on a flag the host has not yet
    enqueued the release of, that deadlocks; during a graph capture it
    invalidates the capture."""

    import torch

    torch.cuda.synchronize()
    gc.collect()


def run_checked(launch, workspace: dict, reference) -> str | None:
    """Run one launch on a NaN-filled output; return a mismatch message or None.
    The result must equal the reference bitwise on every rank."""

    import torch
    import torch.distributed as dist

    out = workspace["out"]
    out.fill_(float("nan"))
    torch.cuda.synchronize()
    dist.barrier()
    launch()
    torch.cuda.synchronize()
    dist.barrier()
    error = None
    if not torch.equal(out, reference):
        bad = (out != reference).nonzero()
        error = f"{bad.shape[0]} mismatches, first at {bad[0].tolist()}"
    failed = torch.tensor([error is not None], device="cuda", dtype=torch.int32)
    dist.all_reduce(failed)
    if int(failed) and error is None:
        error = "another rank mismatched"
    return error


def launchers(impl: str, rank: int, world: int, shape: dict, config: dict,
              workspace: list[dict], shared: dict):
    """``(closures, prologue)`` of one implementation and configuration."""

    from benchmarks.peer import origins

    if impl == "tirx":
        return tirx_launchers(rank, world, shape, config, workspace, shared), None
    if impl == "cutlass":
        return origins.cutlass_launchers(rank, world, shape, config, workspace, shared)
    if impl == "cake":
        return origins.cake_launchers(rank, world, shape, workspace)
    if impl == "cutile":
        return origins.cutile_launchers(rank, world, shape, workspace, shared)
    raise ValueError(impl)


def _time(named: dict[str, tuple], launches: int, trials: int, target_ms: float = 30.0,
          idle_s: float = 0.0):
    """Slowest-rank microseconds per launch of ``{name: (impl, closures, prologue)}``."""

    impls = {name: closures for name, (_, closures, _) in named.items()}
    prologues = {name: prologue for name, (_, _, prologue) in named.items() if prologue}
    eager = frozenset(name for name, (impl, _, _) in named.items() if impl in EAGER)
    return common.slowest_rank(common.time_launches(
        impls, launches=launches, trials=trials, target_ms=target_ms, prologues=prologues,
        eager=eager, idle_s=idle_s))


# ---------------------------------------------------------------------------
# Tuning.


def _log_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tune_rank(rank: int, world: int, impl: str, shape_name: str, log: str,
              todo: list[dict]) -> None:
    import torch
    import torch.distributed._symmetric_memory as symm_mem

    from benchmarks.multimem.origins import CantImplement

    gc.disable()
    symm_mem.set_backend("NVSHMEM")
    shape = SHAPES_BY_NAME[shape_name]
    workspace, shared, reference = allocate(rank, world, shape, torch.device("cuda", rank))

    def record(**fields):
        if rank == 0:
            with open(log, "a") as f:
                f.write(json.dumps({"impl": impl, "shape": shape_name, **fields,
                                    "time": round(time.time(), 1)}) + "\n")

    for config in todo:
        key = config_key(config)
        record(config=key, status="started")
        faulthandler.dump_traceback_later(CANDIDATE_TIMEOUT_S, exit=True)
        try:
            closures, prologue = launchers(impl, rank, world, shape, config, workspace, shared)
        except CantImplement as error:
            record(config=key, status="rejected", message=str(error)[:200])
            continue
        except Exception as error:  # noqa: BLE001 - every rank builds the same kernel
            record(config=key, status="compile_error", message=repr(error)[:400])
            continue
        finally:
            faulthandler.cancel_dump_traceback_later()
            _collect_garbage()
        faulthandler.dump_traceback_later(CANDIDATE_TIMEOUT_S, exit=True)
        error = run_checked(closures[0], workspace[0], reference)
        if error is not None:
            record(config=key, status="wrong", message=error)
            faulthandler.cancel_dump_traceback_later()
            continue
        times = _time({"x": (impl, closures, prologue)}, TUNE_LAUNCHES, TUNE_TRIALS, TUNE_MS)
        faulthandler.cancel_dump_traceback_later()
        record(config=key, status="ok", us=times["x"])
        del closures


def _tune_configs(impl: str, shape_name: str, world: int, log: Path,
                  configs: list[dict]) -> None:
    """Time every one of ``configs`` not yet in the log, surviving hangs and faults."""

    while True:
        done = {r["config"] for r in _log_lines(log)
                if r["impl"] == impl and r["shape"] == shape_name and r["status"] != "started"}
        todo = [c for c in configs if config_key(c) not in done]
        if not todo:
            return
        try:
            common.spawn("benchmarks.peer.all_gather_gemm:tune_rank", world,
                         impl, shape_name, str(log), todo)
            return
        except Exception:  # noqa: BLE001 - a hung or faulted candidate
            traceback.print_exc()
            records = [r for r in _log_lines(log) if r["impl"] == impl and r["shape"] == shape_name]
            finished = {r["config"] for r in records if r["status"] != "started"}
            stuck = [r["config"] for r in records
                     if r["status"] == "started" and r["config"] not in finished]
            if not stuck:
                raise
            with open(log, "a") as f:
                for key in stuck:
                    f.write(json.dumps({"impl": impl, "shape": shape_name, "config": key,
                                        "status": "failed"}) + "\n")


def best_configs(log: Path, impl: str, shape_name: str, top: int = TOP) -> list[dict]:
    ok = sorted((r for r in _log_lines(log) if r["impl"] == impl
                 and r["shape"] == shape_name and r["status"] == "ok"), key=lambda r: r["us"])
    return [json.loads(r["config"]) for r in ok[:top]]


def _best_geometries(log: Path, impl: str, shape_name: str, top: int) -> list[dict]:
    """The ``top`` fastest distinct (2-CTA, MMA tile, cluster) geometries."""

    seen = []
    for config in best_configs(log, impl, shape_name, top=10 ** 6):
        geometry = {key: config[key] for key in ("use_2cta", "mma_tiler", "cluster")}
        if geometry not in seen:
            seen.append(geometry)
    return seen[:top]


def tirx_config(shape: dict, geometry: dict, **schedule) -> dict:
    """A TIRx configuration with every schedule knob explicit (one log key each)."""

    config = {"use_tma_store": True, "copy": "pull", "chunk_rows": shape["m"], "raster": "m",
              "swizzle": 1, **geometry}
    config.update(schedule)
    return config


def tune(impls: list[str], shapes: list[str], world: int, log: Path) -> None:
    """cake and cuTile run their one configuration. CUTLASS times every
    geometry with the TMA-store epilogue, then its ``REFINE_TOP`` fastest
    geometries with the direct store. TIRx starts the same way (pulling whole
    shards), then searches the epilogue store, copy direction and chunk size of
    its ``REFINE_TOP`` fastest geometries, then the raster order and swizzle of
    its ``SCHEDULE_TOP`` fastest configurations."""

    log.parent.mkdir(parents=True, exist_ok=True)
    for shape_name in shapes:
        shape = SHAPES_BY_NAME[shape_name]
        for impl in impls:
            if impl in ("cake", "cutile"):
                _tune_configs(impl, shape_name, world, log, [{}])
            elif impl == "cutlass":
                _tune_configs(impl, shape_name, world, log, geometries(True))
                _tune_configs(impl, shape_name, world, log, [
                    {**geometry, "use_tma_store": False}
                    for geometry in _best_geometries(log, impl, shape_name, REFINE_TOP)])
            else:
                first = [tirx_config(shape, geometry) for geometry in geometries(True)]
                _tune_configs(impl, shape_name, world, log,
                              [c for c in first if tiles(shape, c)])
                second = [tirx_config(shape, geometry, use_tma_store=store, copy=copy,
                                      chunk_rows=rows)
                          for geometry in _best_geometries(log, impl, shape_name, REFINE_TOP)
                          for store in (True, False) for copy in COPIES
                          for rows in chunk_choices(shape)]
                _tune_configs(impl, shape_name, world, log,
                              [c for c in second if tiles(shape, c)])
                third = [{**config, "raster": raster, "swizzle": swizzle}
                         for config in best_configs(log, impl, shape_name, SCHEDULE_TOP)
                         for raster in RASTERS for swizzle in SWIZZLES]
                _tune_configs(impl, shape_name, world, log,
                              [c for c in third if tiles(shape, c)])
            ok = sorted((r for r in _log_lines(log) if r["impl"] == impl
                         and r["shape"] == shape_name and r["status"] == "ok"),
                        key=lambda r: r["us"])
            best = f"{ok[0]['us']:.2f} us {ok[0]['config']}" if ok else "none"
            print(f"[tune] {shape_name} {impl}: {len(ok)} ok, best {best}", flush=True)


# ---------------------------------------------------------------------------
# TIRx against the origins.


def compare_rank(rank: int, world: int, shape_name: str, entries: list[dict]) -> list[dict]:
    """Check each ``{"impl", "config"}`` entry and time all of them interleaved
    on the same buffers."""

    import torch
    import torch.distributed._symmetric_memory as symm_mem

    gc.disable()
    symm_mem.set_backend("NVSHMEM")
    shape = SHAPES_BY_NAME[shape_name]
    workspace, shared, reference = allocate(rank, world, shape, torch.device("cuda", rank))
    named, rows = {}, []
    faulthandler.dump_traceback_later(CANDIDATE_TIMEOUT_S * len(entries), exit=True)
    for index, entry in enumerate(entries):
        closures, prologue = launchers(entry["impl"], rank, world, shape, entry["config"],
                                       workspace, shared)
        _collect_garbage()
        error = run_checked(closures[0], workspace[0], reference)
        rows.append({"shape": shape_name, "impl": entry["impl"],
                     "config": config_key(entry["config"]), "error": error})
        if error is None:
            named[str(index)] = (entry["impl"], closures, prologue)
    times = _time(named, LAUNCHES, TRIALS, idle_s=IDLE_S)
    faulthandler.cancel_dump_traceback_later()
    for index, row in enumerate(rows):
        if str(index) in times:
            row["us"] = times[str(index)]
        if rank == 0:
            print(json.dumps(row), flush=True)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Per shape, from its latest compare run: the fastest origin's best time
    against TIRx's best, and whether every entry was correct."""

    runs = {}
    for row in rows:
        runs.setdefault(row["shape"], {}).setdefault(row["run"], []).append(row)
    summary = []
    for shape in SHAPES:
        if shape["name"] not in runs:
            continue
        group = runs[shape["name"]][max(runs[shape["name"]])]
        timed = [r for r in group if "us" in r]
        origin = [r for r in timed if r["impl"] in ORIGINS]
        tirx = [r for r in timed if r["impl"] == "tirx"]
        row = {"shape": shape["name"], "M": shape["m"], "N": shape["n"], "K": shape["k"],
               "types": f"{shape['ab']}->{shape['c']}", "entries": len(group),
               "correct": all(r["error"] is None for r in group),
               "origin_impls": sorted({r["impl"] for r in origin})}
        for impl in ORIGINS:
            mine = [r["us"] for r in origin if r["impl"] == impl]
            if mine:
                row[f"{impl}_us"] = min(mine)
        if origin and tirx:
            best_origin = min(origin, key=lambda r: r["us"])
            best_tirx = min(tirx, key=lambda r: r["us"])
            row.update(origin=best_origin["impl"], origin_us=best_origin["us"],
                       origin_config=best_origin["config"], tirx_us=best_tirx["us"],
                       tirx_config=best_tirx["config"],
                       ratio=best_tirx["us"] / best_origin["us"])
        summary.append(row)
    return summary


def compare(impls: list[str], shapes: list[str], world: int, log: Path, out: Path,
            top: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    for shape_name in shapes:
        entries = [{"impl": impl, "config": config} for impl in impls
                   for config in best_configs(log, impl, shape_name,
                                              top if impl == "tirx" else ORIGIN_TOP)]
        if not entries:
            print(f"[compare] {shape_name}: no tuned configurations", flush=True)
            continue
        rows = common.spawn("benchmarks.peer.all_gather_gemm:compare_rank", world,
                            shape_name, entries)
        run = round(time.time(), 1)
        with open(out, "a") as f:
            for row in rows:
                f.write(json.dumps({**row, "run": run}) + "\n")
    summary = summarize(_log_lines(out))
    print(common.table([r for r in summary if "ratio" in r], [
        ("shape", "shape", "s"), ("types", "types", "s"), ("origin", "fastest origin", "s"),
        ("origin_us", "origin (us)", ".2f"), ("tirx_us", "TIRx (us)", ".2f"),
        ("ratio", "TIRx / origin", ".3f")]))
    for row in summary:
        if not row["correct"]:
            print(f"[compare] {row['shape']}: an entry was wrong")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("tune", "compare"):
        mode = sub.add_parser(name)
        mode.add_argument("--impls", nargs="*", default=IMPLS)
        mode.add_argument("--world", type=int, default=4)
        mode.add_argument("--shapes", nargs="*", default=[s["name"] for s in SHAPES])
        mode.add_argument("--log", type=Path, default=RESULTS / "all_gather_gemm_tuning.jsonl")
    sub.choices["compare"].add_argument("--top", type=int, default=TOP)
    sub.choices["compare"].add_argument(
        "--out", type=Path, default=RESULTS / "all_gather_gemm_gb200x4.jsonl")
    args = parser.parse_args()
    if args.mode == "tune":
        tune(args.impls, args.shapes, args.world, args.log)
    else:
        compare(args.impls, args.shapes, args.world, args.log, args.out, args.top)


if __name__ == "__main__":
    main()
