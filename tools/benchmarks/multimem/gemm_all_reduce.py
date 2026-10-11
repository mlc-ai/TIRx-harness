"""TIRx ``gemm_all_reduce`` against the GEMM + two-shot all-reduce of CUTLASS's
``distributed_gemm_all_reduce_blackwell.py`` (LDMCxSTMC) and FlashInfer's
``cute_dsl/gemm_allreduce_two_shot.py``, on the same symmetric buffers.

    python -m benchmarks.multimem.gemm_all_reduce tune [--origins cutlass flashinfer] [--shapes NAME ...]
    python -m benchmarks.multimem.gemm_all_reduce compare [--shapes NAME ...]

``tune`` times every configuration each origin accepts on every shape and
appends one JSON line per candidate to ``--log``; it resumes from the log, and a
candidate that hangs or faults the GPU is recorded as failed. ``compare`` takes
each origin's ``TOP`` fastest correct configurations, times them interleaved with
TIRx on the same configurations, and reports each origin's best against TIRx's
best.

C = A @ B^T with K-major A (M x K) and B (N x K) and row-major C; every rank holds
its own A and B, and the kernel leaves sum over ranks of C on every rank.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
from pathlib import Path
import time
import traceback

from benchmarks.multimem import common

# The origins' own shapes: the CUTLASS example's default (TF32 -> f32) and its
# docstring benchmark (f16 -> f16), and FlashInfer's test (TF32 -> f32).
ORIGIN_SHAPES = [
    {"name": "cutlass_default", "m": 256, "n": 256, "k": 512, "ab": "tf32", "c": "f32"},
    {"name": "flashinfer_test", "m": 2048, "n": 2048, "k": 4096, "ab": "tf32", "c": "f32"},
    {"name": "cutlass_doc", "m": 8192, "n": 8192, "k": 8192, "ab": "f16", "c": "f16"},
]
# Tensor-parallel (TP = 4) row-parallel projections, whose partial outputs are
# all-reduced: (hidden, per-rank K) of Llama-3-8B (hidden 4096, intermediate 14336)
# and Llama-3-70B (hidden 8192, intermediate 28672) attention output and MLP down
# projections, at 128 to 8192 tokens, bf16 -> bf16.
LLM_PROJECTIONS = [("8b_o", 4096, 1024), ("8b_down", 4096, 3584),
                   ("70b_o", 8192, 2048), ("70b_down", 8192, 7168)]
LLM_TOKENS = [128, 512, 2048, 8192]
SWEEP_SHAPES = [
    {"name": f"llama{name}_m{m}", "m": m, "n": n, "k": k, "ab": "bf16", "c": "bf16"}
    for name, n, k in LLM_PROJECTIONS for m in LLM_TOKENS
]
SHAPES = ORIGIN_SHAPES + SWEEP_SHAPES
SHAPES_BY_NAME = {shape["name"]: shape for shape in SHAPES}

# The configuration space of the CUTLASS example's ``--benchmark_or_test
# benchmark_all`` (with ``--use_tma_store``, as its docstring runs it); FlashInfer's
# kernel has no raster or swizzle knob but can store C with or without TMA.
TILERS_2CTA = [(256, 256), (256, 192), (256, 128), (256, 64),
               (128, 256), (128, 192), (128, 128), (128, 64)]
TILERS_1CTA = [(128, 256), (128, 192), (128, 128), (128, 64),
               (64, 256), (64, 192), (64, 128), (64, 64)]
CLUSTERS = [(1, 1), (1, 2), (2, 1), (2, 2)]
RASTERS = ["m", "n"]
SWIZZLES = [1, 2, 4, 8]
WORKSPACES = 10
TUNE_LAUNCHES, TUNE_TRIALS, TUNE_MS = 20, 3, 10.0
LAUNCHES, TRIALS = 100, 9
TOP = 3
REFINE_TOP = 5
CANDIDATE_TIMEOUT_S = 180
TORCH_TYPES = {"tf32": "float32", "f16": "float16", "bf16": "bfloat16", "f32": "float32"}
RESULTS = Path(__file__).parent / "results"


def _geometry(config: dict) -> list[dict]:
    rows = []
    for use_2cta, tilers in ((True, TILERS_2CTA), (False, TILERS_1CTA)):
        for cluster in CLUSTERS:
            if use_2cta and cluster[0] % 2:
                continue
            for tiler in tilers:
                rows.append({**config, "use_2cta": use_2cta, "cluster": list(cluster),
                             "mma_tiler": list(tiler)})
    return rows


def full_tiles(shape: dict, config: dict, world: int) -> bool:
    """Every cluster tile lies inside C and splits into whole per-rank slabs:
    the origins' all-reduce warps neither predicate a ragged tile nor a slab
    that is not a whole number of rows."""

    cta_m = config["mma_tiler"][0] // (2 if config["use_2cta"] else 1)
    cta_n = config["mma_tiler"][1]
    clusters_m = shape["m"] // (cta_m * config["cluster"][0])
    clusters_n = shape["n"] // (cta_n * config["cluster"][1])
    if shape["m"] % (cta_m * config["cluster"][0]) or shape["n"] % (cta_n * config["cluster"][1]):
        return False
    if cta_m % world:
        return False
    swizzle = config.get("swizzle", 1)
    swizzled = clusters_n if config.get("raster", "m") == "m" else clusters_m
    return swizzled % swizzle == 0


def _schedules(geometry: dict) -> list[dict]:
    return [{**geometry, "raster": raster, "swizzle": swizzle}
            for raster in RASTERS for swizzle in SWIZZLES]


def candidates(origin: str, shape: dict, world: int, *, geometry_only: bool = False) -> list[dict]:
    """The origin's configurations that tile ``shape``. With ``geometry_only``
    CUTLASS's raster order and swizzle stay at their defaults (m, 1)."""

    if origin == "cutlass":
        geometries = _geometry({"use_tma_store": True})
        if geometry_only:
            space = [{**geometry, "raster": "m", "swizzle": 1} for geometry in geometries]
        else:
            space = [config for geometry in geometries for config in _schedules(geometry)]
    elif origin == "flashinfer":
        space = [geometry for store in (True, False)
                 for geometry in _geometry({"use_tma_store": store})]
    else:
        raise ValueError(origin)
    return [config for config in space if full_tiles(shape, config, world)]


def config_key(config: dict) -> str:
    return json.dumps(config, sort_keys=True)


# ---------------------------------------------------------------------------
# Buffers and correctness.


def _torch_type(name: str):
    import torch

    return getattr(torch, TORCH_TYPES[name])


def allocate(rank: int, world: int, shape: dict, device) -> tuple[list[dict], dict, object]:
    """Workspaces of A and B (one value set, copied ``WORKSPACES`` times, as the
    CUTLASS benchmark does), the shared symmetric C, out and flags, and the
    float64 reference: the exact sum over ranks of each rank's C rounded to C's
    type. Operands are small integers, so the products and the f32 accumulation
    are exact."""

    import torch
    import torch.distributed as dist

    m, n, k = shape["m"], shape["n"], shape["k"]
    generator = torch.Generator(device=device).manual_seed(1000 + rank)
    ab, c = _torch_type(shape["ab"]), _torch_type(shape["c"])
    a = torch.randint(-2, 3, (m, k), generator=generator, device=device).to(ab)
    b = torch.randint(-2, 3, (n, k), generator=generator, device=device).to(ab)
    workspace = [{"a": a.clone(), "b": b.clone()} for _ in range(WORKSPACES)]
    shared = {}
    shared["c"], shared["c_mc"] = common.symmetric((m, n), c, device)
    shared["out"], shared["out_mc"] = common.symmetric((m, n), c, device)
    # The CUTLASS benchmark's sizing: room for 64 x 64 tiles plus per-SM slots.
    shared["flag"], shared["flag_mc"] = common.symmetric(
        ((m // 64) * (n // 64) + 160,), torch.int32, device)
    shared["flag"].zero_()
    reference = (a.double() @ b.double().T).to(c).double()
    dist.all_reduce(reference)
    return workspace, shared, reference


def rounding_error(output, reference):
    """``|output - reference|`` in units of the output type's ulp at ``reference``.

    The NVLS reduction of 16-bit operands (``ld_reduce ... .acc::f32``) is
    faithfully but not correctly rounded: on GB200 bf16 ties go to the odd
    neighbor and a few non-ties round the far way (0.75 ulp), identically for
    every kernel issuing it. So a correct 16-bit result is within one ulp of the
    exact sum, not equal to torch's rounding of it.
    """

    import torch

    precision = {torch.float32: 24, torch.float16: 11, torch.bfloat16: 8}[output.dtype]
    _, exponent = torch.frexp(reference)
    ulp = torch.ldexp(torch.ones_like(reference), exponent - precision)
    return (output.double() - reference).abs() / ulp


def tirx_launchers(rank: int, world: int, shape: dict, config: dict, protocol: str,
                   workspace: list[dict], shared: dict):
    """One launch closure per workspace of TIRx ``gemm_all_reduce`` built with
    ``config`` and the ``protocol`` (and data placement) of that origin."""

    import torch
    from tirx_kernels.runner import compile_kernel

    from benchmarks.multimem import origins
    from tests.numsim.support.multimem_gemm_all_reduce import gemm_all_reduce

    cluster = config["cluster"]
    kernel = gemm_all_reduce(
        shape["m"], shape["n"], shape["k"], shape["ab"], shape["c"], rank, world,
        use_2cta=config["use_2cta"], mma_tiler=tuple(config["mma_tiler"]),
        cluster=tuple(cluster), use_tma_store=config["use_tma_store"],
        raster=config.get("raster", "m"), swizzle=config.get("swizzle", 1), protocol=protocol,
        max_active_clusters=origins.max_active_clusters(cluster[0] * cluster[1]),
        flag_len=shared["flag"].numel())
    executable = compile_kernel(kernel.func)

    def as_bytes(tensor):
        return tensor.view(torch.uint8).reshape(-1)

    fixed = [as_bytes(shared[name]) for name in ("c", "c_mc", "out_mc")]
    fixed += [shared["flag"], shared["flag_mc"]]

    def launcher(ws):
        args = [as_bytes(ws["a"]), as_bytes(ws["b"]), *fixed]
        return lambda: executable(*args)

    return [launcher(ws) for ws in workspace]


def check(origin: str, launch, shared: dict, reference) -> str | None:
    """Run one launch on zeroed outputs; return a mismatch message or None.
    The result must be the exact sum faithfully rounded (``rounding_error``)."""

    error, _ = run_checked(origin, launch, shared, reference)
    return error


def run_checked(origin: str, launch, shared: dict, reference):
    """``check``, also returning this rank's output."""

    import torch
    import torch.distributed as dist

    output = shared["c"] if origin == "flashinfer" else shared["out"]
    shared["c"].zero_()
    shared["out"].zero_()
    shared["flag"].zero_()
    torch.cuda.synchronize()
    dist.barrier()
    launch()
    torch.cuda.synchronize()
    dist.barrier()
    error = None
    ulps = rounding_error(output, reference)
    if not bool((ulps < 1).all()):
        bad = (ulps >= 1).nonzero()
        error = (f"{bad.shape[0]} mismatches, first at {bad[0].tolist()}, "
                 f"worst {float(ulps.max()):.3g} ulp")
    elif int(shared["flag"].count_nonzero()):
        error = "flags left set"
    failed = torch.tensor([error is not None], device="cuda", dtype=torch.int32)
    dist.all_reduce(failed)
    if int(failed) and error is None:
        error = "another rank mismatched"
    return error, output.clone()


# ---------------------------------------------------------------------------
# Tuning the origins.


def _log_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tune_rank(rank: int, world: int, origin: str, shape_name: str, log: str,
              todo: list[dict]) -> None:
    import torch

    from benchmarks.multimem import origins

    shape = SHAPES_BY_NAME[shape_name]
    device = torch.device("cuda", rank)
    workspace, shared, reference = allocate(rank, world, shape, device)

    def record(**fields):
        if rank == 0:
            with open(log, "a") as f:
                f.write(json.dumps({"origin": origin, "shape": shape_name, **fields,
                                    "time": round(time.time(), 1)}) + "\n")

    for config in todo:
        key = config_key(config)
        record(config=key, status="started")
        faulthandler.dump_traceback_later(CANDIDATE_TIMEOUT_S, exit=True)
        try:
            closures = origins.gemm_all_reduce_launchers(
                origin, rank, world, shape, config, workspace, shared)
        except origins.CantImplement as error:
            record(config=key, status="rejected", message=str(error)[:200])
            continue
        except Exception as error:  # noqa: BLE001 - every rank compiles the same IR
            record(config=key, status="compile_error", message=repr(error)[:400])
            continue
        finally:
            faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(CANDIDATE_TIMEOUT_S, exit=True)
        error = check(origin, closures[0], shared, reference)
        if error is not None:
            record(config=key, status="wrong", message=error)
            faulthandler.cancel_dump_traceback_later()
            continue
        times = common.slowest_rank(common.time_launches(
            {"origin": closures}, launches=TUNE_LAUNCHES, trials=TUNE_TRIALS, target_ms=TUNE_MS))
        faulthandler.cancel_dump_traceback_later()
        record(config=key, status="ok", us=times["origin"])
        del closures


def _tune_configs(origin: str, shape_name: str, world: int, log: Path,
                  configs: list[dict]) -> None:
    """Time every one of ``configs`` not yet in the log, surviving hangs and faults."""

    while True:
        done = {r["config"] for r in _log_lines(log)
                if r["origin"] == origin and r["shape"] == shape_name and r["status"] != "started"}
        todo = [c for c in configs if config_key(c) not in done]
        if not todo:
            return
        try:
            common.spawn("benchmarks.multimem.gemm_all_reduce:tune_rank", world,
                         origin, shape_name, str(log), todo)
            return
        except Exception:  # noqa: BLE001 - a hung or faulted candidate
            traceback.print_exc()
            records = [r for r in _log_lines(log)
                       if r["origin"] == origin and r["shape"] == shape_name]
            finished = {r["config"] for r in records if r["status"] != "started"}
            stuck = [r["config"] for r in records
                     if r["status"] == "started" and r["config"] not in finished]
            if not stuck:
                raise
            with open(log, "a") as f:
                for key in stuck:
                    f.write(json.dumps({"origin": origin, "shape": shape_name,
                                        "config": key, "status": "failed"}) + "\n")


def tune(origins_: list[str], shapes: list[str], world: int, log: Path) -> None:
    """Origin shapes search the whole space. On the sweep, CUTLASS's raster and
    swizzle (an L2-locality knob) are searched only around its ``REFINE_TOP``
    fastest geometries, which bounds the sweep's tuning time."""

    log.parent.mkdir(parents=True, exist_ok=True)
    origin_shapes = {shape["name"] for shape in ORIGIN_SHAPES}
    for shape_name in shapes:
        shape = SHAPES_BY_NAME[shape_name]
        for origin in origins_:
            if origin == "cutlass" and shape_name not in origin_shapes:
                _tune_configs(origin, shape_name, world, log,
                              candidates(origin, shape, world, geometry_only=True))
                geometries = []
                for config in best_configs(log, origin, shape_name, top=len(_geometry({}))):
                    geometry = {key: value for key, value in config.items()
                                if key not in ("raster", "swizzle")}
                    if geometry not in geometries:
                        geometries.append(geometry)
                refine = [config for geometry in geometries[:REFINE_TOP]
                          for config in _schedules(geometry) if full_tiles(shape, config, world)]
                _tune_configs(origin, shape_name, world, log, refine)
            else:
                _tune_configs(origin, shape_name, world, log, candidates(origin, shape, world))
            ok = sorted((r for r in _log_lines(log) if r["origin"] == origin
                         and r["shape"] == shape_name and r["status"] == "ok"),
                        key=lambda r: r["us"])
            best = f"{ok[0]['us']:.2f} us {ok[0]['config']}" if ok else "none"
            print(f"[tune] {shape_name} {origin}: {len(ok)} ok, best {best}", flush=True)


def best_configs(log: Path, origin: str, shape_name: str, top: int = TOP) -> list[dict]:
    ok = sorted((r for r in _log_lines(log) if r["origin"] == origin
                 and r["shape"] == shape_name and r["status"] == "ok"), key=lambda r: r["us"])
    return [json.loads(r["config"]) for r in ok[:top]]


# ---------------------------------------------------------------------------
# TIRx against the origins.


def compare_rank(rank: int, world: int, shape_name: str, entries: list[dict],
                 idle_s: float = 0.0) -> list[dict]:
    """Check and time each ``{"origin", "config"}`` entry's origin kernel and
    TIRx kernel, interleaved, on the same buffers (``idle_s``: see
    ``common.time_launches``)."""

    import torch
    import torch.distributed as dist

    from benchmarks.multimem import origins

    shape = SHAPES_BY_NAME[shape_name]
    device = torch.device("cuda", rank)
    workspace, shared, reference = allocate(rank, world, shape, device)
    rows = []
    for entry in entries:
        origin, config = entry["origin"], entry["config"]
        faulthandler.dump_traceback_later(CANDIDATE_TIMEOUT_S, exit=True)
        impls = {
            "origin": origins.gemm_all_reduce_launchers(
                origin, rank, world, shape, config, workspace, shared),
            "tirx": tirx_launchers(rank, world, shape, config, origin, workspace, shared),
        }
        errors, outputs = {}, {}
        for name, closures in impls.items():
            errors[name], outputs[name] = run_checked(origin, closures[0], shared, reference)
        same = torch.tensor([torch.equal(outputs["origin"], outputs["tirx"])], device="cuda",
                            dtype=torch.int32)
        dist.all_reduce(same, op=dist.ReduceOp.MIN)
        row = {"shape": shape_name, "origin": origin, "config": config_key(config),
               "origin_error": errors["origin"], "tirx_error": errors["tirx"],
               "bitwise_equal": bool(int(same))}
        if errors["origin"] is None and errors["tirx"] is None:
            times = common.slowest_rank(
                common.time_launches(impls, launches=LAUNCHES, trials=TRIALS, idle_s=idle_s))
            row.update(origin_us=times["origin"], tirx_us=times["tirx"],
                       ratio=times["tirx"] / times["origin"])
        faulthandler.cancel_dump_traceback_later()
        if rank == 0:
            print(json.dumps(row), flush=True)
        rows.append(row)
        del impls
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Per shape and origin: the origin's best time over its configurations
    against TIRx's best over the same configurations. A configuration's latest
    row replaces its earlier ones."""

    rows = list({(r["shape"], r["origin"], r["config"]): r for r in rows}.values())
    summary = []
    for shape in SHAPES:
        for origin in ("cutlass", "flashinfer"):
            group = [r for r in rows if r["shape"] == shape["name"] and r["origin"] == origin]
            if not group:
                continue
            timed = [r for r in group if "ratio" in r]
            row = {"shape": shape["name"], "M": shape["m"], "N": shape["n"], "K": shape["k"],
                   "types": f"{shape['ab']}->{shape['c']}", "origin": origin,
                   "configs": len(group), "correct": len(timed) == len(group),
                   "bitwise_equal": all(r["bitwise_equal"] for r in group)}
            if timed:
                best_origin = min(timed, key=lambda r: r["origin_us"])
                best_tirx = min(timed, key=lambda r: r["tirx_us"])
                row.update(origin_us=best_origin["origin_us"], origin_config=best_origin["config"],
                           tirx_us=best_tirx["tirx_us"], tirx_config=best_tirx["config"],
                           ratio=best_tirx["tirx_us"] / best_origin["origin_us"])
            summary.append(row)
    return summary


def compare(origins_: list[str], shapes: list[str], world: int, log: Path, out: Path,
            top: int, idle_s: float = 0.0) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    for shape_name in shapes:
        entries = [{"origin": origin, "config": config} for origin in origins_
                   for config in best_configs(log, origin, shape_name, top)]
        if not entries:
            print(f"[compare] {shape_name}: no tuned configurations", flush=True)
            continue
        rows = common.spawn("benchmarks.multimem.gemm_all_reduce:compare_rank", world,
                            shape_name, entries, idle_s)
        with open(out, "a") as f:
            for row in rows:
                f.write(json.dumps({**row, "time": round(time.time(), 1)}) + "\n")
    summary = summarize(_log_lines(out))
    print(common.table([r for r in summary if "ratio" in r], [
        ("shape", "shape", "s"), ("types", "types", "s"), ("origin", "origin", "s"),
        ("origin_us", "origin (us)", ".2f"), ("tirx_us", "TIRx (us)", ".2f"),
        ("ratio", "TIRx / origin", ".3f")]))
    for row in summary:
        if not row["correct"] or not row["bitwise_equal"]:
            print(f"[compare] {row['shape']} {row['origin']}: correct={row['correct']} "
                  f"bitwise_equal={row['bitwise_equal']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("tune", "compare"):
        mode = sub.add_parser(name)
        mode.add_argument("--origins", nargs="*", default=["cutlass", "flashinfer"])
        mode.add_argument("--world", type=int, default=4)
        mode.add_argument("--shapes", nargs="*", default=[s["name"] for s in SHAPES])
        mode.add_argument("--log", type=Path, default=RESULTS / "gemm_all_reduce_tuning.jsonl")
    sub.choices["compare"].add_argument("--top", type=int, default=TOP)
    sub.choices["compare"].add_argument(
        "--out", type=Path, default=RESULTS / "gemm_all_reduce_gb200x4.jsonl")
    sub.choices["compare"].add_argument("--idle-s", type=float, default=0.0)
    args = parser.parse_args()
    if args.mode == "tune":
        tune(args.origins, args.shapes, args.world, args.log)
    else:
        compare(args.origins, args.shapes, args.world, args.log, args.out, args.top, args.idle_s)


if __name__ == "__main__":
    main()
