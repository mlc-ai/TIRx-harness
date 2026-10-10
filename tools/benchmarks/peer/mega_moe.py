"""TIRx ``sm100_fp8_fp4_mega_moe`` against DeepGEMM's ``fp8_fp4_mega_moe`` at
expert parallelism 4.

    python -m benchmarks.peer.mega_moe prepare [--shapes NAME ...] [--jobs N]
    python -m benchmarks.peer.mega_moe compare [--shapes NAME ...] [--rounds N]

The origin is DeepGEMM at the commit the port pins (559d79fb), imported from
``DEEPGEMM_DIR`` (default ``~/src/DeepGEMM-559d79f``, built in place).
``prepare`` compiles both specializations of the port for every shape on the
CPU -- with and without the cumulative expert statistics -- into ``--cache``,
for the SM count of GPU 0 (``--num-sms``): the port, like DeepGEMM, bakes the
SM count into its schedule, and a compile that cannot see the GPU otherwise
assumes a B200's 148 SMs (GB200 has 152). Each build records a hash of the
port's sources: ``prepare`` rebuilds a shape whose sources changed, and
``compare`` refuses to time a build from other sources.
``compare`` runs each shape through the port's distributed harness
(``_run_distributed(config, "bench")``), which follows DeepGEMM's own benchmark
(``tests/test_mega_moe.py``): inputs and routing are drawn and cast as that
test draws and casts them, each rank's output and statistics from one TIRx
launch must equal DeepGEMM's bitwise before anything is timed, and then each
of ``rounds`` rounds times both kernels in one ``deep_gemm.testing.bench_kineto``
session (the kernel's time from the profiler; the L2 flushed and the ranks
barriered outside it, with the all-reduce DeepGEMM's test queues behind its
GPU sleep), alternating their order. A round's time is its slowest
rank's; the reported time is the median over rounds (the harness's mean is
kept as ``*_mean_us``): a process's first round runs on cold clocks and can
take twice as long, for both kernels alike.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from benchmarks.multimem import common

DEEPGEMM_DIR = Path(os.environ.get("DEEPGEMM_DIR", "~/src/DeepGEMM-559d79f")).expanduser()
PORT_DIR = Path(__file__).resolve().parents[2] / "ported" / "deepgemm"
RESULTS = Path(__file__).parent / "results"
CACHE = Path(os.environ.get("TIRX_MEGAMOE_CACHE", "~/.cache/tirx-megamoe")).expanduser()
# Read by ``tirx_kernels.runner.hardware_num_sms``, which sizes the port.
NUM_SMS_ENV = "TIRX_PREPARE_NUM_SMS"
WORLD = 4
ROUNDS = 11


def _shape(name: str, tokens: int, hidden: int, inter: int, experts: int, topk: int,
           shared: int, max_tokens: int | None = None) -> dict:
    return {"name": name, "num_tokens": tokens,
            "num_max_tokens_per_rank": tokens if max_tokens is None else max_tokens,
            "hidden": hidden, "intermediate_hidden": inter, "num_experts": experts,
            "num_topk": topk, "num_shared_experts": shared}


# DeepGEMM's test defaults (it spawns 8 ranks; these run 4), and the 4-rank
# configurations of the port's own test matrix (``CONFIGS`` of
# ``ported/deepgemm/sm100_fp8_fp4_mega_moe.py``) not already among them.
ORIGIN_SHAPES = [
    _shape("deepgemm_default", 8192, 7168, 3072, 384, 6, 1),
    _shape("port_t2_h1024_e8_k1", 2, 1024, 512, 8, 1, 0, max_tokens=4),
    _shape("port_t64_e384_k6", 64, 7168, 3072, 384, 6, 0),
    _shape("port_t256_e384_k6", 256, 7168, 3072, 384, 6, 0),
    _shape("port_t1024_e384_k6", 1024, 7168, 3072, 384, 6, 0),
    _shape("port_t8192_e384_k6", 8192, 7168, 3072, 384, 6, 0),
]
# DeepSeek-V3's MoE layer (256 routed experts, top-8, hidden 7168, expert
# width 2048, one shared expert) from decode to prefill token counts per rank.
DSV3_TOKENS = [1, 16, 64, 256, 1024, 2048, 4096, 8192]
SWEEP_SHAPES = [_shape(f"dsv3_t{tokens}", tokens, 7168, 2048, 256, 8, 1)
                for tokens in DSV3_TOKENS]
SHAPES = ORIGIN_SHAPES + SWEEP_SHAPES
SHAPES_BY_NAME = {shape["name"]: shape for shape in SHAPES}


def config(shape: dict):
    from ported.deepgemm._sm100_fp8_fp4_mega_moe.spec import MegaMoeConfig

    fields = {key: value for key, value in shape.items() if key != "name"}
    result = MegaMoeConfig(num_processes=WORLD, activation_clamp=10.0, fast_math=1, **fields)
    result.validate()
    return result


def device_num_sms() -> int:
    """GPU 0's SM count, read in a child so this process never initializes CUDA."""
    return int(subprocess.check_output([sys.executable, "-c", (
        "import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)")]))


def libraries(shape: dict, cache: Path, num_sms: int) -> dict[bool, Path]:
    """The compiled port of ``shape`` for ``num_sms`` SMs, keyed by whether it
    collects statistics."""
    return {stats: cache / f"sm{num_sms}" / shape["name"] / ("stats.so" if stats else "no_stats.so")
            for stats in (False, True)}


def source_digest() -> str:
    """A hash of the port's sources; a build records the hash it was built from.
    ``data.py`` is the host harness and does not reach the build."""
    digest = hashlib.sha256()
    for path in sorted([*PORT_DIR.glob("_sm100_fp8_fp4_mega_moe/*.py"),
                        PORT_DIR / "sm100_fp8_fp4_mega_moe.py"]):
        if path.name == "data.py":
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _source_stamp(shape: dict, cache: Path, num_sms: int) -> Path:
    return cache / f"sm{num_sms}" / shape["name"] / "source"


def _compile(name: str, cache: str, num_sms: int) -> str:
    from dataclasses import asdict

    from ported.deepgemm._sm100_fp8_fp4_mega_moe.spec import (
        _compile_tirx_mega_moe_for_config,
        _get_mega_moe_cuda_compile_mode,
    )

    os.environ[NUM_SMS_ENV] = str(num_sms)
    shape = SHAPES_BY_NAME[name]
    start = time.time()
    digest = source_digest()
    stamp = _source_stamp(shape, Path(cache), num_sms)
    stale = not stamp.exists() or stamp.read_text().strip() != digest
    for stats, path in libraries(shape, Path(cache), num_sms).items():
        if path.exists() and not stale:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        executable = _compile_tirx_mega_moe_for_config(
            **asdict(config(shape)), collect_stats=stats,
            cuda_compile_mode=_get_mega_moe_cuda_compile_mode())
        partial = path.with_suffix(".partial.so")
        executable.export_library(str(partial))
        partial.rename(path)
    stamp.write_text(digest + "\n")
    return f"[prepare] {name}: {time.time() - start:.0f} s"


def prepare(shapes: list[str], cache: Path, jobs: int, num_sms: int) -> None:
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(jobs, mp_context=context) as pool:
        count = len(shapes)
        for line in pool.map(_compile, shapes, [str(cache)] * count, [num_sms] * count):
            print(line, flush=True)


def _lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compare(shapes: list[str], cache: Path, out: Path, rounds: int) -> None:
    from ported.deepgemm._sm100_fp8_fp4_mega_moe.data import _run_distributed
    from ported.deepgemm._sm100_fp8_fp4_mega_moe.spec import _PREPARED_LIBRARY_ENV

    sys.path.insert(0, str(DEEPGEMM_DIR))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(DEEPGEMM_DIR), *filter(None, [os.environ.get("PYTHONPATH")])])
    os.environ["TIRX_INTERNAL_BENCH_REFERENCES"] = "1"
    num_sms = device_num_sms()
    os.environ[NUM_SMS_ENV] = str(num_sms)
    out.parent.mkdir(parents=True, exist_ok=True)
    digest = source_digest()
    for name in shapes:
        shape = SHAPES_BY_NAME[name]
        paths = libraries(shape, cache, num_sms)
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"run `prepare` first: {missing}")
        stamp = _source_stamp(shape, cache, num_sms)
        built_from = stamp.read_text().strip() if stamp.exists() else None
        if built_from is not None and built_from != digest:
            raise RuntimeError(f"{name} was built from other sources; run `prepare` again")
        for stats, path in paths.items():
            os.environ[_PREPARED_LIBRARY_ENV[stats]] = str(path)
        result = _run_distributed(config(shape), "bench", rounds=rounds, cooldown_s=1.0)
        samples = result["round_samples"]
        row = {"shape": name, "run": round(time.time(), 1), "num_sms": num_sms,
               "source": built_from,
               "config": {k: v for k, v in shape.items() if k != "name"},
               "tirx_us": statistics.median(samples["tirx"]),
               "deepgemm_us": statistics.median(samples["deepgemm"]),
               "tirx_mean_us": result["impls"]["tirx"],
               "deepgemm_mean_us": result["impls"]["deepgemm"],
               "max_abs_diff": result["deepgemm_max_abs_diff"], "round_samples": samples,
               "rank_round_samples": {
                   impl: [rank["round_samples"][impl] for rank in result["rank_results"]]
                   for impl in samples}}
        row["ratio"] = row["tirx_us"] / row["deepgemm_us"]
        print(json.dumps(row), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
    print(summary(_lines(out)))


def summary(rows: list[dict]) -> str:
    """The latest run of each shape: DeepGEMM's and TIRx's times."""
    latest = {}
    for row in rows:
        if row["shape"] not in latest or row["run"] > latest[row["shape"]]["run"]:
            latest[row["shape"]] = row
    table = []
    for shape in SHAPES:
        row = latest.get(shape["name"])
        if row is not None:
            table.append({**row, "tokens": shape["num_tokens"], "experts": shape["num_experts"],
                          "topk": shape["num_topk"], "inter": shape["intermediate_hidden"],
                          "shared": shape["num_shared_experts"]})
    return common.table(table, [
        ("shape", "shape", "s"), ("tokens", "tokens/rank", "d"), ("experts", "experts", "d"),
        ("topk", "top-k", "d"), ("inter", "I", "d"), ("shared", "shared", "d"),
        ("deepgemm_us", "DeepGEMM (us)", ".1f"), ("tirx_us", "TIRx (us)", ".1f"),
        ("ratio", "TIRx / DeepGEMM", ".3f")])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("prepare", "compare"):
        mode = sub.add_parser(name)
        mode.add_argument("--shapes", nargs="*", default=[s["name"] for s in SHAPES])
        mode.add_argument("--cache", type=Path, default=CACHE)
    sub.choices["prepare"].add_argument("--jobs", type=int, default=8)
    sub.choices["prepare"].add_argument("--num-sms", type=int, default=None,
                                        help="SM count to compile for (default: GPU 0's)")
    sub.choices["compare"].add_argument("--rounds", type=int, default=ROUNDS)
    sub.choices["compare"].add_argument("--out", type=Path,
                                        default=RESULTS / "mega_moe_gb200x4.jsonl")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.shapes, args.cache, args.jobs, args.num_sms or device_num_sms())
    else:
        compare(args.shapes, args.cache, args.out, args.rounds)


if __name__ == "__main__":
    main()
