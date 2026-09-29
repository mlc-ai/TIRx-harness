#!/usr/bin/env python3
"""Single CLI for every kernel-evolution benchmark.

    python evolution/benchmark/adapter.py <workload> [version] [--warmup N] [--repeat N]

``<workload>`` is a registered workload directory under candidates/
(``gdn/decode``, ``dsa``, ``fp16_gemm_floor``, ... — a path such as
``candidates/gdn/decode`` or ``.`` from inside the directory works too).
``version`` is a ``vN`` directory holding ``solution.py``, or ``baseline``
(default) for the baseline self-check.

The ``PACKAGED`` table maps every workload to a task in the inline
``flashinfer_bench_evolve`` package beside this adapter, which owns inputs, correctness,
timing and the live baseline. This file only loads ``vN/solution.py`` and hands
it to the task's ``tirx_prepare``/``tirx_run``.

It is also where a benchmark says what it is made of: ``harness_sources`` names the
task files, ``plan`` fixes the config, workloads and candidate for one scoring run,
and ``blobs`` loads the tensors those workloads reference. A runner that cannot read
this checkout — one sending the benchmark somewhere else — gets everything from those
three, and nothing here knows or cares where it goes.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import types
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Candidate paths are a task contract, independent of the adapter's location.
CANDIDATES_ROOT = REPO_ROOT / "candidates"
CANDIDATE_FILENAME = "solution.py"
BENCH_REPO_ROOT = Path(__file__).resolve().parent

# workload dir (relative to candidates/) ->
# (packaged task, default warmup, default repeat, registered shape mode)
# Unsuffixed tasks score the full suite; shape-specific tasks select one shape.
# 'max' uses the package's selection; 'pinned' selects the reviewed UUID below.
PACKAGED = {
    "alphamoe": ("alphamoe", 3, 30, "all"),
    "alphamoe_m128_e512_topk10_k2048_i128_fp8": ("alphamoe", 3, 30, "pinned"),
    "dsa": ("dsa_attention", 10, 50, "all"),
    "flash_attention": ("fa4_forward", 10, 30, "all"),
    "flash_attn_bwd": ("fa4_backward", 10, 30, "all"),
    "fp16_gemm_floor": ("fp16_gemm_floor", 10, 50, "all"),
    "gdn/decode": ("gdn_decode", 10, 50, "all"),
    "gdn/prefill": ("gdn_prefill", 10, 50, "all"),
    "grouped_gemm/fp8": ("grouped_gemm_fp8", 3, 30, "all"),
    "kda/decode": ("kda_decode", 3, 50, "all"),
    "kda/decode_b128_t1_h16_hv32_d128_bf16": ("kda_decode", 3, 50, "pinned"),
    "kda/forward": ("kda_forward", 3, 30, "all"),
    "kda/forward_b1_t8192_h96": ("kda_forward", 3, 30, "max"),
    "kda/backward": ("kda_backward", 2, 20, "all"),
    "kda/backward_b1_t8192_h96": ("kda_backward", 2, 20, "max"),
    "mla_dsv4": ("mla_dsv4", 3, 50, "all"),
    "mla_dsv4_prefill_b2_qsum386_qmax257_h128_d512_swa16384_c16384_topk1152_bf16_hnd_varlen": (
        "mla_dsv4", 3, 50, "pinned"
    ),
    "moe": ("moe", 10, 50, "all"),
    "msa/prefill": ("msa_prefill", 3, 50, "all"),
    "msa/prefill_b1_q4096_kv4096_hq64_hkv4_d128_topk16_bf16_flat": ("msa_prefill", 3, 50, "pinned"),
    "msa/decode": ("msa_decode", 3, 50, "all"),
    "msa/decode_b128_q16_kv4096_hq64_hkv4_d128_topk16_bf16_flat": ("msa_decode", 3, 50, "pinned"),
    "nvfp4_attention": ("nvfp4_attention", 3, 50, "all"),
    "vsa": ("vsa", 3, 50, "all"),
    "vsa_s80000_h8_d128_blk128_topk156_bf16": ("vsa", 3, 50, "pinned"),
}

# The six CAKE shape-specific tasks favor substantial work within the chosen
# contract and baseline arm. Five use fixed lengths; MLA DSV4 explicitly selects
# a varlen prefill row with complete KV pages. NVFP4 already has only one row.
# Keep the exact UUIDs stable instead of relying on lexicographic shape maxima.
PINNED_WORKLOAD_UUIDS = {
    "alphamoe": "alphamoe-qwen3-next-tp4-m128-e512-top10-k2048-i128-fp8",
    "kda_decode": "kda-decode-d128-t1-b128-h16-hv32-standard_decode-precomputed",
    "mla_dsv4": "mla-dsv4-prefill-h128-swa16384-topk4x-c16384-k1024-bf16-hnd",
    "msa_prefill": "prefill_bf16_b1_q4096_kv4096_h64",
    "msa_decode": "mtp_bf16_b128_q16_kv4096_h64",
    "vsa": "vsa-pooled-blk128-s80000-h8-topk156",
}


def workload_key(arg: str) -> str:
    """Normalize a CLI workload argument (dir name or path) to a candidates/ subdir."""

    resolved = Path(arg).resolve()
    if resolved.is_dir() and resolved != CANDIDATES_ROOT and resolved.is_relative_to(CANDIDATES_ROOT):
        key = resolved.relative_to(CANDIDATES_ROOT).as_posix()
    else:
        key = arg.strip("/").removeprefix("candidates/")
    if key in PACKAGED:
        return key
    known = sorted(PACKAGED)
    raise SystemExit(f"unknown workload {arg!r}; choose from: {', '.join(known)}")


def _bench_repo_on_path() -> None:
    """Put the inline benchmark package ahead of an installed copy."""

    if not (BENCH_REPO_ROOT / "flashinfer_bench_evolve").is_dir():
        raise RuntimeError(
            f"Missing benchmark sources at {BENCH_REPO_ROOT}; use a complete repository checkout."
        )
    if str(BENCH_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(BENCH_REPO_ROOT))


def import_benchmark(task_name: str):
    """Import ``flashinfer_bench_evolve.tasks.<task_name>.benchmark`` from this repository."""

    _bench_repo_on_path()
    return importlib.import_module(f"flashinfer_bench_evolve.tasks.{task_name}.benchmark")


def select_workloads(bench, config):
    """Resolve explicit pins; delegate other modes to the packaged benchmark."""

    if config.shape_mode != "pinned":
        return bench.make_workloads(config)
    selected_uuid = PINNED_WORKLOAD_UUIDS.get(bench.TASK_NAME)
    if selected_uuid is None:
        raise ValueError(f"{bench.TASK_NAME}: no pinned workload UUID configured")
    rows = bench.make_workloads(
        replace(config, shape_mode="all", include_official=True, max_official=0)
    )
    selected = [
        row for row in rows
        if row["suite"] == "official" and row["raw"]["uuid"] == selected_uuid
    ]
    if len(selected) != 1:
        raise ValueError(
            f"{bench.TASK_NAME}: expected one official workload {selected_uuid!r}, "
            f"found {len(selected)}"
        )
    # Correctness-only probes are not shape-pinned by the UUID filter above, so
    # a task that publishes them keeps them here, pinned to the selected row's
    # shape. Without this a pinned task would silently drop the suite's
    # numerical-edge gates and score on one timed row alone.
    make_stress = getattr(bench, "make_stress_workloads", None)
    if make_stress is not None:
        axes = selected[0]["axes"]
        selected = selected + make_stress(
            (axes["num_heads"], axes["total_tokens"])
        )
    return selected


def load_solution(workload_dir: Path, version: str):
    """Load `vN/solution.py` without requiring package marker files."""

    solution_path = workload_dir / version / CANDIDATE_FILENAME
    if not solution_path.exists():
        raise FileNotFoundError(f"{solution_path} not found")
    sys.path.insert(0, str(workload_dir))
    rel_parts = workload_dir.relative_to(CANDIDATES_ROOT).parts
    package_stem = "_".join(part.replace("-", "_") for part in rel_parts)
    package_name = f"_tirx_{package_stem}_{version}"
    module_name = f"{package_name}.{solution_path.stem}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(solution_path.parent)]
        sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        module_name,
        solution_path,
        submodule_search_locations=[str(solution_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import solution module: {solution_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_version(
    task_name: str,
    workload_dir: Path,
    version: str,
    *,
    warmup: int,
    repeat: int,
    shape_mode: str,
):
    """Score ``vN/solution.py`` against the task's live baseline (``"baseline"``: self-check)."""

    bench = import_benchmark(task_name)
    config = replace(
        bench.default_config(),
        warmup=warmup,
        iters=repeat,
        shape_mode=shape_mode,
    )
    workloads = select_workloads(bench, config)
    if version == "baseline":
        return bench.run_suite(config, workloads=workloads)
    solution = load_solution(workload_dir, version)
    return bench.run_suite(
        config,
        candidate_fn=bench.tirx_run,
        candidate_prepare_fn=lambda *args: bench.tirx_prepare(solution, *args),
        workloads=workloads,
    )


def harness_sources(task_name: str) -> dict[str, str]:
    """The pinned harness and task files one benchmark needs, verbatim, keyed by package path.

    The package roots, shared benchmark_common, task benchmark/baseline, and
    definition containing its independent correctness oracle.
    """

    package = BENCH_REPO_ROOT / "flashinfer_bench_evolve"
    files = ["__init__.py", "benchmark_common.py", "tasks/__init__.py",
             *(f"tasks/{task_name}/{name}" for name in
               ("__init__.py", "benchmark.py", "baseline.py", "definition.json"))]
    # A task may carry extra modules its benchmark imports lazily, such as the
    # salted holdout probes. Ship the ones it publishes; absent files are not
    # an error, because most tasks have none.
    files += [
        f"tasks/{task_name}/{name}"
        for name in ("holdout.py",)
        if (package / f"tasks/{task_name}/{name}").exists()
    ]
    return {f"flashinfer_bench_evolve/{file}": (package / file).read_text() for file in files}


def plan(task_name: str, workload_dir: Path, version, *, warmup: int, repeat: int, shape_mode: str):
    """What one scoring run is: the config overrides, the workloads it covers, the candidate."""

    bench = import_benchmark(task_name)
    overrides = {
        "warmup": warmup,
        "iters": repeat,
        "shape_mode": shape_mode,
        "device": "cuda:0",  # one benchmark occupies one GPU
    }
    workloads = select_workloads(bench, replace(bench.default_config(), **overrides))
    candidate = None
    if version not in (None, "baseline"):
        candidate = (workload_dir / version / CANDIDATE_FILENAME).read_bytes()
    return overrides, workloads, candidate


def blobs(workloads: list) -> tuple[list[dict], list]:
    """The safetensors ``workloads`` reference, as ``(keys, tensors)`` loaded on the CPU.

    A caller that cannot read the blob files itself gets the tensors and the keys that
    identify them, and needs to know nothing about how the harness stores them.
    """

    import torch

    _bench_repo_on_path()
    from flashinfer_bench_evolve.benchmark_common import load_safetensor

    specs = {}
    for entry in workloads:
        for spec in ((entry.get("raw") or {}).get("inputs") or {}).values():
            if isinstance(spec, dict) and spec.get("type") == "safetensors":
                specs.setdefault((spec["path"], spec["tensor_key"]), spec)
    keys = [{"path": path, "tensor_key": key} for path, key in specs]
    tensors = [load_safetensor(spec, torch.device("cpu")) for spec in specs.values()]
    return keys, tensors


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workload", help=f"registered workload: {', '.join(PACKAGED)}")
    parser.add_argument("version", nargs="?", default=None, help="Version directory (e.g. v0) or 'baseline' (default).")
    parser.add_argument("--warmup", type=int, default=None, help="default: per-workload value")
    parser.add_argument("--repeat", type=int, default=None, help="default: per-workload value")
    args = parser.parse_args(argv)
    key = workload_key(args.workload)
    workload_dir = CANDIDATES_ROOT / key

    task, warmup, repeat, shape_mode = PACKAGED[key]
    run_version(
        task,
        workload_dir,
        "baseline" if args.version is None else args.version,
        warmup=warmup if args.warmup is None else args.warmup,
        repeat=repeat if args.repeat is None else args.repeat,
        shape_mode=shape_mode,
    )


if __name__ == "__main__":
    main()
