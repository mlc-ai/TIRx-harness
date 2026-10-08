"""Prepared full-model scoring shared by the three Qwen3.8 workload groups."""

from __future__ import annotations

import gc
import json
import os
import secrets
import statistics
from pathlib import Path

import torch

from ...benchmark_common import choose_device, config_from_env, summarize
from . import baseline
from .shapes import Case, select


# These sources travel with each group's benchmark to a remote worker. Model
# weights stay on that worker; they are never part of a candidate or upload.
HARNESS_FILES = tuple(
    f"tasks/qwen38/{name}"
    for name in ("__init__.py", "benchmark.py", "baseline.py", "model.py", "shapes.py")
)


def default_config():
    return config_from_env(warmup=3, iters=9, trials=1)


def _model_path():
    """Read weights from the model root, downloading missing files there."""
    root = Path(os.environ.get("TIRX_MODELS_DIR", "/raid/catalyst/models")).expanduser()
    path = root / "Qwen3.8-27B"

    def fetch(filename):
        file = path / filename
        # Complete, provisioned weights also work on read-only offline workers.
        if not file.is_file():
            from huggingface_hub import hf_hub_download
            from .model import MODEL_REVISION

            print(f"Qwen3.8 weights: downloading {filename} to {path}", flush=True)
            hf_hub_download(
                repo_id="Qwen/Qwen3.8-27B", filename=filename,
                revision=MODEL_REVISION, local_dir=path,
            )
        return file

    index = fetch("model.safetensors.index.json")
    shards = set(json.loads(index.read_text())["weight_map"].values())
    for filename in ("config.json", "generation_config.json", *sorted(shards)):
        fetch(filename)
    return path


def make_workloads(group, config=None):
    # Group membership is immutable. Environment limits cannot silently turn a
    # multi-shape task into a subset. The adapter alone resolves single-case pins.
    return [
        {
            "suite": "official",
            "id": case.name,
            "axes": {"batch": case.batch, "segments": case.segments},
            "raw": {"uuid": case.name, "group": case.group},
        }
        for case in select(group)
    ]


def tirx_prepare(solution_module, model, cache, prepared):
    """Expose model data, without reference-model execution methods."""
    data = {
        name: getattr(model, name)
        for name in (
            "config", "eps", "dtype", "device", "layer_types", "embedding",
            "layers", "final_norm", "lm_head", "rope", "rotary_dim",
            "page_size", "max_context",
        )
    }
    data.update(cache=cache, prepared=prepared)
    launch = solution_module.setup(data)
    if not callable(launch):
        raise TypeError("Qwen3.8 solution.setup(data) must return a callable")
    return launch


def tirx_run(launch):
    return launch()


def _check_cache(cache, expected):
    actual = baseline.all_tensors(cache)
    if len(actual) != len(expected):
        raise AssertionError("candidate changed the exposed cache tensor count")
    return max(
        (baseline.check(x, y, f"cache {i}") for i, (x, y) in enumerate(zip(actual, expected))),
        default=0.0,
    )


def _run_case(model, case, cfg, seed, candidate_fn, candidate_prepare_fn):
    """Check one prepared step, then time consecutive calls from matched states."""
    rows, new = baseline.inputs(case, seed)
    cache = baseline.new_cache(model, case)
    baseline.prime(model, cache, case, rows)
    prepared = model._prepare_step(new, cache, return_all_logits=case.all_logits)[2]
    if case.all_logits:
        prepared["verify"] = True
    saved = baseline.save_state(cache)
    reset = lambda: baseline.restore(cache, saved, prepared)
    reference = baseline.Runner(lambda: model._forward(cache, **prepared), case.graph, reset)
    reset()
    expected = baseline.snapshot(reference())
    expected_shape = (sum(case.lengths) if case.all_logits else case.batch, 248320)
    if tuple(expected.shape) != expected_shape or not bool(torch.isfinite(expected).all()):
        raise AssertionError("invalid reference logits")
    expected_cache = [baseline.snapshot(x) for x in baseline.all_tensors(cache)]

    if candidate_fn is None:
        candidate = reference
    else:
        reset()
        launch = candidate_prepare_fn(model, cache, prepared)
        candidate = baseline.Runner(lambda: candidate_fn(launch), case.graph, reset)

    # Check every exposed KV, convolution, recurrent and verification tensor,
    # including prefix preservation, once before timing. Release the large CPU
    # reference snapshots after this gate. Timing advances state within a block,
    # so later logits cannot be compared against this single-step reference.
    reset()
    max_error = baseline.check(candidate(), expected, "candidate logits")
    max_error = max(max_error, _check_cache(cache, expected_cache))
    del expected, expected_cache

    values = {"baseline": [], "candidate": []}
    runners = {"baseline": reference, "candidate": candidate}
    for trial in range(cfg.trials):
        order = list(runners)
        if trial % 2:
            order.reverse()
        for name in order:
            reset()
            for repetition in range(cfg.warmup + cfg.iters):
                output, timing = baseline.timed(runners[name])
                if repetition >= cfg.warmup:
                    values[name].append(timing["gpu_ms"])
            if not bool(torch.isfinite(output).all()):
                raise AssertionError(f"{name}: nonfinite logits after timing block")
    baseline_ms = statistics.median(values["baseline"])
    kernel_ms = statistics.median(values["candidate"])
    return dict(
        passed=True, verdict="PASS", note="", correctness_checks=1,
        max_abs=max_error, matched=1.0, baseline_ms=baseline_ms, kernel_ms=kernel_ms,
        speedup=baseline_ms / kernel_ms, timed=True, timer="cuda_event",
        timing_mode="stateful_blocks",
    )


@torch.inference_mode()
def run_suite(group, config=None, candidate_fn=None, workloads=None, candidate_prepare_fn=None):
    cfg = config or default_config()
    entries = make_workloads(group, cfg) if workloads is None else workloads
    if not entries:
        raise ValueError("the Qwen3.8 workload list cannot be empty")
    if cfg.warmup < 0 or cfg.iters < 1 or cfg.trials < 1:
        raise ValueError("warmup must be nonnegative; repeat and trials must be positive")
    from .model import Qwen38

    model = Qwen38(_model_path(), device=choose_device(cfg.device), max_context=131072)
    configured_seed = os.environ.get("QWEN38_SEED")
    seed = int(configured_seed) if configured_seed is not None else secrets.randbits(31)
    print(f"Qwen3.8 {group}: {len(entries)} cases, seed={seed}", flush=True)
    results = []
    for entry in entries:
        case = Case(entry["id"], entry["raw"]["group"], tuple(map(tuple, entry["axes"]["segments"])))
        if case.group != group:
            raise ValueError(f"{case.name} is not a {group} workload")
        print(f"Preparing {case.name}: (batch, previous, new)={case.segments}", flush=True)
        row = {"suite": entry["suite"], "id": case.name, "axes": entry["axes"]}
        try:
            row.update(_run_case(model, case, cfg, seed, candidate_fn, candidate_prepare_fn))
        except AssertionError as exc:
            row.update(passed=False, verdict="FAIL", note=str(exc), speedup=None,
                       baseline_ms=None, kernel_ms=None, timed=True, timer="cuda_event")
        results.append(row)
        print(row, flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        if not row["passed"]:
            break
    if all(row["passed"] for row in results):
        summarize(results, "batch")
    else:
        print("Qwen3.8 task failed; no aggregate score", flush=True)
    return results
