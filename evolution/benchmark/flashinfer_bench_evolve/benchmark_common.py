#!/usr/bin/env python3
"""Small helpers for the standalone kernel benchmarks."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F


BENCH_ROOT = Path(__file__).resolve().parent
TASKS_ROOT = BENCH_ROOT / "tasks"
# Workload rows reference tensors as "./blob/..." relative to this root. The
# blob subset ships inside the package (flashinfer_bench_evolve/blob/), so the default resolves
# in a checkout and in an installed copy alike; BENCH_TRACE_ROOT points it at
# an external trace dataset instead.
TRACE_ROOT = Path(os.environ.get("BENCH_TRACE_ROOT", BENCH_ROOT))


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


def env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw in (None, "") else raw


DEFAULT_CORRECTNESS_RUNS = env_int("BENCH_CORRECTNESS_RUNS", 5)
DEFAULT_CHECK_AFTER_TIMING = bool(env_int("BENCH_CHECK_AFTER_TIMING", 1))


@dataclass(frozen=True)
class BenchConfig:
    """One evaluation's knobs, captured at call time.

    Task modules build this via ``default_config()`` so a persistent process
    re-reads the environment on every evaluation instead of trusting values
    frozen at first import.
    """

    device: str = "auto"
    warmup: int = 3
    iters: int = 50
    trials: int = 3
    max_official: int = 0
    include_official: bool = True
    shape_mode: str = "all"
    correctness_runs: int = 5
    check_after_timing: bool = True
    require_repeatable_outputs: bool = False
    check_input_dependence: bool = True


def config_from_env(**task_defaults) -> BenchConfig:
    """Read the ``BENCH_*`` knobs now, falling back to the task's defaults."""

    base = BenchConfig(**task_defaults)
    return BenchConfig(
        device=env_str("BENCH_DEVICE", base.device),
        warmup=env_int("BENCH_WARMUP", base.warmup),
        iters=env_int("BENCH_ITERS", base.iters),
        trials=env_int("BENCH_TRIALS", base.trials),
        max_official=env_int("BENCH_MAX_OFFICIAL", base.max_official),
        include_official=bool(
            env_int("BENCH_INCLUDE_OFFICIAL", int(base.include_official))
        ),
        shape_mode=env_str("BENCH_OFFICIAL_SHAPE_MODE", base.shape_mode),
        correctness_runs=env_int("BENCH_CORRECTNESS_RUNS", base.correctness_runs),
        check_after_timing=bool(
            env_int("BENCH_CHECK_AFTER_TIMING", int(base.check_after_timing))
        ),
        require_repeatable_outputs=bool(
            env_int(
                "BENCH_REQUIRE_REPEATABLE_OUTPUTS",
                int(base.require_repeatable_outputs),
            )
        ),
        check_input_dependence=bool(
            env_int("BENCH_CHECK_INPUT_DEPENDENCE", int(base.check_input_dependence))
        ),
    )


def choose_device(device_setting: str) -> torch.device:
    if device_setting != "auto":
        device = torch.device(device_setting)
        torch.cuda.set_device(device)
        return device

    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    candidates = []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_ids = None
    if visible:
        visible_ids = [int(x) for x in visible.split(",") if x.strip().isdigit()]
    for row in rows:
        idx_s, mem_s, util_s = [part.strip() for part in row.split(",")]
        if visible_ids is not None and int(idx_s) not in visible_ids:
            continue
        candidates.append((int(mem_s), int(util_s), int(idx_s)))
    mem, util, idx = min(candidates)
    local_idx = visible_ids.index(idx) if visible_ids is not None else idx
    device = torch.device(f"cuda:{local_idx}")
    torch.cuda.set_device(device)
    print(f"selected GPU cuda:{idx} (memory_used={mem} MiB, util={util}%)")
    return device


def load_workloads(
    path: Path,
    max_workloads: int,
    *,
    shape_mode: str = "all",
    shape_axes: tuple[str, ...] = (),
):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    mode = shape_mode.lower()
    if mode in ("max", "largest"):
        if not shape_axes:
            raise ValueError("shape_axes must be provided when shape_mode='max'")
        if not rows:
            return rows

        def shape_key(row):
            axes = row["workload"]["axes"]
            return tuple(axes[name] for name in shape_axes)

        return [max(rows, key=shape_key)]
    if mode not in ("all", "full"):
        raise ValueError(
            f"Unsupported BENCH_OFFICIAL_SHAPE_MODE={shape_mode!r}; use 'all'/'full' or 'max'"
        )
    if max_workloads > 0:
        rows = rows[:max_workloads]
    return rows


def load_task_workloads(
    task_name: str,
    max_workloads: int,
    *,
    shape_mode: str = "all",
    shape_axes: tuple[str, ...] = (),
):
    """Load one task's published workloads without leaking paths into the task module."""

    return load_workloads(
        _task_file(task_name, "workload.jsonl"),
        max_workloads,
        shape_mode=shape_mode,
        shape_axes=shape_axes,
    )


def include_synthetic_workloads(shape_mode: str) -> bool:
    """Synthetic stress rows follow full official sweeps, not max-shape runs."""

    return shape_mode.lower() in ("all", "full")


def resolve_tensor_path(raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return TRACE_ROOT / p


def load_safetensor(spec: dict, device: torch.device) -> torch.Tensor:
    import safetensors.torch as st

    path = resolve_tensor_path(spec["path"])
    tensor = st.load_file(str(path))[spec["tensor_key"]].contiguous()
    return tensor.to(device=device, non_blocking=True)


def load_kernel(path: Path, module_name: str = "candidate_kernel"):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load kernel from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.run


def load_candidate_override(env_name: str, default_fn, module_name: str):
    """Load an explicitly configured candidate, otherwise reuse the task baseline."""

    raw_path = os.environ.get(env_name)
    if raw_path in (None, ""):
        return default_fn
    return load_kernel(Path(raw_path).expanduser(), module_name)


def _task_file(task_name: str, filename: str) -> Path:
    if not task_name or Path(task_name).name != task_name or task_name in (".", ".."):
        raise ValueError(f"invalid task name: {task_name!r}")
    return TASKS_ROOT / task_name / filename


@lru_cache(maxsize=None)
def load_task_reference(task_name: str, entrypoint: str = "run"):
    """Load the task's independent oracle without importing a GPU baseline."""

    path = _task_file(task_name, "definition.json")
    namespace = {}
    exec(compile(json.loads(path.read_text())["reference"], str(path), "exec"), namespace)
    return namespace[entrypoint]


def rand_tensor(shape, dtype, device, *, positive: bool = False):
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        if positive:
            return (torch.rand(shape, dtype=torch.float32, device=device) * 0.1 + 0.01).to(dtype)
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype is torch.float8_e4m3fn:
        return torch.randn(shape, dtype=torch.float32, device=device).clamp_(-2.0, 2.0).to(dtype)
    if dtype is torch.int32:
        return torch.randint(0, 1024, shape, dtype=dtype, device=device)
    raise ValueError(f"Unsupported dtype: {dtype}")


def normalize_outputs(result):
    if result is None:
        return []
    if isinstance(result, torch.Tensor):
        return [result]
    if isinstance(result, (tuple, list)):
        return list(result)
    return [torch.as_tensor(result)]


def clone_arg(arg):
    if isinstance(arg, torch.Tensor):
        return arg.clone()
    if isinstance(arg, tuple):
        return tuple(clone_arg(x) for x in arg)
    if isinstance(arg, list):
        return [clone_arg(x) for x in arg]
    if isinstance(arg, dict):
        return {k: clone_arg(v) for k, v in arg.items()}
    return arg


def clone_args(args):
    return tuple(clone_arg(arg) for arg in args)


def time_fn(fn, args, device, warmup: int, iters: int):
    try:
        from flashinfer.testing import bench_gpu_time_with_cupti

        times = bench_gpu_time_with_cupti(
            fn=fn,
            dry_run_iters=warmup,
            repeat_iters=iters,
            input_args=tuple(args),
            cold_l2_cache=True,
            use_cuda_graph=False,
        )
        return statistics.median(times), "cupti"
    except Exception as exc:
        print(f"  timer: CUPTI unavailable ({type(exc).__name__}: {exc}); using CUDA events")

    with torch.cuda.device(device):
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(*args)
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end) / iters, "cuda_event"


def compare_outputs(
    candidate,
    baseline,
    atol: float,
    rtol: float,
    required_matched_ratio,
    required_rms_error_ratios=None,
    scale_atol_by_rms: bool = False,
    strict_outputs: bool = False,
):
    """Compare a candidate's outputs against a reference's, element by element.

    With ``scale_atol_by_rms`` the absolute tolerance of each output is ``atol``
    times the RMS of that output's reference tensor instead of ``atol`` itself,
    so tensors of very different magnitude -- an output that decays towards zero
    and a recurrent state of order one -- are held to the same relative standard.

    With ``strict_outputs`` each returned tensor must additionally be a
    densely materialized, contiguous tensor that owns exactly its own storage,
    which rejects views, aliases and padded buffers handed back in place of a
    real result.
    """
    cand = normalize_outputs(candidate)
    base = normalize_outputs(baseline)
    if len(cand) < len(base) or (strict_outputs and len(cand) != len(base)):
        return (
            False,
            float("inf"),
            float("inf"),
            float("inf"),
            0.0,
            "wrong number of outputs",
        )
    if required_rms_error_ratios is not None and len(required_rms_error_ratios) != len(
        base
    ):
        return (
            False,
            float("inf"),
            float("inf"),
            float("inf"),
            0.0,
            "wrong number of normalized RMS error limits",
        )

    max_abs = 0.0
    max_rel = 0.0
    max_rms_ratio = 0.0
    total = 0
    bad = 0
    rms_failures = []
    for output_idx, (got, ref) in enumerate(
        zip(cand[: len(base)], base, strict=True)
    ):
        if tuple(got.shape) != tuple(ref.shape):
            return (
                False,
                float("inf"),
                float("inf"),
                float("inf"),
                0.0,
                "wrong shape",
            )
        if not torch.isfinite(got.float()).all().item():
            return (
                False,
                float("inf"),
                float("inf"),
                float("inf"),
                0.0,
                "non-finite output",
            )
        if not torch.isfinite(ref.float()).all().item():
            return (
                False,
                float("inf"),
                float("inf"),
                float("inf"),
                0.0,
                "non-finite reference",
            )
        if strict_outputs:
            if got.dtype != ref.dtype:
                return (
                    False,
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    0.0,
                    "wrong dtype",
                )
            if got.device != ref.device:
                return (
                    False,
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    0.0,
                    "wrong device",
                )
            expected_bytes = got.numel() * got.element_size()
            if (
                got.layout != torch.strided
                or not got.is_contiguous()
                or got._base is not None
                or got.storage_offset() != 0
                or got.untyped_storage().nbytes() != expected_bytes
            ):
                return (
                    False,
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    0.0,
                    "output is not materialized dense",
                )
        x = got.float()
        y = ref.float()
        abs_error = torch.abs(x - y)
        rel_error = abs_error / (torch.abs(y) + 1e-8)
        output_atol = atol
        if scale_atol_by_rms:
            output_atol = atol * (
                float(torch.mean(torch.square(y)).sqrt().item()) if y.numel() else 0.0
            )
        exceeds = (abs_error > output_atol) & (rel_error > rtol)
        output_max_abs = (
            float(abs_error.max().item()) if abs_error.numel() else 0.0
        )
        max_abs = max(max_abs, output_max_abs)
        max_rel = max(max_rel, float(rel_error.max().item()) if rel_error.numel() else 0.0)
        total += abs_error.numel()
        bad += int(exceeds.sum().item())
        if required_rms_error_ratios is not None:
            if abs_error.numel():
                rms_error = torch.mean(torch.square(x - y)).sqrt()
                rms_reference = torch.mean(torch.square(y)).sqrt()
                rms_ratio = float((rms_error / (rms_reference + 1e-8)).item())
            else:
                rms_ratio = 0.0
            max_rms_ratio = max(max_rms_ratio, rms_ratio)
            rms_limit = float(required_rms_error_ratios[output_idx])
            # Match FLA assert_close: negligible max error passes immediately;
            # otherwise the normalized RMS error ratio must be strictly lower.
            if output_max_abs > 1e-6 and rms_ratio >= rms_limit:
                rms_failures.append(
                    f"output {output_idx} normalized RMS ratio "
                    f"{rms_ratio:.6g} >= {rms_limit:.6g}"
                )

    matched_ratio = 1.0 - bad / total if total else 1.0
    required = 1.0 if required_matched_ratio is None else required_matched_ratio
    failures = []
    if matched_ratio < required:
        failures.append(
            f"elementwise matched ratio {matched_ratio:.6g} < {required:.6g}"
        )
    failures.extend(rms_failures)
    ok = not failures
    return (
        ok,
        max_abs,
        max_rel,
        max_rms_ratio,
        matched_ratio,
        "ok" if ok else "; ".join(failures),
    )


def compare_tensors(
    candidate,
    reference,
    atol,
    rtol,
    required_matched_ratio=None,
    required_rms_error_ratios=None,
    *,
    cast_to_float32=False,
    reference_first=False,
    min_cosine=None,
    max_mae=None,
    absolute_bounds=None,
):
    """Return the harness's six error fields using the upstream close rules.

    ``cast_to_float32`` mirrors PRs which explicitly cast before isclose.
    ``reference_first`` mirrors assertions with the candidate as the second
    operand (the relative-tolerance scale). Diagnostics still use the oracle.
    ``absolute_bounds`` supplies AlphaMoE's per-element accumulation bound.
    Cosine is computed over each entire flattened output in FP32, as in the
    Wan hybrid quality check.

    ``required_rms_error_ratios`` adds one normalized-RMS bound per output on
    top of the upstream elementwise rule. The elementwise rule alone cannot see
    a candidate that degrades every element a little, and on a row whose
    outputs are small relative to ``atol`` it cannot see anything at all, so a
    task that cares about output quality rather than only about outliers should
    set it. The bound is scale free: the all-zero output scores exactly 1.0 on
    every row.
    """
    if required_matched_ratio not in (None, 1.0):
        raise ValueError("FlashInfer checks require all elements")
    got_outputs, ref_outputs = (
        normalize_outputs(candidate),
        normalize_outputs(reference),
    )

    def invalid(note):
        return False, float("inf"), float("inf"), float("inf"), 0.0, note

    if len(got_outputs) != len(ref_outputs):
        return invalid("wrong number of outputs")
    if absolute_bounds is not None and len(absolute_bounds) != len(ref_outputs):
        raise ValueError("one absolute bound is required per output")
    max_abs = max_rel = max_rms = 0.0
    total = bad = 0
    failures = []
    for index, (got, ref) in enumerate(zip(got_outputs, ref_outputs, strict=True)):
        if got.shape != ref.shape:
            return invalid(f"output {index}: wrong shape")
        if got.dtype != ref.dtype:
            return invalid(f"output {index}: wrong dtype ({got.dtype} != {ref.dtype})")
        if got.device != ref.device:
            return invalid(f"output {index}: wrong device")
        if not torch.isfinite(got).all().item():
            return invalid(f"output {index}: non-finite output")
        if not torch.isfinite(ref).all().item():
            return invalid(f"output {index}: non-finite reference")
        x, y = got.float(), ref.float()
        error = (x - y).abs()
        if absolute_bounds is not None:
            bound = absolute_bounds[index]
            if (
                bound.shape != ref.shape
                or not torch.isfinite(bound).all()
                or (bound < 0).any()
            ):
                raise ValueError("invalid elementwise absolute bound")
            close = error <= bound
        else:
            first, second = (x, y) if cast_to_float32 else (got, ref)
            if reference_first:
                first, second = second, first
            close = torch.isclose(
                first,
                second,
                atol=atol,
                rtol=rtol,
                equal_nan=False,
            )
        total += error.numel()
        bad += int((~close).sum().item())
        if error.numel():
            max_abs = max(max_abs, float(error.max().item()))
            max_rel = max(max_rel, float((error / (y.abs() + 1e-8)).max().item()))
            rms = error.square().mean().sqrt() / (y.square().mean().sqrt() + 1e-8)
            rms_ratio = float(rms.item())
            max_rms = max(max_rms, rms_ratio)
            if required_rms_error_ratios is not None:
                if len(required_rms_error_ratios) != len(ref_outputs):
                    raise ValueError("one RMS bound is required per output")
                limit = float(required_rms_error_ratios[index])
                # Match compare_outputs: a negligible maximum error passes
                # immediately, so an exact output never trips the ratio.
                if float(error.max().item()) > 1e-6 and rms_ratio >= limit:
                    failures.append(
                        f"output {index}: normalized RMS ratio "
                        f"{rms_ratio:.6g} >= {limit:.6g}"
                    )
        if min_cosine is not None:
            cosine = float(F.cosine_similarity(x.flatten(), y.flatten(), dim=0).item())
            if not cosine >= min_cosine:
                failures.append(f"output {index}: cosine {cosine:.9g} < {min_cosine}")
        if max_mae is not None:
            mae = float(error.mean().item())
            if not mae <= max_mae:
                failures.append(f"output {index}: MAE {mae:.9g} > {max_mae}")
    if bad:
        failures.insert(0, f"{bad}/{total} elements exceed the FlashInfer tolerance")
    matched = 1.0 - bad / total if total else 1.0
    return not failures, max_abs, max_rel, max_rms, matched, "; ".join(failures) or "ok"


def snapshot_outputs(result):
    """Detach output storage so later prepared-kernel calls cannot overwrite it."""

    return [output.detach().clone() for output in normalize_outputs(result)]


def outputs_equal_exact_values(candidate, expected):
    cand = normalize_outputs(candidate)
    base = normalize_outputs(expected)
    if len(cand) != len(base):
        return False
    return all(
        got.dtype == ref.dtype
        and tuple(got.shape) == tuple(ref.shape)
        and torch.equal(got, ref)
        for got, ref in zip(cand, base, strict=True)
    )


def poison_outputs(result):
    """Poison reusable floating-point outputs before an explicit correctness call."""

    for output in normalize_outputs(result):
        if isinstance(output, torch.Tensor) and (
            output.is_floating_point() or output.is_complex()
        ):
            output.fill_(float("nan"))


def prepare_args(raw_args, prepare_fn=None):
    """Clone one make_inputs draw and run the optional prepare step."""

    args = clone_args(raw_args)
    return args if prepare_fn is None else tuple(prepare_fn(*args))


class KernelRun:
    """One kernel bound to its prepared per-trial arguments.

    Baseline and candidate both go through this unit: run() records the latest
    return value so the poison/stability gate can inspect it, and time()
    measures the exact invocation shape used by the correctness calls.
    """

    def __init__(self, fn, args):
        self.fn = fn
        self.args = args
        self.last_output = None

    def run(self):
        # Release the previously tracked output before invoking so at most one
        # returned output set stays live across repeated (timed) calls.
        self.last_output = None
        self.last_output = self.fn(*self.args)
        return self.last_output

    def poison_last_output(self):
        if self.last_output is not None:
            poison_outputs(self.last_output)

    def time(self, device, warmup: int, iters: int):
        return time_fn(self.run, (), device, warmup, iters)


def reference_outputs(fn, args, device):
    """Produce the correctness reference: one no_grad call, synced and snapshotted."""

    with torch.no_grad():
        result = None if fn is None else fn(*args)
    torch.cuda.synchronize(device)
    return None if fn is None else snapshot_outputs(result)


@dataclass
class CorrectnessCheck:
    ok: bool
    max_abs: float
    max_rel: float
    matched: float
    note: str
    passed_runs: int
    unstable: bool = False
    max_rms_ratio: float = 0.0


class ErrorStats:
    """Worst-case error metrics accumulated across candidate runs."""

    def __init__(self):
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.max_rms_ratio = 0.0
        self.matched = 1.0

    def update(self, abs_err, rel_err, rms_ratio, matched):
        self.max_abs = max(self.max_abs, abs_err)
        self.max_rel = max(self.max_rel, rel_err)
        self.max_rms_ratio = max(self.max_rms_ratio, rms_ratio)
        self.matched = min(self.matched, matched)

    def check(self, *, ok, note, passed_runs, unstable=False) -> CorrectnessCheck:
        return CorrectnessCheck(
            ok=ok,
            max_abs=self.max_abs,
            max_rel=self.max_rel,
            matched=self.matched,
            note=note,
            passed_runs=passed_runs,
            unstable=unstable,
            max_rms_ratio=self.max_rms_ratio,
        )


def check_candidate_runs(
    *,
    candidate_runner,
    reference,
    correctness_fn,
    runs: int,
    require_repeatable_outputs: bool,
    device,
    atol: float,
    rtol: float,
    required_matched_ratio,
    required_rms_error_ratios,
    phase: str,
    repeatability_reference=None,
    compare_fn=None,
):
    """Validate consecutive calls on one prepared candidate instance."""

    stats = ErrorStats()
    first_outputs = repeatability_reference
    for run_idx in range(1, runs + 1):
        try:
            with torch.no_grad():
                candidate_runner.poison_last_output()
                got = candidate_runner.run()
            torch.cuda.synchronize(device)
        except Exception as exc:
            stats.update(float("inf"), float("inf"), float("inf"), 0.0)
            return stats.check(
                ok=False,
                note=f"{phase} run {run_idx}/{runs}: {type(exc).__name__}: {exc}",
                passed_runs=run_idx - 1,
                unstable=run_idx > 1,
            )

        if correctness_fn is not None:
            comparator = compare_outputs if compare_fn is None else compare_fn
            ok, abs_err, rel_err, rms_ratio, run_matched, note = comparator(
                got,
                reference,
                atol,
                rtol,
                required_matched_ratio,
                required_rms_error_ratios,
            )
            stats.update(abs_err, rel_err, rms_ratio, run_matched)
            if not ok:
                return stats.check(
                    ok=False,
                    note=f"{phase} run {run_idx}/{runs}: {note}",
                    passed_runs=run_idx - 1,
                    unstable=run_idx > 1,
                )

        if require_repeatable_outputs:
            if first_outputs is None:
                first_outputs = snapshot_outputs(got)
            elif not outputs_equal_exact_values(got, first_outputs):
                return stats.check(
                    ok=False,
                    note=f"{phase} run {run_idx}/{runs}: outputs are not repeatable across identical calls",
                    passed_runs=run_idx,
                    unstable=True,
                )

    return stats.check(ok=True, note="ok", passed_runs=runs)


def check_observed_outputs(
    *,
    candidate,
    reference,
    correctness_fn,
    atol: float,
    rtol: float,
    required_matched_ratio,
    required_rms_error_ratios,
    phase: str,
    compare_fn=None,
):
    """Check buffers left by a call that already ran, without overwriting them."""

    if correctness_fn is None:
        return CorrectnessCheck(True, 0.0, 0.0, 1.0, "unchecked", 1)
    comparator = compare_outputs if compare_fn is None else compare_fn
    ok, abs_err, rel_err, rms_ratio, matched, note = comparator(
        candidate,
        reference,
        atol,
        rtol,
        required_matched_ratio,
        required_rms_error_ratios,
    )
    return CorrectnessCheck(
        ok=ok,
        max_abs=abs_err,
        max_rel=rel_err,
        matched=matched,
        note="ok" if ok else f"{phase}: {note}",
        passed_runs=int(ok),
        unstable=not ok,
        max_rms_ratio=rms_ratio,
    )


def _tensors(obj):
    """Every tensor reachable from an argument tuple through tuples, lists and dict values."""

    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _tensors(item)


def perturb_inputs_(tensors, seed: int):
    """Add noise of a quarter of each tensor's own spread (its magnitude if constant), in place."""

    for index, tensor in enumerate(tensors):
        values = tensor.float()
        scale = values.std().item() or values.abs().mean().item() or 1.0
        generator = torch.Generator(device=tensor.device).manual_seed(seed + index)
        noise = torch.randn(
            values.shape, generator=generator, device=tensor.device, dtype=torch.float32
        )
        tensor.copy_((values + 0.25 * scale * noise).to(tensor.dtype))


def input_dependence_check(
    *,
    candidate_runner,
    bound_args,
    correctness_fn,
    correctness_prepare,
    device,
    atol: float,
    rtol: float,
    required_matched_ratio,
    required_rms_error_ratios,
    phase: str,
    seed: int,
    compare_fn=None,
):
    """Change a prepared candidate's inputs in place and check one more call on the new values.

    A result computed before the call (in setup, or cached on an earlier call) no longer matches.
    Only finite floating-point inputs with more than one element are changed. Returns None when
    none can be changed or one does not reach the candidate (the prepare step copied it).
    """

    if correctness_fn is None:
        return None
    inputs = [
        tensor
        for tensor in _tensors(bound_args)
        if tensor.is_floating_point() and tensor.numel() > 1 and bool(torch.isfinite(tensor).all())
    ]
    bound = {tensor.untyped_storage().data_ptr() for tensor in _tensors(candidate_runner.args)}
    if not inputs or any(t.untyped_storage().data_ptr() not in bound for t in inputs):
        return None

    perturb_inputs_(inputs, seed)
    reference = reference_outputs(
        correctness_fn, prepare_args(bound_args, correctness_prepare), device
    )
    check = check_candidate_runs(
        candidate_runner=candidate_runner,
        reference=reference,
        correctness_fn=correctness_fn,
        runs=1,
        require_repeatable_outputs=False,
        device=device,
        atol=atol,
        rtol=rtol,
        required_matched_ratio=required_matched_ratio,
        required_rms_error_ratios=required_rms_error_ratios,
        phase=phase,
        compare_fn=compare_fn,
    )
    if not check.ok:
        check.note += " (output did not follow an in-place change of the inputs)"
    return check


def summarize(rows, group_axis: str):
    valid = [r for r in rows if r["passed"] and r["speedup"] and r["speedup"] > 0]
    print()
    print("Summary")
    print(f"  passed: {sum(r['passed'] for r in rows)}/{len(rows)}")
    if not valid:
        print("  no valid speedups")
        return
    speedups = [r["speedup"] for r in valid]
    print(f"  mean speedup: {statistics.mean(speedups):.4f}x")
    print(f"  geomean speedup: {math.exp(statistics.mean(math.log(s) for s in speedups)):.4f}x")
    print(f"  min/max speedup: {min(speedups):.4f}x / {max(speedups):.4f}x")
    for suite in ["official", "large"]:
        suite_rows = [r for r in valid if r["suite"] == suite]
        if suite_rows:
            suite_speedups = [r["speedup"] for r in suite_rows]
            print(
                f"  {suite} geomean: "
                f"{math.exp(statistics.mean(math.log(s) for s in suite_speedups)):.4f}x "
                f"(n={len(suite_rows)})"
            )
    if group_axis:
        print(f"  by {group_axis}:")
        groups = {}
        for row in valid:
            groups.setdefault(row["axes"].get(group_axis), []).append(row)
        for value in sorted(groups):
            speedups_g = [r["speedup"] for r in groups[value]]
            print(
                f"    {value}: geo="
                f"{math.exp(statistics.mean(math.log(s) for s in speedups_g)):.4f}x "
                f"n={len(speedups_g)}"
            )


# Float-valued keys of the result rows appended in run_benchmark below.
# Consumers that serialize rows (stringifying non-finite floats for JSON
# transport) rely on this tuple to restore them — keep it in lockstep with
# the row dict built at rows.append.
NUMERIC_ROW_KEYS = (
    "max_abs",
    "max_rel",
    "max_rms_ratio",
    "matched",
    "baseline_ms",
    "kernel_ms",
    "speedup",
)


def run_benchmark(
    *,
    name: str,
    workloads: list,
    make_inputs,
    baseline_fn,
    candidate_fn,
    baseline_prepare_fn=None,
    candidate_prepare_fn=None,
    baseline_latency_fn=None,
    reference_fn=None,
    reference_prepare_fn=None,
    compare_fn=None,
    timing_baseline_fn=None,
    device: torch.device,
    warmup: int,
    iters: int,
    trials: int,
    atol: float,
    rtol: float,
    required_matched_ratio,
    group_axis: str,
    required_rms_error_ratios=None,
    correctness_runs: int | None = None,
    check_after_timing: bool | None = None,
    require_repeatable_outputs: bool | None = None,
    check_input_dependence: bool | None = None,
):
    if correctness_runs is None:
        correctness_runs = DEFAULT_CORRECTNESS_RUNS
    if check_after_timing is None:
        check_after_timing = DEFAULT_CHECK_AFTER_TIMING
    if check_input_dependence is None:
        check_input_dependence = config_from_env().check_input_dependence
    if require_repeatable_outputs is None:
        require_repeatable_outputs = config_from_env().require_repeatable_outputs
    if correctness_runs < 1:
        raise ValueError("correctness_runs must be at least 1")
    if require_repeatable_outputs and correctness_runs < 2:
        raise ValueError("require_repeatable_outputs needs correctness_runs >= 2")

    print(f"OP: {name}")
    print(f"workloads: {len(workloads)}")
    print(f"device: {device}")
    print(f"warmup/iters/trials: {warmup}/{iters}/{trials}")
    print(
        "correctness runs/post-timing/repeatable-outputs/input-dependence: "
        f"{correctness_runs}/{check_after_timing}/{require_repeatable_outputs}/"
        f"{check_input_dependence}"
    )
    print()

    rows = []
    timer_name = None
    for idx, entry in enumerate(workloads, start=1):
        axes = entry["axes"]
        suite = entry["suite"]
        wid = entry["id"]
        print(f"[{idx:03d}/{len(workloads):03d}] {suite}:{wid} axes={axes}")
        trial_baseline = []
        trial_kernel = []
        max_abs = 0.0
        max_rel = 0.0
        max_rms_ratio = 0.0
        matched = 1.0
        passed = True
        note = "ok"
        verdict = "PASS"
        successful_checks = 0
        entry_timer = None
        # Per-entry overrides let one suite mix probes that differ from the
        # suite defaults: an ill-conditioned regime with looser tolerances, a
        # probe checked against an independent oracle instead of the suite
        # reference, and correctness-only probes that are never timed and so
        # never enter a geomean.
        entry_atol = float(entry.get("atol", atol))
        entry_rtol = float(entry.get("rtol", rtol))
        entry_rms_ratios = entry.get(
            "required_rms_error_ratios", required_rms_error_ratios
        )
        entry_reference_fn = entry.get("reference_fn")
        entry_timed = bool(entry.get("timed", True))

        for trial_idx in range(1, trials + 1):
            try:
                args = tuple(make_inputs(entry, device))
                candidate = KernelRun(candidate_fn, prepare_args(args, candidate_prepare_fn))
                correctness_fn = baseline_fn if reference_fn is None else reference_fn
                correctness_prepare = baseline_prepare_fn if reference_fn is None else reference_prepare_fn
                if entry_reference_fn is not None:
                    correctness_fn = entry_reference_fn
                    correctness_prepare = reference_prepare_fn
                reference = reference_outputs(
                    correctness_fn, prepare_args(args, correctness_prepare), device
                )
            except Exception as exc:
                passed = False
                verdict = "ERROR"
                max_abs = float("inf")
                max_rel = float("inf")
                matched = 0.0
                note = f"preflight trial {trial_idx}: {type(exc).__name__}: {exc}"
                break

            check = check_candidate_runs(
                candidate_runner=candidate,
                reference=reference,
                correctness_fn=correctness_fn,
                runs=correctness_runs,
                require_repeatable_outputs=require_repeatable_outputs,
                device=device,
                atol=entry_atol,
                rtol=entry_rtol,
                required_matched_ratio=required_matched_ratio,
                required_rms_error_ratios=entry_rms_ratios,
                phase=f"preflight trial {trial_idx}",
                compare_fn=compare_fn,
            )
            max_abs = max(max_abs, check.max_abs)
            max_rel = max(max_rel, check.max_rel)
            max_rms_ratio = max(max_rms_ratio, check.max_rms_ratio)
            matched = min(matched, check.matched)
            if not check.ok:
                passed = False
                verdict = "FLAKY" if check.unstable or successful_checks else "FAIL"
                note = check.note
                break
            successful_checks += check.passed_runs

            if not entry_timed:
                # A correctness-only probe: its verdict is the preflight check
                # above, and it never contributes a latency or a speedup.
                break

            try:
                timing_args = tuple(make_inputs(entry, device))
                # Keep the cloned inputs the candidate is bound to, so the
                # input-dependence check can change them in place after timing.
                timing_bound = clone_args(timing_args)
                timing_candidate = KernelRun(
                    candidate_fn,
                    timing_bound
                    if candidate_prepare_fn is None
                    else tuple(candidate_prepare_fn(*timing_bound)),
                )
                timing_baseline_args = prepare_args(timing_args, baseline_prepare_fn)
                timing_base_fn = baseline_fn if timing_baseline_fn is None else timing_baseline_fn
                timing_reference = None
                if check_after_timing and correctness_fn is not None:
                    # Oracle input clones are temporary; keep only the output snapshot.
                    timing_reference = reference_outputs(
                        correctness_fn,
                        timing_baseline_args if reference_fn is None
                        else prepare_args(timing_args, reference_prepare_fn),
                        device,
                    )
                base_ms = None if baseline_latency_fn is None else baseline_latency_fn(entry)
                if base_ms is not None:
                    timer = "frozen_baseline"
                elif timing_base_fn is None:
                    timer = "candidate_only"
                else:
                    base_ms, timer = KernelRun(timing_base_fn, timing_baseline_args).time(
                        device, warmup, iters
                    )
                kernel_ms, timer = timing_candidate.time(device, warmup, iters)
            except Exception as exc:
                passed = False
                verdict = "ERROR"
                note = f"timing trial {trial_idx}: {type(exc).__name__}: {exc}"
                break

            timing_observed_outputs = None
            if check_after_timing:
                timing_observed = timing_candidate.last_output
                observed_check = check_observed_outputs(
                    candidate=timing_observed,
                    reference=timing_reference,
                    correctness_fn=correctness_fn,
                    atol=entry_atol,
                    rtol=entry_rtol,
                    required_matched_ratio=required_matched_ratio,
                    required_rms_error_ratios=entry_rms_ratios,
                    phase=f"post-timing trial {trial_idx} observed output",
                    compare_fn=compare_fn,
                )
                max_abs = max(max_abs, observed_check.max_abs)
                max_rel = max(max_rel, observed_check.max_rel)
                max_rms_ratio = max(
                    max_rms_ratio, observed_check.max_rms_ratio
                )
                matched = min(matched, observed_check.matched)
                if not observed_check.ok:
                    passed = False
                    verdict = "FLAKY"
                    note = observed_check.note
                    break
                successful_checks += observed_check.passed_runs
                if require_repeatable_outputs:
                    timing_observed_outputs = snapshot_outputs(timing_observed)

            if check_after_timing:
                check = check_candidate_runs(
                    candidate_runner=timing_candidate,
                    reference=timing_reference,
                    correctness_fn=correctness_fn,
                    runs=correctness_runs,
                    require_repeatable_outputs=require_repeatable_outputs,
                    device=device,
                    atol=entry_atol,
                    rtol=entry_rtol,
                    required_matched_ratio=required_matched_ratio,
                    required_rms_error_ratios=entry_rms_ratios,
                    phase=f"post-timing trial {trial_idx}",
                    repeatability_reference=timing_observed_outputs,
                    compare_fn=compare_fn,
                )
                max_abs = max(max_abs, check.max_abs)
                max_rel = max(max_rel, check.max_rel)
                max_rms_ratio = max(max_rms_ratio, check.max_rms_ratio)
                matched = min(matched, check.matched)
                if not check.ok:
                    passed = False
                    verdict = "FLAKY"
                    note = check.note
                    break
                successful_checks += check.passed_runs

            if check_input_dependence:
                # Last: it changes the timed instance's inputs. A result computed
                # in setup passes every check above, which reuse setup's inputs.
                dependence = input_dependence_check(
                    candidate_runner=timing_candidate,
                    bound_args=timing_bound,
                    correctness_fn=correctness_fn,
                    correctness_prepare=correctness_prepare,
                    device=device,
                    atol=entry_atol,
                    rtol=entry_rtol,
                    required_matched_ratio=required_matched_ratio,
                    required_rms_error_ratios=entry_rms_ratios,
                    phase=f"input-dependence trial {trial_idx}",
                    seed=trial_idx,
                    compare_fn=compare_fn,
                )
                if dependence is None:
                    if trial_idx == 1 and correctness_fn is not None:
                        print("  input-dependence: skipped (a float input is not bound in place)")
                else:
                    max_abs = max(max_abs, dependence.max_abs)
                    max_rel = max(max_rel, dependence.max_rel)
                    max_rms_ratio = max(max_rms_ratio, dependence.max_rms_ratio)
                    matched = min(matched, dependence.matched)
                    if not dependence.ok:
                        passed = False
                        verdict = "FAIL"
                        note = dependence.note
                        break
                    successful_checks += dependence.passed_runs

            timer_name = timer_name or timer
            # Recorded per row: a downstream consumer's only way to detect
            # a silent CUPTI-to-events degradation.
            entry_timer = timer
            if base_ms is not None:
                trial_baseline.append(base_ms)
            trial_kernel.append(kernel_ms)

        if passed and trial_baseline:
            verdict = "STABLE" if correctness_runs > 1 or check_after_timing else "PASS"
            baseline_ms = statistics.mean(trial_baseline)
            kernel_ms = statistics.mean(trial_kernel)
            speedup = baseline_ms / kernel_ms
            print(f"  baseline: {baseline_ms:.6f} ms")
            print(f"  kernel:   {kernel_ms:.6f} ms")
            print(f"  speedup:  {speedup:.4f}x")
            print(
                f"  correct:  PASS verdict={verdict} checks={successful_checks} "
                f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
                f"max_rms_ratio={max_rms_ratio:.3e} matched={matched:.4f}"
            )
        elif passed and not entry_timed:
            verdict = "STABLE" if correctness_runs > 1 else "PASS"
            baseline_ms = None
            kernel_ms = None
            speedup = None
            print(
                f"  correct:  PASS verdict={verdict} untimed checks={successful_checks} "
                f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
                f"max_rms_ratio={max_rms_ratio:.3e} matched={matched:.4f}"
            )
        elif passed and trial_kernel:
            verdict = "STABLE" if correctness_runs > 1 or check_after_timing else "PASS"
            baseline_ms = None
            kernel_ms = statistics.mean(trial_kernel)
            speedup = None
            print("  baseline: unavailable")
            print(f"  kernel:   {kernel_ms:.6f} ms")
            print(
                f"  correct:  PASS verdict={verdict} checks={successful_checks} "
                f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
                f"max_rms_ratio={max_rms_ratio:.3e} matched={matched:.4f}"
            )
        else:
            baseline_ms = None
            kernel_ms = None
            speedup = None
            print(
                f"  correct:  FAIL verdict={verdict} {note} max_abs={max_abs:.3e} "
                f"max_rel={max_rel:.3e} max_rms_ratio={max_rms_ratio:.3e} "
                f"matched={matched:.4f}"
            )

        rows.append(
            {
                "suite": suite,
                "id": wid,
                "axes": axes,
                "passed": passed,
                "verdict": verdict,
                "note": note,
                "correctness_checks": successful_checks,
                "max_abs": max_abs,
                "max_rel": max_rel,
                "max_rms_ratio": max_rms_ratio,
                "matched": matched,
                "baseline_ms": baseline_ms,
                "kernel_ms": kernel_ms,
                "speedup": speedup,
                "timer": entry_timer,
                "timed": entry_timed,
            }
        )
        print()

    if timer_name:
        print(f"timer: {timer_name}")
    summarize(rows, group_axis)
    return rows
