#!/usr/bin/env python3
"""Reproduce FA4 normalized RMS checks for one task on a Blackwell GPU."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evolution/benchmark"))


def fp32_oracle(q, k, v, causal, scale, grad_out=None):
    """Independent SDPA MATH oracle, one batch/head at a time to bound memory."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention

    backward = grad_out is not None
    result = tuple(torch.empty_like(q, dtype=torch.float32) for _ in range(3 if backward else 1))
    context = torch.enable_grad() if backward else torch.no_grad()
    with context, sdpa_kernel(SDPBackend.MATH):
        for batch in range(q.shape[0]):
            for head in range(q.shape[2]):
                inputs = tuple(
                    x[batch, :, head, :]
                    .float()
                    .detach()
                    .contiguous()[None, None]
                    .requires_grad_(backward)
                    for x in (q, k, v)
                )
                out = scaled_dot_product_attention(
                    *inputs, dropout_p=0.0, is_causal=bool(causal), scale=float(scale)
                )
                values = (
                    torch.autograd.grad(
                        out, inputs, grad_out[batch, :, head, :].float()[None, None]
                    )
                    if backward
                    else (out,)
                )
                for target, value in zip(result, values, strict=True):
                    target[batch, :, head, :].copy_(value[0, 0].detach())
                del out, values, inputs
    return result


def comparison_record(bench, actual, reference, labels, compare_outputs):
    import torch

    tolerances = (bench.ATOL, bench.RTOL, bench.REQUIRED_MATCHED_RATIO)
    legacy = compare_outputs(actual, reference, *tolerances)
    gated = compare_outputs(actual, reference, *tolerances, bench.REQUIRED_RMS_ERROR_RATIOS)

    def fields(values):
        return dict(
            passed=values[0],
            max_abs=values[1],
            max_rel=values[2],
            max_rms_ratio=values[3],
            matched=values[4],
            note=values[5],
        )

    per_output = []
    for name, got, ref, limit in zip(
        labels, actual, reference, bench.REQUIRED_RMS_ERROR_RATIOS, strict=True
    ):
        error = got.float() - ref.float()
        rms_error = torch.mean(error.square()).sqrt().item()
        rms_reference = torch.mean(ref.float().square()).sqrt().item()
        per_output.append(
            dict(
                name=name,
                rms_error_ratio=rms_error / (rms_reference + 1e-8),
                rms_reference=rms_reference,
                max_abs=error.abs().max().item(),
                rms_limit=limit,
            )
        )
    return dict(legacy=fields(legacy), gated=fields(gated), per_output=per_output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("forward", "backward"), required=True)
    parser.add_argument(
        "--max-workloads",
        type=int,
        default=0,
        help="0 runs all five official rows; 1 is a smoke run",
    )
    parser.add_argument(
        "--output", type=Path, help="also save all records, including the summary, as JSONL"
    )
    parser.add_argument(
        "--preflight", action="store_true", help="check GPU and runtime imports only"
    )
    args = parser.parse_args()
    if args.max_workloads < 0:
        parser.error("--max-workloads must be nonnegative")
    records, failed, cases, false_positives, counts = [], 0, 0, {}, {}

    def emit(record):
        def json_safe(value):
            if isinstance(value, float) and not math.isfinite(value):
                return str(value)
            if isinstance(value, dict):
                return {key: json_safe(item) for key, item in value.items()}
            if isinstance(value, tuple | list):
                return [json_safe(item) for item in value]
            return value

        record = json_safe(record)
        records.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)

    try:
        import torch
        from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
        from flashinfer_bench_evolve.benchmark_common import compare_outputs

        if not callable(_flash_attn_fwd) or not callable(_flash_attn_bwd):
            raise RuntimeError("FA4 forward/backward operators are unavailable")
        if not torch.cuda.is_available():
            raise RuntimeError("A B200 CUDA GPU is required")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        capability = torch.cuda.get_device_capability(device)
        if capability[0] < 10:
            raise RuntimeError(f"Expected a Blackwell GPU; device capability is {capability}")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        tasks = (args.task,)
        package_versions = {}
        for package in ("torch", "flash-attn-4", "quack-kernels", "nvidia-cutlass-dsl"):
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = "not installed as this distribution"
        source_hashes = {}
        for name in tasks:
            path = (
                REPO_ROOT
                / "evolution/benchmark/flashinfer_bench_evolve/tasks"
                / f"fa4_{name}/benchmark.py"
            )
            source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            source_revision = revision.stdout.strip() or "unavailable"
        except OSError:
            source_revision = "unavailable"
        emit(
            dict(
                kind="metadata",
                python=sys.version,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device),
                capability=capability,
                package_versions=package_versions,
                tasks=tasks,
                max_workloads=args.max_workloads,
                source_hashes=source_hashes,
                source_revision=source_revision,
                oracle=(
                    "FP32 PyTorch SDPA MATH; sequential batch/head; "
                    "backward autograd; TF32 disabled"
                ),
                forward_runtime="validation-only substitution: FA4 CuTe _flash_attn_fwd",
                repository_forward_baseline="flash_attn.flash_attn_func (FA2)",
                exact_unmodified_forward_baseline_validated=False,
                backward_runtime="unmodified fa4_backward.baseline.run",
            )
        )
        if not args.preflight:
            for task in tasks:
                bench = importlib.import_module(
                    f"flashinfer_bench_evolve.tasks.fa4_{task}.benchmark"
                )
                labels = ("out",) if task == "forward" else ("dq", "dk", "dv")
                if bench.REQUIRED_RMS_ERROR_RATIOS != (0.2,) * len(labels):
                    raise RuntimeError(f"Unexpected {task} normalized RMS limits")
                workloads = bench.make_workloads(
                    bench.BenchConfig(
                        include_official=True, shape_mode="all", max_official=args.max_workloads
                    )
                )
                if args.max_workloads == 0 and len(workloads) != 5:
                    raise RuntimeError(
                        f"Expected five official {task} workloads, got {len(workloads)}"
                    )
                counts[task] = len(workloads)
                false_positives[task] = 0
                for entry in workloads:
                    inputs = bench.make_inputs(entry, device)
                    with torch.no_grad():
                        if task == "forward":
                            q, k, v, causal, scale = inputs
                            out, *_ = _flash_attn_fwd(
                                q=q,
                                k=k,
                                v=v,
                                causal=causal,
                                softmax_scale=float(scale),
                                return_lse=True,
                            )
                            actual = (out.detach(),)
                            grad_out = None
                        else:
                            q, k, v, out, grad_out, lse, causal, scale = inputs
                            baseline = importlib.import_module(
                                "flashinfer_bench_evolve.tasks.fa4_backward.baseline"
                            )
                            actual = tuple(value.detach() for value in baseline.run(*inputs))
                    torch.cuda.synchronize(device)
                    reference = fp32_oracle(q, k, v, causal, scale, grad_out)
                    variants = [("genuine", None, 1.0)]
                    for factor in (0.0, 0.5):
                        for index, label in enumerate(labels):
                            variants.append((f"{factor:g}:{label}", (index,), factor))
                        if len(labels) > 1:
                            variants.append((f"{factor:g}:all", tuple(range(len(labels))), factor))
                    for variant, indices, factor in variants:
                        candidate = tuple(
                            value * factor if indices is not None and index in indices else value
                            for index, value in enumerate(actual)
                        )
                        record = comparison_record(
                            bench, candidate, reference, labels, compare_outputs
                        )
                        expected = indices is None
                        ok = record["gated"]["passed"] is expected
                        if expected:
                            ok &= record["legacy"]["passed"]
                        else:
                            ok &= "normalized RMS ratio" in record["gated"]["note"]
                            false_positives[task] += int(record["legacy"]["passed"])
                        failed += int(not ok)
                        cases += 1
                        emit(
                            dict(
                                kind="case",
                                task=task,
                                workload=entry["raw"]["uuid"],
                                axes=entry["axes"],
                                variant=variant,
                                expected_gate_pass=expected,
                                expectation_met=bool(ok),
                                **record,
                            )
                        )
                    del inputs, actual, reference, candidate, q, k, v, out, grad_out
                    if task == "backward":
                        del lse
                # Causal rows may already fail the old gate. The complete suite
                # must exhibit a false positive, but the first-row smoke need not.
                if args.max_workloads == 0 and false_positives[task] == 0:
                    failed += 1
                    emit(
                        dict(
                            kind="error",
                            task=task,
                            error="Full suite showed no legacy false positives",
                        )
                    )
    except Exception as exc:
        failed += 1
        emit(dict(kind="error", error=f"{type(exc).__name__}: {exc}"))
        traceback.print_exc(file=sys.stderr)
    summary = dict(
        kind="summary",
        passed=failed == 0,
        failed_expectations=failed,
        cases=cases,
        workload_counts=counts,
        legacy_false_positives=false_positives,
        preflight_only=args.preflight,
    )
    emit(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
