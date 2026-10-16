"""Compare FP8, FP32, and promoted sparse MLA against independent references.

This synchronous TIRx example follows the FP8 probability path in
tirx_kernels.mla.mla_dsv4_multishape. It retains sparse gather, online softmax,
attention sinks, and 448-scaled E4M3 probabilities. The optimized production
implementation uses TCGEN/TMA and is run separately by --gpu.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib.metadata import version
from time import perf_counter

import ml_dtypes
import numpy as np

from tirx_harness import numsim
from tvm.script import tirx as T


@dataclass(frozen=True)
class AttentionCase:
    query: np.ndarray
    cache: np.ndarray
    indices: np.ndarray
    lengths: np.ndarray
    sinks: np.ndarray
    score_scale: float

    def arguments(self) -> dict[str, np.ndarray]:
        return {
            "query": self.query.view(np.uint8),
            "cache": self.cache.view(np.uint8),
            "indices": self.indices,
            "lengths": self.lengths,
            "sinks": self.sinks,
            "output": np.zeros(self.query.shape, np.float32),
        }


def make_case(*, queries=4, heads=8, selected=256, dimension=512, seed=7):
    if min(queries, heads, selected, dimension) <= 0 or dimension % 32:
        raise ValueError("positive dimensions and a head dimension divisible by 32 are required")
    rng = np.random.default_rng(seed)
    pool_rows = max(512, selected * 2)
    query = rng.normal(size=(queries, heads, dimension)).astype(ml_dtypes.float8_e4m3fn)
    cache = rng.normal(size=(pool_rows, dimension)).astype(ml_dtypes.float8_e4m3fn)
    indices = np.stack([rng.permutation(pool_rows)[:selected] for _ in range(queries)]).astype(
        np.int32
    )
    indices[:, 13::37] = -1
    lengths = np.maximum(0, selected - np.arange(queries) * 7).astype(np.int32)
    if queries > 2:
        lengths[-1] = 0
    sinks = rng.uniform(-2, 2, heads).astype(np.float32)
    return AttentionCase(query, cache, indices, lengths, sinks, float(np.float32(dimension**-0.55)))


def reference(case: AttentionCase) -> np.ndarray:
    """Evaluate dense attention in FP64 from the same already quantized inputs."""
    query = case.query.astype(np.float64)
    cache = case.cache.astype(np.float64)
    active = (case.indices >= 0) & (
        np.arange(case.indices.shape[1])[None, :] < case.lengths[:, None]
    )
    gathered = cache[np.maximum(case.indices, 0)]
    scores = np.einsum("qhd,qkd->qhk", query, gathered) * case.score_scale
    scores = np.where(active[:, None, :], scores, -np.inf)
    maximum = np.maximum(scores.max(axis=-1), case.sinks[None, :])
    weights = np.exp(scores - maximum[..., None])
    denominator = weights.sum(axis=-1) + np.exp(case.sinks[None, :] - maximum)
    return np.einsum("qhk,qkd->qhd", weights, gathered) / denominator[..., None]


def quantized_reference(case: AttentionCase) -> np.ndarray:
    """Evaluate the blockwise FP8 probability rounding with FP64 accumulation."""
    query = case.query.astype(np.float64)
    cache = case.cache.astype(np.float64)
    maximum = np.broadcast_to(case.sinks.astype(np.float64), query.shape[:2]).copy()
    denominator = np.full(query.shape[:2], 448.0)
    numerator = np.zeros_like(query)
    for start in range(0, case.indices.shape[1], 64):
        indices = case.indices[:, start : start + 64]
        rows = cache[np.maximum(indices, 0)]
        valid = (indices >= 0) & (
            np.arange(start, start + indices.shape[1])[None, :] < case.lengths[:, None]
        )
        scores = np.einsum("qhd,qkd->qhk", query, rows) * case.score_scale
        scores = np.where(valid[:, None, :], scores, -np.inf)
        updated = np.maximum(maximum, scores.max(axis=-1))
        correction = np.exp(maximum - updated)
        weights = np.exp(scores - updated[..., None]) * 448.0
        rounded = weights.astype(ml_dtypes.float8_e4m3fn).astype(np.float64)
        numerator = numerator * correction[..., None] + np.einsum("qhk,qkd->qhd", rounded, rows)
        denominator = denominator * correction + weights.sum(axis=-1)
        maximum = updated
    return numerator / denominator[..., None]


def make_kernel(case: AttentionCase, *, probability_dtype="float8_e4m3fn", wrong=False):
    if probability_dtype not in {"float8_e4m3fn", "float32"}:
        raise ValueError("probabilities must use float8_e4m3fn or float32")
    queries, heads, dimension = case.query.shape
    selected = case.indices.shape[1]
    pool_rows = case.cache.shape[0]
    score_scale = case.score_scale
    output_scale = 1.01 if wrong else 1.0

    @T.prim_func
    def sparse_mla(
        query: T.Buffer((queries, heads, dimension), "float8_e4m3fn"),
        cache: T.Buffer((pool_rows, dimension), "float8_e4m3fn"),
        indices: T.Buffer((queries, selected), "int32"),
        lengths: T.Buffer((queries,), "int32"),
        sinks: T.Buffer((heads,), "float32"),
        output: T.Buffer((queries, heads, dimension), "float32"),
    ):
        T.device_entry()
        task = T.cta_id([queries * heads])
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        query_index = task // heads
        head = task % heads
        accumulator = T.alloc_buffer((dimension // 32,), "float32", scope="local")
        scores = T.alloc_buffer((2,), "float32", scope="local")
        probabilities = T.alloc_buffer((2,), probability_dtype, scope="local")
        maximum = T.alloc_buffer((1,), "float32", scope="local")
        denominator = T.alloc_buffer((1,), "float32", scope="local")
        maximum[0] = sinks[head]
        denominator[0] = T.float32(448)
        for feature in T.serial(dimension // 32):
            accumulator[feature] = T.float32(0)
        for block in T.serial((selected + 63) // 64):
            for half in T.serial(2):
                slot = block * 64 + half * 32 + lane
                scores[half] = T.float32(-float("inf"))
                if slot < selected and slot < lengths[query_index]:
                    row = indices[query_index, slot]
                    if row >= 0:
                        scores[half] = T.float32(0)
                        for feature in T.serial(dimension):
                            scores[half] = (
                                T.cast(query[query_index, head, feature], "float32")
                                * T.cast(cache[row, feature], "float32")
                                + scores[half]
                            )
                        scores[half] = scores[half] * T.float32(score_scale)
            block_maximum: T.float32 = T.cuda.warp_max(T.max(scores[0], scores[1]))
            next_maximum: T.float32 = T.max(maximum[0], block_maximum)
            rescale: T.float32 = T.exp(maximum[0] - next_maximum)
            weight_first: T.float32 = T.exp(scores[0] - next_maximum) * T.float32(448)
            weight_second: T.float32 = T.exp(scores[1] - next_maximum) * T.float32(448)
            denominator[0] = denominator[0] * rescale + T.cuda.warp_sum(
                weight_first + weight_second
            )
            maximum[0] = next_maximum
            probabilities[0] = T.cast(weight_first, probability_dtype)
            probabilities[1] = T.cast(weight_second, probability_dtype)
            for feature in T.serial(dimension // 32):
                accumulator[feature] = accumulator[feature] * rescale
            for position in T.serial(64):
                weight: T.float32 = T.cuda.__shfl_sync(
                    T.uint32(0xFFFFFFFF),
                    T.cast(probabilities[position // 32], "float32"),
                    position % 32,
                    32,
                )
                slot = block * 64 + position
                if slot < selected and slot < lengths[query_index]:
                    row = indices[query_index, slot]
                    if row >= 0:
                        for feature in T.serial(dimension // 32):
                            accumulator[feature] = (
                                weight * T.cast(cache[row, feature * 32 + lane], "float32")
                                + accumulator[feature]
                            )
        for feature in T.serial(dimension // 32):
            output[query_index, head, feature * 32 + lane] = (
                accumulator[feature] / denominator[0] * T.float32(output_scale)
            )

    return sparse_mla


def error_metrics(actual, expected):
    error = actual.astype(np.float64) - expected
    error_norm = float(np.linalg.norm(error))
    reference_norm = float(np.linalg.norm(expected))
    relative_l2 = (
        error_norm / reference_norm if reference_norm else (0.0 if not error_norm else float("inf"))
    )
    return {
        "max_abs": float(np.max(np.abs(error))),
        "rms": float(np.sqrt(np.mean(error**2))),
        "relative_l2": relative_l2,
    }


def run_gpu(kernel, case: AttentionCase):
    import torch

    import tvm

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
    arguments = {
        name: torch.from_numpy(value.copy()).cuda() for name, value in case.arguments().items()
    }
    for name in ("query", "cache"):
        arguments[name] = arguments[name].view(torch.float8_e4m3fn)
    executable(*(arguments[parameter.name] for parameter in kernel.params))
    torch.cuda.synchronize()
    return arguments["output"].cpu().numpy()


def run_production_gpu(case: AttentionCase):
    import torch

    from tirx_kernels.mla import mla_dsv4_multishape

    if case.query.shape[2] != 512 or case.query.shape[1] not in (8, 16, 32, 64, 128):
        raise ValueError("the production MLA requires dimension=512 and heads in 8,16,32,64,128")
    if case.indices.shape[1] < 128:
        raise ValueError("the production MLA requires at least 128 selected slots")
    arguments = {
        name: torch.from_numpy(value.copy()).cuda() for name, value in case.arguments().items()
    }
    inputs = {
        "query": arguments["query"].view(torch.float8_e4m3fn),
        "swa_kv_cache": arguments["cache"].view(torch.float8_e4m3fn),
        "compressed_kv_cache": None,
        "sparse_indices": arguments["indices"],
        "sparse_topk_lens": arguments["lengths"],
        "sinks": arguments["sinks"],
        "output": torch.empty(case.query.shape, dtype=torch.bfloat16, device="cuda"),
        "bmm1_scale": case.score_scale,
        "bmm2_scale": 1.0,
    }
    launch = mla_dsv4_multishape._candidate_setup(
        inputs, case.query.shape[0], case.indices.shape[1]
    )
    launch()
    torch.cuda.synchronize()
    actual = inputs["output"].float().cpu().numpy()
    np.testing.assert_allclose(actual, reference(case), atol=2e-2, rtol=6e-2)
    return actual


def run_example(case: AttentionCase, *, workers=4, gpu=False):
    expected = reference(case)
    quantized = quantized_reference(case)
    records = []
    for name, dtype, precision, wrong in (
        ("fp8", "float8_e4m3fn", "native", False),
        ("fp32", "float32", "native", False),
        ("high", "float8_e4m3fn", "high", False),
        ("wrong_fp8", "float8_e4m3fn", "native", True),
        ("wrong_high", "float8_e4m3fn", "high", True),
    ):
        kernel = make_kernel(case, probability_dtype=dtype, wrong=wrong)
        started = perf_counter()
        module = numsim.transpile(kernel, precision=precision)
        compile_seconds = perf_counter() - started
        started = perf_counter()
        result = numsim.Engine(max_workers=workers).run(
            module, case.arguments(), outputs=("output",)
        )
        execute_seconds = perf_counter() - started
        if result.verdict != "clean":
            raise AssertionError(result.diagnostics)
        actual = result.outputs["output"]
        np.testing.assert_array_equal(actual[case.lengths == 0], 0.0)
        metrics = error_metrics(actual, expected)
        record = {
            "mode": name,
            **metrics,
            "passes_loose_5_percent_l2": metrics["relative_l2"] < 0.05,
            "passes_strict": bool(np.allclose(actual, expected, rtol=1e-5, atol=1e-6)),
            "compile_seconds": compile_seconds,
            "execute_seconds": execute_seconds,
        }
        if name == "fp8":
            np.testing.assert_allclose(actual, quantized, rtol=2e-5, atol=2e-6)
            record["quantized_reference_max_abs"] = error_metrics(actual, quantized)["max_abs"]
        if name in ("fp32", "high"):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        if name == "high":
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        if name == "wrong_high" and record["passes_strict"]:
            raise AssertionError("the strict comparison failed to catch the wrong output scale")
        if gpu and name in ("fp8", "fp32"):
            device_output = run_gpu(kernel, case)
            np.testing.assert_allclose(device_output, actual, rtol=2e-5, atol=2e-6)
            if name == "fp8":
                np.testing.assert_allclose(device_output, quantized, rtol=2e-5, atol=2e-6)
            else:
                np.testing.assert_allclose(device_output, expected, rtol=1e-5, atol=1e-6)
            record["gpu_error"] = error_metrics(device_output, expected)
            record["gpu_vs_numsim_max_abs"] = error_metrics(device_output, actual)["max_abs"]
        records.append(record)
    if gpu:
        records.append(
            {
                "mode": "production_fp8_gpu",
                "package_version": version("tirx-kernels"),
                "output_dtype": "bfloat16",
                "passes_production_tolerances": True,
                **error_metrics(run_production_gpu(case), expected),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--selected", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--gpu", action="store_true", help="also run both implementations on B200/SM100a"
    )
    args = parser.parse_args()
    case = make_case(
        queries=args.queries,
        heads=args.heads,
        selected=args.selected,
        dimension=args.dimension,
        seed=args.seed,
    )
    report = {
        "algorithm": "DeepSeek-V4 sparse MLA",
        "seed": args.seed,
        "query_shape": list(case.query.shape),
        "cache_shape": list(case.cache.shape),
        "selected": args.selected,
        "block_size": 64,
        "workers": args.workers,
        "input_dtype": "float8_e4m3fn",
        "strict_tolerances": {"rtol": 1e-5, "atol": 1e-6},
        "results": run_example(case, workers=args.workers, gpu=args.gpu),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
