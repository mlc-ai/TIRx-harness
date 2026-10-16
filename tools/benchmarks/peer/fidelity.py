"""Baseline-faithful operations for CUTLASS's all-gather GEMM launch schedule."""

from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import math
from pathlib import Path


def prepare_tirx(rank, world, shape, config, workspace, shared):
    """Prepare TIRx replacements without changing the baseline host schedule."""
    import torch
    import tvm_ffi
    from tirx_kernels.runner import compile_kernel

    from benchmarks.multimem.origins import max_active_clusters
    from benchmarks.peer.origins import peer_pointers
    from ported.cutlass.sm100_all_gather_gemm import cutlass_gemm, release_flag, sync_nvl_devices

    cluster = tuple(config["cluster"])
    executables = {
        gated: compile_kernel(
            cutlass_gemm(
                shape["m"], shape["n"], shape["k"], shape["ab"], shape["c"],
                use_2cta=config["use_2cta"], mma_tiler=tuple(config["mma_tiler"]),
                cluster=cluster, use_tma_store=config["use_tma_store"],
                max_active_clusters=max_active_clusters(cluster[0] * cluster[1]),
                gated_a_load=gated,
            ).func,
            cuda_compile_mode="nvcc",
        )
        for gated in (False, True)
    }
    synchronize = compile_kernel(sync_nvl_devices(world), cuda_compile_mode="nvcc")
    release = compile_kernel(release_flag(world), cuda_compile_mode="nvcc")
    addresses = torch.tensor(peer_pointers(shared["arrival"]), dtype=torch.int64, device="cuda")

    def as_bytes(tensor):
        return tensor.view(torch.uint8).reshape(-1)

    arguments = []
    for values in workspace:
        by_peer = []
        for peer in range(world):
            source = values["a"] if peer == rank else shared["pulled"][peer]
            source = as_bytes(source)
            destination = as_bytes(values["out"][peer * shape["m"]:(peer + 1) * shape["m"]])
            by_peer.append((source, source, as_bytes(values["b"]), destination,
                            shared["gate"][peer:peer + 1].view(torch.uint32)))
        arguments.append(by_peer)

    dummy_flags = torch.zeros(world, dtype=torch.int32, device="cuda")
    dummy_counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    dummy_addresses = torch.full(
        (world,), dummy_counter.data_ptr(), dtype=torch.int64, device="cuda")
    warm_arguments = list(arguments[0][rank])
    warm_arguments[-1] = dummy_flags[:1].view(torch.uint32)
    with tvm_ffi.use_torch_stream():
        synchronize(rank, dummy_addresses, dummy_flags, dummy_counter)
        release(dummy_flags, 0)
        for executable in executables.values():
            executable(*warm_arguments)
    torch.cuda.synchronize()

    def launch_gemm(workspace_index, peer, stream):
        with torch.cuda.stream(stream), tvm_ffi.use_torch_stream():
            executables[peer != rank](*arguments[workspace_index][peer])

    def launch_sync(stream):
        with torch.cuda.stream(stream), tvm_ffi.use_torch_stream():
            synchronize(rank, addresses, shared["gate"], shared["arrival"])

    def launch_release(peer, stream):
        with torch.cuda.stream(stream), tvm_ffi.use_torch_stream():
            release(shared["gate"], peer)

    return launch_gemm, launch_sync, launch_release


def compare_rank(rank, world, shape_name, configurations, idle_s):
    import torch
    import torch.distributed as dist

    from benchmarks.multimem import common
    from benchmarks.peer import all_gather_gemm as benchmark, origins

    gc.disable()
    shape = benchmark.SHAPES_BY_NAME[shape_name]
    workspace, shared, reference = benchmark.allocate(rank, world, shape, torch.device("cuda", rank))
    rows = []
    for config in configurations:
        faulthandler.dump_traceback_later(benchmark.CANDIDATE_TIMEOUT_S, exit=True)
        implementations = {
            implementation: origins.cutlass_launchers(
                rank, world, shape, config, workspace, shared, implementation=implementation)[0]
            for implementation in ("cutlass", "tirx")
        }
        errors, outputs = {}, {}
        for name, closures in implementations.items():
            errors[name] = benchmark.run_checked(closures[0], workspace[0], reference)
            outputs[name] = workspace[0]["out"].clone()
        equal = torch.tensor([torch.equal(outputs["cutlass"], outputs["tirx"])],
                             dtype=torch.int32, device="cuda")
        dist.all_reduce(equal, op=dist.ReduceOp.MIN)
        row = {
            "shape": shape_name, "origin": "cutlass", "comparison_kind": "port_fidelity",
            "config": benchmark.config_key(config), "errors": errors,
            "bitwise_equal": bool(equal.item()),
        }
        if not any(errors.values()) and row["bitwise_equal"]:
            times = common.slowest_rank(common.time_launches(
                implementations, launches=benchmark.LAUNCHES, trials=benchmark.TRIALS,
                idle_s=idle_s))
            row.update(origin_us=times["cutlass"], tirx_us=times["tirx"],
                       ratio=times["tirx"] / times["cutlass"])
        if rank == 0:
            print(json.dumps(row), flush=True)
        rows.append(row)
        faulthandler.cancel_dump_traceback_later()
        del implementations, outputs
        benchmark._collect_garbage()
    return rows


def summarize(rows):
    """Use the baseline's best configuration for both sides of each pair."""
    latest = {
        (row["shape"], row["config"]): row for row in rows
        if row.get("comparison_kind") == "port_fidelity" and row.get("origin") == "cutlass"
    }
    summaries = []
    for shape in sorted({shape for shape, _ in latest}):
        group = [row for (name, _), row in latest.items() if name == shape]
        timed = [row for row in group if "ratio" in row]
        summary = {
            "shape": shape, "origin": "cutlass", "configs": len(group),
            "correct": len(timed) == len(group) and all(
                row.get("bitwise_equal")
                and row.get("errors") == {"cutlass": None, "tirx": None}
                for row in group),
        }
        if timed:
            best = min(timed, key=lambda row: row["origin_us"])
            summary.update(config=best["config"], origin_us=best["origin_us"],
                           tirx_us=best["tirx_us"], ratio=best["ratio"])
        summaries.append(summary)
    return summaries


def validate_summary(summary, shapes):
    if not shapes:
        raise RuntimeError("all-gather fidelity validation requires shapes")
    observed = {row["shape"] for row in summary}
    failures = [f"missing {shape}" for shape in sorted(set(shapes) - observed)]
    for row in summary:
        if row["shape"] not in shapes:
            continue
        ratio = row.get("ratio", math.inf)
        if not row["correct"]:
            failures.append(f"{row['shape']}: numerical comparison failed")
        elif not math.isfinite(ratio) or ratio <= 0 or ratio > 1 / 0.99:
            failures.append(f"{row['shape']}: latency ratio {ratio:g} does not meet 99% parity")
    if failures:
        raise RuntimeError("All-gather fidelity validation failed: " + "; ".join(failures))


def main():
    from benchmarks.multimem import common
    from benchmarks.peer import all_gather_gemm as benchmark

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--shapes", nargs="+", default=[shape["name"] for shape in benchmark.SHAPES])
    parser.add_argument("--log", type=Path,
                        default=benchmark.RESULTS / "all_gather_gemm_tuning.jsonl")
    parser.add_argument("--out", type=Path,
                        default=benchmark.RESULTS / "all_gather_gemm_fidelity.jsonl")
    parser.add_argument("--top", type=int, default=benchmark.ORIGIN_TOP)
    parser.add_argument("--idle-s", type=float, default=0.5)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = []
    for shape in args.shapes:
        configurations = benchmark.best_configs(args.log, "cutlass", shape, args.top)
        if not configurations:
            continue
        rows = common.spawn("benchmarks.peer.fidelity:compare_rank", args.world,
                            shape, configurations, args.idle_s)
        completed.extend(rows)
        with args.out.open("a") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    summary = summarize(completed)
    print(common.table([row for row in summary if "ratio" in row], [
        ("shape", "shape", "s"), ("origin_us", "CUTLASS (us)", ".2f"),
        ("tirx_us", "TIRx (us)", ".2f"), ("ratio", "TIRx / CUTLASS", ".3f"),
    ]))
    validate_summary(summary, args.shapes)


if __name__ == "__main__":
    main()
