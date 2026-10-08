"""Qwen3.8 task selection, remote packaging and stateful scoring on CPU."""

import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from evolution.benchmark import adapter
from evolution.preparation.declare import load_task

shared = adapter.import_benchmark("qwen38_decode").shared


@pytest.mark.parametrize(
    "group,prefix,suffix,pin,segments",
    [
        ("decode", "D", "b128_p4096_n1", "D4", ((128, 4096, 1),)),
        ("prefill", "P", "b1_p0_n32768", "P3", ((1, 0, 32768),)),
        ("expand", "S", "b64_p8192_n256", "S6", ((64, 8192, 256),)),
    ],
)
def test_registered_group_and_pin(monkeypatch, group, prefix, suffix, pin, segments):
    monkeypatch.setenv("BENCH_OFFICIAL_SHAPE_MODE", "max")
    monkeypatch.setenv("BENCH_MAX_OFFICIAL", "1")
    monkeypatch.setenv("BENCH_INCLUDE_OFFICIAL", "0")
    for single in (False, True):
        name = f"qwen38_{group}" + (f"_{suffix}" if single else "")
        task = load_task(adapter.REPO_ROOT / "evolution/tasks" / f"{name}.yaml")
        key = adapter.workload_key(task.workload_dir)
        package, warmup, repeat, mode = adapter.PACKAGED[key]
        _, rows, candidate = adapter.plan(
            package,
            adapter.CANDIDATES_ROOT / key,
            "baseline",
            warmup=warmup,
            repeat=repeat,
            shape_mode=mode,
        )
        assert candidate is None
        assert [row["id"] for row in rows] == (
            [pin] if single else [f"{prefix}{i}" for i in range(1, 14)]
        )
        if single:
            assert rows[0]["axes"]["segments"] == segments


def test_remote_bundle_loads_without_checkout(tmp_path):
    from evolution.remote import kcoral_remote

    # Reproduce the worker's in-memory import, with no repository on sys.path.
    for group in ("decode", "prefill", "expand"):
        name = f"qwen38_{group}"
        source = kcoral_remote._bundle(name)
        source += f"""
sys.meta_path.insert(0, _BundleImporter(SOURCES))
bench = importlib.import_module("flashinfer_bench_evolve.tasks.{name}.benchmark")
print(json.dumps(bench.make_workloads()))
"""
        script = tmp_path / f"{name}.py"
        script.write_text(source)
        result = subprocess.run(
            [sys.executable, "-I", str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        rows = json.loads(result.stdout)
        assert len(rows) == 13
        assert {row["raw"]["group"] for row in rows} == {group}
        if group == "expand":
            assert rows[11]["axes"]["segments"] == [[1, 98304, 2048], [128, 4096, 1]]


@pytest.mark.parametrize("fault", [None, "prefix", "nonfinite"])
def test_stateful_blocks_check_once_and_restore_matching_starts(monkeypatch, fault):
    from flashinfer_bench_evolve.tasks.qwen38.shapes import Case

    cache = SimpleNamespace(
        lengths=[2],
        kv={0: (torch.zeros(3, 1), torch.zeros(3, 1))},
        conv={1: torch.ones(1)},
        recurrent={1: torch.full((1,), 2.0)},
        verify_conv_storage={1: torch.zeros(1)},
        verify_recurrent={1: torch.zeros(1)},
    )
    prepared = {"locations": torch.tensor([2])}
    starts = []
    samples = []
    cache_checks = []
    check_cache = shared._check_cache

    def record_cache_check(cache, expected):
        cache_checks.append(True)
        return check_cache(cache, expected)

    monkeypatch.setattr(shared, "_check_cache", record_cache_check)

    def forward(cache, implementation="baseline", **unused):
        starts.append(
            (
                implementation,
                cache.conv[1].item(),
                cache.recurrent[1].item(),
                cache.kv[0][0][2].item(),
                cache.verify_recurrent[1].item(),
            )
        )
        cache.conv[1].add_(3)
        cache.recurrent[1].add_(4)
        for tensor in cache.kv[0]:
            tensor[2] = 5
        cache.verify_conv_storage[1].fill_(6)
        cache.verify_recurrent[1].fill_(7)
        # Logits evolve with state, making a stale single-step comparison fail.
        return torch.full((1, 248320), cache.recurrent[1].item())

    def timed(fn):
        output = fn()
        samples.append(starts[-1])
        return output, {"gpu_ms": 2.0}

    model = SimpleNamespace(
        _forward=forward, _prepare_step=lambda *args, **kwargs: (None, None, prepared)
    )
    monkeypatch.setattr(shared.baseline, "new_cache", lambda *args: cache)
    monkeypatch.setattr(shared.baseline, "prime", lambda *args: None)
    monkeypatch.setattr(shared.baseline, "timed", timed)

    def prepare(model, cache, prepared):
        def launch():
            output = forward(cache, implementation="candidate")
            if fault == "prefix":
                cache.kv[0][0][0] = 1  # Correct logits still must fail the state gate.
            if fault == "nonfinite" and cache.recurrent[1].item() > 6:
                output.fill_(float("nan"))
            return output

        return launch

    cfg = replace(shared.default_config(), trials=2)
    case = Case("P1", "prefill", ((1, 2, 1),))
    if fault:
        with pytest.raises(AssertionError, match="cache 0" if fault == "prefix" else "nonfinite"):
            shared._run_case(model, case, cfg, 123, shared.tirx_run, prepare)
    else:
        row = shared._run_case(model, case, cfg, 123, shared.tirx_run, prepare)
        assert row["passed"] and row["speedup"] == 1.0
        assert row["correctness_checks"] == 1
        assert samples == [
            (name, 1 + 3 * i, 2 + 4 * i, 0 if i == 0 else 5, 0 if i == 0 else 7)
            for name in ("baseline", "candidate", "candidate", "baseline")
            for i in range(cfg.warmup + cfg.iters)
        ]
    assert len(cache_checks) == 1


@pytest.mark.parametrize("lengths", [(0, 0, 0, 0), (0, 5, 0, 7), (5, 5, 5, 5)])
def test_state_restore_preserves_prefix_and_zeros_empty_requests(monkeypatch, lengths):
    tensors = [
        torch.arange(8, dtype=torch.float32).reshape(4, 2),
        torch.arange(24, dtype=torch.float32).reshape(4, 2, 3),
    ]
    for tensor in tensors:
        for slot, length in enumerate(lengths):
            if not length:
                tensor[slot].zero_()
    expected = [tensor.clone() for tensor in tensors]
    cache = SimpleNamespace(
        lengths=list(lengths),
        conv={0: tensors[0]},
        recurrent={0: tensors[1]},
        kv={1: (torch.ones(4, 1), torch.ones(4, 1))},
        verify_conv_storage={},
        verify_recurrent={},
    )
    snapshot = shared.baseline.snapshot
    copied_elements = []

    def record_snapshot(tensor):
        copied_elements.append(tensor.numel())
        return snapshot(tensor)

    monkeypatch.setattr(shared.baseline, "snapshot", record_snapshot)
    saved = shared.baseline.save_state(cache)
    assert sum(copied_elements) == sum(bool(length) for length in lengths) * 8
    for _ in range(2):
        for tensor in tensors:
            tensor.fill_(99)
        shared.baseline.restore(cache, saved, {"locations": torch.tensor([3])})
        for actual, reference in zip(tensors, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for tensor in cache.kv[1]:
            torch.testing.assert_close(tensor[:, 0], torch.tensor([1.0, 1.0, 1.0, 0.0]))
