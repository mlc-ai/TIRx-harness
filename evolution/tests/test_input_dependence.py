"""Check on CPU that the input-dependence check catches results computed before the timed call."""

import pytest
import torch

from evolution.benchmark import adapter

adapter._bench_repo_on_path()
from flashinfer_bench_evolve.benchmark_common import (  # noqa: E402
    KernelRun,
    clone_args,
    input_dependence_check,
    perturb_inputs_,
)


def prepare(mode):
    """A GEMM candidate's setup: returns the timed callable and its bound data."""

    def setup(a, b):
        d, stored, cache = torch.empty(a.shape[0], b.shape[0]), a @ b.T, {}

        def run():
            if mode in ("honest", "copies_input"):
                torch.matmul(a, b.T, out=d)
            elif mode == "precompute":
                d.copy_(stored)
            else:  # cache keyed on pointers, or also on version counters
                key = (a.data_ptr(), b.data_ptr())
                key += (a._version, b._version) if mode == "version_cache" else ()
                cache.setdefault(key, a @ b.T)
                d.copy_(cache[key])

        a = a.clone() if mode == "copies_input" else a
        return run, {"A": a, "B": b, "D": d}

    return setup


def check(mode):
    torch.manual_seed(0)
    bound = clone_args((torch.randn(32, 16), torch.randn(24, 16)))
    candidate = KernelRun(lambda run, data: (run(), data["D"])[1], tuple(prepare(mode)(*bound)))
    candidate.run()  # stands in for the warmup, timing and post-timing calls
    return input_dependence_check(
        candidate_runner=candidate,
        bound_args=bound,
        correctness_fn=lambda a, b: a @ b.T,
        correctness_prepare=None,
        device=torch.device("cpu"),
        atol=1e-4,
        rtol=1e-4,
        required_matched_ratio=1.0,
        required_rms_error_ratios=None,
        phase="input-dependence trial 1",
        seed=1,
    )


@pytest.fixture(autouse=True)
def no_cuda_sync(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)


def test_honest_candidate_passes():
    assert check("honest").ok


@pytest.mark.parametrize("mode", ["precompute", "pointer_cache"])
def test_result_computed_before_the_call_fails(mode):
    result = check(mode)
    assert not result.ok and "did not follow an in-place change" in result.note


def test_version_keyed_cache_is_not_caught():
    # Known limit: the in-place change bumps the version counter, so the cache recomputes.
    assert check("version_cache").ok


def test_skipped_when_prepare_copies_an_input():
    assert check("copies_input") is None


def test_constant_input_is_changed():
    ones = torch.ones(8)
    perturb_inputs_([ones], seed=1)
    assert ones.unique().numel() > 1
