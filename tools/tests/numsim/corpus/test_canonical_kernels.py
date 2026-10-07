"""Numerical correctness for every single-GPU canonical registry kernel."""

from __future__ import annotations

from time import perf_counter

import pytest

from tirx_kernels.registry import discover_kernels
from tirx_harness.numsim import Engine, run_case, transpile
from tirx_harness.numsim.api import ExecutionSubset
from tests.numsim.corpus.canonical_cases import (
    CANONICAL_KERNEL_CASES,
    CANONICAL_KERNEL_MANIFEST,
    MULTI_GPU_ONLY_CANONICAL_KERNELS,
)
from tests.numsim.corpus.kernels.deepgemm import prepare_mega_moe_case
from tests.numsim.support._tirx_kernels import load_tirx_kernel


_MEGA_MOE_MAX_CONFIG_LABEL = "t8192_m8192_h7168_i3072_e384_k6_g1"
# 2026-09-18: this exact Engine.run took 130.602536/136.797175s on the 224-CPU runner,
# with 16 engine workers and pytest -n16/worksteal alongside the Racecheck gate.
# Both runs passed idle-host preflight. Keep the existing 140s guard despite
# its narrow 2.3% headroom; do not widen it to fit a nominal noise margin.
# Preparation/build are excluded; kernels use d11cb9e.
_MEGA_MOE_NUMSIM_WALL_TIME_LIMIT_SECONDS = 140.0


def test_canonical_manifest_exactly_matches_single_gpu_registry() -> None:
    discovered = set(discover_kernels(strict=True))
    manifested = {case.name for case in CANONICAL_KERNEL_MANIFEST}

    assert len(CANONICAL_KERNEL_MANIFEST) == len(manifested)
    assert manifested.isdisjoint(MULTI_GPU_ONLY_CANONICAL_KERNELS)
    assert discovered == manifested | MULTI_GPU_ONLY_CANONICAL_KERNELS


@pytest.mark.parametrize("entry", CANONICAL_KERNEL_CASES, ids=lambda entry: entry.name)
def test_canonical_kernel_matches_independent_numerical_oracle(entry) -> None:
    report = run_case(entry.prepare(), engine=Engine(max_workers=entry.engine_max_workers))

    report.require_ok()
    assert report.verdict == entry.expected_verdict
    if report.verdict == "clean":
        assert report.diagnostics == []
    else:
        assert report.diagnostics
        assert {item["status"] for item in report.diagnostics} == {"review"}
        assert {item["kind"] for item in report.diagnostics} == {"uninitialized_read"}
        if entry.numsim_review_count is not None:
            assert len(report.diagnostics) == entry.numsim_review_count


def test_fastcu_k96_uses_nonzero_scales_across_tmem_words():
    from tests.numsim.corpus.kernels.blockscaled_gemm import prepare_fastcu_nvfp4_gemm_case

    run_case(prepare_fastcu_nvfp4_gemm_case(k=768)).require_ok()


def test_rubin_gather_swiglu_reuses_all_eight_a_stages():
    from tests.numsim.corpus.kernels.blockscaled_gemm import prepare_gather_swiglu_rubin_case

    case = prepare_gather_swiglu_rubin_case(k=2304)
    run_case(case).require_ok()
    for checker in ("synccheck", "racecheck"):
        module = transpile(case.kernel, _analysis_checker=checker)
        result = getattr(Engine(), f"run_{checker}_phase")(
            module, case.args,
        )
        assert result.verdict == "clean", result.to_dict()


@pytest.mark.performance
def test_maximum_mega_moe_numsim_completes_within_performance_budget() -> None:
    module = load_tirx_kernel("sm100_fp8_fp4_mega_moe")
    config_entry = next(
        config for config in module.CONFIGS if config["label"] == _MEGA_MOE_MAX_CONFIG_LABEL
    )
    case = prepare_mega_moe_case(config_entry)
    compiled = transpile(case.kernel)
    expected = case.reference()

    started = perf_counter()
    result = Engine(max_workers=16, native_loop_iteration_budget=10_000_000).run(
        compiled,
        case.args,
        subset=case.subset,
        assumptions=case.assumptions,
        outputs=case.outputs,
    )
    elapsed = perf_counter() - started

    # Named registry configurations deliberately bind exact-zero activations:
    # `y` therefore checks zero preservation, while the stats oracle checks
    # the exact bincount of all 49,152 routes. Separate nonzero evidence is
    # narrower: the default Mega MoE case has four routes, while raw-TCGEN and
    # paired-GPU differentials use CTA/row/K-varying operands and scales to
    # exercise the production K-major mapping directly.
    result.assert_close(expected, case.comparisons)
    assert result.verdict == "clean"
    assert result.diagnostics == []
    assert elapsed < _MEGA_MOE_NUMSIM_WALL_TIME_LIMIT_SECONDS, (
        f"{_MEGA_MOE_MAX_CONFIG_LABEL} NumSim took {elapsed:.3f}s; "
        f"budget is {_MEGA_MOE_NUMSIM_WALL_TIME_LIMIT_SECONDS:.1f}s"
    )


def test_flashmla_small_topk_task_steal_matches_independent_numerical_oracle() -> None:
    entry = next(
        item
        for item in CANONICAL_KERNEL_CASES
        if item.name == "sparse_flashmla_prefill_head128_small_topk_phase1"
    )
    case = entry.prepare()
    # Leave the second logical cluster non-resident.  The resident CTA pair
    # must claim it through CLC and still produce both query rows.
    case.subset = ExecutionSubset(cluster_ids=[0])

    report = run_case(case)

    report.require_ok()
    assert report.verdict == "clean"
    assert report.diagnostics == []
