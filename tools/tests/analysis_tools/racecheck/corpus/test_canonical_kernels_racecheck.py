"""Full-launch native Racecheck gate for canonical kernels with runnable cases."""

from __future__ import annotations

from collections import Counter
from time import perf_counter

import pytest

from tests.numsim.corpus.canonical_cases import CANONICAL_KERNEL_CASES
from tests.numsim.corpus.kernels.deepgemm import prepare_mega_moe_case
from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tirx_harness import numsim
from tirx_harness.numsim.api import ExecutionSubset


# Keep the full persistent grid fixed across hosts, including CPU-only runs.
_MEGA_MOE_RACECHECK_NUM_SMS = 148
_MEGA_MOE_RACECHECK_WORKERS = 16
_MEGA_MOE_RACECHECK_MEDIUM_CONFIG_LABEL = "t64_h2048_i1536_e96_k4_g1"
_MEGA_MOE_RACECHECK_LARGE_CONFIG_LABEL = "t64_h4096_i1536_e96_k4_g1"
# 2026-09-20 UTC, 224-CPU Xeon Platinum 8570, TVM f776cef / kernels d11cb9e:
# pytest -n16/worksteal, 16 engine workers, full 148-SM grid. Three serial
# calibration jobs: one six-case performance selection and two Racecheck-only
# jobs. All 15 Racecheck cases passed; preflight global idle >=50% and external
# CPU use <=50% throughout their measured windows. Original five-case maxima:
# 1.595815/1.868570/2.143231/1.765354/16.243485s.
# User-requested ceiling: 1.5 times each current maximum. Preserve stricter
# existing caps with min(old_cap, floor(maximum * 1.5 * 100) / 100); rounding
# down to 0.01s keeps the ceiling at most 1.5x. Only two_tokens tightens, from
# 2.4s to 2.39s. See pr_comment.md for all observations and the shared TVM
# collection error in the six-case selection; it was not a green package job.
# Before adding large_moe, independent -n16 verification: 5 passed, in case order
# 1.693139/1.765358/2.153020/1.684898/14.284364s. Keep this verification separate
# from the three-round calibration above; all observations remain recorded.
# Preparation and compilation are excluded; correctness checks are unchanged.
_MEGA_MOE_RACECHECK_CASES = [
    pytest.param("p1_tok2_h1024_i512_e2_k1_bm16", 2.39, 2, 8, id="two_tokens"),
    pytest.param("p1_tok16_h1024_i512_e2_k2_bm32", 2.3, 4, 8, id="sixteen_tokens"),
    pytest.param("t8_h1024_i512_e24_k2_g1", 2.7, 24, 96, id="twenty_four_experts"),
    pytest.param("p1_tok2_h1024_i512_e2_k1_bm16_s1", 2.3, 2, 8, id="shared_expert"),
    pytest.param(_MEGA_MOE_RACECHECK_MEDIUM_CONFIG_LABEL, 16.7, 1536, 128, id="medium_moe"),
    # 2026-09-20 UTC, same host/TVM/kernels and -n16/worksteal configuration:
    # full six-case jobs measured 28.561079/31.629900/31.330127s, with accepted
    # CPU load. User-requested 35s cap gives 10.7% headroom over the maximum,
    # tightening the initial 1.5x calibration cap of 47.44s.
    # Focused -n16 verification at 35s: 25.737327s, 1 passed, accepted CPU load.
    # All 2368 tasks and 409256 operations complete. Existing caps stay fixed;
    # the medium-case overruns in these jobs remain recorded in pr_comment.md.
    pytest.param(_MEGA_MOE_RACECHECK_LARGE_CONFIG_LABEL, 35.0, 1536, 128, id="large_moe"),
]


@pytest.mark.parametrize("entry", CANONICAL_KERNEL_CASES, ids=lambda entry: entry.name)
def test_native_racecheck_validates_canonical_kernel(
    entry,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    if entry.name == "alphamoe_fp8_blockscale_qwen3next":
        pytest.xfail("Racecheck does not yet support AlphaMoE's wait over multiple tagged records")
    case = entry.prepare()
    module = numsim.transpile(
        case.kernel,
        cache_dir=tmp_path_factory.getbasetemp() / "canonical-kernels-racecheck-cache",
        _analysis_capable=True,
        _analysis_checker="racecheck",
    )
    engine = numsim.Engine(max_workers=entry.engine_max_workers)
    if entry.racecheck_phase_expected_verdicts:
        assert len(entry.racecheck_phase_expected_verdicts) == len(module.spec.kernels)
    if entry.racecheck_phase_review_counts:
        assert len(entry.racecheck_phase_review_counts) == len(module.spec.kernels)

    for phase_index in range(len(module.spec.kernels)):
        result = engine.run_racecheck_phase(
            module,
            case.args,
            phase_index=phase_index,
            subset=case.subset,
            advance_prefix=True,
        )
        payload = result.to_dict()
        context = (entry.name, phase_index, payload)

        if entry.racecheck_phase_expected_verdicts:
            expected_verdict = entry.racecheck_phase_expected_verdicts[phase_index]
        else:
            expected_verdict = entry.racecheck_expected_verdict or entry.expected_verdict
        assert result.verdict == expected_verdict, context
        if expected_verdict == "review":
            review_items = [item for item in payload["findings"] if item["status"] == "review"]
            review_items.extend(payload["advisories"])
            assert review_items, context
            expected_kinds = entry.racecheck_review_kinds or frozenset({"uninitialized_read"})
            assert {item["kind"] for item in review_items} == expected_kinds, context
            expected_counts = (
                entry.racecheck_phase_review_counts[phase_index]
                if entry.racecheck_phase_review_counts
                else entry.racecheck_review_counts
            )
            if expected_counts:
                assert Counter(item["kind"] for item in review_items) == dict(expected_counts), (
                    context
                )
            assert all(item["status"] == "review" for item in payload["findings"]), context
            assert all(
                item["access_pair"] == "read_write"
                for item in review_items
                if item["kind"] == "tmem_lifetime_review"
            ), context
            if entry.name == "flash_attention4":
                assert payload["advisories"] == [], context
                assert all(
                    finding["prior"]["space"] == "tmem"
                    and finding["current"]["space"] == "tmem"
                    and finding["prior"]["access_kind"] == "read"
                    and finding["current"]["access_kind"] == "write"
                    and finding["ordering_failure"] == "async_lifetime_not_drained"
                    for finding in payload["findings"]
                ), context
        else:
            assert result.findings == [], context
            assert result.advisories == [], context
        assert payload["incomplete"] == [], context
        assert payload["execution_error"] is None, context
        assert payload["stats"]["completed_task_count"] == payload["stats"]["task_count"], context


@pytest.mark.performance
@pytest.mark.parametrize(
    "config_label,limit_seconds,alias_occurrences,alias_bytes", _MEGA_MOE_RACECHECK_CASES
)
def test_real_mega_moe_racecheck_completes_within_performance_budget(
    config_label: str,
    limit_seconds: float,
    alias_occurrences: int,
    alias_bytes: int,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    record_property,
) -> None:
    monkeypatch.setenv("TIRX_DEEPGEMM_NUM_SMS_OVERRIDE", str(_MEGA_MOE_RACECHECK_NUM_SMS))
    monkeypatch.delenv("NUMSIM_PROFILE", raising=False)
    entry = next(item for item in CANONICAL_KERNEL_CASES if item.name == "sm100_fp8_fp4_mega_moe")
    module = load_tirx_kernel("sm100_fp8_fp4_mega_moe")
    configs = {config["label"]: config for config in module.CONFIGS}
    # Keep the canonical construction, but halve this registry shape's hidden
    # width to keep the medium case's execution time in the 10-30s range.
    configs[_MEGA_MOE_RACECHECK_MEDIUM_CONFIG_LABEL] = {
        **configs["t64_h4096_i1536_e96_k4_g1"],
        "hidden": 2048,
        "label": _MEGA_MOE_RACECHECK_MEDIUM_CONFIG_LABEL,
    }
    case = prepare_mega_moe_case(configs[config_label])
    assert case.subset is None
    compiled = numsim.transpile(
        case.kernel,
        cache_dir=tmp_path_factory.getbasetemp() / "mega-moe-racecheck-performance-cache",
        _analysis_capable=True,
        _analysis_checker="racecheck",
    )
    assert len(compiled.spec.kernels) == 1

    started = perf_counter()
    result = numsim.Engine(
        max_workers=_MEGA_MOE_RACECHECK_WORKERS,
        native_loop_iteration_budget=10_000_000,
    ).run_racecheck_phase(compiled, case.args, phase_index=0, subset=case.subset)
    elapsed = perf_counter() - started
    payload = result.to_dict()
    record_property("racecheck_elapsed_seconds", elapsed)
    record_property("racecheck_limit_seconds", limit_seconds)
    record_property("racecheck_config", config_label)
    record_property("racecheck_completed_tasks", payload["stats"]["completed_task_count"])
    record_property(
        "racecheck_completion_operations", payload["stats"]["completion_operation_count"]
    )

    assert result.verdict == entry.racecheck_expected_verdict, payload
    assert result.findings == [], payload
    assert Counter(item["kind"] for item in result.advisories) == dict(
        entry.racecheck_review_counts
    ), payload
    assert payload["incomplete"] == [], payload
    assert payload["execution_error"] is None, payload
    assert payload["stats"]["worker_count"] == _MEGA_MOE_RACECHECK_WORKERS, payload
    assert payload["stats"]["profile"] == {}, payload
    assert (
        payload["stats"]["completed_task_count"]
        == payload["stats"]["task_count"]
        == _MEGA_MOE_RACECHECK_NUM_SMS * 16
    ), payload
    # Verify the expected review's footprint and count as well as completion.
    (advisory,) = result.advisories
    assert advisory["space"] == "shared", payload
    assert advisory["reader_buffer"] == "anonymous_buffer_29", payload
    assert advisory["writer_buffer"] == "anonymous_buffer_8", payload
    assert advisory["occurrences"] == alias_occurrences, payload
    assert advisory["overlaps"] == [
        {"allocation_id": 0, "byte_offset": 0, "byte_len": alias_bytes, "byte_end": alias_bytes}
    ], payload
    assert elapsed < limit_seconds, (
        f"{config_label} Racecheck took {elapsed:.3f}s; "
        f"budget is {limit_seconds:.3f}s"
    )


def test_native_racecheck_validates_flashmla_small_topk_task_steal(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    entry = next(
        item
        for item in CANONICAL_KERNEL_CASES
        if item.name == "sparse_flashmla_prefill_head128_small_topk_phase1"
    )
    case = entry.prepare()
    subset = ExecutionSubset(cluster_ids=[0])
    module = numsim.transpile(
        case.kernel,
        cache_dir=tmp_path_factory.getbasetemp() / "flashmla-task-steal-racecheck-cache",
        _analysis_capable=True,
        _analysis_checker="racecheck",
    )

    result = numsim.Engine(max_workers=16).run_racecheck_phase(
        module, case.args, phase_index=0, subset=subset
    )
    payload = result.to_dict()

    assert result.verdict == "incomplete", payload
    assert result.findings == [], payload
    assert Counter(item["kind"] for item in result.advisories) == dict(
        entry.racecheck_review_counts
    ), payload
    assert payload["incomplete"] == [
        {"kind": "analysis_incomplete", "reason": "subset_execution", "selected_warp_count": 32, "total_warp_count": 64}
    ], payload
    assert payload["execution_error"] is None, payload
    assert payload["stats"]["completed_task_count"] == payload["stats"]["task_count"], payload
