"""Full-launch native Synccheck gate for canonical kernels with runnable cases."""

from __future__ import annotations

import pytest

from tests.numsim.corpus.canonical_cases import CANONICAL_KERNEL_CASES
from tirx_harness import numsim
from tirx_harness.numsim.api import ExecutionSubset


def _resource_limits(*, max_diagnostic_bytes: int = 16 * 1024 * 1024) -> numsim.ResourceLimits:
    return numsim.ResourceLimits(
        max_schedules=16,
        max_backtrack_nodes=100_000,
        max_events_per_run=4_000_000,
        max_total_events=8_000_000,
        max_loop_steps=4_000_000,
        max_wall_time_ms=180_000,
        max_diagnostic_bytes=max_diagnostic_bytes,
    )


@pytest.mark.parametrize("entry", CANONICAL_KERNEL_CASES, ids=lambda entry: entry.name)
def test_native_synccheck_validates_canonical_kernel(
    entry,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    case = entry.prepare()
    module = numsim.transpile(
        case.kernel,
        cache_dir=tmp_path_factory.getbasetemp() / "canonical-kernels-synccheck-cache",
        _analysis_capable=True,
        _default_generated_opt_level=3,
    )
    engine = numsim.Engine(max_workers=entry.engine_max_workers)
    if entry.synccheck_phase_expected_verdicts:
        assert len(entry.synccheck_phase_expected_verdicts) == len(module.spec.kernels)

    for phase_index in range(len(module.spec.kernels)):
        result = engine.run_synccheck_phase(
            module,
            case.args,
            phase_index=phase_index,
            subset=case.subset,
            advance_prefix=True,
            coverage_bounds=numsim.CoverageBounds(
                max_warp_preemptions=0,
                max_completion_schedule_deviations=0,
            ),
            resource_limits=_resource_limits(
                max_diagnostic_bytes=entry.synccheck_max_diagnostic_bytes,
            ),
        )
        payload = result.to_dict()
        context = (entry.name, phase_index, payload)

        expected_verdict = (
            entry.synccheck_phase_expected_verdicts[phase_index]
            if entry.synccheck_phase_expected_verdicts
            else entry.expected_verdict
        )
        assert result.verdict == expected_verdict, context
        assert result.findings == [], context
        if expected_verdict == "review":
            assert result.advisories, context
            assert {item["kind"] for item in result.advisories} == {"uninitialized_read"}, context
        else:
            assert result.advisories == [], context
        assert payload["incomplete"] == [], context
        assert payload["execution_error"] is None, context
        assert payload["stats"]["completed_task_count"] == payload["stats"]["task_count"], context
        assert payload["coverage"]["status"] == "complete_within_bounds", context
        assert payload["coverage"]["eligible_for_clean"] is True, context
        assert payload["search"]["incomplete_reason"] is None, context


def test_native_synccheck_validates_flashmla_small_topk_task_steal(
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
        cache_dir=tmp_path_factory.getbasetemp() / "flashmla-task-steal-synccheck-cache",
        _analysis_capable=True,
    )

    result = numsim.Engine(max_workers=16).run_synccheck_phase(
        module,
        case.args,
        phase_index=0,
        subset=subset,
        assumptions=case.assumptions,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=0,
            max_completion_schedule_deviations=0,
        ),
        resource_limits=_resource_limits(),
    )
    payload = result.to_dict()

    assert result.verdict == "incomplete", payload
    assert result.findings == [], payload
    assert result.advisories == [], payload
    assert payload["incomplete"] == [
        {"kind": "analysis_incomplete", "reason": "subset_execution", "selected_warp_count": 32, "total_warp_count": 64}
    ], payload
    assert payload["execution_error"] is None, payload
    assert payload["stats"]["completed_task_count"] == payload["stats"]["task_count"], payload
    assert payload["coverage"]["status"] == "complete_within_bounds", payload
    assert payload["coverage"]["eligible_for_clean"] is True, payload
    assert payload["coverage"]["termination"] == {
        "kind": "worklist_exhausted",
        "resource_limit": None,
    }, payload
    assert payload["search"]["incomplete_reason"] is None, payload
