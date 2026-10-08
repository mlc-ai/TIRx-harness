"""Every checker entry path must resolve one shared default search budget."""

from __future__ import annotations

from tirx_harness.numsim import api, checker_runner


def test_engine_phase_and_checker_runner_resolve_identical_defaults() -> None:
    assert checker_runner.default_coverage_bounds() == api.default_coverage_bounds()
    assert checker_runner.default_resource_limits() == api.default_resource_limits()


def test_run_synccheck_phase_resolves_the_shared_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def record(self, module, inputs, **kwargs):
        captured.update(kwargs)
        return "phase-result"

    monkeypatch.setattr(api.Engine, "_run_native_synccheck_phase", record)

    assert api.Engine().run_synccheck_phase(object(), {}) == "phase-result"

    assert captured["coverage_bounds"] == checker_runner.default_coverage_bounds()
    assert captured["resource_limits"] == checker_runner.default_resource_limits()
    # The production budget, not a 10,000x smaller private one.
    assert captured["resource_limits"].max_schedules == 10_000
    assert captured["coverage_bounds"] == api.CoverageBounds(2, 2)


def test_explicit_phase_arguments_still_win_over_the_shared_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def record(self, module, inputs, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(api.Engine, "_run_native_synccheck_phase", record)

    bounds = api.CoverageBounds(0, 0)
    limits = api.ResourceLimits(
        max_schedules=1,
        max_backtrack_nodes=1,
        max_events_per_run=1,
        max_total_events=1,
        max_loop_steps=1,
        max_wall_time_ms=1,
        max_diagnostic_bytes=1,
    )

    api.Engine().run_synccheck_phase(object(), {}, coverage_bounds=bounds, resource_limits=limits)

    assert captured["coverage_bounds"] is bounds
    assert captured["resource_limits"] is limits


def test_checker_runner_defaults_are_delegates_not_a_second_copy(monkeypatch) -> None:
    """A second literal copy would keep the old values when ``api`` changes."""

    sentinel_bounds = api.CoverageBounds(7, 7)
    sentinel_limits = api.ResourceLimits(
        max_schedules=13,
        max_backtrack_nodes=13,
        max_events_per_run=13,
        max_total_events=13,
        max_loop_steps=13,
        max_wall_time_ms=13,
        max_diagnostic_bytes=13,
    )
    monkeypatch.setattr(api, "default_coverage_bounds", lambda: sentinel_bounds)
    monkeypatch.setattr(api, "default_resource_limits", lambda: sentinel_limits)

    assert checker_runner.default_coverage_bounds() is sentinel_bounds
    assert checker_runner.default_resource_limits() is sentinel_limits
