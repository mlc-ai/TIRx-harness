"""Root checker implementations backed by the NumSim native engine."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .checker_report import RaceReport, SyncCheckReport
from .checker_runner import run_native_checker

if TYPE_CHECKING:
    from tvm.tir import PrimFunc


def synccheck(kernel: PrimFunc, inputs: dict | list[dict] | None = None) -> SyncCheckReport:
    """Run native synchronization analysis for one concrete invocation.

    A list of per-rank input dicts launches one rank per entry.
    """
    return _run_synccheck(kernel, inputs=inputs)


def racecheck(kernel: PrimFunc, inputs: dict | list[dict] | None = None) -> RaceReport:
    """Run native memory-race analysis for one concrete invocation.

    A list of per-rank input dicts launches one rank per entry.
    """
    return _run_racecheck(kernel, inputs=inputs)


def _run_synccheck(
    kernel: PrimFunc,
    inputs: dict | list[dict] | None = None,
    *,
    coverage_bounds=None,
    resource_limits=None,
    subset=None,
    cache_dir: str | Path | None = None,
    max_workers: int | Literal["auto"] = 8,
    max_polls: int | None = None,
    max_transitions: int | None = None,
    native_loop_iteration_budget: int = 1_000_000,
    native_loop_reschedule_quantum: int = 64,
) -> SyncCheckReport:
    """Run Synccheck with internal execution policy overrides."""
    run = run_native_checker(
        "synccheck",
        kernel,
        inputs=inputs,
        coverage_bounds=coverage_bounds,
        resource_limits=resource_limits,
        subset=subset,
        cache_dir=cache_dir,
        max_workers=max_workers,
        max_polls=max_polls,
        max_transitions=max_transitions,
        native_loop_iteration_budget=native_loop_iteration_budget,
        native_loop_reschedule_quantum=native_loop_reschedule_quantum,
    )
    return SyncCheckReport.from_native(run.payload)


def _run_racecheck(
    kernel: PrimFunc,
    inputs: dict | list[dict] | None = None,
    *,
    subset=None,
    cache_dir: str | Path | None = None,
    max_workers: int | Literal["auto"] = 8,
    max_polls: int | None = None,
    max_transitions: int | None = None,
    native_loop_iteration_budget: int = 1_000_000,
    native_loop_reschedule_quantum: int = 64,
) -> RaceReport:
    """Run Racecheck with internal execution policy overrides."""
    run = run_native_checker(
        "racecheck",
        kernel,
        inputs=inputs,
        subset=subset,
        cache_dir=cache_dir,
        max_workers=max_workers,
        max_polls=max_polls,
        max_transitions=max_transitions,
        native_loop_iteration_budget=native_loop_iteration_budget,
        native_loop_reschedule_quantum=native_loop_reschedule_quantum,
    )
    return RaceReport.from_native(run.payload)


__all__ = ["racecheck", "synccheck"]
