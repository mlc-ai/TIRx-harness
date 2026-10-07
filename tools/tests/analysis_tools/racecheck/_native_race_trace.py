"""Shared direct Racecheck trace helper."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tirx_harness import numsim


def assert_compact_race_execution(execution: Mapping[str, Any]) -> None:
    assert execution["access_count"] > 0
    assert execution["accesses_complete"] is False
    assert execution["accesses"] == []


def _assert_equivalent_execution_error(
    compact: Mapping[str, Any] | None,
    full: Mapping[str, Any] | None,
) -> None:
    if compact == full:
        return
    assert compact is not None and full is not None
    assert full["kind"] == compact["kind"]
    compact_detail = compact["message"].partition(": ")[2]
    assert compact_detail and full["message"].endswith(compact_detail)


def _without_source(value: Any) -> Any:
    """Remove source decoration added by the public report adapter."""
    if isinstance(value, Mapping):
        return {key: _without_source(item) for key, item in value.items() if key != "source"}
    if isinstance(value, list):
        return [_without_source(item) for item in value]
    return value


def full_direct_race_run(
    kernel: Any,
    inputs: dict[str, Any],
    cache_dir: str | Path,
    native_payload: Mapping[str, Any],
    *,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Repeat one public direct RaceCheck execution with the full access journal."""

    assert_compact_race_execution(native_payload)
    module = numsim.transpile(
        kernel,
        cache_dir=cache_dir,
        _default_generated_opt_level=0,
        _analysis_capable=True,
    )
    result = numsim.Engine(max_workers=max_workers).run_racecheck_phase(
        module,
        inputs,
        inspect_accesses=True,
    )
    payload = result.to_dict()
    assert payload["verdict"] == native_payload["verdict"]
    _assert_equivalent_execution_error(
        native_payload.get("execution_error"), payload.get("execution_error")
    )
    assert _without_source(payload["findings"]) == _without_source(native_payload["findings"])
    for key in ("verdict", "findings", "incomplete"):
        assert payload["sync"][key] == native_payload["sync"][key]
    assert payload["access_count"] == native_payload["access_count"], (
        payload["access_count"],
        native_payload["access_count"],
    )
    assert payload["accesses_complete"] is True
    assert len(payload["accesses"]) == payload["access_count"]
    return payload
