"""Load KDA workload and toolset declarations from ``evolution/``.

The task YAML is the sole workload contract: it identifies the benchmark,
authoring language, baseline, timeout, leak bans, and complete inline spec.
The toolset YAML is the sole authority for installed skills and shared bans.
``effective_guard_toolset`` owns the union of shared, task, and cross-run bans.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

# The single benchmark entry point. `python evolution/benchmark/adapter.py
# <workload_dir> vN` dispatches to the registered flashinfer-bench-evolve task.
BENCH_ADAPTER = "evolution/benchmark/adapter.py"

KERNEL_AUTHORING_MODES: frozenset[str] = frozenset({"TIRx-lite", "task"})


@dataclass
class Toolset:
    name: str
    skills: list[str]
    banned_paths: list[str] = field(default_factory=list)

    @property
    def effective_skills(self) -> list[str]:
        return sorted(set(self.skills))


@dataclass
class SotaBaseline:
    name: str


@dataclass
class Task:
    name: str
    workload_dir: str
    kernel_authoring: str
    sota_baseline: SotaBaseline
    spec: str
    banned_paths: list[str] = field(default_factory=list)
    bench_timeout_s: int | None = None


def merge_unique(*lists: list[str]) -> list[str]:
    """Concatenate lists in first-seen order without duplicates."""
    seen: set[str] = set()
    merged: list[str] = []
    for values in lists:
        for value in values or []:
            if value not in seen:
                seen.add(value)
                merged.append(value)
    return merged


def outside_run_bans(repo_root: Path, *runs_roots: Path) -> list[str]:
    """Ban the source checkout and all run roots from a generated worktree."""
    roots = [repo_root, *runs_roots]
    return [f"{path.resolve()}/**" for path in dict.fromkeys(roots)]


def effective_guard_toolset(
    toolset: Toolset, task: Task, *, extra_banned: list[str] | None = None
) -> Toolset:
    """Return the canonical union of toolset, task, and run-specific bans."""
    return replace(
        toolset,
        banned_paths=merge_unique(toolset.banned_paths, task.banned_paths, extra_banned or []),
    )


def _load_yaml(path: Path) -> dict:
    with path.open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level mapping, got {type(data).__name__}")
    return data


def _required_string(data: dict, key: str, path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: '{key}' must be a non-empty string")
    return value


def _string_list(data: dict, key: str, path: Path) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{path}: '{key}' must be a list of non-empty strings")
    return [item.strip() for item in value]


def load_toolset(path: Path) -> Toolset:
    data = _load_yaml(path)
    return Toolset(
        name=_required_string(data, "name", path),
        skills=_string_list(data, "skills", path),
        banned_paths=_string_list(data, "banned_paths", path),
    )


def load_task(path: Path) -> Task:
    data = _load_yaml(path)
    name = _required_string(data, "name", path)
    workload_dir = _required_string(data, "workload_dir", path)
    kernel_authoring = _required_string(data, "kernel_authoring", path)
    if kernel_authoring not in KERNEL_AUTHORING_MODES:
        raise ValueError(
            f"{path}: kernel_authoring must be one of "
            f"{sorted(KERNEL_AUTHORING_MODES)}, got {kernel_authoring!r}"
        )

    baseline = data.get("sota_baseline")
    if not isinstance(baseline, dict):
        raise ValueError(f"{path}: 'sota_baseline' must be a mapping")
    baseline_name = _required_string(baseline, "name", path)

    spec = _required_string(data, "spec", path)
    raw_timeout = data.get("bench_timeout_s")
    if raw_timeout is None:
        bench_timeout_s = None
    else:
        bench_timeout_s = int(raw_timeout)
        if bench_timeout_s < 1:
            raise ValueError(f"{path}: bench_timeout_s must be >= 1, got {bench_timeout_s}")

    return Task(
        name=name,
        workload_dir=workload_dir,
        kernel_authoring=kernel_authoring,
        sota_baseline=SotaBaseline(name=baseline_name),
        spec=spec,
        banned_paths=_string_list(data, "banned_paths", path),
        bench_timeout_s=bench_timeout_s,
    )


def evolution_root(repo_root: Path) -> Path:
    return repo_root / "evolution"


def toolset_path(repo_root: Path, name: str) -> Path:
    return evolution_root(repo_root) / "toolsets" / f"{name}.yaml"


def task_path(repo_root: Path, name: str) -> Path:
    return evolution_root(repo_root) / "tasks" / f"{name}.yaml"
