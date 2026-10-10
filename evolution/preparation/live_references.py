"""Adapt KDA task bans to the standalone tirx-wiki reference fetcher."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import guards
from .worktree import _path_is_wholly_banned

SKILL_REL = Path(".claude/skills/tirx-wiki")
CATALOG_REL = SKILL_REL / "references/sources.json"
FETCH_SCRIPT_REL = SKILL_REL / "scripts/fetch_references.py"
MANUAL_ROOT_REL = SKILL_REL / "references/manuals"
REPOSITORY_ROOT_REL = SKILL_REL / "references/repos"


def kernel_reference_bans(banned_paths: list[str]) -> list[str]:
    """Apply task kernel bans to both discovery paths for the fetched checkout."""
    return [
        f"{directory}/tirx-wiki/references/repos/{path.removeprefix('./')}"
        for path in banned_paths
        if path.removeprefix("./") == "tirx-kernels"
        or path.removeprefix("./").startswith("tirx-kernels/")
        for directory in (".claude/skills", ".agents/skills")
    ]


def excluded_resources(worktree: Path, banned_paths: list[str] | None = None) -> set[str]:
    """Map wholly banned reference paths to the skill fetcher's resource names."""
    catalog = json.loads((worktree / CATALOG_REL).read_text(encoding="utf-8"))
    excluded = set()
    for name in catalog["manuals"]:
        relative = (MANUAL_ROOT_REL / f"{name}.rst").as_posix()
        if _path_is_wholly_banned(relative, banned_paths or []):
            excluded.add(f"manuals/{name}")
    for name in catalog["repositories"]:
        relative = (REPOSITORY_ROOT_REL / name).as_posix()
        if _path_is_wholly_banned(relative, banned_paths or []):
            excluded.add(f"repositories/{name}")
    return excluded


def fetch_live_references(worktree: Path, banned_paths: list[str] | None = None) -> list[Path]:
    """Run the installed skill fetcher with task-banned resources excluded."""
    if not (worktree / FETCH_SCRIPT_REL).is_file():
        return []
    command = [sys.executable, str(worktree / FETCH_SCRIPT_REL)]
    for resource in sorted(excluded_resources(worktree, banned_paths)):
        command.extend(["--exclude", resource])

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"tirx-wiki reference fetch failed: {result.stderr.strip()}")
    if banned_paths:
        guards._warn_unmatched_bans(worktree, kernel_reference_bans(banned_paths))
        guards._sanitize_banned_paths(worktree, banned_paths)
    return [worktree / SKILL_REL / line for line in result.stdout.splitlines() if line]
