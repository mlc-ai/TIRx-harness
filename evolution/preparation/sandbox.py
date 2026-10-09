"""Create one pinned, sanitized, history-free KDA worktree."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from . import claude, codex
from .declare import Toolset
from .environment import Environment
from .guards import _require_banned_matches
from .live_references import fetch_live_references
from .worktree import (
    _path_is_wholly_banned,
    create_worktree,
    prefetch_submodules,
    scrub_worktree_git,
)


def pinned_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def install_skills(worktree: Path, effective) -> None:
    """Copy selected skills before applying task restrictions."""
    names = [
        name
        for name in effective.effective_skills
        if not any(
            _path_is_wholly_banned(f"{directory}/{name}", effective.banned_paths)
            for directory in (".claude/skills", ".agents/skills")
        )
    ]
    source_root = worktree / "skills"
    local = {p.parent.name: p.parent for p in source_root.glob("*/SKILL.md")}
    external = json.loads((source_root / "external.json").read_text())
    if local.keys() & external.keys():
        raise ValueError("Each skill must have exactly one source")
    skills_dir = worktree / ".claude/skills"
    agents_dir = worktree / ".agents/skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = local.get(name)
        if source is None:
            if name not in external:
                raise ValueError(f"Unknown skill: {name}")
            ref = external[name]
            source = source_root / name
            subprocess.run(["git", "clone", "--no-checkout", ref["url"], str(source)], check=True)
            subprocess.run(["git", "checkout", "--detach", ref["revision"]], cwd=source, check=True)
        if not (source / "SKILL.md").is_file():
            raise ValueError(f"Missing skill: {source}/SKILL.md")
        target = skills_dir / name
        shutil.copytree(
            source, target, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache")
        )
        (agents_dir / name).symlink_to(
            os.path.relpath(target, agents_dir), target_is_directory=True
        )
    # Leave no second, unsanitized copy of skill resources in the eval tree.
    shutil.rmtree(worktree / "skills")


def install_toolsets(worktree: Path, effective) -> None:
    """Install the same KDA skill set and guards for Claude and Codex.

    The run directory holding the worktree is allowed alongside it.
    ``outside_run_bans`` bans the source checkout and every run root by
    absolute prefix, which would otherwise also cover the run's own
    orchestration input (``PROMPT.md``, ``flowverse.yaml``, ``manifest.json``)
    and stop the orchestrator from reading its own config. Only this run's
    directory is exempted, so sibling runs stay banned; the worktree itself
    stays first in the allowed list, so the in-repo bans keep matching against
    the worktree-relative form.
    """
    run_dir = [str(worktree.parent.resolve())]
    claude.install_toolset(worktree, effective, extra_allowed_prefixes=run_dir)
    codex.install_toolset(worktree, effective, extra_allowed_prefixes=run_dir)


def up(
    run_dir: Path, repo_root: Path, effective: Toolset, *, pinned: str | None = None
) -> tuple[Path, Environment, Toolset]:
    """Build ``run_dir/worktree`` and install Claude and Codex tool hooks.

    Install packages before guards run and before Git metadata is scrubbed,
    while the native tools build can still verify its submodule revision.
    Sanitization runs again after reference fetches, then git history is rebuilt
    from the sanitized tree so removed blobs cannot be recovered.
    """
    worktree = run_dir / "worktree"
    create_worktree(worktree, repo_root, pinned or pinned_commit(repo_root))
    prefetch_submodules(worktree, effective.banned_paths, repo_root=repo_root)
    environment = Environment.create(worktree)
    # Check now: the reference fetch below also removes these paths from the venv.
    _require_banned_matches(
        environment.site_packages, environment.kernel_bans(effective.banned_paths)
    )
    effective = replace(
        effective,
        # Hooks match paths inside the worktree relative to that root.
        banned_paths=effective.banned_paths
        + [
            str(Path(path).relative_to(worktree))
            for path in environment.kernel_bans(effective.banned_paths)
        ],
    )
    install_skills(worktree, effective)
    install_toolsets(worktree, effective)
    fetch_live_references(worktree, effective.banned_paths)
    environment.sanitize(effective.banned_paths)
    scrub_worktree_git(worktree)
    return worktree, environment, effective
