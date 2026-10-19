"""Build the per-run worktree: a pinned, sanitized, history-free checkout.

The order matters and is load-bearing:

1. ``create_worktree`` — shallow single-commit checkout at the pinned commit.
2. ``prefetch_submodules`` — pull non-banned runtime submodules, preferring
   the parent repo's local copies; sanitize newly materialized sources.
3. The caller creates the worktree's venv and installs packages while the
   native tools build can still check the original submodule metadata.
4. The caller prepares skills, hooks and references, then applies restrictions
   to both the reference copies and the installed packages.
5. ``scrub_worktree_git`` — remove source Git metadata and re-init one clean-room
   commit so history cannot resurrect banned sources. The venv stays ignored.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

from . import guards
from .guards import CANONICAL_KERNELS_SOURCE_DIR


def create_worktree(worktree_path: Path, repo_root: Path, pinned_commit: str) -> None:
    """Materialize the run worktree as a shallow, single-commit clone.

    This was previously ``git worktree add --detach``. A linked worktree
    shares the main repository's object store AND refs, so the agent could
    read deleted or banned content straight out of history. In a live run
    (round 7), agents used ``git log --all`` and ``git show`` to resurrect
    the deliberately deleted seed kernel as their own version.

    A depth-1 fetch of exactly the pinned commit yields a working tree whose
    entire reachable history is ONE parentless commit: nothing older exists in
    the clone's object store, and none of the main repository's refs are
    visible (``git log --all`` shows the single commit; ``git show <main-sha>``
    has no object). Locked-file restoration keeps working against the lone
    clean-room HEAD.
    """
    # Resolve before use: the fetch below runs with `-C worktree_path`, so a
    # RELATIVE repo_root would be resolved against the new checkout dir, not
    # the caller's cwd (git semantics for relative repository paths).
    repo_root = repo_root.resolve()
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    # If a stale checkout exists at that path, remove it (user-initiated
    # re-run). Shallow clones are plain directories: rmtree is the whole story.
    if worktree_path.exists():
        shutil.rmtree(worktree_path)
    # Fetch by a temporary NAMED ref rather than by raw sha: upload-pack
    # refuses arbitrary SHA wants by default, while a named ref is always
    # fetchable. The ref is deleted right after the fetch either way.
    token = f"refs/kda-checkouts/{uuid.uuid4().hex}"
    subprocess.run(
        ["git", "-C", str(repo_root), "update-ref", token, pinned_commit],
        check=True,
        capture_output=True,
    )
    try:
        subprocess.run(["git", "init", "-q", str(worktree_path)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(worktree_path), "fetch", "-q", "--depth", "1", str(repo_root), token],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(worktree_path), "checkout", "-q", "--detach", "FETCH_HEAD"],
            check=True,
            capture_output=True,
        )
    finally:
        subprocess.run(
            ["git", "-C", str(repo_root), "update-ref", "-d", token],
            check=False,
            capture_output=True,
        )
    # Runtime submodules are NOT initialized here — ``prefetch_submodules``
    # pulls every non-banned one at setup time (the only chance: after
    # ``scrub_worktree_git`` nothing can be fetched). Two preparations make
    # its plain `git submodule update`
    # work: `submodule init` registers the URLs in local config, and ssh
    # GitHub URLs are rewritten to https (the eval user has no GitHub ssh
    # key, and a superproject-local url.insteadOf does NOT propagate to the
    # submodule clone subprocess — only the registered submodule.<name>.url
    # is honored).
    subprocess.run(
        ["git", "-C", str(worktree_path), "submodule", "init"], check=False, capture_output=True
    )
    urls = subprocess.run(
        ["git", "-C", str(worktree_path), "config", "--get-regexp", r"^submodule\..*\.url$"],
        capture_output=True,
        text=True,
    )
    for line in (urls.stdout or "").strip().splitlines():
        try:
            key, url = line.split(" ", 1)
        except ValueError:
            continue
        if url.startswith("git@github.com:"):
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(worktree_path),
                    "config",
                    key,
                    url.replace("git@github.com:", "https://github.com/", 1),
                ],
                check=False,
                capture_output=True,
            )


def _path_is_wholly_banned(relative_path: str, banned_paths: list[str]) -> bool:
    """Whether a worktree-relative path sits at or under a banned path.

    Compare against each glob's literal prefix. A resource is wholly banned
    only when its path equals or sits under that prefix. A ban covering just a
    subtree does not skip materialization; the caller fetches the resource and
    then strips the banned subtree.
    """
    path = relative_path.strip("/")
    if not path:
        return False
    for pattern in banned_paths or []:
        lit = guards._literal_glob_prefix(pattern).lstrip("/")
        if not lit:
            continue
        if path == lit or path.startswith(lit + "/"):
            return True
    return False


def prefetch_submodules(
    worktree: Path,
    banned_paths: list[str] | None = None,
    repo_root: Path | None = None,
) -> list[str]:
    """Pre-pull each registered submodule not wholly covered by a ban.

    Submodules start empty in the shallow checkout, and this prefetch is the
    ONLY chance to populate them — ``scrub_worktree_git`` (run right after)
    removes the git metadata an on-demand pull would need. Benchmark sources
    are ordinary repository files and are already present in the checkout.

    Which submodules exist on disk is an orthogonal infra concern. WHOLLY-banned
    submodules are skipped —
    fetching them would expose the SOTA/reference source the ban withholds;
    a sub-tree-only ban is pulled whole and
    stripped by the post-pull sanitize below.

    Best-effort: a failed fetch (network/DNS flake) leaves that submodule
    absent for the whole run. Returns the submodule paths that were attempted.
    """
    gitmodules = worktree / ".gitmodules"
    if not gitmodules.exists():
        return []
    r = subprocess.run(
        [
            "git",
            "-C",
            str(worktree),
            "config",
            "--file",
            ".gitmodules",
            "--get-regexp",
            r"^submodule\..*\.path$",
        ],
        capture_output=True,
        text=True,
    )
    sub_paths = [
        line.split(" ", 1)[1].strip() for line in (r.stdout or "").splitlines() if " " in line
    ]
    needed = [sp for sp in sub_paths if sp and not _path_is_wholly_banned(sp, banned_paths or [])]
    for sp in needed:
        cmd = ["git", "-C", str(worktree)]
        # Prefer cloning from the LOCAL parent-repo submodule copy: the github
        # URL needs credentials a non-interactive run lacks (observed live:
        # `could not read Username for https://github.com`), while the parent
        # repo already has every submodule populated at the pinned gitlink.
        # Falls back to the configured (github) URL when no local copy exists.
        local_src = (repo_root / sp) if repo_root else None
        if local_src is not None and (local_src / ".git").exists():
            cmd += [
                "-c",
                "protocol.file.allow=always",
                "-c",
                f"submodule.{sp}.url={local_src}",
            ]
        cmd += ["submodule", "update", "--init", "--depth", "1", sp]
        subprocess.run(cmd, check=False, capture_output=True)
    # A NEEDED submodule with only a SUB-TREE banned was pulled whole above
    # and must have its banned sub-tree stripped now
    # that the files exist. This also clears any other banned path that a
    # freshly-cloned submodule may have introduced.
    if banned_paths:
        guards._sanitize_banned_paths(worktree, banned_paths)
    return needed


def scrub_worktree_git(worktree: Path) -> None:
    """Rebuild the worktree's git so no banned blob survives in any object/tree.

    The worktree is a depth-1 checkout, but that single commit's TREE still
    holds every file the pinned commit had — INCLUDING the ones
    _sanitize_banned_paths / prefetch removed from the WORKING dir. They stay
    readable via `git show HEAD:<path>`, `git checkout HEAD -- <path>`,
    `git cat-file`, and `git grep HEAD`, none of which the path-token PreToolUse
    hook can catch. Remove every .git (worktree + submodules) and .gitmodules,
    then re-init ONE clean-room commit of the already-sanitized working tree so
    the object store holds only sanitized files. The worktree must REMAIN a git
    repo: the installed hook commands locate themselves via
    `git rev-parse --show-toplevel`. Call AFTER all sanitize/prefetch.

    """
    # STRICT: this scrub IS the leak boundary. A .git that survives keeps the
    # deleted SOTA source reachable in its object store, and a failed init/commit
    # leaves no clean-room HEAD for the hook's rev-parse or bench restore — a
    # setup that cannot guarantee either must fail loudly, not continue.
    # The venv is runtime state, not source history. Avoid traversing its
    # installed packages (and symlinks) when scrubbing repository metadata.
    for directory, dirs, files in os.walk(worktree):
        if Path(directory) == worktree:
            dirs[:] = [name for name in dirs if name != ".venv"]
        if ".git" in dirs or ".git" in files:
            gitpath = Path(directory) / ".git"
            if gitpath.is_dir() and not gitpath.is_symlink():
                shutil.rmtree(gitpath)
            else:
                gitpath.unlink()
            if ".git" in dirs:
                dirs.remove(".git")
    (worktree / ".gitmodules").unlink(missing_ok=True)
    subprocess.run(["git", "init", "-q", str(worktree)], check=True, capture_output=True)
    ident = ["-c", "user.email=kda@local", "-c", "user.name=kda"]
    subprocess.run(["git", "-C", str(worktree), "add", "-A"], check=True, capture_output=True)
    # The standalone checkout is ignored in development, but the sanitized
    # eval source must belong to the history-free snapshot just like its tools.
    if (worktree / CANONICAL_KERNELS_SOURCE_DIR).is_dir():
        subprocess.run(
            ["git", "-C", str(worktree), "add", "-f", "--", str(CANONICAL_KERNELS_SOURCE_DIR)],
            check=True,
            capture_output=True,
        )
    # Runtime skills are installed payloads, ignored by the development repo.
    for relative in (".claude/skills", ".agents/skills"):
        if (worktree / relative).is_dir():
            subprocess.run(
                ["git", "-C", str(worktree), "add", "-f", "--", relative],
                check=True,
                capture_output=True,
            )
    subprocess.run(
        ["git", "-C", str(worktree), *ident, "commit", "-q", "-m", "clean-room"],
        check=True,
        capture_output=True,
    )
