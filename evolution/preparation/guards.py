"""Leak containment for a generated KDA worktree.

Source restrictions use two layers:

1. ``_sanitize_banned_paths`` removes banned files from prepared sources and
   installed packages, then patches parent ``__init__`` re-exports.
2. ``_write_banned_paths_hook`` writes a PreToolUse hook that checks paths in
   file and notebook tools, shell commands, working directories and
   ``WebFetch file://`` URLs, including banned paths outside the worktree.

``_write_process_guard_hook`` separately stops fuzzy
``pkill``-style commands from killing other users' runs. Runtime-specific
registration lives in :mod:`evolution.preparation.claude` and
:mod:`evolution.preparation.codex`.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import stat
from pathlib import Path

CANONICAL_KERNELS_SOURCE_DIR = Path("tirx-kernels")
BANNED_PATHS_HOOK_NAME = "banned-paths.py"
PROCESS_GUARD_HOOK_NAME = "process-guard.py"


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _literal_glob_prefix(pattern: str) -> str:
    s = pattern[2:] if pattern.startswith("./") else pattern
    for i, ch in enumerate(s):
        if ch in "*?[":
            return s[:i].rstrip("/")
    return s.rstrip("/")


def _banned_pattern_targets(worktree: Path, pattern: str) -> list[Path]:
    """Return concrete worktree paths to delete for a banned glob.

    For directory globs such as `tirx_kernels/gemm/**`, deleting only
    currently matched children would leave an importable package shell behind.
    Delete the literal directory prefix instead so Python imports fail.
    """
    norm = pattern[2:] if pattern.startswith("./") else pattern
    prefix = _literal_glob_prefix(norm)
    if norm.endswith("/**") and prefix:
        return [worktree / prefix]
    if not any(ch in norm for ch in "*?["):
        return [worktree / norm]
    return [Path(p) for p in glob.glob(str(worktree / norm), recursive=True)]


def _patch_init_for_removed_module(parent: Path, module_name: str) -> None:
    """Remove simple re-exports for a deleted child module from __init__.py."""
    init_path = parent / "__init__.py"
    if not init_path.exists():
        return
    text = init_path.read_text()
    original = text

    # Drop common import/export lines for the removed child.
    text = re.sub(rf"^from \.{re.escape(module_name)} import .*\n", "", text, flags=re.MULTILINE)
    text = re.sub(rf"^from \. import .*{re.escape(module_name)}.*\n", "", text, flags=re.MULTILINE)
    text = re.sub(rf"^import .*{re.escape(module_name)}.*\n", "", text, flags=re.MULTILINE)

    # Drop string entries from __all__-style lists.
    text = re.sub(rf'"\s*{re.escape(module_name)}\s*",?\s*', "", text)
    text = re.sub(rf"'\s*{re.escape(module_name)}\s*',?\s*", "", text)

    if text != original:
        init_path.write_text(text)


def _sanitize_banned_paths(worktree: Path, banned_paths: list[str]) -> None:
    """Remove banned sources under the supplied worktree or package root.

    Patch parent package exports so permitted modules remain importable after
    banned child modules are removed.
    """
    removed: list[Path] = []
    root = worktree.resolve()
    for pattern in banned_paths:
        for target in _banned_pattern_targets(worktree, pattern):
            try:
                resolved = target.resolve()
            except FileNotFoundError:
                resolved = target.absolute()
            if resolved == root:
                continue
            if not str(resolved).startswith(str(root) + os.sep):
                continue
            if not target.exists() and not target.is_symlink():
                continue
            removed.append(target)
            _remove_path(target)

    for path in removed:
        name = path.stem if path.suffix == ".py" else path.name
        if name and name != "__init__":
            _patch_init_for_removed_module(path.parent, name)


def _require_banned_matches(root: Path, banned_paths: list[str]) -> None:
    """Fail when a ban matches nothing under ``root``, such as after an upstream rename."""
    unmatched = [
        pattern
        for pattern in banned_paths
        if not any(os.path.lexists(target) for target in _banned_pattern_targets(root, pattern))
    ]
    if unmatched:
        raise ValueError(f"Banned paths match nothing under {root}: {unmatched}")


def _write_banned_paths_hook(
    hooks_dir: Path, banned_paths: list[str], allowed_prefixes: list[str] | None = None
) -> None:
    """Write an executable Python hook that rejects banned tool paths.

    The hook checks path arguments, shell command tokens and working
    directories from the JSON payload on stdin. Paths under allowed_prefixes
    are checked relative to the first matching root, with the worktree first.
    This exempts the run from ancestor bans while enforcing its relative bans.
    """
    patterns_py = json.dumps(banned_paths)
    allowed_py = json.dumps(allowed_prefixes or [])
    # hooks_dir is <worktree>/.claude/hooks or <worktree>/.codex/hooks.
    # Bash needs to match on substring of the command string. We derive a set
    # of "literal prefix" substrings from the banned globs (everything up to
    # the first wildcard) and reject if any appears in the Bash command.
    script = f"""#!/usr/bin/env python3
\"\"\"PreToolUse hook: reject Read/Grep/Glob/shell access to BANNED_GLOBS paths.

Agent runtimes invoke hooks by piping a JSON payload on stdin, for example:
    {{"tool_name": "Read", "tool_input": {{"file_path": "..."}}, ...}}
We exit 2 with an error message on stderr to block the tool call.
\"\"\"
import fnmatch
import json
import os
import re
import sys

BANNED_GLOBS = {patterns_py}
ALLOWED_PREFIXES = {allowed_py}
WORKTREE_ROOT = ALLOWED_PREFIXES[0] if ALLOWED_PREFIXES else ""

# Literal prefix of each glob (everything before the first wildcard), used as
# a substring probe for Bash command strings.
def _literal_prefix(g: str) -> str:
    s = g[2:] if g.startswith("./") else g
    for i, ch in enumerate(s):
        if ch in "*?[":
            return s[:i].rstrip("/")
    return s.rstrip("/")

BANNED_PREFIXES = [p for p in (_literal_prefix(g) for g in BANNED_GLOBS) if p]


def _abs(p: str) -> str:
    \"\"\"Resolve `p` to an absolute, symlink-following path. realpath (not
    abspath) so a worktree-internal symlink that points at banned content
    can't smuggle that content past the relative-form ban check.
    \"\"\"
    if not p:
        return p
    # Expand BEFORE anything else: `$HOME/...` / `~/...` name the same file as
    # the literal path, so a ban that only sees the literal form is bypassable.
    p = os.path.expanduser(os.path.expandvars(p))
    if WORKTREE_ROOT and not os.path.isabs(p):
        p = os.path.join(WORKTREE_ROOT, p)
    try:
        return os.path.realpath(p)
    except Exception:
        return p


_SHELL_ESCAPE_RE = re.compile(r"\\\\(.)")


def _shell_unescape(tok: str) -> str:
    \"\"\"Drop single-char backslash escapes the way bash does:
    `g\\\\emm` → `gemm`, `\\\\$X` → `$X`, `\\\\\\\\` → `\\\\`.
    Without this, `cat tirx_kernels/g\\\\emm/foo.py` (a real bash command
    that reads `tirx_kernels/gemm/foo.py`) would slip past a literal
    string match for `tirx_kernels/gemm`.
    \"\"\"
    return _SHELL_ESCAPE_RE.sub(r"\\1", tok)


def _glob_expansions(tok: str) -> list[str]:
    \"\"\"If `tok` contains shell glob wildcards AND looks pathlike (has
    `/`), return its file-system expansion relative to cwd (the worktree
    when the hook fires from a bash subprocess). Empty list if no
    wildcards / no matches.

    Without this, `cat tirx_kernels/gemm*/foo.py` slips past
    a literal-prefix match for `tirx_kernels/gemm/`, but bash expands
    it to a banned path at exec time.
    \"\"\"
    if "/" not in tok or not any(c in tok for c in "*?["):
        return []
    try:
        import glob as _glob
        return _glob.glob(tok)
    except Exception:
        return []


def _rel_under_allowed(path: str) -> str | None:
    \"\"\"If `path` resolves under any ALLOWED_PREFIXES, return its
    representation RELATIVE to that prefix (so banned-glob matching can be
    applied against the worktree-relative form). Returns None if the path
    is NOT under any allowed prefix.

    A returned empty string means `path` IS the allowed prefix root.
    \"\"\"
    ap = _abs(path).rstrip("/")
    for prefix in ALLOWED_PREFIXES:
        prefix_abs = _abs(prefix).rstrip("/")
        if ap == prefix_abs:
            return ""
        if ap.startswith(prefix_abs + "/"):
            return ap[len(prefix_abs) + 1:]
    return None


def _matches_glob_set(target: str, globs: list[str]) -> bool:
    \"\"\"True if `target` (a worktree-relative path string) matches any glob.

    Match policy:
    - fnmatch against the normalized path (worktree-root-relative, so a
      slash-less glob like \"CLAUDE.md\" matches ONLY the root-level file —
      the same anchoring the sanitize step applies; without it a bare-name
      ban would basename-match at any depth and block same-named files the
      run legitimately owns, e.g. a workload dir's own docs)
    - prefix-equals: target equals or starts with the literal directory
      prefix of any glob (e.g. \"tirx_kernels/gemm/**\" → prefix
      \"tirx_kernels/gemm\"; matches \"tirx_kernels/gemm/foo.py\")
    \"\"\"
    if not target:
        return False
    norm = target[2:] if target.startswith("./") else target
    for g in globs:
        g_norm = g[2:] if g.startswith("./") else g
        g_flat = g_norm.replace("**", "*")
        if fnmatch.fnmatch(norm, g_flat):
            return True
        prefix = _literal_prefix(g)
        if prefix and (norm == prefix or norm.startswith(prefix + "/")):
            return True
    return False


def path_matches_any(path: str, globs: list[str]) -> bool:
    \"\"\"True if `path` matches a banned glob.

    Banned globs apply to the path's worktree-relative form. Being inside
    an ALLOWED_PREFIX still requires checking relative bans, including paths
    to sources recreated after setup removed them.

    Paths OUTSIDE all ALLOWED_PREFIXES are matched against banned globs
    via fnmatch + literal-prefix substring on the absolute form (catches
    bypass attempts that point outside the worktree).
    \"\"\"
    rel = _rel_under_allowed(path)
    if rel is not None:
        # In-worktree: banned-glob check on the relative form.
        return _matches_glob_set(rel, globs)
    # Outside any allowed prefix: original fallback matching.
    ap = _abs(path).rstrip("/")
    norm_rel = path[2:] if path.startswith("./") else path
    if _matches_glob_set(norm_rel, globs):
        return True
    for p in BANNED_PREFIXES:
        if "/" not in p:
            # Slash-less prefixes are worktree-root bans (anchored); an
            # outside-the-worktree path can never be that root entry, and a
            # bare-name containment match would block unrelated files (any
            # external pyproject.toml). Outside copies of banned content are
            # covered by the absolute ancestor bans.
            continue
        sep_p = "/" + p.lstrip("/")
        if ap == p or ap.startswith(p + "/") or sep_p in ap + "/":
            return True
    return False


def _any_banned_ref_not_allowed(cmd: str) -> str | None:
    \"\"\"Scan a Bash command for references to banned prefixes.

    For each shell-ish token in `cmd`:
    - If the token resolves UNDER an ALLOWED_PREFIX, its worktree-relative
      form is checked against banned prefixes — banned files inside the
      worktree are rejected.
    - Otherwise, both the token's absolute form and as-given form are
      checked for banned-prefix substrings (catches bypass attempts
      pointing outside the worktree, e.g. to a sibling repo).
    \"\"\"
    if not BANNED_PREFIXES:
        return None
    # Shell-like tokenization: split on whitespace + shell operators, plus
    # strip matching quote chars from each token.
    tokens = re.split(r"[\\s|&<>;()'\\\"]+", cmd)
    for tok_raw in tokens:
        if not tok_raw:
            continue
        # Drop bash backslash escapes so `tirx_kernels/g\\emm/foo.py` is
        # detected as `tirx_kernels/gemm/foo.py`.
        tok = _shell_unescape(tok_raw)
        # Build the candidate set: the unescaped token PLUS, if it has
        # shell wildcards, every path it expands to. bash will read
        # whatever it expands to, so the hook must vet expansions.
        candidates_raw = [tok] + _glob_expansions(tok)
        for cand_raw in candidates_raw:
            rel = _rel_under_allowed(cand_raw)
            candidate = rel if rel is not None else cand_raw
            ap = _abs(cand_raw).rstrip("/")
            for p in BANNED_PREFIXES:
                # Slash-less prefixes (root-level bans like "tests" or
                # "pyproject.toml") are ANCHORED at the worktree root: an
                # anywhere-component match would also hit the agent's own
                # files (v1/tests/..., a vN pyproject.toml). Deeper copies
                # outside the worktree stay covered by the absolute
                # ancestor bans.
                anchored_only = "/" not in p
                if candidate == p or candidate.startswith(p + "/"):
                    return p
                if not anchored_only and (
                    ("/" + p + "/") in candidate or candidate.endswith("/" + p)
                ):
                    return p
                # When token is outside allowed, also check absolute form for
                # bypass via absolute-path-to-banned-content.
                if rel is None and not anchored_only:
                    sep_p = "/" + p.lstrip("/")
                    if ap == p or ap.startswith(p + "/") or sep_p in ap + "/":
                        return p
    return None


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{{}}")
    except Exception:
        return 0
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {{}}) or {{}}
    candidates = []
    if tool_name == "Read":
        candidates.append(tool_input.get("file_path", ""))
    elif tool_name == "Grep":
        if "path" in tool_input:
            candidates.append(tool_input["path"])
    elif tool_name == "Glob":
        if "path" in tool_input:
            candidates.append(tool_input["path"])
        candidates.append(tool_input.get("pattern", ""))
    elif tool_name in ("NotebookRead", "NotebookEdit"):
        candidates.append(tool_input.get("notebook_path", ""))
    elif tool_name == "WebFetch":
        url = str(tool_input.get("url", ""))
        if url.startswith("file://"):
            candidates.append(url[len("file://"):])
    elif tool_name in ("Bash", "exec_command"):
        cmd = str(tool_input.get("command") or tool_input.get("cmd") or "")
        # A command whose working directory sits inside a banned source tree can
        # read banned files by basename: relative tokens escape the token scan
        # (which resolves them against the worktree root, not the real cwd).
        # Codex passes `workdir` in tool_input; Claude passes `cwd` at the payload
        # top level. Reject the command if that cwd is itself a banned path.
        workdir = str(
            tool_input.get("cwd") or tool_input.get("workdir") or payload.get("cwd") or ""
        )
        if workdir and path_matches_any(workdir, BANNED_GLOBS):
            sys.stderr.write(
                f"[kda-flow] banned path access blocked: tool={{tool_name}} cwd={{workdir!r}}\\n"
            )
            return 2
        hit = _any_banned_ref_not_allowed(cmd)
        if hit:
            sys.stderr.write(
                f"[kda-flow] banned path access blocked: tool={{tool_name}} references {{hit!r}}\\n"
            )
            return 2
        return 0
    for c in candidates:
        if c and path_matches_any(str(c), BANNED_GLOBS):
            sys.stderr.write(
                f"[kda-flow] banned path access blocked: tool={{tool_name}} target={{c!r}}\\n"
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
    hook_path = hooks_dir / BANNED_PATHS_HOOK_NAME
    hook_path.write_text(script)
    st = hook_path.stat()
    hook_path.chmod(st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _write_process_guard_hook(hooks_dir: Path) -> None:
    """Write an executable hook that blocks broad process-matching kills.

    This is intentionally separate from the banned-path hook: it protects the
    KDA agent process model, independent of toolset path restrictions.
    """
    script = """#!/usr/bin/env python3
\"\"\"PreToolUse hook: reject broad process-matching kill commands.

`pkill -f ...adapter.py...` and related `pgrep -f ... | kill` forms can
match the agent/runtime command line itself because prompts may contain the
same text. Explicit PID/PGID/parent-PID termination remains allowed.
\"\"\"
import json
import re
import sys


_PYTHON_PROCESS_RE = re.compile(r"\\bpython(?:3(?:\\.\\d+)?)?\\b")
_DANGEROUS_PROCESS_RE = re.compile(r"\\b(?:adapter\\.py|python(?:3(?:\\.\\d+)?)?)\\b")
_FULL_MATCH_FLAG_RE = re.compile(r"(?:^|\\s)(?:-[A-Za-z]*f[A-Za-z]*|--full)(?:\\s|=|$)")


def _has_full_match_flag(text: str) -> bool:
    return _FULL_MATCH_FLAG_RE.search(text) is not None


def _contains_pgrep_f_dangerous_process(text: str) -> bool:
    return (
        re.search(r"\\bpgrep\\b", text) is not None
        and _has_full_match_flag(text)
        and _DANGEROUS_PROCESS_RE.search(text) is not None
    )


def _kill_command_substitutes_pgrep_f_dangerous_process(segment: str) -> bool:
    for m in re.finditer(r"\\$\\(([^)]*)\\)|`([^`]*)`", segment, re.DOTALL):
        inner = m.group(1) if m.group(1) is not None else m.group(2)
        if (
            re.search(r"\\bkill\\b", segment[: m.start()]) is not None
            and _contains_pgrep_f_dangerous_process(inner or "")
        ):
            return True
    return False


def _dangerous_process_kill_reason(cmd: str) -> str | None:
    cmd_l = (cmd or "").lower()
    segments = [s.strip() for s in re.split(r"[;\\n]", cmd_l) if s.strip()]
    for segment in segments:
        if re.search(r"\\bpkill\\b", segment):
            if _has_full_match_flag(segment) and (
                _DANGEROUS_PROCESS_RE.search(segment)
            ):
                return "pkill -f can match the agent/runtime command line"
            if re.search(r"\\badapter\\.py\\b", segment):
                return "pkill adapter.py is too broad"
            if _PYTHON_PROCESS_RE.search(segment):
                return "pkill python is too broad"
        if re.search(r"\\bkillall\\b", segment) and _PYTHON_PROCESS_RE.search(segment):
            return "killall python is too broad"

        # `kill $(pgrep -f ...python-or-bench...)`,
        # ``kill `pgrep -f ...python-or-bench...` ``, and
        # `pgrep -af ...python-or-bench... | awk ... | xargs kill` are as
        # dangerous as direct pkill -f because the match string can occur in
        # the agent prompt or another run's command line.
        if _kill_command_substitutes_pgrep_f_dangerous_process(segment):
            return "kill with pgrep -f is too broad"
        if (
            "|" in segment
            and _contains_pgrep_f_dangerous_process(segment)
            and re.search(r"\\|.*\\bkill\\b", segment) is not None
        ):
            return "pgrep -f piped to kill is too broad"
    return None


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        return 0
    tool_name = payload.get("tool_name", "")
    if tool_name not in ("Bash", "exec_command"):
        return 0
    tool_input = payload.get("tool_input", {}) or {}
    cmd = str(tool_input.get("command") or tool_input.get("cmd") or "")
    process_hit = _dangerous_process_kill_reason(cmd)
    if process_hit:
        sys.stderr.write(
            "[kda-flow] broad process kill blocked: "
            + process_hit
            + "; use timeout, kill <pid>, kill -- -<pgid>, or pkill -P <ppid> instead\\n"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
    hook_path = hooks_dir / PROCESS_GUARD_HOOK_NAME
    hook_path.write_text(script)
    st = hook_path.stat()
    hook_path.chmod(st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
