"""The ``evolve`` command.

``evolve init`` prepares one clean KDA worktree for a Claude or Codex session.
It is intentionally setup-only. The selected agent owns the optimization
session and durable progress lives in the generated worktree.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import time
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

import yaml

from evolution.preparation import sandbox
from evolution.preparation.declare import (
    BENCH_ADAPTER,
    effective_guard_toolset,
    load_task,
    load_toolset,
    outside_run_bans,
    task_path,
    toolset_path,
)
from evolution.preparation.live_references import kernel_reference_bans
from evolution.prompts.render import (
    kernel_authoring_contract,
    kernel_remote_gpu_work,
    kernel_remote_local_checks,
    kernel_research_debugging_contract,
    kernel_search_references,
    render_file,
    shared_rule_blocks,
)

PACKAGE_DIR = Path(__file__).resolve().parent
# The workspace member is installed editable, so the source checkout is the run source.
REPO_ROOT = PACKAGE_DIR.parent
RUNS_ROOT = REPO_ROOT / "kda_flow_runs"
PROMPT_TEMPLATE = PACKAGE_DIR / "prompts" / "PROMPT.md"
REMOTE_TEMPLATE = PACKAGE_DIR / "prompts" / "PROMPT_remote.md"
BENCH_SERVER_URL = "http://127.0.0.1:61886"
TOOLSET_NAME = "kda_flow"
FLOWVERSE_CONFIG_NAME = "flowverse.yaml"


def _prompt(
    *,
    task,
    toolset,
    worktree: Path,
    remote_url: str | None = None,
    python: Path | None = None,
) -> str:
    blocks = shared_rule_blocks(toolset)
    preamble = "You are running as an autonomous kernel-optimization agent."
    environment_blocks = ["1. " + blocks["banned_reads"]]
    bench_timeout_s = task.bench_timeout_s or 120
    interpreter = shlex.quote(str(python or sys.executable))
    bench_command = (
        f"timeout {bench_timeout_s}s {interpreter} "
        f"evolution/benchmark/adapter.py {task.workload_dir} CANDIDATE"
    )
    bench_environment = "env CUDA_VISIBLE_DEVICES=<picked> \\\n"
    remote_section = ""
    if remote_url is not None:
        bench_command = (
            f"{interpreter} evolution/remote/kcoral_remote.py {task.workload_dir} CANDIDATE "
            f"--remote {remote_url} --timeout {bench_timeout_s}"
        )
        bench_environment = ""
        remote_section = "\n\n" + render_file(
            REMOTE_TEMPLATE,
            bench_server_url=remote_url,
            worktree=str(worktree),
            remote_gpu_work=kernel_remote_gpu_work(task),
            remote_local_checks=kernel_remote_local_checks(task),
        ).rstrip("\n")
    else:
        environment_blocks.append("2. " + blocks["gpu"])
    environment = "\n\n".join(environment_blocks)
    body = render_file(
        PROMPT_TEMPLATE,
        task_spec=task.spec.strip(),
        sota_baseline=task.sota_baseline.name,
        workload_dir=task.workload_dir,
        worktree=str(worktree),
        kernel_authoring_contract=kernel_authoring_contract(task),
        kernel_search_references=kernel_search_references(task),
        kernel_research_debugging_contract=kernel_research_debugging_contract(task),
        bench_command=bench_command,
        bench_environment=bench_environment,
        bench_remote_section=remote_section,
    )
    return f"{preamble}\n\n{environment}\n\n{body}\n"


def setup(
    *,
    task_name: str,
    remote_url: str | None = None,
    run_name: str | None = None,
) -> Path:
    if run_name is None:
        stem = f"{task_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        run_id = stem
        suffix = 2
        while (RUNS_ROOT / run_id).exists():
            run_id = f"{stem}-{suffix}"
            suffix += 1
    else:
        run_id = run_name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise SystemExit(f"invalid run id: {run_id!r}")
    run_dir = RUNS_ROOT / run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")

    task = load_task(task_path(REPO_ROOT, task_name))
    toolset = load_toolset(toolset_path(REPO_ROOT, TOOLSET_NAME))
    effective = effective_guard_toolset(
        toolset, task, extra_banned=outside_run_bans(REPO_ROOT, RUNS_ROOT)
    )
    effective = replace(
        effective,
        banned_paths=(
            effective.banned_paths + kernel_reference_bans(effective.banned_paths)
        ),
    )
    run_dir.mkdir(parents=True)
    pinned = sandbox.pinned_commit(REPO_ROOT)
    worktree, environment, effective = sandbox.up(run_dir, REPO_ROOT, effective, pinned=pinned)
    bench = worktree / BENCH_ADAPTER
    if not bench.exists():
        raise SystemExit(f"worktree has no benchmark entry: {bench}")

    prompt_path = run_dir / "PROMPT.md"
    prompt_path.write_text(
        _prompt(
            task=task,
            toolset=effective,
            worktree=worktree,
            remote_url=remote_url,
            python=environment.python,
        )
    )

    manifest = {
        "run_id": run_id,
        "flow": "kda_flow",
        "task": task_name,
        "toolset": TOOLSET_NAME,
        "workload_dir": task.workload_dir,
        "pinned_commit": pinned,
        "kernel_authoring": task.kernel_authoring,
        "bench_server_url": remote_url,
        "started_at": time.time(),
        "venv": str(environment.prefix),
        "python": str(environment.python),
        "kernel_package": str(environment.kernels),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    flowverse_config = {
        "work_paths": [f"{task.workload_dir}/frontier"],
    }
    flowverse_config_path = run_dir / FLOWVERSE_CONFIG_NAME
    flowverse_config_path.write_text(
        yaml.safe_dump(flowverse_config, sort_keys=False), encoding="utf-8"
    )
    print("\nSetup complete. Enter the prepared worktree:\n")
    print(f"run_dir={shlex.quote(str(run_dir))}")
    print('cd "$run_dir/worktree"')
    print(". .venv/bin/activate")
    print("\nThen run your agent. See the launch instructions:")
    print("https://tirxharness.mlc.ai/docs/optimization-runs.html#launch-the-agent")
    return run_dir


def _parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(prog="evolve")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init",
        help="prepare an optimization run",
        description="Prepare one clean KDA worktree for a Claude or Codex session.",
    )
    init.add_argument("--task", required=True)
    init.add_argument(
        "--remote",
        nargs="?",
        const=BENCH_SERVER_URL,
        metavar="URL",
        help="score candidates through the kcoral benchmark server at URL "
        f"(default {BENCH_SERVER_URL})",
    )
    init.add_argument(
        "--name",
        dest="run_name",
        help="exact run directory name (default: <task>-<timestamp>)",
    )
    return parser, init


def main(argv: list[str] | None = None) -> int:
    parser, init = _parser()
    args = parser.parse_args(argv)
    if not (REPO_ROOT / ".git").exists():
        init.error(f"evolution must run from a TIRx-harness checkout, not {REPO_ROOT}")
    if args.remote is not None:
        server = urlparse(args.remote)
        if not (server.hostname and server.port):
            init.error(f"--remote expects http://host:port, got {args.remote!r}")
    try:
        setup(
            task_name=args.task,
            remote_url=args.remote,
            run_name=args.run_name,
        )
    except ValueError as error:
        init.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
