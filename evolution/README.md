# Optimization runs

For installation and launch, read [Optimization Runs](../docs/optimization-runs.md).
This directory owns run preparation and execution tools. Humanize is an
external orchestrator; `setup.py` prepares the run for launch.

## Run preparation

Prepare each optimization run manually before starting the agent. From the source
repository, run:

```bash
python evolution/setup.py --task fp16_gemm_floor --name gemm-review
# For a remote task, add: --remote http://host:port
cat kda_flow_runs/gemm-review/PROMPT.md
```

Setup ends with a short completion summary. Copy its three shell lines to set
`run_dir`, enter the worktree, and activate the run's venv. Follow
[Launch the agent](../docs/optimization-runs.md#launch-the-agent) for your agent's
launch command and prompt. Setup creates a separate worktree and `.venv` for
each run and installs its packages.

### Humanize

Install Humanize separately using its
[installation instructions](https://github.com/humanfia/humanize#install),
then select the Humanize tab in [Launch the agent](../docs/optimization-runs.md#launch-the-agent)
to run two agents in the prepared workspace.

## Responsibilities

| Path | Responsibility |
| --- | --- |
| `setup.py` | Read a task/toolset, prepare the workspace, render the task prompt and write run metadata |
| `tasks/` | Task contracts: math, interface, baseline, candidate path and task-specific restrictions |
| `toolsets/` | Selected skills and shared restrictions |
| `preparation/declare.py` | Parse and combine task/toolset declarations |
| `preparation/worktree.py` | Create a private checkout, initialize permitted submodules and rebuild sanitized Git history |
| `preparation/sandbox.py` | Coordinate source preparation, reference fetching and agent guard installation |
| `preparation/guards.py` | Enforce source-path and process restrictions |
| `preparation/claude.py`, `codex.py` | Configure skills and hooks for each supported agent |
| `preparation/live_references.py` | Fetch permitted references using the installed wiki skill |
| `prompts/render.py` | Fill templates and assemble shared instruction blocks |
| `prompts/PROMPT*.md`, `prompts/rules/` | Instructions rendered for the optimization agents |
| `benchmark/adapter.py` | Map tasks to the inline benchmark package, load candidates and select workloads |
| `remote/` | Kcoral scoring, diagnostic CLIs, request construction and uploaded worker sources |
| `tests/` | Benchmark entry points and remote-tool regressions |

`preparation/` configures the run; it is not the NumSim execution engine and
does not schedule agents. Analysis implementation stays in `tirx_harness/`.
Benchmark inputs, correctness and timing implementations stay in
`benchmark/flashinfer_bench_evolve/`, including task metadata and workload blobs.
The package was imported from `mlc-ai/flashinfer-bench-evolve` at commit
`a5fd91c58c8525b42e68880c0d216fca16f07c7d` and is maintained in this repository.
The KDA forward judge alignment was subsequently ported from
`4d5f7c92101daf0c946a6e1fd38aff6bd62d51c8`, including its shared correctness gates.

## Run outputs

```text
kda_flow_runs/<run-id>/
  manifest.json
  flowverse.yaml
  PROMPT.md
  worktree/
    .venv/                          isolated Python environment for this run
    .claude/skills/                  five configured skills
    .agents/skills/                  links to those skills
    tirx_harness/                     repository source (runtime uses the installed package)
    candidates/<workload>/
      scratch/<candidate>/solution.py
      frontier/<candidate>/solution.py
      frontier/index.json
```

The `kda_flow` manifest/toolset identifies this optimization workflow. Generated
benchmark commands use `evolution/benchmark/adapter.py` locally and
`evolution/remote/kcoral_remote.py` for a remote GPU. Task declarations place
candidate kernels under `candidates/`, separately from the tools that score them.

Setup creates `worktree/.venv` with the Python interpreter used to run setup
and installs the harness and benchmark dependencies from `uv.lock` before
applying task restrictions. The wiki fetcher provides a
`references/repos/tirx-kernels/` checkout for source and README lookup.
Execution uses the installed package.

Task `banned_paths` use the workload/`ported` kernel layout. Native implementations
of the task's workload are restricted alongside its ports and sketches. The same
restrictions apply to the installed package and the fetched wiki checkout.

Setup copies the selected local skills, checks out external skills at the
revisions in `skills/external.json`, and creates the `.agents/skills` links.
The installed skills and fetched references follow the task restrictions.
Setup records the prepared sources in a sanitized Git history.

Launch commands activate the venv, and benchmark commands use its absolute Python
path. The manifest records the venv, interpreter, and installed kernel path. Agents
in the same run share this venv; a new run creates a separate one.

## Tests

Run the benchmark entry-point and remote-tool tests:

```bash
python -m pytest -q evolution/tests
```

The suite tests CLI entry points and remote request handling, with stand-ins
for profiler/server execution. Live GPU profiling and Humanize orchestration
require separate validation. Tool/kernel validation has separate
requirements in the debug/profile skills and `tirx_harness` test guide.
