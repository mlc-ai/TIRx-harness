# Optimization Runs

```{container} lead
An **optimization run** is the complete workflow for a registered workload:
`evolve init` prepares its worktree and Python environment, then an agent
iteratively writes, checks, and benchmarks kernel candidates, saving each with
its results.
```

## How the loop works

Within an optimization run, the **agent loop** repeats these steps: consult
references, write a candidate, check it, and evaluate it on the GPU. Diagnostics
and measurements guide the next revision. The workload keeps the computation,
correctness policy, and scoring rule fixed throughout the run.

````{container} agent-loop-diagram
```{raw} html
:file: _static/agent-loop.svg
```
````

## Prepare a run

Prepare the [source-build prerequisites](installation.md#build-from-source),
install uv, and configure your coding agent. Setup builds the harness from
each run's worktree. Clone the repository and initialize the native
submodule; `uv run` installs the setup tool on first use:

```bash
git clone https://github.com/mlc-ai/TIRx-harness.git
cd TIRx-harness
git submodule update --init thirdparty/tvm-rust-ext
```

Choose a task from {repo}`evolution/tasks` and a GPU matching its contract.
This example uses the B200 FP16 GEMM task:

```bash
uv run --package tirx-evolution evolve init --task fp16_gemm_floor
```

Setup creates an isolated worktree and `.venv` using the Python interpreter
that launched it. It installs the harness and benchmark dependencies from
`uv.lock` into that venv, then prepares the selected skills, references, and
task prompt. This applies to both local and remote GPU runs. The final summary
prints three shell lines to set `run_dir`, enter the worktree, and activate
the run's venv. Copy those lines before launching the agent below.

(select-remote-execution)=
````{admonition} Remote execution
:class: note

For a remote GPU, install kcoral with benchmark dependencies and start the
server. Run these commands from the repository root on the GPU server:

```bash
python -m pip install --group server
python -m kcoral server --host 0.0.0.0 --port 8000 --gpus 0
```

Choose the physical GPU IDs with `--gpus` (for example, `0,1`) and leave the
server running. On the agent machine, replace `your-server` with the server's
hostname or IP address and add `--remote` when preparing the run:

```bash
curl -fsS http://your-server:8000/health
uv run --package tirx-evolution evolve init --task fp16_gemm_floor \
  --remote http://your-server:8000
```

CPU checks stay local. The kcoral adapters run GPU benchmarks and diagnostics
remotely. The server needs the task's runtime dependencies and profiling tools.
````

## Launch the agent

::::{tab-set}
:::{tab-item} Claude Code

Set `run_dir` to the printed path, activate its venv, and start Claude Code:

```bash
run_dir=/absolute/path/printed/as/run_dir
cd "$run_dir/worktree"
source .venv/bin/activate
claude
```

Then enter:

```text
/goal Read ../PROMPT.md and achieve at least 1.1x aggregate speedup over cuBLAS,
while passing correctness checks for every workload.
```

:::
:::{tab-item} Codex

Set `run_dir` to the printed path, activate its venv, and start Codex:

```bash
run_dir=/absolute/path/printed/as/run_dir
cd "$run_dir/worktree"
source .venv/bin/activate
codex
```

Then enter:

```text
/goal Read ../PROMPT.md and achieve at least 1.1x aggregate speedup over cuBLAS,
while passing correctness checks for every workload.
```

:::
:::{tab-item} Humanize
:name: optimize-kernels-with-humanize

Install Humanize separately using its
[installation instructions](https://github.com/humanfia/humanize#install),
then use it to run Flame Chase in the prepared workspace. Set `run_dir` to the
printed path and activate its venv. Replace the agent placeholders with your
configured `CLI/MODEL:EFFORT` specifications:

```bash
run_dir=/absolute/path/printed/as/run_dir
cd "$run_dir/worktree"
source .venv/bin/activate
hmz exec -f 'git+https://github.com/humanfia/flowverse#flame_chase' \
  -a 'first_chaser=<first-agent>' -a 'second_chaser=<second-agent>' \
  -b duration=24h "$(cat "$run_dir/PROMPT.md")"
```

:::
::::

## Inspect the results

The run directory contains:

```text
kda_flow_runs/<run-id>/
├── manifest.json
├── PROMPT.md
├── flowverse.yaml
└── worktree/
    └── candidates/fp16_gemm_floor/
        ├── scratch/
        └── frontier/
            ├── index.json
            └── <name>/solution.py
```

| Result | How to use it |
| --- | --- |
| `manifest.json` | Identify the run's task and setup. |
| `PROMPT.md` | Read the evaluation contract and exact benchmark command. |
| `frontier/index.json` | Find retained candidates, their measured timings, speedups, and reasons for keeping them. |
| `frontier/<name>/solution.py` | Inspect a self-contained candidate implementation. |
| `scratch/` | Inspect temporary experiments and diagnostic artifacts while they are available. |

The frontier retains both the current best and distinct, promising approaches.
Its index is the source of truth for accepted candidates; scratch files may be
removed during cleanup. Accepted frontier changes are committed in the run
worktree.

### Recheck a candidate

Use the command in `PROMPT.md`, replacing `CANDIDATE` with the selected path
relative to the workload, such as `frontier/persistent-tma`. For the local
FP16 GEMM example, the adapter invocation from the generated worktree is:

```bash
python evolution/benchmark/adapter.py fp16_gemm_floor frontier/persistent-tma
```

Replace `frontier/persistent-tma` with an entry that actually exists in your
frontier. Use the venv and absolute Python path printed in the prompt; remote runs use
their prescribed remote scoring command. Passing `baseline` as the candidate
selects the workload's baseline self-check.

Read correctness before speedup. A successful remote request only means the
request completed; the workload must also pass its correctness gate and
produce valid timing. Compare candidates under the same inputs, environment,
and measurement settings. Profiler timings are diagnostic; the official
benchmark determines the score.

Scoring commands can exit with status zero even when workloads fail. Check
every expected row for `PASS` before accepting a candidate.
