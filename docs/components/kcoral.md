# kcoral

```{container} component-subtitle
Remote GPU execution
```

```{container} lead
Run kernels and GPU tools remotely through the harness's command-line adapters.
```

## Key idea

kcoral lets the agent work on one machine while GPU execution happens on
another. The harness provides **adapter CLIs** that upload code and inputs,
run a kernel or diagnostic tool, and bring logs and profiler artifacts back
to the agent. These results guide the next kernel revision.

The adapters package each request independently, so the agent can use a remote
GPU without relying on files left by an earlier run.

## Before you start

Use a local environment with the `kcoral` client; the
[Python environment](../installation.md#install-python-packages)
includes the client. The remote server must already provide the GPU,
kernel dependencies, and tools you want to run. Local GPU access is not needed.
Server setup is described in the {kcoral}`installation guide <README.md>`.

Run the commands below from the repository root or a generated worktree root,
replacing `http://your-server:8000` with your endpoint.

### Model weights on the worker

Model workloads follow the harness's
{repo}`model weights convention <evolution/tasks/README.md#model-weights>`.
Configure their paths in the server's startup environment. For example, if
the server's models are stored under `/data/models`, start it with:

```bash
TIRX_MODELS_DIR=/data/models kcoral server \
  --sandbox bubblewrap --sandbox-readonly-path /data/models
```

Keep the model subdirectories described by that convention. Moving to another
machine changes the server's root path and mount; client commands, task YAMLs
and candidates stay the same. Paths must exist in the environment running
the server, including any enclosing container.

The read-only mount exposes prepared weights to workers. Bubblewrap workers
have isolated networking and a request workspace that is cleared between
requests, so prepare missing model files outside the sandbox in the same
model root before starting benchmarks. KCoral's `--disk-cache-dir` controls
uploaded-file caching, independently of model weight locations.

## Framework adapters

For the Python and profiling examples, put your launch scripts, candidate
modules, and input data in `./inputs/`. Each command uploads that directory's
**contents**, so `./inputs/evaluate.py` is invoked remotely as `evaluate.py`.

The Python, NCU, IKET, and Compute Sanitizer adapters share these options:

| Option | Meaning |
| --- | --- |
| `--remote URL` | kcoral server to execute on. |
| `--send PATH` | Upload a file or a directory's contents; repeat for additional inputs. |
| `-e NAME=VALUE` / `--env NAME=VALUE` | Set an environment variable; repeat as needed. `-e NAME` forwards its local value and fails if unset. |
| `--timeout SECONDS` | Request timeout; default 300 seconds, range 1–3600. |

Put wrapper options before the first `--`. Upload all required scripts,
modules, and data explicitly; packages are not bundled automatically.
Files send their basenames, and directories send their contents. Symlinks
and overlapping upload paths are rejected.

Environment values are literal; the last assignment wins. Other remote
variables are preserved. `CUDA_VISIBLE_DEVICES` and `KCORAL_DIR` are managed
by the worker and cannot be overridden. Use the server's provisioned
environment for diagnostics; do not change its packages or configuration.

(switch-from-local-to-remote-execution)=
### Run Python

Suppose `inputs/` contains your candidate, test data, and `evaluate.py`.
The script checks outputs against a reference, exits nonzero on failure,
and prints correctness results and timings. Locally, run:

```bash
(cd inputs && python evaluate.py)
```

To use an existing kcoral server, run this from the repository root:

```bash
python evolution/remote/kcoral_python.py \
  --remote http://your-server:8000 --send ./inputs -- evaluate.py
```

The adapter uploads the directory's contents, runs `evaluate.py`, and returns
stdout, stderr, and exit status. Your agent receives the script's results
through the same command-result interface. Upload all local modules and data
the script needs; the server must already have its runtime dependencies.

After `--`, pass Python arguments: a script, `-c 'code'`, or `-m module`.
The adapter uses the server's Python with empty stdin.

The Python adapter returns no output files. For profiler reports and traces,
use the [dedicated artifact adapters](#framework-adapters)
and pass their local output paths back to the agent. For packaged workload
scoring, use the [benchmark adapter](#benchmark-a-workload).

### Capture an NCU report

Use `capture.py` to launch the candidate you want to profile:

```bash
python evolution/remote/kcoral_ncu.py \
  --remote http://your-server:8000 --send ./inputs \
  -o ./artifacts/candidate.ncu-rep \
  --set basic --launch-count 1 -- python capture.py

ncu --import ./artifacts/candidate.ncu-rep --page details
```

Wrapper and NCU options go before `--`; the application goes after it.
`-o` names the exact **local report file**. The adapter captures it remotely
and downloads it for inspection with a compatible local NCU installation.
A missing report fails the command; a saved partial report does not imply
successful capture. Remote report import, mode selection, and NCU
configuration files are unsupported. NCU may omit empty environment values
from the profiled application.

For NCU and IKET, use a Python profiling target without shell commands or
nested GPU profilers. A bare `python` or `python3` uses the remote worker's
interpreter; an explicit interpreter path is preserved.

### Capture an IKET timeline

Use `capture_iket.py` to launch an annotated kernel compiled for IKET, following
the {repo}`IKET guide <skills/tirx-profile-kernel/references/iket.md>`:

```bash
python evolution/remote/kcoral_iket.py \
  --remote http://your-server:8000 --send ./inputs \
  --output-dir ./artifacts/iket \
  -- profile --postprocess json -- python capture_iket.py
```

`--output-dir` is a **local directory** for returned artifacts. The first `--`
introduces IKET arguments; the second introduces the application. Change
`--postprocess json` to `--postprocess perfetto` for a Perfetto trace.
Other modes are `html`, `all`, and `none`; use `--keep` to retain intermediates.
The wrapper manages the remote output path and fails if no files are returned.
`TVM_IKET_OFFICIAL_PROFILE` defaults to `cutlass-4.6.0` when unset; an inherited
value or explicit `-e TVM_IKET_OFFICIAL_PROFILE=...` takes precedence.

### Run Compute Sanitizer

Check the launched binary and download its log:

```bash
python evolution/remote/kcoral_compute_sanitizer.py \
  --remote http://your-server:8000 --send ./inputs \
  --fetch check.log --output-dir ./artifacts/sanitizer \
  -- --tool memcheck --error-exitcode 1 --log-file check.log python check.py
```

Sanitizer and application arguments follow `--`. `--error-exitcode 1` makes
findings fail the command. `--fetch check.log` retrieves the remote log as
`./artifacts/sanitizer/check.log`; omit `--fetch`, `--output-dir`, and
`--log-file` to read the diagnostic output directly in the local terminal.
Use remote relative paths for saved logs, records, and XML. Repeat `--fetch`
to retrieve multiple paths, or fetch a directory for generated filenames.
Missing requested files and empty directories fail the command.

### Results and artifact handling

Remote stdout and stderr print locally after completion, up to 16 MiB each;
nonzero tool exit codes propagate. Artifact adapters transfer files separately,
including available partial outputs from failed runs. They require `-f` to
overwrite existing local files; directory exports merge files into the local
directory.

Uploads and returned files are buffered in memory. Temporary worker files are
cleaned up on completion and exceptions, but forced termination may leave files.
Run an adapter with `--help` for its options; implementations live in
{repo}`evolution/remote/ <evolution/remote>`.

### Benchmark a workload

For a [packaged workload](../development/add-workload.md), use `kcoral_remote.py` to upload the
candidate and benchmark inputs together. For example, with a candidate at
`candidates/fp16_gemm_floor/scratch/example/solution.py` in the generated worktree:

```bash
python evolution/remote/kcoral_remote.py fp16_gemm_floor scratch/example \
  --remote http://your-server:8000
```

Replace `scratch/example` with an existing candidate directory relative to
the workload, or with `baseline` to evaluate the reference. The adapter uses
the same workload correctness and timing contract as local scoring.

`kcoral_remote.py --cmd` is a **last-resort escape hatch**. Use it only when
a necessary diagnostic cannot be performed through the dedicated adapters.
Prefer the adapters above for routine execution, debugging, and profiling.

## Use in the agent loop

The [debugging and profiling skills](../installation.md#install-agent-skills) guide the agent to choose these
commands and interpret their results. CPU checks stay local; remote reports
and logs feed back into kernel edits, followed by correctness checks and an
ordinary benchmark run. For an optimization run, select remote execution with
`evolve init --remote URL`; see {ref}`remote setup <select-remote-execution>`.

## Use another execution service

Run the same test and benchmark entry points locally, through kcoral, or through
your execution service. Return exit status, logs, and any requested artifacts.
