# Add a workload

```{container} lead
Give the agent a new computation with a fixed correctness and scoring contract.
```

A workload defines the computation, candidate interface, inputs, correctness
policy, and performance score. The agent changes the candidate while that
contract remains fixed. To use an existing task, go to
[Optimization Runs](../optimization-runs.md).

## Define the contract

Write down the tensor shapes and dtypes, input domain, expected outputs,
allowed implementation choices, target GPU, and measurement boundary. Specify
the independent reference, numerical tolerances, and how results across shapes
combine into a score.

For example, {repo}`fp16_gemm_floor <evolution/tasks/fp16_gemm_floor.yaml>`
requires a `solution.py` exposing:

```text
setup(data, M, N, K) -> callable
```

`data` contains `A`, `B`, and preallocated `D`. Setup prepares the kernel;
each timed invocation reads the inputs and overwrites `D`. The task declaration
defines the permitted setup work and the full evaluation policy.

## Add the benchmark and register it

1. Add the input generation, independent correctness check, timing baseline,
   task definition, and workload rows under
   `evolution/benchmark/flashinfer_bench_evolve/tasks/` in this repository.
2. Map the workload directory to the packaged task in `PACKAGED` in
   {repo}`evolution/benchmark/adapter.py`.

The registry selects the benchmark task, warmup and iteration counts, and
shape-selection mode for both local and remote scoring. Use `all` for a full
suite, `max` for the benchmark's largest-shape selection, or `pinned` for a
reviewed row identified in `PINNED_WORKLOAD_UUIDS`. Register a restricted
shape task separately with an explicit shape suffix.

The adapter gets benchmark source and inputs from this inline package for
remote execution. Benchmark implementations live under `evolution/benchmark/`;
generated candidate implementations live under `candidates/`.

## Declare the optimization task

Add `evolution/tasks/<name>.yaml`. This scaffold shows the required fields;
replace its `spec` with the complete contract:

```yaml
name: my_kernel
workload_dir: candidates/my_kernel
kernel_authoring: TIRx-lite
sota_baseline:
  name: my_reference
spec: |
  Define the computation, tensors, candidate interface, valid inputs,
  correctness policy, target GPU, and scoring rule here.
```

The filename must match `name`, and `workload_dir` must identify the registered
workload. Optional `bench_timeout_s` sets a positive benchmark timeout;
`banned_paths` lists task-specific reference restrictions. Use
`kernel_authoring: task` when the specification defines another authoring
contract.

{repo}`evolution/preparation/declare.py` defines the schema. Shared skills and
reference restrictions belong to {repo}`evolution/toolsets/kda_flow.yaml`.
A YAML declaration needs the benchmark and registry entry above before it can
be evaluated.

## Validate the integration

From the repository root, run the registered baseline on the target GPU:

```bash
python evolution/benchmark/adapter.py my_kernel baseline
```

Check that the expected shapes are evaluated and timings use the declared
measurement boundary. Validate the benchmark's correctness gate with reference
outputs and deliberately corrupted outputs. For remote execution, use the
[remote benchmark adapter](../components/kcoral.md#benchmark-a-workload) with
`baseline`.

Commit the workload files before preparing a run: setup creates the worktree
from the current Git `HEAD`.

```bash
python evolution/setup.py --task my_kernel
```

Follow [Optimization Runs](../optimization-runs.md#launch-the-agent) to launch the
agent with the generated `PROMPT.md`. The agent develops candidates under
`candidates/my_kernel/scratch/<name>/` and evaluates them with the prescribed
benchmark command. Reproducibly passing candidates may enter `frontier/`, as
defined by the {repo}`run prompt <evolution/prompts/PROMPT.md>`.

Submit the benchmark implementation, adapter registration, and task
declaration together in this repository. Include baseline results, correctness
gate validation, input coverage, and the commands used. The
{repo}`workload guide <evolution/tasks/README.md>` owns the registration details.
