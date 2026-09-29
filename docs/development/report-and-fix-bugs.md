# Report and fix bugs

```{container} lead
Turn a failure found during a run into a reproducible fix in the owning project.
```

## Identify what failed

Keep the failing candidate and inputs, then use
[compiler analysis](../components/tools.md) and the debugging or profiling skill to
narrow down the failure. Record the command, expected behavior, actual output,
and environment before reducing the example.

| Failure | Where the fix belongs |
| --- | --- |
| A candidate computes the wrong result or uses invalid synchronization | Fix it in the run; contribute a maintained kernel fix to [mlc-ai/TIRx-kernels](https://github.com/mlc-ai/TIRx-kernels). |
| TIRx-lite authoring or a canonical kernel is defective | [mlc-ai/TIRx-kernels](https://github.com/mlc-ai/TIRx-kernels). |
| NumSim, Synccheck, Racecheck, harness setup, or an adapter CLI is defective | [mlc-ai/TIRx-harness](https://github.com/mlc-ai/TIRx-harness). |
| TIRx lowering or generated code is defective | [apache/tvm](https://github.com/apache/tvm), at the revision used by the failing environment. |
| kcoral's server or execution protocol is defective | [mlc-ai/kcoral](https://github.com/mlc-ai/kcoral); adapter-specific failures belong in [mlc-ai/TIRx-harness](https://github.com/mlc-ai/TIRx-harness). |

Use independent evidence to distinguish a kernel error from a checker or
compiler defect. For example, retain a reference result or a device-side
diagnostic alongside a disputed checker result.

## Preserve a reproducer

The debugging and profiling skills record tool defects under the current
repository's `bugs/`. A useful report directory is:

```text
bugs/<short-description>/
├── report.md
├── reproduce.py
└── artifacts/
```

Include a self-contained reproducer and its inputs or deterministic generator.
Run it and record the exact command and output. The report should state:

- Expected and actual behavior, including the failure location.
- Source revisions and relevant package versions; GPU and CUDA environment
  when device execution is involved.
- Which checks ran, their findings, and the artifacts supporting the diagnosis.

The skills file a report when a tool errors, produces demonstrably wrong
output, or silently misses a known issue. The debugging skill also treats an
`incomplete` NumSim, Synccheck, or Racecheck verdict as a tool bug to report.
Their {repo}`debugging <skills/tirx-debug-kernel/SKILL.md>` and
{repo}`profiling <skills/tirx-profile-kernel/SKILL.md>` instructions define the
full reporting contract.

`bugs/` is a local, ignored backlog in this harness. Attach or link the
reproducer and evidence when opening an issue or PR in the owning repository.
Preserve reports from a run worktree before removing that worktree.

## Fix and verify

Make the change in a development checkout of the owning project. Keep the
run's benchmark and correctness contract fixed while diagnosing it; repair a
benchmark defect in its owning package and evaluate again with the corrected
revision.

First rerun the original reproducer with the same inputs and options. Then
check the affected behavior and add a regression test when it protects the
bug's observable contract. A performance fix that changes synchronization or
cross-thread-visible memory needs both correctness and performance evidence.

## Submit the fix

Describe the trigger, root cause, resulting behavior, and before/after evidence
in the PR. Include the reproduction command and any unresolved limitations.
After verifying the fix, remove the resolved local `bugs/` report as the
skills require; preserve the regression test and relevant evidence with the
submitted change. Rerun the affected kernel case with the fixed dependency
before accepting it into the frontier.
