---
name: tirx-debug-kernel
description: >
  Debug and validate runnable TIRx kernels by selecting and executing the
  applicable correctness tools. Use whenever a kernel fails, produces suspect
  results, or a code change needs evidence about synchronization, memory
  safety, or numerical behavior.
---

# Debug a TIRx Kernel

Use correctness tools to investigate the kernel. Reading a tool guide or
suggesting a command is not a substitute for running the tool when a runnable
case is available.

Use the user's workflow Python environment for imports, adapters, and tool
execution. Resolve TIRx-lite documentation and examples through `$tirx-wiki` in that
same environment; do not assume kernels live inside a skill directory.

## Default order

Prefer pre-GPU checks before device-side debugging:

1. Run every applicable checker first. Synccheck and Racecheck do not reserve a
   GPU, so treat them as the cheap edit-loop filter for synchronization and
   memory-safety defects.
2. If a numerical mismatch remains, run NumSim on the smallest representative
   case after synchronization and race findings are resolved. NumSim provides
   deterministic pre-GPU replay, but its build and simulation cost grows with
   the simulated work; do not start with production-size inputs.
3. Escalate to the workload's normal GPU correctness check. If Synccheck or
   Racecheck does not catch a synchronization or race problem, use Compute
   Sanitizer as a device-side follow-up.

## Tool routing

| Trigger | Required action |
|---|---|
| Synchronization or barrier protocol | Run [Synccheck](references/synccheck.md) |
| Shared, tensor, or cross-thread memory safety | Run [Racecheck](references/racecheck.md) |
| The issue may cross both protocol and memory ordering | Run synccheck and racecheck; run independent CPU-side checks concurrently |
| Deterministic numerical dataflow | Run [NumSim](references/numsim.md) on the smallest representative mismatch after synchronization and race findings are resolved |
| A synchronization or race problem is not caught by pre-GPU checks | Run [Compute Sanitizer](references/compute-sanitizer.md) as a device-side follow-up |

Open the linked guide for the selected tool's current API, inputs, and verdict
semantics. Do not load unrelated guides.

## Workflow

1. Resolve the target kernel and a representative input. Treat user-provided
   launch details as authoritative; otherwise inspect nearby code, tests, and
   benchmarks for the existing construction or launch path.
2. Create only the minimal task-local adapter needed to call the real kernel.
   Do not modify production code merely to make a checker runnable.
3. Follow the default pre-GPU-to-device order before choosing a speculative
   fix. Use the same inputs and options across comparisons.
4. Treat `error` as a failure, `incomplete` as missing evidence rather than a
   pass, and `review` as requiring an explicit disposition.
5. Keep the workload's normal numerical or device correctness check as an
   independent gate.
6. After a fix, rerun the same case before expanding coverage.

## Report tool defects

If a correctness tool routed by this skill errors, produces demonstrably wrong
output, or silently misses a known issue, keep collecting independent evidence
and write a report under the current repository's `bugs/`. State what failed,
expected versus actual behavior, where it surfaced, and the exact reproduction
command and output. Also include and run a reliable self-contained reproducer.
Any `incomplete` verdict from NumSim, Synccheck, or Racecheck is unexpected and
must also be filed as a tool bug. Keep supporting artifacts beside the report;
delete the report when fixed.

## Finish

Report the executed command or runner, kernel and input, verdict and relevant
findings, saved artifacts, and same-case rerun. If execution is blocked, report
the attempted command, exact error, and missing input or dependency. Never
present an unexecuted recommendation as correctness evidence.
