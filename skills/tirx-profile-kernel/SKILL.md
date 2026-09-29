---
name: tirx-profile-kernel
description: >
  Profile and optimize runnable TIRx kernels by selecting and executing the
  applicable measurement and analysis tools. Use whenever performance must be
  measured, explained, compared, or improved.
---

# Profile a TIRx Kernel

Measure the kernel before explaining or optimizing it. Reading a profiler
guide or inspecting source alone is not performance evidence when a runnable
case is available.

Use the user's workflow Python environment for imports, adapters, and tool
execution. Resolve TIRx-lite documentation and examples through `$tirx-wiki` in that
same environment; do not assume kernels live inside a skill directory.

## Tool routing

| Question or trigger | Required action |
|---|---|
| Hardware utilization, stalls, occupancy, or memory traffic | Run [NCU](references/ncu.md) |
| Critical path, pipeline overlap, bubbles, or serialization | Run [IKET](references/iket.md) |
| Load imbalance or makespan tail | Run [IKET](references/iket.md) |
| Generated-code or compiler-resource questions | Run [source dump](references/dump-source.md) |
| Open-ended bottleneck analysis | Run the benchmark plus [IKET](references/iket.md); add NCU when hardware counters are needed |

Open the linked guide for the selected tool's current API, capture procedure,
and interpretation rules. Do not load unrelated guides.

## Workflow

1. Resolve a reproducible launch and representative input. Treat user-provided
   details as authoritative; otherwise inspect nearby code, tests, and
   benchmarks for the existing launch path.
2. Record a stable baseline before profiling. Keep inputs, environment, GPU,
   warmup, and profiler settings fixed across comparisons.
3. Run the tools that answer the question. Pick an idle GPU and do not run GPU
   profilers concurrently on the same device.
4. Base each hypothesis on a specific metric, finding, or generated-code
   artifact. Preserve disagreements between tools instead of forcing one
   explanation.
5. After a change, rerun the same benchmark and affected profilers and report
   both absolute results and the delta.

Never use NCU replay timing as the benchmark result. Mark hardware-behavior
claims `[VERIFY]` when the available artifact or bundled reference does not
establish them.

## Report tool defects

If a performance tool routed by this skill errors, produces demonstrably wrong
output, or silently misses a known issue, keep collecting independent evidence
and write a report under the current repository's `bugs/`. State what failed,
expected versus actual behavior, where it surfaced, and the exact reproduction
command and output. Also include and run a reliable self-contained reproducer.
Keep supporting artifacts beside the report; delete the report when fixed.

## Finish

Report the benchmark and profiler commands, configuration, relevant raw
artifacts, supporting metrics, and like-for-like before/after results. If a
tool is unavailable or fails, continue with applicable independent tools and
state exactly which evidence remains missing. Do not call source-only or
incomplete-model analysis a full profile.
