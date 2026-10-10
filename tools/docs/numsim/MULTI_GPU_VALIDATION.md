# Multi-GPU validation status and resume checklist

This is a draft integration of the multi-GPU NumSim, Racecheck, Synccheck,
kernel-port, and benchmark work. The memory rules and their positive/negative
tests come first in [multimem](MULTI_GPU_MULTIMEM.md) and
[peer memory](MULTI_GPU_PEER.md). This page distinguishes existing evidence
from the work needed to close the correctness and performance goal.

## Current evidence

The October 10, 2026 handoff preserves the prior GB200 result files and ports
on upstream `66681fb`, including the `tirx_harness/` to `tools/` directory
rename. The old GPU job is no longer running. Native and GPU validation of
this integrated branch is pending a replacement Linux/4× GB200 environment.
The local machine has no Cargo or installed harness environment; its attempted
Python test run stopped during collection, not at a kernel test failure.

Local handoff checks passed for all 52 changed Python files' syntax, one JSON
result file and 11,950 JSONL records, 37 local documentation links, 32 named
rule-to-test references, documentation code-block syntax, and patch whitespace.
Sphinx is not installed locally, so a rendered documentation build is also
pending. None of these static checks substitutes for the native or GPU gates.

An offline audit of the checked-in measurements found the following worst
latency ratios. A ratio is `TIRx latency / origin latency`; performance of at
least 99% requires **ratio ≤ 1 / 0.99**, approximately 1.010101. “All shapes”
means every entry in the corresponding benchmark module's `SHAPES`, with the
documented dtypes and world size 4, not every possible kernel shape or device.

| Kernel | Shape coverage | Worst original / repeat ratio | Existing numerical evidence |
|---|---|---|---|
| Two-shot all-reduce | 18 shapes, CUTLASS | 1.002734 / no independent repeat file | Benchmark checks against NCCL and reset flags. |
| GEMM + all-reduce | 19 shapes × CUTLASS and FlashInfer | 1.007572 / 1.007864 | Every retained comparison row reports bitwise equality with its origin; the runner also checks a float64 reference with the stated output-rounding tolerance. |
| All-gather GEMM | 17 shapes, fastest measured supported origin per shape | 1.006654 / 0.986765 | Comparison runs validate output on every rank before timing. The small `cutlass_default` case has CUTLASS only; the other 16 have CUTLASS, cake, and cuTile. |
| Mega MoE | 14 shapes, DeepGEMM | 1.003392 / 1.003785 | Every latest row reports zero maximum output difference; the runner checks the output and cumulative statistics before timing. |

The complete per-shape latency tables and measurement details are in the
[multimem results](MULTI_GPU_MULTIMEM.md#results) and
[peer results](MULTI_GPU_PEER.md#results). Raw data lives in
[`benchmarks/multimem/results`](../../benchmarks/multimem/results/) and
[`benchmarks/peer/results`](../../benchmarks/peer/results/). Recomputing these
ratios is an audit of recorded data, not new GPU correctness or timing evidence.
Result files do not replace fresh validation of the final commit.

The previous session reported 245 peer tests, 348 multimem tests including
GPU microtests, 885 engine tests, and 15 frontend tests passing. These counts
are historical reports; they have not been reproduced on this branch.

## Correctness limits to resolve

- **Full-scale GEMM + all-reduce is not proved race-free.** Its benchmarked
  `cutlass` and `flashinfer` protocols retain the origins' GPU-scoped or
  relaxed signaling. The checker counterexamples diagnose these protocols;
  matching GPU output does not discharge those findings. The `sys_fenced`
  clean variant is currently in the smaller checker kernel. Finish and check
  an appropriately ordered full-scale variant, then measure that exact
  variant against the origins across the complete shape matrix.
- **Mega MoE is `review`, not `clean`.** Its test requires zero physical race
  findings and one `alias_stale_read` advisory. The
  [provenance explanation](MULTI_GPU_PEER.md#litmus-kernels-and-tests) describes
  the unnamed bulk-copy write that leaves an old shared-pool name visible to
  the alias tracker. Resolve or explicitly adjudicate the advisory; do not
  silently drop it or reinterpret `review` as a clean result.
- **Coverage is bounded.** Async multimem copy/reduction/store families and
  fp8 forms are not modeled. The supported-operation table and rejection
  paths must stay consistent. The numerical model has GB200-specific floating
  reduction behavior, with one-ulp tolerance for half sums. A green test set
  is not a claim to cover arbitrary PTX forms, inputs, or GPU architectures.
- **Representative kernels are not identical coverage.** Two-shot uses the
  benchmarked kernel in checker tests. Full-scale multimem GEMM currently uses
  a separate small model for those tests. The two peer ports live in
  `ported/`; the multimem ports are still in `tests/numsim/support/`. Keep
  those distinctions explicit when finishing the example-kernel handoff.

## Resume gates

1. Restore the compatible harness environment on a 4× GB200 host, initialize
   the pinned `thirdparty/tvm-rust-ext` submodule, and install this branch.
   Follow [`tests/AGENTS.md`](../../tests/AGENTS.md); do not substitute an
   unrelated TVM wheel. Record the branch commit, GPU/driver/toolkit versions,
   and origin revisions used by each benchmark.
2. Run the CPU rule tests from both feature documents, including every
   positive control and negative variant, and the engine's
   `cargo test --all-features`. Include rank-local topology, address misuse,
   release/acquire scope, proxy paths, reduction chains, and failed-access
   cleanup. Run the registry/ABI gates and the parent package suite. Compare
   failures with a pristine current-upstream worktree, using identical
   commands serially, before calling a failure pre-existing.
3. Run all live multimem microtests with `-n 1`. A skip due to missing GPUs or
   multicast support is missing validation, not a pass. Keep numerical
   comparisons against both NumSim and the independent reference, and keep
   intentionally invalid ordering cases separate from clean examples.
4. Recheck the final full-scale ports and resolve the correctness limits
   above. Retain source witnesses and all simultaneous report statuses;
   unexpected `incomplete`, deadlock, or race results block completion.
5. Run every declared benchmark shape, then an independent repeat. Check
   host CPU count/load/idle pressure and GPU activity before measurement;
   run no validation or other benchmarks concurrently. Preserve the idle
   interval used to avoid power-history bias, identical timing methods for
   each implementation, origin fixes/exclusions, correctness checks on every
   rank, and raw configuration results. Recalculate each shape's ratio
   without dropping missing or failing shapes, and require ≥99% parity for
   the **correct final kernel**. The prior four-shape multimem idle spot check
   is not a full-matrix revalidation.
6. Update the complete latency tables, exact test commands/results, source
   revisions, and this status page. Only then mark the correctness/performance
   goal complete or move the PR out of draft.

From the repository root, after restoring the environment:

```bash
git submodule update --init thirdparty/tvm-rust-ext
python -m pip install --no-deps --no-build-isolation .
cd tools
python -m pytest -q -n 16 --dist=worksteal -m "not numsim_gpu"
python -m pytest -q -n 1 tests/numsim/microtests/test_multimem.py
(cd src/tirx_harness/numsim/engine-rs && cargo test --all-features)
```

Use the focused commands and benchmark commands in the feature documents
while iterating. The full-suite run above also includes performance tests;
apply the host-idleness preflight required by the test guide first.
