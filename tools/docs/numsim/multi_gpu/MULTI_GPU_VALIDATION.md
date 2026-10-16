# Multi-GPU validation status and resume checklist

This is a draft integration of the multi-GPU NumSim, Racecheck, Synccheck,
kernel-port, and benchmark work. The memory rules and their positive/negative
tests come first in [multimem](MULTI_GPU_MULTIMEM.md) and
[peer memory](MULTI_GPU_PEER.md). This page distinguishes existing evidence
from the work needed to close the correctness and performance goal.

## Port-fidelity requirement

This is a port and TIRx proof of concept, not an optimization project. The
performance gate compares the baseline's instructions, memory-ordering forms,
work partition, pipeline, and launch schedule. Newer instructions, stronger
protocols, fused launches, or redistributed work cannot establish IR parity.
Investigate lowering, generated code, compiler settings, and unintended
differences instead. Keep checker findings separate; do not repair a baseline
silently to make an example clean.

The current audit found:

| Kernel | Fidelity status |
|---|---|
| Two-shot all-reduce | Preserves the baseline's data layout, constexpr loop/rank, and terminal release/relaxed-CAS protocol. All 18 shapes and the independent repeat now pass bitwise equality, independent-reference precision, flag reset, and the unchanged 99% performance gate. |
| GEMM + all-reduce | Restored baseline waits, elected leaders, and barrier sequence; removed the added exit GPU fence. All 19 shapes, both origins, three matched configurations, and the independent repeat now meet the unchanged precision/performance gate. `sys_fenced` remains a separate semantic example. |
| All-gather GEMM | `benchmarks.peer.fidelity` restores CUTLASS's per-shard launches, exact auxiliary kernels, and common host schedule. Both 17-shape GPU rounds pass bitwise/reference correctness and the 99% gate at baseline-selected configurations; worst ratios are 1.001948 / 1.002322. The earlier fused/chunked and NumSim copy-warp variants are separate examples, not fidelity evidence. |
| Mega MoE | Preserves reference timeout, loads, work partition, synchronization, and compile-time activation branches. `sys_fenced` stays separate. Source `85c799e75ba5fd87`, with the final generic compiler patch and shared benchmark inputs, passes all 14 shapes and the independent repeat: worst ratios 1.004591 / 1.006576. Both statistics specializations match every BF16 output bit on all four ranks. The [compiler patch, source, tests, and evidence](lowering_optimizations.md) are explicit prerequisites, not evidence for an unpatched compiler. |

Actual CUTLASS and FlashInfer PTX/CUBIN and TIRx CUBIN were captured with the
existing compiler under the socket lease. The CUTLASS artifact confirms elected
leaders, the read-only producer-tail wait, and no extra GPU fence before the
terminal communication barrier. No newer scoped-alias instructions or private
publication/tuning prototypes are part of the faithful port.

The restored GEMM source
`ed1cb2cbb73d69762aa96df92c26eb7adb072435485001c84d7f760771ad6441`
passed **114 matched-configuration comparisons per round**, with four leased
GPUs, NVRTC 13.2.78, 100 graph launches, nine interleaved trials, and 0.5 s idle.
All 228 rows match their pinned origin bitwise, satisfy the independent
reference, and reset flags. The worst latency ratios over *every configuration*
are **1.007424 / 1.006724** (first/repeat), below `1 / 0.99`. Selecting each
origin's best configuration and using that **same configuration** for TIRx gives
worst ratios **1.004910 / 1.002717**. No instruction upgrade, alternative
publication protocol, launch fusion, or port-only retuning closes this gate.

## Current evidence

The latest Mega MoE correction is described in the
[activation and allocation audit](lowering_optimizations.md#current-source-and-benchmark-correction).
Four activation-specialization regressions fail on the old port; 86 focused
peer numerical/race/sync/fidelity checks pass after the correction. A separate
43-test run covers benchmark input sharing, provenance eligibility, CPU-only
compile target preservation, rendezvous handling, and activation fidelity.
The current source is `85c799e75ba5fd87`; older source stamps and
separate-allocation results below remain historical evidence, not the current
shared-input gate. No kernel schedule or benchmark timing threshold changes.

### Final Mega MoE matrix

The final source is `85c799e75ba5fd87`; the compiler patch SHA-256 is
`09e2422bb37e0260155c8d35b48b50c2d479fcdd5506cf5d7580e29c8efb95f8`.
Validation uses four socket-leased B300 GPUs, 148 SMs per GPU, driver
580.126.09, and CUDA/NVRTC 13.2.78. The implementation uses the same reference
instruction forms, not architecture-specific additions. All shapes retain
`fast_math=1`, activation clamp 10, and the reference's launch configuration.

Every row meets `TIRx/reference <= 1/0.99`. Each comparison uses 11 alternating
paired rounds and the unchanged cold-cache protocol. Outputs and cumulative
statistics match exactly before timing. A separate final-compiler run checks
every BF16 output bit on all four ranks with statistics both enabled and
disabled, for all 14 shapes.

| Shape | First TIRx / reference (us) | First ratio | Repeat TIRx / reference (us) | Repeat ratio |
|---|---:|---:|---:|---:|
| `deepgemm_default` | 2771.000 / 2781.000 | 0.996404 | 2771.000 / 2781.000 | 0.996404 |
| `port_t2_h1024_e8_k1` | 38.281 / 38.465 | 0.995216 | 37.662 / 37.872 | 0.994455 |
| `port_t64_e384_k6` | 555.657 / 555.422 | 1.000423 | 554.907 / 554.728 | 1.000323 |
| `port_t256_e384_k6` | 574.815 / 574.664 | 1.000263 | 574.814 / 574.535 | 1.000486 |
| `port_t1024_e384_k6` | 643.833 / 643.728 | 1.000163 | 643.851 / 643.578 | 1.000424 |
| `port_t8192_e384_k6` | 2433.000 / 2437.000 | 0.998359 | 2435.000 / 2439.000 | 0.998360 |
| `dsv3_t1` | 99.632 / 99.284 | 1.003505 | 99.804 / 99.152 | 1.006576 |
| `dsv3_t16` | 258.397 / 257.646 | 1.002915 | 258.338 / 257.466 | 1.003387 |
| `dsv3_t64` | 293.154 / 292.207 | 1.003241 | 292.668 / 292.292 | 1.001286 |
| `dsv3_t256` | 307.837 / 307.552 | 1.000927 | 307.797 / 307.618 | 1.000582 |
| `dsv3_t1024` | 440.524 / 440.020 | 1.001145 | 441.582 / 441.253 | 1.000746 |
| `dsv3_t2048` | 724.429 / 724.211 | 1.000301 | 723.768 / 723.679 | 1.000123 |
| `dsv3_t4096` | 1450.000 / 1447.000 | 1.002073 | 1449.000 / 1447.000 | 1.001382 |
| `dsv3_t8192` | 2626.000 / 2614.000 | 1.004591 | 2623.000 / 2613.000 | 1.003827 |

The original shared-input matrix and repeat pass all 28 rows. Removing an
unnecessary literal-bound unroll hint changes only the tiny shape's two cubins;
that shape is remeasured above with the rebuilt final compiler. The remaining
26 cubins are byte-identical to the measured images. The first one-token row
had elevated CPU pressure despite 93% instantaneous idle; its retained ratio
is 1.004376. The table uses the additional idle-preflight run (1.003505), not
that row. The repeat, other matrix rows, and final tiny runs have low pressure.
No failed timing is dropped or threshold widened.

The final dependency audit checks all **802** functions from the other three
ports: CUDA source, host LLVM IR, and serialized device/launch metadata match
their previously validated compiler byte-for-byte. A separate audit rechecks
every retained shape/origin/configuration in those first/repeat records:
18 two-shot, 136 all-gather, and 114 GEMM/all-reduce rows per round, all correct
and within 99%. This preserves the earlier evidence rather than inferring
performance from a similar-looking kernel.

With the final compiler, **980 feature tests pass** and the two changed CUDA
codegen/PTX test files pass **352 tests with 37 skips** under the socket lease.
The new polling-loop regression fails before the lowering fix and passes
afterward. The prior 94 multimem and five partial-CTA live cases remain
separate, no-skip evidence. All **734** tracked production/test files in the
remote validation checkout match this worktree. Raw timing and trace artifacts
remain outside Git. The broader-package failures documented below are not
reported as a green full-suite result.

### Earlier validation stages

The faithful two-shot all-reduce passes **18 shapes per round**, twice, on
four socket-leased GPUs with the existing CUDA 13.2.78 toolchain. Worst latency
ratios are **1.003703 / 1.003391**, below the unchanged `1 / 0.99` limit.
Every comparison matches CUTLASS bitwise, passes its independent-reference
precision check, and leaves all flags zero. No port-only instruction or
algorithm change is used.

The exact all-gather components and partial-CTA launch support pass **70
focused CPU tests** on the supported Linux Python 3.13 environment. The Rust
engine passes **888 unit tests, four checker-import tests, and three public
surface tests**. Five partial-CTA GPU microtests pass under the socket lease.
These results cover the actual one-thread helper kernels, without changing
their launch geometry. Partial-CTA collectives remain explicitly unsupported.

The faithful all-gather GPU matrix now has **17 shapes × 8 matched CUTLASS
configurations × 2 rounds** with all rows numerically correct; the
baseline-selected worst ratios are **1.001948 / 1.002322**.

On October 14, the release-mode race-check memory growth was traced to a
byte-occupancy mutation inside `debug_assert!`, which release builds omit.
The mutation now executes independently of the assertion. Both new clearing
regressions fail before the fix; all 13 interval-map tests pass in debug and
optimized release builds afterward. The complete engine suite passes **890
unit tests, four checker-import tests, and three public-surface tests** in
both build modes. The strong-flag-access regressions retain the independent
unordered-payload counterexample; 105 focused checker regressions and 82
additional related cases pass.

The complete four-rank Mega MoE numerical, Racecheck, and Synccheck examples
now pass, including the previously unbounded shared-pool alias test. After
restoring parameter-based peer-offset selection without a shared cache, the
same **four tests pass in 89.83 seconds**, and **26 IR/fidelity gate tests
pass**. That earlier scalar-argument port is superseded by the by-value-array
port described below.
The corresponding generic CUDA parameter-table lowering passes its initial
four-type GPU argument checks; its independent full performance matrix is
running and remains a separate gate.

A fresh supported-Linux run of the complete multimem/peer numerical and
ordering-rule matrix passes **617 tests in 168.50 seconds** with four pytest
workers. Repeating it with the patched compiler and adding the entire NumSim
registry/ABI suites passes **943 tests in 210.27 seconds**. This includes every
full-GEMM protocol/epilogue case, strong-flag positive and unobserved-payload
negative controls, all peer/multicast rule files, and complete Mega MoE checks.
All **94 live multimem GPU microtests pass in 886.48 seconds**, with one pytest
worker and four socket-leased GPUs. No cases are skipped. The workers use the
existing generic CUDA-target helper instead of a fixed device architecture;
kernel instruction forms and NumSim semantics are unchanged.

A subsequent full-package CPU run selects 4,099 cases but terminates after
the shared node's root filesystem fills; it is not a full-suite pass. The
storage failure also interrupts the remaining GPU matrices before they can
finish. The task workspace is moved, checksum-verified, to executable shared
memory behind its original path, recovering 19 GB without removing results or
changing the kernel source stamp (`a3693ef42fc9f8cb`). Completed valid rows are
retained; incomplete files and failure logs are preserved separately while
only unfinished comparisons resume under the same socket lease and unchanged
99% gate. The earlier 943 CPU and 94 GPU passing runs remain separate evidence.

After recovery, the final-source Mega MoE first round passes all 14 shapes
with bitwise-equal output/statistics; its worst latency ratio is 1.0044118.
The independent `dsv3_t1` repeat is 1.0134496 (98.183 versus 96.880 microseconds)
and still fails the 99% gate. Keep that miss distinct from an unrelated
`EADDRINUSE` startup failure: the latter's missing measurement is retried,
whereas the performance miss remains recorded for further lowering work.

The next generic runtime-lowering change removes four per-launch temporary
heap allocations when grouped parameters fit in 32 argument slots. CUDA code
and prepared kernel libraries remain unchanged. Its codegen suite passes
**79 tests with one skip**, and **11 GPU tests pass**, including stack/dynamic
boundary cases and library reloads. Two one-token comparisons pass at
**1.001445 / 1.008477**; the complete stack-packing matrix and repeat remain
open. A separate **13-test** launcher/fidelity run verifies bounded retries
for rendezvous port collisions, queue/environment cleanup, and immediate
propagation of non-startup errors. Bootstrap retries change no kernel or timer.

The rebuilt-compiler regression matrices for the other three kernels are now
complete. An audit checks the exact retained configuration sets, not only the
presence of a shape name. Every row is numerically correct and meets 99%:

| Kernel | Rows per round | Worst ratio, first / repeat | Baseline-selected worst, first / repeat |
|---|---:|---:|---:|
| Two-shot all-reduce | 18 | 1.0026662 / 1.0021546 | Same configurations |
| All-gather GEMM | 17 × 8 = 136 | 1.0034713 / 1.0042671 | 1.0024041 / 1.0034185 |
| GEMM + all-reduce | 19 × 2 × 3 = 114 | 1.0065379 / 1.0062278 | 1.0018165 / 1.0023831 |

These kernels have no scalar-parameter table candidates or `tirx.ptx.mov`
calls in the 802-function scope audit. The later grouped-argument stack path
does not execute for them. At that stage Mega MoE remained unresolved:
its stack-packing first tiny-shape comparison is 35.943/35.436 microseconds
(1.0143075x), despite exact output/statistics. That miss is preserved rather
than replaced by the passing one-token diagnostic comparisons.

The final Mega MoE port replaces the scalar-selection experiment with one
read-only, by-value rank-offset array, matching DeepGEMM's parameter storage.
Plain unsigned local-register addition also becomes visible to normal
lowering; predication, saturation, and other types remain explicit. No
instruction upgrade, alternate protocol, work repartition, or schedule change
is used. The final source stamp is `06dfdd4b74d34861`.

The preceding source stamp `979bd32f26952708` produces exact output and
statistics on all 14 shapes in a fresh four-rank NVRTC matrix. Its ratios are
within 0.42% on 12 shapes; `dsv3_t1` is **1.011856** and `dsv3_t8192` is
1.004211, so the matrix does not establish the 99% gate. The result is
state-sensitive: an isolated 31-round `dsv3_t1` run is 1.007077, and the same
shape after a large shape plus a 10-second cooldown is 1.000252. The matching
`port_t2_h1024_e8_k1` cooldown run is 1.048116, so the remaining tiny-shape
tail is still an open lowering/performance issue rather than a clean pass.
The earlier passing rounds remain historical evidence and are not relabelled.

For the other three kernels, an additional complete compiler-output audit
compares all **802** rank/configuration functions between their measured
compiler and the final compiler: **CUDA source, host LLVM IR, and serialized
device/launch metadata are byte-identical**. Coverage is 456 GEMM/all-reduce,
274 all-gather including its helpers, and 72 two-shot functions. No later
parameter-array or register simplification silently changes those ports.

NumSim ABI **42** snapshots by-value arrays per rank, preserving that policy
through native checker bindings, artifact manifests, and cache identity.
The ABI 42 run passes **106** combined peer numerical/race/sync, parameter,
ABI, and fidelity tests, including all complete Mega MoE cases; another
**41** snapshot/cache tests pass. The Rust suite again passes **890 + 4 + 3**
tests in both debug and release. The compiler suite passes **89 CPU tests
with one skip** and **15 focused GPU tests**, including all six array types,
two extents, direct calls, library reloads, special-value bits, and unsigned
wraparound. New modules explicitly reject older runtimes before launch.
The broad CPU run completes without a resource guard: **4096 passed, 33
failed, one xfailed in 953.43 seconds**, using `-n 16 --dist=worksteal` and
excluding live GPU markers. This is not a green package gate. The pristine
upstream comparison completes too, but its old `LaunchTopology` constructor
cannot consume the pinned frontend's `threads_per_cta` field; its widespread
failures are not a useful blanket explanation of the current failures.

The run exposes stale expectations that strong flag reads must race and that
completed unnamed Mega MoE writes must retain old named provenance. Those
expectations are corrected, not used to restore the false positives. The weak
publication and unrelated-payload negative controls remain. **67** focused
wait/flag/diagnostic/canonical tests now pass, as do **44** parameter/cache/ABI
tests. Three compiler layout negative controls fail before their fix and pass
afterward. The 59 simplifier tests and both source-generation PTX controls
also pass; the latter use source-only fallback, not a newer assembler.

The final ABI 42 feature matrix passes **972 tests in 203.04 seconds**,
including all multimem/peer numerical and ordering rules, registry/ABI checks,
and by-value parameter tests. The final compiler's live rerun passes **99 GPU
microtests in 415.11 seconds**: all 94 multimem cases and five partial-CTA
cases, with four pytest workers under one four-GPU socket lease. All 15
focused compiler GPU cases pass again. Both the current and compatible-parent
source-to-wheel/install/numerical-smoke tests pass after restoring real Git
metadata, the pinned exported dependency revision, and the required `pip`.

The broader CPU gate is still not green. Serial identical-node comparisons
against compatible parent `ee1d815` reproduce ten unsupported-host-`Evaluate`
cases, three stale `cu` bindings, four Rust/LLVM compiler crashes, and the
maximum NumSim CPU-budget failure. That case takes **167.457 / 167.050 seconds**
(current/parent), above the unchanged 140-second budget; the original
149.860-second miss also remains recorded. All six corrected canonical checker
performance cases execute cleanly and complete all tasks; five pass their CPU
budgets, while the medium case takes **17.641 seconds** against 16.7. The
parent's recorded medium-case time is 16.746 seconds, also above that cap.
No threshold is widened. These CPU limits are separate from the GPU
fidelity/parity gate and are not reported as a passing package suite.

## Historical evidence

The latest port correction restores the pinned reference's 32-bit shared
mapping and explicit shared-cluster barrier-arrival forms at four helper
sites, without strengthening the release scope or changing work/scheduling.
Source `979bd32f26952708` passes 82 focused checks and two one-token GPU
comparisons at 1.0000308 / 1.0032366, with exact output/statistics. The tiny
case passes first at 0.9891977 but misses its repeat at **1.0256084**
(33.001/32.177 microseconds). That stage did not establish full parity. The
[lowering evidence](lowering_optimizations.md#restoring-reference-shared-address-forms)
records the exact reference, disassembly delta, and all four comparisons.

After the ABI update, **138 additional targeted tests pass** for the complete
GEMM numerical/race/sync matrix, IR contracts, support-table generation, and
benchmark evidence gates. The faithful all-gather GPU probe matches CUTLASS
bitwise and measures 47.54 versus 47.55 microseconds at the same configuration;
the full shape/configuration matrix and repeat are complete. Nine Mega MoE IR
checks retain per-token serial chunks and distinguish baseline timeout, load,
and fence forms from the separate strengthened example.

The pinned DeepGEMM reference builds with its original kernels and submodules
(`559d79f`, CUTLASS `f3fde583`, fmt `553ec11`). PyTorch 2.14 requires a C++20
override for the host extension; CUDA/NVRTC remains 13.2.78 and the CUDA JIT
instruction choices are unchanged. The benchmark separately checks output
and cumulative statistics before timing and rejects stale build stamps.

The October 10, 2026 handoff preserves the prior GB200 measurement tables and
ports on upstream `66681fb`, including the `tirx_harness/` to `tools/` directory
rename. Raw benchmark JSON/JSONL artifacts are intentionally not committed.
Subsequent validation of runtime sources at `deda386` built the native
frontend and passed **499 CPU tests in 146.23 seconds**, using four pytest
workers with work stealing. This covered the three multimem/peer runtime
files and all ten new Racecheck/Synccheck test files listed in the feature
guides. Later commits through `0d2f86e` only change documentation.

That run used Rust 1.99 and Python 3.11, with the native extension staged
manually after the wheel rejected Python 3.11. It demonstrates those CPU
checks executing, but does not pass the installation gate: the package
requires Python 3.12 or 3.13. Subsequent full-GEMM and Rust checks below use a
supported installed Python 3.13 environment. The broader Python rule and
parent-package suites still need their supported-environment rerun.

The initial live GPU microtest suite did not pass. Its first attempt lacked `cuda.bindings`;
after that dependency was installed, the retry reported failures before the
SSH connection was lost. The final diagnostics were not captured, so no
kernel or hardware cause is established for that attempt. The subsequent
94-case live microtest run above passes under the shared socket lease. It
does not replace the full-kernel performance-equivalence gate.

The subsequent full-scale publication correction passes **29 local tests**
under Python 3.13.2, TVM 0.27.0.post1, and TIRx-kernels 0.1.2.post1:
19 IR contract tests cover system-scope arrivals/polls, alias fences,
one/two-CTA builds, fixed/runtime rank, and explicit origin-ordering opt-in;
10 benchmark-evidence tests reject stale/missing/incorrect or slow results.
These do not execute the kernel's memory effects. On the supported Linux
worker, the full-scale GEMM NumSim/Racecheck/Synccheck matrix now passes:
**42 passed in 126.58 seconds**, covering both protocols, both epilogues,
two/four ranks, three output dtypes, and origin-ordering counterexamples.
The run exposed two preflight issues that are fixed in `6a61962`: direct-store
builders no longer request an unused host TensorMap, and descriptor discovery
skips multicast-window allocations.

Expanded full-GEMM coverage also passes in that supported environment:
**90 tests in 68.96 seconds**, using four pytest workers. This comprises
23 IR/host-descriptor checks and 67 numerical, Racecheck, and Synccheck cases.
The added cases execute multiple persistent clusters, one/two-CTA MMA, both
epilogues, uneven work counts, and an alias-fence removal negative control.
These tests run the committed kernel builder, not a GPU-only optimization
prototype, and do not establish performance parity.

A prior broader Linux Python run stopped after 564 passing cases because a worker's
memory use kept growing; this is not a full-suite pass. Isolated, identical
reruns of
`test_mega_moe_expert_parallel_has_no_stale_pool_alias_advisory` exceeded a
12 GiB resident-memory limit in both the current tree and `13eeeca`. The
complete Mega MoE analysis was therefore unvalidated at that point, rather than
classified as a passing test or a new release-RMW regression. The October 14
release-build correction and end-to-end passing results above supersede that
blocker; the interrupted broader suite is still not a full-suite pass.

The subsequent engine checks reproduced and corrected acceptance of invalid
multimem specializations and stale provenance after unnamed writes. The nine
multimem Rust tests pass. The complete Rust suite on macOS reports 770 passing
and 113 failing tests; an untouched `6b24e8c` engine snapshot reports 763 passing
and the exact same 113 failures. All seven added regression tests pass. The
failure-set comparison introduces no new failures; macOS execution is blocked
by the engine's unsupported worker floating-point environment. A later supported
Linux run at `90d82ea`, with Rust 1.99 and Python 3.13.15, passes **887 Rust unit
tests, 4 architecture/import checks, and 3 public-surface checks**. It includes
a reproduced-and-fixed false positive for a returned release RMW followed by an
acquire fence. The regression also rejects a fence before the read and a
reduction's non-returned read. The installed native frontend still lowers all
28 existing multimem instruction cases. The new bulk-copy provenance kernel
also lowers; execution of it and the complete Mega MoE case remains pending.

The ABI/support-matrix tests and the 29 IR/evidence tests pass together:
**32 passed in 7.62 seconds** with four pytest workers. The broader registry
run exposed the stale ABI header, now corrected; its other 16 failures require
generated native modules that could not link CPython symbols on macOS. Those
execution-dependent checks still need the supported Linux environment.

The separately strengthened `sys_fenced` GPU implementation is numerically bitwise-correct
on the exercised probe shapes, but has **not** passed the 99% performance bar.
That exploratory optimization work is not part of the faithful-port POC.
Its complete 19-shape matrix and independent repeat did not establish the
final gate. Its slowdown is not a measurement of like-for-like IR overhead.
For the port-fidelity gate, preserve the original comparison kernels and the
99% threshold; neither historical data nor a differently optimized prototype
closes it. Every accepted timing must use only the GPUs actually assigned by
the socket lease; an override of that assignment invalidates the run.

On October 13, a fresh four-rank GPU screen covered **all 19 declared shapes
against both pinned origins: 38 shape/origin comparisons**, with one retained
top configuration per pair. Commit `b82113f`'s then-default `sys_fenced` builder
matched each origin bitwise on every rank, satisfied the independent float64
reference's output-rounding tolerance, and reset every flag. The kernel source
SHA-256 was
`362e524feadef94cb9546a92caef4a49b0b6ecbfae5e52b28d985b4d257b21f0`;
the run used Python 3.13.15 and NVRTC 13.4.92 under the shared socket lease.
This expands numerical evidence, **not** the performance claim: its triage
timings used 20 graph launches, 3 interleaved trials, a 10 ms target, and 0.1 s
idle, rather than the final comparison's 100 launches, 9 trials, and top-three
configuration selection. At that stage the full performance gate and independent
repeat remained open. A separately screened optimization prototype also matched the
outputs, but its scoped alias fences are not yet modeled by NumSim; it is
neither the committed implementation nor checker-validated evidence.

The pushed benchmark preflight was then run in the supported Linux environment
with Python 3.13.16, TVM 0.27.0.post1, TIRx-kernels 0.1.2.post1, and Rust 1.99:
two ranks, four ranks, and four ranks with two CTAs per rank all passed
NumSim, Racecheck, and Synccheck. A separate four-rank GPU run of the same
two-shot TIRx kernel matched the float64 reference and left every flag reset,
under the shared GPU lease. The GPU check is a correctness spot check, not the
full performance matrix or the full-scale GEMM gate.

Local handoff checks passed for all 52 changed Python files' syntax, one JSON
result file and 11,950 JSONL records, 37 local documentation links, 32 named
rule-to-test references, documentation code-block syntax, and patch whitespace.
Sphinx is not installed locally, so a rendered documentation build is also
pending. None of these static checks substitutes for the native or GPU gates.

An offline audit of the historical measurements found the following worst
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
[peer results](MULTI_GPU_PEER.md#results). The raw benchmark JSON/JSONL files
are intentionally omitted from this branch. Recomputing these ratios from a
local run is an audit of recorded data, not new GPU correctness or timing
evidence. Recorded tables do not replace fresh validation of the final commit.

The previous session reported 245 peer tests, 348 multimem tests including
GPU microtests, 885 engine tests, and 15 frontend tests passing. These counts
are historical reports; they have not been reproduced on this branch.

## Scope and limitations

- **Full-scale GEMM + all-reduce now has faithful-port validation.** The default
  is now `ordering="origin"`; preserve the source's operations instead of
  compensating for overhead with new instructions or scheduling patterns.
  The explicit `sys_fenced` examples have separate numerical/checker evidence.
  Their clean verdicts do not establish the baseline's safety, and their
  timings do not establish port fidelity. The restored port passes all 19
  shapes, both origins, and the independent repeat as recorded above. Rows
  identify ordering; the gate rejects unlabelled history, another mode's
  measurements, or choosing a different port configuration to hide overhead.
- **Mega MoE's provenance correction passes its end-to-end rerun.** The prior
  result was `review` for one `alias_stale_read` advisory. Focused Rust
  regressions now verify invalidation of overwritten named provenance,
  including partial and sparse writes, without discarding earlier findings.
  The full-kernel test now requires `clean` and passes after correcting the
  independent release-build byte-occupancy mutation described above.
  The [provenance explanation](MULTI_GPU_PEER.md#shared-pool-alias-advisory) describes
  the completed unnamed bulk-copy write and correction. The focused alias and
  peer-kernel checks pass. The broader wiki corpus and relevant performance
  gates remain separate validation obligations.
- **Coverage is bounded.** Async multimem copy/reduction/store families and
  fp8 forms are not modeled. The supported-operation table and rejection
  paths must stay consistent. The numerical model has GB200-specific floating
  reduction behavior, with one-ulp tolerance for half sums. A green test set
  is not a claim to cover arbitrary PTX forms, inputs, or GPU architectures.
- **Full-kernel coverage remains bounded.** The multimem ports now live
  in `ported/cutlass/`; tests own only bindings, references, and focused rule
  kernels. Two-shot retains its existing checker coverage. The new full-scale
  GEMM tests complement the small model and their CPU matrix passes, but that
  result does not replace the final GPU validation or the other corpus gates.

## Reproduction and acceptance gates

1. Restore the compatible harness environment on a four-GPU Linux host, initialize
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
