# NumSim and Native Analysis Engineering

NumSim transpiles TIRx to native Rust and executes a numerical GPU model; native analyses reuse that execution where implemented. Current source and public APIs define implemented interfaces—not intended hardware semantics or stale migration plans. Production code must not import tests or depend on paths outside packaged `tirx_harness`.

## Priorities

1. Correctness: model intended GPU semantics; passing tests or simulator output are not their own oracle.
2. Completeness: state exactly what was checked; unsupported semantics or limits must fail closed.
3. Performance: optimize only after correctness is established, without weakening simulation or analysis.

## Problem-Solving Workflow

- Reproduce with the smallest focused semantic case or real kernel, then validate on every affected representative corpus, including wiki kernels when relevant.
- Before editing disputed semantics, settle the facts and a simple causal algorithm: actors, state, footprints, lifetimes, HB, terminal conditions, counterexamples, and complexity.
- Distinguish issue order, execution/completion order, HB, and register dependency; never infer one from another.
- For every kernel-invalid finding, attach every causal source site or witness its schema supports and explain why the evidence proves invalidity; classify infrastructure failures separately.
- `error` requires proof; `review` is one precise advisory or unresolved risk, never missing coverage or a substitute for a provable error, and must not stop later findings; `incomplete` means execution or coverage cannot support the claim and is never success.
- Triangulate specification wording, canonical implementations, and preferably a focused GPU microtest; flag remaining hardware uncertainty instead of choosing the most convenient source.
- Numerical changes define dtype, rounding, overflow/saturation, special-value, mask/lane, and approximation behavior, then validate it against an independent reference or GPU.
- Diagnose measured bottlenecks. Separate engine build, warm-target/new-kernel transpile, setup/reference work, and engine execution.
- Prefer the smallest complete fix, but never replace requested semantics with an easier approximation or a test-specific exception.

## System Design

- Every native mode claiming complete coverage executes the same full numerical/control/memory path; analyses observe its transition/effect stream and must not project values, skip paths, or change program-visible behavior.
- Generated code owns source-known canonical/layout facts and per-occurrence native control flow; it does not own analysis policy.
- The engine owns mutable launch/runtime state, scheduling, physical resolution, async lifecycle, and HB; shared analysis facts stay under `native_analysis/`, with tool-only state and verdict policy under `native_analysis/<tool>/`.
- Model generic state transitions and effects, not kernel syntax, loop shapes, variable names, or other source-pattern matches.
- Give async operations a shared lifecycle vocabulary: issue, active effect, publication, partial/full completion, wait, and consume.
- Put state at its narrowest real owner; add shared state, locks, or cross-actor effects only after identifying actual conflicting readers/writers and dependencies.
- Declare whole-kernel deadlock only when no modeled actor or pending async effect can make progress; one spinning/polling actor or a threshold is not proof.
- Reuse executors, queues, shadow memory, vector clocks, and analysis infrastructure. A distinct path must name an invariant the shared path cannot satisfy and an actual consumer.
- Keep scheduling reproducible for fixed inputs/config/seed. Add exploration/frontiers only for a named check impossible from canonical execution, with a consuming verifier and explicit coverage; exhausted coverage is `incomplete`.
- Keep the public ABI minimal: merge duplicate facades and require a generated-code caller that cannot use one; version artifact boundaries and reject stale caches after compatibility-affecting ABI/layout/codegen changes.
- Delete superseded experiments, dead compatibility paths, and half-implemented algorithms once the replacement is proven.
- Keep public tool interfaces aligned; user-facing reports and docs expose actionable source evidence and scope, not internal plumbing.

## Evidence, Tests, and Handoff

- Explicitly read `../../../tests/CLAUDE.md` before test or performance work; it is a sibling guide, not inherited here.
- Tool-specific Python tests go under `tests/analysis_tools/<tool>/`; NumSim semantics under `tests/numsim/`; generic native engine/cache coverage under `tests/native_engine/`; private Rust units stay beside implementation.
- Do not weaken existing NumSim tests to accommodate native analysis. Preserve or migrate every non-deprecated contract; intentional contract changes need proof and explicit test deltas.
- Corpus goldens record every finding from a run, including simultaneous statuses, with kind and source evidence; alternatives require a documented genuine source of variability.
- Run focused tests while iterating, then the parent package gate and relevant wiki/corpus gates; deliver no unexpected `incomplete` and report exact blockers.
- Engine changes also run `(cd tools/src/tirx_harness/numsim/engine-rs && cargo test --all-features)`.
- Benchmark the named default optimized artifact and compare like-for-like per-kernel phase timings; test worker scaling only when relevant. Gate changes require clean measurements; temporary relaxation must be explicit and scoped.
- Preserve unaffected numerical/report/performance behavior; name, justify, and test intentional deltas, especially checker conflict classes or memory-ordering scope.
- Keep each PR scoped to the requested tool; justify shared/cross-tool changes and preserve excluded or experimental work separately.
- Use the caller-assigned worktree; otherwise isolate implementation work. Update an owned feature branch against the requested base before handoff and never disturb unrelated changes.
