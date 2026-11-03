# IKET Profiling

Use NVIDIA IKET to inspect the annotated GPU timeline. Let the target kernel
and its trace determine the analysis; do not impose a fixed event taxonomy,
script structure, critical-path algorithm, or report template.

## Canonical usage

Resolve the canonical checkout through `$tirx-wiki` using the same Python
interpreter as the user's workflow, then read the closest annotated kernel
before profiling. All paths below are relative to that checkout:

- FlashAttention-4: `tirx_kernels/ported/flashattention/flash_attention4.py`
  for token ranges, point markers, leader-only sentinel tokens, compilation,
  and single-kernel capture.
- Sparse FlashMLA dispatch: `tirx_kernels/ported/flashmla/flash_mla_sparse_fwd.py`
  for multi-implementation compilation and capture. Its annotations live in
  the selected implementation; use
  `tirx_kernels/ported/flashmla/sparse_prefill_head64_phase1.py` (head64 phase 1)
  as a representative annotated implementation.

They are the source of truth for input preparation, module entry points, and
capture options. Run the selected module with `--help`, preserve its
representative shape and launch path, and write artifacts to a caller-selected
directory. For a new kernel, adapt the nearest canonical example instead of
inventing a separate workflow.

## Annotate and switch capture

Keep one annotated `PrimFunc` for both ordinary execution and IKET. Give each
range a stable, semantic name and store its token in a local `uint32[1]` before
passing that token directly to `range_end`:

```python
def iket_range(name):
    token = txl.alloc_local([1], "uint32")
    txl.assign(token[0], txl.cuda.iket.range_start(name))
    return token

token = iket_range("load-kv")
# work whose duration and overlap matter
txl.cuda.iket.range_end(token[0])
txl.cuda.iket.mark("phase-boundary")  # an instant, not a duration
```

For a range active on only some threads, initialize the token with
`txl.cuda.iket.sentinel_token(name)` and overwrite it with `range_start(name)`
inside the participating path, as FA4's `leader_only` helper does. Keep the
matching `range_end` on the common path; the sentinel makes it a no-op for
non-participants. Do not add synchronization merely to align annotations,
because that changes the pipeline being measured.

Compilation is the on/off switch; do not maintain a second kernel or gate the
annotations behind a source-level profiling flag:

```python
module = tvm_tirx.IRModule({"main": func})

# Off: the TIRx pipeline strips IKET annotations and their token storage.
plain = tvm_tirx.compile(module, target=target, pipeline="tirx")

# On: preserve and lower the annotations for official IKET capture.
instrumented = iket.IketProfiler().compile(
    module, target=target, tir_pipeline="tirx"
)
```

Launch `instrumented` from the replayable callable passed to `iket.run`.
`iket.run` owns capture and postprocessing; it does not replace the IKET-enabled
compile step.

Keep the ordinary benchmark as the performance baseline; IKET instrumentation
timing is diagnostic. Do not run IKET concurrently with another GPU profiler
on the same device.

## Questions to answer

Use the JSON, Perfetto trace, source, and any analysis code you find useful to
answer all three questions:

1. Is an unusually slow operation or span exposed on the critical path?
2. Is missing overlap exposing bubbles that may leave hardware underused?
3. Is insufficient or unevenly distributed work creating a load-balance tail?

The final report must address each numbered question explicitly. For each one,
state a definite conclusion and its evidence. If the current annotations or
capture cannot distinguish the answer, the profile is incomplete: add or
refine the necessary annotations, capture again, and continue until the
question is answered. Do not report insufficient evidence as a conclusion. If
the trace shows no problem, say so and cite the evidence. Never omit a question
because it appears inapplicable or because another finding seems more
important.

How to identify spans, reconstruct overlap, group tasks, or present the result
is analysis-specific. Follow the kernel's roles and annotations rather than a
generic schema.

IKET establishes annotated ordering, duration, overlap, and CTA/warp lifetime.
It does not by itself establish achieved hardware utilization, bandwidth,
occupancy, or hardware stall reasons; use NCU when the conclusion requires
those counters.
