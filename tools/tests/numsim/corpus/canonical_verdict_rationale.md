# Canonical corpus verdict rationale

This ledger explains the non-clean entries at the canonical kernel revision
owned by the wiki gitlink. It is explanatory, not executable configuration:
`canonical_cases.py` is the sole owner of exact verdicts, finding kinds,
counts, and worker limits. Historical same-tree before/after commands and raw
outputs remain in the commits listed below.

Every accepted `review` case completes its independent numerical oracle. A `review`
therefore records a source-visible questionable access whose value cannot
reach the checked result, or a physical ordering ambiguity that the checker
must not silently decide. An `error` is never metadata for an excluded source:
it must be resolved as either a kernel defect or a checker defect.

## Source-backed review entries

### FlexAttention SM103 score-to-probability reuse

The full-block forward case passes the independent output/LSE oracle and
Synccheck. Racecheck retains two TMEM `read_write` reviews: the softmax role
loads score fragments, transforms them into probability registers, and stores
packed probabilities into overlapping TMEM columns. The checker does not prove
load completion from this register dependency, so the
`async_lifetime_not_drained` findings remain visible. This is not an ERROR
waiver or a claim of native SM103 validation on the B200 runner.

### Agent-evolved MSA/VSA TMEM lifetimes

The complete small MSA and two-phase VSA cases retain respectively 16 and 22
TMEM read/write lifetime REVIEWs. Score reuse and output rescaling consume
loaded registers without explicit load-completion waits at every reuse. Native
Racecheck cannot prove those register dependencies; these are not CLEAN claims.
Both numerical cases match the independent sparse-attention oracle. MSA's
separate shared metadata ERROR is repaired by including the MMA warp in the
existing slot-release barrier, so the producer cannot overwrite its masks early.

### MSA uniform FP8 unused query rows

Q's tensor-map copy initializes 16 rows of the 128-row shared-memory MMA
tile. The remaining 112 rows are read by QK but cannot reach the 16 live
query heads' output or LSE. The numerical and B200 GPU executions agree with
the independent attention oracle. Keep these shared-memory uninitialized-read
REVIEWs; they are not the repaired final CORR_SIG phase handoff. The merged kernel
also pins its 168-register entry allocation and repairs the invalid 232/232/96
redistribution with 176/176/152: WG2 releases 2,048 registers, exactly funding
WG0/WG1's increases. The checkers retain register-budget accounting even when the
compiler ignores unconfigured hints; compiler behavior cannot excuse an invalid
source contract.

NumSim and Racecheck gather these FP8 rows in 16-byte atoms: their 896 read reviews
cover the same 14,336 uninitialized bytes previously reported byte by byte.
Both tools retain the complete footprint and REVIEW severity.

### MSA Q1 initial output scratch and TMEM lifetimes

The first correction reads uninitialized TMEM O into `partial[16]`. Its
rescaled value is subsequently discarded by the first PV MMA, whose
accumulator-enable predicate is false. NumSim therefore matches its independent
output/LSE oracle while retaining the propagated register `uninitialized_read`
reviews. The merged kernel gives each row-state update a single owner and
waits for the final O_FULL handoff before CORR_SIG; the former shared-memory
race and barrier-generation ERRORs are gone.

The complete repaired trace additionally exposes TMEM `read_write` reviews:
correction reads old O before its read/modify/write, reads statistics before
the next statistics reuse, and loads scores before later QK reuse. These are
`async_lifetime_not_drained` witnesses, not the repaired ERRORs. A true register
dependency alone does not prove memory completion ordering, so they remain
visible rather than being called safe or clean. Synccheck has no findings and
retains only the initial scratch advisories.

### MSA reverse-prefill score lifetime

The merged kernel preserves the original score registers instead of reloading
the TMEM region after P overwrites it. Both split and final LSE now match the
independent oracle; the overlapping store-to-reload `write_read` ERROR is gone.
The producer retains the original score-load-to-P-store `read_write` REVIEW
(`async_lifetime_not_drained`). This is not a proof of memory completion from
register dataflow. The reduction phase and both Synccheck phases are clean.

### GDN2 recompute register-dependent TMEM reuse

The 17-token case exercises a nonzero initial state, two chunks, and both
checkpoints. NumSim and both Synccheck phases are clean. Racecheck retains two
`read_write` reviews in the main phase: `stage_y` loads the two `state_k`
fragments from TMEM, packs their register values into `y`, and publishes
`protocol[8][0]` only after storing `y` and waiting for those stores. The next
MMA reuses the old TMEM slice after this handoff. The checker deliberately does
not infer the intervening register dependency, so these remain explicit
reviews, not clean findings. The descriptor prologue is clean in both checkers.

### MSA long-prefill register-dependent TMEM reuse

The paged GQA16 case matches the independent FP8 partial-output, per-32-channel
scale, and LSE oracles, and Synccheck is clean. Racecheck retains two TMEM
`read_write` reviews: score-load registers feed arithmetic, exponentiation,
and BF16 conversion before the probability store reuses TMEM. As with the
existing FlashAttention entries, the checker does not infer this register
dataflow. The reviews remain visible; no ERROR is reclassified or excluded.

### Cake ultrasparse BSR TMEM lifetime reviews

The six-selected-block case is numerically and synchronously clean. Racecheck
reports 14 `read_write` reviews on TMEM reuse after `tcgen05.ld`. These remain
completion-order advisories: the checker does not prove the intervening
same-thread register dependencies. Recording their exact count does not claim
that all memory ordering is proved, and does not downgrade an ERROR.

### Cake and MSA M64 split score loads

The compact Cake, both long-sequence Cake, and MSA M64 cases load two score
halves using `16x32bx2.x32` with a half-split offset of 64. The split offset is
in physical columns, not the register count. With this mapping implemented,
their independent numerical oracles and Synccheck runs are clean.

Racecheck retains score-load-to-P-store `async_lifetime_not_drained` reviews:
the two score loads overlap the later `16x32bx2.x16` P store at TMEM byte
ranges [256, 320) and [384, 448). MSA M64 additionally reloads both scores
after computing the temperature sum, yielding the second pair of witnesses.
The source sites are each kernel's `_tmem_load_x32` helper and probability
store; no load-completion wait precedes that reuse. As above, numerical
agreement or register dataflow does not prove the missing memory ordering.
These explicit reviews do not turn an ERROR or incomplete execution into a
pass, and this refresh does not change kernel code to remove them.

### cuDNN BSA combine padding rows

The one-token `cudnn_sm100_bsa_forward_combine_blk64` case exercises a
registry-supported `seqlen_q=1` launch. In
`tirx_kernels/ported/cudnn/bsa/_block_sparse_attention_forward_sm100_blk64/kernel.py`,
`load_o_stage` fills only valid query rows, while the combine loop issues the
two vector shared loads for `row0` and `row1` before applying its valid-row and
positive-weight predicate to the accumulation.

NumSim, Synccheck, and Racecheck therefore observe the same 128 reads from
unwritten padding rows, and all eight checker tasks complete. The independent
split-LSE oracle checks the only valid output row and matches exactly, so the
padding values cannot reach a supported output. `review` records the executed
reads without hiding them as `clean` or misclassifying them as an output race.

### cuDNN GDN recompute unused pair-box accumulator half

The one-token `cudnn_sm100_gdn_recompute_f16` case has one real K chunk in
pair box 0. In
`tirx_kernels/ported/cudnn/linear_attention/gdn_recompute_f16.py`, `issue_k` writes
only that selected 8-KB box, while `fused_kk` deliberately retains the prefill
kernel's M=128 MMA shape across both pair boxes. The source comment states that
half of the resulting accumulator is never read.

NumSim and both native checkers therefore observe 8,192 BF16 reads from the
unwritten box. The independent one-token recurrence checks the complete final
state, so the materialized zeros are not being used as the numerical oracle.
`review` preserves the source-visible dead reads without misclassifying the
intentional inherited MMA shape as a synchronization or memory-safety error.

### DeepGEMM block-scaled FP8 scale padding

The BMM, contiguous and masked M-grouped, and selected FP8 1d1d cases instantiate
`tirx_kernels/ported/deepgemm/_sm100_fp8_fp4_gemm_1d1d/kernel.py`.

- Lines 184-203 derive the logical and 128-word-aligned scale extents.
- Lines 519-533 allocate aligned shared scratch, while lines 940-963 transfer
  only the logical TensorMap box.
- Lines 1559-1576 explicitly load the aligned extent for transposition.
- Lines 338-349 encode the block-scaled MMA descriptor from the logical
  `umma_n`; lines 1291-1335 select the corresponding scale operand.

The diagnostics are therefore real shared-memory padding reads, not inferred
simulator accesses. The descriptor's logical N excludes the padding, and the
bitwise NumPy oracle checks every semantically written output. `clean` would
hide executed reads; `error` would incorrectly claim that the padding can be
selected by the MMA. K-grouped is clean because its logical and aligned scale
extents coincide. The older FP4 paged-MQA review predates this refresh and is
left unchanged; it intentionally has no new exact kind/count contract here.

In the bundled source, the persistent contiguous M-grouped path waits for the prior
task's stage generation and then executes the async-to-generic proxy fence at
line 1562 before its transpose stores reuse shared scale bytes. Its former
`read_write` error is therefore gone; the 64 independently justified padding
reads above remain `review`.

### FlashInfer FP4 RMSNorm reduction scratch

The `flashinfer_rmsnorm_fp4quant` case uses the pinned FP16, H=64, block-16,
E4M3 path. The kernel stages each row through dynamic shared memory, writes
warp partial sums at lines 836-843 of
`tirx_kernels/ported/flashinfer/norm/rmsnorm_fp4quant.py`, and reads the reduction
scratch at lines 844-853 before the scale/pack stores. NumSim reports 480
`uninitialized_read` diagnostics from that shared reduction workspace while
the independent FP32 RMSNorm and bit-level FP4 oracle matches both output
buffers exactly. These are source-visible scratch reads whose zero materialized
value is not accepted as numerical evidence, so the manifest records
`review`, not `clean` or `error`.

### MSA NVFP4 TMEM lifetime review

The `msa_sparse_atten_fwd_nvfp4_kv_sm100` case uses the pinned NVFP4-KV
single-token path with one non-zero K/V block and an independent uniform
attention oracle. NumSim and Synccheck complete cleanly. Racecheck reports two
same-warp TMEM `read_write` reviews, one for each P store at columns 64 and 96.
Both stores consume values derived from the corresponding S-load
registers at columns 64 and 96, while the native checker does not model that
register dataflow. Operation-granular review retirement keeps both witnesses
visible. The manifest records the two reviews with their exact kind/count;
they are not treated as execution errors.

### FlashKDA aligned tiles

The T2-T6 implementations under `tirx_kernels/ported/flashinfer/kda/` initialize
only real token rows/columns but issue aligned `ldmatrix` loads over `sVec`
and, for T5/T6, both Gram halves. Their subsequent coefficient stores and
output maps are guarded by the real token set. The dense FP32 recurrence
oracle independently checks normalization, gates, decay, the recurrent
update, and every BF16 output. The manifest owns the specialization-specific
counts; the common verdict reason is an executed padding read that cannot be
selected by a real-token output map.

### FP32 MTP GDN dead output accumulator reads

The canonical `gdn_decode_fp32_mtp_warp` case selects `ILP_ROWS=4`. In
`tirx_kernels/ported/flashinfer/gdn_decode/gdn_decode_fp32_mtp_warp.py`, lines
1301-1302 initialize two temporary output values by reading `output_sums`.
Lines 1303-1305 immediately replace both values with the initialized
`output_lo` entries before conversion or storage. `output_sums` is the
non-ILP4 accumulator and is intentionally not initialized on this branch.

NumSim, Synccheck, and Racecheck each observe the same 17,408 register reads
from the refreshed kernel: the packed-output path materializes two temporary
`output_sums` pairs before replacing them with the initialized `output_lo`
values. All tasks complete, no checker is incomplete, and the full independent
NumPy recurrence oracle has zero mismatches. The reads are therefore
source-visible but dead;
`review` records them without treating their materialized zeros as numerical
evidence.

### Selective-state-update stochastic-rounding inputs

The canonical `selective_state_update_mtp_vertical` case sets
`philox_rounds=0`. In
`tirx_kernels/ported/flashinfer/mamba/selective_state_update_mtp_vertical.py`, line
610 allocates `row_random`, while lines 611-622 initialize it only when
`PHILOX_ROUNDS > 0`. The active final-state path at lines 758-760 still copies
those registers into `random_words` before calling `_store_state_row`.
Because `_store_state_row` selects its non-stochastic branch, that parameter
is never consumed by conversion or output storage.

This is not missing NumSim semantics: the TIR contains explicit uninitialized
register reads, even though specialization and native optimization can remove
the dead copies. NumSim, Synccheck, and Racecheck all complete, the token-serial
oracle matches the full output and updated state, and `review` preserves the
source read without claiming numerical influence.

### Physical Racecheck reviews

`gdn_prefill_sm100` retains sixteen physical TMEM read/write reviews: ten
same-warp load/transform/reuse chains and six cross-warp lifetime handoffs.
Every reviewed load has all of its destination registers consumed before the
corresponding empty-barrier publication, but Racecheck does not infer that
register dataflow. The former `implicit_tmem` versus
`raw_tcgen05_tmem` advisories were checker-created name splits: raw MMA
footprints now inherit the logical owner of their TMEM anchor without changing
addresses, footprints, ordering, or schedules.

`gdn_cp_prefill_sm100` has twelve same-warp TMEM `read_write` reviews in its
M/N-precompute phase and twelve reviews in its prefill phase. M/N precompute retains four M slice
load/transform/store reviews in warp 0. In warp 4, each of the four N rescaling
stores observes both the earlier N-to-input materialization load and its own N
rescaling load, producing eight reviews. Prefill retains six same-warp chains
and six cross-warp handoffs. At each handoff, all load destinations are consumed
before the empty-barrier publication. PTX ISA 9.3 section 9.7.17.6.4.5 specifies
that same-thread true register dependencies are respected, while Racecheck
deliberately does not infer register dataflow. `review` is therefore the correct
fail-closed verdict; `clean` would claim an ordering proof the checker does not
have.

`cudnn_sm100_bsa_forward_blk64` retains two same-warp TMEM `read_write`
reviews. They are distinct iterations of one load site and cover separate
slice overlaps `[0, 64)` and `[128, 192)` before their corresponding stores.
Operation-granular review retirement keeps both witnesses visible.

`cudnn_sm100_bsa_forward_blk128` retains the same two same-warp TMEM
`read_write` reviews in its persistent score path. Each reviewed `tcgen05.ld`
loads a score fragment into registers, the fragment is consumed by the
softmax and probability stores, and a later iteration reuses the overlapping
TMEM slice. The native checker does not infer that true register dependency,
so the two completion-order witnesses remain `review` rather than being
silently promoted to `clean`; NumSim and Synccheck are clean for this case.

`flash_attention4` is numerically and synchronously clean after upstream
commit `40ff2c034809` added the CTA rendezvous immediately before TMEM
deallocation. Operation-granular review retirement later exposed a separate
score-slot lifetime conflict: the softmax warpgroup loads TMEM scores and later
publishes `p_ready_2`, after which MMA may overwrite the slot. Every loaded score
register is consumed by the softmax and P-store chain before that publication,
so PTX true register dependency supplies load completion. Racecheck retains
eight physical read/write reviews because it does not infer that register
dataflow, including the cross-warp overlap on allocation bytes `[0, 128)`. The
teardown rendezvous remains a separate accepted upstream repair.

## Corrected alias attribution

Raw PTX operations constructed from `address_of(TensorLoad)` retain that
source owner. Descriptor-derived shared accesses do not invent a source view,
and raw TCGEN TMEM accesses retain the anchor owner. These rules removed only
synthetic `alias_stale_read` advisories:

- `flash_attention_backward_sm100` changed from a synthetic TMEM-owner review
  to clean with the same physical access trace.
- `gdn_prefill_sm100` lost its synthetic TMEM aliases but retained its physical
  review findings.
- `flashkda_bf16_fused_m128` lost descriptor- and TMEM-invented aliases while
  retaining five shared-pool alias advisories and its two physical TMEM
  reviews. The refreshed `K` trace does not preserve the Python view-variable
  names in TIR, so the advisories use anonymous logical-view labels.

The refreshed FlashMLA prefill kernels retain one shared-pool alias advisory
per full launch (`flash_mla_sparse_fwd`, head64, and head128-small-top-k), while
the audited head128 case retains two. `stable_sort_topk_by_value` retains three
shared-pool alias advisories. None of these reports contains a physical race
finding or incomplete execution.

The owner repairs do not suppress accesses and do not weaken memory ordering.
Focused controls still report a real race when a required TCGEN commit is
removed.

## Resolved persistent-stage defects

The bundled source also backpressures FlashMLA small-top-k tQ
generation reuse. A consumed mbarrier counts one elected arrival from every
WG0 warp in both CTAs (lines 327-371); CTA0 waits for the prior phase before
republishing tQ-full (lines 416-418), and both consumption paths arrive at
lines 455-456 and 503-504. The forced cluster-0 task-steal run now completes
all 32 selected warps with no physical Racecheck finding and the same one
shared-pool alias advisory as the full launch. Its verdict stays `incomplete`
only because that focused test intentionally excludes the other 32 warps; the
normal full launch completes with `review`.

## Resolved canonical-source defects

`recurrent_kda_decode_one_warp` now orders its initial output fill before the
final cross-lane stores with the upstream warp rendezvous. Its full-launch
NumSim, Synccheck, and Racecheck cases are ordinary corpus entries rather than
an excluded error.

The refreshed `gdn_cp_prefill_sm100` source exposed three independent defects.
First, valid WG2 used `setmaxnreg.dec` counts 72/24/24/72
across its four warps even though `.sync.aligned` requires one warpgroup-wide
action; all four now use 72, balancing the two 216-register consumer
warpgroups. Second, the single-stage `state_input` ring republished its full
barrier without acquiring the prior empty generation; it now acquires the
empty barrier before reuse. Third, `_mn_opt_materialize_x` published
`x_ready` after asynchronous `tcgen05.ld` issue but before load completion,
allowing warp 11 to overwrite TMEM columns 448-511. The load now executes
`tcgen05.wait::ld` before the cross-warp handoff, as required by PTX ISA 9.3
sections 9.7.17.6.2.1.2 and 9.7.17.6.4.4. The original Racecheck overlap was
allocation bytes `[1792, 2048)` between the warp-0 load and warp-11 MMA write.
The prefill output path is different: it consumes every Q-state load register
into the shared-memory output before publishing `p_qstate.empty`, so its
cross-warp lifetime finding is a register-dependency review rather than a source
defect.

The shared-expert MegaMoE combine path exposed a separate source defect when
the corpus began executing `num_shared_experts=1`. Its third combine selection
reused TMA load stage 0 after all lanes had read that stage through the generic
proxy, but the elected lane issued the async-proxy overwrite without a
generic-to-async proxy fence. The canonical source now executes
`fence.proxy.async.shared::cta` followed by `warp_sync` after the completed
generic loads and before that stage can be selected again. This follows the
PTX bulk-copy rule that accesses to one location through generic and async
proxies require a cross-proxy fence. The original Racecheck overlap was shared
allocation bytes `[0, 4)` between the lane-0 `ld.shared.v4.u32` and the next
512-byte TMA write; the repaired full launch has no physical finding and
retains one shared-pool alias advisory.

The pinned Top-K corpus also exposed two single-GPU source ordering defects.
`radix_topk_multi_cta` now performs a CTA barrier before thread 0 publishes
each group-arrival counter, so every histogram/state write is visible before
the next phase consumes it. `fast_topk_clusters` now publishes the cluster
arrival only after the current warp has consumed the previous ping-pong bank;
the old placement allowed the next round to overwrite that bank while it was
still being read. The fix was merged upstream in PR #88 at
`72f3d67f3d7b319a769bb9e4cb5ca7f6d1cc046a`; focused A/B NumSim and Racecheck
runs pass. The final pin also includes the FP4 RMSNorm compatibility fixes
merged in PR #89 at `7b0ba162e9f1671c1d2fedff56c239ee1cfdfb8a`.

## Evidence history and current gates

The evidence-first commits are `9bb9b6068f`, `b8924d3c89`, `a5270fe35f`,
`15cab9a921`, `2fd3dd32c9`, `181778bab6`, `64a08257af`, `d39d309bba`, and
`a596eaab47`. Their paired metadata or checker-contract commits are
`ced187e152`, `c3cf1aba62`, `12c1d81ff2`, `1725e7c552`, `6a7650acb0`,
`2c864f3faa`, `8e1b19009a`, `809f179478`, and `8175b81567`.

Current verification is intentionally executable rather than copied into
additional result documents:

```text
cd tirx_harness
python -m pytest -q tests/numsim/corpus/test_canonical_kernels.py
python -m pytest -q tests/analysis_tools/synccheck/corpus/test_tirx_kernels_synccheck.py
python -m pytest -q tests/analysis_tools/racecheck/corpus/test_canonical_kernels_racecheck.py
```
