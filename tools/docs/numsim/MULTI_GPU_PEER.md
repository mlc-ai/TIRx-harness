# Multi-GPU NVLink peer memory (symmetric memory)

NumSim, Synccheck, and Racecheck run single-node multi-GPU kernels that address
another rank's memory directly over NVLink: a rank loads, stores, does atomics
on, and moves bulk/TMA copies to and from a peer's symmetric-memory buffer, as
NVSHMEM and CUDA symmetric-memory (`torch.distributed._symmetric_memory`)
kernels do. This is unicast peer memory, the counterpart of the multicast
(NVLS) support in [MULTI_GPU_MULTIMEM.md](MULTI_GPU_MULTIMEM.md). Both share
one launch model: one CPU process simulates every rank in a single engine, so
cross-rank flag waits execute, Synccheck checks their completion, and Racecheck
checks cross-rank ordering under the PTX memory consistency model.

Paths below are relative to `tools/`; `engine-rs/` is
`src/tirx_harness/numsim/engine-rs/`. PTX section numbers refer to the PTX
ISA's "Memory Consistency Model" chapter.

## Supported operations

The layer checks peer addresses through the ordinary global-memory path for:

- weak, relaxed, acquire, and release loads/stores and vector accesses;
- 32- and 64-bit atomics, reductions, CAS polling, and system-scope waits;
- bulk and Tensor Memory Accelerator copies through peer symmetric memory;
- reachability of raw peer addresses and tensor maps.

The operation details, ordering rules, counterexamples, and related tests are
listed immediately below. Invalid peer access to private memory is an error;
an address alone does not establish visibility or synchronization.

## Ordering and race semantics

### Check contract and memory-ordering rules

The [multimem check contract](MULTI_GPU_MULTIMEM.md#check-contract-and-limits)
also applies here: analysis executes TIRx IR with concrete bindings, and a
verdict covers the selected specialization, inputs, and control-flow path.
The peer address identifies the target bytes; it does not itself synchronize
with the owner. All ranks run the same kernel/grid, with separate rank-local
shared memory, barriers, and grid coordinates.

1. **Reachability:** a rank may access its own private allocations and mapped
   symmetric replicas. A peer address or tensor map into another rank's
   private allocation is a memory error. Sharing a host NumPy allocation
   among ranks leaves it unowned in this model; use `SymmetricBuffer` when
   testing device ownership and peer mapping.
2. **Physical conflicts:** Racecheck checks overlapping bytes, not equality
   of parameter names. Unordered read/write, write/write, and reuse-after-read
   accesses need a causal dependency or applicable strong-access atomicity.
   A successful numerical result cannot substitute for that dependency.
3. **Scope and observation:** release and acquire patterns must both cover
   the communicating threads; across GPUs that requires `.sys`. The acquire
   must observe the publication, directly or through a qualifying RMW chain.
   A stale flag, relaxed poll, or `.gpu` endpoint supplies no cross-rank
   publication. A fence applies only to operations on its proper side.
4. **Lane participation:** a lane-0 signal carries another lane's writes
   only after the required warp/CTA synchronization; consumers on other lanes
   similarly need the dependency after the acquire.
5. **Proxy visibility and completion:** peer unicast is a generic access.
   A bulk/TMA reader needs generic-to-async visibility, such as an appropriate
   `fence.proxy.async.global` on the publication path. Its shared-memory
   consumer must wait for the load's completion. A bulk/TMA writer must finish
   its destination write before publishing it; `wait_group.read` alone is
   insufficient. Crossing a multicast alias additionally follows the
   [alias rules](MULTI_GPU_MULTIMEM.md#memory-ordering-rules).
6. **Liveness:** Synccheck checks whether modeled participants and pending
   completions can satisfy waits. It does not treat a completed flag poll as
   proof of data visibility. A flag that no actor can advance may prove a
   deadlock; exhausted coverage is `incomplete`.

The tables below pair these rules with positive controls and counterexamples.
See the [PTX memory model](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model)
for the specification, and the
[validation status and resume checklist](MULTI_GPU_VALIDATION.md) for existing
measurements and outstanding gates. Mega MoE currently has an
`alias_stale_read` advisory and therefore a `review` verdict, not `clean`.

### Operation details

Every global-memory instruction NumSim models accepts a peer address. The
address resolves to the peer replica's allocation in the launch's shared
global arena, and the access runs through the ordinary engine path:

| Kind | Forms exercised by the tests |
|---|---|
| Loads and stores | `ld`/`st` weak, `.relaxed`, `.acquire`/`.release` at `.gpu`/`.sys`, `.v4` vectors |
| Atomics | `atom.add`, `atom.cas` (`.relaxed`/`.acquire`, `.gpu`/`.sys`), `red.add` (`.relaxed`/`.release`), 32- and 64-bit |
| Bulk copies | `cp.async.bulk` global to shared (`mbarrier::complete_tx`) and shared to global (`bulk_group`) |
| TMA | `cp.async.bulk.tensor` loads and stores through a `TensorMap` over a peer's replica |
| Polls | `T.cuda.wait_until(scope="sys")` and `atom.cas` spins |

## Related tests

`tests/numsim/support/peer_litmus.py` runs one warp per rank and takes scalar
modes, so one transpile serves a rule's passing form and every counterexample:

| Kernel | Pattern |
|---|---|
| `push` | MP through peer stores. Every lane stores a 16-byte vector into rank `r + 1`'s `inbox`; lane 0 arrives on that rank's `flag` (`ARRIVALS`) and waits on its own (`WAITS`). |
| `pull` | MP through peer loads. Every rank publishes its `outbox` and releases its own `flag` (`PUBLISHES`); lane 0 polls rank `r + 1`'s flag (`POLLS`) and every lane loads that rank's outbox. `reuse` then overwrites the outbox, after the puller's acknowledgement (`ACKS`) or not. |
| `atomics` | Every rank updates rank 0's counters with no synchronization, so only moral strength separates a race from none. |
| `bulk` | 512-byte vectors through the async proxy: pull a peer's outbox into shared memory, or push shared memory into a peer's inbox, with a bulk copy or TMA. |

A Racecheck counterexample asserts the exact set of
(writer rank, reader rank, 16-byte vector) pairs that the rule leaves
unordered, not merely an `error` verdict.

| PTX rule | Passing forms | Counterexamples | Related tests |
|---|---|---|---|
| Release/acquire patterns (8.8, 8.9) | `red.release.sys`; `fence.acq_rel.sys` + relaxed RMW; `st.release.sys`; acquire CAS; relaxed CAS + `fence.acq_rel.sys`; `wait_until(scope="sys")` | `.gpu` on either side (`scope_mismatch`); relaxed arrive or wait; a weak flag store (it races the acquire); a peer store after the release; a read before the acquire fence; lanes off the releasing lane without `warp_sync` | [Racecheck](../../tests/analysis_tools/racecheck/test_native_peer_ordering_rules.py): `test_sys_release_acquire_orders_peer_stores`; `test_scope_short_of_the_peer_rank_orders_no_peer_store`; `test_no_release_acquire_pattern_orders_no_peer_store` |
| Peer loads (`pull`) | `.sys` release and acquire | `.gpu` publication; relaxed publication | [Racecheck](../../tests/analysis_tools/racecheck/test_native_peer_ordering_rules.py): `test_peer_loads_after_a_sys_acquire_are_ordered`; `test_gpu_scoped_publication_orders_no_peer_load`; `test_relaxed_publication_orders_no_peer_load` |
| Reuse after a read (WAR) | acknowledgement released and acquired at `.sys` | relaxed acknowledgement; none | [Racecheck](../../tests/analysis_tools/racecheck/test_native_peer_ordering_rules.py): `test_overwriting_a_buffer_a_peer_still_reads_is_a_war_race` |
| Moral strength (8.7) | `atom.add.sys`, `red.add.sys`, CAS | `atom.add.gpu` across ranks; weak stores from every rank; a weak read of an atomically updated word | [Racecheck](../../tests/analysis_tools/racecheck/test_native_peer_ordering_rules.py): `test_morally_strong_peer_updates_do_not_race`; `test_gpu_scoped_peer_atomics_are_not_morally_strong_across_ranks`; `test_weak_read_of_a_word_peers_update_atomically_races` |
| Async proxy (8.6) | bulk/TMA load after the acquire and `fence.proxy.async.global`; a completed bulk/TMA store published by the release | the load without the proxy fence; the store released before it completes | [Racecheck](../../tests/analysis_tools/racecheck/test_native_peer_ordering_rules.py): `test_async_proxy_load_of_a_peer_buffer_without_a_global_proxy_fence_races`; `test_incomplete_async_proxy_store_to_a_peer_is_not_published` |
| Reachability | symmetric replicas; a rank's own private buffer, atomics included | a peer address into private memory; a `TensorMap` over a peer's private array | [Racecheck](../../tests/analysis_tools/racecheck/test_native_peer_ordering_rules.py): `test_peer_access_to_private_memory_is_a_memory_error`; `test_tensor_map_over_a_peers_private_memory_is_refused_at_launch` |

| File | What it covers |
|---|---|
| `tests/numsim/runtime/test_peer.py` | NumSim values: peer stores, loads, atomics, CAS, bulk and TMA in both directions; the reachability errors; `SymmetricBuffer` validation and `peer_offsets`; both ported kernels against their references; the all-gather barrier |
| `tests/analysis_tools/racecheck/test_native_peer_ordering_rules.py` | The rule table above |
| `tests/analysis_tools/synccheck/test_native_peer_ordering_rules.py` | Every counterexample still completes (Synccheck checks liveness, not ordering); a rank that never arrives deadlocks its receiver; the reachability errors |
| `tests/analysis_tools/racecheck/test_native_peer_kernels.py` | Both ported kernels, world 2 and 4: the all-gather GEMM is clean with both epilogues, and stale flags let its TMA loads run ahead of the copies; its barrier is clean; mega MoE is clean except for one advisory |
| `tests/analysis_tools/synccheck/test_native_peer_kernels.py` | Both ported kernels complete; a barrier with a wrong peer offset deadlocks the skipped rank |

The mega MoE advisory is `alias_stale_read`: combine reuses the shared-pool
prefix that dispatch named `smem_expert_count`. The single-GPU corpus case
reports the same advisory. Physical ordering, within a rank and across the
symmetric buffers, has no finding. The advisory is about names, not values:
- **What it flags:** the first epilogue warp's 16-byte `ld.shared` of its
  combine chunk (`kernel.py`, `lds128`) reads the pool's first 32 bytes, which
  warp 0 last wrote, by name, as `smem_expert_count` (8 experts x 4 bytes at
  this size).
- **Why it is not stale:** the chunk's bytes come from the warp's own
  `cp.async.bulk` load, which the read follows through the load's mbarrier.
  The engine records raw bulk copies without a logical buffer name, and the
  alias tracker skips unnamed writes, so the name it compares against is
  still dispatch's. Physical race checking does see the bulk write, and
  NumSim's output equals the reference.

## Using it

A multi-rank launch passes a list of per-rank input dicts (see the multimem
doc's "Using it"). A symmetric-memory allocation is bound with
`numsim.SymmetricBuffer`, built from one NumPy array per rank and passed to the
same parameter on every rank. Rank `r`'s binding is `replicas[r]`; its kernel
reaches rank `p`'s replica at its own address plus `peer_offsets(r)[p]`, the
byte distance a GPU launcher reads off the symmetric-memory handle
(`buffer_ptrs[p] - buffer_ptrs[r]`):

```python
import numpy as np
from tirx_harness import numsim, racecheck, synccheck

world = 4
inbox = numsim.SymmetricBuffer([np.zeros(128, np.uint32) for _ in range(world)])
flag = numsim.SymmetricBuffer([np.zeros(4, np.uint32) for _ in range(world)])
inputs = [
    {"rank": np.int32(r), "world": np.int32(world),
     "inbox_offsets": inbox.peer_offsets(r), "flag_offsets": flag.peer_offsets(r),
     "inbox": inbox, "flag": flag, "src": src[r], "out": np.zeros(128, np.uint32)}
    for r in range(world)
]

result = numsim.Engine().run(numsim.transpile(push), inputs, outputs=["out"])
rank2_out = result.outputs[numsim.rank_binding_name("out", 2)]  # key "out@rank2"

racecheck(push, inputs).print()
synccheck(push, inputs).print()
```

`push` stores into rank `rank + 1`'s `inbox` at
`inbox + inbox_offsets[rank + 1]`, arrives on that rank's `flag` with
`red.release.sys`, waits on its own flag, and reads its own inbox.
`tests/numsim/support/peer_litmus.py` has the complete kernels. Their `_heap`
carves several buffers from one heap per rank, so a single offset per peer
reaches all of them, as on an NVSHMEM symmetric heap.

The binding contract:

- **Replicas** are one C-contiguous NumPy array per rank, with one dtype and
  shape ("SymmetricBuffer requires one NumPy array per rank", "must share one
  dtype and shape", "must be C-contiguous"). A buffer with the wrong replica
  count fails with "has N replicas for M ranks". Two ranks' replicas may not
  share storage ("shares its storage with").
- **Only symmetric memory is peer-mapped.** Any other array bound by exactly
  one rank is that rank's private device memory. A peer address into it fails,
  because on a GPU it is unmapped, with:
  `rank R addresses rank O's buffer 'name', which is not symmetric memory:
  another GPU maps only symmetric-memory allocations (bind it with
  numsim.SymmetricBuffer)`. NumSim raises `NumSimExecutionError`; both
  checkers return an `error` verdict.
- **Tensor maps** are checked when they are relocated at launch: a rank's
  `TensorMap` over a peer's private array fails with the same message,
  prefixed `TensorMap 'name': `.
- **Multicast replicas** (`MulticastWindow`) are symmetric memory too: a rank
  may address another rank's replica through unicast peer addresses.
- An array bound by several ranks, or by none (a host array the harness
  allocates), is not owned by one rank and stays reachable from every rank.

## How it is implemented

### Bindings

`prepare_rank_bindings` (`src/tirx_harness/numsim/bindings.py`) flattens the
per-rank dicts as for multimem. It also computes `rank_mappings`, one entry
per allocation:
- **Symmetric memory** (a `SymmetricBuffer` replica or a multicast replica):
  `(owner rank, True, binding name)`.
- **A window placeholder, or an allocation bound by zero or by two or more
  ranks:** `None`.
- **Otherwise:** `(owner rank, False, binding name)`, the owner's private
  memory.

The payload field `rank_mappings` carries them to the engine.
`api.py` and `checker_runner.py` treat a `SymmetricBuffer` as a buffer or
pointer binding.

### Engine

`bind_rank_mappings` (`runtime/python.rs`) attaches each mapping to its
allocation as a `RankMapping { rank, symmetric, name }` (`memory.rs`).
`BufferView::check_reachable_from(rank)` is the whole reachability rule. It
rejects an access from a rank other than the owner to a non-symmetric
allocation, and passes everything else. It runs in two places:
- **Raw global addresses:** `operand.rs`, where a 64-bit address operand
  resolves to the allocation that owns it (`observed_address_owner`). Every
  load, store, atomic, and bulk copy goes through this.
- **Tensor maps:** `rewrite_tensor_map_addresses` (`runtime/python.rs`), when
  a `TensorMap`'s base is relocated into the arena at launch.

Everything else is shared with the multimem work:
- **Memory and scope:** all ranks share one global arena, and
  `MemoryScope::required_between_warps` requires `.sys` between warps on
  different ranks.
- **Racecheck:** the global model treats a peer access as an ordinary
  cross-rank access. A scope covers a pair only if it reaches both actors, so
  `.gpu` and narrower releases, acquires, and atomics order nothing across
  ranks. Cross-rank findings report `actor_relation: "cross_rank"`. Bulk and
  TMA copies are async-proxy accesses: a peer's generic writes reach them only
  through `fence.proxy.async.global` after the acquire, and their own writes
  are published only after their completion (`cp.async.bulk.wait_group` or
  the mbarrier) and a release.
- **Synccheck:** needs no peer-specific code. A cross-rank flag spin is an
  ordinary polling actor, and a flag no rank ever sets is a whole-launch
  deadlock.

### Polling policy

Racecheck adjudicates a plain `ld` spin by happens-before: its failed polls
race the arrival on the schedules where they run first. Kernels therefore poll
in one of two ways:
- **A strong RMW poll** (`atom.cas`), which is morally strong with the arrival
  (8.7). The litmus kernels use this, as CUTLASS's and FlashInfer's barriers
  do.
- **`T.cuda.wait_until`**, which declares the spin. Both ports use this.

### ABI

`NUMSIM_ABI_VERSION` is 40 (`rank_mappings`), in both `numsim/abi.py` and
`engine-rs/src/lib.rs`.

## Ported kernels

Both ports stay in the harness for now (`ported/`), to be copied to
[mlc-ai/tirx-kernels](https://github.com/mlc-ai/tirx-kernels) later.

| Port | Origin | NumSim-sized version |
|---|---|---|
| `ported/cutlass/sm100_all_gather_gemm.py` | CUTLASS CuTeDSL `distributed_all_gather_gemm_blackwell.py`, measured against it and FlashInfer `comm/all_gather_matmul` (`cake`, and `auto`, which is cuTile on SM100) | `tests/numsim/support/peer_all_gather_gemm.py`: world 2 and 4, 256x128x128 bf16, 2 chunks per shard, in-kernel copy role |
| `ported/deepgemm/sm100_fp8_fp4_mega_moe.py` (implementation in `_sm100_fp8_fp4_mega_moe/`) | DeepGEMM `fp8_fp4_mega_moe` at `559d79fb` | `tests/numsim/support/peer_mega_moe.py`: world 4, 2 experts per rank, top-2, one shared expert, hidden 256, 4 SMs |

**All-gather GEMM.** `out = all_gather(a) @ b.T` on every rank. It is the
persistent, warp-specialized GEMM of the CUTLASS example (epilogue warps 0-3,
MMA warp 4, TMA warp 5), with the example's configuration space and derived
parameters. CUTLASS launches one GEMM per shard, each remote one gated on a
flag that its copy stream releases. The port covers every shard in one
persistent launch, as FlashInfer's cake does:
- The scheduler walks this rank's shard first, then rank `rank + j`'s at step
  `j`, each in `chunk_rows`-row chunks.
- Before loading a remote chunk, the TMA warp acquires the chunk's flag
  (`wait_until(scope="sys")`) and executes `fence.proxy.async.global`.
- On the GPU, a host schedule fills `scratch`: copy-engine copies, pulled or
  pushed, each followed by a stream write of the chunk's flag.
  `barrier_kernel` clears the flags and aligns the ranks before every launch.
- NumSim launches one kernel per rank and has no copy engine. Its runs build
  the `copy_warps` variant, whose extra warps push the same chunks and
  release the same flags. Everything else is the benchmarked code.

**Mega MoE.** DeepGEMM's fused expert-parallel MoE layer in one persistent
launch per rank:
- dispatch: count, then pull the routed tokens from peer ranks over NVLink;
- L1: FP8 x FP4 GEMM, with SwiGLU and FP8 quantization;
- L2: FP4 GEMM;
- combine: write results into the source rank's buffer, then reduce top-k.

The port reproduces DeepGEMM's block-configuration heuristic, stage count,
register split, warp roles, task scheduler (shared L1, then routed tasks,
then shared L2), and grid-sync and NVLink-barrier structure. Its outputs and
cumulative expert statistics match DeepGEMM bit for bit. The kernel reaches
peers through `symm_rank_offset_<p>` scalars over one `SymmetricBuffer`.
`TIRX_DEEPGEMM_NUM_SMS_OVERRIDE` shrinks the grid for NumSim.

### Where the ports differ from the origins

Mega MoE:
- **NVLink barrier:** DeepGEMM polls with `ld.acquire.sys` and traps after a
  timeout. The port waits without a timeout, because a wait that can time out
  is a loop whose exit condition the checker cannot see.
- **Polls:** DeepGEMM polls the expert receive counts and the L1 task counter
  with `ld.volatile`. The port's scheduler uses `wait_until`, for the polling
  policy above. Its dispatch warps read the receive counts once, with
  `ld.relaxed.sys`, and trap if a count is incomplete: they read after the
  dispatch NVLink barrier, which every rank signals only after adding its
  counts. A `wait_until` closes with an `ld.acquire.sys`, which costs about
  1 us. With two experts per lane, waiting there delayed the pull by 2.3 us.
- **`fence.proxy.async.global` in load-A, after acquiring the L1/L2 readiness
  counters.** TMA reads through the async proxy what dispatch, or the
  previous epilogue, wrote through the generic proxy. DeepGEMM has no fence
  there.
- **Combine entry fences:** combine reuses the shared pool that MMA read and
  dispatch wrote, across the async/generic proxy boundary.
- **One extra workspace grid barrier** (dispatch and load-A) before dispatch
  cleans the workspace. Dispatch and load-A read reusable global workspace
  after the preceding grid rendezvous, and the epilogue's rendezvous cannot
  publish those reads, because it precedes the CTA-local join. The barrier
  is split: each SM arrives once its dispatch pull and its load-A task loop
  are done, and only the clean waits. By then every SM has arrived, so the
  wait is one load. As a full grid sync after the epilogue join, it added
  1.4 us before the clean at 2 tokens per rank. Load-A joins the arrive's
  CTA barrier from its own call site, so that one barrier is the unaligned
  `barrier.sync`: `.aligned` requires every participant to execute the same
  instruction. Every other grid-sync CTA barrier is the aligned `bar.sync`,
  as in DeepGEMM (see [Grid-sync barriers](#grid-sync-barriers-are-aligned)).
- **TMEM release:** DeepGEMM frees TMEM before the NVLink barrier that
  precedes combine. The port frees it right after that barrier. The
  `cta_group::2` free needs both CTAs of the pair to be done with TMEM, and
  the barrier's grid sync, which every epilogue thread of both CTAs joins,
  is what orders the peer CTA's last TMEM read before the free.
- **Combine work split:** at hidden 7168, a token's bf16 row is reduced in
  two chunks. A DeepGEMM epilogue warp takes a token and reduces its chunks
  back to back. A port warp takes (token, chunk) items instead, so a token's
  two chunks reduce on different warps. The slots of each chunk are summed
  in the same order, so the output is unchanged. With DeepGEMM's loop, the
  first two TMA loads of every chunk took 1.3-2.2 us each in the port, but
  only those of the first chunk did in DeepGEMM. Probes ruled out L2
  residency, cache hints, barrier and fence placement, and the TMEM release,
  but did not find the cause. With the aligned grid-sync barriers below in
  place, DeepGEMM's loop still makes the port 2.0% slower than DeepGEMM at
  1 token per rank (86.8 against 85.1 us), and the split makes it 1.7%
  faster (83.7 us).
- **SM count:** the build takes the device's SM count
  (`TIRX_PREPARE_NUM_SMS`; 152 on this GB200) instead of `tirx_kernels`'
  default of 148, so both kernels launch the same grid.

All-gather GEMM:
- **Launch structure:** one persistent launch over every shard (above),
  instead of CUTLASS's one launch per shard.
- **Compiler:** the port's two kernels build with nvcc 13.2, not TVM's
  default compiler, NVRTC 13.0 (see
  [NVRTC 13.0](#nvrtc-130-leaves-a-descriptor-half-unwritten)). The mega MoE
  port already builds with nvcc.
- **Origin wrappers** (`benchmarks/peer/origins.py`) hoist each origin's
  per-call allocation and rendezvous out of the timed launch:
  - CUTLASS: every compiled library is loaded before the first launch. A
    library load waits for the device to idle, so loading a later step's GEMM
    while a gated GEMM spins would deadlock; the example avoids this by
    capturing its graph first.
  - cake: the prologue clears the readiness pad, whose launch epochs a CUDA
    graph bakes in.
  - cuTile: timed eagerly, because its kernels take lists of tensors, which a
    CUDA graph cannot capture. The matmul's flag poll is replaced by an
    atomic read, because FlashInfer's poll compiles to no wait at all (see
    [FlashInfer's cuTile matmul](#flashinfers-cutile-matmul-does-not-wait-for-its-flags)).
- **Symmetric memory:** every implementation uses the NVSHMEM backend of
  `torch.distributed._symmetric_memory`.

### Grid-sync barriers are aligned

A workspace grid sync is a CTA barrier, thread 0's atomic on a global
counter and its poll, then a second CTA barrier. The port first used the
unaligned `barrier.sync` for both CTA barriers, where DeepGEMM uses the
aligned `bar.sync`. With the unaligned form, the warp of thread 0 does not
reconverge after the poll: nvcc emits `WARPSYNC.ALL` after `bar.sync` but
not after `barrier.sync`. The NVLink barrier that follows then runs its
counter load, system-scope fence and peer signal once for lane 0 and once
for lanes 1-31, back to back. That cost 1.3-2.2 us on every rank with work,
and the port was 5.6% slower than DeepGEMM at 2 tokens per rank (34.5
against 32.7 us). With aligned barriers it takes 32.2 us.

Synccheck and Racecheck enforce the `.aligned` contract: a barrier that
mixes aligned and unaligned arrivals is an `aligned_sync_contract_mismatch`.
That check rejected the first version of the fix, which also made the
dispatch/load-A barrier aligned. That barrier stays unaligned (see the
extra workspace grid barrier above).

### NVRTC 13.0 leaves a descriptor half unwritten

On sm_100, a global load or store addresses memory through a 64-bit memory
descriptor in a uniform register pair (`desc[URn]`), which the kernel loads
from the constant bank (`c[0x0][0x358]`). NVRTC 13.0.88 (pip
`nvidia-cuda-nvrtc`) compiled the all-gather TMA warp's flag poll to
`LDG.E.STRONG.SYS R0, desc[UR10][R2.64]` with UR10 set to an unrelated
constant and UR11 never written, so the poll used whatever an earlier kernel
had left in UR11:
- **Where:** 5 of the 1679 configurations of the first tune. All five are
  `cutlass_default` with 64x64 tiles, a 1x1 cluster and the TMA-store
  epilogue. One was among TIRx's three compared configurations.
- **Symptom:** CUTLASS's kernels leave 0xffffffff in UR11. The poll faulted
  with Xid 13 (illegal instruction parameter) in 3 of about 40 compare runs
  that mixed CUTLASS and TIRx, and in none of 12 CUTLASS-only and 12
  TIRx-only runs. A GPU
  core dump put the fault at the poll, with UR10 = 0x40 and
  UR11 = 0xffffffff.
- **Fix:** nvcc 13.2 compiles the same CUDA source with the descriptor loaded
  before the poll, and the benchmark builds the port with it. With it, 20 of
  20 mixed runs passed.

`benchmarks/peer/desc_scan.py` flags every access through a pair with a half
that no earlier instruction writes. It flags the five NVRTC builds above and
nothing in:
- the nvcc builds of every configuration the all-gather tune timed, on every
  rank (1682 configurations, 6728 builds);
- the NVRTC builds of the multimem GEMM + all-reduce (114 configurations, 456
  builds) and two-shot all-reduce (18 builds), at every committed
  configuration and rank;
- the 28 mega MoE libraries, which build with nvcc.

### FlashInfer's cuTile matmul does not wait for its flags

FlashInfer's `auto` backend on SM100 (`all_gather_matmul_cutile.py`, at
`776939f8`) pushes each chunk of a rank's shard into every peer's scratch with
the copy engine, then writes the chunk's flag. Before reading a peer's
chunk, the matmul polls that flag:

```python
signal = ct.load(signal_pad, index=signal_index, shape=(), padding_mode=zero_pad)
while signal == 0:
    signal = ct.load(signal_pad, index=signal_index, shape=(), padding_mode=zero_pad)
```

- **The poll compiles to nothing.** `ct.load` defaults to a weak load, and a
  loop of weak loads has no side effect, so the compiler may delete it. The
  compiled matmul has no load of the 32-bit flag: its only global loads are
  the two 64-bit loads of the input-list pointers, plus the TMA loads. The
  matmul reads a peer's chunk whenever it gets to it.
- **Why it usually works:** each rank multiplies its own shard first, which
  gives the copies a head start. In a warm call, delaying one sender's copy
  stream by about 0.1 ms or 1 ms (`torch.cuda._sleep` before the
  broadcast) made every other rank read that sender's whole shard as the
  NaN it had filled scratch with. That happened in 3 of 3 calls at each
  delay.
- **Where the compare saw it:** the first call in a process compiles the
  barrier kernel, per rank, in 240 to 430 ms. The rank that compiles last
  sends after some ranks have already enqueued their matmul. In both
  failures with host timestamps, exactly the ranks that had enqueued their
  matmul by then read its shard early. In the compare's worker
  processes, the correctness check failed on 5 of 6 such first calls of the
  wrapper, and FlashInfer's own `all_gather_matmul_cutile`, called first,
  failed 6 of 6. A later call in the same process passed, because by then
  the ranks run in lockstep. In a bare process that had not imported
  `cuda.core` (the CUTLASS examples import it), 5 of 5 first calls passed;
  the delayed-sender runs above imported nothing extra, so the race does not
  depend on it.
- **An acquire load is not enough.** With
  `memory_order=ACQUIRE, memory_scope=SYS`, the poll becomes a real loop
  (`LDG.E.STRONG.SYS`, then `CCTL.IVALL`), but each of the four warps that
  run it loads and tests the flag itself, and every iteration starts with a
  128-thread named barrier (`BAR.SYNC 0x1, 0x80`). A warp that sees the
  flag leaves the loop while one that read it a moment earlier goes back to
  the barrier, which then never fills. 9 of 11 runs hung. In one, the hung
  ranks' pads were read from a side stream: every peer's flag was set, and
  the matmul still spun.
- **Fix:** the benchmark polls with an atomic read,
  `ct.atomic_add(signal_pad, signal_index, 0, memory_order=ACQUIRE, memory_scope=SYS)`.
  cuTile issues an atomic from one lane (`ATOMG.E.ADD.STRONG.SYS`) and
  broadcasts its result through a shuffle and shared memory, as it does
  for the CAS in FlashInfer's own barrier kernel, so all warps leave
  together. `origins._cutile_atomic_poll` loads FlashInfer's module with
  exactly that change. With it, the delayed-sender runs (two runs of four
  calls), four first calls in the compare's worker setup, and 18 checks of
  the benchmark's wrapper on NaN-filled scratch all passed, and none hung.

## Benchmark method

All runs: 4x GB200 (sm_100a, 152 SMs), world size 4, torch 2.14.1+cu130,
nvidia-cutlass-dsl 4.7.0.

**Mega MoE** (`benchmarks/peer/mega_moe.py`):
- **Timer:** DeepGEMM's own `deep_gemm.testing.bench_kineto` (DeepGEMM
  `559d79fb`, the commit the port pins) times both kernels from a Kineto
  trace, 30 launches each. Before every launch it flushes L2 (an 8 GB
  write), sleeps the GPU for 2e7 cycles, and aligns the ranks with a stream
  all-reduce, all outside the kernel's time.
- **Rounds:** 11 rounds. Within a round the two kernels run back to back,
  in alternating order from round to round. Between them, 8 GB are zeroed
  and the ranks are realigned, and rounds are 1 s apart. A round's time is
  the slowest rank's, and the reported time is the median over rounds.
- **Correctness:** checked on every benchmarked shape before timing. The
  output and the cumulative statistics must equal DeepGEMM's exactly.
- **Shapes:** DeepGEMM's test default (8192 tokens, hidden 7168,
  intermediate 3072, 384 experts, top-6, one shared expert), the port's own
  matrix at world 4, and the DeepSeek-V3 sweep (hidden 7168, intermediate
  2048, 256 experts, top-8, one shared expert) at 1 to 8192 tokens per rank.

```bash
export DEEPGEMM_DIR=~/src/DeepGEMM-559d79f                   # DeepGEMM checkout at 559d79fb
python -m benchmarks.peer.mega_moe prepare                   # build both libraries per shape
python -m benchmarks.peer.mega_moe compare                   # appends results/mega_moe_gb200x4.jsonl
python -m benchmarks.peer.mega_moe compare --out benchmarks/peer/results/mega_moe_gb200x4_repeat.jsonl
```

**All-gather GEMM** (`benchmarks/peer/all_gather_gemm.py`):
- **Tune:** checks and times each implementation's configurations on every
  shape, appending one line per candidate to the tuning log:
  - CUTLASS: its kernel's MMA tile, cluster, 1- or 2-CTA, and epilogue store.
  - cake and cuTile: their own heuristics.
  - TIRx: CUTLASS's space plus raster order, swizzle, chunk size, and copy
    direction.
- **Compare:** times TIRx's 3 and each origin's 8 fastest correct
  configurations from the tune (cake and cuTile have one each),
  interleaved on the same buffers, and reports TIRx's best against the
  fastest origin. The origins get more
  because the tune's timer has no idle period (below), so an origin's
  fastest configuration under the compare's timer could rank lower in the
  tune.
- **Timer:** each configuration's launches, cycling through 2 to 10
  workspaces of operands (as many as fit in 1 GiB) so that consecutive
  launches do not reuse cached inputs, are captured in one CUDA graph of 40
  launches. cuTile runs eagerly instead. A
  sample replays the graph for about 30 ms. Before every sample the device
  idles for 0.5 s, then the ranks meet at a host barrier. The 15 samples of
  every configuration are interleaved round-robin. A configuration's time
  is the median sample per launch on the slowest rank. The tune uses the
  same timer with 20 launches, 3 samples, 10 ms and no idle period.
- **Why the idle period:** the large-N GEMMs reach the GPU's 1200 W power
  cap within a sample. Without the idle period, 12% of busy 20 ms clock
  samples during a `llama70b_gate_up_m512` compare were power-capped
  (event reason 0x4), mostly at 1290 to 1550 MHz instead of 2062. A sample's clock
  then depends on what ran before it: TIRx's samples, which always follow
  cuTile's slower eager ones, ran 13% faster with cuTile in the compare than
  without it (275 against 318 us), and CUTLASS's 2% faster. With the idle
  period, every configuration of three shapes timed within 0.6% with and
  without cuTile, and 0.9% of busy clock samples were power-capped.
- **Builds and garbage collection:** the TIRx kernels build with nvcc (see
  [NVRTC 13.0](#nvrtc-130-leaves-a-descriptor-half-unwritten)). The worker
  processes disable Python's cyclic garbage collector and collect only while
  the device is idle. A collected CuTeDSL module unloads its library, and the
  unload waits for the device to idle. If that happens inside a CUTLASS
  launch, while a gated GEMM spins on a flag whose release the host has not
  enqueued yet, it deadlocks. During a graph capture, it invalidates the
  capture. In the first tune, eight CUTLASS candidates timed out this way.
  They were re-run with the fix.
- **Correctness:** before timing, every compared configuration runs once on
  a NaN-filled output, which must equal `all_gather(a) @ b.T` bit for bit on
  every rank. Operands are small integers, so every product and the f32
  accumulation are exact.
- **Shapes:** the CUTLASS example's default and docstring runs, FlashInfer's
  benchmark rows (K 8192, N 2048, 1024 to 65536 rows per rank), its
  correctness run (19456) and cake's end-to-end test (384). Plus a
  tensor-parallel sweep: Llama-3-70B QKV (N 2560) and gate/up (N 14336)
  projections at 512, 2048, and 8192 tokens per rank, bf16.

```bash
python -m benchmarks.peer.all_gather_gemm tune        # resumable; results/all_gather_gemm_tuning.jsonl
python -m benchmarks.peer.all_gather_gemm compare     # appends results/all_gather_gemm_gb200x4.jsonl
python -m benchmarks.peer.all_gather_gemm compare --out benchmarks/peer/results/all_gather_gemm_gb200x4_repeat.jsonl
```

## Results

The parity bar is TIRx latency <= origin latency / 0.99 on every shape: the
DeepGEMM kernel for mega MoE, and the faster origin for the all-gather GEMM.
A ratio is TIRx's latency over the origin's, so the bar is a ratio of at most
1.0101.

### Mega MoE

Every shape meets the bar in both runs; the worst ratio is 1.004. On every
shape, the output and the cumulative expert statistics equal DeepGEMM's bit
for bit. Both runs are of the same source (digest `31284daeef41759a`):
`results/mega_moe_gb200x4.jsonl` and the repeat,
`results/mega_moe_gb200x4_repeat.jsonl`. The times are run 1's; the repeat's
are in its file.

| Shape | Tokens per rank | Hidden, I | Experts, top-k, shared | DeepGEMM (us) | TIRx (us) | Ratio | Repeat ratio |
|---|---:|---|---|---:|---:|---:|---:|
| `deepgemm_default` | 8192 | 7168, 3072 | 384, 6, 1 | 2834 | 2828 | 0.998 | 0.999 |
| `port_t2_h1024_e8_k1` | 2 | 1024, 512 | 8, 1, 0 | 32.7 | 32.2 | 0.986 | 0.988 |
| `port_t64_e384_k6` | 64 | 7168, 3072 | 384, 6, 0 | 521.0 | 516.8 | 0.992 | 0.992 |
| `port_t256_e384_k6` | 256 | 7168, 3072 | 384, 6, 0 | 539.8 | 536.1 | 0.993 | 0.993 |
| `port_t1024_e384_k6` | 1024 | 7168, 3072 | 384, 6, 0 | 632.3 | 629.4 | 0.995 | 0.995 |
| `port_t8192_e384_k6` | 8192 | 7168, 3072 | 384, 6, 0 | 2503 | 2500 | 0.999 | 0.999 |
| `dsv3_t1` | 1 | 7168, 2048 | 256, 8, 1 | 85.2 | 83.6 | 0.982 | 0.984 |
| `dsv3_t16` | 16 | 7168, 2048 | 256, 8, 1 | 255.4 | 253.0 | 0.991 | 0.991 |
| `dsv3_t64` | 64 | 7168, 2048 | 256, 8, 1 | 274.9 | 270.9 | 0.985 | 0.985 |
| `dsv3_t256` | 256 | 7168, 2048 | 256, 8, 1 | 291.4 | 287.6 | 0.987 | 0.985 |
| `dsv3_t1024` | 1024 | 7168, 2048 | 256, 8, 1 | 461.3 | 459.0 | 0.995 | 0.997 |
| `dsv3_t2048` | 2048 | 7168, 2048 | 256, 8, 1 | 779.8 | 780.8 | 1.001 | 1.003 |
| `dsv3_t4096` | 4096 | 7168, 2048 | 256, 8, 1 | 1467 | 1464 | 0.998 | 1.002 |
| `dsv3_t8192` | 8192 | 7168, 2048 | 256, 8, 1 | 2653 | 2662 | 1.003 | 1.004 |

### All-gather GEMM

Every shape meets the bar in both runs. The worst ratio is 1.007
(`llama70b_gate_up_m2048` in run 1; 0.986 in the repeat). In each run, all
219 compared configurations (CUTLASS 136, TIRx 51, cake 16, cuTile 16) left
the exact result on every rank. The files are
`results/all_gather_gemm_gb200x4.jsonl` and the repeat,
`results/all_gather_gemm_gb200x4_repeat.jsonl`, from the tuning log
`results/all_gather_gemm_tuning.jsonl`. Each column is that implementation's
best compared configuration, and the faster origin is in bold. The times are
run 1's.

| Shape | M per rank, N, K | Types | CUTLASS (us) | cake (us) | cuTile (us) | TIRx (us) | Ratio | Repeat ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `cutlass_default` | 64, 256, 512 | tf32->f32 | **44.8** | - | - | 34.8 | 0.777 | 0.775 |
| `cutlass_doc` | 2048, 8192, 8192 | f16->f16 | **584.0** | 752.7 | 1036 | 555.1 | 0.951 | 0.950 |
| `flashinfer_m1024` | 1024, 2048, 8192 | bf16->bf16 | **136.2** | 149.4 | 287.9 | 127.9 | 0.939 | 0.938 |
| `flashinfer_m2048` | 2048, 2048, 8192 | bf16->bf16 | **222.2** | 274.5 | 327.9 | 213.3 | 0.960 | 0.960 |
| `flashinfer_m4096` | 4096, 2048, 8192 | bf16->bf16 | **400.0** | 489.9 | 577.3 | 374.8 | 0.937 | 0.940 |
| `flashinfer_m8192` | 8192, 2048, 8192 | bf16->bf16 | **757.2** | 868.5 | 1101 | 684.0 | 0.903 | 0.901 |
| `flashinfer_m16384` | 16384, 2048, 8192 | bf16->bf16 | 1487 | **1472** | 1900 | 1282 | 0.871 | 0.868 |
| `flashinfer_m19456` | 19456, 2048, 8192 | bf16->bf16 | 1731 | **1714** | 2198 | 1505 | 0.878 | 0.882 |
| `flashinfer_m32768` | 32768, 2048, 8192 | bf16->bf16 | **2959** | 3170 | 3742 | 2577 | 0.871 | 0.874 |
| `flashinfer_m65536` | 65536, 2048, 8192 | bf16->bf16 | **5946** | 6233 | 7426 | 5358 | 0.901 | 0.894 |
| `cake_e2e_m384` | 384, 2048, 8192 | bf16->bf16 | **83.3** | 103.5 | 242.9 | 75.5 | 0.907 | 0.902 |
| `llama70b_qkv_m512` | 512, 2560, 8192 | bf16->bf16 | **103.0** | 170.1 | 263.3 | 91.9 | 0.892 | 0.892 |
| `llama70b_qkv_m2048` | 2048, 2560, 8192 | bf16->bf16 | **236.6** | 393.7 | 569.6 | 233.7 | 0.987 | 0.987 |
| `llama70b_qkv_m8192` | 8192, 2560, 8192 | bf16->bf16 | **873.0** | 1225 | 2031 | 740.8 | 0.849 | 0.847 |
| `llama70b_gate_up_m512` | 512, 14336, 8192 | bf16->bf16 | **290.8** | 408.8 | 545.9 | 278.3 | 0.957 | 0.957 |
| `llama70b_gate_up_m2048` | 2048, 14336, 8192 | bf16->bf16 | **981.2** | 1157 | 1514 | 987.8 | 1.007 | 0.986 |
| `llama70b_gate_up_m8192` | 8192, 14336, 8192 | bf16->bf16 | **4790** | 5124 | 7003 | 4473 | 0.934 | 0.940 |

- **The closest shape,** `llama70b_gate_up_m2048`: TIRx took 987.8 and
  984.1 us in the two runs, and CUTLASS's best configuration (2-CTA 256x256,
  cluster 2x1, TMA store, the tile TIRx's best also uses) 981.2 and 998.4 us.
- **cake and cuTile** run only bf16 and f16, so they have no time on the tf32
  `cutlass_default`. Each has one configuration per shape, its own
  heuristic's.
- **CUTLASS's configurations:** its own `can_implement` rejected 16, and one,
  `cutlass_default` with a 2-CTA 128x128 tile, a 2x2 cluster and the direct
  store, left a wrong result on another rank in the tune. It is not compared.
- **TIRx's configurations:** 1682 tuned, all correct. The table's TIRx
  configurations are in the result files.

## Running

From `tools/`, with the harness environment installed:

```bash
python -m pytest -q -n 16 --dist=worksteal \
  tests/numsim/runtime/test_peer.py \
  tests/analysis_tools/racecheck/test_native_peer_ordering_rules.py \
  tests/analysis_tools/racecheck/test_native_peer_kernels.py \
  tests/analysis_tools/synccheck/test_native_peer_ordering_rules.py \
  tests/analysis_tools/synccheck/test_native_peer_kernels.py

(cd src/tirx_harness/numsim/engine-rs && cargo test --all-features)
(cd frontend-rs && LD_LIBRARY_PATH=$(python -c 'import tvm_ffi, os; print(os.path.dirname(tvm_ffi.__file__))')/lib \
   cargo test --all-features)

python -m benchmarks.peer.desc_scan build all_gather
python -m benchmarks.peer.desc_scan build gemm_all_reduce
python -m benchmarks.peer.desc_scan build two_shot
python -m benchmarks.peer.desc_scan ~/.cache/tirx-megamoe/sm152/*/*.so
```

Use the [multimem commands](MULTI_GPU_MULTIMEM.md#running) for its CPU and GPU
regressions. Run live multi-GPU tests with `-n 1` and separately from benchmarks,
so independent launches do not contend for the same devices.

The prior GB200 run reported 245 peer tests, 348 multimem tests (GPU
microtests included), 885 engine tests, and 15 frontend tests passing.
These are historical results, not a rerun of this draft on current upstream.
See [validation status](MULTI_GPU_VALIDATION.md) for the remaining gates.
