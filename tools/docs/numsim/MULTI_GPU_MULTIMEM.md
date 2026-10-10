# Multi-GPU NVLink multimem (NVLS)

NumSim, Synccheck, and Racecheck run single-node multi-GPU kernels that
communicate through NVLink multicast (NVLS) memory: `multimem.ld_reduce`,
`multimem.st`, `multimem.red`, and `fence.proxy.alias`. One CPU process
simulates every rank of the launch in a single engine, so cross-rank flag
barriers execute, Synccheck checks their liveness, and Racecheck checks cross-rank
memory ordering under the PTX memory consistency model.

Paths below are relative to `tools/`; `engine-rs/` is
`src/tirx_harness/numsim/engine-rs/`. PTX section numbers refer to the PTX
ISA's "Memory Consistency Model" chapter.

## Check contract and limits

These tools consume **TIRx IR and concrete per-rank inputs**, not generated PTX
or SASS. NumSim executes the selected numerical, control-flow, and memory
operations; the checkers observe that execution. An arrival inside an
input-dependent branch contributes only when that branch executes. Alternate
inputs, atomic return orders, and the paths they could select are not
enumerated. GPU code generation and performance require separate device tests.

| Tool | What a successful check establishes | What it does not establish |
|---|---|---|
| NumSim | Modeled instructions complete and produce concrete outputs on every rank. Compare those outputs with an independent reference. | Race freedom, all possible inputs, GPU instruction timing, or universal bit-exact hardware numerics. |
| Racecheck | The observed conflicting physical byte ranges have the required synchronization, scope, and proxy ordering. This includes RAW, WAR, WAW, and asynchronous buffer reuse. | A numerical oracle, alternate data-dependent paths, or correctness of unmodeled operations. |
| Synccheck | The selected synchronization program meets modeled barrier/lifecycle rules and liveness obligations. | Visibility of data behind a completed flag wait; an ordering-invalid program can still complete. |

A checker verdict is `clean`, `review` (an advisory), `error` (a detected
violation), or `incomplete` (insufficient execution/coverage). A resource or
loop-exploration limit is not proof of deadlock. Inspect the findings and their
source witnesses; use `require_clean()` when an advisory must also fail a gate.

The model expands a multicast operation into accesses to the replicas' physical
bytes. It does **not** make the whole replica set one indivisible transaction.
A matching pointer, a flag reaching its target, or a numerically correct
simulation does not alone establish a happens-before edge. The consumer must
observe the appropriate release, both endpoints must cover the communicating
threads, and any required proxy bridge must lie on that causal path.

The rules and counterexamples below describe the implemented scope. The PTX
references are [multimem addresses and instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-multimem),
[scopes](https://docs.nvidia.com/cuda/parallel-thread-execution/#scope),
[proxies](https://docs.nvidia.com/cuda/parallel-thread-execution/#proxies),
[release/acquire patterns](https://docs.nvidia.com/cuda/parallel-thread-execution/#release-acquire-patterns),
and [causality](https://docs.nvidia.com/cuda/parallel-thread-execution/#causality-order).
Supported forms, passing examples, and negative tests are separate evidence;
this is not a claim to implement every PTX multimem instruction.

**Draft validation:** existing GPU measurements and their limits are preserved.
The current branch still needs fresh native/GPU validation; see the
[validation status and resume checklist](MULTI_GPU_VALIDATION.md).

## Supported operations

| Op | Forms |
|---|---|
| `tirx.ptx.multimem_ld_reduce{,_f,_f_vec}` | `.add/.min/.max/.and/.or/.xor` over `u32 s32 u64 s64 b32 b64 f16 f16x2 bf16 bf16x2 f32 f64` as the ISA table allows, `.acc::f32` for halves, `.weak` or `.relaxed/.acquire` with `.cta/.cluster/.gpu/.sys`, `.v2/.v4/.v8` |
| `tirx.ptx.multimem_st{,_f,_f_vec}` | `.weak` or `.relaxed/.release` with a scope, vector forms |
| `tirx.ptx.multimem_red{,_f,_f_vec}` | `.add/.min/.max/.and/.or/.xor`, `.relaxed/.release` with a scope; an omitted pair is `.relaxed.sys` |
| `tirx.ptx.fence_proxy` with `proxykind=alias` | `fence.proxy.alias` |
| `tirx.nvshmem.my_pe`, `tirx.nvshmem.n_pes` | launch rank and world size |

An access is 32, 64, or 128 bits wide and aligned to its width. The
`.global` state space is the only one accepted.

Still rejected: `multimem.cp.async.bulk`, `multimem.cp.reduce.async.bulk`
(including `.f32` without `.ftz`), `multimem.red.async`, `multimem.st.async`,
and fp8 element types. ptxas itself rejects `multimem.ld_reduce` on
`e4m3`/`e5m2` for sm_100a, sm_100f, and sm_103a. `engine-rs/SUPPORTED_OPS.md`
carries each row's fidelity note.

## Memory ordering rules

Racecheck matches overlapping physical bytes, including a replica reached through
both its multicast and unicast addresses. The rule implementation is
`engine-rs/src/native_analysis/racecheck/global_race.rs`.

- **Cross-rank relation:**
  - **Actor relation.** Findings between ranks report
    `actor_relation: "cross_rank"` (`GlobalActorRelation::CrossRank`).
  - **Scope.** A scope covers a pair only if it reaches both actors, so
    `.gpu` and narrower scopes order nothing across ranks and yield
    `scope_mismatch` findings.
  - **Acquire fast path.** `launch_wide_scope` lifts the acquire fast path's
    floor from `.gpu` to `.sys` once a launch spans ranks
    (`ReleasePayload::every_head_acquirable_by`).
- **Alias proxy bridges:**
  - **Bridging.** `fence.proxy.alias` (`ProxyAsyncFenceScope::Alias`, applied
    by `apply_proxy_alias_fence`) bridges generic and multicast-alias accesses
    in both directions.
  - **Async to alias.** An async-proxy access, such as a TMA store, reaches a
    multicast-alias access only through both fences in order. A
    `fence.proxy.alias` turns the async-to-generic bridge already recorded
    into `AsyncToAlias`. A `fence.proxy.async` turns the alias-to-generic
    bridge already recorded into `AliasToAsync`.
  - **Placement.** A bridge counts only along the synchronization path (8.9.5);
    an alias fence before the writes or after the reads bridges nothing.
  - **Shared memory** has no modeled virtual aliases, so the alias fence is
    global-only there (`race_shadow.rs`).
- **Coherent strong atomics:**
  - **Rule.** `MemoryProxy::atomics_cohere` treats strong atomics through the
    generic and multicast-alias proxies as atomics on one location. A
    multimem atomic is performed at each replica.
  - **What it enables.** The NVLS flag idiom therefore works without an alias
    fence on the flag itself: arrive with `multimem.red.release.sys`, then
    poll the unicast flag with an acquire.
  - **Where it applies.** The same relation is used for moral strength,
    scope-mismatch detection, release-sequence (RMW chain) ancestry, and the
    acquire's head check. Data published through the barrier still needs the
    alias fence.
- **Which ingredient is reported missing:** `global_ordering_failure` names
  the first missing ingredient.
  - **Alias pairs.** For a pair involving the alias proxy, a missing bridge is
    reported (`missing_proxy_bridge`) only when ordinary synchronization
    already orders the pair, or both accesses are in one warp. Otherwise the
    race is `missing_inter_actor_sync`.
  - **Async-proxy pairs.** These still report the bridge first, because async
    completion is not in the generic clocks.

## Synchronization and liveness

Synccheck has no multimem-specific code:
- **Liveness:** all ranks' warps run in one launch, so a cross-rank flag spin
  is an ordinary polling actor, and a flag that can never reach its target is
  a whole-launch deadlock listing every rank's blocked warps.
- **Fence resource:** `fence.proxy.alias` maps to the global proxy-fence
  resource (`resolved_transition.rs`).
- **Scope of the verdict:** Synccheck decides whether waits complete, not what
  they order. The ordering counterexamples below are clean under Synccheck by
  design, and Racecheck holds each of them to its races.

A TMA/bulk load's `mbarrier::complete_tx` contributes completion only after its
async write. A successful wait on the associated barrier supplies the modeled
dependency for subsequent consumers; an issued copy or a matching barrier
address without that wait does not. For a bulk/TMA store,
`cp.async.bulk.wait_group.read` permits reuse of its source after the reads
finish; publishing its destination requires full completion. The GEMM tests
include both a skipped load wait and a read-only store wait.

## Tests

| File | Covers |
|---|---|
| `tests/numsim/runtime/test_multimem.py` | NumSim values and the binding contract: one-shot all-reduce results, `st`/`red`/`ld_reduce` per replica, f32 `red` flush-to-zero, the NVLS accumulator window, `nvshmem` queries for 1/2/4 ranks, window misuse, replica-count and shape errors. |
| `tests/numsim/runtime/test_multimem_kernels.py` | The two-shot all-reduce and GEMM + all-reduce (3 protocols × f32/bf16/f16) all-reduce on every rank, with flags reset, against float64 references. |
| `tests/analysis_tools/racecheck/test_native_multimem.py` | One-shot all-reduce: an alias fence before or after the barrier is clean; none at all races every vector with `missing_proxy_bridge`; no barrier or a `.gpu` barrier races every cross-rank vector; a plain window store is an error. |
| `tests/analysis_tools/racecheck/test_native_multimem_ordering_rules.py` | The PTX-rule counterexamples (rule map below). Most cases assert the exact (writer rank, reader rank, vector) pairs and `ordering_failure`; the schedule-dependent RMW-chain case asserts required and permitted pair sets. |
| `tests/analysis_tools/racecheck/test_native_multimem_kernels.py` | The ported kernels: the two-shot all-reduce, `sys_fenced` (f32/bf16/f16), and its four clean variants have no findings. The kernel counterexamples are FlashInfer's and CUTLASS's signalling, a relaxed wait, alias fences off the path, a missing staging fence, `wait_group.read` before the arrival, and an MMA that skips its load wait. |
| `tests/analysis_tools/synccheck/test_native_multimem.py` | Every one-shot variant's barrier completes; a plain window store is an error. |
| `tests/analysis_tools/synccheck/test_native_multimem_ordering_rules.py` | Every ordering counterexample still completes. Liveness counterexamples deadlock all ranks: a silent first or peer rank, the wrong flag slot, `multimem.st` instead of `red`. Address misuse in both directions is an error. |
| `tests/analysis_tools/synccheck/test_native_multimem_kernels.py` | The two-shot all-reduce's barrier completes, and the GEMM's pipeline barriers and flag spins complete under all three protocols. An MMA skipping its load wait desynchronizes the mbarriers. A kernel built for 4 ranks, launched on 2, never completes. |
| `tests/numsim/microtests/test_multimem.py` | Live GPU runs against NumSim on 2 and 4 GPUs (next section). |
| `engine-rs` unit tests | `MultimemForm::decode` against the ISA table, rank-order folds, accumulator-window and rounding cases (`multimem.rs`); rank-local topology (`topology.rs`); abort hooks leave no write in flight (`race_check.rs`). |

`tests/numsim/support/runtime_cases.py` registers these tests as the runtime
cases of the new ops, which `test_operation_runtime_cases.py` requires.

### PTX rule map

| Rule | Passing form | Counterexamples | Related tests |
|---|---|---|---|
| 8.2.3 multimem addresses | multimem ops on windows | `ld`/`st`/`red` on a window; `multimem.st`/`red` on a unicast address. Both are memory errors in NumSim, Racecheck, and Synccheck. | [racecheck](../../tests/analysis_tools/racecheck/test_native_multimem_ordering_rules.py): `test_non_multimem_access_to_a_multimem_address_is_a_memory_error`; `test_multimem_access_to_a_unicast_address_is_a_memory_error` |
| 8.5, 8.7 scope | `.sys` release and acquire | a `.gpu` red; `fence.acq_rel.gpu` or `.cta` before a `.sys` red (the release pattern's fence must be `.sys`); a `.gpu` acquire CAS; a relaxed CAS then `fence.acq_rel.gpu` (the acquire pattern's fence must be `.sys`); `wait_until(scope="gpu")`. Every cross-rank pair races, plus cross-rank `scope_mismatch`. | [racecheck](../../tests/analysis_tools/racecheck/test_native_multimem_ordering_rules.py): `test_scope_short_of_the_peer_rank_orders_no_cross_rank_pair` |
| 8.7, 8.7.1 moral strength | concurrent `.sys` `st`/`red`; mixed multicast and unicast `.sys` stores | weak `multimem.st` from every rank (write/write races); `.gpu` `st`/`red` (`scope_mismatch`); a weak poll races the arrivals it reads | [racecheck](../../tests/analysis_tools/racecheck/test_native_multimem_ordering_rules.py): `test_morally_strong_concurrent_writes_do_not_race`; `test_gpu_scoped_multimem_writes_are_not_morally_strong_across_ranks` |
| 8.8 release/acquire patterns | `fence.acq_rel.sys` + relaxed red; relaxed CAS + `fence.acq_rel.sys`; `wait_until(scope="sys")` | relaxed red, relaxed CAS, weak poll, or no wait (all cross-rank pairs race); a store after the release fence is unpublished; a read before the acquire fence is unacquired | [racecheck](../../tests/analysis_tools/racecheck/test_native_multimem_ordering_rules.py): `test_no_release_acquire_pattern_orders_no_cross_rank_pair`; `test_store_after_the_release_fence_is_not_published`; `test_read_before_the_acquire_fence_is_not_acquired` |
| 8.9.2 RMW chains | every arrival `.sys` | one rank's `.gpu` red cuts the chain for the peers after it | [racecheck](../../tests/analysis_tools/racecheck/test_native_multimem_ordering_rules.py): `test_one_gpu_scoped_arrival_breaks_the_flag_rmw_chain` |
| 8.9.4 synchronizes-with | wait reads the arrivals | a flag initialized to `world` is read without observing any release | [racecheck](../../tests/analysis_tools/racecheck/test_native_multimem_ordering_rules.py): `test_no_release_acquire_pattern_orders_no_cross_rank_pair` |
| 8.9.5 causality, cumulativity | `bar.warp.sync` around lane 0's arrival and wait; alias fence after the stores or after the wait | no barrier before (stores precede no release) or after (reads follow no acquire); alias fence absent, before the stores, or after the reads (every pair, even same rank, `missing_proxy_bridge`); `multimem.st` broadcast read through unicast (RAW and WAR) without the alias fence | [racecheck](../../tests/analysis_tools/racecheck/test_native_multimem_ordering_rules.py): `test_alias_fence_off_the_causality_path_bridges_nothing`; `test_lanes_off_the_releasing_lane_need_a_barrier`; `test_multicast_publication_read_through_unicast_needs_the_alias_fence` |
| 8.11.1 reductions form no acquire | strong acquire poll | weak poll, then `red.relaxed.sys` to the flag, then `fence.acq_rel.sys` | [racecheck](../../tests/analysis_tools/racecheck/test_native_multimem_ordering_rules.py): `test_no_release_acquire_pattern_orders_no_cross_rank_pair`; `test_weak_flag_poll_races_the_arrivals` |
| Async-proxy completion | staging fence, full `wait_group` | no staging `fence.proxy.async` (TMA store races the epilogue's SMEM writes); `wait_group.read` before the arrival (the store's global writes are neither complete nor bridged) | [racecheck](../../tests/analysis_tools/racecheck/test_native_multimem_kernels.py): `test_unfenced_staging_races_the_tma_store`; `test_arrival_after_a_read_only_store_wait_publishes_no_tile` |
| Liveness (Synccheck) | every rank adds to every flag replica | a silent rank, the wrong slot, `st` instead of `red`, more expected arrivals than ranks: deadlock with every rank's warps blocked | [synccheck](../../tests/analysis_tools/synccheck/test_native_multimem_ordering_rules.py): `test_flag_that_never_reaches_world_deadlocks_every_rank` |

### Live GPU validation

`tests/numsim/microtests/multigpu.py` launches a kernel once per GPU with
`torch.multiprocessing` and NCCL:
- **Windows:** every array a `MulticastWindow` covers is allocated in
  `torch.distributed._symmetric_memory`, and the window parameter receives
  the handle's `multicast_ptr`.
- **Tensor maps:** a `PairedTensorMap` encodes against the device copy of its
  array.
- **Kernel source:** the same prim_func and per-rank binding builders feed
  NumSim, through `run_gpu_kernel` and `run_gpu_case`.

`test_multimem.py` compares, on 2 and 4 GPUs:

| Test | Comparison |
|---|---|
| `test_multimem_matches_gpu` | Each `MULTIMEM_CASES` form, bitwise. f16/bf16 `ld_reduce.add` is allowed one ulp. |
| `test_ordering_litmus_matches_gpu` | The race-free litmus programs Racecheck proves clean, bitwise. |
| `test_ported_two_shot_all_reduce_matches_gpu` | The two-shot all-reduce, bitwise. |
| `test_ported_gemm_all_reduce_matches_gpu_bitwise` | The GEMM + all-reduce with integer operands (exact in any reduction order), every protocol and C dtype, bitwise. |
| `test_ported_gemm_all_reduce_random_operands_agree_with_gpu` | Random operands. NumSim's tcgen05 reduction associates differently from the tensor core, so both are held to C's rounding error against a float64 reference. |

The tests skip unless the host has enough SM100 GPUs with
`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`. They target sm_100a; the GB200
numerics above were measured with them.

### Running

From `tools/`:

```bash
python -m pytest -q -n 16 \
  tests/numsim/runtime/test_multimem.py \
  tests/numsim/runtime/test_multimem_kernels.py \
  tests/analysis_tools/racecheck/test_native_multimem.py \
  tests/analysis_tools/racecheck/test_native_multimem_kernels.py \
  tests/analysis_tools/racecheck/test_native_multimem_ordering_rules.py \
  tests/analysis_tools/synccheck/test_native_multimem.py \
  tests/analysis_tools/synccheck/test_native_multimem_kernels.py \
  tests/analysis_tools/synccheck/test_native_multimem_ordering_rules.py

# Live GPU comparison; -n 1 keeps multi-GPU launches from sharing devices.
python -m pytest -q -n 1 tests/numsim/microtests/test_multimem.py

(cd src/tirx_harness/numsim/engine-rs && cargo test --all-features)
```

The engine's `numpy_backend` tests embed Python and import numpy, so run
`cargo test` with the project's venv active.

The frontend gate is `cargo test --release` in `frontend-rs/`. Its test binary
needs the environment's `tvm_ffi` on the library path: activate the venv that
provides `tvm-ffi-config`, and add `tvm_ffi/lib` to `LD_LIBRARY_PATH`.

## Using it

A multi-rank launch passes **a list of per-rank input dicts** instead of one
dict. Every rank launches the same kernel and grid; entry `r` binds rank `r`'s
parameters. A multicast address is bound with `numsim.MulticastWindow`, built
from the per-rank unicast arrays behind it and passed to the same parameter on
every rank:

```python
import numpy as np
from tirx_harness import numsim, racecheck, synccheck

world = 4
src = [np.arange(128, dtype=np.float32) * (rank + 1) for rank in range(world)]
data = [np.zeros(128, np.float32) for _ in range(world)]  # rank r's unicast replica
flag = [np.zeros(1, np.uint32) for _ in range(world)]
data_mc = numsim.MulticastWindow(data)                     # one multicast address
flag_mc = numsim.MulticastWindow(flag)
inputs = [
    {"world": np.uint32(world), "src": src[rank], "data": data[rank], "mc": data_mc,
     "flag": flag[rank], "flag_mc": flag_mc, "out": np.zeros(128, np.float32)}
    for rank in range(world)
]

result = numsim.Engine().run(numsim.transpile(all_reduce), inputs, outputs=["out"])
rank2_out = result.outputs[numsim.rank_binding_name("out", 2)]  # key "out@rank2"

racecheck(all_reduce, inputs).print()
synccheck(all_reduce, inputs).print()
```

`all_reduce` is a one-shot all-reduce: each rank stores `src` into its
replica `data`, arrives on `flag_mc`, waits until its `flag` reaches `world`,
and `ld_reduce`s `mc` into `out`. `one_shot_all_reduce` and `rank_inputs` in
`tests/numsim/support/multimem_allreduce.py` are the complete version, with
switches for each ordering ingredient.

The binding contract:

- **Outputs** are keyed `rank_binding_name(name, rank)`, which is
  `"name@rank{rank}"`. A parameter bound to a window has no output of its own;
  read a replica through the rank parameter bound to it (`data` above).
- **Window replicas** are one C-contiguous NumPy array per rank, all with the
  same dtype and shape. Each must start its own allocation (a replica may not
  lie inside another bound array's storage), and no replica may alias another
  replica or a window. A window with the wrong replica count fails with "has
  N replicas for M ranks".
- **Only multimem instructions may access a window** (8.2.3). A plain load,
  store, or atomic on a window address fails with "only multimem operations
  may access it". A multimem instruction on an ordinary unicast address fails
  with "is not in a multicast window". NumSim raises `NumSimExecutionError`;
  both checkers return an `error` verdict.
- **Launch scalars must agree.** Scalars that determine the launch must match
  on every rank ("every rank must launch the same grid").
- **Rank queries.** `T.nvshmem.my_pe()` and `T.nvshmem.n_pes()` return the
  launch rank and the world size. Kernel-visible grid coordinates are
  rank-local: `ctaid`/`nctaid`, `clusterid`/`nclusterid`, `blockIdx`, and
  scope-flat CTA and cluster IDs.
- A dict (not a list) is still a single-device launch. The engine limit is
  `MAX_RANKS = 64` (`engine-rs/src/topology.rs`). Multi-node launches are not
  modeled.

## How it is implemented

### Launch model

`LaunchTopology::with_ranks` (`engine-rs/src/topology.rs`) copies one rank's
grid onto `ranks` devices.
- **IDs:** engine-wide cluster, CTA, and warp IDs stay linear. Each rank owns
  a contiguous block of `clusters_per_rank()` clusters.
- **Rank context:** `WarpContext` (`context.rs`) adds `rank()`,
  `kernel_cluster_id()`, `kernel_cta_id()`, and `kernel_topology()`. The
  frontend emits these for kernel-visible coordinates.

Everything that was implicitly "the grid" is now per rank:
- **Grid participant sets:** `ParticipantContract::grid` (`completion.rs`)
  covers only the issuing rank's warps, as `ScopeInstance::RankGrid`.
- **CLC:** `ClcTaskCounter` (`runtime/launch.rs`) keeps one task counter per
  rank, and `clc_try_cancel` claims from the issuing rank's grid.
- **Scope:** `MemoryScope::required_between_warps` (`physical_access.rs`)
  returns `.sys` for warps on different ranks, so only `.sys` operations
  synchronize across devices (8.5).

All ranks share one physical memory:
- **Global:** one global arena holds every rank's allocations.
- **Launch-wide backings:** SMEM, TMEM, and warp-private backings are
  allocated by rank 0's parameter preparation, then replayed for every later
  rank through `PhysicalMemory::reuse_backing` (`spaces.rs`). Every rank's
  buffer table therefore references the same topology-wide backing.
- **Generated code:** the frontend's templates in `frontend-rs/src/emit/module.rs`
  call `extract_rank_inputs` and `prepare_rank_buffers` (`runtime/python.rs`).
  This builds one buffer table per rank, and each warp's future selects its
  rank's table. The Racecheck global-write seed is collected from every rank's
  table.

### Bindings and multicast windows

`prepare_rank_bindings` (`src/tirx_harness/numsim/bindings.py`) flattens the
per-rank dicts into one binding table under rank-qualified names. Each
`MulticastWindow` becomes a placeholder allocation, which carries the window's
host address but is never read or written, plus one allocation per replica.
The engine payload carries:
- `ranks`: each rank's buffers and scalars.
- `multicast`: a list of `{window, replicas}` allocation indices.
- Replicas are added to the written and output allocation sets; windows are
  removed from them.

`api.py::Engine._prepare_rank_execution` and
`checker_runner.py::_prepare_native_rank_bindings` are the NumSim and checker
entries. `_concretize_checker_launch` enforces the matching launch scalars.

On the engine side, `bind_multicast_windows` (`runtime/python.rs`) calls
`BufferView::bind_multicast_replicas` (`memory.rs`). That attaches the
per-rank replica views to the window allocation and validates that they are
distinct, non-window, at least window-sized allocations of the same arena. A
window rejects data access in one place: `GlobalMemory::allocation_for_view`
returns `MemoryError::MulticastAddressAccess`. Forming a view or pointer
inside a window is still legal (`view_allocation`), because a multimem
instruction needs the address.

### Frontend lowering

`frontend-rs/src/emit/multimem.rs` decodes the canonical form and validates its
state space, known type, semantic/scope pair, and total register width. The
engine's `MultimemForm::decode` additionally checks operation/type combinations.
Together these paths reject:
- non-global state spaces;
- types or reductions the ISA doesn't allow for the element type;
- `.acquire` on `st`/`red` and `.release` on `ld_reduce`;
- a scope on `.weak`;
- widths other than 32, 64, or 128 bits;
- sunk source lanes.

Each form lowers to one awaited `v2::mem::multimem` call specialized by
`v2::mem::variant::Multimem<KIND, TYPE, OP, VEC, SEM, SCOPE>`. Register
operands and results travel as up to four little-endian 32-bit words per lane.
`st` and `red` set `written_global_buffers = None`, because the bytes they
write are replicas that no single buffer binding names.

`emit/sync.rs` accepts `fence.proxy.alias` (no state space) as the `Alias`
proxy-fence marker. `emit/pure.rs` lowers the `nvshmem` rank queries.

### Engine execution

`engine-rs/src/runtime/instructions/multimem.rs` is included from `mem.rs` as
the `multimem` mnemonic. `MultimemForm::decode` turns the const codes into an
element type, a reduction, a width, and `MemoryAccessSemantics`:
- **Proxy:** every access goes through the new `MemoryProxy::MulticastAlias`
  (PTX 8.6: a distinct virtual alias behaves as a different proxy).
- **Weak forms** are plain accesses.
- **Ordered forms** are atomic-class accesses with their order and scope;
  `red` uses the reduction class.

`execute` runs each instruction like this:
1. Group active lanes by the window they address, and check each lane's
   alignment and bounds within its window.
2. For each window, visit the replicas in rank order and perform one ordinary
   engine access per replica through the shared paths:
   - `ld_reduce`: `execute_physical_load`.
   - `st`: `execute_physical_store`.
   - `red`: `execute_atomic_access`, an atomic read-modify-write at each
     replica.

   Each access is labelled `"{buffer}@rank{r}"`. All three observers (NumSim,
   Synccheck, Racecheck) therefore see real per-replica effects.
3. `ld_reduce` then folds the loaded values (`ld_reduce_fold`).

**Numerics.**
- **Integer and bitwise reductions, min/max:** folded in rank order. They are
  exact and order-independent.
- **Float `ld_reduce.add`:** the PTX ISA leaves combination order and
  precision unspecified. NumSim models the GB200 NVLink switch as measured:
  - An exact fixed-point sum (`ExactSum`) in a window anchored at
    `anchor = 16 * ceil(e / 16)` for the largest input exponent `e`.
  - f32/f64 terms are truncated toward zero below `2^(anchor - 95)`.
  - A sum below `2^(anchor - 86)` (f32/f64) or `2^(anchor - 38)` (f16/bf16)
    reads as +0.
  - The result is rounded once, ties to even.
  - The accumulator has no -0; infinities and NaNs propagate.
  - `.acc::f32` changes no f16/bf16 hardware result, so it is modeled
    identically.
  - f32/f64 results match GB200 bit for bit. f16/bf16 sums are within one ulp:
    the switch's half rounding is not round-to-nearest-even, and no input-only
    rule reproduces it.
- **`red`:** reduces at each replica like a unicast atomic. `red.add.f32`
  flushes subnormal operands and results (`add_f32_ftz`).

### Failed-access cleanup

**Abort hooks (engine-wide).** `EngineModeImpl` gained `abort_effect` and
`abort_compact_physical_access` (`engine_mode.rs`). `kernel_engine.rs` calls
them when an access's numeric effect fails after `before_*` admitted it.
Racecheck uses them to release the access's in-flight global write spans and
discard its staged batch (`race_check.rs`). Before this, a failed write left
its spans in flight, and an overlapping atomic on another warp spun forever in
`wait_for_quiescent_global_spans`. That is how a plain `red` to a multicast
window used to hang instead of erroring.

### ABI

`NUMSIM_ABI_VERSION` is 40, in both `numsim/abi.py` and `engine-rs/src/lib.rs`
(39 added multimem; 40 added the peer-memory `rank_mappings`, see
[MULTI_GPU_PEER.md](MULTI_GPU_PEER.md)). Stale artifacts are rejected.

## POC kernels

All live in `tests/numsim/support/` and take scalar mode switches. That way one
transpile serves the passing form and every counterexample.

| Kernel | What it is |
|---|---|
| `multimem_allreduce.one_shot_all_reduce` | One warp per rank, the NVLS barrier pattern of CUTLASS's `MulticastSystemBarrier` and FlashInfer's multimem barrier. It publishes through the unicast replica, arrives with `multimem.red.release.sys`, waits on the unicast flag, then `ld_reduce`s the sum. `VARIANTS` move or drop the alias fences, drop the barrier, use `.gpu` scope, or store plainly to the window. |
| `multimem_kernels.two_shot_all_reduce` | Port of CUTLASS's `all_reduce_two_shot_multimem.py`. Each CTA `ld_reduce`s one 128×128 f32 tile of its rank's chunk, `st`s it to every rank, then joins an SM-wise `multimem.red.release.sys` flag barrier. |
| `multimem_kernels.gemm_all_reduce_two_shot` | Port of FlashInfer's `gemm_allreduce_two_shot.py` and CUTLASS's `distributed_gemm_all_reduce_blackwell.py` (LDMCxSTMC), with their warp roles: TMA producer, tcgen05 MMA issuer, a TMEM-to-SMEM-to-TMA-store epilogue, and all-reduce warps that spin on a per-tile flag and `ld_reduce`/`st` the tile through the multicast C. C is f32, bf16, or f16 (`.acc::f32` for halves). `PROTOCOLS` names the signalling (described after this table). |
| `multimem_litmus.message_passing` | MP over NVLS, one warp per rank. Modes choose the release pattern (`ARRIVALS`), the acquire pattern (`WAITS`), where the alias fence sits (`ALIAS`), whether `bar.warp.sync` carries the other lanes, a store after the release fence, a read before the acquire fence, a pre-initialized flag, a wrong flag slot, and a rank that never arrives. |
| `multimem_litmus.concurrent_writes` | Every rank writes the same multicast words with no synchronization (`WRITES`): weak, `.sys`, or `.gpu` `st`/`red`; mixed multicast and unicast `.sys` stores; and both directions of address-kind misuse. |
| `multimem_litmus.broadcast` | The reverse direction: ranks publish with `multimem.st` and read their unicast replica after the `.sys` barrier (RAW) or before it (WAR), with or without the alias fence. |
| `tests/numsim/microtests/cases/multimem.py` | 28 single-instruction cases (`MULTIMEM_CASES`) covering the `ld_reduce`/`st`/`red` forms, types, and vector widths, with value generators that probe ordering, cancellation, special values, and subnormals. |

The GEMM `PROTOCOLS` signalling:

| Protocol | Arrive | Wait |
|---|---|---|
| `flashinfer` | `fence.acq_rel.gpu; multimem.red.relaxed.gpu; fence.proxy.alias` | relaxed `.gpu` CAS spin |
| `cutlass` | `multimem.red.release.gpu` | acquire `.gpu` CAS spin; no alias fence anywhere |
| `sys_fenced` | `fence.proxy.async.global; fence.proxy.alias; multimem.red.release.sys` | acquire `.sys` CAS spin, then `fence.proxy.alias` |

Each `sys_fenced_*` variant changes one switch of `sys_fenced`.

Four variants stay clean:
- `relaxed_arrive`: a fence followed by a relaxed red is still a release
  pattern.
- `producer_alias_only` and `consumer_alias_only`: one alias fence on the
  store-to-reduce path is enough.
- `no_async_fence`: after `wait_group 0`, the TMA store needs no async fence.

The others are counterexamples:
- Alias fence off the path: `no_alias`, and `alias_after_arrive` (FlashInfer's
  placement).
- `relaxed_wait`.
- Three mainloop ingredients:
  - `no_staging_fence` drops the staging `fence.proxy.async.shared::cta`.
  - `unwaited_load` drops the MMA warp's wait on the TMA loads.
  - `read_only_store_wait` waits with `cp.async.bulk.wait_group.read` rather
    than a full wait before the arrival.

`sys_fenced_gpu_scope` is defined, but no test uses it; the litmus scope tests
and `cutlass` cover `.gpu` scope.

Under the PTX model, both upstream signallings leave the all-reduce's reads
unordered:
- **FlashInfer:** a relaxed `.gpu` CAS acquires nothing, so even a rank's own
  tile is unordered.
- **CUTLASS:** `.gpu` release/acquire never reaches a peer rank, and no alias
  fence bridges the TMA store to the multicast reads.

Racecheck reports both. On GB200, every protocol's output still matches
NumSim's bit for bit: the reported races are orderings the PTX model allows,
not corruption the tests observed.

## Ported kernels and parity with their origins

Two TIRx kernels port production NVLS kernels, and both run as fast as their
origins on 4× GB200. The one-shot all-reduce and the litmus kernels have no
origin kernel, so they are not benchmarked.

| TIRx kernel | Origin |
|---|---|
| `multimem_kernels.two_shot_all_reduce` | CUTLASS CuTeDSL `examples/python/CuTeDSL/cute/blackwell/kernel/distributed/all_reduce_two_shot_multimem.py` |
| `multimem_gemm_all_reduce.gemm_all_reduce`, `protocol="cutlass"` | CUTLASS CuTeDSL `.../distributed/distributed_gemm_all_reduce_blackwell.py`, `Sm100PersistentDenseGemmAllReduceLDMCxSTMCKernel` |
| `multimem_gemm_all_reduce.gemm_all_reduce`, `protocol="flashinfer"` | FlashInfer `flashinfer/cute_dsl/gemm_allreduce_two_shot.py`, `PersistentDenseGemmKernel` with `all_reduce="two_shot"` |

The benchmarked two-shot all-reduce is the same prim_func the NumSim and
checker tests run. It keeps the origin's structure: one 128-thread CTA per
128×128 f32 tile of the rank's chunk. Each thread issues its 32
`ld_reduce`/`st` pairs one after another, and each `st` waits for its
`ld_reduce` to come back from the switch. That serialization is why both
kernels need about 75 µs even for 2 MiB.

`gemm_all_reduce` is a full-scale `tirx_lite` kernel and accepts the origins'
whole configuration space:
- 1- or 2-CTA `tcgen05.mma` and the MMA tile.
- The cluster shape, with TMA multicast.
- The persistent scheduler's raster order and swizzle.
- A TMA-store or direct-store epilogue.

It derives stage counts, the epilogue subtile, TMEM columns and the
all-reduce warp count with the origins' formulas. A configuration therefore
yields the same schedule as the origin built with it.

The checker tests run `multimem_kernels.gemm_all_reduce_two_shot` instead (POC
kernels above). It is a NumSim-sized model with the same warp roles and
signalling.

### Where the ports differ from the origins

- **CUTLASS's store wait.** CUTLASS releases a tile's flag after
  `PipelineTmaStore.producer_tail()`, which is
  `cp.async.bulk.wait_group.read 0`. That waits until the TMA store has read
  shared memory, not until C reaches global memory, so a peer's `ld_reduce` may
  read stale C. Racecheck's `read_only_store_wait` counterexample is this
  pattern. TIRx waits with `cp.async.bulk.wait_group 0`.
- **FlashInfer's direct-store epilogue** (`use_tma_store=False`). Warp 0
  arrives without first joining the other epilogue warps' stores. TIRx joins
  the four warps before the arrive.
- **FlashInfer's final barrier.** It passes the CAS operands swapped (CUTLASS
  issue 2845), so the barrier neither waits for the other ranks nor resets its
  flag. A second launch on the same flags then does not synchronize, and a
  third hangs. The benchmark substitutes CuTeDSL's
  `spin_lock_atom_cas_acquire_wait` for that one helper
  (`benchmarks/multimem/origins.py`). TIRx's barrier needs no fix.

### Benchmark method

`benchmarks/multimem/` holds the benchmarks. They import the origin kernels
unchanged from `$CUTLASS_DIR` and `$FLASHINFER_DIR`; `origins.py` restates
only the host wrappers, so that launches go on the capturing stream.

From `tools/`, on a host with 4 multicast-capable GPUs:

```bash
python -m benchmarks.multimem.two_shot_all_reduce \
  --json benchmarks/multimem/results/two_shot_all_reduce_gb200x4.json
# The origins' configuration search; resumable, appends to
# results/gemm_all_reduce_tuning.jsonl.
python -m benchmarks.multimem.gemm_all_reduce tune
# TIRx against each origin's 3 fastest configurations; appends to
# results/gemm_all_reduce_gb200x4.jsonl (a configuration's latest row counts)
# and prints the table.
python -m benchmarks.multimem.gemm_all_reduce compare
# An independent repeat of the comparison.
python -m benchmarks.multimem.gemm_all_reduce compare \
  --out benchmarks/multimem/results/gemm_all_reduce_gb200x4_repeat.jsonl
# The power-history check (notes under Results): idle 0.5 s before every sample.
python -m benchmarks.multimem.gemm_all_reduce compare --idle-s 0.5 \
  --shapes cutlass_doc llama70b_down_m8192 llama70b_o_m8192 llama8b_o_m2048 \
  --out benchmarks/multimem/results/gemm_all_reduce_gb200x4_idle.jsonl
```

**Timing** (`common.time_launches`). TIRx and the origin run in the same rank
processes and on the same symmetric-memory buffers:
- Each implementation gets one CUDA graph. It replays 100 back-to-back
  launches that cycle through 10 workspaces (copies of the inputs, as
  CUTLASS's benchmark does).
- A trial replays the graph for about 30 ms, or once if one replay takes
  longer. There are 9 trials, interleaved across the implementations so that
  clock and thermal drift affect each alike.
- A rank's time is its median trial. The reported time is the slowest rank's.

**Correctness.** Every compared configuration first runs once on zeroed
outputs and flags:
- *GEMM + all-reduce.* Operands are small integers, so the products and the
  f32 accumulation are exact. Every element must be within one ulp of the
  float64 sum over ranks, rounded to C's type, and the flags must be back to
  zero. The bar is one ulp rather than equality because NVLS `ld_reduce ...
  .acc::f32` on 16-bit data is faithfully but not correctly rounded: bf16 ties
  go to the odd neighbor. TIRx's output must also equal the origin's bit for
  bit.
- *Two-shot all-reduce.* Inputs are random normal f32. The output must
  `assert_close` (rtol 1e-5) to NCCL's all-reduce of the same inputs, and the
  flags must be back to zero.

**Configurations.** `tune` times each origin's configuration space on every
shape:
- *CUTLASS:* the space of its `--benchmark_or_test benchmark_all`, with
  `--use_tma_store` as its docstring runs it. That is 2-CTA and 1-CTA MMA
  tiles, clusters (1,1) through (2,2), raster m or n, and swizzle 1, 2, 4
  or 8. On the sweep shapes, raster and swizzle are searched only around the 5
  fastest geometries.
- *FlashInfer:* the same tiles and clusters, with TMA store or direct store.

`compare` builds TIRx from each of the origin's 3 fastest configurations and
times it interleaved with the origin. It reports the origin's best time
against TIRx's best. The parity bar is TIRx ≤ 1.01× the origin on every
shape.

**Shapes.**

| Kernel | Origin shapes | Sweep |
|---|---|---|
| Two-shot all-reduce | 1024×1024 (CUTLASS's default), 1024×512 (its docstring), f32 | tokens 128 to 16384 × hidden 4096 and 8192, f32 |
| GEMM + all-reduce | `cutlass_default` 256×256×512 tf32→f32 (CUTLASS's default); `flashinfer_test` 2048×2048×4096 tf32→f32 (FlashInfer's test); `cutlass_doc` 8192×8192×8192 f16→f16 (CUTLASS's docstring benchmark) | TP=4 row-parallel projections at M = 128, 512, 2048, 8192 tokens, bf16→bf16: Llama-3-8B o_proj (N 4096, K 1024) and down_proj (4096, 3584); Llama-3-70B o_proj (8192, 2048) and down_proj (8192, 7168) |

The shape lists live in `two_shot_all_reduce.py` and `gemm_all_reduce.py`.

### Results

Every shape meets the bar. The worst ratios are 1.003 for the two-shot
all-reduce and 1.008 for the GEMM + all-reduce. Every compared configuration
passed its correctness check, and every TIRx GEMM + all-reduce output equals
its origin's bit for bit.

Measured on 4× GB200 (sm_100a, driver 580.126.20) with torch 2.14.1+cu130,
NCCL 2.30.7, nvidia-cutlass-dsl 4.7.0, CUTLASS `0b55a2f691d6` and FlashInfer
`776939f82869`. The raw rows are in `benchmarks/multimem/results/`.

**Two-shot all-reduce** against CUTLASS, f32, world size 4. The first two rows
are CUTLASS's shapes; the rest are the sweep.

| M×N (f32) | MiB | CUTLASS (µs) | TIRx (µs) | TIRx / CUTLASS |
|---|---|---|---|---|
| 1024×1024 | 4 | 83.08 | 83.13 | 1.001 |
| 1024×512 | 2 | 75.84 | 75.78 | 0.999 |
| 128×4096 | 2 | 76.18 | 75.68 | 0.993 |
| 256×4096 | 4 | 83.02 | 82.30 | 0.991 |
| 512×4096 | 8 | 85.92 | 85.94 | 1.000 |
| 1024×4096 | 16 | 89.10 | 89.07 | 1.000 |
| 2048×4096 | 32 | 95.53 | 95.36 | 0.998 |
| 4096×4096 | 64 | 159.27 | 158.97 | 0.998 |
| 8192×4096 | 128 | 309.47 | 308.79 | 0.998 |
| 16384×4096 | 256 | 601.77 | 600.99 | 0.999 |
| 128×8192 | 4 | 82.76 | 82.40 | 0.996 |
| 256×8192 | 8 | 85.71 | 85.86 | 1.002 |
| 512×8192 | 16 | 89.06 | 89.12 | 1.001 |
| 1024×8192 | 32 | 95.44 | 95.38 | 0.999 |
| 2048×8192 | 64 | 158.46 | 158.89 | 1.003 |
| 4096×8192 | 128 | 307.99 | 308.02 | 1.000 |
| 8192×8192 | 256 | 598.43 | 598.31 | 1.000 |
| 16384×8192 | 512 | 1181.44 | 1181.19 | 1.000 |

**GEMM + all-reduce**, world size 4. Each row compares the origin's best time
over its 3 fastest configurations against TIRx's best over the same
configurations. The times are from `gemm_all_reduce_gb200x4.jsonl`. "Repeat
run" is the ratio from the independent repeat (`..._repeat.jsonl`).
Configurations read as MMA (1- or 2-CTA) and tile, cluster, then CUTLASS's
raster and swizzle; "st" marks FlashInfer's direct-store epilogue.

| Shape | M×N×K | Types | Origin | Origin's best config | Origin (µs) | TIRx (µs) | TIRx / origin | Repeat run | Correct, bitwise equal |
|---|---|---|---|---|---|---|---|---|---|
| `cutlass_default` | 256×256×512 | tf32→f32 | CUTLASS | 1cta 64x64, cl 2x2, n/2 | 12.47 | 11.69 | 0.937 | 0.934 | yes |
| `cutlass_default` | 256×256×512 | tf32→f32 | FlashInfer | 1cta 64x64, cl 2x2, st | 16.64 | 16.53 | 0.993 | 0.993 | yes |
| `flashinfer_test` | 2048×2048×4096 | tf32→f32 | CUTLASS | 2cta 128x128, cl 2x2, m/1 | 72.12 | 71.72 | 0.994 | 0.994 | yes |
| `flashinfer_test` | 2048×2048×4096 | tf32→f32 | FlashInfer | 2cta 128x128, cl 2x2 | 75.32 | 74.91 | 0.995 | 0.995 | yes |
| `cutlass_doc` | 8192×8192×8192 | f16→f16 | CUTLASS | 2cta 256x256, cl 2x1, n/8 | 663.02 | 664.99 | 1.003 | 0.998 | yes |
| `cutlass_doc` | 8192×8192×8192 | f16→f16 | FlashInfer | 2cta 256x256, cl 2x2 | 719.50 | 722.87 | 1.005 | 0.998 | yes |
| `llama8b_o_m128` | 128×4096×1024 | bf16→bf16 | CUTLASS | 1cta 64x64, cl 2x1, n/1 | 15.45 | 15.57 | 1.008 | 1.008 | yes |
| `llama8b_o_m128` | 128×4096×1024 | bf16→bf16 | FlashInfer | 1cta 64x64, cl 2x1 | 18.93 | 18.65 | 0.985 | 0.985 | yes |
| `llama8b_o_m512` | 512×4096×1024 | bf16→bf16 | CUTLASS | 2cta 256x128, cl 2x1, m/1 | 22.63 | 22.49 | 0.994 | 0.994 | yes |
| `llama8b_o_m512` | 512×4096×1024 | bf16→bf16 | FlashInfer | 2cta 256x128, cl 2x1 | 24.86 | 24.74 | 0.995 | 0.995 | yes |
| `llama8b_o_m2048` | 2048×4096×1024 | bf16→bf16 | CUTLASS | 2cta 128x128, cl 2x1, n/4 | 52.02 | 52.31 | 1.006 | 1.005 | yes |
| `llama8b_o_m2048` | 2048×4096×1024 | bf16→bf16 | FlashInfer | 2cta 256x256, cl 2x2 | 61.36 | 61.00 | 0.994 | 0.994 | yes |
| `llama8b_o_m8192` | 8192×4096×1024 | bf16→bf16 | CUTLASS | 1cta 128x64, cl 1x2, n/4 | 162.79 | 162.70 | 0.999 | 1.000 | yes |
| `llama8b_o_m8192` | 8192×4096×1024 | bf16→bf16 | FlashInfer | 2cta 256x256, cl 2x1 | 176.87 | 176.55 | 0.998 | 0.998 | yes |
| `llama8b_down_m128` | 128×4096×3584 | bf16→bf16 | CUTLASS | 2cta 128x64, cl 2x1, n/1 | 19.56 | 19.53 | 0.999 | 0.999 | yes |
| `llama8b_down_m128` | 128×4096×3584 | bf16→bf16 | FlashInfer | 2cta 128x64, cl 2x1 | 23.23 | 22.88 | 0.985 | 0.985 | yes |
| `llama8b_down_m512` | 512×4096×3584 | bf16→bf16 | CUTLASS | 2cta 128x128, cl 2x1, n/1 | 27.71 | 27.37 | 0.988 | 0.988 | yes |
| `llama8b_down_m512` | 512×4096×3584 | bf16→bf16 | FlashInfer | 2cta 128x256, cl 2x1 | 31.56 | 31.43 | 0.996 | 0.996 | yes |
| `llama8b_down_m2048` | 2048×4096×3584 | bf16→bf16 | CUTLASS | 2cta 128x128, cl 2x1, n/1 | 59.16 | 59.25 | 1.002 | 0.999 | yes |
| `llama8b_down_m2048` | 2048×4096×3584 | bf16→bf16 | FlashInfer | 2cta 256x128, cl 2x2 | 69.99 | 69.57 | 0.994 | 0.995 | yes |
| `llama8b_down_m8192` | 8192×4096×3584 | bf16→bf16 | CUTLASS | 2cta 256x128, cl 2x2, n/1 | 182.46 | 183.10 | 1.003 | 1.006 | yes |
| `llama8b_down_m8192` | 8192×4096×3584 | bf16→bf16 | FlashInfer | 2cta 256x256, cl 2x1 | 198.61 | 197.42 | 0.994 | 0.997 | yes |
| `llama70b_o_m128` | 128×8192×2048 | bf16→bf16 | CUTLASS | 2cta 128x128, cl 2x1, n/1 | 20.12 | 19.73 | 0.980 | 0.981 | yes |
| `llama70b_o_m128` | 128×8192×2048 | bf16→bf16 | FlashInfer | 2cta 128x128, cl 2x2 | 25.25 | 24.98 | 0.989 | 0.989 | yes |
| `llama70b_o_m512` | 512×8192×2048 | bf16→bf16 | CUTLASS | 2cta 128x128, cl 2x2, m/1 | 36.21 | 36.24 | 1.001 | 1.000 | yes |
| `llama70b_o_m512` | 512×8192×2048 | bf16→bf16 | FlashInfer | 2cta 256x128, cl 2x2 | 40.51 | 40.31 | 0.995 | 0.996 | yes |
| `llama70b_o_m2048` | 2048×8192×2048 | bf16→bf16 | CUTLASS | 2cta 128x128, cl 2x1, m/1 | 92.37 | 92.47 | 1.001 | 1.001 | yes |
| `llama70b_o_m2048` | 2048×8192×2048 | bf16→bf16 | FlashInfer | 2cta 128x256, cl 2x1 | 101.61 | 100.88 | 0.993 | 0.993 | yes |
| `llama70b_o_m8192` | 8192×8192×2048 | bf16→bf16 | CUTLASS | 1cta 128x64, cl 1x2, n/4 | 313.46 | 312.38 | 0.997 | 0.997 | yes |
| `llama70b_o_m8192` | 8192×8192×2048 | bf16→bf16 | FlashInfer | 1cta 64x256, cl 1x2 | 335.31 | 335.23 | 1.000 | 1.000 | yes |
| `llama70b_down_m128` | 128×8192×7168 | bf16→bf16 | CUTLASS | 2cta 128x128, cl 2x1, n/1 | 33.19 | 32.68 | 0.985 | 0.984 | yes |
| `llama70b_down_m128` | 128×8192×7168 | bf16→bf16 | FlashInfer | 2cta 128x64, cl 2x1 | 37.19 | 36.64 | 0.985 | 0.985 | yes |
| `llama70b_down_m512` | 512×8192×7168 | bf16→bf16 | CUTLASS | 2cta 256x128, cl 2x1, m/1 | 57.66 | 57.22 | 0.992 | 0.992 | yes |
| `llama70b_down_m512` | 512×8192×7168 | bf16→bf16 | FlashInfer | 2cta 128x256, cl 2x1 | 59.92 | 59.68 | 0.996 | 0.992 | yes |
| `llama70b_down_m2048` | 2048×8192×7168 | bf16→bf16 | CUTLASS | 2cta 128x256, cl 2x1, m/1 | 170.62 | 170.43 | 0.999 | 0.992 | yes |
| `llama70b_down_m2048` | 2048×8192×7168 | bf16→bf16 | FlashInfer | 2cta 256x128, cl 2x1 | 172.48 | 171.39 | 0.994 | 0.989 | yes |
| `llama70b_down_m8192` | 8192×8192×7168 | bf16→bf16 | CUTLASS | 2cta 256x256, cl 2x2, m/4 | 615.89 | 610.34 | 0.991 | 1.005 | yes |
| `llama70b_down_m8192` | 8192×8192×7168 | bf16→bf16 | FlashInfer | 2cta 256x256, cl 2x2 | 680.48 | 678.48 | 0.997 | 0.999 | yes |

Notes on reading these numbers:
- **Large shapes are power-capped.** On `cutlass_doc` each GPU draws about
  1135 W against its 1200 W limit, and SM clocks settle between 1460 and
  1640 MHz, below the 2062 MHz maximum.
  - Ratios on such shapes move by about ±1% between runs. An earlier full
    comparison measured 1.017 against CUTLASS on `cutlass_doc` (668.85 µs
    against 680.22 µs), and three reruns of that shape measured 1.002 to
    1.006.
  - At CUTLASS's best configuration, two sustained 4-second phases per
    kernel, alternating, measured CUTLASS at 666.9 and 667.7 µs and TIRx at
    667.1 and 668.2 µs. Both drew the same power at the same clocks.
  - TIRx's persistent tile order was also checked on the GPU against CuTeDSL's
    `StaticPersistentTileScheduler`. It is identical for every raster and
    swizzle.
  - A power-capped sample's clock also depends on what ran just before it
    (see the all-gather benchmark in
    [MULTI_GPU_PEER.md](MULTI_GPU_PEER.md#benchmark-method)). Here the
    origin and TIRx alternate on the same tile, so each follows the other. A
    rerun of the four heaviest or closest shapes with the device idle 0.5 s
    before every sample (`compare --idle-s 0.5`,
    `gemm_all_reduce_gb200x4_idle.jsonl`) measured, against CUTLASS and
    FlashInfer: `cutlass_doc` 0.988 and 1.001, `llama70b_down_m8192` 0.995
    and 0.995, `llama70b_o_m8192` 0.996 and 0.999, `llama8b_o_m2048` 1.005
    and 0.995. All were correct and bitwise equal. A first such rerun had
    measured 1.010 on `llama70b_down_m8192` against CUTLASS (544.4 against
    539.1 µs). Times drop with the idle period (CUTLASS on `cutlass_doc`:
    591 against 663 µs), as fewer samples reach the cap.
- **The largest steady gap is `llama8b_o_m128` against CUTLASS**: 1.008 in
  every run. It comes from the CUTLASS-protocol exit fence described below.

### What parity took

Two changes closed the last gaps:
- **MMA and TMA issue.** One elected thread runs each of these warps' whole
  loop. CUTLASS does the equivalent: ptxas drops its per-instruction
  `elect.sync`, because `tcgen05.mma` executes once per warp. Electing inside
  the k-loop instead costs an `ELECT`, a branch and a `BSSY`/`BSYNC`
  reconvergence on every k-tile.
  - Where it mattered: with a 2-CTA 128×128 MMA tile (64 rows per CTA), each
    `tcgen05.mma` is about 32 cycles of tensor work, and that per-k-tile
    overhead left the tensor pipe idle.
  - Effect on `flashinfer_test`: the GEMM alone took 57.6 µs against CUTLASS's
    48.6 µs, and 48.4 µs after the change.
  - The same rewrite keeps the accumulate flag warp-uniform (it was assigned
    inside the elected branch, so ptxas needed a `VOTEU` per k-tile). It also
    advances the smem descriptors with 32-bit arithmetic on their low word.
- **CUTLASS-protocol exit barrier.** Each all-reduce thread issues
  `fence.acq_rel.gpu` before the final `bar.sync`. The leader's
  `multimem.red.release.sys` is cumulative over that `bar.sync`, so the fence
  adds no ordering. Its purpose is speed.
  - Observed cause: the `MEMBAR.SYS` behind the `.sys` release completes in
    GPU-wide batches. On 4× GB200, a CTA that issued it more than about 0.5 µs
    after the first CTA on its GPU waited about 1.9 µs longer.
  - Effect on `cutlass_default`: 1.074× without the fence, 0.96× to 0.965× with
    it across four flag-buffer placements.
  - On the other shapes, at CUTLASS's best configuration, the fence moves the
    ratio by less than 1% either way:
    - It helps most on `llama70b_o_m128` (0.982× with it, 0.990× without) and
      `llama70b_down_m128` (0.985×, 0.988×).
    - It costs most on `llama8b_o_m128` (1.008×, 1.001×) and
      `llama8b_down_m128` (0.998×, 0.993×).
    - TIRx keeps it on every shape. CUTLASS's kernel ends with the same
      `multimem` instructions and no fence. Why TIRx needs the fence on
      `cutlass_default` and CUTLASS does not was not isolated.
  - Under FlashInfer's protocol the same fence costs more than it saves
    (1.02× to 1.04× with it, 0.98× to 0.99× without), so that protocol does not
    use it.
