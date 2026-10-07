# Extending NumSim

How to add functionality to the NumSim transpiler and engine.

An **observer** is an `EngineModeImpl`
implementation, a **payload** is an effect variant's field set, a **lowering**
is the Rust emitter function that translates one TIRx op into engine ABI calls,
a **V specialization** is a sealed marker type selecting an instruction's
compile-time behavior, and a **variant catalogue** is static capability data a
family may use to select that marker.

## Adding a new TIRx op

The frontend is the Rust crate `tools/frontend-rs` (crate paths below are
relative to its `src/`): `registry.rs` is the op inventory, and
`emit/<family>.rs` validates and lowers a family's calls. A public TIRx call op
gets one `OpRow` in `registry.rs` with an `emit` callback. For `tirx.ptx.*`,
`ptx_dialect.decode_ptx_call`
(Python, registered as the global function `numsim.frontend.decode_ptx_call`)
is the only wire-ABI decoder and produces `DecodedPtxCall`; `Decoded::new`
calls it on demand and the family functions consume the
decoded operands and modifiers directly. Non-PTX entries read the raw `Call`
where needed. Two routes sit outside that registry and you should know which
one you are in before starting — tile ops dispatch through
`analyze/tile_forms/` and `emit/tile*.rs` (see the end of this section), and
`tirx.address_of` is resolved contextually, because a TensorMap address shares
the generic op name and is detected by `is_tensor_map_address_call` before the
registry lookup in `decode.rs::Decoded::entry`.

**1. Own the op name in its instruction-family module.** Exact PTX op names
come from the target TVM table and are listed once, as the `registry.rs` rows
whose `family` names the owner. A family module names individual ops only where
its behavior differs by op — for example `raw_tcgen.rs`, `raw_memory.rs`,
`atomic_bulk.rs`, or `raw_tma.rs`. Non-PTX names stay with their existing family.

**2. Add family-local validation.** A PTX family's emitter reads named
operands and modifiers from the decoded call (`DecodedPtx`), validates the
engine contract, and selects engine variant markers in `emit/<family>.rs`.
Analysis collects failures by running the same emission traversal.
Keep `decode_ptx_call` as the only positional wire decoder; family emission
must not reconstruct the raw call layout.
Static capability catalogues are appropriate only where the instruction really
has a closed product of engine variants — for example `matrix_variants.rs` and
`ptx_cvt_variants.rs`. Tests should exercise accepted/rejected behavior, not
pin private row counts.

**3. Bind the registry callback.** The row's `emit` field points directly to
the family function; `family` is a documentation label. `schema.rs` rejects
target TVM PTX names without a row; reviewed rows remain registered even when
absent from the target table.

**4. Bind the lowering.** Implement
`fn(&mut Emitter, &Decoded) -> AResult<Option<RustValue>>` in
`emit/<family>.rs`. `Emitter::emit_call` (`emit/calls.rs`) invokes it directly.
Statement lowerings append lines and return `None`; expression lowerings
return `Some(RustValue)`.

**5. Declare fidelity and suspension.** A non-`modeled` op sets `support` and a
`reason` string on its row; an unsupported op is a `support: "rejected"` row.
Set `suspends` for operations that can suspend. The host API returns all bound
buffers unless outputs are selected. Racecheck independently collects possible
global writes while lowering stores: use `emit_explicit_buffer_store` for register
results, and `record_global_write` for any additional structured destination.
Raw writes, including async release, without a static buffer target must set
`written_global_buffers` to
`None`, retaining all host allocations. This monotone emitter fact does not need
a separate IR pass or manifest field. The generated launch passes allocation IDs
from the already prepared buffers directly to Racecheck, preserving host aliases
without changing the Python input payload or host output selection.

Tile ops are a separate route: `registry.rs::TILE_OPS` (names),
`analyze/tile_forms/` (validation in `parse.rs` and the per-family files)
and `emit/tile.rs::emit_tile_call`, with the
per-family emitters in `emit/tile_async_copy.rs`, `emit/tile_gemm_async.rs`
and `emit/tile_tcgen05.rs`. Adding one means a `TILE_OPS` entry, a
validation entry, a lowering arm, and the TVM-side schedule registration that
`test_tile_registry_contract.py` cross-checks.

### What a lowering may and may not do

A lowering validates its call and translates it into exact engine ABI calls.

Allowed: read named fields from the decoded PTX call or the family's parsed
form; name its own sealed V markers as type text (`v2::mem::variant::St<…>`,
`v2::tcgen05::variant::Cp<…>`); use the shared helpers in `analyze/util.rs`
(`dtype_of`, `static_int`, `ffi_text`, `py_repr`) and the emitter's
`RustValue` rather than a private copy; render Rust through `emit/abi.rs`.

Forbidden: assembling a `v2::` call by hand. **Call-position `v2::` emission
belongs in `emit/abi.rs`**, which renders every call prelude from the names,
site and operands a family passes it. The Rust architecture test
(`frontend-rs/tests/architecture.rs`) enforces this mechanically. Naming a variant marker
is explicitly *not* a violation.

Use these `emit/abi.rs` builders instead of literal text (the few listed with
another path are tile-side helpers kept next to their only consumers):

| Need | Builder |
|---|---|
| `(warp, ctx, site)` call | `warp_call`; the families' `*_stateful` wrappers (`emit_sync_stateful`, `emit_tcgen_stateful`, …) build on it |
| `(ctx, site)` call (register family, no warp) | `lane_call`, `lane_call_context` |
| any other prelude | `call`, `context` |
| site identity | `site`, `site_expr` |
| addresses and buffers | `address`, `buffer_address`, `physical_ptr`, `named_buffer` |
| register operands | `register`, `splat`, `per_lane`, `cloned` |
| lane masks | `lane_mask`; `single_lane_mask` (`emit/tile_tcgen05.rs`) |
| mapped elements | `element_ref` |
| mapped views | `mapped_view`, `table_state`, `empty_table_state`, `FN_MAP_STATE` |

`capture_pure_closure` (`emit/tile_async_copy.rs`) is the one shared
implementation of the lazy-capture algorithm for mapper closures; do not
re-derive it per family.

### Checker slices

The Python frontend's `checker_slices.py` — the Synccheck backward slice, the
Racecheck memory-validation slice and the fixed-trace eligibility proof — was
not ported as a whole. The emitted text only ever depended on the
sync slice's `has_unknown_call` bit and the memory-validation slice had no
caller, so what survives is that one predicate:
`fixed_trace_has_unknown_call` in `analyze/tile_forms/mod.rs`. It is true when a
tile sink expression references an unrecognized `Call` without a source-map entry, which
disables fixed-trace verification for the whole kernel, and `emit/module.rs`
evaluates it only for analysis-capable artifacts. A new op needs attention here
only if it can appear in a tile sink expression without a source ID.

### What must pass

Generated source is **expected to change** when you add an op. What is checked in:

- `tests/numsim/registry/test_op_registry_contract.py::test_registry_covers_tirx_cuda_and_ptx_ops`
  — the registry must include all `tirx.cuda.*` / `tirx.ptx.*` op names
  TVM lists. A new TIRx op fails this until registered.
- `tests/numsim/registry/test_operation_runtime_cases.py::test_each_registered_operation_has_a_handwritten_runtime_case`
  — you must add a row to `tests/numsim/support/runtime_cases.py::RUNTIME_CASES`
  naming a real test file and function (`test_runtime_case_selectors_name_existing_tests`
  verifies they exist). Synccheck projects its synchronization-source coverage
  from this same catalog rather than maintaining a second operation-to-test map.
- `tests/numsim/registry/test_engine_support_matrix.py::test_engine_support_matrix_matches_declarative_registries`
  — byte-compares `engine-rs/SUPPORTED_OPS.md` against the rendered registry.
  **Regenerate with `python -m tirx_harness.numsim.transpiler.support_matrix --write`.**
  This is the one genuine blessing flow in the tree.
- `tests/numsim/registry/test_ptx_dialect_decoder.py` — exercises the shared
  PTX wire decoder and proves legacy/non-PTX calls stay outside it.
- Family behavior tests — cover the accepted spelling, adjacent rejected
  spellings, emitted engine instruction, and observable runtime result. Do not
  pin private helper names or catalogue row counts.

## Adding a new engine PTX instruction, and adding a variant of an existing one

### A new mnemonic

**1. Declare the mnemonic in `runtime/instructions/<family>.rs`** — not in
`abi/v2/`. Import the macros with `use super::instruction::{…}` and invoke one
of the six from `abi/v2/instruction.rs` in the declaration block at the top of
the family module, beside the existing invocations — every family does this the
same way (`mem.rs`, `sync.rs`, `reg.rs`, `collective.rs`, `control.rs`,
`tile.rs`). Pick by call shape:

| Macro | Shape |
|---|---|
| `sync_instruction!` | `(engine, ctx, site, args)`, synchronous |
| `sync_instruction_generic_args!` | same, but the operand carrier may be borrowed or owned (`ld`/`st`) |
| `register_instruction!` | `(ctx, site, args)` — no warp receiver, no engine handle |
| `async_instruction!` | awaited, one operand carrier; the `no_args` arm declares an operand-less mnemonic |
| `mapped_instruction!` | whole-tile, views declared as associated spaces; optional trailing `args` |
| `async_mapped_instruction!` | awaited whole-tile |

Each macro generates a **private** `mod $spec` whose `sealed` submodule is only
`pub(super)` — grep `pub(super) mod sealed` in `abi/v2/instruction.rs` — then
re-exports `Variant` under the public name and emits the free entry-point
function. That visibility is why steps 1 and 2 must be the same module: nothing
outside the module that invoked the macro can `impl $spec::sealed::Execute`.
The sealed supertrait is what makes the V set closed — a frontend can name a
marker but cannot add one.

Five of the six macros have their entry point call `<V as Execute>::execute`
directly; `sync_instruction_generic_args!` inserts one hop through its
`sealed::Argument` trait, whose blanket impl forwards to
`<V as Execute<W, Args>>::execute`, so `ld`/`st` accept either operand carrier.

**2. Implement the specializations in that same file** — `impl
<spec>::sealed::Sealed`, `impl <spec>::Variant` (associated `Args` / `Output`,
plus the view spaces for mapped instructions), and `impl <spec>::sealed::Execute`
for each marker.

**3. Re-export from `abi/v2/<family>.rs`.** These modules are `pub use` facades,
not declaration sites — `abi/v2/mem.rs` says so in its own header ("This module
is intentionally declaration-sized"), and eleven of the twelve per-family
modules there contain nothing but the re-export block. Add the new function
name, its `*Variant` name, and any new marker to that
`pub use crate::runtime::instructions::<family>::{…}` block. This is what makes
the mnemonic reachable as `v2::<family>::…` from a generated artifact, and it is
exactly the surface `engine-rs/tests/v2_public_surface.rs` compiles against from
outside the crate. Skipping it leaves the instruction uncallable no matter how
correct the implementation is.

`abi/v2/addr.rs` is the twelfth and the sole exception: it invokes
`sync_instruction!` and implements its variants inline. New families should
follow the other eleven and declare their variants in `runtime/instructions/`.

**4. Use the framework helpers, do not re-write them.**
`runtime/instructions/mod.rs` supplies `begin`, `finish`, and
`singleton_issue_lane`. Use `begin` and `finish` to bracket the instruction's
validation, numerical work, and effects; use `singleton_issue_lane` for
single-lane issue contracts.

**5. Instantiate across the mode axis** with `for_each_engine_mode!`
(`runtime/instructions/mode_axis.rs`). Never name `SyncCheckMode` or
`RaceCheckMode` inside an instruction module — the whole point of the table is
that `runtime/` does not import `native_analysis/`. Choose `test_visible` or
`analysis_only` to match the family's existing instantiated set.

### A variant of an existing mnemonic

Add one row to the family's variant table. Two representative shapes:

```rust
// runtime/instructions/collective.rs — marker => (scope key, diagnostic name, warp-local)
participation_scopes!(
    scope::Warp      => ("participation:warp",      "tile.warp participation",      true),
    scope::Warpgroup => ("participation:warpgroup", "tile.warpgroup participation", true),
    scope::Cta       => ("participation:cta",       "tile.cta participation",       false),
);
```

```rust
// runtime/instructions/sync.rs — operand split, then the shared execute body
arrive_operands!(variant::ArriveLocalCount, (Address<Shared>, R<i64>),
    |args| (args.0, None, Some(args.1), None));
mbarrier_arrive_variant!(variant::ArriveLocalCount, (Address<Shared>, R<i64>));
```

`mbarrier_arrive_variant!` is the reference case: eight PTX spellings collapse
onto one `engine(warp).mbarrier_arrive(…)` call because `ArriveOperands::split`
normalizes the operand tuple first. Other tables in the same style:
`reduce_entries!` / `cta_vote_variants!` (`collective.rs`), `tmem_entries!`
(`tmem.rs`), `setmaxnreg_variant!` (`control.rs`), `bulk_wait_variant!` /
`tma_g2s_variant!` (`async_copy.rs`), the `mem_axis!` / `mem_order_scopes!`
lattice (`mem.rs`), and the eight `packed_narrow_*_variant!` /
`packed_e8m0_*_variant!` macros in `reg.rs`.

A modifier that is numerically observable is a marker axis, never a runtime
flag: `reg.rs` spells `.satfinite`, `.relu`, `.scaled::{n1,n2}::ue8m0`, and
`.pzo` as sealed markers inside
`variant::PackedMode<Round, Saturate, Activation, Scale, Zero>`. The
instantiation only reads their associated constants before calling one
compiled numeric entry in `crate::scalar`. Its trailing parameters default, so
a spelling that predates an axis keeps its emitted variant byte for byte.

A modifier that changes the *operand shape* is a marker axis too: `.rs` is a
rounding marker whose `V::Args` carries the extra `rbits` word, and each
`.scaled::{n1,n2}::ue8m0` mode marker has `V::Args` carry the scale factor.

On the frontend side, update the owning emission family. Families
validate named modifiers in `emit/<family>.rs`; families
with a real closed capability product keep static data beside the emitter,
such as `emit/matrix_variants.rs` and `emit/ptx_cvt_variants.rs`. Add
behavior tests for the new accepted spelling and its fail-closed neighbours. Do
not add a second positional parser, a form that duplicates the decoded call, or
a private-table inventory test.

### Artifact-dependent markers

Raw TCGEN writes the literal
`<artifact-tmem-mode>` placeholder rather than the resolved
`StaticTmem`/`DynamicTmem` marker, because the TMEM mode is a property of the
artifact, not the call site.

### When a raw `tcgen05.mma` variant exceeds the descriptor

The eleven raw MMA entry points in `runtime/tcgen_ops.rs`
(`raw_tcgen05_mma_block_scale_mxf4_e8m0_ss_cta1`,
`raw_tcgen05_mma_block_scale_mxf4nvf4_e2m1_ss_cta1`,
`raw_tcgen05_mma_sp_block_scale_mxf4_e8m0_ss_cta1`,
`raw_tcgen05_mma_sp_f16_ss_cta1`, `raw_tcgen05_mma_f16_f32_ss_cta1`,
`raw_tcgen05_mma_f16_f32_ts_cta1`, `raw_tcgen05_mma_f16_f32_ss_cta2`,
`raw_tcgen05_mma_f16_f32_ts_cta2`, `raw_tcgen05_mma_f8f6f4_cta1`,
`raw_tcgen05_mma_tf32_ts_cta1`,
`raw_tcgen05_mma_block_scale_mxf8f6f4_e2m1_e4m3_e8m0_ss_cta2`) all end by
constructing a `RawMmaTail` and calling `.run(&a_values, &b_values)`. The
per-entry work is decode plus operand gather; the tail is shared. The two
`.kind::mxf4`-family cta_group=1 entries are thin wrappers over one body,
parameterized by `RawTcgenMxf4ScaleSpelling`: `.kind::mxf4`/`.scale_vec::2X`
and `.kind::mxf4nvf4`/`.scale_vec::4X` differ only in the scale type
instruction-descriptor bit 23 encodes, the legal scale-factor IDs, and how many
scale bytes of the TMEM word one instruction reads.

If a new variant fits the existing descriptor, the diff is one `RawMmaTail`
construction. If it does not, these are the touch points, in order:

1. `RawMmaWindow` — add a field only if the new variant
   needs a window property none of `physical`, `context`, `view`,
   `destination_address`, `layout`, `disable_output_lane`, `issuing_lane`,
   `cell_dtype` expresses. A destination element type is a window codec
   (`cell_dtype`), not a new destination variant.
2. `RawMmaDestination` — add a variant if the store *shape* is new.
   The existing three are `Dense` (cell-by-cell over one CTA's window),
   `LaneCells` (one whole TMEM lane per row, block-scaled mxf4), and `Cta2`
   (reaches TMEM through the anchor because the window spans the CTA pair).
   `Cta2` derives its per-CTA row count from `m` — 128 rows at `M=256`, and at
   `M=128` datapath B's 64 rows with the two `N/2` column halves split across
   the 64-lane banks — and carries the pair-wide eight-word
   `disable-output-lane` vector.
3. `RawMmaDestination::read` and `::scatter` — add the
   matching arms. `read` currently dispatches to
   `raw_tcgen05_read_dense_tmem` / `raw_tcgen05_read_cta2_tmem_f32`;
   `scatter` to `raw_tcgen05_scatter_dense`,
   `RawMmaWindow::scatter_lane_cells`, `raw_tcgen05_scatter_cta2_f32`.
4. `RawMmaTail` — add a field only for a genuinely new sequencing
   knob. The existing fields are `m`, `n`, `k`, `enable_input_d`,
   `b_layout`, `destination`, `scale`. `scale` is a `FnOnce() ->
   Result<f32, EngineError>` deliberately: it runs immediately after the
   accumulator read so entries that range-check `scale-input-d` late keep
   surfacing errors in their original order. Preserve that if you touch `run`.
5. One numeric core per accumulator domain: reuse your domain's core, never add
   a second in the same domain. **f32** — `mma_f32_abt_increasing_k`
   (`runtime/tcgen_ops.rs`), shared by raw tcgen05 and typed `tile::gemm*`.
   **f64** — `mma_f64` (`runtime/matrix_ops.rs`) over
   `numsim_fp_env::fma_f64_abt_increasing_k`. **s32** —
   `numsim_fp_env::multiply_accumulate_i32_abt` over an i64 intermediate.
   A non-f32 accumulator needs its own `RawMma*Tail` and `RawMma*Destination`
   plus a matching `TcgenAccumulatorDtype` variant (`runtime/tcgen_work.rs`);
   adding a domain is a reviewed decision.

A missing physical footprint must be declared explicitly. Only
`raw_tcgen05_mma_tf32_ts_cta1`, the two sparse
entries (`AnalysisGapKind::TcgenMma`), and `tcgen05.shift`
(`AnalysisGapKind::TcgenShift`) open one; every other entry records its own
accesses. A new variant of a gap-free family models its footprint too; a new gap
emission needs its own justification.

## Adding a new checker

Worked example: an SMEM bank-conflict checker.

There is **no observer registry**. A checker is a compile-time
`EngineModeImpl` implementation, `EngineMode` is sealed by a private supertrait,
and there is no `dyn EngineMode` anywhere in the crate. One artifact carries one
mode.

**1. Declare the mode and its launch state.** `pub struct BankConflictMode;`
alongside a `BankConflictLaunchState`, in a new module under
`native_analysis/bankcheck/`. Wire it in `lib.rs` with the same
`#[path = "native_analysis/…"] mod …;` pattern the existing checkers use, gated
on a new Cargo feature.

**2. Implement `EngineModeImpl`** (`engine_mode.rs`). Only `LaunchState`,
`GlobalMemoryTransactionGuard<'a>`, `NAME`, `OBSERVES_OPERATIONS`,
`before_operation`, and `after_operation` are required; everything else has a
default. `SyncCheckMode` (`native_analysis/synccheck/sync_check.rs`) is the
minimal-ish reference — the `observes_*` / `controls_*` / `elides_*` gates plus
the four effect and completion hooks. Note its impl block is written
`impl crate::engine_mode::EngineModeImpl for SyncCheckMode`, so grepping for
`impl EngineModeImpl for` misses it. `RaceCheckMode`
(`native_analysis/racecheck/race_check.rs`) additionally opts into the
compact and cached-read fast paths (`OBSERVES_PROXY_MEMORY_DOMAINS`,
`USES_GLOBAL_MEMORY_TRANSACTION`, `USES_CACHED_GLOBAL_READ_FAST_PATH`). A
bank-conflict checker sets `OBSERVES_OPERATIONS = true`, gates
`controls_physical_access` on `PhysicalAccessSpace::Shared`, and leaves the
global-memory transaction machinery at its defaults.

**3. Consume effects, not engine state.** `before_effect` and `after_effect` both
receive `OperationEffect<'_>` borrowed:

- `before_effect` runs after pointer/count/phase resolution and **before** the
  runtime mutates state. Returning `Err` vetoes the operation, so a mode can
  reject a protocol violation without ever observing a partially applied
  effect. Use it for staging and veto.
- `after_effect` runs on a successfully applied effect, before the semantic
  scope closes. Use it for commit and registration.

For bank conflicts, `OperationEffect::PhysicalAccess(batch)` in `after_effect`
carries the per-lane spans; compute the bank histogram there.

**4. Decide your storage policy — it is yours alone.** Delivery is borrowed and
costs nothing. An observer that needs history calls
`OperationEffect::to_owned_effect()` on the subset it keeps and inspects the
stored payload directly. Both ownership forms derive from one field declaration.
Synccheck is the reference consumer — the
shared `engine-rs/src/resolved_transition.rs` (declared at the crate root in
`lib.rs`; not checker-owned) stores
`canonical_sync_payload(effect).to_owned_effect()` in
`ResolvedSynchronizationEffect::from_effect`, only the normalized sync subset.
A folding observer materializes nothing, and
`NumSimMode` compiles away entirely. A bank-conflict checker folds: accumulate
a per-site counter and store nothing.

**5. Report findings.** Each checker owns a status enum and a result struct
(`SyncCheckStatus` / `SyncCheckResult`, `RaceCheckStatus` / `RaceCheckResult`),
a `*_python.rs` serializer (`run_native_sync_check_phase` /
`build_native_sync_check_phase_result`, and the racecheck pair), and a
`#[cfg(feature = "python")]` phase wrapper in `artifact_support.rs`
(`run_synccheck_analysis_phase`, `run_racecheck_analysis_phase`). No
`#[pyfunction]` lives in the engine crate — the artifact's pymodule is emitted
by the frontend crate's `emit/module_template.rs` (reached through
`transpiler/artifact_template.py::emit_rust_module`), which writes
`_native_synccheck_phase` / `_native_racecheck_phase`, and `emit/module.rs`
strips the ones whose feature is off.

**6. Add the mode to the build lattice.** Five coordinated edits:

- `engine-rs/Cargo.toml`: a feature next to `analysis-core` / `racecheck`.
- `runtime/instructions/mode_axis.rs`: one row in `engine_mode_rows!`.
- `runtime/abi_transport.rs`: a `#[cfg]` arm on the `pub type Engine` alias.
- `artifact_support.rs`: a `pub type BankConflictWarpEngine =
  WarpEngine<BankConflictMode>` alias, beside `NumSimWarpEngine` /
  `SyncCheckWarpEngine` / `RaceCheckWarpEngine`.
- Frontend: the `warp_engine_type` selection in `emit/module.rs` and the
  `engine_features` list in `build.py`.

**7. Stay inside the checker-import gate.** `engine-rs/tests/checker_import_gate.rs`
scans every non-test module under `src/native_analysis/` and classifies its
column-0 `use crate::…;` statements against three vocabularies —
`EFFECT_VOCABULARY`, `SHARED_VOCABULARY`, `CHECKER_OWNED` — plus two reasoned
residual allowlists, `ALLOWED_MODULE_PATHS` and `RESIDUAL_ROOT_SYMBOLS`, all
five declared as consts at the top of that file. Import effect payload types
from `crate::effect`, never from `crate::runtime` — the gate's
`classifier_rejects_engine_internals` test pins exactly that regression. The
allowlist ratchets: `allowlist_entries_are_justified_and_live` fails if an entry
is no longer imported, so entries must be deleted as coupling is removed, and a
new entry needs a reason string over 20 characters. Add your checker's own types
to `CHECKER_OWNED`; adding anything to `RESIDUAL_ROOT_SYMBOLS` is a debt entry,
not a normal step.

Read the gate's own caveat before trusting it: import-cleanliness is not
coupling-absence. A payload reached by a method call on a value bound in an
`OperationEffect` match arm needs no `use` at all.

**8. Whole-program information** is computed by the frontend crate's analysis
(`analyze/`) and reaches the checker through the manifest and the emitted
artifact, not through new engine reach-around. Keep instruction-specific
checks in their owning family rather than adding shared analysis passes.

## Adding a new effect variant

Do this when the existing 30-variant vocabulary does not carry the information a
checker needs. Both the delivered and the stored representations are generated
from one declaration, so a variant cannot be added to one and forgotten in the
other.

**Where.** `declare_effect_vocabulary!` is defined in `engine-rs/src/effect.rs`
and has exactly one invocation, in that same file. Add your variant there.

**Field ownership kinds** (from the macro's own `Field ownership kinds:` comment
table, just above the definition):

| Kind | Delivered | Stored |
|---|---|---|
| `copy T` | `T` | `T` |
| `borrow T` | `&'a T` | `T` |
| `opt_copy T` | `Option<T>` | `Option<T>` |
| `opt_borrow T` | `Option<&'a T>` | `Option<T>` |
| `borrow_slice T` | `&'a [T]` | `Box<[T]>` |
| `opt_borrow_slice T` | `Option<&'a [T]>` | `Option<Box<[T]>>` |

Tuple variants take exactly one field; struct variants take named fields:

```rust
MbarrierInitFence {
    barrier_ids: borrow_slice PhysicalBarrierId,
},
MbarrierArrive {
    plan: copy PhysicalMbarrierArrivePlan,
    outcome: opt_copy PhysicalMbarrierArrivalOutcome,
},
SharedBankAccess(borrow PhysicalAccessBatch),
```

**What generates automatically:** the `OperationEffect<'a>` variant, the
`OwnedOperationEffect` variant, the `to_owned_effect()` arm, and the
`as_effect()` arm. Both enums are `#[non_exhaustive]`.

**What you must edit by hand:**

1. `OperationEffect::name()` — exhaustive match; add the arm and its stable
   string. That string appears in checker payloads.
2. The `pub(crate) use` payload re-export block at the top of `effect.rs`. If
   your variant's payload type is not already re-exported there, add it. This
   module is the single import surface a checker may name; the checker-import
   gate depends on it.
3. Any exhaustive `match effect` in the checkers. Synccheck has one in
   `SyncCheckMode::after_effect` — the `profile_count(match effect { … })`
   classification, which has no wildcard arm and will not compile until you
   extend it. Racecheck matches `OperationEffect` in its own `before_effect`
   and `after_effect` bodies. The shared
   `engine-rs/src/resolved_transition.rs` also folds the sync subset through
   `canonical_sync_payload`, which normalizes register/resume pairs before
   storage — a new sync-relevant variant belongs there too.
4. `EFFECT_VOCABULARY` in `engine-rs/tests/checker_import_gate.rs` if checkers
   will import the new payload type by name.

`CompletionActionEffect` and `CompletionEffect`, declared further down
`effect.rs`, are hand-written enums, not generated by
`declare_effect_vocabulary!`. Extending those is a separate edit with no owned
counterpart and no round-trip test.

**Gates that re-run.** The round-trip module `owned_vocabulary_tests` in
`effect.rs` asserts that `to_owned_effect().as_effect()` reproduces the
delivered value exactly, that `name()` survives the round trip, and that the
stored representation is idempotent under re-materialization. Its five tests
sample by *ownership kind*, not by variant, and between them reach 10 of the 30
variants — a new variant is **not** covered
automatically. Add a case through the shared `assert_round_trips` helper: to
`copy_payload_variants_round_trip`, `borrowed_plan_variants_round_trip`, or one
of the three `optional_*_round_trip_present_and_absent` cases if your fields
match a sampled kind, otherwise as a new group. Note `borrow_slice` has no group
at all today — `MbarrierInitFence` is the only variant using it and it is
untested; only `opt_borrow_slice` is sampled.

**Payload A/B discipline.** Publishing a new effect changes what checkers count.
Synccheck's Python payload carries an `effect_summary` and per-kind
`effect_counts`, rendered by `effect_summary_to_py` in `sync_check_python.rs`,
with the diagnostic-byte accounting in
`sync_check_effect_summary_diagnostic_bytes`. Any change to those numbers must
be *declared* in the
change and *asserted by mechanism* — state which effect kind newly fires and
why, and point at the code that makes it fire. A payload delta that is only
observed, never explained, is treated as a regression. Keep the supporting
assertions in the affected checker's tests under `tests/analysis_tools/`.

## The gates

| Gate | Protects | Run it when |
|---|---|---|
| `cargo test --all-features` | engine unit tests, effect round trips, both gate tests below | any engine change |
| `engine-rs/tests/checker_import_gate.rs` | checkers name only effect + shared vocabulary; allowlist ratchets down | any `native_analysis/` change, any new effect payload type |
| `engine-rs/tests/v2_public_surface.rs` | the exact names, paths, associated types, and argument order a generated artifact emits, compiled from outside the crate | any `abi/v2/` change, any variant rename, any signature change |
| Frontend `cargo test --release` | private IR/template tests and emission boundaries | any frontend change |
| `tests/numsim/registry/test_op_registry_contract.py` | registry covers the TIRx op surface; one owner per op | any op added, removed, or renamed |
| `tests/numsim/registry/test_engine_support_matrix.py` | `SUPPORTED_OPS.md` matches the registry — regenerate with `python -m tirx_harness.numsim.transpiler.support_matrix --write` | any op or support-level change |
| `tests/numsim/registry/test_ptx_dialect_decoder.py` | the target-table PTX wire ABI decodes once into named operands/modifiers | any TVM PTX table or decoder change |
| `tests/numsim/registry/test_operation_runtime_cases.py` | every registered op has a hand-written runtime case | any op added |
| Full pytest | NumSim and analysis-tool correctness, representative kernels, and performance regressions | any behavior-affecting change |
| Generated-source byte identity | that a pure move or refactor changed no emission | claiming behavior preservation |
| Full-corpus payload A/B | that engine behavior did not drift | any behavior-affecting engine change |

Run the suite as `python -m pytest -q -n 16 --dist=worksteal`
from `tools/`.

The last two rows are **out-of-tree procedures, not checked-in tests.** There is
no stored generated-source corpus, no payload-diff script, and no bless flow for
either. Both are run by generating in the working tree and in a pristine
baseline worktree and diffing — the byte-identity gate over the representative
kernel cases for all three tools, the payload diff under the `max_workers=1`
regime with wall-clock keys blanked.
Record the evidence in the commit message; that is the only place it lives.
