//! Checker-import gate.
//!
//! The mandated checker boundary is the observer notification: a mode reads the
//! `Effect` payload it is handed, plus shared vocabulary (ids, masks, spans).
//! Reaching around the payload into engine internals must fail the build.
//!
//! This gate enforces that on every non-test module under `src/native_analysis/`
//! by classifying its `use crate::…` statements. Everything not classifiable as
//! effect vocabulary, shared vocabulary, or checker-owned code must appear in
//! the residual allowlist below with a reason. The allowlist ratchets: entries
//! come off as the corresponding coupling is removed, and nothing may be added
//! without a reason recorded here.
//!
//! ## What this gate does NOT prove
//!
//! Import-cleanliness is not coupling-absence. A payload reached by a method
//! call on a value bound in an `OperationEffect` match arm needs no `use` at
//! all — `TcgenLifecyclePlan` and `SetmaxnregPlan` are consumed by synccheck
//! today with zero imports naming them. A green gate means "no checker names an
//! engine internal it was not handed", not "no checker depends on one".
//!
//! ## Scope rule
//!
//! Only column-0 `use crate::…;` statements are gated: those are the module's
//! own imports. Imports inside `mod tests { … }` are indented and unrestricted,
//! since test code may construct engine state directly. A column-0 statement
//! carrying `#[cfg(test)]` on the preceding line is likewise skipped.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Allowed module paths (`use crate::<module>::…`)
// ---------------------------------------------------------------------------

/// One permitted `use crate::<module>::…` reach.
struct ModuleAllowance {
    module: &'static str,
    reason: &'static str,
    /// Exact symbols permitted from this module. Empty means "any symbol",
    /// used only for modules that ARE checker-owned code.
    symbols: &'static [&'static str],
}

const fn allow(
    module: &'static str,
    reason: &'static str,
    symbols: &'static [&'static str],
) -> ModuleAllowance {
    ModuleAllowance {
        module,
        reason,
        symbols,
    }
}

/// A checker may name these engine modules directly. Residual entries pin the
/// exact symbols permitted, so a new reach into an already-allowlisted engine
/// module is still rejected.
const ALLOWED_MODULE_PATHS: &[ModuleAllowance] = &[
    // The effect vocabulary itself — the mandated boundary.
    allow(
        "effect",
        "the effect vocabulary and its payload types: the checker boundary",
        &[],
    ),
    // Modules that ARE native_analysis, reached through the crate root because
    // `lib.rs` is the crate's only module hub. Not coupling.
    allow(
        "race_shadow",
        "checker-owned: src/native_analysis/racecheck/race_shadow.rs",
        &[],
    ),
    allow(
        "transactional_interval_map",
        "checker-owned: src/native_analysis/racecheck/transactional_interval_map.rs",
        &[],
    ),
    allow(
        "sync_check_python",
        "checker-owned: src/native_analysis/synccheck/sync_check_python.rs",
        &[],
    ),
    allow(
        "setmaxnreg_verifier",
        "checker-owned: src/native_analysis/synccheck/setmaxnreg_verifier.rs",
        &[],
    ),
    allow(
        "strict_mbarrier",
        "checker-owned: src/native_analysis/synccheck/strict_mbarrier.rs",
        &[],
    ),
    // ---- residual allowlist (ratchets to empty) ----
    allow(
        "runtime",
        "RESIDUAL: the python shims name the launch surface; no effect payload \
         may be imported from here — payloads come from crate::effect",
        &["ExecutionPolicy", "LaunchSelection"],
    ),
    allow(
        "resolved_transition",
        "RESIDUAL: the owned vocabulary P3 step 2 dissolves; the offline \
         verifier still replays from the fixed-sync log snapshot",
        &[
            "FixedSyncLogSnapshot",
            "FixedSyncSetmaxParticipantSnapshot",
            "FixedSyncSnapshotRecord",
        ],
    ),
    allow(
        "physical_access",
        "RESIDUAL: racecheck's compact shadow needs the compact batch \
         representation, not only the delivered PhysicalAccessBatch",
        &[
            "CompactPhysicalAccessBatch",
            "ProxyMemoryDomain",
            "coalesce_physical_access_batches",
        ],
    ),
    allow(
        "engine_mode",
        "RESIDUAL: cached global reads and their exact-range progress are \
         associated with the observer registration trait a checker implements",
        &[
            "CachedGlobalReadAccess",
            "CachedGlobalReadFinish",
            "GlobalMemoryProgress",
            "GlobalMemoryProgressEpoch",
            "GlobalMemoryProgressSnapshot",
        ],
    ),
    allow(
        "strict_named_barrier",
        "RESIDUAL: synccheck-only protocol machine, same relocation",
        &["StrictNamedBarrierSemanticState"],
    ),
    allow(
        "strict_cluster_barrier",
        "RESIDUAL: synccheck-only protocol machine, same relocation",
        &["StrictClusterBarrierSemanticState"],
    ),
    allow(
        "setmaxnreg",
        "RESIDUAL: setmaxnreg constants and resource ids shared with the engine",
        &[
            "SETMAXNREG_COUNT_GRANULARITY",
            "SETMAXNREG_CTA_REGISTER_POOL",
            "SETMAXNREG_MAX_COUNT",
            "SETMAXNREG_MIN_COUNT",
            "SetmaxnregAction",
            "SetmaxnregResource",
        ],
    ),
    allow(
        "operation",
        "RESIDUAL: race_shadow names LoopFrame directly instead of taking it \
         through the crate-root facade",
        &["LoopFrame"],
    ),
];

// ---------------------------------------------------------------------------
// Allowed crate-root symbols (`use crate::{A, B, …}`)
// ---------------------------------------------------------------------------

/// Effect vocabulary: the delivered payload and everything reachable inside it.
const EFFECT_VOCABULARY: &[&str] = &[
    "AnalysisGapDomain",
    "AnalysisGapEffect",
    "AnalysisGapKind",
    "AsyncGroupCompletionActionId",
    "AsyncGroupMilestone",
    "AsyncTokenId",
    "ClusterBarrierArrivalOutcome",
    "CompletionActionEffect",
    "CompletionEffect",
    "MemoryFenceEffect",
    "NamedBarrierArrivalOutcome",
    "OperationEffect",
    "OwnedOperationEffect",
    "PhysicalAccessBatch",
    "PhysicalCompletionAction",
    "PhysicalCompletionActionId",
    "PhysicalCompletionKind",
    "PhysicalCompletionOutcome",
    "PhysicalMbarrierArrivalOutcome",
    "ProxyAsyncFenceEffect",
    "ProxyAsyncFenceScope",
];

/// Shared vocabulary: ids, masks, spans, contexts, errors, launch shape.
const SHARED_VOCABULARY: &[&str] = &[
    "AllocationId",
    "ClusterBarrierId",
    "ClusterBarrierParticipantExitEvidence",
    "DynamicOpId",
    "EngineError",
    "EngineErrorKind",
    "ExecutionReport",
    "ExecutionStats",
    "LanePhysicalAccess",
    "LaunchTopology",
    "MAX_MBARRIER_EXPECTED_ARRIVALS",
    "MemoryAccessSemantics",
    "MemoryOrder",
    "MemoryProxy",
    "MemoryScope",
    "NamedBarrierId",
    "OperationContext",
    "OperationKind",
    "PhysicalAccessDescriptor",
    "PhysicalAccessKind",
    "PhysicalAccessSpace",
    "PhysicalAllocationId",
    "PhysicalBarrierId",
    "PhysicalByteSpan",
    "SETMAXNREG_COUNT_GRANULARITY",
    "SETMAXNREG_CTA_REGISTER_POOL",
    "SETMAXNREG_WARPS_PER_GROUP",
    "SetmaxnregAction",
    "SetmaxnregResource",
    "setmaxnreg_default_register_count",
    "TMEM_COLUMN_CAPACITY",
    "TcgenLifecycleAction",
    "WARP_SIZE",
    "WarpMask",
    "engine_error_kind",
];

/// Checker-owned types that reach a sibling checker module through the crate
/// root only because `lib.rs` is the crate's module hub. Not coupling.
const CHECKER_OWNED: &[&str] = &[
    "AliasStaleReadAdvisory",
    "BarrierClockPayload",
    "CheckerLaunchContext",
    "DeclaredWordBypassDiagnostic",
    "FixedSyncDeadlock",
    "FixedSyncProgram",
    "FixedSyncProgramBuildError",
    "FixedSyncProgramError",
    "FixedSyncTransition",
    "FixedSyncTransitionEvidence",
    "FixedSyncVerificationError",
    "FixedSyncVerificationIncomplete",
    "FixedSyncVerificationResult",
    "GlobalScopeMismatchDiagnostic",
    "PhysicalRaceFinding",
    "PhysicalRaceKind",
    "PhysicalRaceWitness",
    "RaceBatchValidation",
    "RaceCheckAccessRecord",
    "RaceCheckIncompleteReason",
    "RaceCheckLaunchState",
    "RaceCheckMode",
    "RaceCheckResult",
    "RaceCheckStatus",
    "RaceShadow",
    "RaceShadowError",
    "RaceVectorClock",
    "StrictMbarrierCompletionToken",
    "StrictMbarrierEffect",
    "StrictMbarrierError",
    "StrictMbarrierProtocol",
    "StrictMbarrierSnapshot",
    "StrictMbarrierWaitOutcome",
    "SyncCausalityError",
    "SyncCausalityTracker",
    "SyncCheckEffectOutcome",
    "SyncCheckIncompleteReason",
    "SyncCheckLaunchState",
    "SyncCheckMode",
    "SyncCheckProtocolError",
    "SyncCheckResult",
    "SyncCheckStatus",
    "SyncCheckWaitState",
    "SyncClockPayload",
    "SyncStateFailure",
    "SyncStateSearchLimits",
    "SyncStateSearchOptions",
    "SyncStateSearchTermination",
    "SyncTransitionSystem",
    "SyncVectorClock",
    "UndeclaredProtocolWordDiagnostic",
    "explore_sync_states",
    "retire_barrier_generations",
    "retire_named_barrier_generations",
    "verify_fixed_sync_programs",
];

/// `(symbol, reason)`. Engine internals a checker still names. This list is the
/// ratchet: it may shrink freely; growing it requires a recorded reason.
const RESIDUAL_ROOT_SYMBOLS: &[(&str, &str)] = &[
    (
        "EngineModeImpl",
        "the observer registration trait a checker implements — this IS the \
         boundary, not a violation of it",
    ),
    (
        "ResolvedCompletionEffect",
        "P3 step 2 target: the owned vocabulary being dissolved",
    ),
    (
        "ResolvedMemoryEffect",
        "P3 step 2 target: the owned vocabulary being dissolved",
    ),
    (
        "ResolvedSyncResource",
        "P3 step 2 target: the owned vocabulary being dissolved",
    ),
    (
        "ResolvedSyncResourceKey",
        "P3 step 2 target: the owned vocabulary being dissolved",
    ),
    (
        "ResolvedSynchronizationEffect",
        "P3 step 2 target: the owned vocabulary being dissolved",
    ),
    (
        "ResolvedTransitionLog",
        "P3 step 2 target: storage moves to OwnedOperationEffect",
    ),
    (
        "ResolvedTransitionRegistration",
        "P3 step 2 target: the owned vocabulary being dissolved",
    ),
    (
        "ResolvedTransitionSummary",
        "P3 step 2 target: the owned vocabulary being dissolved",
    ),
    (
        "StrictClusterBarrierError",
        "synccheck-only protocol machine still at src/ root",
    ),
    (
        "StrictClusterBarrierOutcome",
        "synccheck-only protocol machine still at src/ root",
    ),
    (
        "StrictClusterBarrierProtocol",
        "synccheck-only protocol machine still at src/ root",
    ),
    (
        "StrictNamedBarrierError",
        "synccheck-only protocol machine still at src/ root",
    ),
    (
        "StrictNamedBarrierOperation",
        "synccheck-only protocol machine still at src/ root",
    ),
    (
        "StrictNamedBarrierOutcome",
        "synccheck-only protocol machine still at src/ root",
    ),
    (
        "StrictNamedBarrierProtocol",
        "synccheck-only protocol machine still at src/ root",
    ),
    (
        "StrictNamedBarrierSnapshot",
        "synccheck-only protocol machine still at src/ root",
    ),
    (
        "BlockedOperation",
        "scheduler park accounting; the design's blocked-reason-at-park work \
         (§2.3) turns this into effect payload",
    ),
    (
        "CoverageBounds",
        "schedule-search resource accounting, not an effect path",
    ),
    (
        "CoverageStatus",
        "schedule-search resource accounting, not an effect path",
    ),
    (
        "CoverageSummary",
        "schedule-search resource accounting, not an effect path",
    ),
    (
        "CoverageUsage",
        "schedule-search resource accounting, not an effect path",
    ),
    (
        "ResourceAmount",
        "schedule-search resource accounting, not an effect path",
    ),
    (
        "ResourceLimitHit",
        "schedule-search resource accounting, not an effect path",
    ),
    (
        "ResourceLimitKind",
        "schedule-search resource accounting, not an effect path",
    ),
    (
        "ResourceLimits",
        "schedule-search resource accounting, not an effect path",
    ),
    (
        "ResourceUsage",
        "schedule-search resource accounting, not an effect path",
    ),
    (
        "SearchTermination",
        "schedule-search resource accounting, not an effect path",
    ),
    ("ProfileKind", "instrumentation, orthogonal to the boundary"),
    (
        "ProfileTimer",
        "instrumentation, orthogonal to the boundary",
    ),
    (
        "profile_count",
        "instrumentation, orthogonal to the boundary",
    ),
    (
        "profile_reset",
        "instrumentation, orthogonal to the boundary",
    ),
    (
        "profile_snapshot",
        "instrumentation, orthogonal to the boundary",
    ),
    (
        "PhysicalMemory",
        "racecheck reads final memory state to classify a finding",
    ),
    (
        "WarpEngine",
        "racecheck's mode hooks receive the engine handle; narrowing it to the \
         ~10 issue/effect entry points is engine-side work (§2.2)",
    ),
    (
        "MbarrierCompletionCausalToken",
        "synccheck imports it as a bare crate-root path",
    ),
];

// ---------------------------------------------------------------------------
// Scanner
// ---------------------------------------------------------------------------

fn native_analysis_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/native_analysis")
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries = fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", dir.display()))
        .map(|entry| entry.expect("directory entry").path())
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// One gated import statement: `use crate::<rest>;` written at column 0.
#[derive(Debug)]
struct GatedImport {
    file: String,
    statement: String,
}

fn gated_imports(path: &Path) -> Vec<GatedImport> {
    let text = fs::read_to_string(path).expect("read source");
    let file = path
        .strip_prefix(native_analysis_root())
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned();
    let lines = text.lines().collect::<Vec<_>>();
    let mut imports = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        if !line.starts_with("use crate::") {
            index += 1;
            continue;
        }
        // A column-0 `use` carrying `#[cfg(test)]` is test-only.
        if index > 0 && lines[index - 1].trim_start().starts_with("#[cfg(test)]") {
            index += 1;
            continue;
        }
        let mut statement = String::new();
        while index < lines.len() {
            statement.push(' ');
            statement.push_str(lines[index].trim());
            let done = lines[index].trim_end().ends_with(';');
            index += 1;
            if done {
                break;
            }
        }
        imports.push(GatedImport {
            file: file.clone(),
            statement: statement.trim().to_string(),
        });
    }
    imports
}

/// What one gated `use crate::…;` statement names.
#[derive(Debug, Default)]
struct Named {
    /// The engine module reached into, if any, with the symbols taken from it.
    module: Option<(String, Vec<String>)>,
    /// Symbols taken straight off the crate root.
    root_symbols: Vec<String>,
}

fn split_braced(list: &str) -> Vec<String> {
    list.split(',')
        .map(str::trim)
        .filter(|symbol| !symbol.is_empty())
        .map(str::to_string)
        .collect()
}

/// Split a gated statement into the engine module it reaches into (if any) and
/// the crate-root symbols it names.
///
/// `use crate::foo::{Bar, Baz};` reaches module `foo` for `Bar` and `Baz`.
/// `use crate::{A, B};` and the bare `use crate::A;` name crate-root symbols
/// and reach no module.
fn classify(statement: &str) -> Named {
    let body = statement
        .trim_start_matches("use ")
        .trim_end_matches(';')
        .trim();
    let rest = body
        .strip_prefix("crate::")
        .expect("gated on crate:: paths");
    if let Some(inner) = rest.strip_prefix('{') {
        return Named {
            module: None,
            root_symbols: split_braced(inner.trim_end().trim_end_matches('}')),
        };
    }
    match rest.split_once("::") {
        Some((module, tail)) => {
            let tail = tail.trim();
            let symbols = match tail.strip_prefix('{') {
                Some(inner) => split_braced(inner.trim_end().trim_end_matches('}')),
                // `crate::foo::Bar` or a deeper path; take the next segment.
                None => vec![tail.split("::").next().unwrap_or(tail).trim().to_string()],
            };
            Named {
                module: Some((module.trim().to_string(), symbols)),
                root_symbols: Vec::new(),
            }
        }
        // `crate::Symbol` — a crate-root item, same class as a brace import.
        None => Named {
            module: None,
            root_symbols: vec![rest.trim().to_string()],
        },
    }
}

fn allowed_root_symbols() -> BTreeSet<&'static str> {
    EFFECT_VOCABULARY
        .iter()
        .chain(SHARED_VOCABULARY)
        .chain(CHECKER_OWNED)
        .copied()
        .chain(RESIDUAL_ROOT_SYMBOLS.iter().map(|(symbol, _)| *symbol))
        .collect()
}

fn module_allowance(module: &str) -> Option<&'static ModuleAllowance> {
    ALLOWED_MODULE_PATHS
        .iter()
        .find(|allowance| allowance.module == module)
}

fn violations() -> Vec<String> {
    let symbols = allowed_root_symbols();
    let mut found = Vec::new();
    let mut sources = Vec::new();
    rust_sources(&native_analysis_root(), &mut sources);
    for path in &sources {
        for import in gated_imports(path) {
            let named = classify(&import.statement);
            if let Some((module, taken)) = named.module {
                match module_allowance(&module) {
                    None => found.push(format!(
                        "{}: engine module `crate::{module}` is not allowed from a \
                         checker; consume the information through the effect \
                         payload, or add `{module}` to ALLOWED_MODULE_PATHS with a \
                         reason",
                        import.file
                    )),
                    Some(allowance) if !allowance.symbols.is_empty() => {
                        for symbol in &taken {
                            if !allowance.symbols.contains(&symbol.as_str()) {
                                found.push(format!(
                                    "{}: `{symbol}` is a new reach into the \
                                     residually-allowed engine module \
                                     `crate::{module}`; effect payloads must be \
                                     imported from `crate::effect`",
                                    import.file
                                ));
                            }
                        }
                    }
                    Some(_) => {}
                }
            }
            for symbol in named.root_symbols {
                if !symbols.contains(symbol.as_str()) {
                    found.push(format!(
                        "{}: crate-root symbol `{symbol}` is not in the effect \
                         vocabulary, shared vocabulary, checker-owned set, or the \
                         residual allowlist",
                        import.file
                    ));
                }
            }
        }
    }
    found
}

#[test]
fn checkers_import_only_effect_and_shared_vocabulary() {
    let found = violations();
    assert!(
        found.is_empty(),
        "checker-import gate failed ({} violations):\n  {}",
        found.len(),
        found.join("\n  ")
    );
}

/// Vacuity guard. A gate that scans nothing passes everything, so pin that the
/// scanner actually reaches the checker sources and their real import blocks.
#[test]
fn gate_actually_scans_the_checker_sources() {
    let mut sources = Vec::new();
    rust_sources(&native_analysis_root(), &mut sources);
    assert!(
        sources.len() >= 10,
        "expected the native_analysis tree, found {} files",
        sources.len()
    );

    let mut statements = 0usize;
    let mut root_symbols = BTreeSet::new();
    let mut modules = BTreeSet::new();
    for path in &sources {
        for import in gated_imports(path) {
            statements += 1;
            let named = classify(&import.statement);
            if let Some((module, _)) = named.module {
                modules.insert(module);
            }
            root_symbols.extend(named.root_symbols);
        }
    }
    assert!(
        statements >= 20,
        "expected the checker import blocks, found {statements} statements"
    );
    assert!(
        root_symbols.len() >= 100,
        "expected the crate-root facade imports, found {}",
        root_symbols.len()
    );
    // The two anchors that must always be present: the effect vocabulary the
    // boundary is made of, and the module the payload types now come from.
    assert!(
        root_symbols.contains("OperationEffect"),
        "scanner missed the delivered effect type"
    );
    assert!(
        modules.contains("effect"),
        "scanner missed the effect-vocabulary module import"
    );
}

/// The classifier must reject an engine-internal reach. Without this, a broken
/// parser would silently make the gate vacuous.
#[test]
fn classifier_rejects_engine_internals() {
    let symbols = allowed_root_symbols();

    // A module nobody allowlisted.
    let named = classify("use crate::kernel_engine::KernelEngine;");
    let (module, taken) = named.module.expect("module path");
    assert_eq!(module, "kernel_engine");
    assert_eq!(taken, vec!["KernelEngine"]);
    assert!(
        module_allowance("kernel_engine").is_none(),
        "reaching into the engine core must not be allowed"
    );

    // The regression this gate exists to stop: taking an effect payload from
    // `crate::runtime` again instead of from the effect vocabulary. `runtime`
    // is residually allowed for the python shims, so only the pinned symbol
    // set makes this bite.
    let named = classify("use crate::runtime::{PhysicalMbarrierArrivePlan};");
    let (module, taken) = named.module.expect("module path");
    let allowance = module_allowance(&module).expect("runtime is residually allowed");
    assert!(
        !allowance.symbols.contains(&taken[0].as_str()),
        "an effect payload must not be importable from crate::runtime"
    );
    assert!(
        allowance.symbols.contains(&"LaunchSelection"),
        "the python shim's launch surface stays allowed"
    );

    let named = classify("use crate::{DynamicOpId, PhysicalBarrierHub};");
    assert_eq!(
        named.root_symbols,
        vec!["DynamicOpId", "PhysicalBarrierHub"]
    );
    assert!(
        symbols.contains("DynamicOpId"),
        "shared vocabulary is allowed"
    );
    assert!(
        !symbols.contains("PhysicalBarrierHub"),
        "an engine barrier hub must not be importable by a checker"
    );

    // Multi-line blocks must parse the same as single-line ones.
    let named = classify("use crate::{ DynamicOpId, WarpMask, };");
    assert_eq!(named.root_symbols, vec!["DynamicOpId", "WarpMask"]);
}

/// Every allowlist entry must carry a reason, and must still be needed.
#[test]
fn allowlist_entries_are_justified_and_live() {
    for allowance in ALLOWED_MODULE_PATHS {
        assert!(
            allowance.reason.len() > 20,
            "module allowlist entry `{}` needs a real reason",
            allowance.module
        );
    }
    for (symbol, reason) in RESIDUAL_ROOT_SYMBOLS {
        assert!(
            reason.len() > 20,
            "residual allowlist entry `{symbol}` needs a real reason"
        );
    }

    // Ratchet: an allowlist entry nothing imports any more must be deleted, so
    // the list cannot silently accumulate permission for coupling that is gone.
    let mut sources = Vec::new();
    rust_sources(&native_analysis_root(), &mut sources);
    let mut used_modules = BTreeSet::new();
    let mut used_symbols = BTreeSet::new();
    for path in &sources {
        for import in gated_imports(path) {
            let named = classify(&import.statement);
            if let Some((module, _)) = named.module {
                used_modules.insert(module);
            }
            used_symbols.extend(named.root_symbols);
        }
    }
    let stale_modules = ALLOWED_MODULE_PATHS
        .iter()
        .map(|allowance| allowance.module)
        .filter(|module| !used_modules.contains(*module))
        .collect::<Vec<_>>();
    assert!(
        stale_modules.is_empty(),
        "ALLOWED_MODULE_PATHS entries no checker imports any more — delete them: {stale_modules:?}"
    );
    let stale_symbols = RESIDUAL_ROOT_SYMBOLS
        .iter()
        .map(|(symbol, _)| *symbol)
        .filter(|symbol| !used_symbols.contains(*symbol))
        .collect::<Vec<_>>();
    assert!(
        stale_symbols.is_empty(),
        "RESIDUAL_ROOT_SYMBOLS entries no checker imports any more — delete them: {stale_symbols:?}"
    );
}
