//! Racecheck-owned native mode and launch analysis.

mod global_race;
mod tcgen_fence;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::thread::JoinHandle;

use crate::engine_mode::{
    CachedGlobalReadAccess, CachedGlobalReadFinish, GlobalMemoryProgress,
    GlobalMemoryProgressSnapshot,
};
use crate::physical_access::{
    coalesce_physical_access_batches, CompactPhysicalAccessBatch, ProxyMemoryDomain,
};
use crate::race_shadow::{
    tracks_race_conflicts, CompactDirectGeometry, LaneFrontier, PhysicalRaceOrderingFailure,
    RaceAsyncIssueValidation, RaceClockedBatchValidation, RaceCompactBatchValidation,
    RaceDirectSegment, RaceLaneOrder, SharedClockFrontier, SharedLaneFrontiers,
};
// Effect payloads, imported from the effect vocabulary rather than from the
// engine internals that build them.
use crate::effect::{
    TcgenFenceKind, TcgenPipelineOperation, TcgenWorkIssue, TcgenWorkKind, TcgenWorkSet,
};
use crate::transactional_interval_map::TransactionalIntervalMap;
use crate::{
    profile_count, AnalysisGapEffect, AnalysisGapKind,
    AsyncGroupCompletionActionId, AsyncGroupMilestone, AsyncTokenId, BarrierClockPayload,
    CheckerLaunchContext, ClusterBarrierId, ClusterBarrierParticipantExitEvidence,
    CompletionActionEffect, CompletionEffect, DynamicOpId, EngineError, EngineModeImpl,
    ExecutionReport, LanePhysicalAccess, LaunchTopology, MemoryAccessSemantics, OperationContext,
    OperationEffect, OperationKind, PhysicalAccessBatch, PhysicalAccessDescriptor,
    PhysicalAccessKind, PhysicalAccessSpace, PhysicalAllocationId, PhysicalBarrierId,
    PhysicalByteSpan, PhysicalCompletionActionId, PhysicalCompletionKind, PhysicalRaceFinding,
    PhysicalRaceKind, PhysicalRaceWitness, ProfileKind, ProfileTimer, RaceBatchValidation,
    RaceShadow, RaceShadowError, RaceVectorClock, ResolvedTransitionLog, SyncCheckLaunchState,
    SyncCheckMode, SyncCheckResult, SyncCheckStatus, WarpMask, WARP_SIZE,
};
pub(crate) use global_race::scope_covers_warps;
use global_race::{
    tcgen_publication_required, GlobalClockFrontier, GlobalFloor, GlobalRaceShared, GlobalRaceState,
};
pub use global_race::{
    DeclaredWordBypassDiagnostic, GlobalActorRelation, GlobalScopeMismatchDiagnostic,
    UndeclaredProtocolWordDiagnostic,
};
use tcgen_fence::{
    TcgenComponentFrontier, TcgenFenceFrontier, TcgenLaneFrontiers, TcgenPipelineDescriptor,
};

// How much effect journal Racecheck's synccheck peer retains in compact mode.
// Compact Racecheck consumes the strict protocol state and exact effect counts,
// but its normal report does not need an unbounded replay journal. Small
// diagnostics retain their full effects; large launches switch to the same
// exact-count summary used by Synccheck's bounded diagnostic payload. The
// policy belongs to the peer that chooses it, not to the peer that obeys it.
const RACECHECK_COMPACT_EFFECT_DIAGNOSTIC_LIMIT: u64 = 64 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaceCheckStatus {
    Clean,
    Review,
    Incomplete,
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RaceCheckIncompleteReason {
    /// A write escaped the initial allocation set, so earlier reads may have
    /// been omitted. The native caller must repeat with the complete set.
    GlobalWriteSeedIncomplete,
    AnalysisGap {
        operation: DynamicOpId,
        kind: AnalysisGapKind,
    },
    EffectCommitUnobserved {
        operation: DynamicOpId,
        effect: &'static str,
    },
    BarrierGenerationUnavailable {
        operation: DynamicOpId,
        barrier_id: PhysicalBarrierId,
    },
    BarrierPayloadUnavailable {
        operation: DynamicOpId,
        barrier_id: PhysicalBarrierId,
        generation: u64,
    },
    /// A wait on a declared word left its loop on a value no write the
    /// protocol made to that word produces, so the exit was not caused by the
    /// protocol and no edge could be built from it. Either the simulated loop
    /// budget ran out, or something wrote the word outside the primitive --
    /// which the address claim reports separately.
    DeclaredWordWaitUnexplained {
        operation: DynamicOpId,
    },
    /// One word took more writes than its history holds
    /// (`RACECHECK_MAX_DECLARED_WORD_WRITES`). A wait names a position in that
    /// history, so a truncated record cannot answer which write it accepted.
    DeclaredWordHistoryTruncated {
        operation: DynamicOpId,
    },
    /// A strong global write landed on a word some wait declares, but its width
    /// is not one `wait_until` can poll (4 or 8 bytes), so its value never
    /// entered the word's history. The wait names a position in that history,
    /// and a record missing a publication would make a published protocol look
    /// unpublished, so the claim fails closed here instead.
    DeclaredWordWriteUnrecorded {
        operation: DynamicOpId,
    },
    /// A lane that appeared after the global floor collector retired frontier
    /// entries accessed global memory before observing every retired epoch;
    /// a race against a retired record could not be checked.
    RetiredRecordsUnobserved {
        operation: DynamicOpId,
    },
    /// The launch produced more distinct conflicting byte ranges than the
    /// retained set holds (`RACECHECK_MAX_RETAINED_FINDINGS`); `dropped` of
    /// them were counted and not kept.
    FindingsTruncated {
        retained: usize,
        dropped: u64,
    },
    ShadowRejected {
        operation: DynamicOpId,
        reason: String,
    },
    AsyncPayloadAccessUnmodeled {
        operation: DynamicOpId,
    },
    GlobalMemoryModelUnsupported {
        operation: DynamicOpId,
        kind: &'static str,
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RaceCheckAccessRecord {
    operation: DynamicOpId,
    descriptor: PhysicalAccessDescriptor,
    logical_buffer: Option<Box<str>>,
    active_lane_count: usize,
    lanes: Box<[LanePhysicalAccess]>,
}

impl RaceCheckAccessRecord {
    pub const fn operation(&self) -> &DynamicOpId {
        &self.operation
    }

    pub const fn descriptor(&self) -> PhysicalAccessDescriptor {
        self.descriptor
    }

    pub fn logical_buffer(&self) -> Option<&str> {
        self.logical_buffer.as_deref()
    }

    pub const fn active_lane_count(&self) -> usize {
        self.active_lane_count
    }

    pub fn lanes(&self) -> &[LanePhysicalAccess] {
        &self.lanes
    }
}

/// An HB-ordered read through one logical name observed bytes last written
/// through a different logical name on the same exact physical allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AliasStaleReadAdvisory {
    reader_buffer: Box<str>,
    writer_buffer: Box<str>,
    space: PhysicalAccessSpace,
    allocation: PhysicalAllocationId,
    overlaps: Box<[PhysicalByteSpan]>,
    reader_operation: DynamicOpId,
    writer_operation: DynamicOpId,
    occurrences: usize,
}

impl AliasStaleReadAdvisory {
    pub fn reader_buffer(&self) -> &str {
        &self.reader_buffer
    }

    pub fn writer_buffer(&self) -> &str {
        &self.writer_buffer
    }

    pub const fn space(&self) -> PhysicalAccessSpace {
        self.space
    }

    pub const fn allocation(&self) -> PhysicalAllocationId {
        self.allocation
    }

    pub fn overlaps(&self) -> &[PhysicalByteSpan] {
        &self.overlaps
    }

    pub const fn reader_operation(&self) -> &DynamicOpId {
        &self.reader_operation
    }

    pub const fn writer_operation(&self) -> &DynamicOpId {
        &self.writer_operation
    }

    pub const fn occurrences(&self) -> usize {
        self.occurrences
    }
}

/// The racecheck peer's own report material, before it is combined with the
/// synccheck peer's verdict.
struct RaceVerdictInputs<'a> {
    /// Statuses already folded from sub-results, when combining clusters.
    /// Both false for a single launch.
    saw_error: bool,
    saw_incomplete: bool,
    saw_review: bool,
    findings: &'a [PhysicalRaceFinding],
    scope_diagnostics: &'a [GlobalScopeMismatchDiagnostic],
    declared_word_bypasses: &'a [DeclaredWordBypassDiagnostic],
    advisories: &'a [AliasStaleReadAdvisory],
    incomplete_reasons: &'a [RaceCheckIncompleteReason],
}

/// Combine the two peer observers' verdicts into the racecheck verdict.
///
/// A racecheck run reports synccheck's findings as well as its own — that is
/// product behavior, and de-embedding the two checkers does not change it.
/// What changes is that the fold happens once, here, at report time, instead
/// of being written inline wherever a result is built.
///
/// Precedence: a synchronization error or any non-review race finding is an
/// error; otherwise incompleteness on either side; otherwise a race finding, an
/// or alias advisory is a review; otherwise clean.
fn combine_peer_verdicts(sync: SyncCheckStatus, race: &RaceVerdictInputs<'_>) -> RaceCheckStatus {
    let has_error_finding = race
        .findings
        .iter()
        .any(|finding| !finding.requires_unwaited_tmem_load_review());
    if race.saw_error
        || sync == SyncCheckStatus::Error
        || has_error_finding
        || !race.scope_diagnostics.is_empty()
        || !race.declared_word_bypasses.is_empty()
    {
        RaceCheckStatus::Error
    } else if race.saw_incomplete
        || sync == SyncCheckStatus::Incomplete
        || !race.incomplete_reasons.is_empty()
    {
        RaceCheckStatus::Incomplete
    } else if race.saw_review
        || !race.findings.is_empty()
        || !race.advisories.is_empty()
    {
        RaceCheckStatus::Review
    } else {
        RaceCheckStatus::Clean
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RaceCheckResult {
    status: RaceCheckStatus,
    sync: SyncCheckResult,
    findings: Box<[PhysicalRaceFinding]>,
    scope_diagnostics: Box<[GlobalScopeMismatchDiagnostic]>,
    declared_word_bypasses: Box<[DeclaredWordBypassDiagnostic]>,
    /// Internal evidence for hints on proven races, never a standalone finding
    /// or a contributor to the verdict.
    undeclared_protocol_words: Box<[UndeclaredProtocolWordDiagnostic]>,
    advisories: Box<[AliasStaleReadAdvisory]>,
    access_count: usize,
    accesses_complete: bool,
    accesses: Box<[RaceCheckAccessRecord]>,
    incomplete_reasons: Box<[RaceCheckIncompleteReason]>,
    global_memory_model_enabled: bool,
}

impl RaceCheckResult {
    pub fn merge_cluster_results(results: &[Self]) -> Self {
        let sync_results = results
            .iter()
            .map(|result| result.sync.clone())
            .collect::<Vec<_>>();
        let sync = SyncCheckResult::merge_cluster_results(&sync_results);
        let mut findings = Vec::new();
        let mut scope_diagnostics = Vec::new();
        let mut declared_word_bypasses = Vec::new();
        let mut undeclared_protocol_words = Vec::new();
        let mut advisories = Vec::new();
        let mut accesses = Vec::new();
        let mut incomplete_reasons = Vec::new();
        let mut access_count = 0_usize;
        let mut accesses_complete = true;
        let mut saw_error = false;
        let mut saw_incomplete = false;
        let mut saw_review = false;
        let mut global_memory_model_enabled = false;

        for result in results {
            global_memory_model_enabled |= result.global_memory_model_enabled;
            saw_error |= result.status == RaceCheckStatus::Error;
            saw_incomplete |= result.status == RaceCheckStatus::Incomplete;
            saw_review |= result.status == RaceCheckStatus::Review;
            append_report_findings(&mut findings, result.findings.iter().cloned());
            for diagnostic in result.scope_diagnostics.iter().cloned() {
                if !scope_diagnostics.contains(&diagnostic) {
                    scope_diagnostics.push(diagnostic);
                }
            }
            for diagnostic in result.undeclared_protocol_words.iter().cloned() {
                if !undeclared_protocol_words.contains(&diagnostic) {
                    undeclared_protocol_words.push(diagnostic);
                }
            }
            for diagnostic in result.declared_word_bypasses.iter().cloned() {
                if !declared_word_bypasses.contains(&diagnostic) {
                    declared_word_bypasses.push(diagnostic);
                }
            }
            append_report_advisories(&mut advisories, result.advisories.iter().cloned());
            access_count = access_count.saturating_add(result.access_count);
            accesses_complete &= result.accesses_complete;
            accesses.extend(result.accesses.iter().cloned());
            for reason in result.incomplete_reasons.iter().cloned() {
                push_unique_incomplete(&mut incomplete_reasons, reason);
            }
        }

        let status = combine_peer_verdicts(
            sync.status(),
            &RaceVerdictInputs {
                saw_error,
                saw_incomplete,
                saw_review,
                findings: &findings,
                scope_diagnostics: &scope_diagnostics,
                declared_word_bypasses: &declared_word_bypasses,
                advisories: &advisories,
                incomplete_reasons: &incomplete_reasons,
            },
        );
        Self {
            status,
            sync,
            findings: findings.into_boxed_slice(),
            scope_diagnostics: scope_diagnostics.into_boxed_slice(),
            declared_word_bypasses: declared_word_bypasses.into_boxed_slice(),
            undeclared_protocol_words: undeclared_protocol_words.into_boxed_slice(),
            advisories: advisories.into_boxed_slice(),
            access_count,
            accesses_complete,
            accesses: accesses.into_boxed_slice(),
            incomplete_reasons: incomplete_reasons.into_boxed_slice(),
            global_memory_model_enabled,
        }
    }

    pub const fn status(&self) -> RaceCheckStatus {
        self.status
    }

    pub const fn sync(&self) -> &SyncCheckResult {
        &self.sync
    }

    pub fn findings(&self) -> &[PhysicalRaceFinding] {
        &self.findings
    }

    pub fn scope_diagnostics(&self) -> &[GlobalScopeMismatchDiagnostic] {
        &self.scope_diagnostics
    }

    pub fn undeclared_protocol_words(&self) -> &[UndeclaredProtocolWordDiagnostic] {
        &self.undeclared_protocol_words
    }

    pub fn declared_word_bypasses(&self) -> &[DeclaredWordBypassDiagnostic] {
        &self.declared_word_bypasses
    }

    pub fn advisories(&self) -> &[AliasStaleReadAdvisory] {
        &self.advisories
    }

    pub const fn access_count(&self) -> usize {
        self.access_count
    }

    pub const fn accesses_complete(&self) -> bool {
        self.accesses_complete
    }

    pub fn accesses(&self) -> &[RaceCheckAccessRecord] {
        &self.accesses
    }

    pub fn incomplete_reasons(&self) -> &[RaceCheckIncompleteReason] {
        &self.incomplete_reasons
    }

    pub const fn global_memory_model_enabled(&self) -> bool {
        self.global_memory_model_enabled
    }
}

/// Racecheck's own observer state: the memory race shadow and its shards.
///
/// Peer to the [`SyncCheckLaunchState`] beside it in
/// [`RaceCheckLaunchState`]. Keeping it a named struct rather than fields
/// inlined next to the synccheck peer is what makes the composition legible:
/// a racecheck launch is one shared context plus two peer observers, not a
/// synccheck with racecheck fields bolted onto it.
/// Completions between the first global floor collection and launch start;
/// later collections back off geometrically so a launch pays a bounded
/// number of passes over its byte state.
const GLOBAL_FLOOR_GC_FIRST_EVENTS: usize = 4096;

/// Launch-wide trigger state for [`RaceCheckLaunchState::maybe_collect_global_floor`].
struct GlobalFloorGc {
    /// A pass in progress: a floor pass on the worker that took it, or a
    /// clock-node collection on its thread, which clears the flag itself.
    active: Arc<AtomicBool>,
    events: AtomicUsize,
    next: AtomicUsize,
    /// Signature of the last floor a pass walked the byte cells with; an
    /// unchanged floor cannot dominate anything new, so the walk (which
    /// contends with workers for every cell lock) is skipped.
    last_signature: Mutex<Option<(usize, u64, usize, u64)>>,
    /// The thread of the clock-node collection in progress (or of the last
    /// one), joined before the next one starts and when the launch ends.
    collector: Mutex<Option<JoinHandle<()>>>,
}

impl GlobalFloorGc {
    fn new() -> Self {
        Self {
            active: Arc::new(AtomicBool::new(false)),
            events: AtomicUsize::new(0),
            next: AtomicUsize::new(GLOBAL_FLOOR_GC_FIRST_EVENTS),
            last_signature: Mutex::new(None),
            collector: Mutex::new(None),
        }
    }

    fn join_collector(&self) {
        let handle = self
            .collector
            .lock()
            .expect("global racecheck collector handle lock was poisoned")
            .take();
        if let Some(handle) = handle {
            let _ = handle.join();
        }
    }
}

/// Pause between the collector thread's attempts to take every shard's
/// global state at once for the switch of the mark bitmaps.
const CLOCK_NODE_COLLECTOR_RETRY: std::time::Duration = std::time::Duration::from_micros(50);

/// One clock-node collection, on its own thread. The switch of the mark
/// bitmaps and the root snapshot need every shard's global state locked at
/// once, taken with `try_lock` and retried after a pause (the thread never
/// waits on a lock a worker holds, so it cannot take part in a cycle); the
/// mark walk and the sweep run with the workers going.
fn collect_clock_nodes(shards: &[Arc<Mutex<GlobalRaceState>>], shared: &GlobalRaceShared) {
    let roots = loop {
        let mut guards = Vec::with_capacity(shards.len());
        for shard in shards {
            let Ok(guard) = shard.try_lock() else {
                break;
            };
            guards.push(guard);
        }
        if guards.len() == shards.len() {
            let roots = shared.begin_clock_node_mark();
            drop(guards);
            break roots;
        }
        drop(guards);
        std::thread::sleep(CLOCK_NODE_COLLECTOR_RETRY);
    };
    shared.mark_clock_nodes(&roots);
    drop(roots);
    shared.sweep_clock_nodes();
    // The sweep frees node chunks, memo entries and dead lists in bulk;
    // return the freed pages while still on the collector thread, so the
    // launch's peak RSS tracks the live state instead of the heap high-water.
    numsim_fp_env::release_free_heap();
}

struct RaceObserverState {
    global_floor_gc: GlobalFloorGc,
    global_transaction: Box<[RwLock<()>]>,
    global_allocation_epochs: BTreeMap<PhysicalAllocationId, Arc<GlobalAllocationEpoch>>,
    global_shared: Arc<GlobalRaceShared>,
    race_shards: Box<[RaceCheckShard]>,
    first_shard_cluster_id: Option<usize>,
    shard_warps_per_cluster: Option<usize>,
    retain_accesses: bool,
    direct_compact: bool,
    global_memory_model_enabled: bool,
}

/// A racecheck launch: one shared context, two peer observers over one
/// effect stream.
///
/// The engine mode system binds exactly one `LaunchState` per mode
/// (`EngineModeImpl::LaunchState`), so in racecheck mode this type is the
/// single delivery point the runtime knows about. It is not a checker that
/// contains another checker: it holds the two peers side by side, hands each
/// the event it was given (see the `peer_*` methods), and combines their
/// verdicts once at report time (`combine_peer_verdicts`).
pub struct RaceCheckLaunchState {
    /// Launch context shared with the synccheck peer. Racecheck reads its own
    /// launch configuration here, never through the peer.
    context: Arc<CheckerLaunchContext>,
    /// Peer observer: synchronization protocol and causality.
    sync: SyncCheckLaunchState,
    /// Peer observer: physical memory race shadow.
    race: RaceObserverState,
}

const PARALLEL_DROP_MIN_RACE_SHARDS: usize = 8;
const PARALLEL_DROP_RACE_SHARD_WORKERS: usize = 8;

impl Drop for RaceCheckLaunchState {
    fn drop(&mut self) {
        self.race.global_floor_gc.join_collector();
        let shards = std::mem::take(&mut self.race.race_shards).into_vec();
        if shards.len() < PARALLEL_DROP_MIN_RACE_SHARDS {
            drop(shards);
            return;
        }

        let chunk_len = shards.len().div_ceil(PARALLEL_DROP_RACE_SHARD_WORKERS);
        std::thread::scope(|scope| {
            let mut remaining = shards;
            while remaining.len() > chunk_len {
                let tail = remaining.split_off(remaining.len() - chunk_len);
                scope.spawn(move || drop(tail));
            }
            drop(remaining);
        });
    }
}

struct RaceCheckShard {
    global_warp_base: usize,
    global_warp_end: usize,
    uncontrolled_access_count: AtomicUsize,
    compact_global_read_caches: Box<[Mutex<CompactGlobalReadCache>]>,
    state: Mutex<RaceCheckState>,
    /// This shard's actor-side global-memory model (clocks, staged batches,
    /// async tokens, barrier tables); the byte state lives in
    /// `RaceObserverState::global_shared`. Shared with the clock-node
    /// collector thread, which takes every shard's lock for the switch of
    /// the mark bitmaps.
    global: Arc<Mutex<GlobalRaceState>>,
}

#[derive(Default)]
struct GlobalAllocationEpoch {
    epoch: AtomicU64,
    writes_in_flight: AtomicUsize,
    poll_ranges: Mutex<GlobalRangeEpochState>,
}

#[derive(Default)]
struct GlobalRangeEpochState {
    tracked: BTreeMap<(usize, usize), Arc<GlobalRangeEpoch>>,
    active_writes: BTreeMap<(usize, usize), usize>,
    /// Longest tracked range, so a write finds the ranges it overlaps by
    /// start offset instead of scanning every polled range of the
    /// allocation.
    max_tracked_len: usize,
}

impl GlobalRangeEpochState {
    fn tracked_overlapping(
        &self,
        byte_offset: usize,
        byte_end: usize,
    ) -> impl Iterator<Item = &Arc<GlobalRangeEpoch>> {
        let first_start = byte_offset.saturating_sub(self.max_tracked_len);
        self.tracked
            .range((first_start, 0)..(byte_end, 0))
            .filter(move |((_, end), _)| *end > byte_offset)
            .map(|(_, epoch)| epoch)
    }
}

struct GlobalRangeEpoch {
    progress: GlobalMemoryProgress,
    writes_in_flight: AtomicUsize,
}

impl GlobalAllocationEpoch {
    fn tracked_poll_range(&self, byte_offset: usize, byte_end: usize) -> Arc<GlobalRangeEpoch> {
        let mut state = self
            .poll_ranges
            .lock()
            .expect("global poll-range epochs poisoned");
        if let Some(epoch) = state.tracked.get(&(byte_offset, byte_end)) {
            return Arc::clone(epoch);
        }
        let writes_in_flight = state
            .active_writes
            .iter()
            .filter(|((start, end), _)| *start < byte_end && byte_offset < *end)
            .map(|(_, count)| *count)
            .sum();
        let epoch = Arc::new(GlobalRangeEpoch {
            progress: GlobalMemoryProgress::default(),
            writes_in_flight: AtomicUsize::new(writes_in_flight),
        });
        state.max_tracked_len = state.max_tracked_len.max(byte_end - byte_offset);
        state
            .tracked
            .insert((byte_offset, byte_end), Arc::clone(&epoch));
        epoch
    }

    /// Whether a write whose exact span overlaps `[byte_offset, byte_end)`
    /// has begun and not yet committed its Racecheck metadata.
    fn has_write_in_flight(&self, byte_offset: usize, byte_end: usize) -> bool {
        let state = self
            .poll_ranges
            .lock()
            .expect("global poll-range epochs poisoned");
        state
            .active_writes
            .keys()
            .any(|(start, end)| *start < byte_end && byte_offset < *end)
    }

    fn begin_write(&self, byte_offset: usize, byte_end: usize) {
        self.writes_in_flight.fetch_add(1, AtomicOrdering::AcqRel);
        let mut state = self
            .poll_ranges
            .lock()
            .expect("global poll-range epochs poisoned");
        *state
            .active_writes
            .entry((byte_offset, byte_end))
            .or_default() += 1;
        for epoch in state.tracked_overlapping(byte_offset, byte_end) {
            epoch.writes_in_flight.fetch_add(1, AtomicOrdering::AcqRel);
        }
    }

    fn finish_write(&self, byte_offset: usize, byte_end: usize) {
        {
            let mut state = self
                .poll_ranges
                .lock()
                .expect("global poll-range epochs poisoned");
            for epoch in state.tracked_overlapping(byte_offset, byte_end) {
                epoch.progress.record_change();
                let prior = epoch.writes_in_flight.fetch_sub(1, AtomicOrdering::Release);
                debug_assert!(prior > 0);
            }
            let count = state
                .active_writes
                .get_mut(&(byte_offset, byte_end))
                .expect("finished global write was registered");
            *count -= 1;
            if *count == 0 {
                state.active_writes.remove(&(byte_offset, byte_end));
            }
        }
        self.epoch.fetch_add(1, AtomicOrdering::Release);
        let prior = self.writes_in_flight.fetch_sub(1, AtomicOrdering::Release);
        debug_assert!(prior > 0);
    }
}

struct CompactGlobalReadCacheEntry {
    source_op_id: u64,
    descriptor: PhysicalAccessDescriptor,
    lane_spans: Box<[(usize, PhysicalByteSpan)]>,
    allocation_epochs: Box<[(Arc<GlobalAllocationEpoch>, u64)]>,
}

impl CompactGlobalReadCacheEntry {
    fn matches_access(&self, batch: &CompactPhysicalAccessBatch<'_>) -> bool {
        self.source_op_id == batch.operation().id().source_op_id().get()
            && self.descriptor == batch.descriptor()
            && self.lane_spans.iter().copied().eq(batch.lane_spans())
    }

    fn matches_cached_access(&self, access: CachedGlobalReadAccess) -> bool {
        self.source_op_id == access.source_op_id()
            && self.descriptor == access.descriptor()
            && self.lane_spans.as_ref() == [(access.lane(), access.span())].as_slice()
    }
}

#[derive(Default)]
struct CompactGlobalReadCache {
    entry: Option<CompactGlobalReadCacheEntry>,
    pending_sequence: Option<u64>,
    /// Allocation epochs sampled when the pending read began, before its
    /// read-from lookup. An entry admitted with these epochs is invalidated
    /// by any write that finishes after the sample, including one that
    /// commits between the lookup and the entry's admission.
    pending_epochs: Option<Box<[(Arc<GlobalAllocationEpoch>, u64)]>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompactGlobalReadFinish {
    NotCached,
    Stable,
    Reprocess,
}

/// Number of lock stripes the global-memory transaction is spread over;
/// an allocation maps to one stripe, so accesses to different allocations
/// (almost always) run their transactions concurrently.
const GLOBAL_TRANSACTION_STRIPES: usize = 1024;

/// Holds the transaction stripes of one operation's allocations, acquired in
/// ascending stripe order so concurrent transactions never deadlock.
#[derive(Default)]
pub struct GlobalMemoryTransactionGuard<'a> {
    _shared: Vec<RwLockReadGuard<'a, ()>>,
    _exclusive: Vec<RwLockWriteGuard<'a, ()>>,
}

/// Byte granularity of the transaction stripes: two accesses share a stripe
/// only when they touch the same `GLOBAL_TRANSACTION_STRIPE_BYTES` window of
/// one allocation (modulo hash collisions, which only cost concurrency).
const GLOBAL_TRANSACTION_STRIPE_BYTES: usize = 1 << 16;

fn global_transaction_stripes(spans: &[PhysicalByteSpan]) -> Vec<usize> {
    let mut stripes = spans
        .iter()
        .flat_map(|span| {
            let first = span.byte_offset() / GLOBAL_TRANSACTION_STRIPE_BYTES;
            let last =
                (span.byte_end().max(span.byte_offset() + 1) - 1) / GLOBAL_TRANSACTION_STRIPE_BYTES;
            let allocation = span.allocation().get();
            (first..=last).map(move |window| {
                let key = allocation
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(window as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15);
                ((key >> 40) as usize) % GLOBAL_TRANSACTION_STRIPES
            })
        })
        .collect::<Vec<_>>();
    stripes.sort_unstable();
    stripes.dedup();
    stripes
}

struct RaceCheckState {
    topology: Option<LaunchTopology>,
    retain_accesses: bool,
    shadow: RaceShadow,
    lane_shadow: SameWarpLaneShadow,
    pending_direct_segment: Option<PendingDirectSegment>,
    staged_compact_access: Option<StagedCompactAccess>,
    staged_accesses: BTreeMap<DynamicOpId, StagedAccess>,
    staged_async_payloads: BTreeMap<DynamicOpId, StagedAsyncPayload>,
    staged_async_completions: BTreeMap<AsyncTokenId, StagedAsyncCompletion>,
    async_token_clocks: BTreeMap<AsyncTokenId, RaceVectorClock>,
    staged_async_group_issues: BTreeMap<DynamicOpId, StagedAsyncGroupIssue>,
    staged_async_group_completions:
        BTreeMap<AsyncGroupCompletionActionId, StagedAsyncGroupCompletion>,
    async_group_token_clocks: BTreeMap<AsyncTokenId, AsyncGroupTokenClocks>,
    // Arrive-on observes every prior classic copy, including retired groups.
    // This per-issuer frontier contains copy actors only, never issuer history.
    cp_async_completed_payloads: BTreeMap<(usize, usize), BarrierClockPayload>,
    tcgen_work_tokens: BTreeMap<AsyncTokenId, ActiveTcgenWork>,
    reviewed_tcgen_load_tokens: BTreeMap<AsyncTokenId, DynamicOpId>,
    tcgen_pipeline_frontiers:
        BTreeMap<TcgenThreadPipelineKey, BTreeMap<TcgenPipelineDescriptor, RaceVectorClock>>,
    tcgen_thread_epochs: BTreeMap<TcgenThreadKey, u64>,
    tcgen_fenced_frontiers: BTreeMap<TcgenThreadKey, TcgenThreadFenceFrontier>,
    tcgen_published_frontiers: BTreeMap<TcgenThreadKey, TcgenThreadFenceFrontier>,
    tcgen_incoming_frontiers: BTreeMap<TcgenThreadKey, TcgenThreadFenceFrontier>,
    tcgen_wait_frontiers: BTreeMap<TcgenThreadKey, RaceVectorClock>,
    tcgen_capture_cache: BTreeMap<TcgenFenceCaptureKey, TcgenThreadFenceFrontier>,
    tcgen_imported_pipeline_cache: BTreeMap<TcgenPipelineDescriptor, CachedImportedTcgenClock>,
    tcgen_imported_completed_cache: Option<CachedImportedTcgenClock>,
    tcgen_commit_frontiers: BTreeMap<(TcgenThreadPipelineKey, TcgenWorkKind), BarrierClockPayload>,
    staged_arrives: BTreeMap<DynamicOpId, Box<[StagedArrive]>>,
    arrival_completion_payloads: BTreeMap<PhysicalCompletionActionId, ArrivalCompletionPayload>,
    barrier_payloads: BTreeMap<(PhysicalBarrierId, u64), BarrierClockPayload>,
    // Copy completion is observable even by relaxed waits; arrival release
    // history above requires acquire semantics. Neither map duplicates the other.
    barrier_copy_payloads: BTreeMap<(PhysicalBarrierId, u64), BarrierClockPayload>,
    barrier_lane_payloads: BTreeMap<(PhysicalBarrierId, u64), SharedClockFrontier>,
    barrier_tcgen_payloads: BTreeMap<(PhysicalBarrierId, u64), TcgenThreadFenceFrontier>,
    named_barrier_payloads: BTreeMap<(crate::NamedBarrierId, u64), BarrierClockPayload>,
    named_barrier_lane_payloads: BTreeMap<(crate::NamedBarrierId, u64), SharedClockFrontier>,
    named_barrier_tcgen_payloads: BTreeMap<(crate::NamedBarrierId, u64), TcgenThreadFenceFrontier>,
    cluster_barrier_payloads: BTreeMap<(ClusterBarrierId, u64), BarrierClockPayload>,
    cluster_barrier_lane_payloads: BTreeMap<(ClusterBarrierId, u64), SharedClockFrontier>,
    cluster_barrier_tcgen_payloads: BTreeMap<(ClusterBarrierId, u64), TcgenThreadFenceFrontier>,
    findings: Vec<PhysicalRaceFinding>,
    access_count: usize,
    alias_tracker: AliasTracker,
    accesses: Vec<RaceCheckAccessRecord>,
    incomplete_reasons: Vec<RaceCheckIncompleteReason>,
}

struct StagedAccess {
    lane_validation: SameWarpLaneBatchValidation,
    shadow_revision: u64,
    shadow_validation: RaceBatchValidation,
}

struct StagedCompactAccess {
    operation: Arc<DynamicOpId>,
    lane_validation: SameWarpLaneBatchValidation,
    shadow_revision: u64,
    shadow_validation: RaceCompactBatchValidation,
}

struct PendingDirectSegment {
    global_warp_id: usize,
    lane_validation: SameWarpLaneBatchValidation,
    shadow_segment: RaceDirectSegment,
}

struct StagedAsyncPayload {
    compact_batches: Box<[PhysicalAccessBatch]>,
    records: Box<[RaceCheckAccessRecord]>,
    tracked_access_count: usize,
    skipped_access_count: usize,
    token: AsyncTokenId,
    shadow_revision: u64,
    shadow_validation: RaceAsyncIssueValidation,
}

struct StagedAsyncCompletion {
    compact_batches: Box<[PhysicalAccessBatch]>,
    records: Box<[RaceCheckAccessRecord]>,
    tracked_access_count: usize,
    skipped_access_count: usize,
}

struct StagedAsyncGroupIssue {
    tokens: Box<[AsyncTokenId]>,
}

struct ActiveTcgenWork {
    operation: DynamicOpId,
    kind: TcgenWorkKind,
}

#[derive(Clone, Default)]
struct TcgenThreadFenceSnapshot {
    // A before/synchronization/after chain carries all prior fence-eligible
    // TCGEN work (MMA/CP/shift).
    // Descriptors remain attached so the same immutable representation can
    // also feed the unfenced implicit-pipeline path; a thread fence itself
    // does not filter the captured work by descriptor pairing.
    pipeline: BTreeMap<TcgenPipelineDescriptor, RaceVectorClock>,
    // A wait, or a commit whose target mbarrier completed, proves completion
    // and can therefore precede a non-pipelined TCGEN operation.
    completed: Option<RaceVectorClock>,
    // Per-source epochs make repeated barrier generations a constant-time
    // no-op when they carry an immutable frontier already acquired by the
    // destination. A completion upgrade is tracked separately because it
    // strengthens the same captured pipeline without issuing new work.
    lineage: BTreeMap<TcgenThreadKey, u64>,
    completion_lineage: BTreeMap<TcgenThreadKey, u64>,
    // Frontiers received through an execution-ordering handoff remain in the
    // launch-portable representation until a later TCGEN issue actually
    // needs one of their descriptor classes.
    transported: TcgenFenceFrontier,
}

/// Cluster-shard-local TCGEN execution-ordering frontier.
///
/// The contained `RaceVectorClock`s use this shard's async-token registry and
/// therefore never enter launch-wide state. Barrier transport clones this
/// immutable handle; a portable token frontier is produced only at a real
/// global-memory boundary. Each payload stores the join of all acquired facts,
/// using the same descriptor/completion/lineage rules as a local capture. Keeping
/// overlapping per-source snapshots would require repeated dominance scans at
/// every lane and synchronization generation.
#[derive(Clone, Default)]
struct TcgenThreadFenceFrontier {
    snapshot: Arc<TcgenThreadFenceSnapshot>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TcgenFenceCaptureKey {
    thread: TcgenThreadKey,
    commit_cta_group: Option<(u32, TcgenWorkKind)>,
}

struct CachedImportedTcgenClock {
    source: TcgenFenceFrontier,
    clock: RaceVectorClock,
}

#[derive(Clone, Copy, Debug)]
struct ProxyMemoryDomains {
    values: [ProxyMemoryDomain; 3],
    len: u8,
}

impl ProxyMemoryDomains {
    const fn new() -> Self {
        Self {
            values: [ProxyMemoryDomain::Global; 3],
            len: 0,
        }
    }

    fn insert(&mut self, domain: ProxyMemoryDomain) {
        if domain == ProxyMemoryDomain::Other || self.as_slice().contains(&domain) {
            return;
        }
        let index = usize::from(self.len);
        debug_assert!(index < self.values.len());
        self.values[index] = domain;
        self.len += 1;
    }

    fn as_slice(&self) -> &[ProxyMemoryDomain] {
        &self.values[..usize::from(self.len)]
    }
}

/// Same-thread/CTA-group identity shared by all implicit TCGEN pipeline pairs.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TcgenThreadPipelineKey {
    kernel_index: usize,
    global_warp_id: usize,
    active_mask: WarpMask,
    cta_group: u32,
}

/// One simulated PTX thread used by `tcgen05.fence`.
///
/// The fence has no CTA-group qualifier. PTX requires one CTA-group value for
/// all TCGEN instructions in a kernel, so the group remains on issued-work
/// keys and is intentionally absent from the fence identity. Runtime TCGEN
/// forms may collapse a warp-cooperative instruction to its elected issuing
/// lane. Frontiers are stored per lane so a warp-wide fence represents one
/// independent fence per active PTX thread rather than a cross-lane bridge.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TcgenThreadKey {
    kernel_index: usize,
    global_warp_id: usize,
    lane: u8,
}

struct StagedAsyncGroupCompletion {
    records: Box<[RaceCheckAccessRecord]>,
    tracked_access_count: usize,
    skipped_access_count: usize,
    milestone: AsyncGroupMilestone,
    tokens: Box<[AsyncTokenId]>,
    token_clocks: Box<[(AsyncTokenId, RaceVectorClock)]>,
    shadow_revision: u64,
    shadow_validation: RaceClockedBatchValidation,
}

struct AsyncGroupTokenClocks {
    current: RaceVectorClock,
    source_read: Option<RaceVectorClock>,
    full: Option<RaceVectorClock>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WarpLaneOrder {
    common: LaneFrontier,
    observed: [LaneFrontier; WARP_SIZE],
    joined: LaneFrontier,
    incoming_common: SharedClockFrontier,
    incoming_lanes: Option<Box<[SharedClockFrontier; WARP_SIZE]>>,
}

impl Default for WarpLaneOrder {
    fn default() -> Self {
        Self {
            common: [0; WARP_SIZE],
            observed: [[0; WARP_SIZE]; WARP_SIZE],
            joined: [0; WARP_SIZE],
            incoming_common: SharedClockFrontier::default(),
            incoming_lanes: None,
        }
    }
}

impl WarpLaneOrder {
    fn component(&self, lane: usize, source_lane: usize) -> u64 {
        self.common[source_lane].max(self.observed[lane][source_lane])
    }

    fn preview_tick(&self, warp_id: usize, lane: usize) -> Result<LaneEventStamp<'_>, String> {
        let epoch = self
            .component(lane, lane)
            .checked_add(1)
            .ok_or_else(|| format!("lane clock overflowed for warp {warp_id} lane {lane}"))?;
        Ok(LaneEventStamp {
            warp_id,
            lane: lane as u8,
            common: &self.common,
            observed: &self.observed[lane],
            epoch,
        })
    }

    fn tick(&mut self, warp_id: usize, lane: usize) -> Result<(), String> {
        let epoch = self
            .component(lane, lane)
            .checked_add(1)
            .ok_or_else(|| format!("lane clock overflowed for warp {warp_id} lane {lane}"))?;
        self.observed[lane][lane] = epoch;
        self.joined[lane] = self.joined[lane].max(epoch);
        Ok(())
    }

    fn release(&mut self, warp_id: usize, mask: WarpMask) -> Result<LaneFrontier, String> {
        for lane in mask {
            self.tick(warp_id, lane)?;
        }
        if mask.is_full() {
            return Ok(self.joined);
        }
        let mut release = if mask.is_empty() {
            [0; WARP_SIZE]
        } else {
            self.common
        };
        for lane in mask {
            for (released, observed) in release.iter_mut().zip(self.observed[lane]) {
                *released = (*released).max(observed);
            }
        }
        Ok(release)
    }

    fn acquire(
        &mut self,
        warp_id: usize,
        mask: WarpMask,
        release: Option<&LaneFrontier>,
    ) -> Result<(), String> {
        if mask.is_full() {
            if let Some(release) = release {
                for ((common, joined), released) in self
                    .common
                    .iter_mut()
                    .zip(self.joined.iter_mut())
                    .zip(release)
                {
                    *common = (*common).max(*released);
                    *joined = (*joined).max(*common);
                }
            }
            for lane in mask {
                self.tick(warp_id, lane)?;
            }
            return Ok(());
        }
        for lane in mask {
            if let Some(release) = release {
                for ((observed, joined), released) in self.observed[lane]
                    .iter_mut()
                    .zip(self.joined.iter_mut())
                    .zip(release)
                {
                    *observed = (*observed).max(*released);
                    *joined = (*joined).max(*observed);
                }
            }
            self.tick(warp_id, lane)?;
        }
        Ok(())
    }

    fn commit_lane_epoch(&mut self, lane: usize, epoch: u64) {
        debug_assert!(epoch >= self.component(lane, lane));
        self.observed[lane][lane] = epoch;
        self.joined[lane] = self.joined[lane].max(epoch);
    }

    #[cfg(test)]
    fn commit_lane_frontier(&mut self, lane: usize, frontier: LaneFrontier) {
        let common = self.common;
        for (source_lane, ((observed, joined), incoming)) in self.observed[lane]
            .iter_mut()
            .zip(self.joined.iter_mut())
            .zip(frontier)
            .enumerate()
        {
            debug_assert!(incoming >= (*observed).max(common[source_lane]));
            *observed = incoming;
            *joined = (*joined).max(incoming);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LaneEventStamp<'a> {
    warp_id: usize,
    lane: u8,
    common: &'a LaneFrontier,
    observed: &'a LaneFrontier,
    epoch: u64,
}

impl LaneEventStamp<'_> {
    #[cfg(test)]
    fn retained_epoch(&self) -> tests::RetainedLaneEventStamp {
        tests::RetainedLaneEventStamp {
            warp_id: self.warp_id,
            lane: self.lane,
            epoch: self.epoch,
        }
    }

    #[cfg(test)]
    fn observes(&self, prior: &tests::RetainedLaneEventStamp) -> bool {
        let prior_lane = usize::from(prior.lane);
        let observed_epoch = if prior.lane == self.lane {
            self.epoch
        } else {
            self.common[prior_lane].max(self.observed[prior_lane])
        };
        self.warp_id == prior.warp_id && observed_epoch >= prior.epoch
    }

    #[cfg(test)]
    fn materialized_frontier(&self) -> LaneFrontier {
        let mut frontier = std::array::from_fn(|lane| self.common[lane].max(self.observed[lane]));
        frontier[usize::from(self.lane)] = self.epoch;
        frontier
    }
}

struct SameWarpLaneBatchValidation {
    order_index: usize,
    lane_updates: [Option<u64>; WARP_SIZE],
}

impl SameWarpLaneBatchValidation {
    fn merge_order_validation(&mut self, next: Self) {
        debug_assert_eq!(self.order_index, next.order_index);
        for (current, incoming) in self.lane_updates.iter_mut().zip(next.lane_updates) {
            if let Some(incoming) = incoming {
                *current = Some(current.unwrap_or(0).max(incoming));
            }
        }
    }
}

#[derive(Clone)]
struct SameWarpLaneShadow {
    global_warp_base: usize,
    orders: Vec<WarpLaneOrder>,
}

impl SameWarpLaneShadow {
    #[cfg(test)]
    fn new(warp_count: usize) -> Self {
        Self::for_warp_range(0, warp_count)
    }

    fn for_warp_range(global_warp_base: usize, warp_count: usize) -> Self {
        Self {
            global_warp_base,
            orders: vec![WarpLaneOrder::default(); warp_count],
        }
    }

    fn order_index(&self, warp_id: usize) -> Result<usize, String> {
        warp_id
            .checked_sub(self.global_warp_base)
            .filter(|order_index| *order_index < self.orders.len())
            .ok_or_else(|| {
                format!(
                    "warp {warp_id} is outside lane-shadow warp range {}..{}",
                    self.global_warp_base,
                    self.global_warp_base.saturating_add(self.orders.len())
                )
            })
    }

    fn race_lane_order(&self, warp_id: usize) -> Result<RaceLaneOrder<'_>, String> {
        let order_index = self.order_index(warp_id)?;
        Ok(RaceLaneOrder::new(
            warp_id,
            &self.orders[order_index].common,
            &self.orders[order_index].observed,
        )
        .with_incoming(
            self.global_warp_base,
            &self.orders[order_index].incoming_common,
            self.orders[order_index].incoming_lanes.as_deref(),
        ))
    }

    fn validate_order_batch(
        &self,
        batch: &PhysicalAccessBatch,
    ) -> Result<SameWarpLaneBatchValidation, String> {
        self.preview_lanes(
            batch.operation().id().global_warp_id(),
            batch.lanes().iter().map(|lane| lane.provenance().lane()),
        )
    }

    fn validate_compact_order_batch(
        &self,
        batch: &CompactPhysicalAccessBatch<'_>,
    ) -> Result<SameWarpLaneBatchValidation, String> {
        // Lane clocks do not need compact memory geometry or physical spans.
        self.preview_lanes(
            batch.operation().id().global_warp_id(),
            batch.operation().active_mask(),
        )
    }

    fn preview_lanes(
        &self,
        warp_id: usize,
        lanes: impl IntoIterator<Item = usize>,
    ) -> Result<SameWarpLaneBatchValidation, String> {
        let order_index = self.order_index(warp_id)?;
        let order = &self.orders[order_index];
        let mut lane_updates = [None; WARP_SIZE];
        for lane in lanes {
            let stamp = order.preview_tick(warp_id, lane)?;
            lane_updates[lane].get_or_insert(stamp.epoch);
        }
        Ok(SameWarpLaneBatchValidation {
            order_index,
            lane_updates,
        })
    }

    fn commit_batch(&mut self, validation: SameWarpLaneBatchValidation) {
        for (lane, epoch) in validation.lane_updates.into_iter().enumerate() {
            if let Some(epoch) = epoch {
                self.orders[validation.order_index].commit_lane_epoch(lane, epoch);
            }
        }
    }

    fn barrier_release(
        &mut self,
        warp_id: usize,
        mask: WarpMask,
    ) -> Result<SharedClockFrontier, String> {
        let order_index = self.order_index(warp_id)?;
        let order = &mut self.orders[order_index];
        let mut release = SharedClockFrontier::single(warp_id, order.release(warp_id, mask)?);
        if !mask.is_empty() {
            release.merge(&order.incoming_common);
            if let Some(incoming) = &order.incoming_lanes {
                let mut merged: Vec<&SharedClockFrontier> = Vec::new();
                for lane in mask {
                    if merged.iter().any(|prior| prior.shares_storage_with(&incoming[lane])) {
                        continue;
                    }
                    release.merge(&incoming[lane]);
                    merged.push(&incoming[lane]);
                }
            }
        }
        Ok(release)
    }

    fn barrier_acquire(
        &mut self,
        warp_id: usize,
        mask: WarpMask,
        payload: Option<&SharedClockFrontier>,
    ) -> Result<(), String> {
        let order_index = self.order_index(warp_id)?;
        let shard_base = self.global_warp_base;
        let shard_warps = self.orders.len();
        let order = &mut self.orders[order_index];
        if let Some(payload) = payload.filter(|payload| !payload.is_empty()) {
            if mask.is_full() {
                order
                    .incoming_common
                    .merge_within(payload, shard_base, shard_warps);
            } else if !mask.is_empty() {
                let incoming = order.incoming_lanes.get_or_insert_with(|| {
                    Box::new(std::array::from_fn(|_| SharedClockFrontier::default()))
                });
                // Equal immutable inputs have the same exact join. Compute it
                // once and retain sharing across lanes acquiring this payload.
                let mut groups: Vec<(usize, u32)> = Vec::new();
                for lane in mask {
                    match groups.iter_mut().find(|(first, _)| {
                        incoming[*first].shares_storage_with(&incoming[lane])
                    }) {
                        Some((_, lanes)) => *lanes |= 1 << lane,
                        None => groups.push((lane, 1 << lane)),
                    }
                }
                for (first, lanes) in groups {
                    incoming[first].merge_within(payload, shard_base, shard_warps);
                    for lane in WarpMask::from_bits(lanes & !(1 << first)) {
                        incoming[lane] = incoming[first].clone();
                    }
                }
            }
        }
        order.acquire(
            warp_id,
            mask,
            payload.and_then(|payload| payload.release_for(warp_id)),
        )
    }

    fn warp_sync(&mut self, warp_id: usize, mask: WarpMask) -> Result<(), String> {
        let payload = self.barrier_release(warp_id, mask)?;
        if !mask.is_full() {
            return self.barrier_acquire(warp_id, mask, Some(&payload));
        }
        let index = self.order_index(warp_id)?;
        let order = &mut self.orders[index];
        order.acquire(warp_id, mask, payload.release_for(warp_id))?;
        // A full-warp release already joins the common history and every lane's
        // history. All lanes now acquire that exact join: keep it once, without
        // rejoining the old common history or retaining redundant lane copies.
        order.incoming_common = payload;
        order.incoming_lanes = None;
        Ok(())
    }
}

#[cfg(test)]
fn same_warp_conflict_kinds(
    prior: PhysicalAccessKind,
    current: PhysicalAccessKind,
) -> impl Iterator<Item = PhysicalRaceKind> {
    // An RMW may supply either side of a valid read/write conflict witness.
    // Witness selection is not part of the compressed shadow's contract.
    if prior == PhysicalAccessKind::AtomicReadModifyWrite
        && current == PhysicalAccessKind::AtomicReadModifyWrite
    {
        [None; 3]
    } else {
        [
            (prior.writes() && current.reads()).then_some(PhysicalRaceKind::WriteRead),
            (prior.reads() && current.writes()).then_some(PhysicalRaceKind::ReadWrite),
            (prior.writes() && current.writes()).then_some(PhysicalRaceKind::WriteWrite),
        ]
    }
    .into_iter()
    .flatten()
}

#[derive(Clone, Copy)]
struct StagedArrive {
    barrier_id: PhysicalBarrierId,
    generation: u64,
    warp_id: usize,
    active_mask: WarpMask,
}

#[derive(Clone)]
struct ArrivalCompletionPayload {
    operation: DynamicOpId,
    barrier_id: PhysicalBarrierId,
    generation: u64,
    copy_completion: bool,
    payload: BarrierClockPayload,
    lane_payload: SharedClockFrontier,
    tcgen_payload: TcgenThreadFenceFrontier,
}

fn tcgen_pipeline_descriptor(issue: &TcgenWorkIssue) -> TcgenPipelineDescriptor {
    TcgenPipelineDescriptor::new(
        issue.pipeline_operation(),
        issue.cta_group(),
        issue.mma_pipeline_class(),
    )
}

fn tcgen_thread_pipeline_key(
    operation: &OperationContext,
    cta_group: u32,
) -> TcgenThreadPipelineKey {
    TcgenThreadPipelineKey {
        kernel_index: operation.id().kernel_index(),
        global_warp_id: operation.id().global_warp_id(),
        active_mask: operation.active_mask(),
        cta_group,
    }
}

/// Merge acquired TCGEN frontiers into a race state the caller already holds.
///
/// `after_effect` takes the shard's race lock once and keeps it for the whole
/// effect match, so a branch inside it must not reach for the same lock again.
fn merge_tcgen_acquisitions_into(
    state: &mut RaceCheckState,
    operation: &OperationContext,
    acquisitions: TcgenLaneFrontiers,
) {
    for (lane, frontier) in acquisitions {
        state
            .tcgen_incoming_frontiers
            .entry(tcgen_thread_key(operation, lane))
            .or_default()
            .merge_transported(&frontier);
    }
}

fn tcgen_thread_key(operation: &OperationContext, lane: usize) -> TcgenThreadKey {
    TcgenThreadKey {
        kernel_index: operation.id().kernel_index(),
        global_warp_id: operation.id().global_warp_id(),
        lane: u8::try_from(lane).expect("warp lane fits in u8"),
    }
}

fn tcgen_pipeline_key(issue: &TcgenWorkIssue) -> TcgenThreadPipelineKey {
    tcgen_thread_pipeline_key(issue.operation(), issue.cta_group())
}

fn merge_tcgen_clock(
    target: &mut Option<RaceVectorClock>,
    incoming: &RaceVectorClock,
) -> Result<(), EngineError> {
    if let Some(target) = target {
        target
            .merge(incoming)
            .map_err(|error| EngineError::message(error.to_string()))?;
    } else {
        *target = Some(incoming.clone());
    }
    Ok(())
}

fn merge_tcgen_clock_entry<K: Ord>(
    frontiers: &mut BTreeMap<K, RaceVectorClock>,
    key: K,
    incoming: &RaceVectorClock,
) -> Result<(), EngineError> {
    match frontiers.entry(key) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(incoming.clone());
        }
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            entry
                .get_mut()
                .merge(incoming)
                .map_err(|error| EngineError::message(error.to_string()))?;
        }
    }
    Ok(())
}

fn merge_tcgen_descriptor_frontier(
    frontiers: &mut BTreeMap<TcgenPipelineDescriptor, RaceVectorClock>,
    descriptor: TcgenPipelineDescriptor,
    incoming: &RaceVectorClock,
) -> Result<(), EngineError> {
    match frontiers.entry(descriptor) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(incoming.clone());
        }
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            entry
                .get_mut()
                .merge(incoming)
                .map_err(|error| EngineError::message(error.to_string()))?;
        }
    }
    Ok(())
}

impl TcgenThreadFenceFrontier {
    fn snapshot_ptr(&self) -> usize {
        Arc::as_ptr(&self.snapshot) as usize
    }

    fn pipeline(&self) -> &BTreeMap<TcgenPipelineDescriptor, RaceVectorClock> {
        &self.snapshot.pipeline
    }

    fn completed(&self) -> Option<&RaceVectorClock> {
        self.snapshot.completed.as_ref()
    }

    fn transported(&self) -> &TcgenFenceFrontier {
        &self.snapshot.transported
    }

    fn is_empty(&self) -> bool {
        self.snapshot.pipeline.is_empty()
            && self.snapshot.completed.is_none()
            && self.snapshot.transported.is_empty()
    }

    fn merge_pipeline(
        &mut self,
        descriptor: TcgenPipelineDescriptor,
        clock: &RaceVectorClock,
    ) -> Result<(), EngineError> {
        merge_tcgen_descriptor_frontier(
            &mut Arc::make_mut(&mut self.snapshot).pipeline,
            descriptor,
            clock,
        )
    }

    fn merge_completed(&mut self, clock: &RaceVectorClock) -> Result<(), EngineError> {
        merge_tcgen_clock(&mut Arc::make_mut(&mut self.snapshot).completed, clock)
    }

    fn merge_transported(&mut self, frontier: &TcgenFenceFrontier) {
        Arc::make_mut(&mut self.snapshot)
            .transported
            .merge(frontier);
    }

    fn mark_source(&mut self, thread: TcgenThreadKey, epoch: u64) {
        let current = Arc::make_mut(&mut self.snapshot)
            .lineage
            .entry(thread)
            .or_default();
        *current = (*current).max(epoch);
    }

    fn merge(&mut self, other: &Self) -> Result<(), EngineError> {
        if Arc::ptr_eq(&self.snapshot, &other.snapshot) || other.is_empty() {
            return Ok(());
        }
        if self.is_empty() {
            self.snapshot = Arc::clone(&other.snapshot);
            return Ok(());
        }
        let untracked_local = (!other.snapshot.pipeline.is_empty()
            || other.snapshot.completed.is_some())
            && other.snapshot.lineage.is_empty()
            && other.snapshot.completion_lineage.is_empty();
        let adds_local = untracked_local
            || frontier_epochs_advance(&self.snapshot.lineage, &other.snapshot.lineage)
            || frontier_epochs_advance(
                &self.snapshot.completion_lineage,
                &other.snapshot.completion_lineage,
            );
        if !adds_local {
            Arc::make_mut(&mut self.snapshot)
                .transported
                .merge(&other.snapshot.transported);
            return Ok(());
        }
        let snapshot = Arc::make_mut(&mut self.snapshot);
        for (descriptor, clock) in &other.snapshot.pipeline {
            merge_tcgen_descriptor_frontier(&mut snapshot.pipeline, descriptor.clone(), clock)?;
        }
        if let Some(completed) = &other.snapshot.completed {
            merge_tcgen_clock(&mut snapshot.completed, completed)?;
        }
        merge_frontier_epochs(&mut snapshot.lineage, &other.snapshot.lineage);
        merge_frontier_epochs(
            &mut snapshot.completion_lineage,
            &other.snapshot.completion_lineage,
        );
        snapshot.transported.merge(&other.snapshot.transported);
        Ok(())
    }

    fn upgraded_to_completed(&self) -> Result<Self, EngineError> {
        let mut upgraded = self.clone();
        let pipeline = upgraded
            .snapshot
            .pipeline
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for clock in pipeline {
            upgraded.merge_completed(&clock)?;
        }
        let lineage = upgraded.snapshot.lineage.clone();
        merge_frontier_epochs(
            &mut Arc::make_mut(&mut upgraded.snapshot).completion_lineage,
            &lineage,
        );
        let transported = upgraded.snapshot.transported.upgraded_to_completed();
        Arc::make_mut(&mut upgraded.snapshot).transported = transported;
        Ok(upgraded)
    }
}

fn frontier_epochs_advance(
    current: &BTreeMap<TcgenThreadKey, u64>,
    incoming: &BTreeMap<TcgenThreadKey, u64>,
) -> bool {
    incoming
        .iter()
        .any(|(thread, epoch)| *epoch > current.get(thread).copied().unwrap_or_default())
}

fn merge_frontier_epochs(
    current: &mut BTreeMap<TcgenThreadKey, u64>,
    incoming: &BTreeMap<TcgenThreadKey, u64>,
) {
    for (thread, epoch) in incoming {
        let current_epoch = current.entry(thread.clone()).or_default();
        *current_epoch = (*current_epoch).max(*epoch);
    }
}

fn advance_tcgen_thread_epochs(
    state: &mut RaceCheckState,
    operation: &OperationContext,
) -> Result<(), EngineError> {
    for lane in operation.active_mask() {
        let thread = tcgen_thread_key(operation, lane);
        let epoch = state.tcgen_thread_epochs.entry(thread).or_default();
        *epoch = epoch.checked_add(1).ok_or_else(|| {
            EngineError::message(format!(
                "racecheck TCGEN frontier epoch overflow at {} lane {lane}",
                operation.id()
            ))
        })?;
    }
    Ok(())
}

fn portable_tcgen_frontier(
    state: &RaceCheckState,
    local: &TcgenThreadFenceFrontier,
) -> TcgenFenceFrontier {
    let mut portable = local.transported().clone();
    for (descriptor, clock) in local.pipeline() {
        let components: TcgenComponentFrontier = state
            .shadow
            .tcgen_async_frontier(clock)
            .into_iter()
            .collect::<BTreeMap<_, _>>()
            .into();
        portable.merge_pipeline_frontier(descriptor.clone(), &components);
    }
    if let Some(clock) = local.completed() {
        let components: TcgenComponentFrontier = state
            .shadow
            .tcgen_async_frontier(clock)
            .into_iter()
            .collect::<BTreeMap<_, _>>()
            .into();
        portable.merge_completed_frontier(&components);
    }
    portable
}

fn capture_tcgen_fence_frontier(
    state: &mut RaceCheckState,
    operation: &OperationContext,
    lane: usize,
    commit_cta_group: Option<(u32, TcgenWorkKind)>,
) -> Result<TcgenThreadFenceFrontier, EngineError> {
    let thread = tcgen_thread_key(operation, lane);
    let cache_key = TcgenFenceCaptureKey {
        thread: thread.clone(),
        commit_cta_group,
    };
    if let Some(cached) = state.tcgen_capture_cache.get(&cache_key) {
        return Ok(cached.clone());
    }

    let mut local = TcgenThreadFenceFrontier::default();
    let first_key = TcgenThreadPipelineKey {
        kernel_index: operation.id().kernel_index(),
        global_warp_id: operation.id().global_warp_id(),
        active_mask: WarpMask::EMPTY,
        cta_group: 0,
    };
    let last_key = TcgenThreadPipelineKey {
        kernel_index: operation.id().kernel_index(),
        global_warp_id: operation.id().global_warp_id(),
        active_mask: WarpMask::FULL,
        cta_group: u32::MAX,
    };
    for (key, frontiers) in state.tcgen_pipeline_frontiers.range(first_key..=last_key) {
        if !key.active_mask.contains(lane)
            || commit_cta_group.is_some_and(|(cta_group, _)| key.cta_group != cta_group)
        {
            continue;
        }
        for (descriptor, clock) in frontiers {
            if commit_cta_group.is_some_and(|(_, kind)| kind == TcgenWorkKind::MmaSharedARead)
                && descriptor.operation() != TcgenPipelineOperation::MmaSharedARead
            {
                continue;
            }
            if commit_cta_group.is_some() && !descriptor.operation().work_kind().uses_commit() {
                continue;
            }
            local.merge_pipeline(descriptor.clone(), clock)?;
        }
    }

    if commit_cta_group.is_none() {
        if let Some(frontier) = state.tcgen_fenced_frontiers.get(&thread) {
            local.merge(frontier)?;
        }
        if let Some(clock) = state.tcgen_wait_frontiers.get(&thread) {
            local.merge_completed(clock)?;
        }
    }

    if !local.pipeline().is_empty() || local.completed().is_some() {
        if let Some(epoch) = state.tcgen_thread_epochs.get(&thread).copied() {
            local.mark_source(thread, epoch);
        }
    }

    state.tcgen_capture_cache.insert(cache_key, local.clone());
    Ok(local)
}

fn apply_captured_tcgen_fence(
    state: &mut RaceCheckState,
    operation: &OperationContext,
    captured_frontiers: &BTreeMap<usize, TcgenThreadFenceFrontier>,
) -> Result<(), EngineError> {
    for lane in operation.active_mask() {
        let thread = tcgen_thread_key(operation, lane);
        let Some(captured) = captured_frontiers.get(&lane) else {
            continue;
        };
        state
            .tcgen_published_frontiers
            .entry(thread.clone())
            .or_default()
            .merge(captured)?;
        state
            .tcgen_fenced_frontiers
            .entry(thread.clone())
            .or_default()
            .merge(captured)?;
    }
    Ok(())
}

fn apply_transported_tcgen_fence(
    state: &mut RaceCheckState,
    operation: &OperationContext,
) -> Result<(), EngineError> {
    // Lanes whose incoming and fenced frontiers are the same two snapshots
    // get one join and keep sharing it.
    let key = |lane: usize| tcgen_thread_key(operation, lane);
    let mut groups: Vec<((usize, usize), u32)> = Vec::new();
    for lane in operation.active_mask() {
        let thread = key(lane);
        let Some(incoming) = state.tcgen_incoming_frontiers.get(&thread) else {
            continue;
        };
        let identity = (
            incoming.snapshot_ptr(),
            state
                .tcgen_fenced_frontiers
                .get(&thread)
                .map_or(0, TcgenThreadFenceFrontier::snapshot_ptr),
        );
        match groups.iter_mut().find(|(current, _)| *current == identity) {
            Some((_, lanes)) => *lanes |= 1 << lane,
            None => groups.push((identity, 1 << lane)),
        }
    }
    for (_, lanes) in groups {
        let lanes = WarpMask::from_bits(lanes);
        let first = lanes.into_iter().next().expect("a lane group is not empty");
        let execution_frontier = state
            .tcgen_incoming_frontiers
            .get(&key(first))
            .cloned()
            .expect("a grouped lane has an incoming frontier");
        let mut merged = state
            .tcgen_fenced_frontiers
            .get(&key(first))
            .cloned()
            .unwrap_or_default();
        merged.merge(&execution_frontier)?;
        for lane in lanes {
            state.tcgen_fenced_frontiers.insert(key(lane), merged.clone());
        }
    }
    state.tcgen_capture_cache.clear();
    Ok(())
}

fn tcgen_release_mask(
    state: &RaceCheckState,
    operation: &OperationContext,
    global_warp_id: usize,
    mask: WarpMask,
) -> Result<TcgenThreadFenceFrontier, EngineError> {
    let mut payload = TcgenThreadFenceFrontier::default();
    for lane in mask {
        let thread = TcgenThreadKey {
            kernel_index: operation.id().kernel_index(),
            global_warp_id,
            lane: lane as u8,
        };
        if let Some(frontier) = state.tcgen_published_frontiers.get(&thread) {
            payload.merge(frontier)?;
        }
    }
    Ok(payload)
}

fn tcgen_acquire_mask(
    state: &mut RaceCheckState,
    operation: &OperationContext,
    global_warp_id: usize,
    mask: WarpMask,
    payload: &TcgenThreadFenceFrontier,
) -> Result<(), EngineError> {
    if payload.is_empty() {
        return Ok(());
    }
    // Lanes whose frontier is one snapshot get one join and keep sharing it;
    // merging lane by lane would clone that snapshot once per lane.
    let key = |lane: usize| TcgenThreadKey {
        kernel_index: operation.id().kernel_index(),
        global_warp_id,
        lane: lane as u8,
    };
    let mut groups: Vec<(usize, u32)> = Vec::new();
    for lane in mask {
        let snapshot = state
            .tcgen_incoming_frontiers
            .get(&key(lane))
            .map_or(0, TcgenThreadFenceFrontier::snapshot_ptr);
        match groups.iter_mut().find(|(current, _)| *current == snapshot) {
            Some((_, lanes)) => *lanes |= 1 << lane,
            None => groups.push((snapshot, 1 << lane)),
        }
    }
    for (_, lanes) in groups {
        let lanes = WarpMask::from_bits(lanes);
        let first = lanes.into_iter().next().expect("a lane group is not empty");
        let mut merged = state
            .tcgen_incoming_frontiers
            .get(&key(first))
            .cloned()
            .unwrap_or_default();
        merged.merge(payload)?;
        for lane in lanes {
            state.tcgen_incoming_frontiers.insert(key(lane), merged.clone());
        }
    }
    Ok(())
}

fn retain_physical_barrier_payloads(
    state: &mut RaceCheckState,
    barrier: PhysicalBarrierId,
    generation: u64,
    pin: Option<u64>,
) {
    use crate::sync_causality::retire_barrier_generations_except;
    retire_barrier_generations_except(
        &mut state.barrier_payloads,
        barrier,
        generation,
        RETAINED_BARRIER_GENERATIONS,
        pin,
    );
    retire_barrier_generations_except(
        &mut state.barrier_copy_payloads,
        barrier,
        generation,
        RETAINED_BARRIER_GENERATIONS,
        pin,
    );
    retire_barrier_generations_except(
        &mut state.barrier_lane_payloads,
        barrier,
        generation,
        RETAINED_BARRIER_GENERATIONS,
        pin,
    );
    retire_barrier_generations_except(
        &mut state.barrier_tcgen_payloads,
        barrier,
        generation,
        RETAINED_BARRIER_GENERATIONS,
        pin,
    );
}

fn merge_tcgen_barrier_payload<B: BarrierGenerationWindow>(
    payloads: &mut BTreeMap<(B, u64), TcgenThreadFenceFrontier>,
    key: (B, u64),
    payload: &TcgenThreadFenceFrontier,
) -> Result<(), EngineError> {
    if !payload.is_empty() {
        payloads.entry(key).or_default().merge(payload)?;
    }
    B::retire_generations(payloads, key.0, key.1);
    Ok(())
}

fn publish_mbarrier_arrival(
    state: &mut RaceCheckState,
    operation: &OperationContext,
    arrive: StagedArrive,
    release: Option<&BarrierClockPayload>,
) -> Result<(), EngineError> {
    let key = (arrive.barrier_id, arrive.generation);
    if let Some(payload) = release {
        let lane_payload = state
            .lane_shadow
            .barrier_release(arrive.warp_id, arrive.active_mask)
            .map_err(EngineError::message)?;
        match state.barrier_payloads.entry(key) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(payload.clone());
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                entry
                    .get_mut()
                    .merge(payload)
                    .map_err(|error| EngineError::message(error.to_string()))?;
            }
        }
        merge_lane_barrier_payload(&mut state.barrier_lane_payloads, key, lane_payload);
    } else {
        // A fully modeled phase can contain only relaxed arrivals. Record its
        // empty publication, keeping absence reserved for unavailable history.
        state
            .barrier_payloads
            .entry(key)
            .or_insert_with(|| BarrierClockPayload::from_clock(state.shadow.empty_clock()));
    }
    // PTX 9.7.18.6.4.4 explicitly composes relaxed mbarrier arrival/wait with
    // before/after-thread-sync fences. This execution frontier is independent
    // of the ordinary memory release above.
    let tcgen = tcgen_release_mask(state, operation, arrive.warp_id, arrive.active_mask)?;
    merge_tcgen_barrier_payload(&mut state.barrier_tcgen_payloads, key, &tcgen)
}

fn imported_tcgen_pipeline_clock(
    state: &mut RaceCheckState,
    source: &TcgenFenceFrontier,
    descriptor: &TcgenPipelineDescriptor,
) -> Option<RaceVectorClock> {
    let components = source.pipeline().get(descriptor)?;
    if let Some(cached) = state.tcgen_imported_pipeline_cache.get(descriptor) {
        if cached.source.shares_snapshot_with(source) {
            return Some(cached.clock.clone());
        }
    }
    let clock = state.shadow.tcgen_frontier_from_async_components(
        components
            .iter()
            .map(|(token, epoch)| (token.clone(), *epoch)),
    );
    state.tcgen_imported_pipeline_cache.insert(
        descriptor.clone(),
        CachedImportedTcgenClock {
            source: source.clone(),
            clock: clock.clone(),
        },
    );
    Some(clock)
}

fn imported_tcgen_completed_clock(
    state: &mut RaceCheckState,
    source: &TcgenFenceFrontier,
) -> Option<RaceVectorClock> {
    if source.completed().is_empty() {
        return None;
    }
    if let Some(cached) = &state.tcgen_imported_completed_cache {
        if cached.source.shares_snapshot_with(source) {
            return Some(cached.clock.clone());
        }
    }
    let clock = state.shadow.tcgen_frontier_from_async_components(
        source
            .completed()
            .iter()
            .map(|(token, epoch)| (token.clone(), *epoch)),
    );
    state.tcgen_imported_completed_cache = Some(CachedImportedTcgenClock {
        source: source.clone(),
        clock: clock.clone(),
    });
    Some(clock)
}

fn tcgen_pipeline_predecessor(
    state: &mut RaceCheckState,
    issue: &TcgenWorkIssue,
) -> Result<Option<RaceVectorClock>, EngineError> {
    let descriptor = tcgen_pipeline_descriptor(issue);
    let mut predecessor = None;
    if let Some(frontiers) = state
        .tcgen_pipeline_frontiers
        .get(&tcgen_pipeline_key(issue))
    {
        for (source, frontier) in frontiers {
            if source.pipelines_into(&descriptor) {
                merge_tcgen_clock(&mut predecessor, frontier)?;
            }
        }
    }

    // Descriptor pairing applies only to the unfenced implicit-pipeline path
    // above. PTX thread fences order every captured prior TCGEN operation, so
    // every participating PTX thread consumes its complete fenced frontier.
    let mut cross_thread_frontier = None;
    for lane in issue.operation().active_mask() {
        let thread = tcgen_thread_key(issue.operation(), lane);
        let mut lane_frontier = state.tcgen_wait_frontiers.get(&thread).cloned();
        let fenced = state.tcgen_fenced_frontiers.get(&thread).cloned();
        if let Some(frontier) = fenced {
            if let Some(completed) = frontier.completed() {
                merge_tcgen_clock(&mut lane_frontier, completed)?;
            }
            for pipeline_frontier in frontier.pipeline().values() {
                merge_tcgen_clock(&mut lane_frontier, pipeline_frontier)?;
            }
            if let Some(completed) = imported_tcgen_completed_clock(state, frontier.transported()) {
                merge_tcgen_clock(&mut lane_frontier, &completed)?;
            }
            for source in frontier.transported().pipeline().keys() {
                if let Some(imported) =
                    imported_tcgen_pipeline_clock(state, frontier.transported(), source)
                {
                    merge_tcgen_clock(&mut lane_frontier, &imported)?;
                }
            }
        }
        let Some(lane_frontier) = lane_frontier else {
            return Ok(predecessor);
        };
        merge_tcgen_clock(&mut cross_thread_frontier, &lane_frontier)?;
    }
    if let Some(cross_thread_frontier) = cross_thread_frontier {
        merge_tcgen_clock(&mut predecessor, &cross_thread_frontier)?;
    }
    Ok(predecessor)
}

fn new_race_check_state(
    topology: Option<LaunchTopology>,
    retain_accesses: bool,
    global_warp_base: usize,
    warp_count: usize,
) -> RaceCheckState {
    RaceCheckState {
        topology,
        retain_accesses,
        shadow: RaceShadow::for_warp_range_with_topology(global_warp_base, warp_count, topology),
        lane_shadow: SameWarpLaneShadow::for_warp_range(global_warp_base, warp_count),
        pending_direct_segment: None,
        staged_compact_access: None,
        staged_accesses: BTreeMap::new(),
        staged_async_payloads: BTreeMap::new(),
        staged_async_completions: BTreeMap::new(),
        async_token_clocks: BTreeMap::new(),
        staged_async_group_issues: BTreeMap::new(),
        staged_async_group_completions: BTreeMap::new(),
        async_group_token_clocks: BTreeMap::new(),
        cp_async_completed_payloads: BTreeMap::new(),
        tcgen_work_tokens: BTreeMap::new(),
        reviewed_tcgen_load_tokens: BTreeMap::new(),
        tcgen_pipeline_frontiers: BTreeMap::new(),
        tcgen_thread_epochs: BTreeMap::new(),
        tcgen_fenced_frontiers: BTreeMap::new(),
        tcgen_published_frontiers: BTreeMap::new(),
        tcgen_incoming_frontiers: BTreeMap::new(),
        tcgen_wait_frontiers: BTreeMap::new(),
        tcgen_capture_cache: BTreeMap::new(),
        tcgen_imported_pipeline_cache: BTreeMap::new(),
        tcgen_imported_completed_cache: None,
        tcgen_commit_frontiers: BTreeMap::new(),
        staged_arrives: BTreeMap::new(),
        arrival_completion_payloads: BTreeMap::new(),
        barrier_payloads: BTreeMap::new(),
        barrier_copy_payloads: BTreeMap::new(),
        barrier_lane_payloads: BTreeMap::new(),
        barrier_tcgen_payloads: BTreeMap::new(),
        named_barrier_payloads: BTreeMap::new(),
        named_barrier_lane_payloads: BTreeMap::new(),
        named_barrier_tcgen_payloads: BTreeMap::new(),
        cluster_barrier_payloads: BTreeMap::new(),
        cluster_barrier_lane_payloads: BTreeMap::new(),
        cluster_barrier_tcgen_payloads: BTreeMap::new(),
        findings: Vec::new(),
        access_count: 0,
        alias_tracker: AliasTracker::default(),
        accesses: Vec::new(),
        incomplete_reasons: Vec::new(),
    }
}

impl RaceCheckLaunchState {
    pub fn new(warp_count: usize) -> Self {
        Self::with_optional_topology(
            0,
            warp_count,
            None,
            ResolvedTransitionLog::default(),
            true,
            None,
            false,
        )
    }

    pub fn for_topology(topology: LaunchTopology) -> Self {
        Self::with_optional_topology(
            0,
            topology.warp_count(),
            Some(topology),
            ResolvedTransitionLog::default(),
            true,
            None,
            false,
        )
    }

    pub fn for_topology_and_global_write_allocations(
        topology: LaunchTopology,
        allocations: impl IntoIterator<Item = PhysicalAllocationId>,
    ) -> Self {
        Self::with_optional_topology(
            0,
            topology.warp_count(),
            Some(topology),
            ResolvedTransitionLog::default(),
            true,
            Some(allocations.into_iter().collect()),
            false,
        )
    }

    pub fn for_cluster(topology: LaunchTopology, cluster_id: usize, retain_accesses: bool) -> Self {
        let warp_range = topology
            .cluster_warp_range(cluster_id)
            .expect("cluster racecheck state requires a valid cluster ID");
        Self::with_optional_topology(
            warp_range.start,
            warp_range.len(),
            Some(topology),
            ResolvedTransitionLog::default(),
            retain_accesses,
            None,
            false,
        )
    }

    pub fn with_summary_for_cluster_and_global_write_allocations(
        topology: LaunchTopology,
        cluster_id: usize,
        transitions: ResolvedTransitionLog,
        allocations: impl IntoIterator<Item = PhysicalAllocationId>,
    ) -> Self {
        let warp_range = topology
            .cluster_warp_range(cluster_id)
            .expect("cluster racecheck state requires a valid cluster ID");
        Self::with_optional_topology(
            warp_range.start,
            warp_range.len(),
            Some(topology),
            transitions,
            false,
            Some(allocations.into_iter().collect()),
            false,
        )
    }

    pub fn with_direct_summary_for_cluster_and_global_write_allocations(
        topology: LaunchTopology,
        cluster_id: usize,
        allocations: impl IntoIterator<Item = PhysicalAllocationId>,
    ) -> Self {
        let warp_range = topology
            .cluster_warp_range(cluster_id)
            .expect("cluster racecheck state requires a valid cluster ID");
        Self::with_optional_topology(
            warp_range.start,
            warp_range.len(),
            Some(topology),
            ResolvedTransitionLog::default(),
            false,
            Some(allocations.into_iter().collect()),
            true,
        )
    }

    pub fn with_direct_summary_for_topology_and_global_write_allocations(
        topology: LaunchTopology,
        allocations: impl IntoIterator<Item = PhysicalAllocationId>,
    ) -> Self {
        Self::with_optional_topology(
            0,
            topology.warp_count(),
            Some(topology),
            ResolvedTransitionLog::default(),
            false,
            Some(allocations.into_iter().collect()),
            true,
        )
    }

    fn with_optional_topology(
        global_warp_base: usize,
        warp_count: usize,
        topology: Option<LaunchTopology>,
        transitions: ResolvedTransitionLog,
        retain_accesses: bool,
        global_write_allocations: Option<std::collections::BTreeSet<PhysicalAllocationId>>,
        direct_compact: bool,
    ) -> Self {
        let global_shadow_allocations = global_write_allocations.clone();
        let global_allocation_epochs = global_shadow_allocations
            .iter()
            .flatten()
            .copied()
            .map(|allocation| (allocation, Arc::new(GlobalAllocationEpoch::default())))
            .collect();
        // One launch context, referenced by both peer observers. Every value
        // racecheck used to read back out of the embedded synccheck state now
        // comes from here, so the two can no longer disagree.
        //
        // Direct-compact mode records no resolved transitions, and in that mode
        // the caller-supplied log is deliberately dropped: neither peer may
        // reach it. The remaining two knobs are synccheck-private and stay a
        // constructor argument.
        let uses_compact_memory_analysis = global_write_allocations.is_some();
        let records_resolved_transitions = !(uses_compact_memory_analysis && direct_compact);
        let context = Arc::new(CheckerLaunchContext::new(
            topology,
            if records_resolved_transitions {
                transitions
            } else {
                ResolvedTransitionLog::default()
            },
            global_write_allocations,
            records_resolved_transitions,
        ));
        let sync = SyncCheckLaunchState::with_shared_context(
            Arc::clone(&context),
            topology.is_some() && uses_compact_memory_analysis && !direct_compact,
            if records_resolved_transitions || topology.is_none() {
                u64::MAX
            } else {
                RACECHECK_COMPACT_EFFECT_DIAGNOSTIC_LIMIT
            },
        );
        let requested_end = global_warp_base
            .checked_add(warp_count)
            .expect("racecheck warp range must not overflow");
        let mut shard_ranges = Vec::new();
        if let Some(topology) = topology {
            for cluster_id in 0..topology.clusters() {
                let cluster_range = topology
                    .cluster_warp_range(cluster_id)
                    .expect("topology cluster ID came from its declared range");
                let start = cluster_range.start.max(global_warp_base);
                let end = cluster_range.end.min(requested_end);
                if start < end {
                    shard_ranges.push(start..end);
                }
            }
        }
        if shard_ranges.is_empty() {
            shard_ranges.push(global_warp_base..requested_end);
        }
        let first_shard_cluster_id = topology.map(|topology| {
            shard_ranges
                .first()
                .expect("racecheck always constructs at least one shard")
                .start
                / topology.warps_per_cluster()
        });
        let shard_warps_per_cluster = topology.map(LaunchTopology::warps_per_cluster);
        let global_shared = Arc::new(GlobalRaceShared::new(global_shadow_allocations));
        let race_shards = shard_ranges
            .into_iter()
            .map(|range| {
                let global_warp_base = range.start;
                let global_warp_end = range.end;
                let shard_warp_count = range.len();
                RaceCheckShard {
                    global_warp_base,
                    global_warp_end,
                    uncontrolled_access_count: AtomicUsize::new(0),
                    compact_global_read_caches: (0..shard_warp_count)
                        .map(|_| Mutex::new(CompactGlobalReadCache::default()))
                        .collect(),
                    state: Mutex::new(new_race_check_state(
                        topology,
                        retain_accesses,
                        global_warp_base,
                        shard_warp_count,
                    )),
                    global: Arc::new(Mutex::new(GlobalRaceState::for_warp_range(
                        topology,
                        global_warp_base,
                        shard_warp_count,
                        Arc::clone(&global_shared),
                    ))),
                }
            })
            .collect();
        Self {
            context,
            sync,
            race: RaceObserverState {
                global_floor_gc: GlobalFloorGc::new(),
                global_transaction: (0..GLOBAL_TRANSACTION_STRIPES)
                    .map(|_| RwLock::new(()))
                    .collect(),
                global_allocation_epochs,
                global_shared,
                race_shards,
                first_shard_cluster_id,
                shard_warps_per_cluster,
                retain_accesses,
                direct_compact,
                global_memory_model_enabled: true,
            },
        }
    }

    pub fn set_global_memory_model_enabled(&mut self, enabled: bool) {
        self.race.global_memory_model_enabled = enabled;
    }

    pub fn transition_log(&self) -> &ResolvedTransitionLog {
        self.context.transition_log()
    }

    fn records_resolved_transitions(&self) -> bool {
        self.context.records_resolved_transitions()
    }

    /// Whether a resolved batch participates in Racecheck's conflict model.
    ///
    /// Global batches still reach the runtime for numeric execution, exact
    /// address/OOB validation, accounting, and optional access inspection.
    /// They do not need vector-clock or interval-shadow work while global
    /// publication/read-from ordering is outside the checked race domain.
    fn tracks_race_conflict_batch(&self, batch: &PhysicalAccessBatch) -> bool {
        tracks_race_conflicts(batch.descriptor().space())
    }

    fn shard_for_warp(&self, global_warp_id: usize) -> Result<&RaceCheckShard, EngineError> {
        // Topology-backed shards are emitted in contiguous cluster order, so
        // the hot per-access lookup is direct instead of a binary search.
        let shard_index = match (
            self.race.first_shard_cluster_id,
            self.race.shard_warps_per_cluster,
        ) {
            (Some(first_cluster), Some(warps_per_cluster)) => (global_warp_id / warps_per_cluster)
                .checked_sub(first_cluster)
                .unwrap_or(self.race.race_shards.len()),
            _ => 0,
        };
        self.race
            .race_shards
            .get(shard_index)
            .filter(|shard| {
                shard.global_warp_base <= global_warp_id && global_warp_id < shard.global_warp_end
            })
            .ok_or_else(|| {
                EngineError::message(format!(
                    "racecheck warp {global_warp_id} is outside every launch shard"
                ))
            })
    }

    fn race_for_warp(
        &self,
        global_warp_id: usize,
    ) -> Result<MutexGuard<'_, RaceCheckState>, EngineError> {
        let shard = self.shard_for_warp(global_warp_id)?;
        Ok(shard.state.lock().expect("race-check state poisoned"))
    }

    /// The global-memory model of the shard owning `global_warp_id`.
    fn declared_word_candidates(
        &self,
        span: PhysicalByteSpan,
        warp_id: usize,
        lane: usize,
    ) -> Result<(usize, Vec<u64>), EngineError> {
        // The numerical word can satisfy the wait before its writer commits
        // the corresponding history. Select history only after that commit.
        self.wait_for_quiescent_global_spans(std::iter::once(span));
        Ok(self
            .global_for_warp(warp_id)?
            .declared_word_candidates(span, warp_id, lane))
    }

    fn global_for_warp(
        &self,
        global_warp_id: usize,
    ) -> Result<MutexGuard<'_, GlobalRaceState>, EngineError> {
        let shard = self.shard_for_warp(global_warp_id)?;
        Ok(shard
            .global
            .lock()
            .expect("racecheck global state poisoned"))
    }

    fn global_for_token(
        &self,
        token: &AsyncTokenId,
    ) -> Result<MutexGuard<'_, GlobalRaceState>, EngineError> {
        self.global_for_warp(token.issue_operation().global_warp_id())
    }

    fn merge_global_frontier(
        &self,
        operation: &OperationContext,
        frontier: &GlobalClockFrontier,
    ) -> Result<(), EngineError> {
        if frontier.is_empty() {
            return Ok(());
        }
        let mut state = self.race_for_operation(operation)?;
        Self::commit_pending_direct_segment(&mut state);
        // The lanes of one batch usually acquire the same frontier; acquiring
        // it under their joint mask joins it once (into the common payload
        // when every lane takes it) instead of once per lane.
        let mut groups: Vec<(&SharedClockFrontier, u32)> = Vec::new();
        for (&lane, incoming) in frontier {
            match groups.iter_mut().find(|(current, _)| *current == incoming) {
                Some((_, lanes)) => *lanes |= 1 << lane,
                None => groups.push((incoming, 1 << lane)),
            }
        }
        for (incoming, lanes) in groups {
            state
                .lane_shadow
                .barrier_acquire(
                    operation.id().global_warp_id(),
                    WarpMask::from_bits(lanes),
                    Some(incoming),
                )
                .map_err(EngineError::message)?;
        }
        Ok(())
    }

    fn shared_frontier_for_operation(
        &self,
        operation: &OperationContext,
    ) -> Result<SharedLaneFrontiers, EngineError> {
        let mut state = self.race_for_operation(operation)?;
        Self::commit_pending_direct_segment(&mut state);
        let warp = operation.id().global_warp_id();
        let base = state.lane_shadow.global_warp_base;
        operation
            .active_mask()
            .into_iter()
            .map(|lane| {
                let mask = WarpMask::from_bits(1 << lane);
                let mut frontier = state
                    .lane_shadow
                    .barrier_release(warp, mask)
                    .map_err(EngineError::message)?;
                let clock = state
                    .shadow
                    .memory_publication(warp, mask)
                    .map_err(|error| EngineError::message(error.to_string()))?;
                frontier.merge_clock(base, &clock);
                Ok((lane, frontier))
            })
            .collect()
    }

    #[cfg(feature = "profile")]
    pub(crate) fn global_replay_diagnostic_line(&self) -> Option<String> {
        // The replay oracle re-runs one shard's event log in order; with
        // several shards their logs interleave arbitrarily, so only a
        // single-shard launch reports an exact replay.
        self.race.global_memory_model_enabled.then(|| {
            if self.race.race_shards.len() == 1 {
                self.race.race_shards[0]
                    .global
                    .lock()
                    .expect("racecheck global state poisoned")
                    .replay_diagnostic_line()
            } else {
                format!(
                    "{{\"replay_available\":false,\"shards\":{}}}",
                    self.race.race_shards.len()
                )
            }
        })
    }

    fn compact_global_read_allocation_epochs(
        &self,
        batch: &CompactPhysicalAccessBatch<'_>,
    ) -> Option<Box<[(Arc<GlobalAllocationEpoch>, u64)]>> {
        let allocations = batch
            .lane_spans()
            .map(|(_, span)| span.allocation())
            .collect::<BTreeSet<_>>();
        let mut epochs = Vec::with_capacity(allocations.len());
        for allocation in allocations {
            let state = self.race.global_allocation_epochs.get(&allocation)?;
            if state.writes_in_flight.load(AtomicOrdering::Acquire) != 0 {
                return None;
            }
            epochs.push((Arc::clone(state), state.epoch.load(AtomicOrdering::Acquire)));
        }
        for (state, epoch) in &epochs {
            if state.writes_in_flight.load(AtomicOrdering::Acquire) != 0
                || state.epoch.load(AtomicOrdering::Acquire) != *epoch
            {
                return None;
            }
        }
        Some(epochs.into_boxed_slice())
    }

    fn compact_global_read_entry_is_current(&self, entry: &CompactGlobalReadCacheEntry) -> bool {
        Self::compact_global_read_epochs_are_current(&entry.allocation_epochs)
    }

    fn compact_global_read_epochs_are_current(
        epochs: &[(Arc<GlobalAllocationEpoch>, u64)],
    ) -> bool {
        for (state, epoch) in epochs {
            if state.writes_in_flight.load(AtomicOrdering::Acquire) != 0
                || state.epoch.load(AtomicOrdering::Acquire) != *epoch
            {
                return false;
            }
        }
        true
    }

    /// Park an atomic-class global access until no overlapping write is
    /// between its numerical effect and its Racecheck metadata commit.
    ///
    /// A pre-reserved atomic RMW joins no global transaction, so a polling
    /// acquire load (or an RMW reading from a release store) could otherwise
    /// resolve its read-from version before the writer's version is
    /// published while its numerical read already observes the new value.
    /// That drops the writer's release heads from the acquire and reports the
    /// data the release ordered as racing. Waiting on the exact in-flight
    /// spans keeps the physical and shadow read-from orders aligned without
    /// serializing against unrelated writes to the same allocation.
    fn wait_for_quiescent_atomic_access(
        &self,
        semantics: MemoryAccessSemantics,
        spans: impl Iterator<Item = PhysicalByteSpan>,
    ) {
        if !self.race.global_memory_model_enabled || !semantics.class().is_atomic_class() {
            return;
        }
        self.wait_for_quiescent_global_spans(spans);
    }

    fn wait_for_quiescent_global_spans(&self, spans: impl Iterator<Item = PhysicalByteSpan>) {
        for span in spans {
            let Some(state) = self.race.global_allocation_epochs.get(&span.allocation()) else {
                continue;
            };
            let mut waited = false;
            while state.has_write_in_flight(span.byte_offset(), span.byte_end()) {
                waited = true;
                std::thread::yield_now();
            }
            if waited {
            }
        }
    }

    fn begin_global_writes(&self, allocations: impl IntoIterator<Item = PhysicalAllocationId>) {
        for allocation in allocations.into_iter().collect::<BTreeSet<_>>() {
            if let Some(state) = self.race.global_allocation_epochs.get(&allocation) {
                state.begin_write(0, usize::MAX);
            }
        }
    }

    fn finish_global_writes(&self, allocations: impl IntoIterator<Item = PhysicalAllocationId>) {
        for allocation in allocations.into_iter().collect::<BTreeSet<_>>() {
            if let Some(state) = self.race.global_allocation_epochs.get(&allocation) {
                state.finish_write(0, usize::MAX);
            }
        }
    }

    fn begin_global_write_spans(&self, spans: impl IntoIterator<Item = PhysicalByteSpan>) {
        for span in spans {
            if let Some(state) = self.race.global_allocation_epochs.get(&span.allocation()) {
                state.begin_write(span.byte_offset(), span.byte_end());
            }
        }
    }

    fn finish_global_write_spans(&self, spans: impl IntoIterator<Item = PhysicalByteSpan>) {
        for span in spans {
            if let Some(state) = self.race.global_allocation_epochs.get(&span.allocation()) {
                state.finish_write(span.byte_offset(), span.byte_end());
            }
        }
    }

    fn global_write_allocations<'a>(
        batches: impl IntoIterator<Item = &'a PhysicalAccessBatch>,
    ) -> Vec<PhysicalAllocationId> {
        batches
            .into_iter()
            .filter(|batch| {
                batch.descriptor().space() == PhysicalAccessSpace::Global
                    && batch.descriptor().kind().writes()
            })
            .flat_map(|batch| batch.lanes())
            .flat_map(|lane| lane.footprint().spans())
            .map(|span| span.allocation())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn async_group_global_write_allocations(
        action: &crate::AsyncGroupCompletionAction,
    ) -> Vec<PhysicalAllocationId> {
        let batches = action.members().iter().flat_map(|member| {
            match action.milestone() {
                AsyncGroupMilestone::SourceReadComplete => member.source_accesses(),
                AsyncGroupMilestone::FullComplete => member.destination_accesses(),
            }
            .iter()
        });
        Self::global_write_allocations(batches)
    }

    fn completion_action_global_write_allocations(
        effect: CompletionActionEffect<'_>,
    ) -> Vec<PhysicalAllocationId> {
        match effect {
            CompletionActionEffect::DeferredPayload(action) => {
                Self::global_write_allocations(action.completion_accesses())
            }
            CompletionActionEffect::AsyncGroup(action) => {
                Self::async_group_global_write_allocations(action)
            }
            CompletionActionEffect::PhysicalMbarrier(_) | CompletionActionEffect::Setmaxnreg(_) => {
                Vec::new()
            }
        }
    }

    fn completion_global_write_allocations(
        effect: CompletionEffect<'_>,
    ) -> Vec<PhysicalAllocationId> {
        match effect {
            CompletionEffect::DeferredPayload(outcome) => {
                Self::global_write_allocations(outcome.action().completion_accesses())
            }
            CompletionEffect::AsyncGroup(outcome) => {
                Self::async_group_global_write_allocations(outcome.action())
            }
            CompletionEffect::PhysicalMbarrier(_) | CompletionEffect::Setmaxnreg(_) => Vec::new(),
        }
    }

    fn compact_global_read_cache(
        &self,
        global_warp_id: usize,
    ) -> Result<MutexGuard<'_, CompactGlobalReadCache>, EngineError> {
        let shard = self.shard_for_warp(global_warp_id)?;
        let local_warp_id = global_warp_id - shard.global_warp_base;
        Ok(shard.compact_global_read_caches[local_warp_id]
            .lock()
            .expect("racecheck compact global-read cache poisoned"))
    }

    /// Repeated pure loads with the same source, semantics, and exact spans
    /// share one retained race/HB event while their allocation epochs remain
    /// unchanged. Non-load operations invalidate the per-warp entry, and an
    /// overlapping write either blocks cache admission or forces post-numeric
    /// reprocessing, so this is equally valid for plain and atomic polls.
    fn cacheable_compact_global_read(batch: &CompactPhysicalAccessBatch<'_>) -> bool {
        batch.descriptor().space() == PhysicalAccessSpace::Global
            && batch.descriptor().kind() == PhysicalAccessKind::Read
    }

    fn begin_compact_global_read(
        &self,
        batch: &CompactPhysicalAccessBatch<'_>,
    ) -> Result<bool, EngineError> {
        if !Self::cacheable_compact_global_read(batch) {
            return Ok(false);
        }
        let mut cache = self.compact_global_read_cache(batch.operation().id().global_warp_id())?;
        if let Some(pending_sequence) = cache.pending_sequence {
            return Err(EngineError::message(format!(
                "racecheck compact global read {} began before warp sequence {} finished",
                batch.operation().id(),
                pending_sequence,
            )));
        }
        let hit = cache.entry.as_ref().is_some_and(|entry| {
            entry.matches_access(batch) && self.compact_global_read_entry_is_current(entry)
        });
        profile_count(if hit {
            ProfileKind::RaceGlobalReadCacheHit
        } else {
            ProfileKind::RaceGlobalReadCacheMiss
        });
        cache.pending_sequence = Some(batch.operation().id().per_warp_sequence());
        cache.pending_epochs = (!hit)
            .then(|| self.compact_global_read_allocation_epochs(batch))
            .flatten();
        Ok(hit)
    }

    fn begin_cached_global_read(
        &self,
        access: CachedGlobalReadAccess,
    ) -> Result<bool, EngineError> {
        let mut cache = self.compact_global_read_cache(access.global_warp_id())?;
        if let Some(pending_sequence) = cache.pending_sequence {
            return Err(EngineError::message(format!(
                "racecheck cached global read at warp sequence {} began before warp sequence {} finished",
                access.per_warp_sequence(),
                pending_sequence,
            )));
        }
        let hit = cache.entry.as_ref().is_some_and(|entry| {
            entry.matches_cached_access(access) && self.compact_global_read_entry_is_current(entry)
        });
        if hit {
            cache.pending_sequence = Some(access.per_warp_sequence());
            profile_count(ProfileKind::RaceGlobalReadCacheHit);
        }
        Ok(hit)
    }

    fn remember_compact_global_read(
        &self,
        batch: &CompactPhysicalAccessBatch<'_>,
    ) -> Result<(), EngineError> {
        if !Self::cacheable_compact_global_read(batch) {
            return Ok(());
        }
        let mut cache = self.compact_global_read_cache(batch.operation().id().global_warp_id())?;
        let allocation_epochs = cache.pending_epochs.take();
        let pending_sequence = cache.pending_sequence.ok_or_else(|| {
            EngineError::message(format!(
                "racecheck compact global read {} has no pending cache entry",
                batch.operation().id()
            ))
        })?;
        if pending_sequence != batch.operation().id().per_warp_sequence() {
            return Err(EngineError::message(format!(
                "racecheck compact global read {} tried to update cache for warp sequence {}",
                batch.operation().id(),
                pending_sequence,
            )));
        }
        if let Some(allocation_epochs) = allocation_epochs {
            cache.entry = Some(CompactGlobalReadCacheEntry {
                source_op_id: batch.operation().id().source_op_id().get(),
                descriptor: batch.descriptor(),
                lane_spans: batch.lane_spans().collect(),
                allocation_epochs,
            });
        }
        Ok(())
    }

    fn finish_compact_global_read(
        &self,
        batch: &CompactPhysicalAccessBatch<'_>,
    ) -> Result<CompactGlobalReadFinish, EngineError> {
        if !Self::cacheable_compact_global_read(batch) {
            return Ok(CompactGlobalReadFinish::NotCached);
        }
        let mut cache = self.compact_global_read_cache(batch.operation().id().global_warp_id())?;
        cache.pending_epochs = None;
        let pending_sequence = cache.pending_sequence.take().ok_or_else(|| {
            EngineError::message(format!(
                "racecheck compact global read {} has no pending cache entry",
                batch.operation().id()
            ))
        })?;
        if pending_sequence != batch.operation().id().per_warp_sequence() {
            return Err(EngineError::message(format!(
                "racecheck compact global read {} finished cache entry for warp sequence {}",
                batch.operation().id(),
                pending_sequence,
            )));
        }
        if cache.entry.as_ref().is_some_and(|entry| {
            entry.matches_access(batch) && self.compact_global_read_entry_is_current(entry)
        }) {
            return Ok(CompactGlobalReadFinish::Stable);
        }
        cache.entry = None;
        profile_count(ProfileKind::RaceGlobalReadReprocess);
        Ok(CompactGlobalReadFinish::Reprocess)
    }

    fn finish_cached_global_read(
        &self,
        access: CachedGlobalReadAccess,
    ) -> Result<CachedGlobalReadFinish, EngineError> {
        let mut cache = self.compact_global_read_cache(access.global_warp_id())?;
        let pending_sequence = cache.pending_sequence.take().ok_or_else(|| {
            EngineError::message(format!(
                "racecheck cached global read at warp sequence {} has no pending cache entry",
                access.per_warp_sequence(),
            ))
        })?;
        if pending_sequence != access.per_warp_sequence() {
            return Err(EngineError::message(format!(
                "racecheck cached global read at warp sequence {} finished cache entry for warp sequence {}",
                access.per_warp_sequence(),
                pending_sequence,
            )));
        }
        if cache.entry.as_ref().is_some_and(|entry| {
            entry.matches_cached_access(access) && self.compact_global_read_entry_is_current(entry)
        }) {
            return Ok(CachedGlobalReadFinish::Stable);
        }
        cache.entry = None;
        profile_count(ProfileKind::RaceGlobalReadReprocess);
        Ok(CachedGlobalReadFinish::Reprocess)
    }

    fn invalidate_compact_global_read_cache(
        &self,
        global_warp_id: usize,
    ) -> Result<(), EngineError> {
        self.compact_global_read_cache(global_warp_id)?.entry = None;
        Ok(())
    }

    fn race_for_operation(
        &self,
        operation: &OperationContext,
    ) -> Result<MutexGuard<'_, RaceCheckState>, EngineError> {
        self.race_for_warp(operation.id().global_warp_id())
    }

    fn tcgen_publications_for_operation(
        &self,
        operation: &OperationContext,
        required: bool,
    ) -> Result<TcgenLaneFrontiers, EngineError> {
        if !required {
            return Ok(TcgenLaneFrontiers::new());
        }
        let state = self.race_for_operation(operation)?;
        let mut publications = TcgenLaneFrontiers::new();
        for lane in operation.active_mask() {
            let thread = tcgen_thread_key(operation, lane);
            if let Some(frontier) = state.tcgen_published_frontiers.get(&thread) {
                publications.insert(lane, portable_tcgen_frontier(&state, frontier));
            }
        }
        Ok(publications)
    }

    fn merge_tcgen_acquisitions(
        &self,
        operation: &OperationContext,
        acquisitions: TcgenLaneFrontiers,
    ) -> Result<(), EngineError> {
        if acquisitions.is_empty() {
            return Ok(());
        }
        let mut state = self.race_for_operation(operation)?;
        merge_tcgen_acquisitions_into(&mut state, operation, acquisitions);
        Ok(())
    }

    fn race_for_token(
        &self,
        token: &AsyncTokenId,
    ) -> Result<MutexGuard<'_, RaceCheckState>, EngineError> {
        self.race_for_warp(token.issue_operation().global_warp_id())
    }

    fn race_for_async_group_action(
        &self,
        action: &crate::AsyncGroupCompletionAction,
    ) -> Result<MutexGuard<'_, RaceCheckState>, EngineError> {
        let member = action.members().first().ok_or_else(|| {
            EngineError::message(format!(
                "racecheck async-group action {} has no members",
                action.id().get()
            ))
        })?;
        self.race_for_token(member.token())
    }

    fn race_for_arrival_completion(
        &self,
        action: crate::PhysicalCompletionAction,
    ) -> Result<MutexGuard<'_, RaceCheckState>, EngineError> {
        let PhysicalCompletionKind::Arrival { warp_id, .. } = action.kind() else {
            return Err(EngineError::message(format!(
                "racecheck action {} is not an arrival completion",
                action.id().get()
            )));
        };
        let state = self.race_for_warp(warp_id)?;
        if !state.arrival_completion_payloads.contains_key(&action.id()) {
            return Err(EngineError::message(format!(
                "racecheck arrival action {} has no release payload in warp {warp_id}'s shard",
                action.id().get()
            )));
        }
        Ok(state)
    }

    #[cfg(test)]
    fn live_async_group_token_count(&self) -> usize {
        self.race
            .race_shards
            .iter()
            .map(|shard| {
                shard
                    .state
                    .lock()
                    .expect("race-check state poisoned")
                    .async_group_token_clocks
                    .len()
            })
            .sum()
    }

    // ---- peer delivery -----------------------------------------------------
    //
    // `EngineModeImpl` binds one LaunchState per mode, so the runtime notifies
    // exactly one observer per launch. In racecheck mode that observer is this
    // state, and these are the seam where it hands the synccheck peer the same
    // event it was handed. Nothing is interpreted or translated here: the peer
    // sees the stream, not a racecheck-mediated view of it.
    //
    // Every stream-delivery reach into `self.sync` goes through this block, and
    // the peer is reachable from nowhere else — de-embedding removed the
    // accessor that used to hand `&SyncCheckLaunchState` out to callers. Two
    // NON-delivery reaches remain outside it, deliberately, and are the whole
    // residue of the embedding:
    //
    //   * `finish_with` reads the peer's `result*()` — the (b)-class verdict
    //     consumption that `combine_peer_verdicts` folds. Report time, not
    //     stream time.
    //   * `before_effect`'s MbarrierArrive/MbarrierArriveBatch arms query
    //     `staged_mbarrier_arrive_generation(s)` — the barrier-generation
    //     oracle, a genuine information dependency that needs barrier outcomes
    //     published as effects before it can be folded out.
    //
    // A new `self.sync` reach that is neither of those belongs here.

    fn peer_before_effect(
        &self,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        <SyncCheckMode as EngineModeImpl>::before_effect(&self.sync, operation, effect)
    }

    fn peer_after_effect(
        &self,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        <SyncCheckMode as EngineModeImpl>::after_effect(&self.sync, operation, effect)
    }

    fn peer_before_completion(
        &self,
        effect: CompletionActionEffect<'_>,
    ) -> Result<(), EngineError> {
        <SyncCheckMode as EngineModeImpl>::before_completion(&self.sync, effect)
    }

    fn peer_after_completion(&self, effect: CompletionEffect<'_>) -> Result<(), EngineError> {
        <SyncCheckMode as EngineModeImpl>::after_completion(&self.sync, effect)
    }

    fn peer_after_operation(&self, operation: &OperationContext) -> Result<(), EngineError> {
        <SyncCheckMode as EngineModeImpl>::after_operation(&self.sync, operation)
    }

    fn peer_controls_physical_access(
        &self,
        kind: OperationKind,
        space: PhysicalAccessSpace,
    ) -> bool {
        <SyncCheckMode as EngineModeImpl>::controls_physical_access(&self.sync, kind, space)
    }

    fn peer_controls_physical_access_allocation(
        &self,
        kind: OperationKind,
        space: PhysicalAccessSpace,
        allocation: Option<PhysicalAllocationId>,
    ) -> bool {
        <SyncCheckMode as EngineModeImpl>::controls_physical_access_allocation(
            &self.sync, kind, space, allocation,
        )
    }

    fn commit_pending_direct_segment(state: &mut RaceCheckState) {
        let _profile = ProfileTimer::new(ProfileKind::RaceCommitPending);
        let Some(pending) = state.pending_direct_segment.take() else {
            return;
        };
        let shadow_reviews = state.shadow.commit_direct_segment(pending.shadow_segment);
        state.lane_shadow.commit_batch(pending.lane_validation);
        commit_review_findings(state, shadow_reviews);
    }

    fn flush_pending_direct_for_operation(
        &self,
        operation: &OperationContext,
    ) -> Result<(), EngineError> {
        let mut state = self.race_for_operation(operation)?;
        Self::commit_pending_direct_segment(&mut state);
        Ok(())
    }

    fn flush_all_pending_direct(&self) {
        for shard in &self.race.race_shards {
            let mut state = shard.state.lock().expect("race-check state poisoned");
            Self::commit_pending_direct_segment(&mut state);
        }
    }

    /// Validate and record a compact access after its numeric effect succeeds.
    ///
    /// Racecheck never changes the numeric value produced by an access. A
    /// numeric failure therefore publishes no shadow state, while a checker
    /// rejection terminates execution and discards the already-mutated
    /// numerical run. On the successful path this permits one locked,
    /// in-place interval traversal instead of a prevalidation token followed
    /// by a second commit traversal.
    fn apply_compact_physical_access_after_numeric(
        &self,
        batch: &CompactPhysicalAccessBatch<'_>,
    ) -> Result<(), EngineError> {
        let _profile = ProfileTimer::new(ProfileKind::RaceCompactApply);
        let operation = batch.operation();
        if self.race.global_memory_model_enabled
            && batch.descriptor().space() == PhysicalAccessSpace::Shared
            && (batch.descriptor().kind().writes()
                || batch.descriptor().memory_semantics().order().is_strong())
        {
            // The engine holds the same byte transaction as async issue:
            // numeric bytes and their read-from version cannot diverge.
            let publications = self.tcgen_publications_for_operation(
                operation,
                tcgen_publication_required(batch.descriptor()),
            )?;
            let shared = batch
                .descriptor()
                .memory_semantics()
                .order()
                .has_release()
                .then(|| self.shared_frontier_for_operation(operation))
                .transpose()?;
            let (acquisitions, frontier) = {
                let mut global = self.global_for_warp(operation.id().global_warp_id())?;
                let result = global
                    .before_compact_batch(batch, &publications, shared.as_ref())
                    .map_err(EngineError::message)?;
                global
                    .after_compact_batch(batch)
                    .map_err(EngineError::message)?;
                result
            };
            self.merge_tcgen_acquisitions(operation, acquisitions)?;
            self.merge_global_frontier(operation, &frontier)?;
        }
        let mut state = self.race_for_operation(operation)?;
        debug_assert!(state.staged_compact_access.is_none());
        debug_assert!(!state.staged_accesses.contains_key(operation.id()));

        let global_warp_id = operation.id().global_warp_id();
        let direct_geometry = {
            let _profile = ProfileTimer::new(ProfileKind::RaceCompactGeometry);
            state.shadow.compact_direct_geometry(batch)
        };
        let supports_direct_segment = direct_geometry.is_some();
        if state
            .pending_direct_segment
            .as_ref()
            .is_some_and(|pending| {
                pending.global_warp_id != global_warp_id || !supports_direct_segment
            })
        {
            Self::commit_pending_direct_segment(&mut state);
        }

        let lane_validation = match {
            let _profile = ProfileTimer::new(ProfileKind::RaceLaneValidate);
            state.lane_shadow.validate_compact_order_batch(batch)
        } {
            Ok(validation) => validation,
            Err(reason) => {
                push_unique_incomplete(
                    &mut state.incomplete_reasons,
                    RaceCheckIncompleteReason::ShadowRejected {
                        operation: operation.id().clone(),
                        reason: reason.clone(),
                    },
                );
                return Err(EngineError::message(format!(
                    "racecheck could not validate same-warp lane order for {}: {reason}",
                    operation.id()
                )));
            }
        };

        let shadow_result = {
            let _profile = ProfileTimer::new(ProfileKind::RaceShadowApply);
            let RaceCheckState {
                lane_shadow,
                pending_direct_segment,
                shadow,
                ..
            } = &mut *state;
            let lane_order = lane_shadow
                .race_lane_order(global_warp_id)
                .expect("validated compact lane shadow has an order for this warp");
            shadow.apply_compact_batch_after_numeric(
                batch,
                &lane_order,
                direct_geometry,
                pending_direct_segment
                    .as_ref()
                    .map(|pending| &pending.shadow_segment),
            )
        };
        let (new_shadow_segment, shadow_reviews) = match shadow_result {
            Ok(result) => result,
            Err(RaceShadowError::Race(finding)) => {
                let message = race_rejection_message(operation.id(), &finding);
                state.findings.push(finding);
                return Err(EngineError::message(message));
            }
            Err(error) => {
                push_unique_incomplete(
                    &mut state.incomplete_reasons,
                    RaceCheckIncompleteReason::ShadowRejected {
                        operation: operation.id().clone(),
                        reason: error.to_string(),
                    },
                );
                return Err(EngineError::message(format!(
                    "racecheck could not validate {}: {error}",
                    operation.id()
                )));
            }
        };
        commit_review_findings(&mut state, shadow_reviews);

        {
            if supports_direct_segment {
                if let Some(pending) = state.pending_direct_segment.as_mut() {
                    debug_assert_eq!(pending.global_warp_id, global_warp_id);
                    debug_assert!(new_shadow_segment.is_none());
                    pending
                        .lane_validation
                        .merge_order_validation(lane_validation);
                } else {
                    state.pending_direct_segment = Some(PendingDirectSegment {
                        global_warp_id,
                        lane_validation,
                        shadow_segment: new_shadow_segment
                            .expect("a new direct segment returns its shadow state"),
                    });
                }
            } else {
                debug_assert!(state.pending_direct_segment.is_none());
                debug_assert!(new_shadow_segment.is_none());
                state.lane_shadow.commit_batch(lane_validation);
            }
        }

        state.access_count = state.access_count.saturating_add(1);
        {
            let _profile = ProfileTimer::new(ProfileKind::RaceAliasObserve);
            state
                .alias_tracker
                .observe_compact_batch_with_geometry(batch, direct_geometry);
        }
        debug_assert!(
            !self.records_resolved_transitions(),
            "compact physical batches do not retain an owned transition payload"
        );
        Ok(())
    }

    fn before_compact_physical_access(
        &self,
        batch: &CompactPhysicalAccessBatch<'_>,
    ) -> Result<(), EngineError> {
        if self.race.global_memory_model_enabled
            && batch.descriptor().space() == PhysicalAccessSpace::Global
        {
            if self.begin_compact_global_read(batch)? {
                return Ok(());
            }
            self.wait_for_quiescent_atomic_access(
                batch.descriptor().memory_semantics(),
                batch.lane_spans().map(|(_, span)| span),
            );
            let write_spans = batch
                .descriptor()
                .kind()
                .writes()
                .then(|| batch.lane_spans().map(|(_, span)| span).collect::<Vec<_>>())
                .unwrap_or_default();
            self.begin_global_write_spans(write_spans.iter().copied());
            if Self::defers_plain_compact_write(batch) {
                return Ok(());
            }
            let tcgen_publications = self.tcgen_publications_for_operation(
                batch.operation(),
                tcgen_publication_required(batch.descriptor()),
            )?;
            let shared_frontier = batch
                .descriptor()
                .memory_semantics()
                .order()
                .has_release()
                .then(|| self.shared_frontier_for_operation(batch.operation()))
                .transpose()?;
            let result = {
                let mut global = {
                    let _profile = batch
                        .descriptor()
                        .kind()
                        .reads()
                        .then(|| ProfileTimer::new(ProfileKind::RaceGlobalReadLock));
                    self.global_for_warp(batch.operation().id().global_warp_id())?
                };
                let _profile = batch
                    .descriptor()
                    .kind()
                    .reads()
                    .then(|| ProfileTimer::new(ProfileKind::RaceGlobalReadApply));
                global
                    .before_compact_batch(batch, &tcgen_publications, shared_frontier.as_ref())
                    .map_err(EngineError::message)
            };
            let (tcgen_acquisitions, global_frontier) = match result {
                Ok(preparation) => preparation,
                Err(error) => {
                    self.finish_global_write_spans(write_spans);
                    return Err(error);
                }
            };
            self.merge_tcgen_acquisitions(batch.operation(), tcgen_acquisitions)?;
            self.merge_global_frontier(batch.operation(), &global_frontier)?;
            self.remember_compact_global_read(batch)?;
            return Ok(());
        }
        let operation = batch.operation();
        let mut state = self.race_for_operation(operation)?;
        if state.staged_compact_access.is_some()
            || state.staged_accesses.contains_key(operation.id())
        {
            return Err(EngineError::message(format!(
                "racecheck operation {} already has a staged physical access",
                operation.id()
            )));
        }
        let global_warp_id = operation.id().global_warp_id();
        let direct_geometry = state.shadow.compact_direct_geometry(batch);
        let supports_direct_segment = direct_geometry.is_some();
        let must_flush_pending = state
            .pending_direct_segment
            .as_ref()
            .is_some_and(|pending| {
                pending.global_warp_id != global_warp_id || !supports_direct_segment
            });
        if must_flush_pending {
            Self::commit_pending_direct_segment(&mut state);
        }
        let lane_result = state.lane_shadow.validate_compact_order_batch(batch);
        let lane_validation = match lane_result {
            Ok(validation) => validation,
            Err(reason) => {
                push_unique_incomplete(
                    &mut state.incomplete_reasons,
                    RaceCheckIncompleteReason::ShadowRejected {
                        operation: operation.id().clone(),
                        reason: reason.clone(),
                    },
                );
                return Err(EngineError::message(format!(
                    "racecheck could not validate same-warp lane order for {}: {reason}",
                    operation.id()
                )));
            }
        };
        let shadow_result = {
            let RaceCheckState {
                lane_shadow,
                pending_direct_segment,
                shadow,
                ..
            } = &mut *state;
            let lane_order = lane_shadow
                .race_lane_order(global_warp_id)
                .expect("validated compact lane shadow has an order for this warp");
            if supports_direct_segment {
                match pending_direct_segment.as_ref() {
                    Some(pending) => shadow.validate_compact_batch_for_direct_segment(
                        batch,
                        &lane_order,
                        direct_geometry.expect("direct geometry was already resolved"),
                        &pending.shadow_segment,
                    ),
                    None => shadow.validate_compact_batch_for_new_direct_segment(
                        batch,
                        &lane_order,
                        direct_geometry.expect("direct geometry was already resolved"),
                    ),
                }
            } else {
                shadow.validate_compact_batch_for_direct_commit(batch, &lane_order, direct_geometry)
            }
        };
        let shadow_validation = match shadow_result {
            Ok(validation) => validation,
            Err(RaceShadowError::Race(finding)) => {
                let message = race_rejection_message(operation.id(), &finding);
                state.findings.push(finding);
                return Err(EngineError::message(message));
            }
            Err(error) => {
                push_unique_incomplete(
                    &mut state.incomplete_reasons,
                    RaceCheckIncompleteReason::ShadowRejected {
                        operation: operation.id().clone(),
                        reason: error.to_string(),
                    },
                );
                return Err(EngineError::message(format!(
                    "racecheck could not validate {}: {error}",
                    operation.id()
                )));
            }
        };
        let shadow_revision = state.shadow.revision();
        state.staged_compact_access = Some(StagedCompactAccess {
            operation: operation.shared_id(),
            lane_validation,
            shadow_revision,
            shadow_validation,
        });
        Ok(())
    }

    fn after_compact_physical_access(
        &self,
        batch: &CompactPhysicalAccessBatch<'_>,
    ) -> Result<(), EngineError> {
        if self.race.global_memory_model_enabled
            && batch.descriptor().space() == PhysicalAccessSpace::Global
        {
            // Every path in this branch returns early, so the shared accounting
            // at the end of this function never runs for a global access. Count
            // it here instead: the access executed, and every other regime
            // counts it -- the journal through `prepare_access_batch_commit`,
            // and the pre-global-model engine through the uncontrolled counter.
            // Reporting fewer accesses in the compact summary would make the
            // report depend on whether the caller asked for the journal.
            {
                let mut state = self.race_for_operation(batch.operation())?;
                state.access_count = state.access_count.saturating_add(1);
            }
            match self.finish_compact_global_read(batch)? {
                CompactGlobalReadFinish::Stable => return Ok(()),
                CompactGlobalReadFinish::Reprocess => {
                    let tcgen_publications = self.tcgen_publications_for_operation(
                        batch.operation(),
                        tcgen_publication_required(batch.descriptor()),
                    )?;
                    let spans = batch.lane_spans().map(|(_, span)| span).collect::<Vec<_>>();
                    let _transaction =
                        <RaceCheckMode as EngineModeImpl>::begin_global_memory_transaction(
                            self, true, &spans,
                        )?;
                    self.wait_for_quiescent_atomic_access(
                        batch.descriptor().memory_semantics(),
                        spans.iter().copied(),
                    );
                    let (tcgen_acquisitions, global_frontier) = self
                        .global_for_warp(batch.operation().id().global_warp_id())?
                        .before_compact_batch(batch, &tcgen_publications, None)
                        .map_err(EngineError::message)?;
                    drop(_transaction);
                    self.merge_tcgen_acquisitions(batch.operation(), tcgen_acquisitions)?;
                    self.merge_global_frontier(batch.operation(), &global_frontier)?;
                    return Ok(());
                }
                CompactGlobalReadFinish::NotCached => {}
            }
            if batch.descriptor().kind().writes() {
                let write_spans = batch.lane_spans().map(|(_, span)| span).collect::<Vec<_>>();
                let result = if Self::defers_plain_compact_write(batch) {
                    let mut global = {
                        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalPlainWriteLock);
                        self.global_for_warp(batch.operation().id().global_warp_id())?
                    };
                    let _profile = ProfileTimer::new(ProfileKind::RaceGlobalPlainWriteApply);
                    global
                        .apply_plain_compact_write_after_numeric(batch)
                        .map_err(EngineError::message)
                } else {
                    self.global_for_warp(batch.operation().id().global_warp_id())?
                        .after_compact_batch(batch)
                        .map(|()| GlobalClockFrontier::default())
                        .map_err(EngineError::message)
                };
                self.finish_global_write_spans(write_spans);
                let frontier = result?;
                self.merge_global_frontier(batch.operation(), &frontier)?;
                return Ok(());
            }
            return Ok(());
        }
        let operation = batch.operation();
        let mut state = self.race_for_operation(operation)?;
        let mut staged = state.staged_compact_access.take().ok_or_else(|| {
            EngineError::message(format!(
                "racecheck operation {} has no staged physical access",
                operation.id()
            ))
        })?;
        if staged.operation.as_ref() != operation.id() {
            return Err(EngineError::message(format!(
                "racecheck staged compact operation {} was committed as {}",
                staged.operation,
                operation.id()
            )));
        }
        let shadow_validation = if state.shadow.revision() == staged.shadow_revision {
            staged.shadow_validation
        } else {
            // Completion sources can publish a same-cluster async effect
            // between the synchronous prevalidation and numeric commit. The
            // footprint was still approved before numeric execution; validate
            // it once more against the newer linearization point before
            // merging its shadow update.
            let global_warp_id = operation.id().global_warp_id();
            let direct_geometry = state.shadow.compact_direct_geometry(batch);
            let supports_direct_segment = direct_geometry.is_some();
            let must_flush_pending = state
                .pending_direct_segment
                .as_ref()
                .is_some_and(|pending| {
                    pending.global_warp_id != global_warp_id || !supports_direct_segment
                });
            if must_flush_pending {
                Self::commit_pending_direct_segment(&mut state);
            }
            let lane_result = state.lane_shadow.validate_compact_order_batch(batch);
            staged.lane_validation = match lane_result {
                Ok(validation) => validation,
                Err(reason) => {
                    push_unique_incomplete(
                        &mut state.incomplete_reasons,
                        RaceCheckIncompleteReason::ShadowRejected {
                            operation: operation.id().clone(),
                            reason: reason.clone(),
                        },
                    );
                    return Err(EngineError::message(format!(
                        "racecheck could not commit same-warp lane order for {}: {reason}",
                        operation.id()
                    )));
                }
            };
            let validation = {
                let RaceCheckState {
                    lane_shadow,
                    pending_direct_segment,
                    shadow,
                    ..
                } = &mut *state;
                let lane_order = lane_shadow
                    .race_lane_order(global_warp_id)
                    .expect("staged compact lane shadow has an order for this warp");
                if supports_direct_segment {
                    match pending_direct_segment.as_ref() {
                        Some(pending) => shadow.validate_compact_batch_for_direct_segment(
                            batch,
                            &lane_order,
                            direct_geometry.expect("direct geometry was already resolved"),
                            &pending.shadow_segment,
                        ),
                        None => shadow.validate_compact_batch_for_new_direct_segment(
                            batch,
                            &lane_order,
                            direct_geometry.expect("direct geometry was already resolved"),
                        ),
                    }
                } else {
                    shadow.validate_compact_batch_for_direct_commit(
                        batch,
                        &lane_order,
                        direct_geometry,
                    )
                }
            };
            match validation {
                Ok(validation) => validation,
                Err(RaceShadowError::Race(finding)) => {
                    let message =
                        format!("racecheck rejected {} at commit: {finding}", operation.id());
                    state.findings.push(finding);
                    return Err(EngineError::message(message));
                }
                Err(error) => {
                    push_unique_incomplete(
                        &mut state.incomplete_reasons,
                        RaceCheckIncompleteReason::ShadowRejected {
                            operation: operation.id().clone(),
                            reason: error.to_string(),
                        },
                    );
                    return Err(EngineError::message(format!(
                        "racecheck could not commit {}: {error}",
                        operation.id()
                    )));
                }
            }
        };
        let (committed_segment, shadow_reviews) = {
            let RaceCheckState {
                lane_shadow,
                shadow,
                ..
            } = &mut *state;
            let lane_order = lane_shadow
                .race_lane_order(operation.id().global_warp_id())
                .expect("staged compact lane shadow has an order for this warp");
            shadow.commit_compact_validation(shadow_validation, batch, &lane_order)
        };
        commit_review_findings(&mut state, shadow_reviews);
        {
            if let Some(segment) = committed_segment {
                if let Some(pending) = state.pending_direct_segment.as_mut() {
                    debug_assert_eq!(pending.global_warp_id, operation.id().global_warp_id());
                    pending
                        .lane_validation
                        .merge_order_validation(staged.lane_validation);
                } else {
                    state.pending_direct_segment = Some(PendingDirectSegment {
                        global_warp_id: operation.id().global_warp_id(),
                        lane_validation: staged.lane_validation,
                        shadow_segment: segment,
                    });
                }
            } else {
                debug_assert!(state.pending_direct_segment.is_none());
                state.lane_shadow.commit_batch(staged.lane_validation);
            }
        }
        state.access_count = state.access_count.saturating_add(1);
        state.alias_tracker.observe_compact_batch(batch);
        debug_assert!(
            !self.records_resolved_transitions(),
            "compact physical batches do not retain an owned transition payload"
        );
        Ok(())
    }

    pub fn result(&self) -> RaceCheckResult {
        self.result_with_terminal_validation(true, &[])
    }

    pub fn result_before_aborted_execution(&self) -> RaceCheckResult {
        self.result_with_terminal_validation(false, &[])
    }

    pub fn result_for_execution(&self, execution: &ExecutionReport) -> RaceCheckResult {
        let participant_exits = execution.cluster_barrier_participant_exit_evidence();
        self.result_with_terminal_validation(execution.is_success(), &participant_exits)
    }

    fn result_with_terminal_validation(
        &self,
        validate_terminal_state: bool,
        participant_exits: &[ClusterBarrierParticipantExitEvidence],
    ) -> RaceCheckResult {
        if self.race.direct_compact {
            self.flush_all_pending_direct();
        }
        let has_global_error_finding = self
            .race
            .global_shared
            .findings()
            .iter()
            .any(|finding| !finding.requires_unwaited_tmem_load_review());
        let has_error_finding = has_global_error_finding
            || self.race.race_shards.iter().any(|shard| {
                shard
                    .state
                    .lock()
                    .expect("race-check state poisoned")
                    .findings
                    .iter()
                    .any(|finding| !finding.requires_unwaited_tmem_load_review())
            });
        let sync = if !participant_exits.is_empty() && !has_error_finding {
            self.sync
                .result_before_aborted_execution_with_cluster_barrier_participant_exits(
                    participant_exits,
                )
        } else if !validate_terminal_state || has_error_finding {
            self.sync.result_before_aborted_execution()
        } else {
            self.sync.result()
        };
        let mut findings = Vec::new();
        let mut scope_diagnostics = Vec::new();
        let mut declared_word_bypasses = Vec::new();
        let mut undeclared_protocol_words = Vec::new();
        let mut advisories = Vec::new();
        let mut accesses = Vec::new();
        let mut incomplete_reasons = Vec::new();
        let mut access_count = 0_usize;
        let mut accesses_complete = true;
        for shard in &self.race.race_shards {
            access_count = access_count.saturating_add(
                shard
                    .uncontrolled_access_count
                    .load(AtomicOrdering::Relaxed),
            );
            let state = shard.state.lock().expect("race-check state poisoned");
            append_report_findings(&mut findings, state.findings.iter().cloned());
            append_report_advisories(&mut advisories, state.alias_tracker.advisories());
            accesses.extend(state.accesses.iter().cloned());
            incomplete_reasons.extend(state.incomplete_reasons.iter().cloned());
            access_count = access_count.saturating_add(state.access_count);
            accesses_complete &= state.retain_accesses;
            if validate_terminal_state {
                if let Some(staged) = state.staged_compact_access.as_ref() {
                    incomplete_reasons.push(RaceCheckIncompleteReason::EffectCommitUnobserved {
                        operation: staged.operation.as_ref().clone(),
                        effect: "compact_physical_access",
                    });
                }
                incomplete_reasons.extend(state.staged_accesses.keys().map(|operation| {
                    RaceCheckIncompleteReason::EffectCommitUnobserved {
                        operation: operation.clone(),
                        effect: "physical_access",
                    }
                }));
                incomplete_reasons.extend(state.staged_async_payloads.keys().map(|operation| {
                    RaceCheckIncompleteReason::EffectCommitUnobserved {
                        operation: operation.clone(),
                        effect: "async_payload",
                    }
                }));
                incomplete_reasons.extend(state.staged_async_completions.keys().map(|token| {
                    RaceCheckIncompleteReason::EffectCommitUnobserved {
                        operation: token.issue_operation().clone(),
                        effect: "async_payload_completion",
                    }
                }));
                incomplete_reasons.extend(state.staged_async_group_issues.keys().map(
                    |operation| RaceCheckIncompleteReason::EffectCommitUnobserved {
                        operation: operation.clone(),
                        effect: "async_group_issue",
                    },
                ));
                incomplete_reasons.extend(
                    state
                        .staged_async_group_completions
                        .values()
                        .flat_map(|completion| completion.tokens.iter())
                        .map(|token| RaceCheckIncompleteReason::EffectCommitUnobserved {
                            operation: token.issue_operation().clone(),
                            effect: "async_group_completion",
                        }),
                );
                incomplete_reasons.extend(state.staged_arrives.keys().map(|operation| {
                    RaceCheckIncompleteReason::EffectCommitUnobserved {
                        operation: operation.clone(),
                        effect: "mbarrier.arrive",
                    }
                }));
                incomplete_reasons.extend(
                    state
                        .tcgen_work_tokens
                        .values()
                        // An outstanding tcgen05.ld has already published its
                        // TMEM read witness. If no later conflicting TMEM
                        // access reached the review path, warp termination
                        // closes that read lifetime without requiring
                        // register-provenance modeling. Store and commit-bound
                        // work can still leave memory effects unobserved.
                        .filter(|work| work.kind != TcgenWorkKind::Load)
                        .map(|work| RaceCheckIncompleteReason::EffectCommitUnobserved {
                            operation: work.operation.clone(),
                            effect: "tcgen work completion",
                        }),
                );
                incomplete_reasons.extend(state.arrival_completion_payloads.values().map(
                    |pending| RaceCheckIncompleteReason::EffectCommitUnobserved {
                        operation: pending.operation.clone(),
                        effect: "tcgen05.commit completion",
                    },
                ));
            }
        }
        if self.race.global_memory_model_enabled {
            let shared = &self.race.global_shared;
            append_report_findings(&mut findings, shared.findings());
            let (retained, dropped) = shared.findings_retained();
            if dropped > 0 {
                push_unique_incomplete(
                    &mut incomplete_reasons,
                    RaceCheckIncompleteReason::FindingsTruncated { retained, dropped },
                );
            }
            scope_diagnostics.extend(shared.scope_diagnostics());
            declared_word_bypasses.extend(shared.declared_word_bypasses());
            undeclared_protocol_words.extend(shared.undeclared_protocol_words());
            for reason in shared.incomplete_reasons() {
                push_unique_incomplete(&mut incomplete_reasons, reason);
            }
            if validate_terminal_state {
                for shard in &self.race.race_shards {
                    let global = shard
                        .global
                        .lock()
                        .expect("racecheck global state poisoned");
                    for operation in global.staged_operations() {
                        push_unique_incomplete(
                            &mut incomplete_reasons,
                            RaceCheckIncompleteReason::EffectCommitUnobserved {
                                operation: operation.clone(),
                                effect: "global_physical_access",
                            },
                        );
                    }
                }
            }
        }
        let status = combine_peer_verdicts(
            sync.status(),
            &RaceVerdictInputs {
                saw_error: false,
                saw_incomplete: false,
                saw_review: false,
                findings: &findings,
                scope_diagnostics: &scope_diagnostics,
                declared_word_bypasses: &declared_word_bypasses,
                advisories: &advisories,
                incomplete_reasons: &incomplete_reasons,
            },
        );
        RaceCheckResult {
            status,
            sync,
            findings: findings.into_boxed_slice(),
            scope_diagnostics: scope_diagnostics.into_boxed_slice(),
            declared_word_bypasses: declared_word_bypasses.into_boxed_slice(),
            undeclared_protocol_words: undeclared_protocol_words.into_boxed_slice(),
            advisories: advisories.into_boxed_slice(),
            access_count,
            accesses_complete,
            accesses: accesses.into_boxed_slice(),
            incomplete_reasons: incomplete_reasons.into_boxed_slice(),
            global_memory_model_enabled: self.race.global_memory_model_enabled,
        }
    }

    fn defers_plain_compact_write(batch: &CompactPhysicalAccessBatch<'_>) -> bool {
        batch.descriptor().kind() == PhysicalAccessKind::Write
            && batch.descriptor().memory_semantics() == MemoryAccessSemantics::plain()
    }

    fn before_effect(
        &self,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        if self.race.direct_compact {
            self.flush_pending_direct_for_operation(operation)?;
        }
        let staged_global = self.race.global_memory_model_enabled
            && matches!(
                effect,
                OperationEffect::PhysicalAccess(batch)
                    if batch.descriptor().space().has_read_from_versions()
            );
        let global_write_spans = match effect {
            OperationEffect::PhysicalAccess(batch)
                if self.race.global_memory_model_enabled
                    && batch.descriptor().space() == PhysicalAccessSpace::Global
                    && batch.descriptor().kind().writes() =>
            {
                batch
                    .lanes()
                    .iter()
                    .flat_map(|lane| lane.footprint().spans())
                    .copied()
                    .collect::<Vec<_>>()
            }
            _ => Vec::new(),
        };
        if let OperationEffect::PhysicalAccess(batch) = effect {
            if batch.descriptor().space() == PhysicalAccessSpace::Global {
                self.wait_for_quiescent_atomic_access(
                    batch.descriptor().memory_semantics(),
                    batch
                        .lanes()
                        .iter()
                        .flat_map(|lane| lane.footprint().spans())
                        .copied(),
                );
            }
        }
        self.begin_global_write_spans(global_write_spans.iter().copied());
        if let OperationEffect::PhysicalAccess(batch) = effect {
            if self.race.global_memory_model_enabled
                && batch.descriptor().space().has_read_from_versions()
            {
                let tcgen_publications = self.tcgen_publications_for_operation(
                    operation,
                    tcgen_publication_required(batch.descriptor()),
                )?;
                let shared_frontier = batch
                    .descriptor()
                    .memory_semantics()
                    .order()
                    .has_release()
                    .then(|| self.shared_frontier_for_operation(operation))
                    .transpose()?;
                let result = self
                    .global_for_warp(operation.id().global_warp_id())?
                    .before_batch(batch, &tcgen_publications, shared_frontier.as_ref())
                    .map_err(EngineError::message);
                match result {
                    Ok((tcgen_acquisitions, global_frontier)) => {
                        self.merge_tcgen_acquisitions(operation, tcgen_acquisitions)?;
                        self.merge_global_frontier(operation, &global_frontier)?;
                    }
                    Err(error) => {
                        self.finish_global_write_spans(global_write_spans.iter().copied());
                        return Err(error);
                    }
                }
            }
        }
        if let OperationEffect::AsyncPayload(payload) = effect {
            return self.before_async_payload(operation, payload, effect);
        }
        if let OperationEffect::AsyncGroupIssue(issue) = effect {
            return self.before_async_group_issue(operation, issue, effect);
        }
        if let OperationEffect::AsyncGroupIssueBatch(batch) = effect {
            return self.before_async_group_issue_batch(operation, batch, effect);
        }
        if let OperationEffect::TcgenWorkIssue(issue) = effect {
            self.peer_before_effect(operation, effect)?;
            return self.commit_tcgen_work_issue_at_issue(operation, issue);
        }
        if let Err(error) = self.peer_before_effect(operation, effect) {
            if staged_global {
                self.global_for_warp(operation.id().global_warp_id())?
                    .discard_batch(operation.id());
            }
            self.finish_global_write_spans(global_write_spans);
            return Err(error);
        }
        if let OperationEffect::PhysicalAccess(batch) = effect {
            if !self.tracks_race_conflict_batch(batch) {
                return Ok(());
            }
        }

        let mut state = self.race_for_operation(operation)?;
        match effect {
            OperationEffect::PhysicalAccess(batch) => {
                if state.staged_accesses.contains_key(operation.id()) {
                    return Err(EngineError::message(format!(
                        "racecheck operation {} already has a staged physical access",
                        operation.id()
                    )));
                }
                let lane_result = state.lane_shadow.validate_order_batch(batch);
                let lane_validation = match lane_result {
                    Ok(validation) => validation,
                    Err(reason) => {
                        push_unique_incomplete(
                            &mut state.incomplete_reasons,
                            RaceCheckIncompleteReason::ShadowRejected {
                                operation: operation.id().clone(),
                                reason: reason.clone(),
                            },
                        );
                        return Err(EngineError::message(format!(
                            "racecheck could not validate same-warp lane order for {}: {reason}",
                            operation.id()
                        )));
                    }
                };
                let RaceCheckState {
                    lane_shadow,
                    shadow,
                    ..
                } = &mut *state;
                let lane_order = lane_shadow
                    .race_lane_order(operation.id().global_warp_id())
                    .expect("validated lane shadow has an order for this warp");
                let shadow_result = shadow.validate_batch_with_lane_order(batch, &lane_order);
                match shadow_result {
                    Ok(validation) => {
                        // The synchronous numeric access cannot yield, but a
                        // completion pump on another cluster worker may still
                        // mutate this cluster's shadow before `after_effect`.
                        // Retain the sparse validation and reuse it when the
                        // shadow revision proves no such mutation occurred.
                        let shadow_revision = state.shadow.revision();
                        state.staged_accesses.insert(
                            operation.id().clone(),
                            StagedAccess {
                                lane_validation,
                                shadow_revision,
                                shadow_validation: validation,
                            },
                        );
                    }
                    Err(RaceShadowError::Race(finding)) => {
                        let message = race_rejection_message(operation.id(), &finding);
                        state.findings.push(finding);
                        return Err(EngineError::message(message));
                    }
                    Err(error) => {
                        push_unique_incomplete(
                            &mut state.incomplete_reasons,
                            RaceCheckIncompleteReason::ShadowRejected {
                                operation: operation.id().clone(),
                                reason: error.to_string(),
                            },
                        );
                        return Err(EngineError::message(format!(
                            "racecheck could not validate {}: {error}",
                            operation.id()
                        )));
                    }
                }
            }
            OperationEffect::MbarrierArrive { plan, .. } => {
                let Some(generation) = self.sync.staged_mbarrier_arrive_generation(operation.id())
                else {
                    push_unique_incomplete(
                        &mut state.incomplete_reasons,
                        RaceCheckIncompleteReason::BarrierGenerationUnavailable {
                            operation: operation.id().clone(),
                            barrier_id: plan.barrier_id(),
                        },
                    );
                    return Err(EngineError::message(format!(
                        "racecheck could not resolve generation for {}",
                        operation.id()
                    )));
                };
                state.staged_arrives.insert(
                    operation.id().clone(),
                    Box::new([StagedArrive {
                        barrier_id: plan.barrier_id(),
                        generation,
                        warp_id: plan.warp_id(),
                        active_mask: operation.active_mask(),
                    }]),
                );
            }
            OperationEffect::MbarrierArriveBatch { plan, .. } => {
                let Some(generations) =
                    self.sync.staged_mbarrier_arrive_generations(operation.id())
                else {
                    let barrier_id = plan.entries()[0].plan().barrier_id();
                    push_unique_incomplete(
                        &mut state.incomplete_reasons,
                        RaceCheckIncompleteReason::BarrierGenerationUnavailable {
                            operation: operation.id().clone(),
                            barrier_id,
                        },
                    );
                    return Err(EngineError::message(format!(
                        "racecheck could not resolve batch generations for {}",
                        operation.id(),
                    )));
                };
                if generations.len() != plan.entries().len() {
                    return Err(EngineError::message(format!(
                        "racecheck resolved {} generations for {} mbarrier batch targets at {}",
                        generations.len(),
                        plan.entries().len(),
                        operation.id(),
                    )));
                }
                let mut arrives = Vec::with_capacity(plan.entries().len());
                for (entry, (resolved_barrier_id, generation)) in
                    plan.entries().iter().zip(generations)
                {
                    let target = entry.plan();
                    if target.barrier_id() != resolved_barrier_id {
                        return Err(EngineError::message(format!(
                            "racecheck generation order disagrees with mbarrier batch at {}: {:?} vs {:?}",
                            operation.id(),
                            target.barrier_id(),
                            resolved_barrier_id,
                        )));
                    }
                    arrives.push(StagedArrive {
                        barrier_id: target.barrier_id(),
                        generation,
                        warp_id: target.warp_id(),
                        active_mask: entry.arrival_mask(),
                    });
                }
                state
                    .staged_arrives
                    .insert(operation.id().clone(), arrives.into_boxed_slice());
            }
            OperationEffect::MbarrierInvalidate { .. }
            | OperationEffect::MbarrierInit(_)
            | OperationEffect::MbarrierExpectTx { .. }
            | OperationEffect::TcgenWait { .. }
            | OperationEffect::AsyncGroupCommit { .. }
            | OperationEffect::CpAsyncMbarrierArrive { .. }
            | OperationEffect::AsyncGroupWait { .. }
            | OperationEffect::MemoryFence(_)
            | OperationEffect::ProxyAsyncFence(_)
            | OperationEffect::TensorMap(_)
            | OperationEffect::MbarrierInitFence { .. }
            | OperationEffect::WarpSync(_)
            | OperationEffect::TcgenFence(_)
            | OperationEffect::MbarrierWait { .. }
            | OperationEffect::DeclaredWordWait { .. }
            | OperationEffect::MbarrierCompletionIssue { .. }
            | OperationEffect::TcgenCommitIssue { .. }
            | OperationEffect::AnalysisGap(_)
            | OperationEffect::NamedBarrierArrive { .. }
            | OperationEffect::NamedBarrierSyncRegister { .. }
            | OperationEffect::NamedBarrierSyncResume(_)
            | OperationEffect::ClusterBarrierArrive { .. }
            | OperationEffect::ClusterBarrierWaitRegister { .. }
            | OperationEffect::ClusterBarrierWaitResume(_)
            | OperationEffect::TcgenLifecycleRegister(_)
            | OperationEffect::TcgenLifecycleResume(_)
            | OperationEffect::SetmaxnregRegister(_)
            | OperationEffect::SetmaxnregResume(_) => {}
            OperationEffect::AsyncPayload(_) => unreachable!(),
            OperationEffect::AsyncGroupIssue(_) => unreachable!(),
            OperationEffect::AsyncGroupIssueBatch(_) => unreachable!(),
            OperationEffect::TcgenWorkIssue(_) => unreachable!(),
        }
        Ok(())
    }

    fn before_async_group_issue(
        &self,
        operation: &OperationContext,
        issue: &crate::AsyncGroupIssueEffect,
        effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        {
            let mut state = self.race_for_operation(operation)?;
            if self.race.direct_compact {
                Self::commit_pending_direct_segment(&mut state);
            }
            if state.staged_async_group_issues.contains_key(operation.id())
                || state.async_group_token_clocks.contains_key(issue.token())
            {
                return Err(EngineError::message(format!(
                    "racecheck async-group token {:?} is already staged or active",
                    issue.token()
                )));
            }
            let lane_order = state
                .lane_shadow
                .race_lane_order(operation.id().global_warp_id())
                .map_err(EngineError::message)?;
            state
                .shadow
                .preview_async_token_after_clock(
                    operation.id().global_warp_id(),
                    operation.active_mask(),
                    issue.token(),
                    None,
                    Some(&lane_order),
                )
                .map_err(|error| EngineError::message(error.to_string()))?;
            state.staged_async_group_issues.insert(
                operation.id().clone(),
                StagedAsyncGroupIssue {
                    tokens: Box::new([issue.token().clone()]),
                },
            );
        }
        if let Err(error) =
            <SyncCheckMode as EngineModeImpl>::before_effect(&self.sync, operation, effect)
        {
            self.race_for_operation(operation)?
                .staged_async_group_issues
                .remove(operation.id());
            return Err(error);
        }
        Ok(())
    }

    fn before_async_group_issue_batch(
        &self,
        operation: &OperationContext,
        batch: &crate::AsyncGroupIssueBatchEffect,
        effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        {
            let mut state = self.race_for_operation(operation)?;
            if self.race.direct_compact {
                Self::commit_pending_direct_segment(&mut state);
            }
            if state.staged_async_group_issues.contains_key(operation.id())
                || batch
                    .members()
                    .iter()
                    .any(|member| state.async_group_token_clocks.contains_key(member.token()))
            {
                return Err(EngineError::message(format!(
                    "racecheck async-group batch at {} is already staged or active",
                    operation.id()
                )));
            }
            for member in batch.members() {
                let lane_order = state
                    .lane_shadow
                    .race_lane_order(operation.id().global_warp_id())
                    .map_err(EngineError::message)?;
                state
                    .shadow
                    .preview_async_token_after_clock(
                        operation.id().global_warp_id(),
                        member.operation().active_mask(),
                        member.token(),
                        None,
                        Some(&lane_order),
                    )
                    .map_err(|error| EngineError::message(error.to_string()))?;
            }
            state.staged_async_group_issues.insert(
                operation.id().clone(),
                StagedAsyncGroupIssue {
                    tokens: batch
                        .members()
                        .iter()
                        .map(|member| member.token().clone())
                        .collect(),
                },
            );
        }
        if let Err(error) = self.peer_before_effect(operation, effect) {
            self.race_for_operation(operation)?
                .staged_async_group_issues
                .remove(operation.id());
            return Err(error);
        }
        Ok(())
    }

    fn before_async_payload(
        &self,
        operation: &OperationContext,
        payload: &crate::AsyncPayloadEffect,
        effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        {
            let mut state = self.race_for_operation(operation)?;
            if state.staged_accesses.contains_key(operation.id())
                || state.staged_async_payloads.contains_key(operation.id())
            {
                return Err(EngineError::message(format!(
                    "racecheck operation {} already has a staged memory effect",
                    operation.id()
                )));
            }

            let retain_accesses = state.retain_accesses;
            let tracked_access_count = payload
                .issue_accesses()
                .iter()
                .filter(|batch| self.tracks_race_conflict_batch(batch))
                .fold(0_usize, |count, batch| {
                    count.saturating_add(batch.semantic_access_count())
                });
            let skipped_access_count = payload
                .issue_accesses()
                .iter()
                .filter(|batch| !self.tracks_race_conflict_batch(batch))
                .fold(0_usize, |count, batch| {
                    count.saturating_add(batch.semantic_access_count())
                });
            let records = if retain_accesses {
                access_records(payload.issue_accesses())
            } else {
                Vec::new()
            };
            let compact_batches = match coalesce_physical_access_batches(
                payload
                    .issue_accesses()
                    .iter()
                    .filter(|batch| self.tracks_race_conflict_batch(batch)),
            ) {
                Ok(batches) => batches,
                Err(error) => {
                    push_unique_incomplete(
                        &mut state.incomplete_reasons,
                        RaceCheckIncompleteReason::ShadowRejected {
                            operation: operation.id().clone(),
                            reason: format!("could not compact async footprints: {error}"),
                        },
                    );
                    return Err(EngineError::message(format!(
                        "racecheck could not compact async payload {}: {error}",
                        operation.id()
                    )));
                }
            };
            let shadow_revision = state.shadow.revision();
            let RaceCheckState {
                lane_shadow,
                shadow,
                ..
            } = &mut *state;
            let lane_order = lane_shadow
                .race_lane_order(operation.id().global_warp_id())
                .map_err(EngineError::message)?;
            let shadow_validation = match shadow.validate_async_issue_batches(
                operation.id().global_warp_id(),
                operation.active_mask(),
                payload.token(),
                compact_batches.iter(),
                Some(&lane_order),
            ) {
                Ok(validation) => validation,
                Err(RaceShadowError::Race(finding)) => {
                    let message = race_rejection_message(operation.id(), &finding);
                    state.findings.push(finding);
                    return Err(EngineError::message(message));
                }
                Err(error) => {
                    push_unique_incomplete(
                        &mut state.incomplete_reasons,
                        RaceCheckIncompleteReason::ShadowRejected {
                            operation: operation.id().clone(),
                            reason: error.to_string(),
                        },
                    );
                    return Err(EngineError::message(format!(
                        "racecheck could not validate async payload {}: {error}",
                        operation.id()
                    )));
                }
            };
            state.staged_async_payloads.insert(
                operation.id().clone(),
                StagedAsyncPayload {
                    compact_batches,
                    records: records.into_boxed_slice(),
                    tracked_access_count,
                    skipped_access_count,
                    token: payload.token().clone(),
                    shadow_revision,
                    shadow_validation,
                },
            );
        }

        if let Err(error) = self.peer_before_effect(operation, effect) {
            self.race_for_operation(operation)?
                .staged_async_payloads
                .remove(operation.id());
            return Err(error);
        }
        Ok(())
    }

    fn commit_tcgen_work_issue_at_issue(
        &self,
        operation: &OperationContext,
        issue: &TcgenWorkIssue,
    ) -> Result<(), EngineError> {
        if issue.operation() != operation {
            return Err(EngineError::message(format!(
                "racecheck {} issue context {} disagrees with effect context {}",
                issue.kind().name(),
                issue.operation().id(),
                operation.id(),
            )));
        }
        let mut state = self.race_for_operation(operation)?;
        if state.tcgen_work_tokens.contains_key(issue.token())
            || state.reviewed_tcgen_load_tokens.contains_key(issue.token())
        {
            return Err(EngineError::message(format!(
                "racecheck {} token {:?} is already active",
                issue.kind().name(),
                issue.token(),
            )));
        }

        let (
            pipeline_descriptor,
            token_clock,
            records,
            tracked_access_count,
            skipped_access_count,
            compact_batches,
        ) = {
            let pipeline_descriptor = tcgen_pipeline_descriptor(issue);
            let pipeline_predecessor = tcgen_pipeline_predecessor(&mut state, issue)?;
            let RaceCheckState {
                lane_shadow,
                shadow,
                ..
            } = &mut *state;
            let lane_order = lane_shadow
                .race_lane_order(operation.id().global_warp_id())
                .map_err(EngineError::message)?;
            let token_clock = shadow
                .fork_tcgen_token_after_clock(
                    operation.id().global_warp_id(),
                    operation.active_mask(),
                    issue.token(),
                    pipeline_predecessor.as_ref(),
                    Some(&lane_order),
                )
                .map_err(|error| EngineError::message(error.to_string()))?;
            let retain_accesses = state.retain_accesses;
            let mut records = retain_accesses
                .then(|| Vec::with_capacity(issue.accesses().len()))
                .unwrap_or_default();
            let mut tracked_access_count = 0_usize;
            let mut skipped_access_count = 0_usize;
            if retain_accesses {
                records.extend(access_records(issue.accesses()));
            }
            for batch in issue.accesses() {
                if !self.tracks_race_conflict_batch(batch) {
                    skipped_access_count =
                        skipped_access_count.saturating_add(batch.semantic_access_count());
                    continue;
                }
                tracked_access_count =
                    tracked_access_count.saturating_add(batch.semantic_access_count());
            }
            let compact_batches = match coalesce_physical_access_batches(
                issue
                    .accesses()
                    .iter()
                    .filter(|batch| self.tracks_race_conflict_batch(batch)),
            ) {
                Ok(batches) => batches,
                Err(error) => {
                    push_unique_incomplete(
                        &mut state.incomplete_reasons,
                        RaceCheckIncompleteReason::ShadowRejected {
                            operation: operation.id().clone(),
                            reason: format!("could not compact TCGEN footprints: {error}"),
                        },
                    );
                    return Err(EngineError::message(format!(
                        "racecheck could not compact {} issue {:?}: {error}",
                        issue.kind().name(),
                        issue.token(),
                    )));
                }
            };
            (
                pipeline_descriptor,
                token_clock,
                records,
                tracked_access_count,
                skipped_access_count,
                compact_batches,
            )
        };
        let shadow_result = state.shadow.validate_and_commit_batches_at_clock(
            &compact_batches,
            &token_clock,
            issue.token(),
            issue.kind() == TcgenWorkKind::Load,
        );
        let review_findings = match shadow_result {
            Ok(findings) => findings,
            Err(RaceShadowError::Race(finding)) => {
                let message = format!(
                    "racecheck rejected {} issue {:?}: {finding}",
                    issue.kind().name(),
                    issue.token(),
                );
                state.findings.push(finding);
                return Err(EngineError::message(message));
            }
            Err(error) => {
                push_unique_incomplete(
                    &mut state.incomplete_reasons,
                    RaceCheckIncompleteReason::ShadowRejected {
                        operation: operation.id().clone(),
                        reason: error.to_string(),
                    },
                );
                return Err(EngineError::message(format!(
                    "racecheck could not validate {} issue {:?}: {error}",
                    issue.kind().name(),
                    issue.token(),
                )));
            }
        };
        // LD/ST use their dedicated waits rather than the MMA/CP/shift thread
        // fence frontier. Retaining only commit-bound work also keeps this map
        // bounded to descriptors that a later fence can consume.
        if issue.pipeline_operation().work_kind().uses_commit() {
            merge_tcgen_descriptor_frontier(
                state
                    .tcgen_pipeline_frontiers
                    .entry(tcgen_pipeline_key(issue))
                    .or_default(),
                pipeline_descriptor,
                &token_clock,
            )?;
            advance_tcgen_thread_epochs(&mut state, operation)?;
            state.tcgen_capture_cache.clear();
        }
        commit_staged_accesses(
            &mut state,
            records.into_boxed_slice(),
            tracked_access_count,
            skipped_access_count,
            issue
                .accesses()
                .iter()
                .filter(|batch| self.tracks_race_conflict_batch(batch)),
        );
        state.tcgen_work_tokens.insert(
            issue.token().clone(),
            ActiveTcgenWork {
                operation: operation.id().clone(),
                kind: issue.kind(),
            },
        );
        commit_review_findings(&mut state, review_findings);
        Ok(())
    }

    fn complete_tcgen_work_set(
        state: &mut RaceCheckState,
        work: &TcgenWorkSet,
    ) -> Result<Option<BarrierClockPayload>, EngineError> {
        let mut active_tokens = Vec::with_capacity(work.tokens().len());
        for token in work.tokens() {
            let issue_operation = if let Some(active) = state.tcgen_work_tokens.get(token) {
                if active.kind != work.kind()
                    && !(work.kind() == TcgenWorkKind::Commit
                        && active.kind == TcgenWorkKind::MmaSharedARead)
                {
                    return Err(EngineError::message(format!(
                        "racecheck {} completion references {} token {token:?}",
                        work.kind().name(),
                        active.kind.name(),
                    )));
                }
                active_tokens.push(token.clone());
                &active.operation
            } else if let Some(operation) = state.reviewed_tcgen_load_tokens.get(token) {
                if work.kind() != TcgenWorkKind::Load {
                    return Err(EngineError::message(format!(
                        "racecheck {} completion references reviewed tcgen05.ld token {token:?}",
                        work.kind().name(),
                    )));
                }
                operation
            } else {
                return Err(EngineError::message(format!(
                    "racecheck {} completion references unknown token {token:?}",
                    work.kind().name(),
                )));
            };
            if issue_operation.global_warp_id() != work.global_warp_id() {
                return Err(EngineError::message(format!(
                    "racecheck {} token {token:?} belongs to warp {}, completion was planned for warp {}",
                    work.kind().name(),
                    issue_operation.global_warp_id(),
                    work.global_warp_id(),
                )));
            }
        }
        let completed_clocks = state
            .shadow
            .complete_async_actors(&active_tokens)
            .map_err(|error| EngineError::message(error.to_string()))?;
        let mut payload: Option<BarrierClockPayload> = None;
        for completed_clock in completed_clocks {
            let completed = BarrierClockPayload::from_clock(completed_clock);
            if let Some(payload) = &mut payload {
                payload
                    .merge(&completed)
                    .map_err(|error| EngineError::message(error.to_string()))?;
            } else {
                payload = Some(completed);
            }
        }
        for token in work.tokens() {
            if state.tcgen_work_tokens.remove(token).is_none() {
                state
                    .reviewed_tcgen_load_tokens
                    .remove(token)
                    .expect("validated reviewed TCGEN load remains known until its wait");
            }
        }
        Ok(payload)
    }

    fn after_effect(
        &self,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        // The embedded synchronization state has no RaceCheck physical-memory
        // semantics; its PhysicalAccess branch only owns Synccheck's
        // sync-relevant atomic lane-order gap. Commit physical accesses through
        // the race shadow while forwarding other effects to the sync state.
        if !matches!(effect, OperationEffect::PhysicalAccess(_)) {
            self.peer_after_effect(operation, effect)?;
        }
        if matches!(effect, OperationEffect::TcgenWorkIssue(_)) {
            return Ok(());
        }
        let mut captured_tcgen_frontiers = BTreeMap::new();
        let before_cta_group = match effect {
            OperationEffect::TcgenFence(TcgenFenceKind::BeforeThreadSync) => Some(None),
            OperationEffect::TcgenCommitIssue {
                work,
                actions: Some(_),
                ..
            } => Some(Some((
                work.cta_group().ok_or_else(|| {
                    EngineError::message(format!(
                        "racecheck tcgen05.commit at {} has no CTA group",
                        operation.id()
                    ))
                })?,
                work.kind(),
            ))),
            _ => None,
        };
        if let Some(commit_cta_group) = before_cta_group {
            let mut state = self.race_for_operation(operation)?;
            for lane in operation.active_mask() {
                captured_tcgen_frontiers.insert(
                    lane,
                    capture_tcgen_fence_frontier(&mut state, operation, lane, commit_cta_group)?,
                );
            }
        }
        match effect {
            OperationEffect::PhysicalAccess(batch)
                if self.race.global_memory_model_enabled
                    && batch.descriptor().space().has_read_from_versions() =>
            {
                if batch.descriptor().kind().writes() {
                    let write_spans = batch
                        .lanes()
                        .iter()
                        .flat_map(|lane| lane.footprint().spans())
                        .copied()
                        .filter(|_| batch.descriptor().space() == PhysicalAccessSpace::Global)
                        .collect::<Vec<_>>();
                    let result = self
                        .global_for_warp(operation.id().global_warp_id())?
                        .after_batch(batch)
                        .map_err(EngineError::message);
                    self.finish_global_write_spans(write_spans);
                    result?;
                }
                // Global reads commit their HB/read-from state in
                // `before_batch`, before the numerical load. They must still
                // fall through to the ordinary RaceCheck commit path so
                // access accounting, alias tracking, and an explicitly
                // requested journal retain the exact read metadata.
            }
            OperationEffect::MemoryFence(fence) => {
                let tcgen_publications =
                    self.tcgen_publications_for_operation(operation, fence.order().has_release())?;
                let shared = fence
                    .order()
                    .has_release()
                    .then(|| self.shared_frontier_for_operation(operation))
                    .transpose()?;
                let (tcgen_acquisitions, frontier) = self
                    .global_for_warp(operation.id().global_warp_id())?
                    .fence(operation, fence, &tcgen_publications, shared.as_ref())
                    .map_err(EngineError::message)?;
                self.merge_tcgen_acquisitions(operation, tcgen_acquisitions)?;
                self.merge_global_frontier(operation, &frontier)?;
            }
            _ => {}
        }
        if self.race.global_memory_model_enabled
            && matches!(
                effect,
                OperationEffect::MbarrierInit(_)
                    | OperationEffect::MbarrierInvalidate { .. }
                    | OperationEffect::MbarrierArrive { .. }
                    | OperationEffect::MbarrierArriveBatch { .. }
                    | OperationEffect::MbarrierWait { .. }
            | OperationEffect::DeclaredWordWait { .. }
                    | OperationEffect::NamedBarrierArrive { .. }
                    | OperationEffect::NamedBarrierSyncRegister { .. }
                    | OperationEffect::NamedBarrierSyncResume(_)
                    | OperationEffect::ClusterBarrierArrive { .. }
                    | OperationEffect::ClusterBarrierWaitResume(_)
                    | OperationEffect::TcgenCommitIssue {
                        actions: Some(_),
                        ..
                    }
            )
        {
            // Barrier effects do not access numeric global memory. The global
            // state mutex gives them a model order relative to physical-access
            // commits; a concurrent write from another warp may legally fall
            // on either side of that order, while same-warp execution is
            // already sequential. Numeric reads remain protected by the
            // global-memory transaction across the actual load.
            let mut global = self.global_for_warp(operation.id().global_warp_id())?;
            match effect {
                OperationEffect::MbarrierInit(plan) => {
                    global.reset_physical_barriers(plan.barrier_ids());
                }
                OperationEffect::MbarrierInvalidate { barrier_ids } => {
                    global.reset_physical_barriers(barrier_ids);
                }
                OperationEffect::MbarrierArrive {
                    plan,
                    outcome: Some(outcome),
                } => {
                    if plan.is_release() {
                        global
                            .physical_barrier_release(
                                plan.barrier_id(),
                                outcome.generation(),
                                plan.warp_id(),
                                operation.active_mask(),
                            )
                            .map_err(EngineError::message)?;
                    }
                    global.retain_physical_barrier_generations(
                        plan.barrier_id(),
                        outcome.generation(),
                        outcome.conditional_completed_generation(),
                    );
                }
                OperationEffect::MbarrierArriveBatch {
                    plan,
                    outcome: Some(outcome),
                } => {
                    for (entry, outcome) in plan.entries().iter().zip(outcome.outcomes()) {
                        if entry.plan().is_release() {
                            global
                                .physical_barrier_release(
                                    entry.plan().barrier_id(),
                                    outcome.generation(),
                                    entry.plan().warp_id(),
                                    entry.arrival_mask(),
                                )
                                .map_err(EngineError::message)?;
                        }
                        global.retain_physical_barrier_generations(
                            entry.plan().barrier_id(),
                            outcome.generation(),
                            outcome.conditional_completed_generation(),
                        );
                    }
                }
                OperationEffect::TcgenCommitIssue {
                    plan,
                    actions: Some(actions),
                    ..
                } => {
                    // Commit has already applied its implicit before fence.
                    // Its deferred arrival additionally certifies that the
                    // selected operations completed. A waiter cannot acquire
                    // either payload until that action actually fires.
                    for action in actions {
                        global
                            .physical_barrier_release(
                                action.barrier_id(),
                                action.generation(),
                                plan.warp_id(),
                                operation.active_mask(),
                            )
                            .map_err(EngineError::message)?;
                    }
                }
                OperationEffect::MbarrierWait {
                    plan,
                    outcome: Some(outcome),
                } => {
                    if let Some(generation) = outcome.completed_generation() {
                        global
                            .physical_barrier_acquire(
                               Some(operation.id()),
                                plan.barrier_id(),
                                generation,
                                plan.warp_id(),
                                operation.active_mask(),
                                plan.has_acquire(),
                            )
                            .map_err(EngineError::message)?;
                    }
                }
                OperationEffect::NamedBarrierArrive {
                    plan,
                    outcome: Some(outcome),
                } => {
                    global
                        .named_barrier_release(
                            plan.barrier_id(),
                            outcome.generation(),
                            plan.warp_id(),
                            plan.arrival_mask(),
                        )
                        .map_err(EngineError::message)?;
                }
                OperationEffect::NamedBarrierSyncRegister {
                    plan,
                    outcome: Some(outcome),
                } => {
                    global
                        .named_barrier_release(
                            plan.barrier_id(),
                            outcome.generation(),
                            plan.warp_id(),
                            plan.arrival_mask(),
                        )
                        .map_err(EngineError::message)?;
                }
                OperationEffect::NamedBarrierSyncResume(plan) => {
                    global
                        .named_barrier_acquire(
                           Some(operation.id()),
                            plan.barrier_id(),
                            plan.generation(),
                            plan.warp_id(),
                            plan.arrival_mask(),
                        )
                        .map_err(EngineError::message)?;
                }
                OperationEffect::ClusterBarrierArrive {
                    plan,
                    outcome: Some(outcome),
                } => {
                    global
                        .cluster_barrier_release(
                            plan.barrier_id(),
                            outcome.generation(),
                            plan.warp_id(),
                            plan.arrival_mask(),
                            plan.publishes_memory(),
                        )
                        .map_err(EngineError::message)?;
                }
                OperationEffect::ClusterBarrierWaitResume(plan)
                    if plan.plan().acquires_memory() =>
                {
                    global
                        .cluster_barrier_acquire(
                           Some(operation.id()),
                            plan.plan().barrier_id(),
                            plan.generation(),
                            plan.plan().warp_id(),
                            plan.plan().arrival_mask(),
                        )
                        .map_err(EngineError::message)?;
                }
                _ => {}
            }
        }

        let mut state = self.race_for_operation(operation)?;
        match effect {
            OperationEffect::TcgenWorkIssue(_) => unreachable!(),
            OperationEffect::PhysicalAccess(batch) => {
                if !self.tracks_race_conflict_batch(batch) {
                    let retained_records = prepare_access_batch_commit(&mut state, batch);
                    state.accesses.extend(retained_records);
                    if self.records_resolved_transitions() {
                        self.context
                            .transition_log()
                            .register_operation_effect(
                                operation,
                                OperationEffect::PhysicalAccess(batch),
                            )
                            .map_err(|error| {
                                EngineError::message(format!(
                                    "racecheck could not record resolved physical effect at {}: {error}",
                                    operation.id()
                                ))
                            })?;
                    }
                    return Ok(());
                }
                let staged = state
                    .staged_accesses
                    .remove(operation.id())
                    .ok_or_else(|| {
                        EngineError::message(format!(
                            "racecheck operation {} has no staged physical access",
                            operation.id()
                        ))
                    })?;
                let validation = if state.shadow.revision() == staged.shadow_revision {
                    staged.shadow_validation
                } else {
                    let RaceCheckState {
                        lane_shadow,
                        shadow,
                        ..
                    } = &mut *state;
                    let lane_order = lane_shadow
                        .race_lane_order(operation.id().global_warp_id())
                        .expect("staged lane shadow has an order for this warp");
                    let revalidated = shadow.validate_batch_with_lane_order(batch, &lane_order);
                    match revalidated {
                        Ok(validation) => validation,
                        Err(RaceShadowError::Race(finding)) => {
                            let message = format!(
                                "racecheck rejected {} at commit: {finding}",
                                operation.id()
                            );
                            state.findings.push(finding);
                            return Err(EngineError::message(message));
                        }
                        Err(error) => {
                            push_unique_incomplete(
                                &mut state.incomplete_reasons,
                                RaceCheckIncompleteReason::ShadowRejected {
                                    operation: operation.id().clone(),
                                    reason: error.to_string(),
                                },
                            );
                            return Err(EngineError::message(format!(
                                "racecheck could not commit {}: {error}",
                                operation.id()
                            )));
                        }
                    }
                };
                let retained_records = prepare_access_batch_commit(&mut state, batch);
                let shadow_reviews = state.shadow.commit_validation(validation);
                state.lane_shadow.commit_batch(staged.lane_validation);
                commit_review_findings(&mut state, shadow_reviews);
                state.accesses.extend(retained_records);
                if self.records_resolved_transitions() {
                    self.context
                        .transition_log()
                        .register_operation_effect(
                            operation,
                            OperationEffect::PhysicalAccess(batch),
                        )
                        .map_err(|error| {
                            EngineError::message(format!(
                                "racecheck could not record resolved physical effect at {}: {error}",
                                operation.id()
                            ))
                        })?;
                }
            }
            OperationEffect::AsyncPayload(payload) => {
                let staged = state
                    .staged_async_payloads
                    .remove(operation.id())
                    .ok_or_else(|| {
                        EngineError::message(format!(
                            "racecheck operation {} has no staged async payload",
                            operation.id()
                        ))
                    })?;
                if staged.token != *payload.token() {
                    return Err(EngineError::message(format!(
                        "racecheck async payload {} changed token before commit",
                        operation.id()
                    )));
                }
                let shadow_validation = if state.shadow.revision() == staged.shadow_revision {
                    staged.shadow_validation
                } else {
                    // A completion may advance the same cluster while the
                    // numeric async issue is in flight. The exact footprint
                    // was still approved before numeric execution; rebuild
                    // only when that newer linearization point exists.
                    let RaceCheckState {
                        lane_shadow,
                        shadow,
                        ..
                    } = &mut *state;
                    let lane_order = lane_shadow
                        .race_lane_order(operation.id().global_warp_id())
                        .map_err(EngineError::message)?;
                    match shadow.validate_async_issue_batches(
                        operation.id().global_warp_id(),
                        operation.active_mask(),
                        &staged.token,
                        staged.compact_batches.iter(),
                        Some(&lane_order),
                    ) {
                        Ok(validation) => validation,
                        Err(RaceShadowError::Race(finding)) => {
                            let message = format!(
                                "racecheck rejected async payload {} at commit: {finding}",
                                operation.id()
                            );
                            state.findings.push(finding);
                            return Err(EngineError::message(message));
                        }
                        Err(error) => {
                            push_unique_incomplete(
                                &mut state.incomplete_reasons,
                                RaceCheckIncompleteReason::ShadowRejected {
                                    operation: operation.id().clone(),
                                    reason: error.to_string(),
                                },
                            );
                            return Err(EngineError::message(format!(
                                "racecheck could not commit async payload {}: {error}",
                                operation.id()
                            )));
                        }
                    }
                };
                let (mut token_clock, review_findings) = state
                    .shadow
                    .commit_async_issue_validation(shadow_validation)
                    .map_err(|error| {
                        EngineError::message(format!(
                            "racecheck could not publish async payload {}: {error}",
                            operation.id()
                        ))
                    })?;
                commit_review_findings(&mut state, review_findings);
                commit_staged_accesses(
                    &mut state,
                    staged.records,
                    staged.tracked_access_count,
                    staged.skipped_access_count,
                    payload
                        .issue_accesses()
                        .iter()
                        .filter(|batch| self.tracks_race_conflict_batch(batch)),
                );
                for domain in async_proxy_domains(payload.issue_accesses())
                    .as_slice()
                    .iter()
                    .copied()
                {
                    token_clock
                        .apply_implicit_async_completion(domain)
                        .map_err(|error| EngineError::message(error.to_string()))?;
                }
                state.async_token_clocks.insert(staged.token, token_clock);
            }
            OperationEffect::AsyncGroupIssue(issue) => {
                let staged = state
                    .staged_async_group_issues
                    .remove(operation.id())
                    .ok_or_else(|| {
                        EngineError::message(format!(
                            "racecheck operation {} has no staged async-group issue",
                            operation.id()
                        ))
                    })?;
                if staged.tokens.as_ref() != [issue.token().clone()] {
                    return Err(EngineError::message(format!(
                        "racecheck async-group issue {} changed token before commit",
                        operation.id()
                    )));
                }
                let RaceCheckState {
                    lane_shadow,
                    shadow,
                    ..
                } = &mut *state;
                let lane_order = lane_shadow
                    .race_lane_order(operation.id().global_warp_id())
                    .map_err(EngineError::message)?;
                let issue_clock = shadow
                    .fork_async_token_after_clock(
                        operation.id().global_warp_id(),
                        operation.active_mask(),
                        issue.token(),
                        None,
                        Some(&lane_order),
                    )
                    .map_err(|error| EngineError::message(error.to_string()))?;
                state.async_group_token_clocks.insert(
                    issue.token().clone(),
                    AsyncGroupTokenClocks {
                        current: issue_clock,
                        source_read: None,
                        full: None,
                    },
                );
            }
            OperationEffect::AsyncGroupIssueBatch(batch) => {
                let staged = state
                    .staged_async_group_issues
                    .remove(operation.id())
                    .ok_or_else(|| {
                        EngineError::message(format!(
                            "racecheck operation {} has no staged async-group batch",
                            operation.id()
                        ))
                    })?;
                let tokens = batch
                    .members()
                    .iter()
                    .map(|member| member.token().clone())
                    .collect::<Vec<_>>();
                if staged.tokens.as_ref() != tokens.as_slice() {
                    return Err(EngineError::message(format!(
                        "racecheck async-group batch {} changed tokens before commit",
                        operation.id()
                    )));
                }
                for member in batch.members() {
                    let RaceCheckState {
                        lane_shadow,
                        shadow,
                        ..
                    } = &mut *state;
                    let lane_order = lane_shadow
                        .race_lane_order(operation.id().global_warp_id())
                        .map_err(EngineError::message)?;
                    let issue_clock = shadow
                        .fork_async_token_after_clock(
                            operation.id().global_warp_id(),
                            member.operation().active_mask(),
                            member.token(),
                            None,
                            Some(&lane_order),
                        )
                        .map_err(|error| EngineError::message(error.to_string()))?;
                    state.async_group_token_clocks.insert(
                        member.token().clone(),
                        AsyncGroupTokenClocks {
                            current: issue_clock,
                            source_read: None,
                            full: None,
                        },
                    );
                }
            }
            OperationEffect::AsyncGroupCommit { .. } => {}
            OperationEffect::AsyncGroupWait {
                outcome: Some(outcome),
                ..
            } => {
                let mut payload: Option<BarrierClockPayload> = None;
                let mut retired_tokens = Vec::new();
                for group in outcome.groups() {
                    for member in group.members() {
                        let clocks = state
                            .async_group_token_clocks
                            .get(member.token())
                            .ok_or_else(|| {
                                EngineError::message(format!(
                                    "racecheck async-group token {:?} has no clock state",
                                    member.token()
                                ))
                            })?;
                        let clock = match group.milestone() {
                            AsyncGroupMilestone::SourceReadComplete => clocks.source_read.as_ref(),
                            AsyncGroupMilestone::FullComplete => clocks.full.as_ref(),
                        }
                        .ok_or_else(|| {
                            EngineError::message(format!(
                                "racecheck async-group token {:?} has not reached {:?}",
                                member.token(),
                                group.milestone()
                            ))
                        })?;
                        let acquired = BarrierClockPayload::from_clock(clock.clone());
                        if let Some(payload) = &mut payload {
                            payload
                                .merge(&acquired)
                                .map_err(|error| EngineError::message(error.to_string()))?;
                        } else {
                            payload = Some(acquired);
                        }
                        if !outcome.plan().read_only() {
                            retired_tokens.push(member.token().clone());
                        }
                    }
                }
                if let Some(payload) = payload {
                    state
                        .shadow
                        .barrier_acquire_masked(
                            operation.id().global_warp_id(),
                            operation.active_mask(),
                            &payload,
                        )
                        .map_err(|error| EngineError::message(error.to_string()))?;
                }
                for token in retired_tokens {
                    state
                        .async_group_token_clocks
                        .remove(&token)
                        .ok_or_else(|| {
                            EngineError::message(format!(
                                "racecheck async-group token {token:?} disappeared before full-wait retirement"
                            ))
                        })?;
                }
            }
            OperationEffect::AsyncGroupWait { outcome: None, .. } => {
                return Err(EngineError::message(format!(
                    "racecheck observed uncommitted async-group wait at {}",
                    operation.id()
                )));
            }
            OperationEffect::MemoryFence(_) | OperationEffect::TensorMap(_) => {}
            // The engine owns the fenced set; synccheck consumes it. Racecheck
            // layers nothing on the fence itself.
            OperationEffect::MbarrierInitFence { .. } => {}
            // Synccheck owns mbarrier generation accounting. A standalone
            // expectation neither releases nor acquires a memory payload.
            OperationEffect::MbarrierExpectTx { .. } => {}
            // Intercepted by `RaceCheckMode::after_effect` before this dispatch
            // so it can take the global-transaction lock in the original order.
            OperationEffect::WarpSync(_) => {}
            OperationEffect::ProxyAsyncFence(proxy) => {
                let RaceCheckState {
                    lane_shadow,
                    shadow,
                    ..
                } = &mut *state;
                let lane_order = lane_shadow
                    .race_lane_order(operation.id().global_warp_id())
                    .map_err(EngineError::message)?;
                shadow
                    .proxy_async_fence_masked(
                        operation.id().global_warp_id(),
                        operation.active_mask(),
                        proxy.scope(),
                        Some(&lane_order),
                    )
                    .map_err(|error| EngineError::message(error.to_string()))?;
            }
            OperationEffect::TcgenFence(kind) => match kind {
                TcgenFenceKind::BeforeThreadSync => {
                    apply_captured_tcgen_fence(&mut state, operation, &captured_tcgen_frontiers)?;
                }
                TcgenFenceKind::AfterThreadSync => {
                    apply_transported_tcgen_fence(&mut state, operation)?;
                }
            },
            OperationEffect::AnalysisGap(gap) => {
                if !<RaceCheckMode as EngineModeImpl>::observes_analysis_gap(self, gap.kind()) {
                    return Ok(());
                }
                push_unique_incomplete(
                    &mut state.incomplete_reasons,
                    RaceCheckIncompleteReason::AnalysisGap {
                        operation: operation.id().clone(),
                        kind: gap.kind(),
                    },
                );
            }
            OperationEffect::MbarrierInit(plan) => {
                for barrier_id in plan.barrier_ids() {
                    state
                        .barrier_payloads
                        .retain(|(existing, _), _| existing != barrier_id);
                    state
                        .barrier_copy_payloads
                        .retain(|(existing, _), _| existing != barrier_id);
                    state
                        .barrier_lane_payloads
                        .retain(|(existing, _), _| existing != barrier_id);
                    state
                        .barrier_tcgen_payloads
                        .retain(|(existing, _), _| existing != barrier_id);
                }
            }
            OperationEffect::MbarrierInvalidate { barrier_ids } => {
                for id in barrier_ids {
                    state
                        .barrier_payloads
                        .retain(|(existing, _), _| existing != id);
                    state
                        .barrier_copy_payloads
                        .retain(|(existing, _), _| existing != id);
                    state
                        .barrier_lane_payloads
                        .retain(|(existing, _), _| existing != id);
                    state
                        .barrier_tcgen_payloads
                        .retain(|(existing, _), _| existing != id);
                }
            }
            OperationEffect::MbarrierArrive { plan, outcome } => {
                let arrives = state.staged_arrives.remove(operation.id()).ok_or_else(|| {
                    EngineError::message(format!(
                        "racecheck operation {} has no staged mbarrier arrival",
                        operation.id()
                    ))
                })?;
                let payload = plan
                    .is_release()
                    .then(|| {
                        state
                            .shadow
                            .barrier_release_masked(
                                operation.id().global_warp_id(),
                                operation.active_mask(),
                            )
                            .map_err(|error| EngineError::message(error.to_string()))
                    })
                    .transpose()?;
                for arrive in arrives {
                    publish_mbarrier_arrival(&mut state, operation, arrive, payload.as_ref())?;
                    let payload_key = (arrive.barrier_id, arrive.generation);
                    if let Some(outcome) = outcome {
                        retain_physical_barrier_payloads(
                            &mut state,
                            payload_key.0,
                            payload_key.1,
                            outcome.conditional_completed_generation(),
                        );
                    }
                }
            }
            OperationEffect::MbarrierArriveBatch { plan, outcome } => {
                let arrives = state.staged_arrives.remove(operation.id()).ok_or_else(|| {
                    EngineError::message(format!(
                        "racecheck operation {} has no staged mbarrier arrival batch",
                        operation.id()
                    ))
                })?;
                let payload = plan
                    .entries()
                    .iter()
                    .any(|entry| entry.plan().is_release())
                    .then(|| {
                        state
                            .shadow
                            .barrier_release_masked(
                                operation.id().global_warp_id(),
                                operation.active_mask(),
                            )
                            .map_err(|error| EngineError::message(error.to_string()))
                    })
                    .transpose()?;
                for (index, arrive) in arrives.iter().copied().enumerate() {
                    let release = if plan.entries()[index].plan().is_release() {
                        payload.as_ref()
                    } else {
                        None
                    };
                    publish_mbarrier_arrival(&mut state, operation, arrive, release)?;
                    let payload_key = (arrive.barrier_id, arrive.generation);
                    if let Some(outcome) = outcome.and_then(|batch| batch.outcomes().get(index)) {
                        retain_physical_barrier_payloads(
                            &mut state,
                            payload_key.0,
                            payload_key.1,
                            outcome.conditional_completed_generation(),
                        );
                    }
                }
            }
            OperationEffect::DeclaredWordWait { plan } => {
                if self.race.global_memory_model_enabled {
                    let (acquisitions, shared_frontier) = self
                        .global_for_warp(plan.warp_id())?
                        .apply_declared_word_wait(operation.id(), plan)
                        .map_err(EngineError::message)?;
                    // `state` above is this shard's race lock, held for the
                    // whole match; merge through it rather than taking it
                    // again.
                    merge_tcgen_acquisitions_into(&mut state, operation, acquisitions);
                    drop(state);
                    self.merge_global_frontier(operation, &shared_frontier)?;
                    state = self.race_for_operation(operation)?;
                }
            }
            OperationEffect::MbarrierWait { plan, outcome } => {
                let generation = outcome.and_then(|outcome| outcome.completed_generation());
                if let Some(generation) = generation {
                    // TCGEN completion is an execution observation, including
                    // for relaxed queries. Its existing after-thread-sync fence
                    // still controls when these incoming frontiers are usable.
                    let tcgen_payload = state
                        .barrier_tcgen_payloads
                        .get(&(plan.barrier_id(), generation))
                        .cloned()
                        .unwrap_or_default();
                    tcgen_acquire_mask(
                        &mut state,
                        operation,
                        plan.warp_id(),
                        operation.active_mask(),
                        &tcgen_payload,
                    )?;
                    let completion = state
                        .barrier_copy_payloads
                        .get(&(plan.barrier_id(), generation))
                        .cloned();
                    if let Some(completion) = &completion {
                        state
                            .shadow
                            .barrier_acquire_masked(
                                plan.warp_id(),
                                operation.active_mask(),
                                completion,
                            )
                            .map_err(|error| EngineError::message(error.to_string()))?;
                    }
                    if !plan.has_acquire() {
                        return Ok(());
                    }
                    let payload = state
                        .barrier_payloads
                        .get(&(plan.barrier_id(), generation))
                        .cloned();
                    let Some(payload) = payload else {
                        if completion.is_some() {
                            return Ok(());
                        }
                        push_unique_incomplete(
                            &mut state.incomplete_reasons,
                            RaceCheckIncompleteReason::BarrierPayloadUnavailable {
                                operation: operation.id().clone(),
                                barrier_id: plan.barrier_id(),
                                generation,
                            },
                        );
                        return Ok(());
                    };
                    state
                        .shadow
                        .barrier_acquire_masked(plan.warp_id(), operation.active_mask(), &payload)
                        .map_err(|error| EngineError::message(error.to_string()))?;
                    let lane_payload = state
                        .barrier_lane_payloads
                        .get(&(plan.barrier_id(), generation))
                        .cloned();
                    state
                        .lane_shadow
                        .barrier_acquire(
                            plan.warp_id(),
                            operation.active_mask(),
                            lane_payload.as_ref(),
                        )
                        .map_err(EngineError::message)?;
                }
            }
            OperationEffect::MbarrierCompletionIssue { plan, .. } => {
                if !plan.is_counter_only()
                    && plan
                        .completions()
                        .iter()
                        .any(|(_, transactions)| *transactions != 0)
                {
                    push_unique_incomplete(
                        &mut state.incomplete_reasons,
                        RaceCheckIncompleteReason::AsyncPayloadAccessUnmodeled {
                            operation: operation.id().clone(),
                        },
                    );
                }
            }
            OperationEffect::TcgenCommitIssue {
                work,
                actions: Some(actions),
                ..
            } => {
                // PTX defines commit as an implicit before-thread-sync fence
                // for the same-thread, same-CTA-group MMA/CP/shift operations
                // it tracks. Reuse the explicit fence path for that ordering;
                // the remainder of this branch models completion delivery.
                apply_captured_tcgen_fence(&mut state, operation, &captured_tcgen_frontiers)?;
                for action in actions {
                    let PhysicalCompletionKind::Arrival { .. } = action.kind() else {
                        return Err(EngineError::message(format!(
                            "racecheck tcgen05.commit at {} did not produce arrival completions",
                            operation.id()
                        )));
                    };
                    if state.arrival_completion_payloads.contains_key(&action.id()) {
                        return Err(EngineError::message(format!(
                            "racecheck tcgen05.commit action {} was reused",
                            action.id()
                        )));
                    }
                }
                let cta_group = work.cta_group().ok_or_else(|| {
                    EngineError::message(format!(
                        "racecheck tcgen05.commit at {} has no CTA group",
                        operation.id()
                    ))
                })?;
                // Completion upgrades only the work selected by this commit;
                // the immutable TCGEN payload is delivered when the deferred
                // mbarrier arrival actually completes.
                let completed_work = Self::complete_tcgen_work_set(&mut state, work)?;
                let frontier_key = tcgen_thread_pipeline_key(operation, cta_group);
                if let Some(completion_frontier) = completed_work {
                    match state
                        .tcgen_commit_frontiers
                        .entry((frontier_key.clone(), work.kind()))
                    {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert(completion_frontier);
                        }
                        std::collections::btree_map::Entry::Occupied(mut entry) => {
                            entry
                                .get_mut()
                                .merge(&completion_frontier)
                                .map_err(|error| EngineError::message(error.to_string()))?;
                        }
                    }
                }
                let mut payload = state
                    .shadow
                    .barrier_release_masked(
                        operation.id().global_warp_id(),
                        operation.active_mask(),
                    )
                    .map_err(|error| EngineError::message(error.to_string()))?;
                // Each target mbarrier tracks all prior commit-bound work from
                // the issuing thread, rather than popping a one-shot FIFO.
                // Preserve the cumulative completion frontier when the same
                // MMA is committed to more than one barrier.
                for kind in [TcgenWorkKind::Commit, TcgenWorkKind::MmaSharedARead] {
                    if work.kind() == TcgenWorkKind::MmaSharedARead && kind != work.kind() {
                        continue;
                    }
                    if let Some(completed_work) = state
                        .tcgen_commit_frontiers
                        .get(&(frontier_key.clone(), kind))
                    {
                        payload
                            .merge(completed_work)
                            .map_err(|error| EngineError::message(error.to_string()))?;
                    }
                }
                let lane_payload = state
                    .lane_shadow
                    .barrier_release(operation.id().global_warp_id(), operation.active_mask())
                    .map_err(EngineError::message)?;
                let mut tcgen_payload = tcgen_release_mask(
                    &state,
                    operation,
                    operation.id().global_warp_id(),
                    operation.active_mask(),
                )?;
                for captured in captured_tcgen_frontiers.values() {
                    tcgen_payload.merge(&captured.upgraded_to_completed()?)?;
                }
                for action in actions {
                    state.arrival_completion_payloads.insert(
                        action.id(),
                        ArrivalCompletionPayload {
                            operation: operation.id().clone(),
                            barrier_id: action.barrier_id(),
                            generation: action.generation(),
                            copy_completion: false,
                            payload: payload.clone(),
                            lane_payload: lane_payload.clone(),
                            tcgen_payload: tcgen_payload.clone(),
                        },
                    );
                }
            }
            OperationEffect::TcgenCommitIssue { actions: None, .. } => {
                return Err(EngineError::message(format!(
                    "racecheck observed uncommitted tcgen05.commit issue at {}",
                    operation.id()
                )));
            }
            OperationEffect::TcgenWait { work } => {
                if let Some(payload) = Self::complete_tcgen_work_set(&mut state, work)? {
                    let completion_frontier = payload.clock().clone();
                    state
                        .shadow
                        .barrier_acquire_masked(
                            operation.id().global_warp_id(),
                            operation.active_mask(),
                            &payload,
                        )
                        .map_err(|error| EngineError::message(error.to_string()))?;
                    // A wait pipelines the LD/ST operations it tracks with
                    // later same-thread TCGEN work. Completion also returns
                    // the tracked operation to ordinary thread execution; an
                    // explicit before fence is still needed before publishing
                    // it through a later inter-thread synchronization.
                    for lane in operation.active_mask() {
                        merge_tcgen_clock_entry(
                            &mut state.tcgen_wait_frontiers,
                            tcgen_thread_key(operation, lane),
                            &completion_frontier,
                        )?;
                    }
                    advance_tcgen_thread_epochs(&mut state, operation)?;
                    state.tcgen_capture_cache.clear();
                }
            }
            OperationEffect::NamedBarrierArrive {
                plan,
                outcome: Some(outcome),
            } => {
                let tcgen_payload =
                    tcgen_release_mask(&state, operation, plan.warp_id(), plan.arrival_mask())?;
                let payload = {
                    state
                        .shadow
                        .barrier_release_masked(plan.warp_id(), plan.arrival_mask())
                        .map_err(|error| EngineError::message(error.to_string()))?
                };
                let lane_payload = {
                    state
                        .lane_shadow
                        .barrier_release(plan.warp_id(), plan.arrival_mask())
                        .map_err(EngineError::message)?
                };
                let payload_key = (plan.barrier_id(), outcome.generation());
                match state.named_barrier_payloads.entry(payload_key) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(payload);
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        entry
                            .get_mut()
                            .merge(&payload)
                            .map_err(|error| EngineError::message(error.to_string()))?;
                    }
                }
                retire_named_barrier_generations(
                    &mut state.named_barrier_payloads,
                    payload_key.0,
                    payload_key.1,
                );
                merge_lane_barrier_payload(
                    &mut state.named_barrier_lane_payloads,
                    payload_key,
                    lane_payload,
                );
                merge_tcgen_barrier_payload(
                    &mut state.named_barrier_tcgen_payloads,
                    payload_key,
                    &tcgen_payload,
                )?;
            }
            OperationEffect::NamedBarrierArrive { outcome: None, .. } => {
                return Err(EngineError::message(format!(
                    "racecheck observed uncommitted bar.arrive registration at {}",
                    operation.id()
                )));
            }
            OperationEffect::NamedBarrierSyncRegister {
                plan,
                outcome: Some(outcome),
            } => {
                let tcgen_payload =
                    tcgen_release_mask(&state, operation, plan.warp_id(), plan.arrival_mask())?;
                let payload = {
                    state
                        .shadow
                        .barrier_release_masked(plan.warp_id(), plan.arrival_mask())
                        .map_err(|error| EngineError::message(error.to_string()))?
                };
                let lane_payload = {
                    state
                        .lane_shadow
                        .barrier_release(plan.warp_id(), plan.arrival_mask())
                        .map_err(EngineError::message)?
                };
                let payload_key = (plan.barrier_id(), outcome.generation());
                match state.named_barrier_payloads.entry(payload_key) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(payload);
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        entry
                            .get_mut()
                            .merge(&payload)
                            .map_err(|error| EngineError::message(error.to_string()))?;
                    }
                }
                retire_named_barrier_generations(
                    &mut state.named_barrier_payloads,
                    payload_key.0,
                    payload_key.1,
                );
                merge_lane_barrier_payload(
                    &mut state.named_barrier_lane_payloads,
                    payload_key,
                    lane_payload,
                );
                merge_tcgen_barrier_payload(
                    &mut state.named_barrier_tcgen_payloads,
                    payload_key,
                    &tcgen_payload,
                )?;
            }
            OperationEffect::NamedBarrierSyncRegister { outcome: None, .. } => {
                return Err(EngineError::message(format!(
                    "racecheck observed uncommitted bar.sync registration at {}",
                    operation.id()
                )));
            }
            OperationEffect::NamedBarrierSyncResume(plan) => {
                let payload = state
                    .named_barrier_payloads
                    .get(&(plan.barrier_id(), plan.generation()))
                    .cloned();
                let Some(payload) = payload else {
                    push_unique_incomplete(
                        &mut state.incomplete_reasons,
                        RaceCheckIncompleteReason::ShadowRejected {
                            operation: operation.id().clone(),
                            reason: format!(
                                "named barrier {:?} generation {} release payload is unavailable",
                                plan.barrier_id(),
                                plan.generation(),
                            ),
                        },
                    );
                    return Err(EngineError::message(format!(
                        "racecheck could not acquire named barrier {:?} generation {} at {}",
                        plan.barrier_id(),
                        plan.generation(),
                        operation.id(),
                    )));
                };
                state
                    .shadow
                    .barrier_acquire_masked(plan.warp_id(), plan.arrival_mask(), &payload)
                    .map_err(|error| EngineError::message(error.to_string()))?;
                let lane_payload = state
                    .named_barrier_lane_payloads
                    .get(&(plan.barrier_id(), plan.generation()))
                    .cloned();
                state
                    .lane_shadow
                    .barrier_acquire(plan.warp_id(), plan.arrival_mask(), lane_payload.as_ref())
                    .map_err(EngineError::message)?;
                let tcgen_payload = state
                    .named_barrier_tcgen_payloads
                    .get(&(plan.barrier_id(), plan.generation()))
                    .cloned()
                    .unwrap_or_default();
                tcgen_acquire_mask(
                    &mut state,
                    operation,
                    plan.warp_id(),
                    plan.arrival_mask(),
                    &tcgen_payload,
                )?;
            }
            OperationEffect::ClusterBarrierArrive {
                plan,
                outcome: Some(outcome),
            } => {
                let payload_key = (plan.barrier_id(), outcome.generation());
                let tcgen_payload =
                    tcgen_release_mask(&state, operation, plan.warp_id(), plan.arrival_mask())?;
                merge_tcgen_barrier_payload(
                    &mut state.cluster_barrier_tcgen_payloads,
                    payload_key,
                    &tcgen_payload,
                )?;
                if plan.publishes_memory() {
                    let payload = state
                        .shadow
                        .barrier_release_masked(plan.warp_id(), plan.arrival_mask())
                        .map_err(|error| EngineError::message(error.to_string()))?;
                    let lane_payload = state
                        .lane_shadow
                        .barrier_release(plan.warp_id(), plan.arrival_mask())
                        .map_err(EngineError::message)?;
                    merge_cluster_barrier_payload(
                        &mut state.cluster_barrier_payloads,
                        plan.barrier_id(),
                        outcome.generation(),
                        payload,
                    )?;
                    merge_lane_barrier_payload(
                        &mut state.cluster_barrier_lane_payloads,
                        payload_key,
                        lane_payload,
                    );
                }
            }
            OperationEffect::ClusterBarrierArrive { outcome: None, .. } => {
                return Err(EngineError::message(format!(
                    "racecheck observed uncommitted barrier.cluster.arrive at {}",
                    operation.id()
                )));
            }
            OperationEffect::ClusterBarrierWaitRegister { outcome: None, .. } => {
                return Err(EngineError::message(format!(
                    "racecheck observed uncommitted cluster-barrier registration at {}",
                    operation.id()
                )));
            }
            OperationEffect::ClusterBarrierWaitRegister {
                outcome: Some(_), ..
            } => {}
            OperationEffect::ClusterBarrierWaitResume(plan) => {
                if plan.plan().acquires_memory() {
                    let payload = state
                        .cluster_barrier_payloads
                        .get(&(plan.plan().barrier_id(), plan.generation()))
                        .cloned();
                    acquire_cluster_barrier_payload(
                        &mut state.shadow,
                        payload.as_ref(),
                        plan.plan().warp_id(),
                        plan.plan().arrival_mask(),
                    )?;
                    let lane_payload = state
                        .cluster_barrier_lane_payloads
                        .get(&(plan.plan().barrier_id(), plan.generation()))
                        .cloned();
                    state
                        .lane_shadow
                        .barrier_acquire(
                            plan.plan().warp_id(),
                            plan.plan().arrival_mask(),
                            lane_payload.as_ref(),
                        )
                        .map_err(EngineError::message)?;
                    let tcgen_payload = state
                        .cluster_barrier_tcgen_payloads
                        .get(&(plan.plan().barrier_id(), plan.generation()))
                        .cloned()
                        .unwrap_or_default();
                    tcgen_acquire_mask(
                        &mut state,
                        operation,
                        plan.plan().warp_id(),
                        plan.plan().arrival_mask(),
                        &tcgen_payload,
                    )?;
                }
            }
            OperationEffect::TcgenLifecycleRegister(_)
            | OperationEffect::TcgenLifecycleResume(_)
            | OperationEffect::SetmaxnregRegister(_)
            | OperationEffect::SetmaxnregResume(_) => {}
            OperationEffect::CpAsyncMbarrierArrive {
                plan,
                outcome: Some(outcome),
                actions: Some(actions),
                ..
            } => {
                let immediately_ready = outcome
                    .immediately_ready_physical_actions()
                    .iter()
                    .map(|action| action.id())
                    .collect::<BTreeSet<_>>();
                for ((lane, barrier_id), action) in plan.targets().zip(actions) {
                    if !immediately_ready.contains(&action.id()) {
                        continue;
                    }
                    if barrier_id != action.barrier_id() {
                        return Err(EngineError::message(format!(
                            "cp.async arrive-on action {} changed barrier identity",
                            action.id()
                        )));
                    }
                    // No pending work does not imply no prior copies: an
                    // explicit wait may already have retired their groups.
                    let payload = state
                        .cp_async_completed_payloads
                        .get(&(operation.id().global_warp_id(), lane))
                        .cloned()
                        .unwrap_or_else(|| {
                            BarrierClockPayload::from_clock(state.shadow.empty_clock())
                        });
                    if state
                        .arrival_completion_payloads
                        .insert(
                            action.id(),
                            ArrivalCompletionPayload {
                                operation: operation.id().clone(),
                                barrier_id,
                                generation: action.generation(),
                                copy_completion: true,
                                payload,
                                lane_payload: SharedClockFrontier::default(),
                                tcgen_payload: TcgenThreadFenceFrontier::default(),
                            },
                        )
                        .is_some()
                    {
                        return Err(EngineError::message(format!(
                            "racecheck arrival action {} was reused",
                            action.id()
                        )));
                    }
                }
            }
            OperationEffect::CpAsyncMbarrierArrive { .. } => {}
        }
        drop(state);
        if self.race.global_memory_model_enabled {
            match effect {
                OperationEffect::AsyncPayload(payload) => {
                    self.global_for_warp(operation.id().global_warp_id())?
                        .begin_async_token(
                            payload.token(),
                            operation,
                            payload.issue_accesses(),
                            payload.completion_accesses(),
                            true,
                        )
                        .map_err(EngineError::message)?;
                }
                OperationEffect::AsyncGroupIssue(issue) => {
                    self.global_for_warp(operation.id().global_warp_id())?
                        .begin_async_token(
                            issue.token(),
                            operation,
                            issue.source_accesses(),
                            issue.destination_accesses(),
                            false,
                        )
                        .map_err(EngineError::message)?;
                }
                OperationEffect::AsyncGroupIssueBatch(batch) => {
                    let release = batch.domain() == crate::AsyncGroupDomain::Release;
                    let publications = self.tcgen_publications_for_operation(operation, release)?;
                    let shared = release
                        .then(|| self.shared_frontier_for_operation(operation))
                        .transpose()?;
                    let mut global = self.global_for_warp(operation.id().global_warp_id())?;
                    for member in batch.members() {
                        global
                            .begin_async_token(
                                member.token(),
                                member.operation(),
                                member.source_accesses(),
                                member.destination_accesses(),
                                false,
                            )
                            .map_err(EngineError::message)?;
                        if let Some(shared) = &shared {
                            let lane = member
                                .operation()
                                .active_mask()
                                .first_active()
                                .expect("one issuing lane");
                            global.bind_async_release_frontiers(
                                member.token(),
                                publications.get(&lane).cloned().unwrap_or_default(),
                                shared.get(&lane).cloned().unwrap_or_default(),
                            );
                        }
                    }
                }
                OperationEffect::AsyncGroupWait {
                    outcome: Some(outcome),
                    ..
                } => {
                    let mut global = self.global_for_warp(operation.id().global_warp_id())?;
                    let mut retired = Vec::new();
                    for group in outcome.groups() {
                        for member in group.members() {
                            global
                                .acquire_async_token(operation, member.token(), group.milestone())
                                .map_err(EngineError::message)?;
                            if !outcome.plan().read_only() {
                                retired.push(member.token().clone());
                            }
                        }
                    }
                    for token in retired {
                        global.retire_async_token(&token);
                    }
                }
                OperationEffect::CpAsyncMbarrierArrive {
                    plan,
                    outcome: Some(outcome),
                    actions: Some(actions),
                    ..
                } => {
                    let mut global = self.global_for_warp(operation.id().global_warp_id())?;
                    for ((lane, barrier_id), action) in plan.targets().zip(actions) {
                        if outcome
                            .immediately_ready_physical_actions()
                            .contains(action)
                        {
                            global.publish_cp_async_completion(
                                operation.id().global_warp_id(),
                                lane,
                                barrier_id,
                                action.generation(),
                            );
                        }
                    }
                }
                OperationEffect::ProxyAsyncFence(proxy) => {
                    self.global_for_warp(operation.id().global_warp_id())?
                        .proxy_async_fence(operation, proxy)
                        .map_err(EngineError::message)?;
                }
                OperationEffect::TensorMap(observation) => {
                    self.global_for_warp(operation.id().global_warp_id())?
                        .tensor_map_observation(operation, observation)
                        .map_err(EngineError::message)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn before_completion(&self, effect: CompletionActionEffect<'_>) -> Result<(), EngineError> {
        self.peer_before_completion(effect)?;
        let action = match effect {
            CompletionActionEffect::PhysicalMbarrier(_) => return Ok(()),
            CompletionActionEffect::DeferredPayload(action) => action,
            CompletionActionEffect::AsyncGroup(action) => {
                self.before_async_group_completion(action)?;
                self.begin_global_writes(Self::completion_action_global_write_allocations(effect));
                return Ok(());
            }
            CompletionActionEffect::Setmaxnreg(_) => return Ok(()),
        };
        let mut state = self.race_for_token(action.token())?;
        if self.race.direct_compact {
            Self::commit_pending_direct_segment(&mut state);
        }
        if state.staged_async_completions.contains_key(action.token()) {
            return Err(EngineError::message(format!(
                "racecheck async token {:?} already has a staged completion",
                action.token()
            )));
        }
        if !state.async_token_clocks.contains_key(action.token()) {
            return Err(EngineError::message(format!(
                "racecheck async token {:?} has no issue clock",
                action.token()
            )));
        }
        let retain_accesses = state.retain_accesses;
        let mut records = retain_accesses
            .then(|| Vec::with_capacity(action.completion_accesses().len()))
            .unwrap_or_default();
        let mut tracked_access_count = 0_usize;
        let mut skipped_access_count = 0_usize;
        if retain_accesses {
            records.extend(access_records(action.completion_accesses()));
        }
        for batch in action.completion_accesses() {
            if !self.tracks_race_conflict_batch(batch) {
                skipped_access_count =
                    skipped_access_count.saturating_add(batch.semantic_access_count());
                continue;
            }
            tracked_access_count =
                tracked_access_count.saturating_add(batch.semantic_access_count());
        }
        let compact_batches = match coalesce_physical_access_batches(
            action
                .completion_accesses()
                .iter()
                .filter(|batch| self.tracks_race_conflict_batch(batch)),
        ) {
            Ok(batches) => batches,
            Err(error) => {
                push_unique_incomplete(
                    &mut state.incomplete_reasons,
                    RaceCheckIncompleteReason::ShadowRejected {
                        operation: action.operation().id().clone(),
                        reason: format!("could not compact async footprints: {error}"),
                    },
                );
                return Err(EngineError::message(format!(
                    "racecheck could not compact async completion {:?}: {error}",
                    action.token()
                )));
            }
        };
        state.staged_async_completions.insert(
            action.token().clone(),
            StagedAsyncCompletion {
                compact_batches,
                records: records.into_boxed_slice(),
                tracked_access_count,
                skipped_access_count,
            },
        );
        self.begin_global_writes(Self::completion_action_global_write_allocations(effect));
        Ok(())
    }

    fn before_async_group_completion(
        &self,
        action: &crate::AsyncGroupCompletionAction,
    ) -> Result<(), EngineError> {
        let mut state = self.race_for_async_group_action(action)?;
        if self.race.direct_compact {
            Self::commit_pending_direct_segment(&mut state);
        }
        if state
            .staged_async_group_completions
            .contains_key(&action.id())
        {
            return Err(EngineError::message(format!(
                "racecheck async-group action {} already has a staged completion",
                action.id().get()
            )));
        }
        let shadow_revision = state.shadow.revision();
        let mut shadow_validation = state.shadow.empty_clocked_batch_validation();
        let mut token_clocks = Vec::with_capacity(action.members().len());
        let retain_accesses = state.retain_accesses;
        let mut records = retain_accesses
            .then(|| {
                Vec::with_capacity(
                    action
                        .members()
                        .iter()
                        .map(|member| match action.milestone() {
                            AsyncGroupMilestone::SourceReadComplete => {
                                member.source_accesses().len()
                            }
                            AsyncGroupMilestone::FullComplete => {
                                member.destination_accesses().len()
                            }
                        })
                        .sum(),
                )
            })
            .unwrap_or_default();
        let mut tracked_access_count = 0_usize;
        let mut skipped_access_count = 0_usize;
        for member in action.members() {
            if !state.async_group_token_clocks.contains_key(member.token()) {
                return Err(EngineError::message(format!(
                    "racecheck async-group token {:?} has no issue clock",
                    member.token()
                )));
            }
            let token_clock = state
                .shadow
                .preview_advance_async_actor(member.token())
                .map_err(|error| EngineError::message(error.to_string()))?;
            let accesses = match action.milestone() {
                AsyncGroupMilestone::SourceReadComplete => member.source_accesses(),
                AsyncGroupMilestone::FullComplete => member.destination_accesses(),
            };
            if retain_accesses {
                records.extend(access_records(accesses));
            }
            for batch in accesses {
                if !self.tracks_race_conflict_batch(batch) {
                    skipped_access_count =
                        skipped_access_count.saturating_add(batch.semantic_access_count());
                    continue;
                }
                tracked_access_count =
                    tracked_access_count.saturating_add(batch.semantic_access_count());
            }
            if let Err(error) = state.shadow.extend_batches_at_clock(
                &mut shadow_validation,
                accesses
                    .iter()
                    .filter(|batch| self.tracks_race_conflict_batch(batch)),
                &token_clock,
                member.token(),
            ) {
                match error {
                    RaceShadowError::Race(finding) => {
                        let message = format!(
                            "racecheck rejected async-group completion {:?}: {finding}",
                            member.token()
                        );
                        state.findings.push(finding);
                        return Err(EngineError::message(message));
                    }
                    error => {
                        push_unique_incomplete(
                            &mut state.incomplete_reasons,
                            RaceCheckIncompleteReason::ShadowRejected {
                                operation: member.operation().id().clone(),
                                reason: error.to_string(),
                            },
                        );
                        return Err(EngineError::message(format!(
                            "racecheck could not validate async-group completion {:?}: {error}",
                            member.token()
                        )));
                    }
                }
            }
            token_clocks.push((member.token().clone(), token_clock));
        }
        state.staged_async_group_completions.insert(
            action.id(),
            StagedAsyncGroupCompletion {
                records: records.into_boxed_slice(),
                tracked_access_count,
                skipped_access_count,
                milestone: action.milestone(),
                tokens: action
                    .members()
                    .iter()
                    .map(|member| member.token().clone())
                    .collect(),
                token_clocks: token_clocks.into_boxed_slice(),
                shadow_revision,
                shadow_validation,
            },
        );
        Ok(())
    }

    fn after_completion(&self, effect: CompletionEffect<'_>) -> Result<(), EngineError> {
        let global_write_allocations = Self::completion_global_write_allocations(effect);
        let result = self.after_completion_impl(effect);
        self.finish_global_writes(global_write_allocations);
        self.maybe_collect_global_floor();
        result
    }

    /// Every `GLOBAL_FLOOR_GC_FIRST_EVENTS`, then at geometrically growing
    /// intervals, retire the global frontier entries and barrier payloads that
    /// every actor in the launch already dominates. One collector runs at a
    /// time; it only ever tries to lock shards and byte cells, so it can never
    /// wait on a worker that might be waiting on it.
    fn maybe_collect_global_floor(&self) {
        if !self.race.global_memory_model_enabled {
            return;
        }
        let gc = &self.race.global_floor_gc;
        let events = gc.events.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        let floor_due = events >= gc.next.load(AtomicOrdering::Relaxed);
        if !floor_due && !self.race.global_shared.clock_nodes_sweep_due() {
            return;
        }
        if gc
            .active
            .compare_exchange(
                false,
                true,
                AtomicOrdering::Acquire,
                AtomicOrdering::Relaxed,
            )
            .is_err()
        {
            return;
        }
        if !floor_due {
            // Only the clock-node slabs asked for this pass: hand it to the
            // collector thread (which clears `active` once it swept) and
            // leave the floor schedule alone.
            self.spawn_clock_node_collector();
            return;
        }
        if self.collect_global_floor().is_some() {
            gc.next.store(
                events
                    .saturating_mul(2)
                    .max(events.saturating_add(GLOBAL_FLOOR_GC_FIRST_EVENTS)),
                AtomicOrdering::Relaxed,
            );
        } else {
            // Some shard was busy: try again soon rather than backing off.
            gc.next.store(
                events.saturating_add(GLOBAL_FLOOR_GC_FIRST_EVENTS / 8),
                AtomicOrdering::Relaxed,
            );
        }
        gc.active.store(false, AtomicOrdering::Release);
    }

    /// Returns `(frontier entries retired, barrier payloads retired)`, or
    /// `None` when a shard could not be locked without waiting.
    /// One clock-node collection: a quiescent point to switch the mark
    /// bitmaps and snapshot the root lists, the mark walk with the workers
    /// running, and a second quiescent point to free the unmarked nodes.
    /// Returns whether both quiescent points were reached; a collection whose
    /// sweep could not lock every shard is retried at the next pass (the
    /// allocation barrier keeps the mark valid meanwhile).
    /// Run one clock-node collection on its own thread: a mark walk of the
    /// e384 configs takes seconds, and on a worker it stalled every cluster
    /// waiting on that worker's publications.
    fn spawn_clock_node_collector(&self) {
        let gc = &self.race.global_floor_gc;
        let mut collector = gc
            .collector
            .lock()
            .expect("global racecheck collector handle lock was poisoned");
        if let Some(previous) = collector.take() {
            // Finished: it cleared `active` before it returned.
            let _ = previous.join();
        }
        let shards: Vec<Arc<Mutex<GlobalRaceState>>> = self
            .race
            .race_shards
            .iter()
            .map(|shard| Arc::clone(&shard.global))
            .collect();
        let shared = Arc::clone(&self.race.global_shared);
        let active = Arc::clone(&gc.active);
        let spawned = std::thread::Builder::new()
            .name("numsim-racecheck-gc".to_string())
            .spawn(move || {
                crate::worker_affinity::unconfine_current_thread();
                collect_clock_nodes(&shards, &shared);
                active.store(false, AtomicOrdering::Release);
            });
        match spawned {
            Ok(handle) => *collector = Some(handle),
            Err(_) => {
                // No thread to be had: collect on this worker.
                drop(collector);
                let shards: Vec<Arc<Mutex<GlobalRaceState>>> = self
                    .race
                    .race_shards
                    .iter()
                    .map(|shard| Arc::clone(&shard.global))
                    .collect();
                collect_clock_nodes(&shards, &self.race.global_shared);
                gc.active.store(false, AtomicOrdering::Release);
            }
        }
    }

    fn collect_global_floor(&self) -> Option<(usize, usize)> {
        let mut guards = Vec::with_capacity(self.race.race_shards.len());
        for shard in &self.race.race_shards {
            guards.push(shard.global.try_lock().ok()?);
        }
        let warp_count = self
            .race
            .race_shards
            .iter()
            .map(|shard| shard.global_warp_end)
            .max()
            .unwrap_or(0);
        let mut floor = GlobalFloor::new(warp_count, self.race.global_shared.async_slot_count());
        let mut present = 0;
        for guard in &guards {
            present += guard.meet_actor_clocks(&mut floor);
            if floor.is_bottom() {
                return Some((0, 0));
            }
        }
        if present == 0 {
            return Some((0, 0));
        }
        floor.finish(&self.race.global_shared);
        // A floor with no component above zero dominates nothing, and a floor
        // equal to the one the previous pass walked with dominates nothing
        // new (floors only grow); either way skip the cell walk, which takes
        // every byte cell's write lock and contends with the workers.
        let signature = floor.signature();
        if signature.0 == 0 && signature.2 == 0 {
            return Some((0, 0));
        }
        {
            let mut last = self
                .race
                .global_floor_gc
                .last_signature
                .lock()
                .expect("global floor signature lock was poisoned");
            if *last == Some(signature) {
                return Some((0, 0));
            }
            *last = Some(signature);
        }
        // Publish the watermark while every shard is still locked, so a lane
        // created after this point is marked a laggard before the entries go.
        self.race.global_shared.note_retirement_floor(&floor);
        let mut payloads = 0;
        for guard in &mut guards {
            payloads += guard.retire_dominated_barriers(&floor);
        }
        drop(guards);
        let (retired, _skipped) = self.race.global_shared.retire_dominated(&floor);
        Some((retired, payloads))
    }

    fn after_completion_impl(&self, effect: CompletionEffect<'_>) -> Result<(), EngineError> {
        self.peer_after_completion(effect)?;
        let outcome = match effect {
            CompletionEffect::PhysicalMbarrier(outcome) => {
                if !matches!(
                    outcome.action().kind(),
                    PhysicalCompletionKind::Arrival { .. }
                ) {
                    return Ok(());
                }
                let action = outcome.action();
                let mut state = self.race_for_arrival_completion(action)?;
                if self.race.direct_compact {
                    Self::commit_pending_direct_segment(&mut state);
                }
                let pending = state
                    .arrival_completion_payloads
                    .remove(&action.id())
                    .expect("the selected racecheck shard contains this completion payload");
                if pending.barrier_id != action.barrier_id()
                    || pending.generation != action.generation()
                {
                    return Err(EngineError::message(format!(
                        "racecheck arrival action {} changed identity before completion",
                        action.id()
                    )));
                }
                let payload_key = (pending.barrier_id, pending.generation);
                let payloads = if pending.copy_completion {
                    &mut state.barrier_copy_payloads
                } else {
                    &mut state.barrier_payloads
                };
                match payloads.entry(payload_key) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(pending.payload);
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        entry
                            .get_mut()
                            .merge(&pending.payload)
                            .map_err(|error| EngineError::message(error.to_string()))?;
                    }
                }
                merge_lane_barrier_payload(
                    &mut state.barrier_lane_payloads,
                    payload_key,
                    pending.lane_payload,
                );
                merge_tcgen_barrier_payload(
                    &mut state.barrier_tcgen_payloads,
                    payload_key,
                    &pending.tcgen_payload,
                )?;
                retain_physical_barrier_payloads(
                    &mut state,
                    payload_key.0,
                    payload_key.1,
                    outcome.conditional_completed_generation(),
                );
                drop(state);
                if self.race.global_memory_model_enabled {
                    self.global_for_warp(pending.operation.global_warp_id())?
                        .retain_physical_barrier_generations(
                            payload_key.0,
                            payload_key.1,
                            outcome.conditional_completed_generation(),
                        );
                }
                return Ok(());
            }
            CompletionEffect::DeferredPayload(outcome) => outcome,
            CompletionEffect::AsyncGroup(outcome) => {
                self.after_async_group_completion(outcome)?;
                if self.race.global_memory_model_enabled {
                    let action = outcome.action();
                    let first = action.members().first().ok_or_else(|| {
                        EngineError::message("async group completion has no members")
                    })?;
                    let mut global = self.global_for_token(first.token())?;
                    for member in action.members() {
                        let accesses = match action.milestone() {
                            AsyncGroupMilestone::SourceReadComplete => member.source_accesses(),
                            AsyncGroupMilestone::FullComplete => member.destination_accesses(),
                        };
                        global
                            .complete_async_token(member.token(), action.milestone(), accesses)
                            .map_err(EngineError::message)?;
                        if action.milestone() == AsyncGroupMilestone::FullComplete {
                            if member.domain() == crate::AsyncGroupDomain::CpAsync {
                                global
                                    .retain_cp_async_completion(
                                        action.group_id().global_warp_id(),
                                        action.group_id().lane(),
                                        member.token(),
                                    )
                                    .map_err(EngineError::message)?;
                            }
                            if member.domain() == crate::AsyncGroupDomain::Release {
                                global.retire_async_token(member.token());
                            }
                        }
                    }
                    for physical_action in action.physical_actions() {
                        global.publish_cp_async_completion(
                            action.group_id().global_warp_id(),
                            action.group_id().lane(),
                            physical_action.barrier_id(),
                            physical_action.generation(),
                        );
                    }
                }
                return Ok(());
            }
            CompletionEffect::Setmaxnreg(_) => return Ok(()),
        };
        let action = outcome.action();
        let mut state = self.race_for_token(action.token())?;
        if self.race.direct_compact {
            Self::commit_pending_direct_segment(&mut state);
        }
        let staged = state
            .staged_async_completions
            .remove(action.token())
            .ok_or_else(|| {
                EngineError::message(format!(
                    "racecheck async token {:?} has no staged completion",
                    action.token()
                ))
            })?;
        let mut token_clock = state
            .async_token_clocks
            .get(action.token())
            .cloned()
            .expect("staged async completion retains its token clock");
        let review_findings = match state.shadow.validate_and_commit_batches_at_clock(
            &staged.compact_batches,
            &token_clock,
            action.token(),
            false,
        ) {
            Ok(findings) => findings,
            Err(RaceShadowError::Race(finding)) => {
                let finding = refine_clocked_race_finding(
                    &mut state.shadow,
                    action
                        .completion_accesses()
                        .iter()
                        .filter(|batch| self.tracks_race_conflict_batch(batch)),
                    &token_clock,
                    action.token(),
                    finding,
                );
                let message = format!(
                    "racecheck rejected async completion {:?}: {finding}",
                    action.token()
                );
                state.findings.push(finding);
                return Err(EngineError::message(message));
            }
            Err(error) => {
                push_unique_incomplete(
                    &mut state.incomplete_reasons,
                    RaceCheckIncompleteReason::ShadowRejected {
                        operation: action.operation().id().clone(),
                        reason: error.to_string(),
                    },
                );
                return Err(EngineError::message(format!(
                    "racecheck could not commit async completion {:?}: {error}",
                    action.token()
                )));
            }
        };
        commit_review_findings(&mut state, review_findings);
        commit_staged_accesses(
            &mut state,
            staged.records,
            staged.tracked_access_count,
            staged.skipped_access_count,
            action
                .completion_accesses()
                .iter()
                .filter(|batch| self.tracks_race_conflict_batch(batch)),
        );
        for domain in async_proxy_domains(action.completion_accesses())
            .as_slice()
            .iter()
            .copied()
        {
            token_clock
                .apply_implicit_async_completion(domain)
                .map_err(|error| EngineError::message(error.to_string()))?;
        }
        let completion = token_clock
            .copy_completion_projection(action.token())
            .map_err(|error| EngineError::message(error.to_string()))?;
        for physical_action in action.physical_actions() {
            let payload = BarrierClockPayload::from_clock(completion.clone());
            match state
                .barrier_copy_payloads
                .entry((physical_action.barrier_id(), physical_action.generation()))
            {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(payload);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    entry
                        .get_mut()
                        .merge(&payload)
                        .map_err(|error| EngineError::message(error.to_string()))?;
                }
            }
        }
        for completion in outcome.physical_outcomes() {
            retain_physical_barrier_payloads(
                &mut state,
                completion.action().barrier_id(),
                completion.action().generation(),
                completion.conditional_completed_generation(),
            );
        }
        state
            .shadow
            .retire_async_actor(action.token())
            .map_err(|error| EngineError::message(error.to_string()))?;
        state
            .async_token_clocks
            .remove(action.token())
            .expect("completed async token retains its issue clock until payload publication");
        drop(state);
        if self.race.global_memory_model_enabled {
            let mut global = self.global_for_token(action.token())?;
            global
                .complete_async_token(
                    action.token(),
                    AsyncGroupMilestone::FullComplete,
                    action.completion_accesses(),
                )
                .map_err(EngineError::message)?;
            for physical_action in action.physical_actions() {
                global
                    .publish_async_token_to_physical_barrier(
                        action.token(),
                        physical_action.barrier_id(),
                        physical_action.generation(),
                    )
                    .map_err(EngineError::message)?;
            }
            global.retire_async_token(action.token());
            for completion in outcome.physical_outcomes() {
                global.retain_physical_barrier_generations(
                    completion.action().barrier_id(),
                    completion.action().generation(),
                    completion.conditional_completed_generation(),
                );
            }
        }
        Ok(())
    }

    fn after_async_group_completion(
        &self,
        outcome: &crate::AsyncGroupCompletionOutcome,
    ) -> Result<(), EngineError> {
        let action = outcome.action();
        let mut state = self.race_for_async_group_action(action)?;
        if self.race.direct_compact {
            Self::commit_pending_direct_segment(&mut state);
        }
        let staged = state
            .staged_async_group_completions
            .remove(&action.id())
            .ok_or_else(|| {
                EngineError::message(format!(
                    "racecheck async-group action {} has no staged completion",
                    action.id().get()
                ))
            })?;
        if staged.milestone != action.milestone() {
            return Err(EngineError::message(format!(
                "racecheck async-group action {} changed milestone after validation",
                action.id().get()
            )));
        }
        let token_clocks = if state.shadow.revision() == staged.shadow_revision
            && staged.token_clocks.iter().all(|(token, expected)| {
                state
                    .shadow
                    .preview_advance_async_actor(token)
                    .is_ok_and(|clock| clock == *expected)
            }) {
            for (token, expected) in staged.token_clocks.iter() {
                let committed = state
                    .shadow
                    .advance_async_actor(token)
                    .map_err(|error| EngineError::message(error.to_string()))?;
                debug_assert_eq!(&committed, expected);
            }
            let review_findings = state
                .shadow
                .commit_clocked_batches(staged.shadow_validation);
            commit_review_findings(&mut state, review_findings);
            staged.token_clocks.into_vec()
        } else {
            let mut candidate_shadow = state.shadow.clone();
            let mut token_clocks = Vec::with_capacity(action.members().len());
            let mut review_findings = Vec::new();
            for member in action.members() {
                if !state.async_group_token_clocks.contains_key(member.token()) {
                    return Err(EngineError::message(format!(
                        "racecheck async-group token {:?} disappeared before completion commit",
                        member.token()
                    )));
                }
                let token_clock = candidate_shadow
                    .advance_async_actor(member.token())
                    .map_err(|error| EngineError::message(error.to_string()))?;
                let accesses = match action.milestone() {
                    AsyncGroupMilestone::SourceReadComplete => member.source_accesses(),
                    AsyncGroupMilestone::FullComplete => member.destination_accesses(),
                };
                for batch in accesses {
                    if !self.tracks_race_conflict_batch(batch) {
                        continue;
                    }
                    match candidate_shadow.check_batch_at_clock_for_async_token(
                        batch,
                        &token_clock,
                        member.token(),
                    ) {
                        Ok(reviews) => review_findings.extend(reviews),
                        Err(error) => match error {
                            RaceShadowError::Race(finding) => {
                                let message = format!(
                                    "racecheck rejected async-group completion {:?} at commit: {finding}",
                                    member.token()
                                );
                                state.findings.push(finding);
                                return Err(EngineError::message(message));
                            }
                            error => {
                                push_unique_incomplete(
                                    &mut state.incomplete_reasons,
                                    RaceCheckIncompleteReason::ShadowRejected {
                                        operation: member.operation().id().clone(),
                                        reason: error.to_string(),
                                    },
                                );
                                return Err(EngineError::message(format!(
                                    "racecheck could not commit async-group completion {:?}: {error}",
                                    member.token()
                                )));
                            }
                        },
                    }
                }
                token_clocks.push((member.token().clone(), token_clock));
            }
            state.shadow = candidate_shadow;
            commit_review_findings(&mut state, review_findings);
            token_clocks
        };
        let tracked_completion_batches = action.members().iter().flat_map(|member| {
            match action.milestone() {
                AsyncGroupMilestone::SourceReadComplete => member.source_accesses(),
                AsyncGroupMilestone::FullComplete => member.destination_accesses(),
            }
            .iter()
        });
        commit_staged_accesses(
            &mut state,
            staged.records,
            staged.tracked_access_count,
            staged.skipped_access_count,
            tracked_completion_batches.filter(|batch| self.tracks_race_conflict_batch(batch)),
        );
        for (token, _validated_clock) in token_clocks {
            let member = action
                .members()
                .iter()
                .find(|member| member.token() == &token)
                .expect("completion token came from this action");
            let accesses = match staged.milestone {
                AsyncGroupMilestone::SourceReadComplete => member.source_accesses(),
                AsyncGroupMilestone::FullComplete => member.destination_accesses(),
            };
            let clock = state
                .shadow
                .apply_implicit_async_completion(
                    &token,
                    async_proxy_domains(accesses).as_slice().iter().copied(),
                )
                .map_err(|error| EngineError::message(error.to_string()))?;
            let token_state = state
                .async_group_token_clocks
                .get_mut(&token)
                .ok_or_else(|| {
                    EngineError::message(format!(
                        "racecheck async-group token {token:?} disappeared before completion"
                    ))
                })?;
            token_state.current = clock.clone();
            match staged.milestone {
                AsyncGroupMilestone::SourceReadComplete => token_state.source_read = Some(clock),
                AsyncGroupMilestone::FullComplete => {
                    token_state.full = Some(clock.clone());
                    if member.domain() == crate::AsyncGroupDomain::CpAsync {
                        let contribution = BarrierClockPayload::from_clock(
                            clock
                                .copy_completion_projection(&token)
                                .map_err(|error| EngineError::message(error.to_string()))?,
                        );
                        let key = (action.group_id().global_warp_id(), action.group_id().lane());
                        match state.cp_async_completed_payloads.entry(key) {
                            std::collections::btree_map::Entry::Vacant(entry) => {
                                entry.insert(contribution);
                            }
                            std::collections::btree_map::Entry::Occupied(mut entry) => {
                                entry
                                    .get_mut()
                                    .merge(&contribution)
                                    .map_err(|error| EngineError::message(error.to_string()))?;
                            }
                        }
                    }
                }
            }
            if staged.milestone == AsyncGroupMilestone::FullComplete {
                state
                    .shadow
                    .retire_async_actor(&token)
                    .map_err(|error| EngineError::message(error.to_string()))?;
                if member.domain() == crate::AsyncGroupDomain::Release {
                    state.async_group_token_clocks.remove(&token);
                }
            }
        }
        if !action.physical_actions().is_empty() {
            if action.milestone() != AsyncGroupMilestone::FullComplete {
                return Err(EngineError::message(format!(
                    "async-group action {} released an mbarrier arrival before full completion",
                    action.id().get()
                )));
            }
            let payload = state
                .cp_async_completed_payloads
                .get(&(action.group_id().global_warp_id(), action.group_id().lane()))
                .cloned()
                .ok_or_else(|| {
                    EngineError::message(format!(
                        "async-group action {} has physical arrivals but no completed async tokens",
                        action.id().get()
                    ))
                })?;
            let operation = action
                .members()
                .first()
                .expect("physical async-group action has a member")
                .operation()
                .id()
                .clone();
            for &physical_action in action.physical_actions() {
                if state
                    .arrival_completion_payloads
                    .insert(
                        physical_action.id(),
                        ArrivalCompletionPayload {
                            operation: operation.clone(),
                            barrier_id: physical_action.barrier_id(),
                            generation: physical_action.generation(),
                            copy_completion: true,
                            payload: payload.clone(),
                            lane_payload: SharedClockFrontier::default(),
                            tcgen_payload: TcgenThreadFenceFrontier::default(),
                        },
                    )
                    .is_some()
                {
                    return Err(EngineError::message(format!(
                        "racecheck arrival action {} was reused",
                        physical_action.id()
                    )));
                }
            }
        }
        Ok(())
    }
}

fn merge_cluster_barrier_payload(
    payloads: &mut BTreeMap<(ClusterBarrierId, u64), BarrierClockPayload>,
    barrier_id: ClusterBarrierId,
    generation: u64,
    payload: BarrierClockPayload,
) -> Result<(), EngineError> {
    match payloads.entry((barrier_id, generation)) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(payload);
        }
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            entry
                .get_mut()
                .merge(&payload)
                .map_err(|error| EngineError::message(error.to_string()))?;
        }
    }
    retire_named_barrier_generations(payloads, barrier_id, generation);
    Ok(())
}

/// Newest generations kept per barrier in every `(barrier, generation)`-keyed
/// payload table. A wait resolves to the barrier's last completed generation
/// at the moment it becomes ready, so a payload is only requested while its
/// generation is the newest completed one (plus the scheduling gap before the
/// waiter's after-effect runs); everything older is unreachable and used to
/// accumulate for the whole launch (1.3 M entries, ~35 GB on MegaMoE
/// t128_m128). A request below the window is reported as an incomplete
/// reason rather than served from a missing entry.
pub(crate) const RETAINED_BARRIER_GENERATIONS: u64 = 8;

/// Drops `barrier`'s generations below `newest - RETAINED_BARRIER_GENERATIONS`
/// from a `(barrier, generation)`-keyed table; returns the first generation
/// still retained (0 when nothing can be retired yet).
pub(crate) fn retire_barrier_generations<B: Ord + Copy, V>(
    table: &mut BTreeMap<(B, u64), V>,
    barrier: B,
    newest: u64,
) -> u64 {
    crate::sync_causality::retire_barrier_generations_with(
        table,
        barrier,
        newest,
        RETAINED_BARRIER_GENERATIONS,
    )
}

/// Named and cluster barriers resume the generation they registered for, so a
/// warp woken but not yet scheduled can ask for a generation the barrier has
/// moved past; their tables are small and keep a wider window.
pub(crate) const RETAINED_NAMED_BARRIER_GENERATIONS: u64 = 64;

pub(crate) fn retire_named_barrier_generations<B: Ord + Copy, V>(
    table: &mut BTreeMap<(B, u64), V>,
    barrier: B,
    newest: u64,
) -> u64 {
    crate::sync_causality::retire_barrier_generations_with(
        table,
        barrier,
        newest,
        RETAINED_NAMED_BARRIER_GENERATIONS,
    )
}

fn merge_lane_barrier_payload<B: BarrierGenerationWindow>(
    payloads: &mut BTreeMap<(B, u64), SharedClockFrontier>,
    key: (B, u64),
    payload: SharedClockFrontier,
) {
    match payloads.entry(key) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(payload);
        }
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            entry.get_mut().merge(&payload);
        }
    }
    B::retire_generations(payloads, key.0, key.1);
}

/// Which generation window a `(barrier, generation)`-keyed table uses: the
/// physical-mbarrier window resolves waits to the latest completed generation,
/// named/cluster resumes name their registered generation and keep more.
trait BarrierGenerationWindow: Ord + Copy {
    fn retire_generations<V>(table: &mut BTreeMap<(Self, u64), V>, barrier: Self, newest: u64);
}

impl BarrierGenerationWindow for PhysicalBarrierId {
    fn retire_generations<V>(_table: &mut BTreeMap<(Self, u64), V>, _barrier: Self, _newest: u64) {
        // Physical payloads retire together after the numeric outcome supplies
        // the conditional pin; a partial merge has no authority to retire it.
    }
}

impl BarrierGenerationWindow for crate::NamedBarrierId {
    fn retire_generations<V>(table: &mut BTreeMap<(Self, u64), V>, barrier: Self, newest: u64) {
        retire_named_barrier_generations(table, barrier, newest);
    }
}

impl BarrierGenerationWindow for ClusterBarrierId {
    fn retire_generations<V>(table: &mut BTreeMap<(Self, u64), V>, barrier: Self, newest: u64) {
        retire_named_barrier_generations(table, barrier, newest);
    }
}

fn acquire_cluster_barrier_payload(
    shadow: &mut RaceShadow,
    payload: Option<&BarrierClockPayload>,
    warp_id: usize,
    active_mask: WarpMask,
) -> Result<(), EngineError> {
    let Some(payload) = payload else {
        // A generation containing only relaxed arrivals intentionally carries no HB payload.
        return Ok(());
    };
    shadow
        .barrier_acquire_masked(warp_id, active_mask, payload)
        .map_err(|error| EngineError::message(error.to_string()))
}

fn access_record(batch: &PhysicalAccessBatch) -> RaceCheckAccessRecord {
    RaceCheckAccessRecord {
        operation: batch.operation().id().clone(),
        descriptor: batch.descriptor(),
        logical_buffer: batch.logical_buffer().map(Into::into),
        active_lane_count: batch.lanes().len(),
        lanes: batch.lanes().to_vec().into_boxed_slice(),
    }
}

/// One report record per transfer unit of a multi-unit transfer batch,
/// exactly as when every unit was its own batch.
fn transfer_unit_records(batch: &PhysicalAccessBatch) -> Vec<RaceCheckAccessRecord> {
    let mut records = Vec::with_capacity(batch.semantic_access_count());
    for lane in batch.lanes() {
        for span in lane.footprint().spans() {
            records.push(RaceCheckAccessRecord {
                operation: batch.operation().id().clone(),
                descriptor: batch.descriptor().with_byte_width(span.byte_len()),
                logical_buffer: batch.logical_buffer().map(Into::into),
                active_lane_count: 1,
                lanes: Box::new([lane.with_span(*span)]),
            });
        }
    }
    records
}

/// The report records of a batch list, in the order one batch per transfer
/// unit (and target) would have produced: multi-unit transfer batches expand
/// unit by unit, and multicast siblings interleave their targets per unit.
fn access_records(batches: &[PhysicalAccessBatch]) -> Vec<RaceCheckAccessRecord> {
    let mut records = Vec::with_capacity(batches.len());
    let mut index = 0;
    while index < batches.len() {
        let batch = &batches[index];
        if !batch.transfer_units() {
            records.push(access_record(batch));
            index += 1;
            continue;
        }
        let siblings = batch.transfer_siblings();
        let group = &batches[index..(index + siblings.max(1)).min(batches.len())];
        let interleaved = siblings > 1
            && group.len() == siblings
            && group.iter().all(|sibling| {
                sibling.transfer_siblings() == siblings
                    && sibling.semantic_access_count() == batch.semantic_access_count()
            });
        if !interleaved {
            records.extend(transfer_unit_records(batch));
            index += 1;
            continue;
        }
        let per_target = group.iter().map(transfer_unit_records).collect::<Vec<_>>();
        let unit_count = per_target[0].len();
        let mut per_target = per_target
            .into_iter()
            .map(Vec::into_iter)
            .collect::<Vec<_>>();
        for _ in 0..unit_count {
            for target in per_target.iter_mut() {
                records.push(
                    target
                        .next()
                        .expect("multicast siblings carry the same unit count"),
                );
            }
        }
        index += siblings;
    }
    records
}

fn async_proxy_domains<'a>(
    batches: impl IntoIterator<Item = &'a PhysicalAccessBatch>,
) -> ProxyMemoryDomains {
    let mut domains = ProxyMemoryDomains::new();
    for batch in batches {
        let descriptor = batch.descriptor();
        if descriptor.memory_semantics().proxy() == crate::MemoryProxy::Async {
            domains.insert(descriptor.proxy_memory_domain());
        }
    }
    domains
}

/// Clean-path validation may coalesce one async operation's element batches.
/// If it finds a race, replay only that failing validation with the original
/// fragments so the reported overlap retains source-level element precision.
fn refine_clocked_race_finding<'a>(
    shadow: &mut RaceShadow,
    batches: impl IntoIterator<Item = &'a PhysicalAccessBatch>,
    event_clock: &RaceVectorClock,
    token: &AsyncTokenId,
    fallback: PhysicalRaceFinding,
) -> PhysicalRaceFinding {
    match shadow.validate_batches_at_clock(batches, event_clock, token) {
        Err(RaceShadowError::Race(finding)) => finding,
        _ => fallback,
    }
}

fn prepare_access_batch_commit(
    state: &mut RaceCheckState,
    batch: &PhysicalAccessBatch,
) -> Vec<RaceCheckAccessRecord> {
    state.access_count = state
        .access_count
        .saturating_add(batch.semantic_access_count());
    state.alias_tracker.observe_batch(batch);
    if !state.retain_accesses {
        return Vec::new();
    }
    access_records(std::slice::from_ref(batch))
}

fn commit_access_record(state: &mut RaceCheckState, record: RaceCheckAccessRecord) {
    state.access_count = state.access_count.saturating_add(1);
    state.alias_tracker.observe_record(&record);
    if state.retain_accesses {
        state.accesses.push(record);
    }
}

fn commit_access_records(
    state: &mut RaceCheckState,
    records: impl IntoIterator<Item = RaceCheckAccessRecord>,
) {
    for record in records {
        commit_access_record(state, record);
    }
}

/// Commit async-access accounting without materializing report records when
/// the caller requested the compact summary.  The original batch count stays
/// stable, and alias provenance still observes every exact physical access;
/// only the otherwise-discarded evidence clones are omitted.
fn commit_staged_accesses<'a>(
    state: &mut RaceCheckState,
    records: Box<[RaceCheckAccessRecord]>,
    tracked_access_count: usize,
    skipped_access_count: usize,
    tracked_batches: impl IntoIterator<Item = &'a PhysicalAccessBatch>,
) {
    if state.retain_accesses {
        debug_assert_eq!(
            records.len(),
            tracked_access_count.saturating_add(skipped_access_count)
        );
        commit_access_records(state, records);
    } else {
        debug_assert!(records.is_empty());
        state.access_count = state
            .access_count
            .saturating_add(tracked_access_count)
            .saturating_add(skipped_access_count);
        for batch in tracked_batches {
            state.alias_tracker.observe_batch(batch);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AliasWriter {
    logical_buffer: Option<Arc<str>>,
    operation: Arc<DynamicOpId>,
}

#[derive(Clone, Copy)]
struct AliasWriterRef<'a> {
    logical_buffer: Option<&'a str>,
    operation: &'a DynamicOpId,
}

impl AliasWriterRef<'_> {
    fn into_owned(self) -> AliasWriter {
        AliasWriter {
            logical_buffer: self.logical_buffer.map(Arc::from),
            operation: Arc::new(self.operation.clone()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AliasWriterSegment {
    byte_offset: usize,
    byte_end: usize,
    writer_index: usize,
    writer: Arc<AliasWriter>,
}

#[derive(Clone, Debug)]
struct AliasAdvisoryAccumulator {
    reader_operation: DynamicOpId,
    writer_operation: DynamicOpId,
    spans: Vec<PhysicalByteSpan>,
    occurrences: usize,
}

type AliasAdvisoryKey = (
    PhysicalAccessSpace,
    PhysicalAllocationId,
    Arc<str>,
    Arc<str>,
    usize,
    crate::StaticOpId,
    usize,
    crate::StaticOpId,
);

#[derive(Debug, Default)]
struct AliasTracker {
    last_writers: BTreeMap<(PhysicalAccessSpace, PhysicalAllocationId), AliasAllocationWriters>,
    advisories: BTreeMap<AliasAdvisoryKey, AliasAdvisoryAccumulator>,
}

#[derive(Debug, Default)]
struct AliasAllocationWriters {
    segments: TransactionalIntervalMap<AliasWriterSegment>,
    writer_bounds: Vec<AliasWriterBounds>,
}

#[derive(Debug)]
struct AliasWriterBounds {
    logical_buffer: Arc<str>,
    byte_offset: usize,
    byte_end: usize,
}

impl AliasAllocationWriters {
    fn note_writer(&mut self, logical_buffer: &str, byte_offset: usize, byte_end: usize) -> usize {
        if let Some((index, bounds)) = self
            .writer_bounds
            .iter_mut()
            .enumerate()
            .find(|(_, bounds)| bounds.logical_buffer.as_ref() == logical_buffer)
        {
            bounds.byte_offset = bounds.byte_offset.min(byte_offset);
            bounds.byte_end = bounds.byte_end.max(byte_end);
            index
        } else {
            let index = self.writer_bounds.len();
            self.writer_bounds.push(AliasWriterBounds {
                logical_buffer: Arc::from(logical_buffer),
                byte_offset,
                byte_end,
            });
            index
        }
    }

    #[inline(always)]
    fn range_owned_by(&self, writer_index: usize, byte_offset: usize, byte_end: usize) -> bool {
        let (covering, covered_until) =
            self.segments
                .state_until(byte_offset, byte_end, |segment| segment.byte_end);
        covered_until == byte_end
            && covering.is_some_and(|segment| segment.writer_index == writer_index)
    }

    #[inline(always)]
    fn compact_write_owned_by(
        &self,
        batch: &CompactPhysicalAccessBatch<'_>,
        geometry: CompactDirectGeometry,
        writer_index: usize,
    ) -> bool {
        let (byte_offset, byte_end) = geometry.byte_range();
        if geometry.has_contiguous_coverage() {
            return self.range_owned_by(writer_index, byte_offset, byte_end);
        }
        let duplicate_lane_order = geometry.duplicate_lane_order();
        if duplicate_lane_order.is_empty() {
            return batch.lane_spans().all(|(_, span)| {
                self.range_owned_by(writer_index, span.byte_offset(), span.byte_end())
            });
        }
        let mut previous_span = None;
        duplicate_lane_order.iter().copied().all(|lane| {
            let span = batch
                .lane_span(usize::from(lane))
                .expect("direct geometry retains only active lanes");
            if previous_span == Some(span) {
                return true;
            }
            previous_span = Some(span);
            self.range_owned_by(writer_index, span.byte_offset(), span.byte_end())
        })
    }

    fn overwrite_writer(
        &mut self,
        writer_index: usize,
        byte_offset: usize,
        byte_end: usize,
        writer: AliasWriterRef<'_>,
        retained_writer: &mut Option<Arc<AliasWriter>>,
    ) {
        overwrite_alias_writer(
            &mut self.segments,
            writer_index,
            byte_offset,
            byte_end,
            writer,
            retained_writer,
        );
    }

    fn cannot_alias_read(&self, logical_buffer: &str, byte_offset: usize, byte_end: usize) -> bool {
        self.segments.is_empty()
            || self.writer_bounds.iter().all(|bounds| {
                bounds.logical_buffer.as_ref() == logical_buffer
                    || bounds.byte_end <= byte_offset
                    || byte_end <= bounds.byte_offset
            })
    }
}

fn overwrite_alias_writer_general(
    segments: &mut BTreeMap<usize, AliasWriterSegment>,
    writer_index: usize,
    byte_offset: usize,
    byte_end: usize,
    writer: Arc<AliasWriter>,
) {
    if let Some(segment) = segments
        .get_mut(&byte_offset)
        .filter(|segment| segment.byte_end == byte_end)
    {
        segment.writer_index = writer_index;
        segment.writer = writer;
        return;
    }
    let mut overlapping = Vec::new();
    if let Some((&key, segment)) = segments.range(..=byte_offset).next_back() {
        if segment.byte_end > byte_offset {
            overlapping.push(key);
        }
    }
    overlapping.extend(segments.range(byte_offset..byte_end).map(|(&key, _)| key));
    overlapping.sort_unstable();
    overlapping.dedup();

    let left = overlapping
        .first()
        .and_then(|key| segments.get(key))
        .and_then(|segment| {
            (segment.byte_offset < byte_offset).then(|| AliasWriterSegment {
                byte_offset: segment.byte_offset,
                byte_end: byte_offset,
                writer_index: segment.writer_index,
                writer: segment.writer.clone(),
            })
        });
    let right = overlapping.last().and_then(|key| {
        segments.get(key).and_then(|segment| {
            (segment.byte_end > byte_end).then(|| AliasWriterSegment {
                byte_offset: byte_end,
                byte_end: segment.byte_end,
                writer_index: segment.writer_index,
                writer: segment.writer.clone(),
            })
        })
    });

    for key in overlapping {
        segments.remove(&key);
    }
    if let Some(left) = left {
        segments.insert(left.byte_offset, left);
    }
    segments.insert(
        byte_offset,
        AliasWriterSegment {
            byte_offset,
            byte_end,
            writer_index,
            writer,
        },
    );
    if let Some(right) = right {
        segments.insert(right.byte_offset, right);
    }

    // Only the newly inserted interval and its two boundaries can become
    // mergeable. Merge by key so the operation stays logarithmic and never
    // shifts unrelated intervals.
    let mut current_key = byte_offset;
    if let Some((&previous_key, previous)) = segments.range(..current_key).next_back() {
        let current = segments
            .get(&current_key)
            .expect("the replacement interval was inserted");
        if previous.byte_end == current.byte_offset && previous.writer == current.writer {
            let previous_start = previous.byte_offset;
            let current_end = current.byte_end;
            let current_writer_index = current.writer_index;
            let current_writer = current.writer.clone();
            segments.remove(&previous_key);
            segments.remove(&current_key);
            segments.insert(
                previous_start,
                AliasWriterSegment {
                    byte_offset: previous_start,
                    byte_end: current_end,
                    writer_index: current_writer_index,
                    writer: current_writer,
                },
            );
            current_key = previous_start;
        }
    }
    if let Some((&next_key, next)) = segments
        .range((
            std::ops::Bound::Excluded(current_key),
            std::ops::Bound::Unbounded,
        ))
        .next()
    {
        let current = segments
            .get(&current_key)
            .expect("the replacement interval or its merge was inserted");
        if current.byte_end == next.byte_offset && current.writer == next.writer {
            let current_start = current.byte_offset;
            let next_end = next.byte_end;
            let current_writer_index = current.writer_index;
            let current_writer = current.writer.clone();
            segments.remove(&current_key);
            segments.remove(&next_key);
            segments.insert(
                current_start,
                AliasWriterSegment {
                    byte_offset: current_start,
                    byte_end: next_end,
                    writer_index: current_writer_index,
                    writer: current_writer,
                },
            );
        }
    }
}

fn overwrite_alias_writer(
    segments: &mut TransactionalIntervalMap<AliasWriterSegment>,
    writer_index: usize,
    byte_offset: usize,
    byte_end: usize,
    writer: AliasWriterRef<'_>,
    retained_writer: &mut Option<Arc<AliasWriter>>,
) {
    let (covering, covered_until) =
        segments.state_until(byte_offset, byte_end, |segment| segment.byte_end);
    if covered_until == byte_end
        && covering.is_some_and(|segment| segment.writer_index == writer_index)
    {
        // Alias advisories distinguish logical buffers, not successive writes
        // through the same view. A contained write cannot change any future
        // alias answer, so do not split and immediately reinsert its already
        // equivalent provenance interval.
        return;
    }
    if let Some(segment) = segments
        .get_mut(byte_offset)
        .filter(|segment| segment.byte_end == byte_end)
    {
        // Replacing provenance with another write through the same logical
        // buffer cannot change whether a later read aliases that interval.
        // Retaining the earlier exact writer also keeps valid source evidence
        // while avoiding an Arc replacement for the dominant repeated-write
        // case.
        if segment.writer_index == writer_index {
            return;
        }
        segment.writer_index = writer_index;
        segment.writer =
            Arc::clone(retained_writer.get_or_insert_with(|| Arc::new(writer.into_owned())));
        return;
    }
    segments.prepare_span(byte_offset, byte_end);
    let writer = Arc::clone(retained_writer.get_or_insert_with(|| Arc::new(writer.into_owned())));
    let replacement = vec![AliasWriterSegment {
        byte_offset,
        byte_end,
        writer_index,
        writer,
    }];
    if let Err(replacement) = segments.try_replace_range(
        byte_offset,
        byte_end,
        replacement,
        |segment| segment.byte_offset,
        |segment| segment.byte_end,
        |segment, split_start, split_end| AliasWriterSegment {
            byte_offset: split_start,
            byte_end: split_end,
            writer_index: segment.writer_index,
            writer: Arc::clone(&segment.writer),
        },
    ) {
        let replacement = replacement
            .into_iter()
            .next()
            .expect("one alias-writer replacement is retained");
        overwrite_alias_writer_general(
            segments.general_mut(),
            writer_index,
            byte_offset,
            byte_end,
            replacement.writer,
        );
    }
}

fn merge_alias_spans(mut spans: Vec<PhysicalByteSpan>) -> Box<[PhysicalByteSpan]> {
    spans.sort();
    let mut merged: Vec<PhysicalByteSpan> = Vec::with_capacity(spans.len());
    for span in spans {
        let Some(previous) = merged.last_mut() else {
            merged.push(span);
            continue;
        };
        if previous.allocation() == span.allocation() && previous.byte_end() >= span.byte_offset() {
            let byte_end = previous.byte_end().max(span.byte_end());
            *previous = PhysicalByteSpan::new(
                previous.allocation(),
                previous.byte_offset(),
                byte_end - previous.byte_offset(),
            )
            .expect("merged alias advisory span is non-empty and in-bounds");
        } else {
            merged.push(span);
        }
    }
    merged.into_boxed_slice()
}

fn common_lane_allocation_range(
    lanes: &[LanePhysicalAccess],
) -> Option<(PhysicalAllocationId, usize, usize)> {
    let mut spans = lanes
        .iter()
        .flat_map(|lane| lane.footprint().spans().iter());
    let first = spans.next()?;
    let allocation = first.allocation();
    let mut byte_offset = first.byte_offset();
    let mut byte_end = first.byte_end();
    for span in spans {
        if span.allocation() != allocation {
            return None;
        }
        byte_offset = byte_offset.min(span.byte_offset());
        byte_end = byte_end.max(span.byte_end());
    }
    Some((allocation, byte_offset, byte_end))
}

fn observe_alias_read_span(
    advisories: &mut BTreeMap<AliasAdvisoryKey, AliasAdvisoryAccumulator>,
    segments: &TransactionalIntervalMap<AliasWriterSegment>,
    operation: &DynamicOpId,
    space: PhysicalAccessSpace,
    reader_buffer: &str,
    span: &PhysicalByteSpan,
) {
    let mut cursor = span.byte_offset();
    while cursor < span.byte_end() {
        let (segment, next) =
            segments.state_until(cursor, span.byte_end(), |segment| segment.byte_end);
        let Some(segment) = segment else {
            debug_assert!(next > cursor, "alias-writer interval scan must advance");
            cursor = next;
            continue;
        };
        let Some(writer_buffer) = segment.writer.logical_buffer.as_ref() else {
            cursor = next;
            continue;
        };
        if writer_buffer.as_ref() == reader_buffer {
            cursor = next;
            continue;
        }

        // Indexed writer geometry deliberately stays scalar so repeated lane
        // writes update in place. Preserve the old merged-interval advisory
        // semantics by coalescing adjacent fragments from the same writer at
        // read time.
        let mut overlap_end = next;
        while overlap_end < span.byte_end() {
            let (following, following_end) =
                segments.state_until(overlap_end, span.byte_end(), |segment| segment.byte_end);
            let Some(following) = following else {
                break;
            };
            if following.byte_offset != overlap_end
                || following.writer != segment.writer
                || following.writer.logical_buffer.as_deref() == Some(reader_buffer)
            {
                break;
            }
            overlap_end = following_end;
        }
        let overlap = PhysicalByteSpan::new(span.allocation(), cursor, overlap_end - cursor)
            .expect("overlapping alias spans have a non-empty intersection");
        let key = (
            space,
            span.allocation(),
            Arc::from(reader_buffer),
            Arc::clone(writer_buffer),
            operation.global_warp_id(),
            operation.source_op_id(),
            segment.writer.operation.global_warp_id(),
            segment.writer.operation.source_op_id(),
        );
        let advisory = advisories
            .entry(key)
            .or_insert_with(|| AliasAdvisoryAccumulator {
                reader_operation: operation.clone(),
                writer_operation: segment.writer.operation.as_ref().clone(),
                spans: Vec::new(),
                occurrences: 0,
            });
        advisory.spans.push(overlap);
        advisory.occurrences = advisory.occurrences.saturating_add(1);
        cursor = overlap_end;
    }
}

impl AliasTracker {
    fn observe_compact_batch(&mut self, batch: &CompactPhysicalAccessBatch<'_>) {
        self.observe_compact_batch_with_geometry(batch, None);
    }

    fn observe_compact_batch_with_geometry(
        &mut self,
        batch: &CompactPhysicalAccessBatch<'_>,
        geometry: Option<CompactDirectGeometry>,
    ) {
        let descriptor = batch.descriptor();
        let space = descriptor.space();
        if !matches!(
            space,
            PhysicalAccessSpace::Shared | PhysicalAccessSpace::Tmem
        ) {
            crate::profile_count(ProfileKind::RaceAliasUntrackedSpace);
            return;
        }
        crate::profile_count(if space == PhysicalAccessSpace::Shared {
            ProfileKind::RaceAliasShared
        } else {
            ProfileKind::RaceAliasTmem
        });
        let operation = batch.operation().id();
        let Some(logical_buffer) = batch.logical_buffer() else {
            // Alias advisories require both logical buffer identities. Avoid
            // scanning every lane merely to rediscover that an anonymous
            // access can neither produce nor consume one.
            crate::profile_count(ProfileKind::RaceAliasNoLogicalBuffer);
            return;
        };
        let common_allocation_range = geometry
            .map(|geometry| {
                let (byte_offset, byte_end) = geometry.byte_range();
                (geometry.allocation(), byte_offset, byte_end)
            })
            .or_else(|| {
                let mut spans = batch.lane_spans();
                let (_, first) = spans.next()?;
                let allocation = first.allocation();
                let mut byte_offset = first.byte_offset();
                let mut byte_end = first.byte_end();
                for (_, span) in spans {
                    if span.allocation() != allocation {
                        return None;
                    }
                    byte_offset = byte_offset.min(span.byte_offset());
                    byte_end = byte_end.max(span.byte_end());
                }
                Some((allocation, byte_offset, byte_end))
            });
        if common_allocation_range.is_some() {
            crate::profile_count(ProfileKind::RaceAliasCommonAllocation);
        }

        if descriptor.kind().reads() {
            crate::profile_count(ProfileKind::RaceAliasRead);
            if let Some((allocation, byte_offset, byte_end)) = common_allocation_range {
                if let Some(writers) = self.last_writers.get(&(space, allocation)) {
                    if writers.cannot_alias_read(logical_buffer, byte_offset, byte_end) {
                        crate::profile_count(ProfileKind::RaceAliasUniformRead);
                        if !descriptor.kind().writes() {
                            return;
                        }
                    } else {
                        for (_, span) in batch.lane_spans() {
                            observe_alias_read_span(
                                &mut self.advisories,
                                &writers.segments,
                                operation,
                                space,
                                logical_buffer,
                                &span,
                            );
                        }
                    }
                }
            } else {
                for (_, span) in batch.lane_spans() {
                    let Some(writers) = self.last_writers.get(&(space, span.allocation())) else {
                        continue;
                    };
                    if writers.cannot_alias_read(
                        logical_buffer,
                        span.byte_offset(),
                        span.byte_end(),
                    ) {
                        continue;
                    }
                    observe_alias_read_span(
                        &mut self.advisories,
                        &writers.segments,
                        operation,
                        space,
                        logical_buffer,
                        &span,
                    );
                }
            }
        }

        // An anonymous writer can never produce an alias advisory: read-side
        // comparison deliberately requires both logical-buffer identities.
        // Avoid building byte-interval provenance that no result can observe.
        if descriptor.kind().writes() {
            crate::profile_count(ProfileKind::RaceAliasWrite);
            let writer = AliasWriterRef {
                logical_buffer: Some(logical_buffer),
                operation,
            };
            let mut retained_writer = None;
            if let Some((allocation, byte_offset, byte_end)) = common_allocation_range {
                let writers = self.last_writers.entry((space, allocation)).or_default();
                let writer_index = writers.note_writer(logical_buffer, byte_offset, byte_end);
                if geometry.is_some_and(|geometry| {
                    writers.compact_write_owned_by(batch, geometry, writer_index)
                }) {
                    crate::profile_count(ProfileKind::RaceAliasBatchOwnedHit);
                    return;
                }
                crate::profile_count(ProfileKind::RaceAliasBatchOwnedMiss);
                if geometry.is_some_and(CompactDirectGeometry::has_contiguous_coverage) {
                    writers.overwrite_writer(
                        writer_index,
                        byte_offset,
                        byte_end,
                        writer,
                        &mut retained_writer,
                    );
                    return;
                }
                let duplicate_lane_order = geometry
                    .as_ref()
                    .map(CompactDirectGeometry::duplicate_lane_order)
                    .unwrap_or_default();
                if duplicate_lane_order.is_empty() {
                    for (_, span) in batch.lane_spans() {
                        writers.overwrite_writer(
                            writer_index,
                            span.byte_offset(),
                            span.byte_end(),
                            writer,
                            &mut retained_writer,
                        );
                    }
                } else {
                    let mut previous_span = None;
                    for &lane in duplicate_lane_order {
                        let span = batch
                            .lane_span(usize::from(lane))
                            .expect("direct geometry retains only active lanes");
                        if previous_span == Some(span) {
                            continue;
                        }
                        writers.overwrite_writer(
                            writer_index,
                            span.byte_offset(),
                            span.byte_end(),
                            writer,
                            &mut retained_writer,
                        );
                        previous_span = Some(span);
                    }
                }
            } else {
                for (_, span) in batch.lane_spans() {
                    let writers = self
                        .last_writers
                        .entry((space, span.allocation()))
                        .or_default();
                    let writer_index =
                        writers.note_writer(logical_buffer, span.byte_offset(), span.byte_end());
                    writers.overwrite_writer(
                        writer_index,
                        span.byte_offset(),
                        span.byte_end(),
                        writer,
                        &mut retained_writer,
                    );
                }
            }
        }
    }

    fn observe_batch(&mut self, batch: &PhysicalAccessBatch) {
        self.observe(
            batch.operation().shared_id(),
            batch.descriptor(),
            batch.shared_logical_buffer(),
            batch.lanes(),
        );
    }

    fn observe_record(&mut self, access: &RaceCheckAccessRecord) {
        self.observe(
            Arc::new(access.operation().clone()),
            access.descriptor(),
            access.logical_buffer().map(Arc::from),
            access.lanes(),
        );
    }

    fn observe(
        &mut self,
        operation: Arc<DynamicOpId>,
        descriptor: PhysicalAccessDescriptor,
        logical_buffer: Option<Arc<str>>,
        lanes: &[LanePhysicalAccess],
    ) {
        let space = descriptor.space();
        if !matches!(
            space,
            PhysicalAccessSpace::Shared | PhysicalAccessSpace::Tmem
        ) {
            return;
        }
        let Some(logical_buffer) = logical_buffer else {
            return;
        };
        let common_allocation_range = common_lane_allocation_range(lanes);

        if descriptor.kind().reads() {
            if let Some((allocation, byte_offset, byte_end)) = common_allocation_range {
                if let Some(writers) = self.last_writers.get(&(space, allocation)) {
                    if writers.cannot_alias_read(logical_buffer.as_ref(), byte_offset, byte_end) {
                        crate::profile_count(ProfileKind::RaceAliasUniformRead);
                        if !descriptor.kind().writes() {
                            return;
                        }
                    } else {
                        for lane in lanes {
                            for span in lane.footprint().spans() {
                                observe_alias_read_span(
                                    &mut self.advisories,
                                    &writers.segments,
                                    operation.as_ref(),
                                    space,
                                    logical_buffer.as_ref(),
                                    span,
                                );
                            }
                        }
                    }
                }
            } else {
                for lane in lanes {
                    for span in lane.footprint().spans() {
                        let Some(writers) = self.last_writers.get(&(space, span.allocation()))
                        else {
                            continue;
                        };
                        if writers.cannot_alias_read(
                            logical_buffer.as_ref(),
                            span.byte_offset(),
                            span.byte_end(),
                        ) {
                            continue;
                        }
                        observe_alias_read_span(
                            &mut self.advisories,
                            &writers.segments,
                            operation.as_ref(),
                            space,
                            logical_buffer.as_ref(),
                            span,
                        );
                    }
                }
            }
        }

        // See the compact path above: unnamed writers are not observable by
        // this advisory and need no retained interval state.
        if descriptor.kind().writes() {
            let writer = AliasWriterRef {
                logical_buffer: Some(logical_buffer.as_ref()),
                operation: operation.as_ref(),
            };
            let mut retained_writer = None;
            if let Some((allocation, byte_offset, byte_end)) = common_allocation_range {
                let writers = self.last_writers.entry((space, allocation)).or_default();
                let writer_index =
                    writers.note_writer(logical_buffer.as_ref(), byte_offset, byte_end);
                for lane in lanes {
                    for span in lane.footprint().spans() {
                        writers.overwrite_writer(
                            writer_index,
                            span.byte_offset(),
                            span.byte_end(),
                            writer,
                            &mut retained_writer,
                        );
                    }
                }
            } else {
                for lane in lanes {
                    for span in lane.footprint().spans() {
                        let writers = self
                            .last_writers
                            .entry((space, span.allocation()))
                            .or_default();
                        let writer_index = writers.note_writer(
                            logical_buffer.as_ref(),
                            span.byte_offset(),
                            span.byte_end(),
                        );
                        writers.overwrite_writer(
                            writer_index,
                            span.byte_offset(),
                            span.byte_end(),
                            writer,
                            &mut retained_writer,
                        );
                    }
                }
            }
        }
    }

    fn advisories(&self) -> Vec<AliasStaleReadAdvisory> {
        self.advisories
            .iter()
            .map(
                |((space, allocation, reader_buffer, writer_buffer, ..), advisory)| {
                    AliasStaleReadAdvisory {
                        reader_buffer: reader_buffer.as_ref().into(),
                        writer_buffer: writer_buffer.as_ref().into(),
                        space: *space,
                        allocation: *allocation,
                        overlaps: merge_alias_spans(advisory.spans.clone()),
                        reader_operation: advisory.reader_operation.clone(),
                        writer_operation: advisory.writer_operation.clone(),
                        occurrences: advisory.occurrences,
                    }
                },
            )
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RaceCheckMode;

impl EngineModeImpl for RaceCheckMode {
    type LaunchState = RaceCheckLaunchState;
    type GlobalMemoryTransactionGuard<'a> = GlobalMemoryTransactionGuard<'a>;

    const NAME: &'static str = "racecheck";
    const OBSERVES_OPERATIONS: bool = true;
    const OBSERVES_PROXY_MEMORY_DOMAINS: bool = true;
    const USES_GLOBAL_MEMORY_TRANSACTION: bool = true;
    const USES_CACHED_GLOBAL_READ_FAST_PATH: bool = true;

    fn begin_global_memory_transaction<'a>(
        state: &'a Self::LaunchState,
        exclusive: bool,
        spans: &[PhysicalByteSpan],
    ) -> Result<Self::GlobalMemoryTransactionGuard<'a>, EngineError> {
        if !state.race.global_memory_model_enabled || spans.is_empty() {
            return Ok(GlobalMemoryTransactionGuard::default());
        }
        let stripes = global_transaction_stripes(spans);
        if exclusive {
            let _profile = ProfileTimer::new(ProfileKind::RaceGlobalTransactionExclusiveWait);
            Ok(GlobalMemoryTransactionGuard {
                _shared: Vec::new(),
                _exclusive: stripes
                    .into_iter()
                    .map(|stripe| {
                        state.race.global_transaction[stripe]
                            .write()
                            .expect("racecheck global-memory transaction lock was poisoned")
                    })
                    .collect(),
            })
        } else {
            let _profile = ProfileTimer::new(ProfileKind::RaceGlobalTransactionSharedWait);
            Ok(GlobalMemoryTransactionGuard {
                _shared: stripes
                    .into_iter()
                    .map(|stripe| {
                        state.race.global_transaction[stripe]
                            .read()
                            .expect("racecheck global-memory transaction lock was poisoned")
                    })
                    .collect(),
                _exclusive: Vec::new(),
            })
        }
    }

    fn observes_analysis_gap(_state: &Self::LaunchState, kind: AnalysisGapKind) -> bool {
        !matches!(kind, AnalysisGapKind::AtomicLaneSerialization)
    }

    fn controls_physical_access(
        state: &Self::LaunchState,
        kind: OperationKind,
        space: PhysicalAccessSpace,
    ) -> bool {
        // A requested access journal covers register/local traffic too.  The
        // global-allocation optimization may put Synccheck in compact-memory
        // mode, but that must not narrow Racecheck's diagnostic evidence.
        if state.race.retain_accesses {
            return true;
        }
        if state.context.uses_compact_memory_analysis() {
            return !matches!(
                space,
                PhysicalAccessSpace::Local | PhysicalAccessSpace::Register
            );
        }
        state.peer_controls_physical_access(kind, space)
    }

    fn elides_physical_access_resolution_for_buffer(
        state: &Self::LaunchState,
        kind: OperationKind,
        buffer: &crate::runtime::RuntimeBuffer,
    ) -> bool {
        state.race.direct_compact
            && matches!(kind, OperationKind::Load | OperationKind::Store)
            && matches!(
                buffer.uniform_physical_space(),
                Some(PhysicalAccessSpace::Local | PhysicalAccessSpace::Register)
            )
    }

    fn controls_physical_access_allocation(
        state: &Self::LaunchState,
        kind: OperationKind,
        space: PhysicalAccessSpace,
        allocation: Option<PhysicalAllocationId>,
    ) -> bool {
        // Access inspection is an exact journal, independent of which global
        // allocations the launch-wide HB shadow can safely omit.  Keep
        // read-only inputs observable here; GlobalRaceState applies the
        // allocation filter only to its sparse byte/version state.
        if state.race.retain_accesses {
            return Self::controls_physical_access(state, kind, space);
        }
        if state.context.uses_compact_memory_analysis() {
            if !Self::controls_physical_access(state, kind, space) {
                return false;
            }
            if kind == OperationKind::Load && space == PhysicalAccessSpace::Global {
                return allocation
                    .is_none_or(|allocation| state.context.tracks_global_allocation(allocation));
            }
            return true;
        }
        state.peer_controls_physical_access_allocation(kind, space, allocation)
    }

    fn observes_physical_access_batch(
        state: &Self::LaunchState,
        descriptor: PhysicalAccessDescriptor,
        _mask: WarpMask,
    ) -> bool {
        state.race.retain_accesses
            || tracks_race_conflicts(descriptor.space())
            || (state.race.global_memory_model_enabled
                && descriptor.space() == PhysicalAccessSpace::Global)
    }

    fn resolves_async_accesses(_state: &Self::LaunchState) -> bool {
        true
    }

    fn compacts_async_accesses(state: &Self::LaunchState) -> bool {
        state.race.direct_compact
    }

    fn compacts_direct_physical_access(
        state: &Self::LaunchState,
        descriptor: PhysicalAccessDescriptor,
        _mask: WarpMask,
        atomic_return_sync_relevant: bool,
    ) -> bool {
        // A strong global write publishes: it may be the write some wait names
        // as the one that released it, and naming it needs the value the word
        // holds afterwards. That post-image is read back only on the full
        // batch path, so the summary-only path cannot carry a publication.
        // Publishers are spelled in raw PTX now, so this is the shape every
        // declared protocol takes, not a rare one.
        // Generic proxy only: a declared word is polled by `wait_until`,
        // which is an ordinary load, so its publisher is an ordinary store.
        // An async-proxy release (`st_async`, a bulk store) publishes through
        // the async lifecycle instead, and pulling it off the summary path
        // costs its release payload without buying any word history.
        let publishes = descriptor.space() == PhysicalAccessSpace::Global
            && descriptor.kind().writes()
            && descriptor.memory_semantics().proxy() == crate::MemoryProxy::Generic
            && descriptor.memory_semantics().class().is_atomic_class();
        state.race.direct_compact
            && !state.race.retain_accesses
            && !atomic_return_sync_relevant
            // The compact shared shadow retains proxy, but not strong scope.
            // Use the existing semantic witness path for scoped operations.
            && !(descriptor.space() == PhysicalAccessSpace::Shared
                && descriptor.memory_semantics().order().is_strong())
            && !publishes
            && (tracks_race_conflicts(descriptor.space())
                || (state.race.global_memory_model_enabled
                    && descriptor.space() == PhysicalAccessSpace::Global))
    }

    fn applies_compact_physical_access_after_numeric(
        state: &Self::LaunchState,
        descriptor: PhysicalAccessDescriptor,
    ) -> bool {
        state.race.direct_compact
            && !(state.race.global_memory_model_enabled
                && descriptor.space() == PhysicalAccessSpace::Global)
    }

    fn before_compact_physical_access(
        state: &Self::LaunchState,
        batch: &CompactPhysicalAccessBatch<'_>,
    ) -> Result<(), EngineError> {
        debug_assert!(state.race.direct_compact);
        debug_assert!(!state.race.retain_accesses);
        debug_assert!(
            tracks_race_conflicts(batch.descriptor().space())
                || (state.race.global_memory_model_enabled
                    && batch.descriptor().space() == PhysicalAccessSpace::Global)
        );
        debug_assert!(!batch.atomic_return_sync_relevant());
        state.before_compact_physical_access(batch)
    }

    fn after_compact_physical_access(
        state: &Self::LaunchState,
        batch: &CompactPhysicalAccessBatch<'_>,
    ) -> Result<(), EngineError> {
        debug_assert!(state.race.direct_compact);
        debug_assert!(!state.race.retain_accesses);
        debug_assert!(
            tracks_race_conflicts(batch.descriptor().space())
                || (state.race.global_memory_model_enabled
                    && batch.descriptor().space() == PhysicalAccessSpace::Global)
        );
        debug_assert!(!batch.atomic_return_sync_relevant());
        if state.race.global_memory_model_enabled
            && batch.descriptor().space() == PhysicalAccessSpace::Global
        {
            state.after_compact_physical_access(batch)
        } else {
            state.apply_compact_physical_access_after_numeric(batch)
        }
    }

    fn begin_cached_global_read(
        state: &Self::LaunchState,
        access: CachedGlobalReadAccess,
    ) -> Result<bool, EngineError> {
        if !state.race.global_memory_model_enabled {
            return Ok(false);
        }
        state.begin_cached_global_read(access)
    }

    fn finish_cached_global_read(
        state: &Self::LaunchState,
        access: CachedGlobalReadAccess,
    ) -> Result<CachedGlobalReadFinish, EngineError> {
        state.finish_cached_global_read(access)
    }

    fn declared_word_candidates(
        state: &Self::LaunchState,
        span: PhysicalByteSpan,
        warp_id: usize,
        lane: usize,
    ) -> (usize, Vec<u64>) {
        if !state.race.global_memory_model_enabled {
            return (0, Vec::new());
        }
        state
            .declared_word_candidates(span, warp_id, lane)
            .unwrap_or((0, Vec::new()))
    }

    fn global_memory_progress_snapshot(
        state: &Self::LaunchState,
        spans: impl IntoIterator<Item = PhysicalByteSpan>,
    ) -> Option<GlobalMemoryProgressSnapshot> {
        if !state.race.global_memory_model_enabled {
            return None;
        }
        let mut progresses = Vec::new();
        for span in spans {
            let allocation = state
                .race
                .global_allocation_epochs
                .get(&span.allocation())?;
            progresses.push(
                allocation
                    .tracked_poll_range(span.byte_offset(), span.byte_end())
                    .progress
                    .clone(),
            );
        }
        let snapshot = GlobalMemoryProgressSnapshot::capture(progresses);
        (!snapshot.is_empty()).then_some(snapshot)
    }

    fn observes_tcgen_accesses(_state: &Self::LaunchState) -> bool {
        true
    }

    fn after_unobserved_physical_access(
        state: &Self::LaunchState,
        global_warp_id: usize,
    ) -> Result<(), EngineError> {
        let shard = state.shard_for_warp(global_warp_id)?;
        let _ = shard.uncontrolled_access_count.fetch_update(
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
            |count| count.checked_add(1),
        );
        Ok(())
    }

    fn before_operation(
        state: &Self::LaunchState,
        operation: &OperationContext,
    ) -> Result<(), EngineError> {
        if operation.kind() != OperationKind::Load {
            state.invalidate_compact_global_read_cache(operation.id().global_warp_id())?;
        }
        Ok(())
    }

    fn after_operation(
        state: &Self::LaunchState,
        operation: &OperationContext,
    ) -> Result<(), EngineError> {
        state.peer_after_operation(operation)
    }

    fn before_effect(
        state: &Self::LaunchState,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        state.before_effect(operation, effect)
    }

    fn after_effect(
        state: &Self::LaunchState,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        // Intercepted before the shadow dispatch so the global-transaction and
        // per-warp lock order stays exactly what the removed `after_warp_sync`
        // hook took.
        if let OperationEffect::WarpSync(sync) = effect {
            return Self::apply_warp_sync(state, operation, sync.mask());
        }
        state.after_effect(operation, effect)
    }

    fn before_completion(
        state: &Self::LaunchState,
        effect: CompletionActionEffect<'_>,
    ) -> Result<(), EngineError> {
        state.before_completion(effect)
    }

    fn after_completion(
        state: &Self::LaunchState,
        effect: CompletionEffect<'_>,
    ) -> Result<(), EngineError> {
        state.after_completion(effect)
    }

    fn warp_collective_rendezvous(
        state: &Self::LaunchState,
        operation: &OperationContext,
        mask: WarpMask,
    ) -> Result<(), EngineError> {
        RaceCheckMode::apply_warp_collective_rendezvous(state, operation, mask)
    }
}

impl RaceCheckMode {
    /// Apply one completed same-warp lane rendezvous to the race shadows.
    fn apply_warp_sync(
        state: &<Self as EngineModeImpl>::LaunchState,
        operation: &OperationContext,
        mask: WarpMask,
    ) -> Result<(), EngineError> {
        if state.race.global_memory_model_enabled {
            // A warp sync only joins the warp's own lane clocks, which no
            // other shard touches; it needs no allocation transaction.
            state
                .global_for_warp(operation.id().global_warp_id())?
                .warp_sync(operation.id().global_warp_id(), mask)
                .map_err(EngineError::message)?;
        }
        let mut race = state.race_for_operation(operation)?;
        if state.race.direct_compact {
            RaceCheckLaunchState::commit_pending_direct_segment(&mut race);
        }
        let tcgen_payload =
            tcgen_release_mask(&race, operation, operation.id().global_warp_id(), mask)?;
        race.shadow
            .warp_sync(operation.id().global_warp_id(), mask)
            .map_err(|error| EngineError::message(error.to_string()))?;
        race.lane_shadow
            .warp_sync(operation.id().global_warp_id(), mask)
            .map_err(EngineError::message)?;
        tcgen_acquire_mask(
            &mut race,
            operation,
            operation.id().global_warp_id(),
            mask,
            &tcgen_payload,
        )?;
        Ok(())
    }

    fn apply_warp_collective_rendezvous(
        state: &<Self as EngineModeImpl>::LaunchState,
        operation: &OperationContext,
        mask: WarpMask,
    ) -> Result<(), EngineError> {
        let mut race = state.race_for_operation(operation)?;
        if state.race.direct_compact {
            RaceCheckLaunchState::commit_pending_direct_segment(&mut race);
        }
        race.shadow
            .warp_sync(operation.id().global_warp_id(), mask)
            .map_err(|error| EngineError::message(error.to_string()))?;
        race.lane_shadow
            .warp_sync(operation.id().global_warp_id(), mask)
            .map_err(EngineError::message)?;
        Ok(())
    }
}

fn commit_review_findings(
    state: &mut RaceCheckState,
    review_findings: impl IntoIterator<Item = PhysicalRaceFinding>,
) {
    let review_findings = review_findings.into_iter().collect::<Vec<_>>();
    let reviewed_load_operations = review_findings
        .iter()
        .filter_map(PhysicalRaceFinding::reviewed_tmem_load_operation)
        .cloned()
        .collect::<BTreeSet<_>>();
    if !reviewed_load_operations.is_empty() {
        let tokens = state
            .tcgen_work_tokens
            .iter()
            .filter(|(_, work)| {
                work.kind == TcgenWorkKind::Load
                    && reviewed_load_operations.contains(&work.operation)
            })
            .map(|(token, _)| token.clone())
            .collect::<Vec<_>>();
        if let Err(error) = state.shadow.retire_async_actors_for_review(&tokens) {
            let operation = tokens
                .first()
                .and_then(|token| state.tcgen_work_tokens.get(token))
                .map(|work| work.operation.clone())
                .or_else(|| {
                    review_findings
                        .first()
                        .map(|finding| finding.current().operation().clone())
                });
            if let Some(operation) = operation {
                push_unique_incomplete(
                    &mut state.incomplete_reasons,
                    RaceCheckIncompleteReason::ShadowRejected {
                        operation,
                        reason: format!(
                            "could not retire reviewed tcgen05.ld async actors: {error}"
                        ),
                    },
                );
            }
        } else {
            for token in tokens {
                let work = state
                    .tcgen_work_tokens
                    .remove(&token)
                    .expect("reviewed TCGEN load was selected from the active token map");
                debug_assert_eq!(work.kind, TcgenWorkKind::Load);
                let replaced = state
                    .reviewed_tcgen_load_tokens
                    .insert(token, work.operation);
                debug_assert!(replaced.is_none());
            }
        }
    }
    record_review_findings(&mut state.findings, review_findings);
}

fn record_review_findings(
    findings: &mut Vec<PhysicalRaceFinding>,
    review_findings: impl IntoIterator<Item = PhysicalRaceFinding>,
) {
    for finding in review_findings {
        debug_assert!(finding.requires_unwaited_tmem_load_review());
        if findings
            .iter()
            .any(|existing| same_review_site(existing, &finding))
        {
            continue;
        }
        findings.push(finding);
    }
}

fn append_report_findings(
    findings: &mut Vec<PhysicalRaceFinding>,
    new_findings: impl IntoIterator<Item = PhysicalRaceFinding>,
) {
    for finding in new_findings {
        if finding.requires_unwaited_tmem_load_review() {
            record_review_findings(findings, [finding]);
        } else {
            findings.push(finding);
        }
    }
}

fn append_report_advisories(
    advisories: &mut Vec<AliasStaleReadAdvisory>,
    new_advisories: impl IntoIterator<Item = AliasStaleReadAdvisory>,
) {
    for mut advisory in new_advisories {
        let Some(existing) = advisories
            .iter_mut()
            .find(|existing| same_static_advisory_site(existing, &advisory))
        else {
            advisories.push(advisory);
            continue;
        };
        let occurrences = existing.occurrences.saturating_add(advisory.occurrences);
        let candidate_key = (
            &advisory.reader_operation,
            &advisory.writer_operation,
            advisory.allocation,
        );
        let existing_key = (
            &existing.reader_operation,
            &existing.writer_operation,
            existing.allocation,
        );
        if candidate_key < existing_key {
            advisory.occurrences = occurrences;
            *existing = advisory;
        } else {
            existing.occurrences = occurrences;
        }
    }
}

fn same_static_advisory_site(
    left: &AliasStaleReadAdvisory,
    right: &AliasStaleReadAdvisory,
) -> bool {
    left.reader_buffer == right.reader_buffer
        && left.writer_buffer == right.writer_buffer
        && left.space == right.space
        && left.reader_operation.kernel_index() == right.reader_operation.kernel_index()
        && left.reader_operation.source_op_id() == right.reader_operation.source_op_id()
        && left.writer_operation.kernel_index() == right.writer_operation.kernel_index()
        && left.writer_operation.source_op_id() == right.writer_operation.source_op_id()
}

// One representative is sufficient for a review whose resolution depends on
// the same pair of static instructions. Keeping per-warp/lane/iteration copies
// would make a completed launch emit an unbounded number of identical actions.
fn same_review_site(left: &PhysicalRaceFinding, right: &PhysicalRaceFinding) -> bool {
    left.kind() == right.kind()
        && same_static_witness_site(left.prior(), right.prior())
        && same_static_witness_site(left.current(), right.current())
}

fn same_static_witness_site(left: &PhysicalRaceWitness, right: &PhysicalRaceWitness) -> bool {
    left.operation().kernel_index() == right.operation().kernel_index()
        && left.operation().source_op_id() == right.operation().source_op_id()
        && left.kind() == right.kind()
        && left.space() == right.space()
}

fn push_unique_incomplete(
    incomplete: &mut Vec<RaceCheckIncompleteReason>,
    reason: RaceCheckIncompleteReason,
) {
    let already_present = match &reason {
        RaceCheckIncompleteReason::AnalysisGap {
            operation,
            kind,
        } => incomplete.iter().any(|existing| {
            matches!(
                existing,
                RaceCheckIncompleteReason::AnalysisGap {
                    operation: existing_operation,
                    kind: existing_kind,
                } if existing_operation.kernel_index() == operation.kernel_index()
                    && existing_operation.source_op_id() == operation.source_op_id()
                    && existing_kind == kind
            )
        }),
        _ => incomplete.contains(&reason),
    };
    if !already_present {
        incomplete.push(reason);
    }
}

fn race_rejection_message(operation: &DynamicOpId, finding: &PhysicalRaceFinding) -> String {
    let prior = finding.prior();
    let current = finding.current();
    if prior.operation().global_warp_id() == current.operation().global_warp_id()
        && prior.lane() != current.lane()
    {
        format!("racecheck rejected same-warp lane access {operation}: {finding}")
    } else {
        format!("racecheck rejected {operation}: {finding}")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn barrier_generation_window_keeps_the_newest_generations_only() {
        use super::{retire_barrier_generations, RETAINED_BARRIER_GENERATIONS};
        let mut table: std::collections::BTreeMap<(u32, u64), u64> =
            std::collections::BTreeMap::new();
        for generation in 0..3 {
            table.insert((7, generation), generation);
            assert_eq!(retire_barrier_generations(&mut table, 7, generation), 0);
        }
        assert_eq!(table.len(), 3);
        // Another barrier's generations are untouched.
        table.insert((9, 100), 100);
        let newest = RETAINED_BARRIER_GENERATIONS + 2;
        table.insert((7, newest), newest);
        assert_eq!(retire_barrier_generations(&mut table, 7, newest), 2);
        assert!(!table.contains_key(&(7, 0)) && !table.contains_key(&(7, 1)));
        assert!(table.contains_key(&(7, 2)) && table.contains_key(&(7, newest)));
        assert!(table.contains_key(&(9, 100)));
        // An out-of-order older publication never widens the retired range.
        table.insert((7, 3), 3);
        assert_eq!(retire_barrier_generations(&mut table, 7, 3), 0);
        assert!(table.contains_key(&(7, 2)));
    }

    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use crate::physical_access::CompactPhysicalAccessBatch;
    use crate::race_shadow::{tracks_race_conflicts, PhysicalRaceOrderingFailure};
    use crate::runtime::{
        plan_physical_mbarrier_arrive, plan_physical_mbarrier_init, plan_physical_mbarrier_wait,
        plan_tcgen_commit_issue, run_kernel_engine_launch, run_kernel_engine_launch_report,
        ClusterBarrierArrivalSemantics, ExecutionPolicy, LaunchSelection,
        PhysicalMbarrierCompletionIssuePlan, PhysicalMbarrierWaitOutcome, PhysicalPtr,
        RuntimeBuffer, TcgenAccumulatorDtype, TcgenMmaPipelineClass, TcgenPipelineOperation,
        TcgenWorkIssue, TcgenWorkKind, TcgenWorkSet,
    };
    use crate::{
        AnalysisGapKind, AsyncGroupDomain, AsyncGroupHub, AsyncGroupIssueEffect,
        AsyncPayloadEffect, CompletionActionEffect, CompletionEffect, CtaId, DeferredPayloadHub,
        DynamicOpId, EngineModeImpl, ExecutionReport, LaunchTopology, MbarrierCompletionAction,
        MbarrierCompletionOutcome, OperationContext, OperationEffect, OperationKind,
        PhysicalAccessBatch, PhysicalAccessDescriptor, PhysicalAccessKind, PhysicalAccessSpace,
        PhysicalAllocationId, PhysicalBarrierHub, PhysicalByteSpan, PhysicalMemory,
        PhysicalRaceFinding, PhysicalRaceKind, PhysicalRaceWitness, ProxyAsyncFenceEffect,
        ProxyAsyncFenceScope, ResolvedTransitionSummary, StaticOpId, SyncCheckEffectKind,
        TcgenFenceKind, WarpMask, WarpValue, WARP_SIZE,
    };

    use super::{
        append_report_advisories, overwrite_alias_writer_general, same_warp_conflict_kinds,
        AliasAllocationWriters, AliasTracker, AliasWriter, AliasWriterRef, AliasWriterSegment,
        CompactGlobalReadFinish, LaneFrontier, RaceCheckIncompleteReason, RaceCheckLaunchState,
        RaceCheckMode, RaceCheckStatus, SameWarpLaneBatchValidation, SameWarpLaneShadow,
        SharedClockFrontier, SyncCheckStatus, WarpLaneOrder,
    };

    fn operation(warp_id: usize, sequence: u64, kind: OperationKind) -> OperationContext {
        masked_operation(warp_id, sequence, kind, WarpMask::from_lanes([0]).unwrap())
    }

    #[test]
    fn joined_tcgen_frontier_keeps_pipeline_and_completion_facts_separate() {
        use super::{TcgenPipelineDescriptor, TcgenThreadFenceFrontier, TcgenThreadKey};
        use crate::race_shadow::RaceShadow;
        use crate::AsyncTokenId;

        let mut shadow = RaceShadow::new(2);
        let first = AsyncTokenId::new(operation(0, 1, OperationKind::AsyncIssue).id().clone(), 0);
        let second = AsyncTokenId::new(operation(1, 1, OperationKind::AsyncIssue).id().clone(), 0);
        let first_clock = shadow
            .fork_tcgen_token_after_clock(0, WarpMask::FULL, &first, None, None)
            .unwrap();
        let second_clock = shadow
            .fork_tcgen_token_after_clock(1, WarpMask::FULL, &second, None, None)
            .unwrap();
        let first_source = TcgenThreadKey {
            kernel_index: 0,
            global_warp_id: 0,
            lane: 0,
        };
        let second_source = TcgenThreadKey {
            kernel_index: 0,
            global_warp_id: 1,
            lane: 0,
        };
        let copy = TcgenPipelineDescriptor::new(TcgenPipelineOperation::Copy, 1, None);
        let shift = TcgenPipelineDescriptor::new(TcgenPipelineOperation::Shift, 1, None);
        let mut first_frontier = TcgenThreadFenceFrontier::default();
        first_frontier
            .merge_pipeline(copy.clone(), &first_clock)
            .unwrap();
        first_frontier.mark_source(first_source.clone(), 1);
        let mut second_frontier = TcgenThreadFenceFrontier::default();
        second_frontier
            .merge_pipeline(shift.clone(), &second_clock)
            .unwrap();
        second_frontier.mark_source(second_source.clone(), 1);

        let mut joined = first_frontier.upgraded_to_completed().unwrap();
        joined.merge(&second_frontier).unwrap();
        // Joining an execution-ordering handoff must not turn the second
        // source's pending work into completion of that work.
        assert_eq!(joined.pipeline().len(), 2);
        assert_eq!(
            joined.pipeline()[&copy].async_component(&first),
            first_clock.async_component(&first)
        );
        assert_eq!(
            joined.pipeline()[&shift].async_component(&second),
            second_clock.async_component(&second)
        );
        assert_eq!(
            joined.completed().unwrap().async_component(&first),
            first_clock.async_component(&first)
        );
        assert_eq!(joined.completed().unwrap().async_component(&second), 0);
        assert_eq!(joined.snapshot.lineage.len(), 2);
        assert_eq!(
            joined.snapshot.completion_lineage.get(&first_source),
            Some(&1)
        );
        assert_eq!(joined.snapshot.completion_lineage.get(&second_source), None);

        // A later completion strengthens the same source epoch. A stale
        // pipeline-only publication cannot erase or manufacture that fact.
        joined
            .merge(&second_frontier.upgraded_to_completed().unwrap())
            .unwrap();
        joined.merge(&first_frontier).unwrap();
        assert_eq!(
            joined.completed().unwrap().async_component(&second),
            second_clock.async_component(&second)
        );
        assert_eq!(joined.snapshot.completion_lineage.len(), 2);
    }

    fn after_init_fence(
        state: &RaceCheckLaunchState,
        warp_id: usize,
        sequence: u64,
        barrier_ids: &[crate::PhysicalBarrierId],
    ) -> Result<(), crate::EngineError> {
        let operation = masked_operation(
            warp_id,
            sequence,
            OperationKind::MbarrierInitFence,
            WarpMask::FULL,
        );
        after(
            state,
            &operation,
            OperationEffect::MbarrierInitFence { barrier_ids },
        )
    }

    fn alias_writer(name: &str, sequence: u64) -> Arc<AliasWriter> {
        let operation = operation(0, sequence, OperationKind::Store);
        Arc::new(AliasWriter {
            logical_buffer: Some(name.into()),
            operation: operation.shared_id(),
        })
    }

    fn alias_segment(
        byte_offset: usize,
        byte_end: usize,
        writer_index: usize,
        writer: Arc<AliasWriter>,
    ) -> AliasWriterSegment {
        AliasWriterSegment {
            byte_offset,
            byte_end,
            writer_index,
            writer,
        }
    }

    #[test]
    fn alias_writer_updates_splice_only_the_overlapping_interval() {
        let first = alias_writer("first", 0);
        let second = alias_writer("second", 1);
        let third = alias_writer("third", 2);
        let mut segments = BTreeMap::new();

        overwrite_alias_writer_general(&mut segments, 0, 10, 20, first.clone());
        overwrite_alias_writer_general(&mut segments, 1, 12, 18, second.clone());
        assert_eq!(
            segments.values().cloned().collect::<Vec<_>>(),
            vec![
                alias_segment(10, 12, 0, first.clone()),
                alias_segment(12, 18, 1, second.clone()),
                alias_segment(18, 20, 0, first.clone()),
            ]
        );

        overwrite_alias_writer_general(&mut segments, 2, 5, 14, third.clone());
        assert_eq!(
            segments.values().cloned().collect::<Vec<_>>(),
            vec![
                alias_segment(5, 14, 2, third.clone()),
                alias_segment(14, 18, 1, second.clone()),
                alias_segment(18, 20, 0, first),
            ]
        );

        overwrite_alias_writer_general(&mut segments, 1, 14, 20, second.clone());
        assert_eq!(
            segments.values().cloned().collect::<Vec<_>>(),
            vec![
                alias_segment(5, 14, 2, third),
                alias_segment(14, 20, 1, second),
            ]
        );
    }

    #[test]
    fn alias_writer_updates_merge_adjacent_equal_writers() {
        let writer = alias_writer("same", 0);
        let mut segments = BTreeMap::new();
        overwrite_alias_writer_general(&mut segments, 0, 0, 4, writer.clone());
        overwrite_alias_writer_general(&mut segments, 0, 8, 12, writer.clone());
        overwrite_alias_writer_general(&mut segments, 0, 4, 8, writer.clone());
        assert_eq!(
            segments.values().cloned().collect::<Vec<_>>(),
            vec![alias_segment(0, 12, 0, writer)]
        );
    }

    #[test]
    fn alias_writer_indices_do_not_conflate_distinct_logical_buffers() {
        let first_operation = operation(0, 0, OperationKind::Store);
        let second_operation = operation(0, 1, OperationKind::Store);
        let mut writers = AliasAllocationWriters::default();
        let mut retained_writer = None;

        let first_index = writers.note_writer("first", 0, 16);
        writers.overwrite_writer(
            first_index,
            0,
            16,
            AliasWriterRef {
                logical_buffer: Some("first"),
                operation: first_operation.id(),
            },
            &mut retained_writer,
        );
        assert!(writers.range_owned_by(first_index, 0, 16));

        let second_index = writers.note_writer("second", 4, 12);
        assert_ne!(first_index, second_index);
        retained_writer = None;
        writers.overwrite_writer(
            second_index,
            4,
            12,
            AliasWriterRef {
                logical_buffer: Some("second"),
                operation: second_operation.id(),
            },
            &mut retained_writer,
        );

        assert!(writers.range_owned_by(first_index, 0, 4));
        assert!(!writers.range_owned_by(first_index, 4, 12));
        assert!(writers.range_owned_by(second_index, 4, 12));
        assert!(writers.range_owned_by(first_index, 12, 16));
    }

    #[test]
    fn cluster_state_uses_cluster_local_clock_dimensions_with_global_warp_ids() {
        let topology = LaunchTopology::new(8, 1, 8).unwrap();
        let state = RaceCheckLaunchState::for_cluster(topology, 7, false);
        let race = state.race_for_warp(56).unwrap();

        assert_eq!(race.shadow.warp_count(), 8);
        assert!(race.shadow.warp_clock(56).is_some());
        assert!(race.shadow.warp_clock(63).is_some());
        assert!(race.shadow.warp_clock(55).is_none());
        assert_eq!(race.lane_shadow.global_warp_base, 56);
        assert_eq!(race.lane_shadow.orders.len(), 8);
    }

    #[test]
    fn direct_summary_skips_numeric_memory_gates_without_disabling_online_shadowing() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let mut direct =
            RaceCheckLaunchState::with_direct_summary_for_cluster_and_global_write_allocations(
                topology,
                0,
                std::iter::empty(),
            );
        direct.set_global_memory_model_enabled(false);

        assert!(<RaceCheckMode as EngineModeImpl>::controls_physical_access(
            &direct,
            OperationKind::Load,
            PhysicalAccessSpace::Shared,
        ));
        let global =
            PhysicalAccessDescriptor::new(PhysicalAccessKind::Read, PhysicalAccessSpace::Global, 4)
                .unwrap();
        let shared =
            PhysicalAccessDescriptor::new(PhysicalAccessKind::Read, PhysicalAccessSpace::Shared, 4)
                .unwrap();
        assert!(
            !<RaceCheckMode as EngineModeImpl>::observes_physical_access_batch(
                &direct,
                global,
                WarpMask::FULL,
            )
        );
        direct.set_global_memory_model_enabled(true);
        assert!(
            <RaceCheckMode as EngineModeImpl>::observes_physical_access_batch(
                &direct,
                global,
                WarpMask::FULL,
            )
        );
        assert!(
            <RaceCheckMode as EngineModeImpl>::observes_physical_access_batch(
                &direct,
                shared,
                WarpMask::FULL,
            )
        );
        assert!(!direct.records_resolved_transitions());

        let diagnostic = RaceCheckLaunchState::for_topology(topology);
        assert!(
            <RaceCheckMode as EngineModeImpl>::observes_physical_access_batch(
                &diagnostic,
                global,
                WarpMask::FULL,
            )
        );

        let differential =
            RaceCheckLaunchState::with_summary_for_cluster_and_global_write_allocations(
                topology,
                0,
                crate::ResolvedTransitionLog::default(),
                std::iter::empty(),
            );
        assert!(differential.records_resolved_transitions());
    }

    #[test]
    fn compact_atomic_poll_cache_is_allocation_epoch_guarded() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let watched = PhysicalAllocationId::new(7);
        let unrelated = PhysicalAllocationId::new(8);
        let state =
            RaceCheckLaunchState::with_direct_summary_for_topology_and_global_write_allocations(
                topology,
                [watched, unrelated],
            );
        let descriptor =
            PhysicalAccessDescriptor::new(PhysicalAccessKind::Read, PhysicalAccessSpace::Global, 4)
                .unwrap()
                .with_memory_semantics(crate::MemoryAccessSemantics::scoped(
                    crate::MemoryOrder::Acquire,
                    crate::MemoryScope::Gpu,
                    crate::MemoryProxy::Generic,
                    crate::MemoryAccessClass::Atomic,
                ));
        let make_operation = |sequence| {
            OperationContext::new(
                DynamicOpId::new(0, 0, sequence, StaticOpId::new(99), []),
                OperationKind::Load,
                WarpMask::from_bits(1),
            )
        };
        let lane_spans = std::array::from_fn(|lane| {
            (lane == 0).then_some(PhysicalByteSpan::new(watched, 0, 4).unwrap())
        });

        let first_operation = make_operation(1);
        let first =
            CompactPhysicalAccessBatch::new(&first_operation, descriptor, None, false, &lane_spans);
        assert!(!state.begin_compact_global_read(&first).unwrap());
        state.remember_compact_global_read(&first).unwrap();
        assert_eq!(
            state.finish_compact_global_read(&first).unwrap(),
            CompactGlobalReadFinish::Stable,
        );

        state.begin_global_writes([unrelated]);
        state.finish_global_writes([unrelated]);
        let second_operation = make_operation(2);
        let second = CompactPhysicalAccessBatch::new(
            &second_operation,
            descriptor,
            None,
            false,
            &lane_spans,
        );
        assert!(state.begin_compact_global_read(&second).unwrap());
        assert_eq!(
            state.finish_compact_global_read(&second).unwrap(),
            CompactGlobalReadFinish::Stable,
        );

        state.begin_global_writes([watched]);
        let third_operation = make_operation(3);
        let third =
            CompactPhysicalAccessBatch::new(&third_operation, descriptor, None, false, &lane_spans);
        assert!(!state.begin_compact_global_read(&third).unwrap());
        state.remember_compact_global_read(&third).unwrap();
        state.finish_global_writes([watched]);
        assert_eq!(
            state.finish_compact_global_read(&third).unwrap(),
            CompactGlobalReadFinish::Reprocess,
        );

        let fourth_operation = make_operation(4);
        let fourth = CompactPhysicalAccessBatch::new(
            &fourth_operation,
            descriptor,
            None,
            false,
            &lane_spans,
        );
        assert!(!state.begin_compact_global_read(&fourth).unwrap());
        assert_eq!(
            state.finish_compact_global_read(&fourth).unwrap(),
            CompactGlobalReadFinish::Reprocess,
        );

        let plain_descriptor =
            PhysicalAccessDescriptor::new(PhysicalAccessKind::Read, PhysicalAccessSpace::Global, 4)
                .unwrap();
        let fifth_operation = make_operation(5);
        let fifth = CompactPhysicalAccessBatch::new(
            &fifth_operation,
            plain_descriptor,
            None,
            false,
            &lane_spans,
        );
        assert!(!state.begin_compact_global_read(&fifth).unwrap());
        state.remember_compact_global_read(&fifth).unwrap();
        assert_eq!(
            state.finish_compact_global_read(&fifth).unwrap(),
            CompactGlobalReadFinish::Stable,
        );

        let sixth_operation = make_operation(6);
        let sixth = CompactPhysicalAccessBatch::new(
            &sixth_operation,
            plain_descriptor,
            None,
            false,
            &lane_spans,
        );
        assert!(state.begin_compact_global_read(&sixth).unwrap());
        assert_eq!(
            state.finish_compact_global_read(&sixth).unwrap(),
            CompactGlobalReadFinish::Stable,
        );
    }

    #[test]
    fn declared_wait_history_waits_for_its_word_publication() {
        use std::sync::mpsc::{channel, RecvTimeoutError};
        use std::time::Duration;

        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let allocation = PhysicalAllocationId::new(7);
        let state =
            RaceCheckLaunchState::for_topology_and_global_write_allocations(topology, [allocation]);
        let span = PhysicalByteSpan::new(allocation, 0, 4).unwrap();
        let unrelated = PhysicalByteSpan::new(allocation, 8, 4).unwrap();
        let operation = masked_operation(0, 0, OperationKind::Store, WarpMask::from_bits(1));
        let descriptor = PhysicalAccessDescriptor::new(
            PhysicalAccessKind::Write,
            PhysicalAccessSpace::Global,
            4,
        )
        .unwrap()
        .with_memory_semantics(crate::MemoryAccessSemantics::scoped(
            crate::MemoryOrder::Release,
            crate::MemoryScope::Gpu,
            crate::MemoryProxy::Generic,
            crate::MemoryAccessClass::Atomic,
        ));
        let batch = PhysicalAccessBatch::resolve(operation.clone(), descriptor, |_| {
            Ok::<_, std::convert::Infallible>(vec![span])
        })
        .unwrap()
        .with_declared_values(Arc::from([1_u64]));
        before(&state, &operation, OperationEffect::PhysicalAccess(&batch)).unwrap();
        // Another word in the allocation need not wait for this publication.
        assert_eq!(
            state.declared_word_candidates(unrelated, 1, 0).unwrap(),
            (0, vec![])
        );
        std::thread::scope(|scope| {
            let (started_tx, started_rx) = channel();
            let (result_tx, result_rx) = channel();
            let reader_state = &state;
            let reader = scope.spawn(move || {
                started_tx.send(()).unwrap();
                result_tx
                    .send(reader_state.declared_word_candidates(span, 1, 0).unwrap())
                    .unwrap();
            });
            started_rx.recv().unwrap();
            let early = result_rx.recv_timeout(Duration::from_millis(50));
            // Always release the reader, including when the old implementation
            // returned stale history and the assertion below must fail.
            after(&state, &operation, OperationEffect::PhysicalAccess(&batch)).unwrap();
            let completed = match &early {
                Ok(result) => result.clone(),
                Err(RecvTimeoutError::Timeout) => {
                    result_rx.recv_timeout(Duration::from_secs(2)).unwrap()
                }
                Err(error) => panic!("reader failed: {error}"),
            };
            reader.join().unwrap();
            assert!(
                matches!(early, Err(RecvTimeoutError::Timeout)),
                "returned before publication: {early:?}"
            );
            assert_eq!(completed, (0, vec![1]));
        });
    }

    #[test]
    fn compact_atomic_poll_cache_rejects_writes_between_lookup_and_admission() {
        // The read-from lookup runs before the entry is admitted; a write that
        // begins and finishes in between must invalidate the entry, or the
        // next poll would skip the acquire of the version it now observes.
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let watched = PhysicalAllocationId::new(7);
        let state =
            RaceCheckLaunchState::with_direct_summary_for_topology_and_global_write_allocations(
                topology,
                [watched],
            );
        let descriptor =
            PhysicalAccessDescriptor::new(PhysicalAccessKind::Read, PhysicalAccessSpace::Global, 4)
                .unwrap()
                .with_memory_semantics(crate::MemoryAccessSemantics::scoped(
                    crate::MemoryOrder::Acquire,
                    crate::MemoryScope::Gpu,
                    crate::MemoryProxy::Generic,
                    crate::MemoryAccessClass::Atomic,
                ));
        let operation = OperationContext::new(
            DynamicOpId::new(0, 0, 1, StaticOpId::new(99), []),
            OperationKind::Load,
            WarpMask::from_bits(1),
        );
        let lane_spans = std::array::from_fn(|lane| {
            (lane == 0).then_some(PhysicalByteSpan::new(watched, 0, 4).unwrap())
        });
        let batch =
            CompactPhysicalAccessBatch::new(&operation, descriptor, None, false, &lane_spans);
        assert!(!state.begin_compact_global_read(&batch).unwrap());
        state.begin_global_writes([watched]);
        state.finish_global_writes([watched]);
        state.remember_compact_global_read(&batch).unwrap();
        assert_eq!(
            state.finish_compact_global_read(&batch).unwrap(),
            CompactGlobalReadFinish::Reprocess,
        );
    }

    #[test]
    fn access_inspection_keeps_read_only_global_allocations_outside_the_hb_shadow() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let output = PhysicalAllocationId::new(7);
        let input = PhysicalAllocationId::new(9);
        let diagnostic =
            RaceCheckLaunchState::for_topology_and_global_write_allocations(topology, [output]);

        assert!(
            <RaceCheckMode as EngineModeImpl>::controls_physical_access_allocation(
                &diagnostic,
                OperationKind::Load,
                PhysicalAccessSpace::Global,
                Some(input),
            )
        );
        assert!(
            <RaceCheckMode as EngineModeImpl>::controls_physical_access_allocation(
                &diagnostic,
                OperationKind::Store,
                PhysicalAccessSpace::Global,
                Some(output),
            )
        );
        assert!(<RaceCheckMode as EngineModeImpl>::controls_physical_access(
            &diagnostic,
            OperationKind::Load,
            PhysicalAccessSpace::Register,
        ));
        for shard in &diagnostic.race.race_shards {
            assert_eq!(
                shard
                    .global
                    .lock()
                    .expect("racecheck global state poisoned")
                    .retained_state_counts(),
                (0, 0, 0)
            );
        }
    }

    #[test]
    fn uncontrolled_access_count_is_atomic_saturating_and_snapshot_stable() {
        const THREADS: usize = 8;
        const ACCESSES_PER_THREAD: usize = 4_096;

        let state = Arc::new(RaceCheckLaunchState::new(1));
        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                let state = Arc::clone(&state);
                scope.spawn(move || {
                    for _ in 0..ACCESSES_PER_THREAD {
                        <RaceCheckMode as EngineModeImpl>::after_unobserved_physical_access(
                            &state, 0,
                        )
                        .unwrap();
                    }
                });
            }
        });
        state.race_for_warp(0).unwrap().access_count = 3;

        let expected = THREADS * ACCESSES_PER_THREAD + 3;
        let first = state.result_before_aborted_execution();
        assert_eq!(first.access_count(), expected);
        assert_eq!(first.clone().access_count(), expected);
        assert_eq!(
            state.result_before_aborted_execution().access_count(),
            expected,
            "reading a result must not consume or reset the launch counter",
        );

        state
            .shard_for_warp(0)
            .unwrap()
            .uncontrolled_access_count
            .store(usize::MAX, Ordering::Relaxed);
        <RaceCheckMode as EngineModeImpl>::after_unobserved_physical_access(&state, 0).unwrap();
        assert_eq!(
            state.result_before_aborted_execution().access_count(),
            usize::MAX,
        );
    }

    fn masked_operation(
        warp_id: usize,
        sequence: u64,
        kind: OperationKind,
        mask: WarpMask,
    ) -> OperationContext {
        OperationContext::new(
            DynamicOpId::new(0, warp_id, sequence, StaticOpId::new(sequence + 1), []),
            kind,
            mask,
        )
    }

    fn batch(operation: OperationContext, kind: PhysicalAccessKind) -> PhysicalAccessBatch {
        batch_at(operation, kind, 7, 16)
    }

    fn batch_at(
        operation: OperationContext,
        kind: PhysicalAccessKind,
        allocation: u64,
        byte_offset: usize,
    ) -> PhysicalAccessBatch {
        batch_range(operation, kind, allocation, byte_offset, 4)
    }

    fn global_batch_at(
        operation: OperationContext,
        kind: PhysicalAccessKind,
        allocation: u64,
        byte_offset: usize,
    ) -> PhysicalAccessBatch {
        let descriptor =
            PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Global, 4).unwrap();
        PhysicalAccessBatch::resolve(operation, descriptor, |_| {
            Ok::<_, std::convert::Infallible>(vec![PhysicalByteSpan::new(
                PhysicalAllocationId::new(allocation),
                byte_offset,
                4,
            )
            .unwrap()])
        })
        .unwrap()
    }

    fn batch_range(
        operation: OperationContext,
        kind: PhysicalAccessKind,
        allocation: u64,
        byte_offset: usize,
        byte_len: usize,
    ) -> PhysicalAccessBatch {
        let descriptor =
            PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Shared, byte_len).unwrap();
        PhysicalAccessBatch::resolve(operation, descriptor, |_| {
            Ok::<_, std::convert::Infallible>(vec![PhysicalByteSpan::new(
                PhysicalAllocationId::new(allocation),
                byte_offset,
                byte_len,
            )
            .unwrap()])
        })
        .unwrap()
    }

    fn tmem_batch_range(
        operation: OperationContext,
        kind: PhysicalAccessKind,
        allocation: u64,
        byte_offset: usize,
        byte_len: usize,
    ) -> PhysicalAccessBatch {
        let descriptor =
            PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Tmem, byte_len).unwrap();
        PhysicalAccessBatch::resolve(operation, descriptor, |_| {
            Ok::<_, std::convert::Infallible>(vec![PhysicalByteSpan::new(
                PhysicalAllocationId::new(allocation),
                byte_offset,
                byte_len,
            )
            .unwrap()])
        })
        .unwrap()
    }

    // Exercise the real lane-clock + shared conflict-shadow composition.
    struct SameWarpExecution {
        lane: SameWarpLaneShadow,
        shadow: crate::race_shadow::RaceShadow,
    }

    impl SameWarpExecution {
        fn new() -> Self {
            Self {
                lane: SameWarpLaneShadow::new(1),
                shadow: crate::race_shadow::RaceShadow::new(1),
            }
        }

        fn validate_batch(
            &mut self,
            batch: &PhysicalAccessBatch,
        ) -> Result<
            (
                SameWarpLaneBatchValidation,
                crate::race_shadow::RaceBatchValidation,
            ),
            crate::race_shadow::RaceShadowError,
        > {
            let lane = self.lane.validate_order_batch(batch).unwrap();
            let order = self
                .lane
                .race_lane_order(batch.operation().id().global_warp_id())
                .unwrap();
            let shadow = self.shadow.validate_batch_with_lane_order(batch, &order)?;
            Ok((lane, shadow))
        }

        fn commit_batch(
            &mut self,
            (lane, shadow): (
                SameWarpLaneBatchValidation,
                crate::race_shadow::RaceBatchValidation,
            ),
        ) -> Vec<PhysicalRaceFinding> {
            self.lane.commit_batch(lane);
            self.shadow.commit_validation(shadow)
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct RetainedLaneEventStamp {
        pub(super) warp_id: usize,
        pub(super) lane: u8,
        pub(super) epoch: u64,
    }

    #[derive(Clone)]
    struct ReferenceLaneAccess {
        stamp: RetainedLaneEventStamp,
        witness: PhysicalRaceWitness,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct ReferenceSameWarpByteKey {
        warp_id: usize,
        space: PhysicalAccessSpace,
        allocation: PhysicalAllocationId,
        byte_offset: usize,
    }

    struct ReferenceSameWarpLaneBatchValidation {
        order_index: usize,
        lane_updates: Box<[(u8, LaneFrontier)]>,
        byte_updates: BTreeMap<ReferenceSameWarpByteKey, Vec<ReferenceLaneAccess>>,
        review_findings: Vec<PhysicalRaceFinding>,
    }

    #[derive(Clone)]
    struct ReferenceSameWarpLaneShadow {
        global_warp_base: usize,
        orders: Vec<WarpLaneOrder>,
        bytes: BTreeMap<ReferenceSameWarpByteKey, Vec<ReferenceLaneAccess>>,
    }

    impl ReferenceSameWarpLaneShadow {
        fn new(warp_count: usize) -> Self {
            Self {
                global_warp_base: 0,
                orders: vec![WarpLaneOrder::default(); warp_count],
                bytes: BTreeMap::new(),
            }
        }

        fn order_index(&self, warp_id: usize) -> usize {
            warp_id
                .checked_sub(self.global_warp_base)
                .filter(|order_index| *order_index < self.orders.len())
                .expect("the reference shadow test uses an in-range warp")
        }

        fn validate_batch(
            &self,
            batch: &PhysicalAccessBatch,
        ) -> Result<ReferenceSameWarpLaneBatchValidation, Vec<PhysicalRaceFinding>> {
            let warp_id = batch.operation().id().global_warp_id();
            let order_index = self.order_index(warp_id);
            let order = &self.orders[order_index];
            let descriptor = batch.descriptor();
            let tracks_conflicts = tracks_race_conflicts(descriptor.space());
            let mut byte_updates = BTreeMap::new();
            let mut lane_updates = Vec::with_capacity(batch.lanes().len());
            let mut review_findings = Vec::new();
            let mut conflicts = Vec::new();

            for lane in batch.lanes() {
                let lane_id = lane.provenance().lane();
                let stamp = order.preview_tick(warp_id, lane_id).unwrap();
                lane_updates.push((stamp.lane, stamp.materialized_frontier()));
                for span in lane
                    .footprint()
                    .spans()
                    .iter()
                    .copied()
                    .filter(|_| tracks_conflicts)
                {
                    let current = ReferenceLaneAccess {
                        stamp: stamp.retained_epoch(),
                        witness: PhysicalRaceWitness::from_lane(
                            lane,
                            descriptor.kind(),
                            descriptor.space(),
                            span,
                        ),
                    };
                    for byte_offset in span.byte_offset()..span.byte_end() {
                        let key = ReferenceSameWarpByteKey {
                            warp_id,
                            space: descriptor.space(),
                            allocation: span.allocation(),
                            byte_offset,
                        };
                        let byte_state = byte_updates
                            .entry(key)
                            .or_insert_with(|| self.bytes.get(&key).cloned().unwrap_or_default());
                        for prior in byte_state.iter() {
                            if stamp.observes(&prior.stamp) {
                                continue;
                            }
                            for kind in same_warp_conflict_kinds(
                                prior.witness.kind(),
                                current.witness.kind(),
                            ) {
                                let start = prior
                                    .witness
                                    .span()
                                    .byte_offset()
                                    .max(current.witness.span().byte_offset());
                                let end = prior
                                    .witness
                                    .span()
                                    .byte_end()
                                    .min(current.witness.span().byte_end());
                                let overlap =
                                    PhysicalByteSpan::new(span.allocation(), start, end - start)
                                        .unwrap();
                                let finding = PhysicalRaceFinding::new(
                                    kind,
                                    PhysicalRaceOrderingFailure::MissingSameWarpLaneOrder,
                                    prior.witness.clone(),
                                    current.witness.clone(),
                                    overlap,
                                );
                                if finding.requires_unwaited_tmem_load_review() {
                                    if !review_findings.contains(&finding) {
                                        review_findings.push(finding);
                                    }
                                } else if !conflicts.contains(&finding) {
                                    conflicts.push(finding);
                                }
                            }
                        }
                        byte_state.push(current.clone());
                    }
                }
            }
            if !conflicts.is_empty() {
                return Err(conflicts);
            }
            Ok(ReferenceSameWarpLaneBatchValidation {
                order_index,
                lane_updates: lane_updates.into_boxed_slice(),
                byte_updates,
                review_findings,
            })
        }

        fn commit_batch(&mut self, validation: ReferenceSameWarpLaneBatchValidation) {
            for (lane, frontier) in validation.lane_updates {
                self.orders[validation.order_index]
                    .commit_lane_frontier(usize::from(lane), frontier);
            }
            for (byte, state) in validation.byte_updates {
                self.bytes.insert(byte, state);
            }
        }

        fn barrier_release(&mut self, warp_id: usize, mask: WarpMask) -> SharedClockFrontier {
            let order_index = self.order_index(warp_id);
            SharedClockFrontier::single(
                warp_id,
                self.orders[order_index].release(warp_id, mask).unwrap(),
            )
        }

        fn barrier_acquire(
            &mut self,
            warp_id: usize,
            mask: WarpMask,
            payload: Option<&SharedClockFrontier>,
        ) {
            let order_index = self.order_index(warp_id);
            self.orders[order_index]
                .acquire(
                    warp_id,
                    mask,
                    payload.and_then(|payload| payload.release_for(warp_id)),
                )
                .unwrap();
        }
    }

    fn named_batch(
        operation: OperationContext,
        kind: PhysicalAccessKind,
        logical_buffer: &str,
    ) -> PhysicalAccessBatch {
        batch(operation, kind).with_logical_buffer(logical_buffer)
    }

    fn commit_access(state: &RaceCheckLaunchState, batch: &PhysicalAccessBatch) {
        let operation = batch.operation();
        before(state, operation, OperationEffect::PhysicalAccess(batch)).unwrap();
        after(state, operation, OperationEffect::PhysicalAccess(batch)).unwrap();
    }

    fn before(
        state: &RaceCheckLaunchState,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
    ) -> Result<(), crate::EngineError> {
        <RaceCheckMode as EngineModeImpl>::before_effect(state, operation, effect)
    }

    fn after(
        state: &RaceCheckLaunchState,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
    ) -> Result<(), crate::EngineError> {
        <RaceCheckMode as EngineModeImpl>::after_effect(state, operation, effect)
    }

    fn staged_arrive(plan: crate::runtime::PhysicalMbarrierArrivePlan) -> OperationEffect<'static> {
        OperationEffect::MbarrierArrive {
            plan,
            outcome: None,
        }
    }

    #[test]
    fn exact_pool_alias_provenance_is_review_with_strict_precedence() {
        let state = RaceCheckLaunchState::new(2);
        let writer = named_batch(
            operation(0, 0, OperationKind::Store),
            PhysicalAccessKind::Write,
            "B_shared",
        );
        let reader = named_batch(
            operation(0, 1, OperationKind::Load),
            PhysicalAccessKind::Read,
            "A_shared",
        );
        commit_access(&state, &writer);
        commit_access(&state, &reader);

        let review = state.result();
        assert_eq!(review.status(), RaceCheckStatus::Review);
        assert_eq!(review.access_count(), 2);
        assert!(review.accesses_complete());
        assert_eq!(review.accesses().len(), 2);
        assert_eq!(review.advisories().len(), 1);
        let advisory = &review.advisories()[0];
        assert_eq!(advisory.reader_buffer(), "A_shared");
        assert_eq!(advisory.writer_buffer(), "B_shared");
        assert_eq!(advisory.space(), PhysicalAccessSpace::Shared);
        assert_eq!(advisory.allocation(), PhysicalAllocationId::new(7));
        assert_eq!(advisory.overlaps().len(), 1);
        assert_eq!(advisory.overlaps()[0].byte_offset(), 16);
        assert_eq!(advisory.overlaps()[0].byte_len(), 4);

        let uncommitted = named_batch(
            operation(0, 2, OperationKind::Store),
            PhysicalAccessKind::Write,
            "A_shared",
        );
        before(
            &state,
            uncommitted.operation(),
            OperationEffect::PhysicalAccess(&uncommitted),
        )
        .unwrap();
        let incomplete = state.result();
        assert_eq!(incomplete.status(), RaceCheckStatus::Incomplete);
        assert_eq!(incomplete.advisories().len(), 1);

        let racing = named_batch(
            operation(1, 0, OperationKind::Store),
            PhysicalAccessKind::Write,
            "C_shared",
        );
        assert!(before(
            &state,
            racing.operation(),
            OperationEffect::PhysicalAccess(&racing),
        )
        .is_err());
        let error = state.result();
        assert_eq!(error.status(), RaceCheckStatus::Error);
        assert_eq!(error.advisories().len(), 1);
    }

    #[test]
    fn report_coalesces_dynamic_alias_advisories_by_static_site() {
        let mut tracker = AliasTracker::default();
        for (warp_id, allocation) in [(1, 8), (0, 7)] {
            let writer = batch_at(
                operation(warp_id, 0, OperationKind::Store),
                PhysicalAccessKind::Write,
                allocation,
                16,
            )
            .with_logical_buffer("B_shared");
            let reader = batch_at(
                operation(warp_id, 1, OperationKind::Load),
                PhysicalAccessKind::Read,
                allocation,
                16,
            )
            .with_logical_buffer("A_shared");
            tracker.observe_batch(&writer);
            tracker.observe_batch(&reader);
        }

        let dynamic = tracker.advisories();
        assert_eq!(dynamic.len(), 2);
        let mut report = Vec::new();
        append_report_advisories(&mut report, dynamic.into_iter().rev());
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].occurrences(), 2);
        assert_eq!(report[0].reader_operation().global_warp_id(), 0);
        assert_eq!(report[0].writer_operation().global_warp_id(), 0);
        assert_eq!(report[0].allocation(), PhysicalAllocationId::new(7));
    }

    #[test]
    fn same_logical_name_and_never_written_alias_are_clean() {
        let state = RaceCheckLaunchState::new(1);
        let first_read = named_batch(
            operation(0, 0, OperationKind::Load),
            PhysicalAccessKind::Read,
            "A_shared",
        );
        let writer = named_batch(
            operation(0, 1, OperationKind::Store),
            PhysicalAccessKind::Write,
            "A_shared",
        );
        let second_read = named_batch(
            operation(0, 2, OperationKind::Load),
            PhysicalAccessKind::Read,
            "A_shared",
        );
        commit_access(&state, &first_read);
        commit_access(&state, &writer);
        commit_access(&state, &second_read);

        let result = state.result();
        assert_eq!(result.status(), RaceCheckStatus::Clean);
        assert!(result.advisories().is_empty());
    }

    #[test]
    fn disjoint_different_logical_buffers_are_proven_clean() {
        let mut tracker = AliasTracker::default();
        let writer = batch_at(
            operation(0, 0, OperationKind::Store),
            PhysicalAccessKind::Write,
            7,
            16,
        )
        .with_logical_buffer("B_shared");
        let reader = batch_at(
            operation(0, 1, OperationKind::Load),
            PhysicalAccessKind::Read,
            7,
            32,
        )
        .with_logical_buffer("A_shared");

        tracker.observe_batch(&writer);
        let writers = tracker
            .last_writers
            .get(&(PhysicalAccessSpace::Shared, PhysicalAllocationId::new(7)))
            .expect("the named writer is retained");
        assert!(writers.cannot_alias_read("A_shared", 32, 36));
        assert!(!writers.cannot_alias_read("A_shared", 16, 20));

        tracker.observe_batch(&reader);
        assert!(tracker.advisories().is_empty());
    }

    fn committed_arrive(
        plan: crate::runtime::PhysicalMbarrierArrivePlan,
        outcome: crate::PhysicalMbarrierArrivalOutcome,
    ) -> OperationEffect<'static> {
        OperationEffect::MbarrierArrive {
            plan,
            outcome: Some(outcome),
        }
    }

    fn run_cluster_split_hb(
        arrival_semantics: ClusterBarrierArrivalSemantics,
    ) -> (Arc<RaceCheckLaunchState>, ExecutionReport) {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocations = (0..2)
            .map(|cta_id| {
                let owner = CtaId::new(topology, 0, cta_id).unwrap();
                physical.shared().allocate_cta_zeroed(owner, 4).unwrap()
            })
            .collect();
        // Both CTAs name CTA 0's DSM cell explicitly. This keeps the cluster
        // barrier HB regression in Racecheck's checked shared-memory domain.
        let buffer = RuntimeBuffer::RemoteShared {
            allocations: Arc::new(allocations),
            byte_offsets: WarpValue::splat(0_i64),
            byte_len: 4,
            target_cta_ids: WarpValue::splat(0_i64),
            virtual_base: 0,
        };
        let state = Arc::new(RaceCheckLaunchState::for_topology(topology));
        let report = run_kernel_engine_launch_report::<RaceCheckMode, _, _>(
            physical,
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let buffer = buffer.clone();
                async move {
                    let warp_id = warp.context().global_warp_id();
                    let lane_mask = WarpMask::from_lanes([0]).unwrap();
                    let indices = WarpValue::splat(0_i64);
                    if warp_id == 0 {
                        let context = warp.context().with_active_mask(lane_mask);
                        let store = warp.begin_operation(
                            context,
                            StaticOpId::new(3000),
                            OperationKind::Store,
                            [],
                        )?;
                        warp.runtime_physical_access(
                            Some(&store),
                            PhysicalAccessDescriptor::new(
                                PhysicalAccessKind::Write,
                                PhysicalAccessSpace::Shared,
                                4,
                            )
                            .unwrap(),
                            &buffer,
                            &indices,
                            lane_mask,
                            || Ok(()),
                        )?;
                        warp.finish_operation(&store)?;
                    }

                    let arrive = warp.begin_operation(
                        warp.context(),
                        StaticOpId::new(3001),
                        OperationKind::Collective,
                        [],
                    )?;
                    warp.cluster_barrier_arrive(
                        Some(&arrive),
                        WarpMask::FULL,
                        arrival_semantics == ClusterBarrierArrivalSemantics::Release,
                        true,
                    )?;
                    warp.finish_operation(&arrive)?;

                    let wait = warp.begin_operation(
                        warp.context(),
                        StaticOpId::new(3002),
                        OperationKind::Collective,
                        [],
                    )?;
                    warp.cluster_barrier_wait(Some(&wait), WarpMask::FULL, true)
                        .await?;
                    warp.finish_operation(&wait)?;

                    if warp_id == 1 {
                        let context = warp.context().with_active_mask(lane_mask);
                        let load = warp.begin_operation(
                            context,
                            StaticOpId::new(3003),
                            OperationKind::Load,
                            [],
                        )?;
                        warp.runtime_physical_access(
                            Some(&load),
                            PhysicalAccessDescriptor::new(
                                PhysicalAccessKind::Read,
                                PhysicalAccessSpace::Shared,
                                4,
                            )
                            .unwrap(),
                            &buffer,
                            &indices,
                            lane_mask,
                            || Ok(()),
                        )?;
                        warp.finish_operation(&load)?;
                    }
                    Ok(())
                }
            },
        );
        (state, report)
    }

    #[test]
    fn summary_evidence_counts_accesses_and_preserves_alias_advisories() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let state = RaceCheckLaunchState::with_optional_topology(
            0,
            topology.warp_count(),
            Some(topology),
            crate::ResolvedTransitionLog::default(),
            false,
            None,
            false,
        );
        let writer = named_batch(
            operation(0, 0, OperationKind::Store),
            PhysicalAccessKind::Write,
            "B_shared",
        );
        let reader = named_batch(
            operation(0, 1, OperationKind::Load),
            PhysicalAccessKind::Read,
            "A_shared",
        );
        commit_access(&state, &writer);
        commit_access(&state, &reader);

        let result = state.result();
        assert_eq!(result.status(), RaceCheckStatus::Review);
        assert_eq!(result.access_count(), 2);
        assert!(!result.accesses_complete());
        assert!(result.accesses().is_empty());
        assert_eq!(result.advisories().len(), 1);
        assert_eq!(result.advisories()[0].reader_buffer(), "A_shared");
        assert_eq!(result.advisories()[0].writer_buffer(), "B_shared");
    }

    #[test]
    fn summary_evidence_does_not_retain_access_records() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let state = RaceCheckLaunchState::with_optional_topology(
            0,
            topology.warp_count(),
            Some(topology),
            crate::ResolvedTransitionLog::default(),
            false,
            None,
            false,
        );
        let descriptor = PhysicalAccessDescriptor::new(
            PhysicalAccessKind::Write,
            PhysicalAccessSpace::Global,
            4,
        )
        .unwrap();
        let global =
            PhysicalAccessBatch::resolve(operation(0, 0, OperationKind::Store), descriptor, |_| {
                Ok::<_, std::convert::Infallible>(vec![PhysicalByteSpan::new(
                    PhysicalAllocationId::new(9),
                    0,
                    4,
                )
                .unwrap()])
            })
            .unwrap();
        let shared = batch_at(
            operation(0, 1, OperationKind::Store),
            PhysicalAccessKind::Write,
            10,
            0,
        );

        commit_access(&state, &global);
        commit_access(&state, &shared);

        let result = state.result();
        assert_eq!(result.access_count(), 2);
        assert!(!result.accesses_complete());
        assert!(result.accesses().is_empty());
    }

    #[test]
    fn diagnostic_results_keep_full_accesses() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let state = RaceCheckLaunchState::with_optional_topology(
            0,
            topology.warp_count(),
            Some(topology),
            crate::ResolvedTransitionLog::default(),
            true,
            None,
            false,
        );
        let global_descriptor = PhysicalAccessDescriptor::new(
            PhysicalAccessKind::Write,
            PhysicalAccessSpace::Global,
            4,
        )
        .unwrap();
        let global = PhysicalAccessBatch::resolve(
            operation(0, 0, OperationKind::Store),
            global_descriptor,
            |_| {
                Ok::<_, std::convert::Infallible>(vec![PhysicalByteSpan::new(
                    PhysicalAllocationId::new(9),
                    0,
                    4,
                )
                .unwrap()])
            },
        )
        .unwrap();
        let shared = batch_at(
            operation(0, 1, OperationKind::Store),
            PhysicalAccessKind::Write,
            10,
            0,
        );

        commit_access(&state, &global);
        commit_access(&state, &shared);

        let result = state.result();
        assert!(result.accesses_complete());
        assert_eq!(result.accesses().len(), 2);
    }

    #[test]
    fn unordered_writes_report_exact_physical_race_before_numeric_commit() {
        let state = RaceCheckLaunchState::new(2);
        let first_op = operation(0, 0, OperationKind::Store);
        let first = batch(first_op.clone(), PhysicalAccessKind::Write);
        before(&state, &first_op, OperationEffect::PhysicalAccess(&first)).unwrap();
        after(&state, &first_op, OperationEffect::PhysicalAccess(&first)).unwrap();

        let second_op = operation(1, 0, OperationKind::Store);
        let second = batch(second_op.clone(), PhysicalAccessKind::Write);
        let error =
            before(&state, &second_op, OperationEffect::PhysicalAccess(&second)).unwrap_err();

        assert!(error.to_string().contains("unordered write/write"));
        let result = state.result();
        assert_eq!(result.status(), RaceCheckStatus::Error);
        assert_eq!(result.findings().len(), 1);
        assert_eq!(result.findings()[0].current().operation(), second_op.id());
        assert_eq!(result.findings()[0].overlap().byte_offset(), 16);
    }

    #[test]
    fn staged_access_revalidates_after_a_concurrent_shadow_commit() {
        let state = RaceCheckLaunchState::new(2);
        let staged_op = operation(0, 0, OperationKind::Store);
        let staged = batch(staged_op.clone(), PhysicalAccessKind::Write);
        before(&state, &staged_op, OperationEffect::PhysicalAccess(&staged)).unwrap();

        let concurrent_op = operation(1, 0, OperationKind::Store);
        let concurrent = batch(concurrent_op, PhysicalAccessKind::Write);
        commit_access(&state, &concurrent);

        let error =
            after(&state, &staged_op, OperationEffect::PhysicalAccess(&staged)).unwrap_err();
        assert!(error.to_string().contains("unordered write/write"));
        let result = state.result();
        assert_eq!(result.status(), RaceCheckStatus::Error);
        assert_eq!(result.findings().len(), 1);
        assert_eq!(result.findings()[0].current().operation(), staged_op.id());
    }

    #[test]
    fn compressed_warp_lane_order_matches_naive_matrix() {
        struct NaiveWarpLaneOrder {
            observed: [LaneFrontier; WARP_SIZE],
        }

        impl NaiveWarpLaneOrder {
            fn tick(&mut self, lane: usize) {
                self.observed[lane][lane] = self.observed[lane][lane].checked_add(1).unwrap();
            }

            fn release(&mut self, mask: WarpMask) -> LaneFrontier {
                let mut release = [0; WARP_SIZE];
                for lane in mask {
                    self.tick(lane);
                    for (released, observed) in release.iter_mut().zip(self.observed[lane]) {
                        *released = (*released).max(observed);
                    }
                }
                release
            }

            fn acquire(&mut self, mask: WarpMask, release: Option<&LaneFrontier>) {
                for lane in mask {
                    if let Some(release) = release {
                        for (observed, released) in self.observed[lane].iter_mut().zip(release) {
                            *observed = (*observed).max(*released);
                        }
                    }
                    self.tick(lane);
                }
            }

            fn preview_tick(&self, lane: usize) -> LaneFrontier {
                let mut frontier = self.observed[lane];
                frontier[lane] = frontier[lane].checked_add(1).unwrap();
                frontier
            }
        }

        fn next_random(seed: &mut u64) -> u64 {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *seed
        }

        fn random_mask(seed: &mut u64, step: usize) -> WarpMask {
            match step % 13 {
                0 => WarpMask::FULL,
                1 => WarpMask::EMPTY,
                _ => WarpMask::from_bits(next_random(seed) as u32),
            }
        }

        let mut compressed = WarpLaneOrder::default();
        let mut naive = NaiveWarpLaneOrder {
            observed: [[0; WARP_SIZE]; WARP_SIZE],
        };
        let mut seed = 0xc10c_cafe_5eed_1234_u64;

        for step in 0..4_096 {
            match next_random(&mut seed) % 6 {
                0 => {
                    let lane = next_random(&mut seed) as usize % WARP_SIZE;
                    compressed.tick(7, lane).unwrap();
                    naive.tick(lane);
                }
                1 => {
                    let mask = random_mask(&mut seed, step);
                    assert_eq!(
                        compressed.release(7, mask).unwrap(),
                        naive.release(mask),
                        "release at step {step}",
                    );
                }
                2 | 3 => {
                    let mask = random_mask(&mut seed, step);
                    let release = std::array::from_fn(|_| next_random(&mut seed) % 8_192);
                    let release = (step % 3 != 0).then_some(release);
                    compressed.acquire(7, mask, release.as_ref()).unwrap();
                    naive.acquire(mask, release.as_ref());
                }
                4 => {
                    let lane = next_random(&mut seed) as usize % WARP_SIZE;
                    let stamp = compressed.preview_tick(7, lane).unwrap();
                    let frontier = stamp.materialized_frontier();
                    assert_eq!(frontier, naive.preview_tick(lane), "preview at step {step}",);
                    compressed.commit_lane_epoch(lane, stamp.epoch);
                    naive.observed[lane] = frontier;
                }
                _ => {
                    let lane = next_random(&mut seed) as usize % WARP_SIZE;
                    let frontier = std::array::from_fn(|source_lane| {
                        naive.observed[lane][source_lane] + next_random(&mut seed) % 4
                    });
                    compressed.commit_lane_frontier(lane, frontier);
                    naive.observed[lane] = frontier;
                }
            }

            let materialized: [LaneFrontier; WARP_SIZE] = std::array::from_fn(|lane| {
                std::array::from_fn(|source_lane| compressed.component(lane, source_lane))
            });
            assert_eq!(materialized, naive.observed, "state at step {step}");
            let joined = std::array::from_fn(|source_lane| {
                naive
                    .observed
                    .iter()
                    .map(|frontier| frontier[source_lane])
                    .max()
                    .unwrap()
            });
            assert_eq!(compressed.joined, joined, "joined state at step {step}");
        }
    }

    #[test]
    fn different_lanes_in_one_warp_are_not_implicitly_program_ordered() {
        let state = RaceCheckLaunchState::new(1);
        let writer = masked_operation(
            0,
            0,
            OperationKind::Store,
            WarpMask::from_lanes([0]).unwrap(),
        );
        let write = batch(writer.clone(), PhysicalAccessKind::Write);
        before(&state, &writer, OperationEffect::PhysicalAccess(&write)).unwrap();
        after(&state, &writer, OperationEffect::PhysicalAccess(&write)).unwrap();

        let reader = masked_operation(
            0,
            1,
            OperationKind::Load,
            WarpMask::from_lanes([1]).unwrap(),
        );
        let read = batch(reader.clone(), PhysicalAccessKind::Read);
        let error = before(&state, &reader, OperationEffect::PhysicalAccess(&read)).unwrap_err();

        assert!(error.to_string().contains("same-warp lane access"));
        let result = state.result();
        assert_eq!(result.status(), RaceCheckStatus::Error);
        assert_eq!(result.findings().len(), 1);
        assert_eq!(
            result.findings()[0].kind(),
            crate::PhysicalRaceKind::WriteRead
        );
        assert_eq!(result.findings()[0].prior().lane(), 0);
        assert_eq!(result.findings()[0].current().lane(), 1);
    }

    #[test]
    fn same_lane_program_order_and_explicit_lane_barrier_are_ordered() {
        let mut shadow = SameWarpExecution::new();
        let lane_zero = WarpMask::from_lanes([0]).unwrap();
        let lane_one = WarpMask::from_lanes([1]).unwrap();
        let full = WarpMask::FULL;

        let first = masked_operation(0, 0, OperationKind::Store, lane_zero);
        let write = batch(first, PhysicalAccessKind::Write);
        let validation = shadow.validate_batch(&write).unwrap();
        shadow.commit_batch(validation);

        let same_lane = masked_operation(0, 1, OperationKind::Load, lane_zero);
        let read = batch(same_lane, PhysicalAccessKind::Read);
        let validation = shadow.validate_batch(&read).unwrap();
        shadow.commit_batch(validation);

        let release = shadow.lane.barrier_release(0, full).unwrap();
        shadow.lane.barrier_acquire(0, full, Some(&release)).unwrap();
        let other_lane = masked_operation(0, 2, OperationKind::Load, lane_one);
        let read = batch(other_lane, PhysicalAccessKind::Read);
        shadow.validate_batch(&read).unwrap();
    }

    #[test]
    fn atomic_frontier_does_not_hide_an_older_non_atomic_lane_race() {
        let mut shadow = SameWarpExecution::new();
        let lane_zero = WarpMask::from_lanes([0]).unwrap();
        let lane_one = WarpMask::from_lanes([1]).unwrap();

        let write = batch(
            masked_operation(0, 0, OperationKind::Store, lane_zero),
            PhysicalAccessKind::Write,
        );
        let validation = shadow.validate_batch(&write).unwrap();
        shadow.commit_batch(validation);

        let first_atomic = batch(
            masked_operation(0, 1, OperationKind::Atomic, lane_zero),
            PhysicalAccessKind::AtomicReadModifyWrite,
        );
        let validation = shadow.validate_batch(&first_atomic).unwrap();
        shadow.commit_batch(validation);

        let second_atomic = batch(
            masked_operation(0, 2, OperationKind::Atomic, lane_one),
            PhysicalAccessKind::AtomicReadModifyWrite,
        );
        let finding = match shadow.validate_batch(&second_atomic) {
            Err(crate::race_shadow::RaceShadowError::Race(finding)) => finding,
            Err(error) => panic!("unexpected shadow error: {error}"),
            Ok(_) => panic!("an earlier non-atomic write must remain on the conflict frontier"),
        };
        assert_eq!(finding.kind(), crate::PhysicalRaceKind::WriteRead);
        assert_eq!(finding.prior().kind(), PhysicalAccessKind::Write);
        assert_eq!(
            finding.current().kind(),
            PhysicalAccessKind::AtomicReadModifyWrite
        );

        // Two atomic modifiers are mutually allowed, but do not synchronize
        // their lanes. A later plain read still conflicts with the other lane.
        let mut shadow = SameWarpExecution::new();
        for (sequence, mask) in [(0, lane_zero), (1, lane_one)] {
            let atomic = batch(
                masked_operation(0, sequence, OperationKind::Atomic, mask),
                PhysicalAccessKind::AtomicReadModifyWrite,
            );
            let validation = shadow.validate_batch(&atomic).unwrap();
            assert!(shadow.commit_batch(validation).is_empty());
        }
        // Warp-only GC must not discard another lane's unobserved atomic.
        shadow.shadow.gc_dominated_frontier();
        let read = batch(
            masked_operation(0, 2, OperationKind::Load, lane_one),
            PhysicalAccessKind::Read,
        );
        let Err(crate::race_shadow::RaceShadowError::Race(finding)) = shadow.validate_batch(&read)
        else {
            panic!("atomic modification order must not publish lane history")
        };
        assert_eq!(finding.kind(), PhysicalRaceKind::WriteRead);
        assert_eq!(finding.prior().lane(), 0);
        assert_eq!(finding.current().lane(), 1);
    }

    #[test]
    fn same_warp_multi_span_finding_retains_the_first_conflicting_reader() {
        let mut shadow = SameWarpExecution::new();
        for (sequence, lane, allocation, byte_offset) in
            [(0, 3, 7, 20), (1, 1, 7, 20), (2, 0, 8, 0)]
        {
            let read = batch_range(
                masked_operation(
                    0,
                    sequence,
                    OperationKind::Load,
                    WarpMask::from_lanes([lane]).unwrap(),
                ),
                PhysicalAccessKind::Read,
                allocation,
                byte_offset,
                8,
            );
            let validation = shadow.validate_batch(&read).unwrap();
            shadow.commit_batch(validation);
        }

        let operation = masked_operation(
            0,
            3,
            OperationKind::Store,
            WarpMask::from_lanes([2]).unwrap(),
        );
        let descriptor = PhysicalAccessDescriptor::new(
            PhysicalAccessKind::Write,
            PhysicalAccessSpace::Shared,
            24,
        )
        .unwrap();
        let write = PhysicalAccessBatch::resolve(operation, descriptor, |_| {
            Ok::<_, std::convert::Infallible>(vec![
                PhysicalByteSpan::new(PhysicalAllocationId::new(7), 16, 16).unwrap(),
                PhysicalByteSpan::new(PhysicalAllocationId::new(8), 0, 8).unwrap(),
            ])
        })
        .unwrap();
        let finding = match shadow.validate_batch(&write) {
            Err(crate::race_shadow::RaceShadowError::Race(finding)) => finding,
            Err(reason) => {
                panic!("unexpected lane-shadow state error: {reason}")
            }
            Ok(_) => panic!("the first physical span must conflict with earlier lane reads"),
        };
        assert_eq!(finding.kind(), PhysicalRaceKind::ReadWrite);
        // The shared conflict shadow visits retained readers in issue order,
        // rather than the removed lane shadow's lane-sorted witness order.
        assert_eq!(finding.prior().lane(), 3);
        assert_eq!(finding.prior().operation().per_warp_sequence(), 0);
        assert_eq!(finding.current().lane(), 2);
        assert_eq!(finding.overlap().allocation(), PhysicalAllocationId::new(7));
        assert_eq!(finding.overlap().byte_offset(), 20);
        assert_eq!(finding.overlap().byte_end(), 28);
    }

    #[test]
    fn same_warp_lanes_in_one_interval_batch_conflict_transactionally() {
        let mut shadow = SameWarpExecution::new();
        let operation = masked_operation(
            0,
            0,
            OperationKind::Store,
            WarpMask::from_lanes([1, 3]).unwrap(),
        );
        let write = batch_range(operation, PhysicalAccessKind::Write, 7, 0, 8);
        let finding = match shadow.validate_batch(&write) {
            Err(crate::race_shadow::RaceShadowError::Race(finding)) => finding,
            Err(error) => panic!("unexpected shadow error: {error}"),
            Ok(_) => panic!("different lanes in one SIMT instruction must remain unordered"),
        };
        assert_eq!(finding.kind(), PhysicalRaceKind::WriteWrite);
        assert_eq!(finding.prior().lane(), 1);
        assert_eq!(finding.current().lane(), 3);
        assert_eq!(shadow.shadow.tracked_interval_count(), 0);
    }

    #[test]
    fn full_warp_sync_compaction_preserves_partial_lane_history() {
        let mut clocks = crate::race_shadow::RaceShadow::new(2);
        let mut foreign_clocks = crate::race_shadow::RaceShadow::new(2);
        let mut payloads = Vec::new();
        for step in 0..8 {
            clocks
                .proxy_async_fence(step as usize % 2, ProxyAsyncFenceScope::SharedCta)
                .unwrap();
            let mut payload = SharedClockFrontier::single(1, [step + 1; WARP_SIZE]);
            payload.merge_clock(0, &clocks.barrier_release(step as usize % 2).unwrap());
            payload.merge_clock(8, &foreign_clocks.barrier_release(step as usize % 2).unwrap());
            payloads.push(payload);
        }
        let mut compact = SameWarpLaneShadow::new(2);
        let mut reference = compact.clone();
        for step in 0..48 {
            let mask = WarpMask::from_bits([0x5555_5555, 0xaaaa_aaaa, 1, u32::MAX][step % 4]);
            let payload = &payloads[step % payloads.len()];
            compact.barrier_acquire(0, mask, Some(payload)).unwrap();
            reference.barrier_acquire(0, mask, Some(payload)).unwrap();
            let sync_mask = if step % 3 == 0 { WarpMask::FULL } else { mask };
            compact.warp_sync(0, sync_mask).unwrap();
            let release = reference.barrier_release(0, sync_mask).unwrap();
            reference.barrier_acquire(0, sync_mask, Some(&release)).unwrap();
            if sync_mask.is_full() {
                assert!(compact.orders[0].incoming_lanes.is_none());
            }
            // Probe each lane's exported history independently. Compaction must
            // not publish another lane's new history after a partial acquire.
            for lane in 0..WARP_SIZE {
                let probe = WarpMask::from_bits(1 << lane);
                let actual = compact.clone().barrier_release(0, probe).unwrap();
                let expected = reference.clone().barrier_release(0, probe).unwrap();
                assert_eq!(actual, expected, "step {step}, lane {lane}");
            }
        }
    }

    #[test]
    fn shared_frontier_lane_groups_match_independent_acquires_and_releases() {
        let mut clocks = crate::race_shadow::RaceShadow::new(2);
        let mut first = SharedClockFrontier::single(1, [3; WARP_SIZE]);
        first.merge_clock(0, &clocks.barrier_release(0).unwrap());
        let mut second = first.clone();
        second.merge_clock(0, &clocks.barrier_release(1).unwrap());
        // The same lane rows do not imply the same asynchronous/proxy clocks.
        assert!(!first.shares_storage_with(&second));
        let mut foreign = SharedClockFrontier::single(9, [7; WARP_SIZE]);
        foreign.merge_clock(0, &clocks.barrier_release(0).unwrap());

        let mut grouped = SameWarpLaneShadow::new(2);
        let mut reference = WarpLaneOrder::default();
        for (bits, payload) in [
            (0x5555_5555, &first),
            (0xaaaa_aaaa, &second),
            (0x3333_3333, &foreign),
            (0x7777_7777, &second),
            (u32::MAX, &first),
            (0, &second),
        ] {
            let mask = WarpMask::from_bits(bits);
            grouped.barrier_acquire(0, mask, Some(payload)).unwrap();
            if mask.is_full() {
                reference.incoming_common.merge_within(payload, 0, 2);
            } else if !mask.is_empty() {
                let lanes = reference.incoming_lanes.get_or_insert_with(|| {
                    Box::new(std::array::from_fn(|_| SharedClockFrontier::default()))
                });
                for lane in mask {
                    lanes[lane].merge_within(payload, 0, 2);
                }
            }
            reference.acquire(0, mask, payload.release_for(0)).unwrap();
            assert_eq!(grouped.orders[0], reference);

            let actual = grouped.barrier_release(0, mask).unwrap();
            let mut expected = SharedClockFrontier::single(0, reference.release(0, mask).unwrap());
            if !mask.is_empty() {
                expected.merge(&reference.incoming_common);
                if let Some(lanes) = &reference.incoming_lanes {
                    for lane in mask {
                        expected.merge(&lanes[lane]);
                    }
                }
            }
            assert_eq!(actual, expected);
            assert_eq!(grouped.orders[0], reference);
        }
    }

    #[test]
    fn same_warp_interval_shadow_matches_per_byte_reference() {
        fn next_random(seed: &mut u64) -> u64 {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *seed
        }

        let mut interval = SameWarpExecution::new();
        let mut reference = ReferenceSameWarpLaneShadow::new(1);
        let mut seed = 0x5a17_1e5e_1bad_c0de_u64;

        for sequence in 0..2_000_u64 {
            if next_random(&mut seed).is_multiple_of(11) {
                let mut lanes = (0..4)
                    .filter(|lane| next_random(&mut seed) & (1_u64 << lane) != 0)
                    .collect::<Vec<_>>();
                if lanes.is_empty() {
                    lanes.push((next_random(&mut seed) % 4) as usize);
                }
                let mask = WarpMask::from_lanes(lanes).unwrap();
                let actual_payload = interval.lane.barrier_release(0, mask).unwrap();
                let reference_payload = reference.barrier_release(0, mask);
                assert_eq!(actual_payload, reference_payload, "release at {sequence}");
                interval
                    .lane
                    .barrier_acquire(0, mask, Some(&actual_payload))
                    .unwrap();
                reference.barrier_acquire(0, mask, Some(&reference_payload));
            } else {
                let mut lanes = (0..4)
                    .filter(|lane| next_random(&mut seed) & (1_u64 << lane) != 0)
                    .collect::<Vec<_>>();
                if lanes.is_empty() {
                    lanes.push((next_random(&mut seed) % 4) as usize);
                }
                let mask = WarpMask::from_lanes(lanes.iter().copied()).unwrap();
                let kind = match next_random(&mut seed) % 3 {
                    0 => PhysicalAccessKind::Read,
                    1 => PhysicalAccessKind::Write,
                    _ => PhysicalAccessKind::AtomicReadModifyWrite,
                };
                let operation_kind = match kind {
                    PhysicalAccessKind::Read => OperationKind::Load,
                    PhysicalAccessKind::Write => OperationKind::Store,
                    PhysicalAccessKind::AtomicReadModifyWrite => OperationKind::Atomic,
                };
                let space = match next_random(&mut seed) % 3 {
                    0 => PhysicalAccessSpace::Shared,
                    1 => PhysicalAccessSpace::Global,
                    _ => PhysicalAccessSpace::Tmem,
                };
                let mut lane_spans = BTreeMap::new();
                for lane in lanes {
                    let first_len = (next_random(&mut seed) % 8 + 1) as usize;
                    let first_offset = (next_random(&mut seed) % 64) as usize;
                    let mut spans = vec![PhysicalByteSpan::new(
                        PhysicalAllocationId::new(next_random(&mut seed) % 2),
                        first_offset,
                        first_len,
                    )
                    .unwrap()];
                    if next_random(&mut seed).is_multiple_of(3) {
                        let second_len = (next_random(&mut seed) % 8 + 1) as usize;
                        let second_offset = (next_random(&mut seed) % 64) as usize;
                        spans.push(
                            PhysicalByteSpan::new(
                                PhysicalAllocationId::new(2 + next_random(&mut seed) % 2),
                                second_offset,
                                second_len,
                            )
                            .unwrap(),
                        );
                    }
                    lane_spans.insert(lane, spans);
                }
                let operation = masked_operation(0, sequence, operation_kind, mask);
                let batch = PhysicalAccessBatch::resolve_lane_widths(
                    operation,
                    kind,
                    space,
                    |provenance| {
                        Ok::<_, std::convert::Infallible>(
                            lane_spans
                                .get(&provenance.lane())
                                .expect("every active lane has a generated footprint")
                                .clone(),
                        )
                    },
                )
                .unwrap();

                match (
                    reference.validate_batch(&batch),
                    interval.validate_batch(&batch),
                ) {
                    (Ok(reference_validation), Ok(interval_validation)) => {
                        let actual_reviews = interval.commit_batch(interval_validation);
                        assert_eq!(
                            actual_reviews.is_empty(),
                            reference_validation.review_findings.is_empty(),
                            "review presence at {sequence}",
                        );
                        assert!(actual_reviews
                            .iter()
                            .all(PhysicalRaceFinding::requires_unwaited_tmem_load_review));
                        reference.commit_batch(reference_validation);
                    }
                    (
                        Err(reference_findings),
                        Err(crate::race_shadow::RaceShadowError::Race(interval_finding)),
                    ) => {
                        assert!(
                            reference_findings.contains(&interval_finding),
                            "invalid witness at {sequence}: {interval_finding:?}; valid: {reference_findings:?}",
                        );
                    }
                    (_, Err(error))
                        if !matches!(error, crate::race_shadow::RaceShadowError::Race(_)) =>
                    {
                        panic!("unexpected shadow error at {sequence}: {error}")
                    }
                    (Ok(_), Err(crate::race_shadow::RaceShadowError::Race(finding))) => {
                        panic!("interval shadow found an extra race at {sequence}: {finding:?}")
                    }
                    (Err(finding), Ok(_)) => {
                        panic!("interval shadow missed a race at {sequence}: {finding:?}")
                    }
                    (_, Err(error)) => panic!("unexpected shadow error: {error}"),
                }
            }

            // This per-byte oracle owns same-warp order. Compare every original
            // clock field; foreign-lane transport has separate relay controls.
            assert_eq!(interval.lane.orders.len(), reference.orders.len());
            for (actual, expected) in interval.lane.orders.iter().zip(&reference.orders) {
                assert_eq!(
                    (actual.common, &actual.observed, actual.joined),
                    (expected.common, &expected.observed, expected.joined),
                    "orders at {sequence}",
                );
            }
        }
    }

    #[test]
    fn cluster_release_arrive_and_acquire_wait_order_cross_cta_memory() {
        let (state, report) = run_cluster_split_hb(ClusterBarrierArrivalSemantics::Release);
        assert!(report.is_success(), "{:?}", report.error());
        let result = state.result();
        assert_eq!(result.status(), RaceCheckStatus::Clean);
        assert!(result.findings().is_empty());
        assert_eq!(result.accesses().len(), 2);
    }

    #[test]
    fn cluster_relaxed_arrive_does_not_publish_a_release_clock() {
        let (state, report) = run_cluster_split_hb(ClusterBarrierArrivalSemantics::Relaxed);
        assert!(report.error().is_some());
        let result = state.result();
        assert_eq!(result.status(), RaceCheckStatus::Error);
        assert_eq!(result.findings().len(), 1, "{:?}", report.error());
        assert_eq!(
            result.findings()[0].kind(),
            crate::PhysicalRaceKind::WriteRead
        );
    }

    #[test]
    fn exact_async_payload_write_reports_ww_without_unmodeled_payload_fallback() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let physical = PhysicalMemory::new(topology);
        let owner = CtaId::new(topology, 0, 0).unwrap();
        let allocation = physical.shared().allocate_cta_zeroed(owner, 8).unwrap();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Shared {
                allocations: Arc::new(vec![allocation]),
                byte_offset: 0,
                byte_len: 8,
                backing_byte_len: 8,
                virtual_base: 0,
            },
            WarpValue::splat(0_i64),
            8,
        );
        let context = topology.warp_contexts().next().unwrap();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let init = plan_physical_mbarrier_init(&context, &pointer, mask, 1).unwrap();
        let state = Arc::new(RaceCheckLaunchState::new(2));
        let mbarriers = Arc::new(PhysicalBarrierHub::new());
        // Keep this manually driven operation outside the runtime launch's
        // sequence-number range below.
        let init_op = operation(0, 100, OperationKind::Barrier);
        before(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        init.apply(&mbarriers).unwrap();
        after(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after_init_fence(&state, 0, 101, init.barrier_ids()).unwrap();
        run_kernel_engine_launch::<RaceCheckMode, _, _>(
            physical.clone(),
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            |mut warp| async move {
                let sync = warp.begin_operation(
                    warp.context(),
                    StaticOpId::new(2),
                    OperationKind::Collective,
                    [],
                )?;
                warp.named_barrier_sync_with_alignment(Some(&sync), 15, 64, WarpMask::FULL, true)
                    .await?;
                warp.finish_operation(&sync)?;
                Ok(())
            },
        )
        .unwrap();
        let first_op = operation(0, 1, OperationKind::Store);
        let first = batch(first_op.clone(), PhysicalAccessKind::Write);
        before(&state, &first_op, OperationEffect::PhysicalAccess(&first)).unwrap();
        after(&state, &first_op, OperationEffect::PhysicalAccess(&first)).unwrap();

        // The runtime launch above already used sequence 0 on both warps.
        let async_op = operation(1, 100, OperationKind::AsyncIssue);
        let async_write = batch(async_op.clone(), PhysicalAccessKind::Write);
        let barrier_id = init.barrier_ids()[0];
        let deferred = DeferredPayloadHub::new(Arc::clone(&mbarriers));
        let mut payload = AsyncPayloadEffect::new(
            async_op.clone(),
            [async_write],
            PhysicalMbarrierCompletionIssuePlan::single(barrier_id, 4),
        )
        .unwrap();
        before(&state, &async_op, OperationEffect::AsyncPayload(&payload)).unwrap();
        let action_ids = deferred.enqueue_payload(&payload, 0).unwrap();
        let action_id = action_ids[0];
        payload.bind_completion_action_ids(action_ids).unwrap();
        after(&state, &async_op, OperationEffect::AsyncPayload(&payload)).unwrap();
        let MbarrierCompletionAction::DeferredPayload(action) = deferred
            .pending_completion_actions()
            .into_iter()
            .next()
            .unwrap()
        else {
            panic!("payload must expose one logical completion")
        };
        <RaceCheckMode as EngineModeImpl>::before_completion(
            &state,
            CompletionActionEffect::DeferredPayload(&action),
        )
        .unwrap();
        let staged_result = state.result();
        assert_eq!(staged_result.status(), RaceCheckStatus::Incomplete);
        assert!(staged_result.incomplete_reasons().iter().any(|reason| {
            matches!(
                reason,
                RaceCheckIncompleteReason::EffectCommitUnobserved {
                    effect: "async_payload_completion",
                    ..
                }
            )
        }));
        let MbarrierCompletionOutcome::DeferredPayload(outcome) =
            deferred.apply_completion_detailed(action_id).unwrap()
        else {
            panic!("payload must produce one logical completion outcome")
        };
        let error = <RaceCheckMode as EngineModeImpl>::after_completion(
            &state,
            CompletionEffect::DeferredPayload(&outcome),
        )
        .unwrap_err();

        assert!(error.to_string().contains("unordered write/write"));
        let result = state.result();
        assert_eq!(result.status(), RaceCheckStatus::Error);
        assert_eq!(result.findings().len(), 1);
        assert!(result.incomplete_reasons().iter().all(|reason| !matches!(
            reason,
            RaceCheckIncompleteReason::AsyncPayloadAccessUnmodeled { .. }
        )));
    }

    #[test]
    fn missing_after_effect_does_not_commit_shadow_state() {
        let state = RaceCheckLaunchState::new(2);
        let abandoned_op = operation(0, 0, OperationKind::Store);
        let abandoned = batch(abandoned_op.clone(), PhysicalAccessKind::Write);
        before(
            &state,
            &abandoned_op,
            OperationEffect::PhysicalAccess(&abandoned),
        )
        .unwrap();

        let committed_op = operation(1, 0, OperationKind::Store);
        let committed = batch(committed_op.clone(), PhysicalAccessKind::Write);
        before(
            &state,
            &committed_op,
            OperationEffect::PhysicalAccess(&committed),
        )
        .unwrap();
        after(
            &state,
            &committed_op,
            OperationEffect::PhysicalAccess(&committed),
        )
        .unwrap();

        let result = state.result();
        assert_eq!(result.status(), RaceCheckStatus::Incomplete);
        assert!(result.findings().is_empty());
        assert_eq!(result.accesses().len(), 1);
    }

    #[test]
    fn async_payload_issue_is_incomplete_until_race_accesses_are_modeled() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let owner = CtaId::new(topology, 0, 0).unwrap();
        let allocation = physical.shared().allocate_cta_zeroed(owner, 8).unwrap();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Shared {
                allocations: Arc::new(vec![allocation]),
                byte_offset: 0,
                byte_len: 8,
                backing_byte_len: 8,
                virtual_base: 0,
            },
            WarpValue::splat(0_i64),
            8,
        );
        let context = topology.warp_contexts().next().unwrap();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let init = plan_physical_mbarrier_init(&context, &pointer, mask, 1).unwrap();
        let arrive = plan_physical_mbarrier_arrive(&context, &pointer, mask, None, Some(16))
            .unwrap()
            .unwrap();
        let issue = PhysicalMbarrierCompletionIssuePlan::single(arrive.barrier_id(), 16);
        let hub = PhysicalBarrierHub::new();
        let state = RaceCheckLaunchState::new(1);

        let init_op = operation(0, 0, OperationKind::Barrier);
        before(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        init.apply(&hub).unwrap();
        after(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after_init_fence(&state, 0, 1, init.barrier_ids()).unwrap();

        let issue_op = operation(0, 2, OperationKind::AsyncIssue);
        before(
            &state,
            &issue_op,
            OperationEffect::MbarrierCompletionIssue {
                plan: &issue,
                action_ids: None,
            },
        )
        .unwrap();
        let action_ids = issue.apply(&hub).unwrap();
        after(
            &state,
            &issue_op,
            OperationEffect::MbarrierCompletionIssue {
                plan: &issue,
                action_ids: Some(&action_ids),
            },
        )
        .unwrap();
        let ResolvedTransitionSummary::Completion(issue_summary) = state
            .transition_log()
            .completion_summary(action_ids[0].get())
            .unwrap()
        else {
            panic!("committed issue must record its completion resource");
        };
        assert_eq!(issue_summary.resource().generation(), Some(0));

        let arrive_op = operation(0, 3, OperationKind::Barrier);
        before(&state, &arrive_op, staged_arrive(arrive)).unwrap();
        let arrive_outcome = arrive.apply(&hub).unwrap();
        after(&state, &arrive_op, committed_arrive(arrive, arrive_outcome)).unwrap();
        let action = hub.pending_completion_actions()[0];
        <RaceCheckMode as EngineModeImpl>::before_completion(
            &state,
            CompletionActionEffect::PhysicalMbarrier(&action),
        )
        .unwrap();
        let outcome = hub.apply_completion_detailed(action_ids[0]).unwrap();
        <RaceCheckMode as EngineModeImpl>::after_completion(
            &state,
            CompletionEffect::PhysicalMbarrier(&outcome),
        )
        .unwrap();

        let result = state.result();
        assert_eq!(result.sync().status(), SyncCheckStatus::Clean);
        assert_eq!(result.status(), RaceCheckStatus::Incomplete);
        assert!(matches!(
            result.incomplete_reasons(),
            [RaceCheckIncompleteReason::AsyncPayloadAccessUnmodeled { .. }]
        ));
    }

    #[test]
    fn exact_async_payload_commits_all_accesses_and_completion_binding() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let owner = CtaId::new(topology, 0, 0).unwrap();
        let allocation = physical.shared().allocate_cta_zeroed(owner, 8).unwrap();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Shared {
                allocations: Arc::new(vec![allocation]),
                byte_offset: 0,
                byte_len: 8,
                backing_byte_len: 8,
                virtual_base: 0,
            },
            WarpValue::splat(0_i64),
            8,
        );
        let context = topology.warp_contexts().next().unwrap();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let init = plan_physical_mbarrier_init(&context, &pointer, mask, 1).unwrap();
        let arrive = plan_physical_mbarrier_arrive(&context, &pointer, mask, None, Some(16))
            .unwrap()
            .unwrap();
        let hub = Arc::new(PhysicalBarrierHub::new());
        let deferred = DeferredPayloadHub::new(Arc::clone(&hub));
        let global_destination = PhysicalAllocationId::new(71);
        let state = RaceCheckLaunchState::for_topology_and_global_write_allocations(
            topology,
            [global_destination],
        );

        let init_op = operation(0, 0, OperationKind::Barrier);
        before(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        init.apply(&hub).unwrap();
        after(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after_init_fence(&state, 0, 1, init.barrier_ids()).unwrap();

        let issue_op = operation(0, 2, OperationKind::AsyncIssue);
        let source = batch_at(issue_op.clone(), PhysicalAccessKind::Read, 70, 0);
        let destination = global_batch_at(issue_op.clone(), PhysicalAccessKind::Write, 71, 32);
        let mut payload = AsyncPayloadEffect::new(
            issue_op.clone(),
            [source, destination],
            PhysicalMbarrierCompletionIssuePlan::single(arrive.barrier_id(), 16),
        )
        .unwrap();
        before(&state, &issue_op, OperationEffect::AsyncPayload(&payload)).unwrap();
        let action_ids = deferred.enqueue_payload(&payload, 0).unwrap();
        let action_id = action_ids[0];
        payload.bind_completion_action_ids(action_ids).unwrap();
        after(&state, &issue_op, OperationEffect::AsyncPayload(&payload)).unwrap();

        let arrive_op = operation(0, 3, OperationKind::Barrier);
        before(&state, &arrive_op, staged_arrive(arrive)).unwrap();
        let arrive_outcome = arrive.apply(&hub).unwrap();
        after(&state, &arrive_op, committed_arrive(arrive, arrive_outcome)).unwrap();
        let MbarrierCompletionAction::DeferredPayload(action) = deferred
            .pending_completion_actions()
            .into_iter()
            .next()
            .unwrap()
        else {
            panic!("payload must expose one logical completion")
        };
        <RaceCheckMode as EngineModeImpl>::before_completion(
            &state,
            CompletionActionEffect::DeferredPayload(&action),
        )
        .unwrap();
        let destination_epoch = state.race.global_allocation_epochs[&global_destination]
            .tracked_poll_range(0, usize::MAX);
        let destination_progress_before = destination_epoch.progress.snapshot();
        assert_eq!(
            destination_epoch.writes_in_flight.load(Ordering::Acquire),
            1,
        );
        let MbarrierCompletionOutcome::DeferredPayload(outcome) =
            deferred.apply_completion_detailed(action_id).unwrap()
        else {
            panic!("payload must produce one logical completion outcome")
        };
        <RaceCheckMode as EngineModeImpl>::after_completion(
            &state,
            CompletionEffect::DeferredPayload(&outcome),
        )
        .unwrap();
        assert_eq!(
            destination_epoch.writes_in_flight.load(Ordering::Acquire),
            0,
        );
        assert_ne!(
            destination_epoch.progress.snapshot(),
            destination_progress_before,
        );

        let result = state.result();
        assert_eq!(result.status(), RaceCheckStatus::Clean);
        assert_eq!(result.sync().status(), SyncCheckStatus::Clean);
        assert_eq!(result.accesses().len(), 2);
        assert!(result
            .accesses()
            .iter()
            .all(|access| access.operation() == issue_op.id()));
        assert!(result.incomplete_reasons().is_empty());
        assert!(matches!(
            state.transition_log().operation_summary(issue_op.id()),
            Some(ResolvedTransitionSummary::AsyncPayload(_))
        ));
        let ResolvedTransitionSummary::Completion(completion) = state
            .transition_log()
            .completion_summary(action_id.get())
            .unwrap()
        else {
            panic!("composite payload action must stay indexed")
        };
        assert_eq!(completion.resource().generation(), Some(0));
    }

    #[test]
    fn mbarrier_release_acquire_orders_cross_warp_memory() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let physical = PhysicalMemory::new(topology);
        let owner = CtaId::new(topology, 0, 0).unwrap();
        let allocation = physical.shared().allocate_cta_zeroed(owner, 64).unwrap();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Shared {
                allocations: Arc::new(vec![allocation]),
                byte_offset: 0,
                byte_len: 64,
                backing_byte_len: 64,
                virtual_base: 0,
            },
            WarpValue::splat(0_i64),
            8,
        );
        let mut contexts = topology.warp_contexts();
        let context0 = contexts.next().unwrap();
        let context1 = contexts.next().unwrap();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let init = plan_physical_mbarrier_init(&context0, &pointer, mask, 1).unwrap();
        let arrive = plan_physical_mbarrier_arrive(&context0, &pointer, mask, None, None)
            .unwrap()
            .unwrap();
        let wait = plan_physical_mbarrier_wait(&context1, &pointer, mask, 0)
            .unwrap()
            .unwrap();
        let state = RaceCheckLaunchState::new(2);

        let init_op = operation(0, 0, OperationKind::Barrier);
        before(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after_init_fence(&state, 0, 1, init.barrier_ids()).unwrap();

        let write_op = operation(0, 2, OperationKind::Store);
        let write = batch(write_op.clone(), PhysicalAccessKind::Write);
        before(&state, &write_op, OperationEffect::PhysicalAccess(&write)).unwrap();
        after(&state, &write_op, OperationEffect::PhysicalAccess(&write)).unwrap();

        let arrive_op = operation(0, 3, OperationKind::Barrier);
        before(&state, &arrive_op, staged_arrive(arrive)).unwrap();
        after(
            &state,
            &arrive_op,
            committed_arrive(
                arrive,
                crate::PhysicalMbarrierArrivalOutcome::new(0, true).with_pending_arrivals_before(1),
            ),
        )
        .unwrap();

        let wait_op = operation(1, 0, OperationKind::Barrier);
        before(
            &state,
            &wait_op,
            OperationEffect::MbarrierWait {
                plan: wait,
                outcome: None,
            },
        )
        .unwrap();
        after(
            &state,
            &wait_op,
            OperationEffect::MbarrierWait {
                plan: wait,
                outcome: Some(crate::runtime::PhysicalMbarrierWaitOutcome::new(Some(0))),
            },
        )
        .unwrap();

        let read_op = operation(1, 1, OperationKind::Load);
        let read = batch(read_op.clone(), PhysicalAccessKind::Read);
        before(&state, &read_op, OperationEffect::PhysicalAccess(&read)).unwrap();
        after(&state, &read_op, OperationEffect::PhysicalAccess(&read)).unwrap();

        let result = state.result();
        assert_eq!(result.status(), RaceCheckStatus::Clean);
        assert!(result.findings().is_empty());
        assert_eq!(result.accesses().len(), 2);
    }

    #[test]
    fn mma_pipeline_uses_accumulator_dtype_not_destination_address() {
        let lane_zero = WarpMask::from_lanes([0]).unwrap();
        let make_operation = |sequence| {
            OperationContext::new(
                DynamicOpId::new(0, 0, sequence, StaticOpId::new(642 + sequence), []),
                OperationKind::TcgenWork,
                lane_zero,
            )
        };
        let make_issue = |operation: OperationContext, destination_offset: usize, first: bool| {
            let tmem_write = PhysicalAccessBatch::resolve(
                operation.clone(),
                PhysicalAccessDescriptor::new(
                    PhysicalAccessKind::Write,
                    PhysicalAccessSpace::Tmem,
                    4,
                )
                .unwrap(),
                |_| {
                    Ok::<_, std::convert::Infallible>(vec![PhysicalByteSpan::new(
                        PhysicalAllocationId::new(99),
                        destination_offset,
                        4,
                    )
                    .unwrap()])
                },
            )
            .unwrap();
            let mut accesses = vec![tmem_write];
            if first {
                // Force a conflict when the second MMA writes offset 16;
                // physical destination addresses are deliberately not part
                // of the pipeline identity.
                accesses.push(tmem_batch_range(
                    operation.clone(),
                    PhysicalAccessKind::Read,
                    99,
                    16,
                    4,
                ));
            }
            TcgenWorkIssue::new(
                operation,
                1,
                TcgenPipelineOperation::Mma,
                Some(TcgenMmaPipelineClass::new(
                    128,
                    128,
                    16,
                    TcgenAccumulatorDtype::F32,
                )),
                accesses,
            )
            .unwrap()
        };

        let run = |second_destination| {
            let state = RaceCheckLaunchState::new(1);
            for (sequence, destination) in [(0, 0), (1, second_destination)] {
                let operation = make_operation(sequence);
                let issue = make_issue(operation.clone(), destination, sequence == 0);
                before(&state, &operation, OperationEffect::TcgenWorkIssue(&issue))?;
                after(&state, &operation, OperationEffect::TcgenWorkIssue(&issue))?;
            }
            Ok::<_, crate::EngineError>(state.result_before_aborted_execution())
        };

        for second_destination in [0, 16] {
            let result = run(second_destination)
                .expect("same accumulator dtype and shape must be implicitly pipelined");
            assert_eq!(result.status(), RaceCheckStatus::Clean);
            assert!(result.findings().is_empty());
        }
    }

    #[test]
    fn different_mma_shape_is_not_implicitly_ordered() {
        let lane_zero = WarpMask::from_lanes([0]).unwrap();
        for (case, first_shape, second_shape) in [
            (0_u64, [128, 64, 16], [128, 128, 16]),
            (1_u64, [128, 64, 16], [128, 64, 32]),
        ] {
            let state = RaceCheckLaunchState::new(1);
            for (sequence, instruction_shape) in [(0_u64, first_shape), (1_u64, second_shape)] {
                let operation = OperationContext::new(
                    DynamicOpId::new(
                        0,
                        0,
                        sequence,
                        StaticOpId::new(700 + case * 2 + sequence),
                        [],
                    ),
                    OperationKind::TcgenWork,
                    lane_zero,
                );
                let tmem_write = PhysicalAccessBatch::resolve(
                    operation.clone(),
                    PhysicalAccessDescriptor::new(
                        PhysicalAccessKind::Write,
                        PhysicalAccessSpace::Tmem,
                        4,
                    )
                    .unwrap(),
                    |_| {
                        Ok::<_, std::convert::Infallible>(vec![PhysicalByteSpan::new(
                            PhysicalAllocationId::new(99),
                            0,
                            4,
                        )
                        .unwrap()])
                    },
                )
                .unwrap();
                let issue = TcgenWorkIssue::new(
                    operation.clone(),
                    1,
                    TcgenPipelineOperation::Mma,
                    Some(TcgenMmaPipelineClass::new(
                        instruction_shape[0],
                        instruction_shape[1],
                        instruction_shape[2],
                        TcgenAccumulatorDtype::F32,
                    )),
                    [tmem_write],
                )
                .unwrap();
                let issue_result =
                    before(&state, &operation, OperationEffect::TcgenWorkIssue(&issue));
                if sequence == 1 {
                    assert!(issue_result.is_err());
                    break;
                }
                issue_result.unwrap();
                after(&state, &operation, OperationEffect::TcgenWorkIssue(&issue)).unwrap();
            }

            let error = state.result_before_aborted_execution();
            assert_eq!(error.status(), RaceCheckStatus::Error);
            assert_eq!(error.findings().len(), 1);
            assert_eq!(error.findings()[0].kind(), PhysicalRaceKind::WriteWrite);
            assert!(!error.findings()[0].requires_unwaited_tmem_load_review());
        }
    }

    #[test]
    fn tcgen_implicit_pipeline_accepts_exactly_the_ptx_pairings() {
        use TcgenPipelineOperation::{Copy, Copy4x256b, Mma, Shift};

        let operations = [Mma, Copy, Copy4x256b, Shift];
        let lane_zero = WarpMask::from_lanes([0]).unwrap();
        for source in operations {
            for destination in operations {
                let state = RaceCheckLaunchState::new(1);
                let mut second_result = Ok(());
                for (sequence, pipeline_operation) in [(0_u64, source), (1, destination)] {
                    let operation = OperationContext::new(
                        DynamicOpId::new(0, 0, sequence, StaticOpId::new(740 + sequence), []),
                        OperationKind::TcgenWork,
                        lane_zero,
                    );
                    let write =
                        tmem_batch_range(operation.clone(), PhysicalAccessKind::Write, 99, 0, 4);
                    let mma_class = (pipeline_operation == Mma).then_some(
                        TcgenMmaPipelineClass::new(128, 64, 16, TcgenAccumulatorDtype::F32),
                    );
                    let issue = TcgenWorkIssue::new(
                        operation.clone(),
                        1,
                        pipeline_operation,
                        mma_class,
                        [write],
                    )
                    .unwrap();
                    let result =
                        before(&state, &operation, OperationEffect::TcgenWorkIssue(&issue));
                    if sequence == 1 {
                        second_result = result;
                        break;
                    }
                    result.unwrap();
                    after(&state, &operation, OperationEffect::TcgenWorkIssue(&issue)).unwrap();
                }

                let pipelines = matches!(
                    (source, destination),
                    (Mma, Mma)
                        | (Copy, Mma)
                        | (Copy4x256b, Mma)
                        | (Shift, Mma)
                        | (Shift, Copy4x256b)
                        | (Mma, Shift)
                );
                assert_eq!(
                    second_result.is_ok(),
                    pipelines,
                    "unexpected implicit TCGEN pipeline for {source:?} -> {destination:?}",
                );
            }
        }
    }

    #[test]
    fn tcgen_implicit_mma_pipeline_requires_a_matching_accumulator_dtype() {
        // PTX orders MMA into MMA only when the instruction shape *and* the
        // accumulator type match. `kind::f8f6f4` can name either a `.f32` or a
        // `.f16` destination at the same M/N/K, so the dtype is the only field
        // separating these two descriptors: same-dtype pairs pipeline and the
        // overlapping writes are ordered, mixed-dtype pairs do not and the same
        // two writes must be reported as a write-write race.
        let lane_zero = WarpMask::from_lanes([0]).unwrap();
        // `None` means the second MMA was accepted, so the pair pipelined and
        // the overlapping writes are ordered.
        let run = |first: TcgenAccumulatorDtype, second: TcgenAccumulatorDtype| {
            let state = RaceCheckLaunchState::new(1);
            let mut rejected = false;
            for (sequence, accumulator) in [(0_u64, first), (1, second)] {
                let operation = OperationContext::new(
                    DynamicOpId::new(0, 0, sequence, StaticOpId::new(760 + sequence), []),
                    OperationKind::TcgenWork,
                    lane_zero,
                );
                let write =
                    tmem_batch_range(operation.clone(), PhysicalAccessKind::Write, 99, 0, 4);
                let issue = TcgenWorkIssue::new(
                    operation.clone(),
                    1,
                    TcgenPipelineOperation::Mma,
                    Some(TcgenMmaPipelineClass::new(128, 16, 32, accumulator)),
                    [write],
                )
                .unwrap();
                let result = before(&state, &operation, OperationEffect::TcgenWorkIssue(&issue));
                if sequence == 1 {
                    rejected = result.is_err();
                    break;
                }
                result.unwrap();
                after(&state, &operation, OperationEffect::TcgenWorkIssue(&issue)).unwrap();
            }
            rejected.then(|| state.result_before_aborted_execution())
        };

        assert!(run(TcgenAccumulatorDtype::F32, TcgenAccumulatorDtype::F32).is_none());
        assert!(run(TcgenAccumulatorDtype::F16, TcgenAccumulatorDtype::F16).is_none());

        for (first, second) in [
            (TcgenAccumulatorDtype::F32, TcgenAccumulatorDtype::F16),
            (TcgenAccumulatorDtype::F16, TcgenAccumulatorDtype::F32),
        ] {
            let error = run(first, second).expect("mixed accumulator dtypes must not pipeline");
            assert_eq!(error.status(), RaceCheckStatus::Error);
            assert_eq!(error.findings().len(), 1);
            assert_eq!(error.findings()[0].kind(), PhysicalRaceKind::WriteWrite);
        }
    }

    #[test]
    fn tcgen_shared_completion_does_not_implicitly_bridge_to_generic_reuse() {
        let run = |complete: bool, fence_after_completion: bool| {
            let state = RaceCheckLaunchState::new(1);
            let issue_operation = operation(0, 0, OperationKind::TcgenWork);
            let async_read =
                batch_range(issue_operation.clone(), PhysicalAccessKind::Read, 99, 0, 4);
            let issue = TcgenWorkIssue::new(
                issue_operation.clone(),
                1,
                TcgenPipelineOperation::Copy,
                None,
                [async_read],
            )
            .unwrap();
            assert_eq!(
                issue.accesses()[0].descriptor().memory_semantics().proxy(),
                crate::MemoryProxy::Async,
            );
            before(
                &state,
                &issue_operation,
                OperationEffect::TcgenWorkIssue(&issue),
            )
            .unwrap();
            after(
                &state,
                &issue_operation,
                OperationEffect::TcgenWorkIssue(&issue),
            )
            .unwrap();

            let work = TcgenWorkSet::new(
                TcgenWorkKind::Commit,
                Some(1),
                0,
                [(0, vec![issue.token().clone()].into_boxed_slice())],
            );
            if complete {
                let mut race = state.race_for_operation(&issue_operation).unwrap();
                let payload = RaceCheckLaunchState::complete_tcgen_work_set(&mut race, &work)
                    .unwrap()
                    .expect("the completed TCGEN read publishes its source-complete payload");
                race.shadow.barrier_acquire(0, &payload).unwrap();
            }

            if fence_after_completion {
                let fence_operation = operation(0, 1, OperationKind::Fence);
                let fence = ProxyAsyncFenceEffect::new(ProxyAsyncFenceScope::SharedCta, 0, 0, 0);
                before(
                    &state,
                    &fence_operation,
                    OperationEffect::ProxyAsyncFence(fence),
                )
                .unwrap();
                after(
                    &state,
                    &fence_operation,
                    OperationEffect::ProxyAsyncFence(fence),
                )
                .unwrap();
            }

            let generic_write_operation = operation(0, 2, OperationKind::Store);
            let generic_write = batch_range(
                generic_write_operation.clone(),
                PhysicalAccessKind::Write,
                99,
                0,
                4,
            );
            before(
                &state,
                &generic_write_operation,
                OperationEffect::PhysicalAccess(&generic_write),
            )
        };

        assert!(run(false, false).is_err());
        assert!(run(false, true).is_err());
        assert!(run(true, false).is_err());
        assert!(run(true, true).is_ok());
    }

    #[test]
    fn tcgen_shared_issue_honors_standard_proxy_fence_ordering() {
        let run = |with_proxy_fence: bool, with_tcgen_fence: bool| {
            let state = RaceCheckLaunchState::new(1);
            let generic_write_operation = operation(0, 0, OperationKind::Store);
            let generic_write = batch_range(
                generic_write_operation.clone(),
                PhysicalAccessKind::Write,
                99,
                0,
                4,
            );
            commit_access(&state, &generic_write);

            if with_proxy_fence {
                let fence_operation = operation(0, 1, OperationKind::Fence);
                let fence = ProxyAsyncFenceEffect::new(ProxyAsyncFenceScope::SharedCta, 0, 0, 0);
                before(
                    &state,
                    &fence_operation,
                    OperationEffect::ProxyAsyncFence(fence),
                )
                .unwrap();
                after(
                    &state,
                    &fence_operation,
                    OperationEffect::ProxyAsyncFence(fence),
                )
                .unwrap();
            }

            if with_tcgen_fence {
                let fence_operation = operation(0, 2, OperationKind::Fence);
                before(
                    &state,
                    &fence_operation,
                    OperationEffect::TcgenFence(TcgenFenceKind::AfterThreadSync),
                )
                .unwrap();
                after(
                    &state,
                    &fence_operation,
                    OperationEffect::TcgenFence(TcgenFenceKind::AfterThreadSync),
                )
                .unwrap();
            }

            let issue_operation = operation(0, 3, OperationKind::TcgenWork);
            let async_read =
                batch_range(issue_operation.clone(), PhysicalAccessKind::Read, 99, 0, 4);
            let issue = TcgenWorkIssue::new(
                issue_operation.clone(),
                1,
                TcgenPipelineOperation::Copy,
                None,
                [async_read],
            )
            .unwrap();
            before(
                &state,
                &issue_operation,
                OperationEffect::TcgenWorkIssue(&issue),
            )
        };

        assert!(run(false, false).is_err());
        assert!(run(false, true).is_err());
        let proxy_ordered = run(true, false);
        assert!(proxy_ordered.is_ok(), "{proxy_ordered:?}");
        assert!(run(true, true).is_ok());
    }

    #[test]
    fn same_warp_tmem_review_completes_prior_loads_but_retains_store() {
        let state = RaceCheckLaunchState::new(2);
        let issue = |operation: OperationContext,
                     kind: TcgenWorkKind,
                     byte_offset: usize,
                     byte_len: usize| {
            let access = tmem_batch_range(
                operation.clone(),
                if kind == TcgenWorkKind::Load {
                    PhysicalAccessKind::Read
                } else {
                    PhysicalAccessKind::Write
                },
                99,
                byte_offset,
                byte_len,
            );
            let pipeline_operation = match kind {
                TcgenWorkKind::Commit => TcgenPipelineOperation::Copy,
                TcgenWorkKind::MmaSharedARead => panic!("transfer test does not issue MMA A reads"),
                TcgenWorkKind::Load => TcgenPipelineOperation::Load,
                TcgenWorkKind::Store => TcgenPipelineOperation::Store,
            };
            TcgenWorkIssue::new(operation, 1, pipeline_operation, None, [access]).unwrap()
        };

        let first_load_op = operation(0, 0, OperationKind::TcgenWork);
        let first_load = issue(first_load_op.clone(), TcgenWorkKind::Load, 0, 4);
        before(
            &state,
            &first_load_op,
            OperationEffect::TcgenWorkIssue(&first_load),
        )
        .unwrap();
        after(
            &state,
            &first_load_op,
            OperationEffect::TcgenWorkIssue(&first_load),
        )
        .unwrap();

        let second_load_op = operation(0, 1, OperationKind::TcgenWork);
        let second_load = issue(second_load_op.clone(), TcgenWorkKind::Load, 4, 4);
        before(
            &state,
            &second_load_op,
            OperationEffect::TcgenWorkIssue(&second_load),
        )
        .unwrap();
        after(
            &state,
            &second_load_op,
            OperationEffect::TcgenWorkIssue(&second_load),
        )
        .unwrap();

        let store_op = operation(0, 2, OperationKind::TcgenWork);
        let store = issue(store_op.clone(), TcgenWorkKind::Store, 0, 8);
        before(&state, &store_op, OperationEffect::TcgenWorkIssue(&store)).unwrap();
        after(&state, &store_op, OperationEffect::TcgenWorkIssue(&store)).unwrap();

        {
            let race = state.race_for_operation(&store_op).unwrap();
            assert_eq!(race.reviewed_tcgen_load_tokens.len(), 2);
            assert_eq!(race.tcgen_work_tokens.len(), 1);
            assert_eq!(
                race.tcgen_work_tokens
                    .get(store.token())
                    .map(|work| work.kind),
                Some(TcgenWorkKind::Store),
            );
            assert_eq!(race.shadow.active_async_actor_count(), 1);
            assert!(race
                .findings
                .iter()
                .any(PhysicalRaceFinding::requires_unwaited_tmem_load_review));
        }

        // A later load is tracked normally. A real wait may contain both that
        // active token and the earlier review-completed tokens.
        let later_load_op = operation(0, 3, OperationKind::TcgenWork);
        let later_load = issue(later_load_op.clone(), TcgenWorkKind::Load, 16, 4);
        before(
            &state,
            &later_load_op,
            OperationEffect::TcgenWorkIssue(&later_load),
        )
        .unwrap();
        after(
            &state,
            &later_load_op,
            OperationEffect::TcgenWorkIssue(&later_load),
        )
        .unwrap();
        let load_work = TcgenWorkSet::new(
            TcgenWorkKind::Load,
            None,
            0,
            [(
                0,
                vec![
                    first_load.token().clone(),
                    second_load.token().clone(),
                    later_load.token().clone(),
                ]
                .into_boxed_slice(),
            )],
        );
        let wait_op = operation(0, 4, OperationKind::TcgenWork);
        before(
            &state,
            &wait_op,
            OperationEffect::TcgenWait { work: &load_work },
        )
        .unwrap();
        after(
            &state,
            &wait_op,
            OperationEffect::TcgenWait { work: &load_work },
        )
        .unwrap();

        {
            let race = state.race_for_operation(&store_op).unwrap();
            assert!(race.reviewed_tcgen_load_tokens.is_empty());
            assert_eq!(race.tcgen_work_tokens.len(), 1);
            assert!(race.tcgen_work_tokens.contains_key(store.token()));
            assert_eq!(race.shadow.active_async_actor_count(), 1);
        }

        // The triggering store remains a write witness after both the review
        // and the later load wait.
        let other_warp_read_op = operation(1, 0, OperationKind::Load);
        let other_warp_read = tmem_batch_range(
            other_warp_read_op.clone(),
            PhysicalAccessKind::Read,
            99,
            0,
            4,
        );
        assert!(before(
            &state,
            &other_warp_read_op,
            OperationEffect::PhysicalAccess(&other_warp_read),
        )
        .is_err());
        let result = state.result_before_aborted_execution();
        assert_eq!(result.status(), RaceCheckStatus::Error);
        assert!(result.findings().iter().any(|finding| {
            !finding.requires_unwaited_tmem_load_review()
                && finding.kind() == PhysicalRaceKind::WriteRead
                && finding.prior().operation() == store_op.id()
        }));
    }

    #[test]
    fn later_tmem_load_does_not_review_an_earlier_write() {
        let state = RaceCheckLaunchState::new(1);
        let issue = |operation: OperationContext, kind: TcgenWorkKind| {
            let access = tmem_batch_range(
                operation.clone(),
                if kind == TcgenWorkKind::Load {
                    PhysicalAccessKind::Read
                } else {
                    PhysicalAccessKind::Write
                },
                99,
                0,
                4,
            );
            let pipeline_operation = match kind {
                TcgenWorkKind::Commit => TcgenPipelineOperation::Copy,
                TcgenWorkKind::MmaSharedARead => panic!("transfer test does not issue MMA A reads"),
                TcgenWorkKind::Load => TcgenPipelineOperation::Load,
                TcgenWorkKind::Store => TcgenPipelineOperation::Store,
            };
            TcgenWorkIssue::new(operation, 1, pipeline_operation, None, [access]).unwrap()
        };

        let store_op = operation(0, 0, OperationKind::TcgenWork);
        let store = issue(store_op.clone(), TcgenWorkKind::Store);
        before(&state, &store_op, OperationEffect::TcgenWorkIssue(&store)).unwrap();
        after(&state, &store_op, OperationEffect::TcgenWorkIssue(&store)).unwrap();

        let load_op = operation(0, 1, OperationKind::TcgenWork);
        let load = issue(load_op.clone(), TcgenWorkKind::Load);
        assert!(before(&state, &load_op, OperationEffect::TcgenWorkIssue(&load),).is_err());

        let result = state.result_before_aborted_execution();
        assert_eq!(result.status(), RaceCheckStatus::Error);
        assert_eq!(result.findings().len(), 1);
        assert_eq!(result.findings()[0].kind(), PhysicalRaceKind::WriteRead);
        assert!(!result.findings()[0].requires_unwaited_tmem_load_review());
    }

    #[test]
    fn terminal_unconflicted_tcgen_load_needs_no_register_completion_model() {
        let state = RaceCheckLaunchState::new(1);
        let load_op = operation(0, 0, OperationKind::TcgenWork);
        let load = TcgenWorkIssue::new(
            load_op.clone(),
            1,
            TcgenPipelineOperation::Load,
            None,
            [tmem_batch_range(
                load_op.clone(),
                PhysicalAccessKind::Read,
                99,
                0,
                4,
            )],
        )
        .unwrap();
        before(&state, &load_op, OperationEffect::TcgenWorkIssue(&load)).unwrap();
        after(&state, &load_op, OperationEffect::TcgenWorkIssue(&load)).unwrap();

        let result = state.result();
        assert_eq!(result.status(), RaceCheckStatus::Clean);
        assert!(result.incomplete_reasons().is_empty());
    }

    #[test]
    fn later_empty_tcgen_commit_republishes_prior_mma_completion() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let physical = PhysicalMemory::new(topology);
        let owner = CtaId::new(topology, 0, 0).unwrap();
        let allocation = physical.shared().allocate_cta_zeroed(owner, 64).unwrap();
        let buffer = RuntimeBuffer::Shared {
            allocations: Arc::new(vec![allocation]),
            byte_offset: 0,
            byte_len: 64,
            backing_byte_len: 64,
            virtual_base: 0,
        };
        let first_pointer = PhysicalPtr::new(buffer.clone(), WarpValue::splat(0_i64), 8);
        let second_pointer = PhysicalPtr::new(buffer, WarpValue::splat(1_i64), 8);
        let mut contexts = topology.warp_contexts();
        let consumer = contexts.next().unwrap();
        let producer = contexts.next().unwrap();
        let lane_zero = WarpMask::from_lanes([0]).unwrap();
        let state = RaceCheckLaunchState::new(2);
        let hub = PhysicalBarrierHub::new();

        // Both barriers are initialized before the fence, so the fence covers
        // both. Ascending byte offset is already barrier-id order.
        let mut fenced_ids = Vec::new();
        for (sequence, pointer) in [(0_u64, &first_pointer), (1_u64, &second_pointer)] {
            let init = plan_physical_mbarrier_init(&producer, pointer, lane_zero, 1).unwrap();
            let init_op = operation(1, sequence, OperationKind::Barrier);
            before(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
            init.apply(&hub).unwrap();
            after(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
            fenced_ids.extend_from_slice(init.barrier_ids());
        }
        after_init_fence(&state, 1, 2, &fenced_ids).unwrap();

        let mma_op = operation(1, 3, OperationKind::TcgenWork);
        let tmem_write = PhysicalAccessBatch::resolve(
            mma_op.clone(),
            PhysicalAccessDescriptor::new(PhysicalAccessKind::Write, PhysicalAccessSpace::Tmem, 4)
                .unwrap(),
            |_| {
                Ok::<_, std::convert::Infallible>(vec![PhysicalByteSpan::new(
                    PhysicalAllocationId::new(99),
                    0,
                    4,
                )
                .unwrap()])
            },
        )
        .unwrap();
        let issue = TcgenWorkIssue::new(
            mma_op.clone(),
            1,
            TcgenPipelineOperation::Mma,
            Some(TcgenMmaPipelineClass::new(
                128,
                128,
                16,
                TcgenAccumulatorDtype::F32,
            )),
            [tmem_write],
        )
        .unwrap();
        before(&state, &mma_op, OperationEffect::TcgenWorkIssue(&issue)).unwrap();
        after(&state, &mma_op, OperationEffect::TcgenWorkIssue(&issue)).unwrap();

        let first_work = TcgenWorkSet::new(
            TcgenWorkKind::Commit,
            Some(1),
            1,
            [(0, vec![issue.token().clone()].into_boxed_slice())],
        );
        let empty_work = TcgenWorkSet::new(
            TcgenWorkKind::Commit,
            Some(1),
            1,
            [(0, Vec::new().into_boxed_slice())],
        );

        for (sequence, pointer, work) in [
            (4_u64, &first_pointer, &first_work),
            (5_u64, &second_pointer, &empty_work),
        ] {
            let plan = plan_tcgen_commit_issue(&producer, pointer, lane_zero, None)
                .unwrap()
                .unwrap();
            let commit_op = operation(1, sequence, OperationKind::AsyncIssue);
            before(
                &state,
                &commit_op,
                OperationEffect::TcgenCommitIssue {
                    plan: &plan,
                    work,
                    actions: None,
                },
            )
            .unwrap();
            let actions = plan.apply(&hub).unwrap();
            after(
                &state,
                &commit_op,
                OperationEffect::TcgenCommitIssue {
                    plan: &plan,
                    work,
                    actions: Some(&actions),
                },
            )
            .unwrap();
            let action = actions[0];
            <RaceCheckMode as EngineModeImpl>::before_completion(
                &state,
                CompletionActionEffect::PhysicalMbarrier(&action),
            )
            .unwrap();
            let outcome = hub.apply_completion_detailed(action.id()).unwrap();
            <RaceCheckMode as EngineModeImpl>::after_completion(
                &state,
                CompletionEffect::PhysicalMbarrier(&outcome),
            )
            .unwrap();
        }

        let wait = plan_physical_mbarrier_wait(&consumer, &second_pointer, lane_zero, 0)
            .unwrap()
            .unwrap();
        let wait_op = operation(0, 0, OperationKind::Barrier);
        before(
            &state,
            &wait_op,
            OperationEffect::MbarrierWait {
                plan: wait,
                outcome: None,
            },
        )
        .unwrap();
        after(
            &state,
            &wait_op,
            OperationEffect::MbarrierWait {
                plan: wait,
                outcome: Some(PhysicalMbarrierWaitOutcome::new(Some(0))),
            },
        )
        .unwrap();

        let fence_op = operation(0, 1, OperationKind::Fence);
        before(
            &state,
            &fence_op,
            OperationEffect::TcgenFence(TcgenFenceKind::AfterThreadSync),
        )
        .unwrap();
        after(
            &state,
            &fence_op,
            OperationEffect::TcgenFence(TcgenFenceKind::AfterThreadSync),
        )
        .unwrap();

        let read_op = operation(0, 2, OperationKind::TcgenWork);
        let tmem_read = PhysicalAccessBatch::resolve(
            read_op.clone(),
            PhysicalAccessDescriptor::new(PhysicalAccessKind::Read, PhysicalAccessSpace::Tmem, 4)
                .unwrap(),
            |_| {
                Ok::<_, std::convert::Infallible>(vec![PhysicalByteSpan::new(
                    PhysicalAllocationId::new(99),
                    0,
                    4,
                )
                .unwrap()])
            },
        )
        .unwrap();
        let read = TcgenWorkIssue::new(
            read_op.clone(),
            1,
            TcgenPipelineOperation::Load,
            None,
            [tmem_read],
        )
        .unwrap();
        before(&state, &read_op, OperationEffect::TcgenWorkIssue(&read)).unwrap();

        assert!(state
            .result_before_aborted_execution()
            .findings()
            .is_empty());
    }

    #[test]
    fn named_barrier_register_resume_orders_cross_warp_memory() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let physical = PhysicalMemory::new(topology);
        let owner = CtaId::new(topology, 0, 0).unwrap();
        let allocation = physical.shared().allocate_cta_zeroed(owner, 64).unwrap();
        let buffer = RuntimeBuffer::Shared {
            allocations: Arc::new(vec![allocation]),
            byte_offset: 0,
            byte_len: 64,
            backing_byte_len: 64,
            virtual_base: 0,
        };
        let state = Arc::new(RaceCheckLaunchState::new(2));
        let report = run_kernel_engine_launch_report::<RaceCheckMode, _, _>(
            physical,
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let buffer = buffer.clone();
                async move {
                    let warp_id = warp.context().global_warp_id();
                    let lane_mask = WarpMask::from_lanes([0]).unwrap();
                    let indices = WarpValue::splat(0_i64);
                    if warp_id == 0 {
                        let context = warp.context().with_active_mask(lane_mask);
                        let store = warp.begin_operation(
                            context,
                            StaticOpId::new(2000),
                            OperationKind::Store,
                            [],
                        )?;
                        warp.runtime_physical_access(
                            Some(&store),
                            PhysicalAccessDescriptor::new(
                                PhysicalAccessKind::Write,
                                PhysicalAccessSpace::Shared,
                                4,
                            )
                            .unwrap(),
                            &buffer,
                            &indices,
                            lane_mask,
                            || Ok(()),
                        )?;
                        warp.finish_operation(&store)?;
                    }

                    let sync = warp.begin_operation(
                        warp.context(),
                        StaticOpId::new(2001),
                        OperationKind::Collective,
                        [],
                    )?;
                    warp.named_barrier_sync_with_alignment(
                        Some(&sync),
                        6,
                        64,
                        WarpMask::FULL,
                        true,
                    )
                    .await?;
                    warp.finish_operation(&sync)?;

                    if warp_id == 1 {
                        let context = warp.context().with_active_mask(lane_mask);
                        let load = warp.begin_operation(
                            context,
                            StaticOpId::new(2002),
                            OperationKind::Load,
                            [],
                        )?;
                        warp.runtime_physical_access(
                            Some(&load),
                            PhysicalAccessDescriptor::new(
                                PhysicalAccessKind::Read,
                                PhysicalAccessSpace::Shared,
                                4,
                            )
                            .unwrap(),
                            &buffer,
                            &indices,
                            lane_mask,
                            || Ok(()),
                        )?;
                        warp.finish_operation(&load)?;
                    }
                    Ok(())
                }
            },
        );

        assert!(report.is_success(), "{:?}", report.error());
        let result = state.result();
        assert_eq!(result.status(), RaceCheckStatus::Clean);
        assert!(result.findings().is_empty());
        assert_eq!(result.accesses().len(), 2);
    }

    #[test]
    fn named_barrier_arrive_releases_and_sync_resume_acquires_mixed_payload() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let physical = PhysicalMemory::new(topology);
        let owner = CtaId::new(topology, 0, 0).unwrap();
        let allocation = physical.shared().allocate_cta_zeroed(owner, 64).unwrap();
        let buffer = RuntimeBuffer::Shared {
            allocations: Arc::new(vec![allocation]),
            byte_offset: 0,
            byte_len: 64,
            backing_byte_len: 64,
            virtual_base: 0,
        };
        let state = Arc::new(RaceCheckLaunchState::new(2));
        let report = run_kernel_engine_launch_report::<RaceCheckMode, _, _>(
            physical,
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let buffer = buffer.clone();
                async move {
                    let warp_id = warp.context().global_warp_id();
                    let lane_mask = WarpMask::from_lanes([0]).unwrap();
                    let indices = WarpValue::splat(0_i64);
                    if warp_id == 0 {
                        let store = warp.begin_operation(
                            warp.context().with_active_mask(lane_mask),
                            StaticOpId::new(2010),
                            OperationKind::Store,
                            [],
                        )?;
                        warp.runtime_physical_access(
                            Some(&store),
                            PhysicalAccessDescriptor::new(
                                PhysicalAccessKind::Write,
                                PhysicalAccessSpace::Shared,
                                4,
                            )
                            .unwrap(),
                            &buffer,
                            &indices,
                            lane_mask,
                            || Ok(()),
                        )?;
                        warp.finish_operation(&store)?;
                    }

                    let barrier = warp.begin_operation(
                        warp.context(),
                        StaticOpId::new(2011),
                        OperationKind::Collective,
                        [],
                    )?;
                    if warp_id == 0 {
                        warp.named_barrier_arrive(Some(&barrier), 8, 64, WarpMask::FULL)?;
                    } else {
                        warp.named_barrier_sync_with_alignment(
                            Some(&barrier),
                            8,
                            64,
                            WarpMask::FULL,
                            true,
                        )
                        .await?;
                    }
                    warp.finish_operation(&barrier)?;

                    if warp_id == 1 {
                        let load = warp.begin_operation(
                            warp.context().with_active_mask(lane_mask),
                            StaticOpId::new(2012),
                            OperationKind::Load,
                            [],
                        )?;
                        warp.runtime_physical_access(
                            Some(&load),
                            PhysicalAccessDescriptor::new(
                                PhysicalAccessKind::Read,
                                PhysicalAccessSpace::Shared,
                                4,
                            )
                            .unwrap(),
                            &buffer,
                            &indices,
                            lane_mask,
                            || Ok(()),
                        )?;
                        warp.finish_operation(&load)?;
                    }
                    Ok(())
                }
            },
        );

        assert!(report.is_success(), "{:?}", report.error());
        let result = state.result();
        assert_eq!(result.status(), RaceCheckStatus::Clean);
        assert!(result.findings().is_empty());
        assert_eq!(result.accesses().len(), 2);
        assert_eq!(
            result
                .sync()
                .effects()
                .iter()
                .filter(|effect| effect.effect() == SyncCheckEffectKind::NamedBarrierArrive)
                .count(),
            1
        );
    }

    #[test]
    fn second_mbarrier_generation_uses_the_staged_arrive_generation() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let physical = PhysicalMemory::new(topology);
        let owner = CtaId::new(topology, 0, 0).unwrap();
        let allocation = physical.shared().allocate_cta_zeroed(owner, 64).unwrap();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Shared {
                allocations: Arc::new(vec![allocation]),
                byte_offset: 0,
                byte_len: 64,
                backing_byte_len: 64,
                virtual_base: 0,
            },
            WarpValue::splat(0_i64),
            8,
        );
        let mut contexts = topology.warp_contexts();
        let context0 = contexts.next().unwrap();
        let context1 = contexts.next().unwrap();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let init = plan_physical_mbarrier_init(&context0, &pointer, mask, 1).unwrap();
        let arrive0 = plan_physical_mbarrier_arrive(&context0, &pointer, mask, None, None)
            .unwrap()
            .unwrap();
        let wait0 = plan_physical_mbarrier_wait(&context1, &pointer, mask, 0)
            .unwrap()
            .unwrap();
        let arrive1 = plan_physical_mbarrier_arrive(&context1, &pointer, mask, None, None)
            .unwrap()
            .unwrap();
        let wait1 = plan_physical_mbarrier_wait(&context0, &pointer, mask, 1)
            .unwrap()
            .unwrap();
        let state = RaceCheckLaunchState::new(2);

        let init_op = operation(0, 0, OperationKind::Barrier);
        before(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after_init_fence(&state, 0, 1, init.barrier_ids()).unwrap();

        let first_write_op = operation(0, 2, OperationKind::Store);
        let first_write = batch(first_write_op.clone(), PhysicalAccessKind::Write);
        before(
            &state,
            &first_write_op,
            OperationEffect::PhysicalAccess(&first_write),
        )
        .unwrap();
        after(
            &state,
            &first_write_op,
            OperationEffect::PhysicalAccess(&first_write),
        )
        .unwrap();

        let arrive0_op = operation(0, 3, OperationKind::Barrier);
        before(&state, &arrive0_op, staged_arrive(arrive0)).unwrap();
        after(
            &state,
            &arrive0_op,
            committed_arrive(
                arrive0,
                crate::PhysicalMbarrierArrivalOutcome::new(0, true).with_pending_arrivals_before(1),
            ),
        )
        .unwrap();

        let wait0_op = operation(1, 0, OperationKind::Barrier);
        before(
            &state,
            &wait0_op,
            OperationEffect::MbarrierWait {
                plan: wait0,
                outcome: None,
            },
        )
        .unwrap();
        after(
            &state,
            &wait0_op,
            OperationEffect::MbarrierWait {
                plan: wait0,
                outcome: Some(crate::runtime::PhysicalMbarrierWaitOutcome::new(Some(0))),
            },
        )
        .unwrap();

        let second_write_op = operation(1, 1, OperationKind::Store);
        let second_write = batch(second_write_op.clone(), PhysicalAccessKind::Write);
        before(
            &state,
            &second_write_op,
            OperationEffect::PhysicalAccess(&second_write),
        )
        .unwrap();
        after(
            &state,
            &second_write_op,
            OperationEffect::PhysicalAccess(&second_write),
        )
        .unwrap();

        let arrive1_op = operation(1, 2, OperationKind::Barrier);
        before(&state, &arrive1_op, staged_arrive(arrive1)).unwrap();
        assert_eq!(
            state
                .sync
                .staged_mbarrier_arrive_generation(arrive1_op.id()),
            Some(1)
        );
        after(
            &state,
            &arrive1_op,
            committed_arrive(
                arrive1,
                crate::PhysicalMbarrierArrivalOutcome::new(1, true).with_pending_arrivals_before(1),
            ),
        )
        .unwrap();

        let wait1_op = operation(0, 4, OperationKind::Barrier);
        before(
            &state,
            &wait1_op,
            OperationEffect::MbarrierWait {
                plan: wait1,
                outcome: None,
            },
        )
        .unwrap();
        after(
            &state,
            &wait1_op,
            OperationEffect::MbarrierWait {
                plan: wait1,
                outcome: Some(crate::runtime::PhysicalMbarrierWaitOutcome::new(Some(1))),
            },
        )
        .unwrap();

        let final_read_op = operation(0, 5, OperationKind::Load);
        let final_read = batch(final_read_op.clone(), PhysicalAccessKind::Read);
        before(
            &state,
            &final_read_op,
            OperationEffect::PhysicalAccess(&final_read),
        )
        .unwrap();
        after(
            &state,
            &final_read_op,
            OperationEffect::PhysicalAccess(&final_read),
        )
        .unwrap();

        let result = state.result();
        assert_eq!(result.status(), RaceCheckStatus::Clean);
        assert!(result.findings().is_empty());
        assert!(result.incomplete_reasons().is_empty());
        assert_eq!(result.accesses().len(), 3);
    }

    #[test]
    fn direct_launch_rejects_race_before_second_numeric_store() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let physical = PhysicalMemory::new(topology);
        let owner = CtaId::new(topology, 0, 0).unwrap();
        let allocation = physical.shared().allocate_cta_zeroed(owner, 4).unwrap();
        let buffer = RuntimeBuffer::Shared {
            allocations: Arc::new(vec![allocation]),
            byte_offset: 0,
            byte_len: 4,
            backing_byte_len: 4,
            virtual_base: 0,
        };
        let state = Arc::new(RaceCheckLaunchState::new(topology.warp_count()));
        let numeric_stores = Arc::new(AtomicUsize::new(0));
        let numeric_stores_for_launch = Arc::clone(&numeric_stores);
        let error = run_kernel_engine_launch::<RaceCheckMode, _, _>(
            physical,
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let buffer = buffer.clone();
                let numeric_stores = Arc::clone(&numeric_stores_for_launch);
                async move {
                    let context = warp
                        .context()
                        .with_active_mask(WarpMask::from_lanes([0]).expect("lane zero is valid"));
                    let operation = warp.begin_operation(
                        context,
                        StaticOpId::new(200),
                        OperationKind::Store,
                        [],
                    )?;
                    let descriptor = PhysicalAccessDescriptor::new(
                        PhysicalAccessKind::Write,
                        PhysicalAccessSpace::Shared,
                        4,
                    )
                    .expect("four-byte store is valid");
                    warp.runtime_physical_access(
                        Some(&operation),
                        descriptor,
                        &buffer,
                        &WarpValue::splat(0_i64),
                        context.active_mask(),
                        || {
                            numeric_stores.fetch_add(1, Ordering::SeqCst);
                            Ok(())
                        },
                    )?;
                    warp.finish_operation(&operation)?;
                    Ok(())
                }
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("unordered write/write"));
        assert_eq!(numeric_stores.load(Ordering::SeqCst), 1);
        let result = state.result();
        assert_eq!(result.status(), RaceCheckStatus::Error);
        assert_eq!(result.findings().len(), 1);
    }

    #[test]
    fn full_async_group_wait_retires_token_state_across_many_groups() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let lane_zero = WarpMask::from_lanes([0]).unwrap();
        let hub = AsyncGroupHub::new(topology);
        let state = RaceCheckLaunchState::for_topology(topology);

        for ordinal in 0..10_000_u64 {
            let sequence = ordinal * 3;
            let issue_operation = operation(0, sequence, OperationKind::AsyncIssue);
            let issue =
                AsyncGroupIssueEffect::new(issue_operation.clone(), AsyncGroupDomain::Bulk, [], [])
                    .unwrap();
            before(
                &state,
                &issue_operation,
                OperationEffect::AsyncGroupIssue(&issue),
            )
            .unwrap();
            hub.issue_exact(&context, issue.clone(), Vec::new())
                .unwrap();
            after(
                &state,
                &issue_operation,
                OperationEffect::AsyncGroupIssue(&issue),
            )
            .unwrap();

            let commit_operation = operation(0, sequence + 1, OperationKind::Barrier);
            let commit_plan = hub
                .commit_plan(&context, AsyncGroupDomain::Bulk, lane_zero)
                .unwrap();
            before(
                &state,
                &commit_operation,
                OperationEffect::AsyncGroupCommit {
                    plan: &commit_plan,
                    outcome: None,
                },
            )
            .unwrap();
            let commit_outcome = hub.commit_detailed(commit_plan.clone()).unwrap();
            after(
                &state,
                &commit_operation,
                OperationEffect::AsyncGroupCommit {
                    plan: &commit_plan,
                    outcome: Some(&commit_outcome),
                },
            )
            .unwrap();
            let group = &commit_outcome.groups()[0];
            for action in [group.source_read_action(), group.full_action()] {
                <RaceCheckMode as EngineModeImpl>::before_completion(
                    &state,
                    CompletionActionEffect::AsyncGroup(&action),
                )
                .unwrap();
                let outcome = hub
                    .apply_completion_action_detailed_with_outcome(&action, |_| Ok(()))
                    .unwrap();
                <RaceCheckMode as EngineModeImpl>::after_completion(
                    &state,
                    CompletionEffect::AsyncGroup(&outcome),
                )
                .unwrap();
            }

            let wait_operation = operation(0, sequence + 2, OperationKind::Barrier);
            let wait_plan = hub
                .wait_plan(&context, AsyncGroupDomain::Bulk, lane_zero, 0, false)
                .unwrap();
            before(
                &state,
                &wait_operation,
                OperationEffect::AsyncGroupWait {
                    plan: &wait_plan,
                    outcome: None,
                },
            )
            .unwrap();
            let wait_outcome = hub.complete_wait(wait_plan.clone()).unwrap();
            after(
                &state,
                &wait_operation,
                OperationEffect::AsyncGroupWait {
                    plan: &wait_plan,
                    outcome: Some(&wait_outcome),
                },
            )
            .unwrap();

            assert_eq!(state.live_async_group_token_count(), 0);
            state.transition_log().begin_replay();
        }
    }
}
