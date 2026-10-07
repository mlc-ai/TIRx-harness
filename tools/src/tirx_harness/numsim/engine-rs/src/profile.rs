#[cfg(feature = "profile")]
use std::cell::RefCell;
#[cfg(feature = "profile")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "profile")]
use std::time::Instant;

#[cfg(feature = "profile")]
const KIND_COUNT: usize = 167;

#[derive(Clone, Copy)]
pub enum ProfileKind {
    WorkerTotal = 0,
    FuturePoll = 1,
    CompletionPump = 2,
    GmemRead = 3,
    GmemWrite = 4,
    GmemAtomic = 5,
    PrivateRead = 6,
    PrivateWrite = 7,
    PrivateAtomic = 8,
    NumpyTotal = 9,
    NumpyInput = 10,
    NumpyMatmul = 11,
    NumpyOutput = 12,
    SyncBeforeEffect = 13,
    SyncAfterEffect = 14,
    SyncTransitionEffect = 15,
    SyncTransitionClock = 16,
    SyncCompletionLookup = 17,
    SyncEffectMemory = 19,
    SyncEffectAsyncPayload = 20,
    SyncEffectAsyncGroup = 21,
    SyncEffectMbarrier = 22,
    SyncEffectTcgen = 23,
    SyncEffectNamedCluster = 24,
    SyncEffectLifecycleSetmax = 25,
    SyncEffectOther = 26,
    OperationBegin = 27,
    MmaIncreasingK = 28,
    CompletionDeferredDiscovery = 29,
    CompletionDeferredBefore = 30,
    CompletionDeferredApply = 31,
    SyncPrepareCompletion = 32,
    SyncAfterCompletion = 33,
    CompletionAsyncDiscovery = 34,
    CompletionAsyncApply = 35,
    AsyncDomainApply = 36,
    AsyncReadyWaiters = 37,
    AsyncPublishOutcome = 38,
    AsyncGlobalPublish = 39,
    AsyncEnabledValidation = 40,
    AsyncActionMetadata = 41,
    AsyncPublishWriteCount = 42,
    AsyncPublishByteCount = 43,
    AsyncPublishResolve = 44,
    AsyncPublishPlan = 45,
    AsyncPublishLock = 46,
    AsyncPublishStore = 47,
    AsyncPublishNotify = 48,
    PrivateWriteResolvedBatch = 49,
    PrivateWriteScalar = 50,
    PrivateWriteBatch = 51,
    PrivateWritePublishBatch = 52,
    PrivateWriteRows = 53,
    SyncBeforeNamed = 54,
    SyncAfterNamed = 55,
    SyncBeforeCluster = 56,
    SyncAfterCluster = 57,
    SyncBeforeStrict = 58,
    SyncAfterStrict = 59,
    SyncRecordEffect = 60,
    SyncApplyCausality = 61,
    RaceCompactApply = 62,
    RaceLaneValidate = 63,
    RaceShadowApply = 64,
    RaceAliasObserve = 65,
    RaceCommitPending = 66,
    RaceCompactGeometry = 67,
    RaceOperationRegister = 68,
    RaceShadowSegments = 69,
    RaceDirectGeometry = 70,
    RaceSparseGeometry = 71,
    RaceDuplicateGeometry = 72,
    RaceShadowRead = 73,
    RaceShadowWrite = 74,
    RaceShadowTmem = 75,
    RaceAliasRead = 76,
    RaceAliasWrite = 77,
    RaceAliasShared = 78,
    RaceAliasTmem = 79,
    RaceAliasCommonAllocation = 80,
    RaceAliasNoLogicalBuffer = 81,
    RaceAliasUntrackedSpace = 82,
    RaceAliasUniformRead = 83,
    RaceSparseShadow = 84,
    RaceSparseValidate = 85,
    RaceSparseCommit = 86,
    RaceSparsePatchApply = 87,
    RaceSparsePatchBaseExact = 88,
    RaceSparsePatchGeneral = 89,
    RaceCommitClock = 90,
    RaceCommitAllocation = 91,
    RaceCommitRetire = 92,
    RaceCommitSafePoint = 93,
    TcgenResolveAccesses = 94,
    TcgenBeforeEffect = 95,
    TcgenNumericEffect = 96,
    TcgenAfterEffect = 97,
    RaceCommitOrderedExact = 98,
    RaceCommitOrderedSingle = 99,
    RaceCommitOrderedContiguous = 100,
    RaceCommitOrderedGroups = 101,
    RaceCommitIndexedReplace = 102,
    RaceCommitGeneralReplace = 103,
    GmemReadReadonly = 104,
    GmemReadOneStripe = 105,
    GmemReadUniform = 106,
    GmemReadContiguous = 107,
    GmemReadGeneral = 108,
    RaceSparseFirstOverlap = 109,
    RaceSparseAllocationMismatch = 110,
    RaceSparsePartialOverlap = 111,
    RaceSparseRead = 112,
    RaceSparseWrite = 113,
    RaceSparseGlobal = 114,
    RaceSparseShared = 115,
    RaceSparseTmem = 116,
    RaceDirectSegmentExact = 117,
    RaceDirectSegmentEmpty = 118,
    RaceDirectSegmentNonexact = 119,
    RacePriorAtomic = 120,
    RacePriorSameWarp = 121,
    RacePriorSameOperation = 122,
    RacePriorCrossWarp = 123,
    RaceAccessFrontierEmpty = 124,
    RaceAccessFrontierOneSame = 125,
    RaceAccessFrontierOneReplace = 126,
    RaceAccessFrontierOneMany = 127,
    RaceAccessFrontierMany = 128,
    RaceGeometryMonotonic = 129,
    RaceGeometryDuplicate = 130,
    RacePriorSameLane = 131,
    RacePriorOtherLane = 132,
    RaceAliasBatchOwnedHit = 133,
    RaceAliasBatchOwnedMiss = 134,
    RaceClockedCommonAllocation = 135,
    RaceClockedMixedAllocation = 136,
    RaceGlobalTransactionSharedWait = 137,
    RaceGlobalTransactionExclusiveWait = 138,
    RaceGlobalPlainWriteLock = 139,
    RaceGlobalPlainWriteApply = 140,
    RaceGlobalDenseActors = 141,
    RaceGlobalReadCacheHit = 142,
    RaceGlobalReadCacheMiss = 143,
    RaceGlobalReadReprocess = 144,
    RaceGlobalReadApply = 145,
    RaceGlobalClockMerge = 146,
    RaceGlobalFence = 147,
    RaceGlobalWarpSync = 148,
    RaceGlobalPhysicalRelease = 149,
    RaceGlobalPhysicalAcquire = 150,
    RaceGlobalNamedCluster = 151,
    RaceGlobalAsyncIssue = 152,
    RaceGlobalAsyncComplete = 153,
    RaceGlobalAsyncAcquire = 154,
    RaceGlobalAsyncPublish = 155,
    RaceGlobalProxyFence = 159,
    RaceGlobalReadLock = 160,
    AtomicReservationSync = 161,
    RaceGlobalTransactionPhysical = 162,
    RaceGlobalTransactionFence = 163,
    RaceGlobalTransactionAsyncIssue = 164,
    RaceGlobalTransactionCompletion = 165,
    RaceGlobalTransactionReadReprocess = 166,
}

#[cfg(feature = "profile")]
const NAMES: [&str; KIND_COUNT] = [
    "worker_total",
    "future_poll",
    "completion_pump",
    "gmem_read",
    "gmem_write",
    "gmem_atomic",
    "private_read",
    "private_write",
    "private_atomic",
    "numpy_total",
    "numpy_input",
    "numpy_matmul",
    "numpy_output",
    "sync_before_effect",
    "sync_after_effect",
    "sync_transition_effect",
    "sync_transition_clock",
    "sync_completion_lookup",
    // Slot 18: retired with the synccheck `after_operation` body. The table
    // is indexed by discriminant, so the slot stays to keep 19.. aligned.
    "retired_sync_after_operation",
    "sync_effect_memory",
    "sync_effect_async_payload",
    "sync_effect_async_group",
    "sync_effect_mbarrier",
    "sync_effect_tcgen",
    "sync_effect_named_cluster",
    "sync_effect_lifecycle_setmax",
    "sync_effect_other",
    "operation_begin",
    "mma_increasing_k",
    "completion_deferred_discovery",
    "completion_deferred_before",
    "completion_deferred_apply",
    "sync_prepare_completion",
    "sync_after_completion",
    "completion_async_discovery",
    "completion_async_apply",
    "async_domain_apply",
    "async_ready_waiters",
    "async_publish_outcome",
    "async_global_publish",
    "async_enabled_validation",
    "async_action_metadata",
    "async_publish_write_count",
    "async_publish_byte_count",
    "async_publish_resolve",
    "async_publish_plan",
    "async_publish_lock",
    "async_publish_store",
    "async_publish_notify",
    "private_write_resolved_batch",
    "private_write_scalar",
    "private_write_batch",
    "private_write_publish_batch",
    "private_write_rows",
    "sync_before_named",
    "sync_after_named",
    "sync_before_cluster",
    "sync_after_cluster",
    "sync_before_strict",
    "sync_after_strict",
    "sync_record_effect",
    "sync_apply_causality",
    "race_compact_apply",
    "race_lane_validate",
    "race_shadow_apply",
    "race_alias_observe",
    "race_commit_pending",
    "race_compact_geometry",
    "race_operation_register",
    "race_shadow_segments",
    "race_direct_geometry",
    "race_sparse_geometry",
    "race_duplicate_geometry",
    "race_shadow_read",
    "race_shadow_write",
    "race_shadow_tmem",
    "race_alias_read",
    "race_alias_write",
    "race_alias_shared",
    "race_alias_tmem",
    "race_alias_common_allocation",
    "race_alias_no_logical_buffer",
    "race_alias_untracked_space",
    "race_alias_uniform_read",
    "race_sparse_shadow",
    "race_sparse_validate",
    "race_sparse_commit",
    "race_sparse_patch_apply",
    "race_sparse_patch_base_exact",
    "race_sparse_patch_general",
    "race_commit_clock",
    "race_commit_allocation",
    "race_commit_retire",
    "race_commit_safe_point",
    "tcgen_resolve_accesses",
    "tcgen_before_effect",
    "tcgen_numeric_effect",
    "tcgen_after_effect",
    "race_commit_ordered_exact",
    "race_commit_ordered_single",
    "race_commit_ordered_contiguous",
    "race_commit_ordered_groups",
    "race_commit_indexed_replace",
    "race_commit_general_replace",
    "gmem_read_readonly",
    "gmem_read_one_stripe",
    "gmem_read_uniform",
    "gmem_read_contiguous",
    "gmem_read_general",
    "race_sparse_first_overlap",
    "race_sparse_allocation_mismatch",
    "race_sparse_partial_overlap",
    "race_sparse_read",
    "race_sparse_write",
    "race_sparse_global",
    "race_sparse_shared",
    "race_sparse_tmem",
    "race_direct_segment_exact",
    "race_direct_segment_empty",
    "race_direct_segment_nonexact",
    "race_prior_atomic",
    "race_prior_same_warp",
    "race_prior_same_operation",
    "race_prior_cross_warp",
    "race_access_frontier_empty",
    "race_access_frontier_one_same",
    "race_access_frontier_one_replace",
    "race_access_frontier_one_many",
    "race_access_frontier_many",
    "race_geometry_monotonic",
    "race_geometry_duplicate",
    "race_prior_same_lane",
    "race_prior_other_lane",
    "race_alias_batch_owned_hit",
    "race_alias_batch_owned_miss",
    "race_clocked_common_allocation",
    "race_clocked_mixed_allocation",
    "race_global_transaction_shared_wait",
    "race_global_transaction_exclusive_wait",
    "race_global_plain_write_lock",
    "race_global_plain_write_apply",
    "race_global_dense_actors",
    "race_global_read_cache_hit",
    "race_global_read_cache_miss",
    "race_global_read_reprocess",
    "race_global_read_apply",
    "race_global_clock_merge",
    "race_global_fence",
    "race_global_warp_sync",
    "race_global_physical_release",
    "race_global_physical_acquire",
    "race_global_named_cluster",
    "race_global_async_issue",
    "race_global_async_complete",
    "race_global_async_acquire",
    "race_global_async_publish",
    // Slots 156..=158 belonged to an unreachable prevalidated-lane path.
    "retired_race_direct_segment_covered",
    "retired_race_direct_ordering_cache_hit",
    "retired_race_direct_ordering_cache_miss",
    "race_global_proxy_fence",
    "race_global_read_lock",
    "atomic_reservation_sync",
    "race_global_transaction_physical",
    "race_global_transaction_fence",
    "race_global_transaction_async_issue",
    "race_global_transaction_completion",
    "race_global_transaction_read_reprocess",
];

#[cfg(feature = "profile")]
#[derive(Clone, Copy, Default)]
struct Entry {
    count: u64,
    nanos: u64,
}

#[cfg(feature = "profile")]
struct LocalProfile {
    entries: [Entry; KIND_COUNT],
}

#[cfg(feature = "profile")]
impl Default for LocalProfile {
    fn default() -> Self {
        Self {
            entries: [Entry::default(); KIND_COUNT],
        }
    }
}

#[cfg(feature = "profile")]
impl LocalProfile {
    fn flush(&mut self) {
        for (index, entry) in self.entries.iter_mut().enumerate() {
            GLOBAL_COUNTS[index].fetch_add(entry.count, Ordering::Relaxed);
            GLOBAL_NANOS[index].fetch_add(entry.nanos, Ordering::Relaxed);
            *entry = Entry::default();
        }
    }
}

#[cfg(feature = "profile")]
impl Drop for LocalProfile {
    fn drop(&mut self) {
        self.flush();
    }
}

#[cfg(feature = "profile")]
static GLOBAL_COUNTS: [AtomicU64; KIND_COUNT] = [const { AtomicU64::new(0) }; KIND_COUNT];
#[cfg(feature = "profile")]
static GLOBAL_NANOS: [AtomicU64; KIND_COUNT] = [const { AtomicU64::new(0) }; KIND_COUNT];

#[cfg(feature = "profile")]
thread_local! {
    static LOCAL_PROFILE: RefCell<LocalProfile> = RefCell::new(LocalProfile::default());
}

pub struct ProfileTimer {
    #[cfg(feature = "profile")]
    kind: ProfileKind,
    #[cfg(feature = "profile")]
    start: Instant,
}

impl ProfileTimer {
    pub fn new(kind: ProfileKind) -> Self {
        #[cfg(feature = "profile")]
        {
            Self {
                kind,
                start: Instant::now(),
            }
        }
        #[cfg(not(feature = "profile"))]
        {
            let _ = kind;
            Self {}
        }
    }
}

impl Drop for ProfileTimer {
    fn drop(&mut self) {
        #[cfg(feature = "profile")]
        {
            let nanos = self.start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
            LOCAL_PROFILE.with(|profile| {
                let mut profile = profile.borrow_mut();
                let entry = &mut profile.entries[self.kind as usize];
                entry.count = entry.count.saturating_add(1);
                entry.nanos = entry.nanos.saturating_add(nanos);
            });
        }
    }
}

pub fn profile_count(kind: ProfileKind) {
    profile_count_by(kind, 1);
}

pub fn profile_count_by(kind: ProfileKind, count: u64) {
    #[cfg(feature = "profile")]
    {
        LOCAL_PROFILE.with(|profile| {
            let mut profile = profile.borrow_mut();
            let entry = &mut profile.entries[kind as usize];
            entry.count = entry.count.saturating_add(count);
        });
    }
    #[cfg(not(feature = "profile"))]
    {
        let _ = (kind, count);
    }
}

pub fn profile_reset() {
    #[cfg(feature = "profile")]
    {
        LOCAL_PROFILE.with(|profile| *profile.borrow_mut() = LocalProfile::default());
        for value in &GLOBAL_COUNTS {
            value.store(0, Ordering::Relaxed);
        }
        for value in &GLOBAL_NANOS {
            value.store(0, Ordering::Relaxed);
        }
    }
}

pub fn profile_snapshot() -> Vec<(&'static str, u64, u64)> {
    #[cfg(feature = "profile")]
    {
        LOCAL_PROFILE.with(|profile| profile.borrow_mut().flush());
        NAMES
            .iter()
            .enumerate()
            .map(|(index, name)| {
                (
                    *name,
                    GLOBAL_COUNTS[index].load(Ordering::Relaxed),
                    GLOBAL_NANOS[index].load(Ordering::Relaxed),
                )
            })
            .collect()
    }
    #[cfg(not(feature = "profile"))]
    {
        Vec::new()
    }
}
