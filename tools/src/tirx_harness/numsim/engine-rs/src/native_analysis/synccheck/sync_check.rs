//! Synccheck-owned native execution and protocol analysis.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// Effect payloads, imported from the effect vocabulary rather than from the
// engine internals that build them.
use crate::effect::{
    ClusterBarrierArrivePlan, ClusterBarrierRegistrationOutcome, ClusterBarrierWaitPlan,
    ClusterBarrierWaitResumePlan, CpAsyncMbarrierArrivePlan, NamedBarrierArrivePlan,
    NamedBarrierSyncPlan, NamedBarrierSyncRegistrationOutcome, NamedBarrierSyncResumePlan,
    PhysicalMbarrierArrivalBatchOutcome, PhysicalMbarrierArriveBatchPlan,
    PhysicalMbarrierArrivePlan, PhysicalMbarrierCompletionIssuePlan,
    PhysicalMbarrierExpectTxOutcome, PhysicalMbarrierExpectTxPlan, PhysicalMbarrierInitPlan,
    PhysicalMbarrierWaitPlan, TcgenCommitIssuePlan,
};
use crate::MbarrierCompletionCausalToken;
use crate::{
    profile_count, retire_named_barrier_generations, AnalysisGapEffect,
    AnalysisGapKind, CheckerLaunchContext, ClusterBarrierArrivalOutcome, ClusterBarrierId,
    ClusterBarrierParticipantExitEvidence, CompletionActionEffect, CompletionEffect, DynamicOpId,
    EngineError, ExecutionReport, LaunchTopology, NamedBarrierArrivalOutcome, OperationContext,
    OperationEffect, OperationKind, PhysicalAccessDescriptor, PhysicalAccessKind,
    PhysicalAccessSpace, PhysicalAllocationId, PhysicalBarrierId, PhysicalCompletionAction,
    PhysicalCompletionActionId, PhysicalCompletionKind, PhysicalCompletionOutcome,
    PhysicalMbarrierArrivalOutcome, ProfileKind, ProfileTimer, ResolvedCompletionEffect,
    ResolvedMemoryEffect, ResolvedSyncResource, ResolvedSynchronizationEffect,
    ResolvedTransitionLog, ResolvedTransitionRegistration, ResolvedTransitionSummary,
    StrictClusterBarrierError, StrictClusterBarrierOutcome, StrictClusterBarrierProtocol,
    StrictMbarrierCompletionToken, StrictMbarrierEffect, StrictMbarrierError,
    StrictMbarrierProtocol, StrictMbarrierSnapshot, StrictMbarrierWaitOutcome,
    StrictNamedBarrierError, StrictNamedBarrierOperation, StrictNamedBarrierOutcome,
    StrictNamedBarrierProtocol, StrictNamedBarrierSnapshot, SyncCausalityError,
    SyncCausalityTracker, SyncClockPayload, SyncVectorClock, TcgenLifecycleAction, WarpMask,
};

/// Top-level verdict for the currently committed native sync-check state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncCheckStatus {
    Clean,
    Incomplete,
    Error,
}

/// Checker-visible kind of a fully resolved synchronization effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum SyncCheckEffectKind {
    MbarrierInvalidate,
    MbarrierInit,
    MbarrierArrive,
    MbarrierWait,
    MbarrierCompletionIssue,
    CpAsyncMbarrierArrive,
    TcgenCommitIssue,
    MbarrierCompletion,
    NamedBarrierArrive,
    NamedBarrierSyncRegister,
    NamedBarrierSyncResume,
    ClusterBarrierArrive,
    ClusterBarrierWaitRegister,
    ClusterBarrierWaitResume,
    TcgenAllocRegister,
    TcgenAllocResume,
    TcgenDeallocRegister,
    TcgenDeallocResume,
    TcgenRelinquishRegister,
    TcgenRelinquishResume,
    SetmaxnregRegister,
    SetmaxnregResume,
    MbarrierExpectTx,
}

impl SyncCheckEffectKind {
    const COUNT: usize = 23;
    const ALL: [Self; Self::COUNT] = [
        Self::MbarrierInvalidate,
        Self::MbarrierInit,
        Self::MbarrierArrive,
        Self::MbarrierWait,
        Self::MbarrierCompletionIssue,
        Self::CpAsyncMbarrierArrive,
        Self::TcgenCommitIssue,
        Self::MbarrierCompletion,
        Self::NamedBarrierArrive,
        Self::NamedBarrierSyncRegister,
        Self::NamedBarrierSyncResume,
        Self::ClusterBarrierArrive,
        Self::ClusterBarrierWaitRegister,
        Self::ClusterBarrierWaitResume,
        Self::TcgenAllocRegister,
        Self::TcgenAllocResume,
        Self::TcgenDeallocRegister,
        Self::TcgenDeallocResume,
        Self::TcgenRelinquishRegister,
        Self::TcgenRelinquishResume,
        Self::SetmaxnregRegister,
        Self::SetmaxnregResume,
        Self::MbarrierExpectTx,
    ];

    const fn index(self) -> usize {
        self as usize
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::MbarrierInvalidate => "mbarrier.inval",
            Self::MbarrierInit => "mbarrier.init",
            Self::MbarrierArrive => "mbarrier.arrive",
            Self::MbarrierWait => "mbarrier.wait",
            Self::MbarrierCompletionIssue => "mbarrier.completion_issue",
            Self::CpAsyncMbarrierArrive => "cp.async.mbarrier.arrive",
            Self::TcgenCommitIssue => "tcgen05.commit.issue",
            Self::MbarrierCompletion => "mbarrier.complete_tx",
            Self::NamedBarrierArrive => "bar.arrive.register",
            Self::NamedBarrierSyncRegister => "bar.sync.register",
            Self::NamedBarrierSyncResume => "bar.sync.resume",
            Self::ClusterBarrierArrive => "barrier.cluster.arrive",
            Self::ClusterBarrierWaitRegister => "barrier.cluster.wait.register",
            Self::ClusterBarrierWaitResume => "barrier.cluster.wait.resume",
            Self::TcgenAllocRegister => "tcgen05.alloc.register",
            Self::TcgenAllocResume => "tcgen05.alloc.resume",
            Self::TcgenDeallocRegister => "tcgen05.dealloc.register",
            Self::TcgenDeallocResume => "tcgen05.dealloc.resume",
            Self::TcgenRelinquishRegister => "tcgen05.relinquish_alloc_permit.register",
            Self::TcgenRelinquishResume => "tcgen05.relinquish_alloc_permit.resume",
            Self::SetmaxnregRegister => "setmaxnreg.register",
            Self::SetmaxnregResume => "setmaxnreg.resume",
            Self::MbarrierExpectTx => "mbarrier.expect_tx",
        }
    }
}

/// Strict disposition observed while validating or committing a parity wait.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SyncCheckWaitState {
    Ready {
        generation: Option<u64>,
        consumed_now: bool,
    },
    Registered {
        generation: u64,
    },
}

impl From<StrictMbarrierWaitOutcome> for SyncCheckWaitState {
    fn from(outcome: StrictMbarrierWaitOutcome) -> Self {
        match outcome {
            StrictMbarrierWaitOutcome::Ready {
                generation,
                consumed_now,
            } => Self::Ready {
                generation,
                consumed_now,
            },
            StrictMbarrierWaitOutcome::Registered { generation } => Self::Registered { generation },
        }
    }
}

/// Typed successful effect record retained for report adapters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyncCheckEffectRecord {
    operation: DynamicOpId,
    effect: SyncCheckEffectKind,
    outcome: SyncCheckEffectOutcome,
}

impl SyncCheckEffectRecord {
    pub const fn operation(&self) -> &DynamicOpId {
        &self.operation
    }

    pub const fn effect(&self) -> SyncCheckEffectKind {
        self.effect
    }

    pub const fn outcome(&self) -> &SyncCheckEffectOutcome {
        &self.outcome
    }
}

/// Checker-specific information produced by a successfully committed effect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SyncCheckEffectOutcome {
    Applied,
    MbarrierArrive {
        completed_generation: Option<u64>,
        consumed_generation: Option<u64>,
        ready_warps: Box<[usize]>,
    },
    MbarrierArriveBatch {
        arrivals: Box<[SyncCheckMbarrierArrivalRecord]>,
    },
    MbarrierWait {
        staged: SyncCheckWaitState,
        committed: Option<SyncCheckWaitState>,
    },
    MbarrierCompletionIssue {
        actions: Box<[SyncCheckCompletionIssue]>,
    },
    MbarrierCompletion {
        action_id: PhysicalCompletionActionId,
        barrier_id: PhysicalBarrierId,
        generation: u64,
        transactions: u64,
        completion_kind: PhysicalCompletionKind,
        completed_generation: Option<u64>,
        ready_warps: Box<[usize]>,
    },
    NamedBarrierArrive {
        generation: u64,
        completed_now: bool,
    },
    ClusterBarrier {
        generation: u64,
        completed_now: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyncCheckMbarrierArrivalRecord {
    barrier_id: PhysicalBarrierId,
    generation: u64,
    completed_generation: Option<u64>,
    consumed_generation: Option<u64>,
    ready_warps: Box<[usize]>,
}

impl SyncCheckMbarrierArrivalRecord {
    pub const fn barrier_id(&self) -> PhysicalBarrierId {
        self.barrier_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn completed_generation(&self) -> Option<u64> {
        self.completed_generation
    }

    pub const fn consumed_generation(&self) -> Option<u64> {
        self.consumed_generation
    }

    pub fn ready_warps(&self) -> &[usize] {
        &self.ready_warps
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyncCheckCompletionIssue {
    action_id: PhysicalCompletionActionId,
    barrier_id: PhysicalBarrierId,
    generation: Option<u64>,
    transactions: u64,
}

impl SyncCheckCompletionIssue {
    pub const fn action_id(&self) -> PhysicalCompletionActionId {
        self.action_id
    }

    pub const fn barrier_id(&self) -> PhysicalBarrierId {
        self.barrier_id
    }

    pub const fn generation(&self) -> Option<u64> {
        self.generation
    }

    pub const fn transactions(&self) -> u64 {
        self.transactions
    }
}

// Keep this charge identical to the Python report adapter's structured
// diagnostic accounting. Recording it incrementally lets a bounded launch
// discard the success journal without losing exact resource usage.
const STRUCTURED_RECORD_OVERHEAD: u64 = 16;
const STRUCTURED_SEQUENCE_OVERHEAD: u64 = 8;
const STRUCTURED_SCALAR_BYTES: u64 = 8;
const STRUCTURED_BOOL_BYTES: u64 = 1;
const STRUCTURED_OPTION_TAG_BYTES: u64 = 1;

#[cfg(test)]
pub(crate) fn sync_check_effect_diagnostic_bytes(effect: &SyncCheckEffectRecord) -> u64 {
    sync_check_effect_fields_diagnostic_bytes(effect.operation(), effect.effect(), effect.outcome())
}

fn sync_check_effect_fields_diagnostic_bytes(
    operation: &DynamicOpId,
    effect: SyncCheckEffectKind,
    outcome: &SyncCheckEffectOutcome,
) -> u64 {
    STRUCTURED_RECORD_OVERHEAD
        .saturating_add(dynamic_op_diagnostic_bytes(operation))
        .saturating_add(text_bytes(effect.name()))
        .saturating_add(sync_check_effect_outcome_diagnostic_bytes(outcome))
}

fn sync_check_effect_outcome_diagnostic_bytes(outcome: &SyncCheckEffectOutcome) -> u64 {
    let bytes = STRUCTURED_RECORD_OVERHEAD;
    match outcome {
        SyncCheckEffectOutcome::Applied => bytes.saturating_add(text_bytes("applied")),
        SyncCheckEffectOutcome::MbarrierArrive {
            completed_generation,
            consumed_generation,
            ready_warps,
        } => bytes
            .saturating_add(text_bytes("mbarrier_arrive"))
            .saturating_add(optional_u64_diagnostic_bytes(*completed_generation))
            .saturating_add(optional_u64_diagnostic_bytes(*consumed_generation))
            .saturating_add(usize_slice_diagnostic_bytes(ready_warps)),
        SyncCheckEffectOutcome::MbarrierArriveBatch { arrivals } => {
            let mut bytes = bytes
                .saturating_add(text_bytes("mbarrier_arrive_batch"))
                .saturating_add(STRUCTURED_SEQUENCE_OVERHEAD);
            for arrival in arrivals {
                bytes = bytes
                    .saturating_add(STRUCTURED_RECORD_OVERHEAD)
                    .saturating_add(physical_barrier_diagnostic_bytes(arrival.barrier_id()))
                    .saturating_add(STRUCTURED_SCALAR_BYTES)
                    .saturating_add(optional_u64_diagnostic_bytes(
                        arrival.completed_generation(),
                    ))
                    .saturating_add(optional_u64_diagnostic_bytes(arrival.consumed_generation()))
                    .saturating_add(usize_slice_diagnostic_bytes(arrival.ready_warps()));
            }
            bytes
        }
        SyncCheckEffectOutcome::MbarrierWait { staged, committed } => bytes
            .saturating_add(text_bytes("mbarrier_wait"))
            .saturating_add(sync_check_wait_state_diagnostic_bytes(staged))
            .saturating_add(STRUCTURED_OPTION_TAG_BYTES)
            .saturating_add(
                committed
                    .as_ref()
                    .map_or(0, sync_check_wait_state_diagnostic_bytes),
            ),
        SyncCheckEffectOutcome::MbarrierCompletionIssue { actions } => {
            let mut bytes = bytes
                .saturating_add(text_bytes("mbarrier_completion_issue"))
                .saturating_add(STRUCTURED_SEQUENCE_OVERHEAD);
            for action in actions {
                bytes = bytes
                    .saturating_add(STRUCTURED_RECORD_OVERHEAD)
                    .saturating_add(STRUCTURED_SCALAR_BYTES)
                    .saturating_add(physical_barrier_diagnostic_bytes(action.barrier_id()))
                    .saturating_add(optional_u64_diagnostic_bytes(action.generation()))
                    .saturating_add(STRUCTURED_SCALAR_BYTES);
            }
            bytes
        }
        SyncCheckEffectOutcome::MbarrierCompletion {
            barrier_id,
            completion_kind,
            completed_generation,
            ready_warps,
            ..
        } => bytes
            .saturating_add(text_bytes("mbarrier_completion"))
            .saturating_add(STRUCTURED_SCALAR_BYTES)
            .saturating_add(physical_barrier_diagnostic_bytes(*barrier_id))
            .saturating_add(STRUCTURED_SCALAR_BYTES.saturating_mul(2))
            .saturating_add(match completion_kind {
                PhysicalCompletionKind::Transaction { .. } => {
                    text_bytes("transaction").saturating_add(STRUCTURED_SCALAR_BYTES)
                }
                PhysicalCompletionKind::Arrival { .. } => {
                    text_bytes("arrival").saturating_add(STRUCTURED_SCALAR_BYTES.saturating_mul(2))
                }
            })
            .saturating_add(optional_u64_diagnostic_bytes(*completed_generation))
            .saturating_add(usize_slice_diagnostic_bytes(ready_warps)),
        SyncCheckEffectOutcome::NamedBarrierArrive { .. } => bytes
            .saturating_add(text_bytes("named_barrier_arrive"))
            .saturating_add(STRUCTURED_SCALAR_BYTES)
            .saturating_add(STRUCTURED_BOOL_BYTES),
        SyncCheckEffectOutcome::ClusterBarrier { .. } => bytes
            .saturating_add(text_bytes("cluster_barrier"))
            .saturating_add(STRUCTURED_SCALAR_BYTES)
            .saturating_add(STRUCTURED_BOOL_BYTES),
    }
}

fn sync_check_wait_state_diagnostic_bytes(state: &SyncCheckWaitState) -> u64 {
    match state {
        SyncCheckWaitState::Ready { generation, .. } => STRUCTURED_RECORD_OVERHEAD
            .saturating_add(text_bytes("ready"))
            .saturating_add(optional_u64_diagnostic_bytes(*generation))
            .saturating_add(STRUCTURED_BOOL_BYTES),
        SyncCheckWaitState::Registered { .. } => STRUCTURED_RECORD_OVERHEAD
            .saturating_add(text_bytes("registered"))
            .saturating_add(STRUCTURED_SCALAR_BYTES),
    }
}

fn dynamic_op_diagnostic_bytes(operation: &DynamicOpId) -> u64 {
    STRUCTURED_RECORD_OVERHEAD
        .saturating_add(STRUCTURED_SCALAR_BYTES.saturating_mul(4))
        .saturating_add(STRUCTURED_SEQUENCE_OVERHEAD)
        .saturating_add(
            count_bytes(operation.loop_frames().len())
                .saturating_mul(STRUCTURED_RECORD_OVERHEAD + STRUCTURED_SCALAR_BYTES * 2),
        )
}

fn physical_barrier_diagnostic_bytes(_barrier: PhysicalBarrierId) -> u64 {
    STRUCTURED_RECORD_OVERHEAD.saturating_add(STRUCTURED_SCALAR_BYTES.saturating_mul(3))
}

fn optional_u64_diagnostic_bytes(value: Option<u64>) -> u64 {
    STRUCTURED_OPTION_TAG_BYTES.saturating_add(value.map_or(0, |_| STRUCTURED_SCALAR_BYTES))
}

fn usize_slice_diagnostic_bytes(values: &[usize]) -> u64 {
    STRUCTURED_SEQUENCE_OVERHEAD
        .saturating_add(count_bytes(values.len()).saturating_mul(STRUCTURED_SCALAR_BYTES))
}

fn text_bytes(value: &str) -> u64 {
    count_bytes(value.len())
}

fn count_bytes(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// One strict protocol rejection with its exact dynamic source identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SyncCheckProtocolError {
    Mbarrier(StrictMbarrierError),
    NamedBarrier(StrictNamedBarrierError),
    ClusterBarrier(StrictClusterBarrierError),
    Causality(SyncCausalityError),
}

impl fmt::Display for SyncCheckProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mbarrier(error) => error.fmt(f),
            Self::NamedBarrier(error) => error.fmt(f),
            Self::ClusterBarrier(error) => error.fmt(f),
            Self::Causality(error) => error.fmt(f),
        }
    }
}

impl Error for SyncCheckProtocolError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Mbarrier(error) => Some(error),
            Self::NamedBarrier(error) => Some(error),
            Self::ClusterBarrier(error) => Some(error),
            Self::Causality(error) => Some(error),
        }
    }
}

impl From<StrictMbarrierError> for SyncCheckProtocolError {
    fn from(error: StrictMbarrierError) -> Self {
        Self::Mbarrier(error)
    }
}

impl From<StrictNamedBarrierError> for SyncCheckProtocolError {
    fn from(error: StrictNamedBarrierError) -> Self {
        Self::NamedBarrier(error)
    }
}

impl From<StrictClusterBarrierError> for SyncCheckProtocolError {
    fn from(error: StrictClusterBarrierError) -> Self {
        Self::ClusterBarrier(error)
    }
}

impl From<SyncCausalityError> for SyncCheckProtocolError {
    fn from(error: SyncCausalityError) -> Self {
        Self::Causality(error)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyncCheckFinding {
    operation: DynamicOpId,
    effect: SyncCheckEffectKind,
    error: SyncCheckProtocolError,
}

impl SyncCheckFinding {
    pub const fn operation(&self) -> &DynamicOpId {
        &self.operation
    }

    pub const fn effect(&self) -> SyncCheckEffectKind {
        self.effect
    }

    pub const fn error(&self) -> &SyncCheckProtocolError {
        &self.error
    }
}

/// Typed reason why the current vertical slice cannot certify a clean result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SyncCheckIncompleteReason {
    AnalysisGap {
        operation: DynamicOpId,
        kind: AnalysisGapKind,
    },
    CompletionActionUnobserved {
        action_id: PhysicalCompletionActionId,
        barrier_id: PhysicalBarrierId,
        generation: u64,
    },
    CompletionTransitionUnobserved {
        operation: DynamicOpId,
        barrier_id: PhysicalBarrierId,
        generation: u64,
    },
    /// `before_effect` staged a candidate, but no successful `after_effect`
    /// was observed. This is expected when the numeric runtime rejects/cancels
    /// the effect and proves that no strict state was committed prematurely.
    EffectCommitUnobserved {
        operation: DynamicOpId,
        effect: SyncCheckEffectKind,
    },
    ClusterBarrierParticipantExitUnmodeled {
        barrier_id: ClusterBarrierId,
        generation: u64,
        missing_warps: Box<[usize]>,
    },
    ClusterBarrierUnalignedUnmodeled {
        operation: DynamicOpId,
        effect: SyncCheckEffectKind,
    },
    ClusterBarrierRearrivalWithoutWaitUnmodeled {
        operation: DynamicOpId,
        barrier_id: ClusterBarrierId,
        generation: u64,
        warp_id: usize,
    },
}

/// Immutable primitive result used by the eventual Python report adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyncCheckResult {
    status: SyncCheckStatus,
    findings: Box<[SyncCheckFinding]>,
    effects: Box<[SyncCheckEffectRecord]>,
    effect_counts: [u64; SyncCheckEffectKind::COUNT],
    full_effect_diagnostic_bytes: u64,
    effect_journal_complete: bool,
    incomplete_reasons: Box<[SyncCheckIncompleteReason]>,
}

impl SyncCheckResult {
    pub fn merge_cluster_results(results: &[Self]) -> Self {
        let mut findings = Vec::new();
        let mut effects = Vec::new();
        let mut effect_counts = [0_u64; SyncCheckEffectKind::COUNT];
        let mut full_effect_diagnostic_bytes = 0_u64;
        let effect_journal_complete = results.iter().all(|result| result.effect_journal_complete);
        let mut incomplete_reasons = Vec::new();
        let mut saw_error = false;
        let mut saw_incomplete = false;

        for result in results {
            saw_error |= result.status == SyncCheckStatus::Error;
            saw_incomplete |= result.status == SyncCheckStatus::Incomplete;
            findings.extend(result.findings.iter().cloned());
            if effect_journal_complete {
                effects.extend(result.effects.iter().cloned());
            }
            for (total, count) in effect_counts.iter_mut().zip(result.effect_counts) {
                *total = total.saturating_add(count);
            }
            full_effect_diagnostic_bytes =
                full_effect_diagnostic_bytes.saturating_add(result.full_effect_diagnostic_bytes);
            for reason in result.incomplete_reasons.iter().cloned() {
                push_unique_incomplete(&mut incomplete_reasons, reason);
            }
        }

        let status = if saw_error || !findings.is_empty() {
            SyncCheckStatus::Error
        } else if saw_incomplete || !incomplete_reasons.is_empty() {
            SyncCheckStatus::Incomplete
        } else {
            SyncCheckStatus::Clean
        };
        Self {
            status,
            findings: findings.into_boxed_slice(),
            effects: effects.into_boxed_slice(),
            effect_counts,
            full_effect_diagnostic_bytes,
            effect_journal_complete,
            incomplete_reasons: incomplete_reasons.into_boxed_slice(),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_test_applied_effects(count: usize) -> Self {
        let effects = (0..count)
            .map(|sequence| SyncCheckEffectRecord {
                operation: DynamicOpId::new(
                    0,
                    0,
                    u64::try_from(sequence).unwrap_or(u64::MAX),
                    crate::StaticOpId::new(1),
                    [],
                ),
                effect: SyncCheckEffectKind::NamedBarrierSyncRegister,
                outcome: SyncCheckEffectOutcome::Applied,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut effect_counts = [0; SyncCheckEffectKind::COUNT];
        effect_counts[SyncCheckEffectKind::NamedBarrierSyncRegister.index()] =
            u64::try_from(count).unwrap_or(u64::MAX);
        let full_effect_diagnostic_bytes = effects
            .iter()
            .map(sync_check_effect_diagnostic_bytes)
            .fold(0_u64, u64::saturating_add);
        Self {
            status: SyncCheckStatus::Clean,
            findings: Box::new([]),
            effects,
            effect_counts,
            full_effect_diagnostic_bytes,
            effect_journal_complete: true,
            incomplete_reasons: Box::new([]),
        }
    }

    pub const fn status(&self) -> SyncCheckStatus {
        self.status
    }

    pub fn findings(&self) -> &[SyncCheckFinding] {
        &self.findings
    }

    pub fn effects(&self) -> &[SyncCheckEffectRecord] {
        &self.effects
    }

    pub fn effect_counts(&self) -> impl Iterator<Item = (SyncCheckEffectKind, u64)> + '_ {
        SyncCheckEffectKind::ALL
            .into_iter()
            .zip(self.effect_counts)
            .filter(|(_, count)| *count != 0)
    }

    pub fn total_effect_count(&self) -> u64 {
        self.effect_counts
            .iter()
            .copied()
            .fold(0_u64, u64::saturating_add)
    }

    pub const fn full_effect_diagnostic_bytes(&self) -> u64 {
        self.full_effect_diagnostic_bytes
    }

    pub const fn effect_journal_complete(&self) -> bool {
        self.effect_journal_complete
    }

    pub fn incomplete_reasons(&self) -> &[SyncCheckIncompleteReason] {
        &self.incomplete_reasons
    }
}

/// Launch-wide state shared by every native sync-check warp.
pub struct SyncCheckLaunchState {
    inner: Mutex<SyncCheckState>,
    cluster_inners: Box<[Mutex<SyncCheckState>]>,
    /// Per-launch context shared with the peer racecheck observer when this
    /// state is one half of a racecheck run. Synccheck-only launches own a
    /// context of their own; either way the fields below are read through it.
    context: Arc<CheckerLaunchContext>,
    records_only_fixed_sync_transitions: bool,
    effect_diagnostic_limit: u64,
    retained_effect_diagnostic_bytes: AtomicU64,
    effect_journal_overflowed: AtomicBool,
}

impl Default for SyncCheckLaunchState {
    fn default() -> Self {
        Self::with_shared_context(Arc::new(CheckerLaunchContext::default()), false, u64::MAX)
    }
}

struct SyncCheckState {
    protocol: StrictMbarrierProtocol,
    named_protocol: StrictNamedBarrierProtocol,
    cluster_protocol: StrictClusterBarrierProtocol,
    causality: SyncCausalityTracker,
    named_causal_payloads: BTreeMap<(crate::NamedBarrierId, u64), SyncClockPayload>,
    cluster_causal_payloads: BTreeMap<(ClusterBarrierId, u64), SyncClockPayload>,
    pending_mbarrier_init_fences: BTreeSet<usize>,
    revision: u64,
    named_revision: u64,
    cluster_revision: u64,
    staged: BTreeMap<DynamicOpId, StagedSyncEffect>,
    staged_named: BTreeMap<DynamicOpId, StagedNamedBarrierEffect>,
    staged_cluster: BTreeMap<DynamicOpId, StagedClusterBarrierEffect>,
    staged_completions: BTreeMap<PhysicalCompletionActionId, StagedMbarrierCompletion>,
    completion_tokens: BTreeMap<PhysicalCompletionActionId, StrictCompletionBinding>,
    findings: Vec<SyncCheckFinding>,
    effects: Vec<SyncCheckEffectRecord>,
    effect_counts: [u64; SyncCheckEffectKind::COUNT],
    full_effect_diagnostic_bytes: u64,
    retains_effect_journal: bool,
    incomplete_reasons: Vec<SyncCheckIncompleteReason>,
}

impl SyncCheckState {
    /// Build a state whose strict named-barrier protocol knows the launch
    /// CTA thread count. `None` (no topology) skips full-CTA-contract checks.
    fn with_cta_thread_count(cta_thread_count: Option<u64>) -> Self {
        Self {
            named_protocol: StrictNamedBarrierProtocol::new(cta_thread_count),
            ..Self::default()
        }
    }
}

impl Default for SyncCheckState {
    fn default() -> Self {
        Self {
            protocol: StrictMbarrierProtocol::default(),
            named_protocol: StrictNamedBarrierProtocol::default(),
            cluster_protocol: StrictClusterBarrierProtocol::default(),
            causality: SyncCausalityTracker::default(),
            named_causal_payloads: BTreeMap::new(),
            cluster_causal_payloads: BTreeMap::new(),
            pending_mbarrier_init_fences: BTreeSet::new(),
            revision: 0,
            named_revision: 0,
            cluster_revision: 0,
            staged: BTreeMap::new(),
            staged_named: BTreeMap::new(),
            staged_cluster: BTreeMap::new(),
            staged_completions: BTreeMap::new(),
            completion_tokens: BTreeMap::new(),
            findings: Vec::new(),
            effects: Vec::new(),
            effect_counts: [0; SyncCheckEffectKind::COUNT],
            full_effect_diagnostic_bytes: 0,
            retains_effect_journal: true,
            incomplete_reasons: Vec::new(),
        }
    }
}

#[derive(Clone)]
struct StrictCompletionBinding {
    token: StrictMbarrierCompletionToken,
    causal_token: MbarrierCompletionCausalToken,
    kind: PhysicalCompletionKind,
}

impl SyncCheckLaunchState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_external_analysis_gap(
        &self,
        operation: &OperationContext,
        gap: AnalysisGapEffect,
    ) -> Result<(), EngineError> {
        self.after_effect(operation, OperationEffect::AnalysisGap(gap))
    }

    /// Build a synccheck observer over an already-shared launch context.
    ///
    /// This is the one constructor; the named variants below build a context
    /// and call it. Racecheck calls it directly with the context it also keeps
    /// for itself, so the two peers read one object instead of racecheck
    /// reading its own constructor inputs back out of this state.
    ///
    /// A topology in the context builds one independent protocol/clock shard
    /// per cluster. Physical mbarriers and named barriers are CTA-scoped while
    /// the hardware cluster barrier is cluster-scoped, so no Synccheck protocol
    /// resource spans two clusters: sharding removes false launch-wide lock
    /// contention without changing any resource's transition order.
    pub(crate) fn with_shared_context(
        context: Arc<CheckerLaunchContext>,
        records_only_fixed_sync_transitions: bool,
        effect_diagnostic_limit: u64,
    ) -> Self {
        let cta_thread_count = context
            .topology()
            .map(|topology| (topology.warps_per_cta() * crate::WARP_SIZE) as u64);
        Self {
            inner: Mutex::new(SyncCheckState::with_cta_thread_count(cta_thread_count)),
            cluster_inners: (0..context.cluster_shards())
                .map(|_| Mutex::new(SyncCheckState::with_cta_thread_count(cta_thread_count)))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            context,
            records_only_fixed_sync_transitions,
            effect_diagnostic_limit,
            retained_effect_diagnostic_bytes: AtomicU64::new(0),
            effect_journal_overflowed: AtomicBool::new(false),
        }
    }

    fn with_options(
        topology: Option<LaunchTopology>,
        transitions: ResolvedTransitionLog,
        global_write_allocations: Option<BTreeSet<PhysicalAllocationId>>,
        records_resolved_transitions: bool,
        records_only_fixed_sync_transitions: bool,
        effect_diagnostic_limit: u64,
    ) -> Self {
        Self::with_shared_context(
            Arc::new(CheckerLaunchContext::new(
                topology,
                transitions,
                global_write_allocations,
                records_resolved_transitions,
            )),
            records_only_fixed_sync_transitions,
            effect_diagnostic_limit,
        )
    }

    pub fn with_topology_transition_log_and_global_write_allocations(
        topology: LaunchTopology,
        transitions: ResolvedTransitionLog,
        allocations: impl IntoIterator<Item = PhysicalAllocationId>,
        effect_diagnostic_limit: u64,
    ) -> Self {
        Self::with_options(
            Some(topology),
            transitions,
            Some(allocations.into_iter().collect()),
            true,
            true,
            effect_diagnostic_limit,
        )
    }

    /// Compact memory analysis with no topology and no transition log.
    ///
    /// Only Synccheck's own tests build this shape now: the racecheck peer
    /// that used to select it goes through [`Self::with_shared_context`].
    #[cfg(test)]
    fn with_global_write_allocations(
        allocations: impl IntoIterator<Item = PhysicalAllocationId>,
    ) -> Self {
        Self::with_options(
            None,
            ResolvedTransitionLog::default(),
            Some(allocations.into_iter().collect()),
            false,
            false,
            u64::MAX,
        )
    }

    /// Live-staging launch state with a topology: the strict protocols run
    /// during execution with the CTA thread count derived from the topology,
    /// so contract-derived full-CTA checks are active.
    #[cfg(test)]
    fn with_topology_for_strict_protocols(topology: LaunchTopology) -> Self {
        Self::with_options(
            Some(topology),
            ResolvedTransitionLog::default(),
            None,
            true,
            false,
            u64::MAX,
        )
    }

    fn record_effect(
        &self,
        state: &mut SyncCheckState,
        operation: &DynamicOpId,
        effect: SyncCheckEffectKind,
        outcome: SyncCheckEffectOutcome,
    ) {
        let _profile_timer = ProfileTimer::new(ProfileKind::SyncRecordEffect);
        let effect_index = effect.index();
        state.effect_counts[effect_index] = state.effect_counts[effect_index].saturating_add(1);
        let diagnostic_bytes =
            sync_check_effect_fields_diagnostic_bytes(operation, effect, &outcome);
        state.full_effect_diagnostic_bytes = state
            .full_effect_diagnostic_bytes
            .saturating_add(diagnostic_bytes);

        if !state.retains_effect_journal {
            return;
        }
        if self.effect_diagnostic_limit == u64::MAX {
            state.effects.push(SyncCheckEffectRecord {
                operation: operation.clone(),
                effect,
                outcome,
            });
            return;
        }
        if self.effect_journal_overflowed.load(Ordering::Acquire) {
            state.effects.clear();
            state.retains_effect_journal = false;
            return;
        }
        let retained = self.retained_effect_diagnostic_bytes.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| {
                current
                    .checked_add(diagnostic_bytes)
                    .filter(|next| *next <= self.effect_diagnostic_limit)
            },
        );
        if retained.is_ok() {
            state.effects.push(SyncCheckEffectRecord {
                operation: operation.clone(),
                effect,
                outcome,
            });
        } else {
            self.effect_journal_overflowed
                .store(true, Ordering::Release);
            state.effects.clear();
            state.retains_effect_journal = false;
        }
    }

    fn inner_for_global_warp(&self, global_warp_id: usize) -> &Mutex<SyncCheckState> {
        let Some(topology) = self.context.topology() else {
            return &self.inner;
        };
        let cluster_id = topology
            .cluster_id_for_warp(global_warp_id)
            .expect("sync-check operation warp belongs to its launch topology");
        &self.cluster_inners[cluster_id]
    }

    fn inner_for_operation(&self, operation: &DynamicOpId) -> &Mutex<SyncCheckState> {
        self.inner_for_global_warp(operation.global_warp_id())
    }

    fn inner_for_global_cta(&self, global_cta_id: usize) -> &Mutex<SyncCheckState> {
        let Some(topology) = self.context.topology() else {
            return &self.inner;
        };
        assert!(
            global_cta_id < topology.cta_count(),
            "sync-check CTA belongs to its launch topology"
        );
        &self.cluster_inners[global_cta_id / topology.ctas_per_cluster()]
    }

    fn inner_for_physical_barrier(&self, barrier_id: PhysicalBarrierId) -> &Mutex<SyncCheckState> {
        self.inner_for_global_cta(barrier_id.target_global_cta_id())
    }

    fn inner_for_named_barrier(&self, barrier_id: crate::NamedBarrierId) -> &Mutex<SyncCheckState> {
        self.inner_for_global_cta(barrier_id.global_cta_id())
    }

    fn inner_for_cluster_barrier(&self, barrier_id: ClusterBarrierId) -> &Mutex<SyncCheckState> {
        match self.context.topology() {
            Some(topology) => {
                assert!(
                    barrier_id.cluster_id() < topology.clusters(),
                    "sync-check cluster barrier belongs to its launch topology"
                );
                &self.cluster_inners[barrier_id.cluster_id()]
            }
            None => &self.inner,
        }
    }

    pub fn tracks_global_allocation(&self, allocation: PhysicalAllocationId) -> bool {
        self.context.tracks_global_allocation(allocation)
    }

    pub fn uses_compact_memory_analysis(&self) -> bool {
        self.context.uses_compact_memory_analysis()
    }

    pub fn records_resolved_transitions(&self) -> bool {
        self.context.records_resolved_transitions()
    }

    const fn records_only_fixed_sync_transitions(&self) -> bool {
        self.records_only_fixed_sync_transitions
    }

    pub fn transition_log(&self) -> &ResolvedTransitionLog {
        self.context.transition_log()
    }

    fn recorded_operation_clock(
        &self,
        operation: &DynamicOpId,
        effect: OperationEffect<'_>,
    ) -> Result<Option<SyncVectorClock>, EngineError> {
        let state = self
            .inner_for_operation(operation)
            .lock()
            .expect("sync-check state poisoned");
        if matches!(effect, OperationEffect::MbarrierWait { outcome: None, .. }) {
            return state
                .causality
                .preview_program_tick(operation.global_warp_id())
                .map(Some)
                .map_err(|error| {
                    EngineError::message(format!(
                        "synccheck could not record blocking wait issue clock at {operation}: {error}"
                    ))
                });
        }
        Ok(state
            .causality
            .warp_clock(operation.global_warp_id())
            .cloned())
    }

    fn completion_source_clocks(
        &self,
        operation: &DynamicOpId,
    ) -> Vec<(PhysicalCompletionActionId, SyncVectorClock)> {
        let state = self
            .inner_for_operation(operation)
            .lock()
            .expect("sync-check state poisoned");
        state
            .completion_tokens
            .iter()
            .filter_map(|(&action_id, binding)| {
                (binding.token.issue_witness() == Some(operation))
                    .then(|| (action_id, binding.causal_token.issuer_clock().clone()))
            })
            .collect()
    }

    pub fn snapshot(&self, barrier_id: PhysicalBarrierId) -> StrictMbarrierSnapshot {
        self.inner_for_physical_barrier(barrier_id)
            .lock()
            .expect("sync-check state poisoned")
            .protocol
            .snapshot(barrier_id)
    }

    pub fn staged_mbarrier_arrive_generation(&self, operation: &DynamicOpId) -> Option<u64> {
        let state = self
            .inner_for_operation(operation)
            .lock()
            .expect("sync-check state poisoned");
        let staged = state.staged.get(operation)?;
        let OwnedSyncEffect::Arrive { plan, .. } = &staged.effect else {
            return None;
        };
        staged
            .candidate
            .as_ref()?
            .snapshot(plan.barrier_id())
            .generation()
    }

    pub fn staged_mbarrier_arrive_generations(
        &self,
        operation: &DynamicOpId,
    ) -> Option<Vec<(PhysicalBarrierId, u64)>> {
        let state = self
            .inner_for_operation(operation)
            .lock()
            .expect("sync-check state poisoned");
        let staged = state.staged.get(operation)?;
        let candidate = staged.candidate.as_ref()?;
        match &staged.effect {
            OwnedSyncEffect::Arrive { plan, .. } => candidate
                .snapshot(plan.barrier_id())
                .generation()
                .map(|generation| vec![(plan.barrier_id(), generation)]),
            OwnedSyncEffect::ArriveBatch { plan, .. } => plan
                .entries()
                .iter()
                .map(|entry| {
                    let barrier_id = entry.plan().barrier_id();
                    candidate
                        .snapshot(barrier_id)
                        .generation()
                        .map(|generation| (barrier_id, generation))
                })
                .collect::<Option<Vec<_>>>(),
            _ => None,
        }
    }

    pub fn named_snapshot(&self, barrier_id: crate::NamedBarrierId) -> StrictNamedBarrierSnapshot {
        self.inner_for_named_barrier(barrier_id)
            .lock()
            .expect("sync-check state poisoned")
            .named_protocol
            .snapshot(barrier_id)
    }

    pub fn result(&self) -> SyncCheckResult {
        self.result_with_terminal_validation(true, &[])
    }

    pub fn result_before_aborted_execution(&self) -> SyncCheckResult {
        self.result_with_terminal_validation(false, &[])
    }

    pub(crate) fn result_before_aborted_execution_with_cluster_barrier_participant_exits(
        &self,
        participant_exits: &[ClusterBarrierParticipantExitEvidence],
    ) -> SyncCheckResult {
        self.result_with_terminal_validation(false, participant_exits)
    }

    pub fn result_for_execution(&self, execution: &ExecutionReport) -> SyncCheckResult {
        let participant_exits = execution.cluster_barrier_participant_exit_evidence();
        self.result_with_terminal_validation(execution.is_success(), &participant_exits)
    }

    fn result_with_terminal_validation(
        &self,
        validate_terminal_state: bool,
        participant_exits: &[ClusterBarrierParticipantExitEvidence],
    ) -> SyncCheckResult {
        let retain_effect_journal = !self.effect_journal_overflowed.load(Ordering::Acquire);
        let mut partials = Vec::new();
        if self.cluster_inners.is_empty() {
            let state = self.inner.lock().expect("sync-check state poisoned");
            partials.push(Self::result_for_state(
                &state,
                validate_terminal_state,
                participant_exits,
                retain_effect_journal,
            ));
        } else {
            for (cluster_id, inner) in self.cluster_inners.iter().enumerate() {
                let cluster_participant_exits = participant_exits
                    .iter()
                    .filter(|exit| exit.cluster_id() == cluster_id)
                    .cloned()
                    .collect::<Vec<_>>();
                let state = inner.lock().expect("sync-check state poisoned");
                partials.push(Self::result_for_state(
                    &state,
                    validate_terminal_state,
                    &cluster_participant_exits,
                    retain_effect_journal,
                ));
            }
        }

        let mut findings = Vec::new();
        let mut effects = Vec::new();
        let mut effect_counts = [0_u64; SyncCheckEffectKind::COUNT];
        let mut full_effect_diagnostic_bytes = 0_u64;
        let effect_journal_complete = partials
            .iter()
            .all(|partial| partial.effect_journal_complete);
        let mut incomplete_reasons = Vec::new();
        for partial in partials {
            for finding in partial.findings {
                if !findings.contains(&finding) {
                    findings.push(finding);
                }
            }
            if effect_journal_complete {
                effects.extend(partial.effects);
            }
            for (total, count) in effect_counts.iter_mut().zip(partial.effect_counts) {
                *total = total.saturating_add(count);
            }
            full_effect_diagnostic_bytes =
                full_effect_diagnostic_bytes.saturating_add(partial.full_effect_diagnostic_bytes);
            for reason in partial.incomplete_reasons {
                push_unique_incomplete(&mut incomplete_reasons, reason);
            }
        }
        if !self.cluster_inners.is_empty() {
            findings.sort_by(|left, right| {
                left.operation()
                    .cmp(right.operation())
                    .then_with(|| left.effect().cmp(&right.effect()))
            });
            effects.sort_by(|left, right| {
                left.operation()
                    .cmp(right.operation())
                    .then_with(|| left.effect().cmp(&right.effect()))
            });
        }
        let status = if !findings.is_empty() {
            SyncCheckStatus::Error
        } else if !incomplete_reasons.is_empty() {
            SyncCheckStatus::Incomplete
        } else {
            SyncCheckStatus::Clean
        };
        SyncCheckResult {
            status,
            findings: findings.into_boxed_slice(),
            effects: effects.into_boxed_slice(),
            effect_counts,
            full_effect_diagnostic_bytes,
            effect_journal_complete,
            incomplete_reasons: incomplete_reasons.into_boxed_slice(),
        }
    }

    fn result_for_state(
        state: &SyncCheckState,
        validate_terminal_state: bool,
        participant_exits: &[ClusterBarrierParticipantExitEvidence],
        retain_effect_journal: bool,
    ) -> SyncCheckResult {
        let mut findings = state.findings.clone();
        if validate_terminal_state && findings.is_empty() {
            findings.extend(
                state
                    .named_protocol
                    .full_cta_aligned_nonuniform_errors()
                    .into_iter()
                    .filter_map(|error| {
                        let operation = error.witness()?.clone();
                        Some(SyncCheckFinding {
                            operation,
                            effect: SyncCheckEffectKind::NamedBarrierSyncRegister,
                            error: error.into(),
                        })
                    }),
            );
        }
        let mut incomplete_reasons = state.incomplete_reasons.clone();
        if validate_terminal_state {
            incomplete_reasons.extend(state.staged.iter().map(|(operation, staged)| {
                SyncCheckIncompleteReason::EffectCommitUnobserved {
                    operation: operation.clone(),
                    effect: staged.effect.kind(),
                }
            }));
            incomplete_reasons.extend(state.staged_named.iter().map(|(operation, staged)| {
                SyncCheckIncompleteReason::EffectCommitUnobserved {
                    operation: operation.clone(),
                    effect: staged.effect.kind(),
                }
            }));
            incomplete_reasons.extend(state.staged_cluster.iter().map(|(operation, staged)| {
                SyncCheckIncompleteReason::EffectCommitUnobserved {
                    operation: operation.clone(),
                    effect: staged.effect.kind(),
                }
            }));
        }
        if findings.is_empty() && (validate_terminal_state || !participant_exits.is_empty()) {
            let incomplete_generations = state.cluster_protocol.incomplete_generations();
            if validate_terminal_state {
                incomplete_reasons.extend(incomplete_generations.into_iter().map(|generation| {
                    SyncCheckIncompleteReason::ClusterBarrierParticipantExitUnmodeled {
                        barrier_id: generation.barrier_id(),
                        generation: generation.generation(),
                        missing_warps: generation.missing_warps().into(),
                    }
                }));
            } else {
                for participant_exit in participant_exits {
                    let matching = incomplete_generations.iter().find(|generation| {
                        generation.barrier_id().cluster_id() == participant_exit.cluster_id()
                            && generation.generation() == participant_exit.generation()
                            && participant_exit
                                .exited_warps()
                                .iter()
                                .all(|warp_id| generation.missing_warps().contains(warp_id))
                    });
                    debug_assert!(
                        matching.is_some(),
                        "controlled cluster-barrier exit evidence must match strict protocol state"
                    );
                    if let Some(generation) = matching {
                        incomplete_reasons.push(
                            SyncCheckIncompleteReason::ClusterBarrierParticipantExitUnmodeled {
                                barrier_id: generation.barrier_id(),
                                generation: generation.generation(),
                                missing_warps: participant_exit.exited_warps().into(),
                            },
                        );
                    }
                }
            }
        }
        if validate_terminal_state {
            incomplete_reasons.extend(state.completion_tokens.iter().map(
                |(&action_id, binding)| SyncCheckIncompleteReason::CompletionActionUnobserved {
                    action_id,
                    barrier_id: binding.token.barrier_id(),
                    generation: binding.token.generation(),
                },
            ));
        }
        let status = if !findings.is_empty() {
            SyncCheckStatus::Error
        } else if !incomplete_reasons.is_empty() {
            SyncCheckStatus::Incomplete
        } else {
            SyncCheckStatus::Clean
        };
        SyncCheckResult {
            status,
            findings: findings.into_boxed_slice(),
            effects: if retain_effect_journal && state.retains_effect_journal {
                state.effects.clone().into_boxed_slice()
            } else {
                Box::new([])
            },
            effect_counts: state.effect_counts,
            full_effect_diagnostic_bytes: state.full_effect_diagnostic_bytes,
            effect_journal_complete: retain_effect_journal && state.retains_effect_journal,
            incomplete_reasons: incomplete_reasons.into_boxed_slice(),
        }
    }

    fn resolved_completion_issues(&self, operation: &DynamicOpId) -> Vec<ResolvedCompletionEffect> {
        let state = self
            .inner_for_operation(operation)
            .lock()
            .expect("sync-check state poisoned");
        state
            .completion_tokens
            .iter()
            .filter_map(|(&action_id, binding)| {
                if binding.token.issue_witness() != Some(operation) {
                    return None;
                }
                let PhysicalCompletionKind::Transaction { transactions } = binding.kind else {
                    return None;
                };
                (transactions != 0).then(|| {
                    ResolvedCompletionEffect::new(
                        action_id.get(),
                        ResolvedSyncResource::physical_mbarrier(
                            binding.token.barrier_id(),
                            Some(binding.token.generation()),
                        ),
                        transactions,
                    )
                })
            })
            .collect()
    }

    fn before_effect(
        &self,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        if let Some(effect) = OwnedClusterBarrierEffect::from_effect(effect) {
            return self.before_cluster_barrier_effect(operation, effect);
        }
        if let Some(effect) = OwnedNamedBarrierEffect::from_effect(effect) {
            return self.before_named_barrier_effect(operation, effect);
        }
        let Some(effect) = OwnedSyncEffect::from_effect(effect) else {
            return Ok(());
        };
        let _profile_timer = ProfileTimer::new(ProfileKind::SyncBeforeStrict);
        if matches!(
            effect,
            OwnedSyncEffect::ExpectTx {
                outcome: Some(_),
                ..
            } | OwnedSyncEffect::Arrive {
                outcome: Some(_),
                ..
            } | OwnedSyncEffect::ArriveBatch {
                outcome: Some(_),
                ..
            }
        ) {
            return Err(EngineError::message(format!(
                "synccheck mbarrier operation at {} was staged with a committed outcome",
                operation.id()
            )));
        }
        let mut state = self
            .inner_for_operation(operation.id())
            .lock()
            .expect("sync-check state poisoned");
        if state.staged.contains_key(operation.id()) {
            return Err(EngineError::message(format!(
                "synccheck operation {} already has a staged effect",
                operation.id()
            )));
        }

        let candidate = state.protocol.clone();
        let preview = match effect.apply(&candidate, operation.id()) {
            Ok(preview) => preview,
            Err(error) => {
                let message = format!(
                    "synccheck rejected {} at {}: {error}",
                    effect.kind().name(),
                    operation.id()
                );
                state.findings.push(SyncCheckFinding {
                    operation: operation.id().clone(),
                    effect: effect.kind(),
                    error: error.into(),
                });
                return Err(EngineError::message(message));
            }
        };

        // A blocking numeric wait registers its waker before returning
        // `Pending`, while `after_effect` is not called until that waiter is
        // woken.  Publish the matching strict waiter now so a synchronous
        // arrive (or a completion pump on another worker) can observe it.
        // All other effects remain transactional until `after_effect`.
        let (candidate, revision) = if matches!(
            preview,
            SyncEffectPreview::Wait(SyncCheckWaitState::Registered { .. })
        ) {
            state.protocol = candidate;
            state.revision = state
                .revision
                .checked_add(1)
                .ok_or_else(|| EngineError::message("synccheck protocol revision overflow"))?;
            (None, state.revision)
        } else {
            (Some(candidate), state.revision)
        };
        state.staged.insert(
            operation.id().clone(),
            StagedSyncEffect {
                effect,
                preview,
                candidate,
                revision,
            },
        );
        Ok(())
    }

    fn after_effect(
        &self,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        let numeric_wait = match effect {
            OperationEffect::MbarrierWait { plan, outcome } => Some((plan, outcome)),
            _ => None,
        };
        let tcgen_kind = match effect {
            OperationEffect::TcgenLifecycleRegister(plan) => {
                Some(tcgen_effect_kind(plan.action(), false))
            }
            OperationEffect::TcgenLifecycleResume(plan) => {
                Some(tcgen_effect_kind(plan.plan().action(), true))
            }
            _ => None,
        };
        if let Some(effect) = tcgen_kind {
            let mut state = self
                .inner_for_operation(operation.id())
                .lock()
                .expect("sync-check state poisoned");
            state
                .causality
                .program_tick(operation.id().global_warp_id())
                .map_err(|error| {
                    EngineError::message(format!(
                        "synccheck could not order TCGEN lifecycle operation {}: {error}",
                        operation.id()
                    ))
                })?;
            self.record_effect(
                &mut state,
                operation.id(),
                effect,
                SyncCheckEffectOutcome::Applied,
            );
            return Ok(());
        }
        let setmax_kind = match effect {
            OperationEffect::SetmaxnregRegister(_) => Some(SyncCheckEffectKind::SetmaxnregRegister),
            OperationEffect::SetmaxnregResume(_) => Some(SyncCheckEffectKind::SetmaxnregResume),
            _ => None,
        };
        if let Some(effect) = setmax_kind {
            let mut state = self
                .inner_for_operation(operation.id())
                .lock()
                .expect("sync-check state poisoned");
            state
                .causality
                .program_tick(operation.id().global_warp_id())
                .map_err(|error| {
                    EngineError::message(format!(
                        "synccheck could not order setmaxnreg operation {}: {error}",
                        operation.id()
                    ))
                })?;
            self.record_effect(
                &mut state,
                operation.id(),
                effect,
                SyncCheckEffectOutcome::Applied,
            );
            return Ok(());
        }
        if let OperationEffect::AnalysisGap(gap) = effect {
            if !<SyncCheckMode as crate::engine_mode::EngineModeImpl>::observes_analysis_gap(
                self,
                gap.kind(),
            ) {
                return Ok(());
            }
            let mut state = self
                .inner_for_operation(operation.id())
                .lock()
                .expect("sync-check state poisoned");
            push_unique_incomplete(
                &mut state.incomplete_reasons,
                SyncCheckIncompleteReason::AnalysisGap {
                    operation: operation.id().clone(),
                    kind: gap.kind(),
                },
            );
            return Ok(());
        }
        if let OperationEffect::PhysicalAccess(batch) = effect {
            if batch.atomic_return_sync_relevant()
                && batch.descriptor().kind() == PhysicalAccessKind::AtomicReadModifyWrite
                && batch.has_inter_lane_overlap()
            {
                let mut state = self
                    .inner_for_operation(operation.id())
                    .lock()
                    .expect("sync-check state poisoned");
                push_unique_incomplete(
                    &mut state.incomplete_reasons,
                    SyncCheckIncompleteReason::AnalysisGap {
                        operation: operation.id().clone(),
                        kind: AnalysisGapKind::AtomicLaneSerialization,
                    },
                );
            }
            return Ok(());
        }
        if let Some(effect) = OwnedClusterBarrierEffect::from_effect(effect) {
            return self.after_cluster_barrier_effect(operation, effect);
        }
        if let Some(effect) = OwnedNamedBarrierEffect::from_effect(effect) {
            return self.after_named_barrier_effect(operation, effect);
        }
        let Some(effect) = OwnedSyncEffect::from_effect(effect) else {
            return Ok(());
        };
        let _profile_timer = ProfileTimer::new(ProfileKind::SyncAfterStrict);
        let mut state = self
            .inner_for_operation(operation.id())
            .lock()
            .expect("sync-check state poisoned");
        let mut staged = state.staged.remove(operation.id()).ok_or_else(|| {
            EngineError::message(format!(
                "synccheck operation {} has no staged {} effect",
                operation.id(),
                effect.kind().name()
            ))
        })?;
        if !staged.effect.same_staged_effect(&effect) {
            return Err(EngineError::message(format!(
                "synccheck effect changed between validation and commit at {}",
                operation.id()
            )));
        }
        if staged.candidate.is_some() && state.revision != staged.revision {
            let candidate = state.protocol.clone();
            let preview = match staged.effect.apply(&candidate, operation.id()) {
                Ok(preview) => preview,
                Err(error) => {
                    let message = format!(
                        "synccheck rejected rebased {} at {}: {error}",
                        staged.effect.kind().name(),
                        operation.id(),
                    );
                    state.findings.push(SyncCheckFinding {
                        operation: operation.id().clone(),
                        effect: staged.effect.kind(),
                        error: error.into(),
                    });
                    return Err(EngineError::message(message));
                }
            };
            staged.candidate = match &preview {
                SyncEffectPreview::Wait(SyncCheckWaitState::Registered { .. }) => None,
                _ => Some(candidate),
            };
            staged.preview = preview;
            staged.revision = state.revision;
        }
        match &staged.preview {
            SyncEffectPreview::ExpectTx { expected } => {
                let OwnedSyncEffect::ExpectTx {
                    outcome: Some(actual),
                    ..
                } = &effect
                else {
                    return Err(EngineError::message(format!(
                        "synccheck mbarrier.expect_tx at {} committed without numeric generations",
                        operation.id()
                    )));
                };
                if actual.generations() != expected.as_ref() {
                    return Err(EngineError::message(format!(
                        "synccheck mbarrier.expect_tx at {} disagrees with numeric generations: strict {:?}, numeric {:?}",
                        operation.id(),
                        expected,
                        actual.generations(),
                    )));
                }
            }
            SyncEffectPreview::Arrive { expected, .. } => {
                let OwnedSyncEffect::Arrive {
                    outcome: Some(actual),
                    ..
                } = &effect
                else {
                    return Err(EngineError::message(format!(
                        "synccheck mbarrier arrival at {} committed without a numeric outcome",
                        operation.id()
                    )));
                };
                if !actual.same_primary_transition(*expected) {
                    return Err(EngineError::message(format!(
                        "synccheck mbarrier arrival at {} disagrees with numeric outcome: strict {:?}, numeric {:?}",
                        operation.id(),
                        expected,
                        actual,
                    )));
                }
            }
            SyncEffectPreview::ArriveBatch { expected, .. } => {
                let OwnedSyncEffect::ArriveBatch {
                    outcome: Some(actual),
                    ..
                } = &effect
                else {
                    return Err(EngineError::message(format!(
                        "synccheck mbarrier arrival batch at {} committed without numeric outcomes",
                        operation.id()
                    )));
                };
                if actual.outcomes().len() != expected.len()
                    || actual
                        .outcomes()
                        .iter()
                        .zip(expected.iter())
                        .any(|(actual, expected)| !actual.same_primary_transition(*expected))
                {
                    return Err(EngineError::message(format!(
                        "synccheck mbarrier arrival batch at {} disagrees with numeric outcomes: strict {:?}, numeric {:?}",
                        operation.id(),
                        expected,
                        actual.outcomes(),
                    )));
                }
            }
            _ => {}
        }

        if let Some(candidate) = staged.candidate {
            if let (
                SyncEffectPreview::Wait(SyncCheckWaitState::Ready {
                    generation: Some(completed_generation),
                    ..
                }),
                Some((_, Some(outcome))),
            ) = (&staged.preview, numeric_wait.as_ref())
            {
                let Some(numeric_generation) = outcome.completed_generation() else {
                    return Err(EngineError::message(format!(
                        "synccheck ready mbarrier wait at {} has no numeric completed generation",
                        operation.id()
                    )));
                };
                if numeric_generation != *completed_generation {
                    return Err(EngineError::message(format!(
                        "synccheck ready mbarrier wait at {} disagrees on generation: strict {}, numeric {}",
                        operation.id(),
                        completed_generation,
                        numeric_generation,
                    )));
                }
            }
            let completion_commit = match &staged.preview {
                SyncEffectPreview::CompletionIssue { tokens } => {
                    let OwnedSyncEffect::CompletionIssue { plan, .. } = &staged.effect else {
                        unreachable!("completion preview belongs to a completion issue")
                    };
                    let action_ids = effect.completion_action_ids().ok_or_else(|| {
                        EngineError::message(format!(
                            "synccheck completion issue at {} committed without action IDs",
                            operation.id()
                        ))
                    })?;
                    if action_ids.len() != plan.completions().len()
                        || action_ids.len() != tokens.len()
                    {
                        return Err(EngineError::message(format!(
                            "synccheck completion issue at {} produced {} action IDs for {} targets",
                            operation.id(),
                            action_ids.len(),
                            plan.completions().len()
                        )));
                    }
                    let mut seen = BTreeSet::new();
                    let mut actions = Vec::with_capacity(action_ids.len());
                    let mut bindings = Vec::new();
                    for ((&(barrier_id, transactions), token), &action_id) in plan
                        .completions()
                        .iter()
                        .zip(tokens.iter())
                        .zip(action_ids.iter())
                    {
                        if !seen.insert(action_id)
                            || state.completion_tokens.contains_key(&action_id)
                        {
                            return Err(EngineError::message(format!(
                                "synccheck completion action ID {action_id} was reused"
                            )));
                        }
                        let generation = token.as_ref().map(|token| token.generation());
                        if (transactions == 0) != token.is_none() {
                            return Err(EngineError::message(format!(
                                "synccheck completion token presence disagrees with {transactions} transaction bytes for action {action_id}"
                            )));
                        }
                        actions.push(SyncCheckCompletionIssue {
                            action_id,
                            barrier_id,
                            generation,
                            transactions,
                        });
                        if let Some(token) = token {
                            let causal_token = match state.causality.mbarrier_completion_issue(
                                barrier_id,
                                token.generation(),
                                operation.id().global_warp_id(),
                            ) {
                                Ok(token) => token,
                                Err(error) => {
                                    return Err(record_causality_error(
                                        &mut state,
                                        operation.id(),
                                        SyncCheckEffectKind::MbarrierCompletionIssue,
                                        error,
                                    ));
                                }
                            };
                            bindings.push((
                                action_id,
                                StrictCompletionBinding {
                                    token: token.clone(),
                                    causal_token,
                                    kind: PhysicalCompletionKind::Transaction { transactions },
                                },
                            ));
                        }
                    }
                    Some((
                        SyncCheckEffectKind::MbarrierCompletionIssue,
                        actions.into_boxed_slice(),
                        bindings,
                    ))
                }
                SyncEffectPreview::CpAsyncMbarrierArrive { tokens } => {
                    let OwnedSyncEffect::CpAsyncMbarrierArrive {
                        plan,
                        actions: Some(actions),
                    } = &effect
                    else {
                        return Err(EngineError::message(format!(
                            "synccheck cp.async.mbarrier.arrive at {} committed without actions",
                            operation.id()
                        )));
                    };
                    let targets = plan.targets().collect::<Vec<_>>();
                    if actions.len() != targets.len() || actions.len() != tokens.len() {
                        return Err(EngineError::message(format!(
                            "synccheck cp.async.mbarrier.arrive at {} produced {} actions for {} lanes",
                            operation.id(),
                            actions.len(),
                            targets.len()
                        )));
                    }
                    let mut seen = BTreeSet::new();
                    let mut issues = Vec::with_capacity(actions.len());
                    let mut bindings = Vec::with_capacity(actions.len());
                    for (((_, barrier_id), token), &action) in targets
                        .iter()
                        .copied()
                        .zip(tokens.iter())
                        .zip(actions.iter())
                    {
                        if !seen.insert(action.id())
                            || state.completion_tokens.contains_key(&action.id())
                            || action.barrier_id() != barrier_id
                            || action.generation() != token.generation()
                            || action.arrival() != Some((plan.warp_id(), 1))
                        {
                            return Err(EngineError::message(format!(
                                "synccheck cp.async.mbarrier.arrive action at {} disagrees with its exact lane target",
                                operation.id()
                            )));
                        }
                        issues.push(SyncCheckCompletionIssue {
                            action_id: action.id(),
                            barrier_id,
                            generation: Some(action.generation()),
                            transactions: 0,
                        });
                        let causal_token = match state.causality.mbarrier_completion_issue(
                            barrier_id,
                            token.generation(),
                            operation.id().global_warp_id(),
                        ) {
                            Ok(token) => token,
                            Err(error) => {
                                return Err(record_causality_error(
                                    &mut state,
                                    operation.id(),
                                    SyncCheckEffectKind::CpAsyncMbarrierArrive,
                                    error,
                                ));
                            }
                        };
                        bindings.push((
                            action.id(),
                            StrictCompletionBinding {
                                token: token.clone(),
                                causal_token,
                                kind: action.kind(),
                            },
                        ));
                    }
                    Some((
                        SyncCheckEffectKind::CpAsyncMbarrierArrive,
                        issues.into_boxed_slice(),
                        bindings,
                    ))
                }
                SyncEffectPreview::TcgenCommitIssue { tokens } => {
                    let OwnedSyncEffect::TcgenCommitIssue {
                        plan,
                        actions: Some(actions),
                    } = &effect
                    else {
                        return Err(EngineError::message(format!(
                            "synccheck tcgen05.commit issue at {} committed without actions",
                            operation.id()
                        )));
                    };
                    if actions.len() != plan.barrier_ids().len() || actions.len() != tokens.len() {
                        return Err(EngineError::message(format!(
                            "synccheck tcgen05.commit issue at {} produced {} actions for {} targets",
                            operation.id(),
                            actions.len(),
                            plan.barrier_ids().len()
                        )));
                    }
                    let mut seen = BTreeSet::new();
                    let mut issues = Vec::with_capacity(actions.len());
                    let mut bindings = Vec::with_capacity(actions.len());
                    for ((&barrier_id, token), &action) in plan
                        .barrier_ids()
                        .iter()
                        .zip(tokens.iter())
                        .zip(actions.iter())
                    {
                        if !seen.insert(action.id())
                            || state.completion_tokens.contains_key(&action.id())
                            || action.barrier_id() != barrier_id
                            || action.generation() != token.generation()
                            || action.arrival() != Some((plan.warp_id(), plan.arrival_count()))
                        {
                            return Err(EngineError::message(format!(
                                "synccheck tcgen05.commit action at {} disagrees with its exact issue plan",
                                operation.id()
                            )));
                        }
                        issues.push(SyncCheckCompletionIssue {
                            action_id: action.id(),
                            barrier_id: action.barrier_id(),
                            generation: Some(action.generation()),
                            transactions: 0,
                        });
                        let causal_token = match state.causality.mbarrier_completion_issue(
                            barrier_id,
                            token.generation(),
                            operation.id().global_warp_id(),
                        ) {
                            Ok(token) => token,
                            Err(error) => {
                                return Err(record_causality_error(
                                    &mut state,
                                    operation.id(),
                                    SyncCheckEffectKind::TcgenCommitIssue,
                                    error,
                                ));
                            }
                        };
                        bindings.push((
                            action.id(),
                            StrictCompletionBinding {
                                token: token.clone(),
                                causal_token,
                                kind: action.kind(),
                            },
                        ));
                    }
                    Some((
                        SyncCheckEffectKind::TcgenCommitIssue,
                        issues.into_boxed_slice(),
                        bindings,
                    ))
                }
                SyncEffectPreview::Applied
                | SyncEffectPreview::ExpectTx { .. }
                | SyncEffectPreview::Arrive { .. }
                | SyncEffectPreview::ArriveBatch { .. }
                | SyncEffectPreview::Wait(_) => None,
            };
            let next_revision = state
                .revision
                .checked_add(1)
                .ok_or_else(|| EngineError::message("synccheck protocol revision overflow"))?;
            state.protocol = candidate;
            state.revision = next_revision;
            let causality = {
                let _profile_timer = ProfileTimer::new(ProfileKind::SyncApplyCausality);
                apply_mbarrier_commit_causality(&mut state, operation, &effect, &staged.preview)
            };
            if let Err(error) = causality {
                return Err(record_causality_error(
                    &mut state,
                    operation.id(),
                    effect.kind(),
                    error,
                ));
            }
            if let Some((effect_kind, actions, bindings)) = completion_commit {
                state.completion_tokens.extend(bindings);
                self.record_effect(
                    &mut state,
                    operation.id(),
                    effect_kind,
                    SyncCheckEffectOutcome::MbarrierCompletionIssue { actions },
                );
            } else {
                self.record_effect(
                    &mut state,
                    operation.id(),
                    effect.kind(),
                    outcome_from_preview(staged.preview, None),
                );
            }
            return Ok(());
        }

        let SyncEffectPreview::Wait(initial_wait) = staged.preview else {
            return Err(EngineError::message(format!(
                "synccheck staged effect at {} lost its commit candidate",
                operation.id()
            )));
        };
        let registered_generation = match &initial_wait {
            SyncCheckWaitState::Registered { generation } => *generation,
            SyncCheckWaitState::Ready { .. } => {
                unreachable!("only a registered wait is committed after numeric suspension")
            }
        };
        let OwnedSyncEffect::Wait {
            plan: wait_plan, ..
        } = &effect
        else {
            unreachable!("registered wait commit belongs to a wait effect")
        };
        let snapshot = state.protocol.snapshot(wait_plan.barrier_id());
        if snapshot.last_completed_generation() != Some(registered_generation)
            || snapshot.last_completed_phase() != wait_plan.requested_phase()
        {
            push_unique_incomplete(
                &mut state.incomplete_reasons,
                SyncCheckIncompleteReason::CompletionTransitionUnobserved {
                    operation: operation.id().clone(),
                    barrier_id: wait_plan.barrier_id(),
                    generation: registered_generation,
                },
            );
            return Err(EngineError::message(format!(
                "synccheck numeric wait at {} succeeded without a typed completion for generation {registered_generation}",
                operation.id()
            )));
        }
        let candidate = state.protocol.clone();
        let committed = match staged.effect.apply(&candidate, operation.id()) {
            Ok(SyncEffectPreview::Wait(wait)) => wait,
            Ok(_) => unreachable!("registered wait replays as a wait"),
            Err(error) => {
                let message = format!(
                    "synccheck could not commit successful wait at {}: {error}",
                    operation.id()
                );
                state.findings.push(SyncCheckFinding {
                    operation: operation.id().clone(),
                    effect: effect.kind(),
                    error: error.into(),
                });
                return Err(EngineError::message(message));
            }
        };
        let finalized_wait = match committed {
            SyncCheckWaitState::Ready {
                generation: Some(completed_generation),
                ..
            } => {
                let Some((plan, Some(outcome))) = numeric_wait else {
                    return Err(EngineError::message(format!(
                        "synccheck blocking mbarrier wait at {} resumed without a numeric outcome",
                        operation.id()
                    )));
                };
                let Some(numeric_generation) = outcome.completed_generation() else {
                    return Err(EngineError::message(format!(
                        "synccheck blocking mbarrier wait at {} resumed without a completed numeric generation",
                        operation.id()
                    )));
                };
                if registered_generation != completed_generation
                    || numeric_generation != completed_generation
                {
                    return Err(EngineError::message(format!(
                        "synccheck blocking mbarrier wait at {} disagrees on generation: registered {}, strict completion {}, numeric completion {}",
                        operation.id(),
                        registered_generation,
                        completed_generation,
                        numeric_generation,
                    )));
                }
                state.protocol = candidate;
                state.revision = state
                    .revision
                    .checked_add(1)
                    .ok_or_else(|| EngineError::message("synccheck protocol revision overflow"))?;
                if let Err(error) = apply_mbarrier_wait_causality(
                    &mut state,
                    operation.id(),
                    plan,
                    completed_generation,
                    true,
                ) {
                    return Err(record_causality_error(
                        &mut state,
                        operation.id(),
                        SyncCheckEffectKind::MbarrierWait,
                        error,
                    ));
                }
                Some(committed)
            }
            SyncCheckWaitState::Ready {
                generation: None, ..
            } => {
                return Err(EngineError::message(format!(
                    "synccheck blocking mbarrier wait at {} resumed without a strict completed generation",
                    operation.id()
                )));
            }
            SyncCheckWaitState::Registered { generation } => {
                let OwnedSyncEffect::Wait { plan, .. } = &effect else {
                    unreachable!("registered wait commit belongs to a wait effect")
                };
                push_unique_incomplete(
                    &mut state.incomplete_reasons,
                    SyncCheckIncompleteReason::CompletionTransitionUnobserved {
                        operation: operation.id().clone(),
                        barrier_id: plan.barrier_id(),
                        generation,
                    },
                );
                return Err(EngineError::message(format!(
                    "synccheck numeric wait at {} succeeded without a typed completion for generation {generation}",
                    operation.id()
                )));
            }
        };
        self.record_effect(
            &mut state,
            operation.id(),
            effect.kind(),
            SyncCheckEffectOutcome::MbarrierWait {
                staged: initial_wait,
                committed: finalized_wait,
            },
        );
        Ok(())
    }

    fn before_named_barrier_effect(
        &self,
        operation: &OperationContext,
        effect: OwnedNamedBarrierEffect,
    ) -> Result<(), EngineError> {
        let _profile_timer = ProfileTimer::new(ProfileKind::SyncBeforeNamed);
        if self.records_only_fixed_sync_transitions {
            let control_check = match &effect {
                OwnedNamedBarrierEffect::Arrive {
                    plan,
                    outcome: None,
                } => Some((
                    plan.barrier_id(),
                    StrictNamedBarrierOperation::Arrive,
                    SyncCheckEffectKind::NamedBarrierArrive,
                )),
                OwnedNamedBarrierEffect::Register {
                    plan,
                    outcome: None,
                } => plan.aligned().then_some((
                    plan.barrier_id(),
                    StrictNamedBarrierOperation::Sync,
                    SyncCheckEffectKind::NamedBarrierSyncRegister,
                )),
                OwnedNamedBarrierEffect::Resume(_) => None,
                OwnedNamedBarrierEffect::Arrive {
                    outcome: Some(_), ..
                }
                | OwnedNamedBarrierEffect::Register {
                    outcome: Some(_), ..
                } => {
                    return Err(EngineError::message(format!(
                        "synccheck {} at {} was staged with a committed outcome",
                        effect.kind().name(),
                        operation.id()
                    )));
                }
            };
            if let Some((barrier_id, kind, effect_kind)) = control_check {
                if let Err(error) = crate::strict_named_barrier::validate_named_barrier_control(
                    operation, barrier_id, kind,
                ) {
                    let mut state = self
                        .inner_for_operation(operation.id())
                        .lock()
                        .expect("sync-check state poisoned");
                    let message = format!(
                        "synccheck rejected {} at {}: {error}",
                        effect_kind.name(),
                        operation.id()
                    );
                    state.findings.push(SyncCheckFinding {
                        operation: operation.id().clone(),
                        effect: effect_kind,
                        error: error.into(),
                    });
                    return Err(EngineError::message(message));
                }
            }
            return Ok(());
        }
        let mut state = self
            .inner_for_operation(operation.id())
            .lock()
            .expect("sync-check state poisoned");
        if state.staged_named.contains_key(operation.id())
            || state.staged.contains_key(operation.id())
        {
            return Err(EngineError::message(format!(
                "synccheck operation {} already has a staged effect",
                operation.id()
            )));
        }

        let control_check = match &effect {
            OwnedNamedBarrierEffect::Arrive { plan, .. } => Some((
                plan.barrier_id(),
                StrictNamedBarrierOperation::Arrive,
                SyncCheckEffectKind::NamedBarrierArrive,
            )),
            OwnedNamedBarrierEffect::Register { plan, .. } => plan.aligned().then_some((
                plan.barrier_id(),
                StrictNamedBarrierOperation::Sync,
                SyncCheckEffectKind::NamedBarrierSyncRegister,
            )),
            OwnedNamedBarrierEffect::Resume(_) => None,
        };
        if let Some((barrier_id, kind, effect_kind)) = control_check {
            if let Err(error) = crate::strict_named_barrier::validate_named_barrier_control(
                operation, barrier_id, kind,
            ) {
                let message = format!(
                    "synccheck rejected {} at {}: {error}",
                    effect_kind.name(),
                    operation.id()
                );
                state.findings.push(SyncCheckFinding {
                    operation: operation.id().clone(),
                    effect: effect_kind,
                    error: error.into(),
                });
                return Err(EngineError::message(message));
            }
        }

        let staged = match effect {
            OwnedNamedBarrierEffect::Arrive {
                plan,
                outcome: None,
            } => {
                let candidate = state.named_protocol.clone();
                let outcome = match candidate.arrive(
                    plan.barrier_id(),
                    plan.expected_arrivals(),
                    plan.warp_id(),
                    plan.arrival_mask(),
                    Some(operation.id().clone()),
                ) {
                    Ok(outcome) => outcome,
                    Err(error) => {
                        let message = format!(
                            "synccheck rejected bar.arrive registration at {}: {error}",
                            operation.id()
                        );
                        state.findings.push(SyncCheckFinding {
                            operation: operation.id().clone(),
                            effect: SyncCheckEffectKind::NamedBarrierArrive,
                            error: error.into(),
                        });
                        return Err(EngineError::message(message));
                    }
                };
                let StrictNamedBarrierOutcome::Arrived {
                    generation,
                    completed,
                    ..
                } = outcome
                else {
                    unreachable!("bar.arrive strict registration must produce an arrival")
                };
                StagedNamedBarrierEffect {
                    effect: OwnedNamedBarrierEffect::Arrive {
                        plan,
                        outcome: None,
                    },
                    candidate: Some(candidate),
                    expected_outcome: Some(ExpectedNamedBarrierOutcome::Arrive(
                        NamedBarrierArrivalOutcome::new(generation, completed),
                    )),
                    revision: state.named_revision,
                }
            }
            OwnedNamedBarrierEffect::Arrive {
                outcome: Some(_), ..
            } => {
                return Err(EngineError::message(format!(
                    "synccheck bar.arrive registration at {} was staged with a committed outcome",
                    operation.id()
                )));
            }
            OwnedNamedBarrierEffect::Register {
                plan,
                outcome: None,
            } => {
                let candidate = state.named_protocol.clone();
                let outcome = match candidate.sync_with_alignment(
                    plan.barrier_id(),
                    plan.expected_arrivals(),
                    plan.warp_id(),
                    plan.arrival_mask(),
                    plan.aligned(),
                    Some(operation.id().clone()),
                ) {
                    Ok(outcome) => outcome,
                    Err(error) => {
                        let message = format!(
                            "synccheck rejected bar.sync registration at {}: {error}",
                            operation.id()
                        );
                        state.findings.push(SyncCheckFinding {
                            operation: operation.id().clone(),
                            effect: SyncCheckEffectKind::NamedBarrierSyncRegister,
                            error: error.into(),
                        });
                        return Err(EngineError::message(message));
                    }
                };
                let generation = outcome.generation();
                let completed_now = match outcome {
                    StrictNamedBarrierOutcome::Ready { completed_now, .. } => completed_now,
                    StrictNamedBarrierOutcome::Registered { .. } => false,
                    StrictNamedBarrierOutcome::Arrived { .. } => {
                        unreachable!("bar.sync strict registration cannot produce bar.arrive")
                    }
                };
                StagedNamedBarrierEffect {
                    effect: OwnedNamedBarrierEffect::Register {
                        plan,
                        outcome: None,
                    },
                    candidate: Some(candidate),
                    expected_outcome: Some(ExpectedNamedBarrierOutcome::Sync(
                        NamedBarrierSyncRegistrationOutcome::new(generation, completed_now),
                    )),
                    revision: state.named_revision,
                }
            }
            OwnedNamedBarrierEffect::Register {
                outcome: Some(_), ..
            } => {
                return Err(EngineError::message(format!(
                    "synccheck bar.sync registration at {} was staged with a committed outcome",
                    operation.id()
                )));
            }
            OwnedNamedBarrierEffect::Resume(plan) => {
                let candidate = state.named_protocol.clone();
                if let Err(error) = candidate.resume_with_alignment(
                    plan.barrier_id(),
                    plan.warp_id(),
                    plan.arrival_mask(),
                    plan.generation(),
                    plan.plan().aligned(),
                    Some(operation.id().clone()),
                ) {
                    let message = format!(
                        "synccheck rejected bar.sync resume at {}: {error}",
                        operation.id()
                    );
                    state.findings.push(SyncCheckFinding {
                        operation: operation.id().clone(),
                        effect: SyncCheckEffectKind::NamedBarrierSyncResume,
                        error: error.into(),
                    });
                    return Err(EngineError::message(message));
                }
                StagedNamedBarrierEffect {
                    effect: OwnedNamedBarrierEffect::Resume(plan),
                    candidate: Some(candidate),
                    expected_outcome: None,
                    revision: state.named_revision,
                }
            }
        };
        state.staged_named.insert(operation.id().clone(), staged);
        Ok(())
    }

    fn after_named_barrier_effect(
        &self,
        operation: &OperationContext,
        effect: OwnedNamedBarrierEffect,
    ) -> Result<(), EngineError> {
        let _profile_timer = ProfileTimer::new(ProfileKind::SyncAfterNamed);
        if self.records_only_fixed_sync_transitions {
            let record_outcome = match &effect {
                OwnedNamedBarrierEffect::Arrive {
                    outcome: Some(outcome),
                    ..
                } => SyncCheckEffectOutcome::NamedBarrierArrive {
                    generation: outcome.generation(),
                    completed_now: outcome.completed_now(),
                },
                OwnedNamedBarrierEffect::Register {
                    outcome: Some(_), ..
                }
                | OwnedNamedBarrierEffect::Resume(_) => SyncCheckEffectOutcome::Applied,
                OwnedNamedBarrierEffect::Arrive { outcome: None, .. }
                | OwnedNamedBarrierEffect::Register { outcome: None, .. } => {
                    return Err(EngineError::message(format!(
                        "synccheck {} at {} committed without a numeric outcome",
                        effect.kind().name(),
                        operation.id()
                    )));
                }
            };
            let mut state = self
                .inner_for_operation(operation.id())
                .lock()
                .expect("sync-check state poisoned");
            let causality = {
                let _profile_timer = ProfileTimer::new(ProfileKind::SyncApplyCausality);
                apply_named_barrier_causality(&mut state, &effect)
            };
            if let Err(error) = causality {
                return Err(record_causality_error(
                    &mut state,
                    operation.id(),
                    effect.kind(),
                    error,
                ));
            }
            self.record_effect(&mut state, operation.id(), effect.kind(), record_outcome);
            return Ok(());
        }
        let mut state = self
            .inner_for_operation(operation.id())
            .lock()
            .expect("sync-check state poisoned");
        let mut staged = state.staged_named.remove(operation.id()).ok_or_else(|| {
            EngineError::message(format!(
                "synccheck operation {} has no staged {} effect",
                operation.id(),
                effect.kind().name(),
            ))
        })?;
        if state.named_revision != staged.revision {
            let candidate = state.named_protocol.clone();
            let expected_outcome = match &staged.effect {
                OwnedNamedBarrierEffect::Arrive {
                    plan,
                    outcome: None,
                } => candidate
                    .arrive(
                        plan.barrier_id(),
                        plan.expected_arrivals(),
                        plan.warp_id(),
                        plan.arrival_mask(),
                        Some(operation.id().clone()),
                    )
                    .map(|outcome| {
                        let StrictNamedBarrierOutcome::Arrived {
                            generation,
                            completed,
                            ..
                        } = outcome
                        else {
                            unreachable!("bar.arrive rebase must produce an arrival")
                        };
                        Some(ExpectedNamedBarrierOutcome::Arrive(
                            NamedBarrierArrivalOutcome::new(generation, completed),
                        ))
                    }),
                OwnedNamedBarrierEffect::Register {
                    plan,
                    outcome: None,
                } => candidate
                    .sync_with_alignment(
                        plan.barrier_id(),
                        plan.expected_arrivals(),
                        plan.warp_id(),
                        plan.arrival_mask(),
                        plan.aligned(),
                        Some(operation.id().clone()),
                    )
                    .map(|outcome| {
                        let generation = outcome.generation();
                        let completed_now = match outcome {
                            StrictNamedBarrierOutcome::Ready { completed_now, .. } => completed_now,
                            StrictNamedBarrierOutcome::Registered { .. } => false,
                            StrictNamedBarrierOutcome::Arrived { .. } => {
                                unreachable!("bar.sync rebase cannot produce bar.arrive")
                            }
                        };
                        Some(ExpectedNamedBarrierOutcome::Sync(
                            NamedBarrierSyncRegistrationOutcome::new(generation, completed_now),
                        ))
                    }),
                OwnedNamedBarrierEffect::Resume(plan) => candidate
                    .resume_with_alignment(
                        plan.barrier_id(),
                        plan.warp_id(),
                        plan.arrival_mask(),
                        plan.generation(),
                        plan.plan().aligned(),
                        Some(operation.id().clone()),
                    )
                    .map(|()| None),
                OwnedNamedBarrierEffect::Arrive {
                    outcome: Some(_), ..
                }
                | OwnedNamedBarrierEffect::Register {
                    outcome: Some(_), ..
                } => unreachable!("staged named-barrier effects have no numeric outcome"),
            };
            staged.expected_outcome = match expected_outcome {
                Ok(outcome) => outcome,
                Err(error) => {
                    let message = format!(
                        "synccheck rejected rebased {} at {}: {error}",
                        staged.effect.kind().name(),
                        operation.id(),
                    );
                    state.findings.push(SyncCheckFinding {
                        operation: operation.id().clone(),
                        effect: staged.effect.kind(),
                        error: error.into(),
                    });
                    return Err(EngineError::message(message));
                }
            };
            staged.candidate = Some(candidate);
            staged.revision = state.named_revision;
        }
        let record_outcome = match (&staged.effect, &effect) {
            (
                OwnedNamedBarrierEffect::Arrive {
                    plan: staged_plan, ..
                },
                OwnedNamedBarrierEffect::Arrive {
                    plan: committed_plan,
                    outcome: Some(outcome),
                },
            ) if staged_plan == committed_plan => {
                let ExpectedNamedBarrierOutcome::Arrive(expected) = staged
                    .expected_outcome
                    .expect("named barrier arrival has an expected outcome")
                else {
                    unreachable!("bar.arrive staged an incompatible expected outcome")
                };
                if expected != *outcome {
                    return Err(EngineError::message(format!(
                        "synccheck bar.arrive registration at {} disagrees with numeric outcome: strict {:?}, numeric {:?}",
                        operation.id(),
                        expected,
                        outcome,
                    )));
                }
                SyncCheckEffectOutcome::NamedBarrierArrive {
                    generation: outcome.generation(),
                    completed_now: outcome.completed_now(),
                }
            }
            (
                OwnedNamedBarrierEffect::Register {
                    plan: staged_plan, ..
                },
                OwnedNamedBarrierEffect::Register {
                    plan: committed_plan,
                    outcome: Some(outcome),
                },
            ) if staged_plan == committed_plan => {
                let ExpectedNamedBarrierOutcome::Sync(expected) = staged
                    .expected_outcome
                    .expect("named barrier sync registration has an expected outcome")
                else {
                    unreachable!("bar.sync staged an incompatible expected outcome")
                };
                if expected != *outcome {
                    return Err(EngineError::message(format!(
                        "synccheck bar.sync registration at {} disagrees with numeric outcome: strict {:?}, numeric {:?}",
                        operation.id(),
                        expected,
                        outcome,
                    )));
                }
                SyncCheckEffectOutcome::Applied
            }
            (
                OwnedNamedBarrierEffect::Resume(staged_plan),
                OwnedNamedBarrierEffect::Resume(committed_plan),
            ) if staged_plan == committed_plan => SyncCheckEffectOutcome::Applied,
            _ => {
                return Err(EngineError::message(format!(
                    "synccheck named-barrier effect changed between validation and commit at {}",
                    operation.id()
                )));
            }
        };
        state.named_protocol = staged
            .candidate
            .expect("named barrier effect has a commit candidate");
        state.named_revision = state.named_revision.checked_add(1).ok_or_else(|| {
            EngineError::message("synccheck named-barrier protocol revision overflow")
        })?;
        let causality = {
            let _profile_timer = ProfileTimer::new(ProfileKind::SyncApplyCausality);
            apply_named_barrier_causality(&mut state, &effect)
        };
        if let Err(error) = causality {
            return Err(record_causality_error(
                &mut state,
                operation.id(),
                effect.kind(),
                error,
            ));
        }
        self.record_effect(&mut state, operation.id(), effect.kind(), record_outcome);
        Ok(())
    }

    fn before_cluster_barrier_effect(
        &self,
        operation: &OperationContext,
        effect: OwnedClusterBarrierEffect,
    ) -> Result<(), EngineError> {
        let _profile_timer = ProfileTimer::new(ProfileKind::SyncBeforeCluster);
        let barrier_id = effect.barrier_id();
        let mut state = self
            .inner_for_cluster_barrier(barrier_id)
            .lock()
            .expect("sync-check state poisoned");
        if state.staged_cluster.contains_key(operation.id())
            || state.staged_named.contains_key(operation.id())
            || state.staged.contains_key(operation.id())
        {
            return Err(EngineError::message(format!(
                "synccheck operation {} already has a staged effect",
                operation.id()
            )));
        }
        let candidate = state.cluster_protocol.clone();
        let strict_result: Result<Option<StrictClusterBarrierOutcome>, StrictClusterBarrierError> =
            match &effect {
                OwnedClusterBarrierEffect::Arrive {
                    plan,
                    outcome: None,
                } => candidate
                    .arrive(
                        plan.barrier_id(),
                        plan.participant_warps(),
                        plan.warp_id(),
                        plan.arrival_mask(),
                        Some(operation.id().clone()),
                    )
                    .map(Some),
                OwnedClusterBarrierEffect::WaitRegister {
                    plan,
                    outcome: None,
                } => candidate
                    .wait_register(
                        plan.barrier_id(),
                        plan.participant_warps(),
                        plan.warp_id(),
                        plan.arrival_mask(),
                        Some(operation.id().clone()),
                    )
                    .map(Some),
                OwnedClusterBarrierEffect::WaitResume(plan) => candidate
                    .resume(
                        plan.plan().barrier_id(),
                        plan.generation(),
                        plan.plan().warp_id(),
                        Some(operation.id().clone()),
                    )
                    .map(|()| None),
                _ => {
                    return Err(EngineError::message(format!(
                        "synccheck {} at {} was staged with a committed outcome",
                        effect.kind().name(),
                        operation.id()
                    )));
                }
            };
        let expected_outcome = match strict_result {
            Ok(outcome) => outcome,
            Err(error) => {
                let message = format!(
                    "synccheck rejected {} at {}: {error}",
                    effect.kind().name(),
                    operation.id()
                );
                state.findings.push(SyncCheckFinding {
                    operation: operation.id().clone(),
                    effect: effect.kind(),
                    error: error.into(),
                });
                return Err(EngineError::message(message));
            }
        };
        let rearrival_without_wait =
            expected_outcome.is_some_and(StrictClusterBarrierOutcome::rearrival_without_wait);
        let revision = state.cluster_revision;
        state.staged_cluster.insert(
            operation.id().clone(),
            StagedClusterBarrierEffect {
                effect,
                candidate: Some(candidate),
                expected_outcome,
                rearrival_without_wait,
                revision,
            },
        );
        Ok(())
    }

    fn after_cluster_barrier_effect(
        &self,
        operation: &OperationContext,
        effect: OwnedClusterBarrierEffect,
    ) -> Result<(), EngineError> {
        let _profile_timer = ProfileTimer::new(ProfileKind::SyncAfterCluster);
        let barrier_id = effect.barrier_id();
        let mut state = self
            .inner_for_cluster_barrier(barrier_id)
            .lock()
            .expect("sync-check state poisoned");
        let staged = state.staged_cluster.remove(operation.id()).ok_or_else(|| {
            EngineError::message(format!(
                "synccheck operation {} has no staged {} effect",
                operation.id(),
                effect.kind().name(),
            ))
        })?;
        let record_outcome = match (&staged.effect, &effect) {
            (
                OwnedClusterBarrierEffect::Arrive {
                    plan: staged_plan, ..
                },
                OwnedClusterBarrierEffect::Arrive {
                    plan: committed_plan,
                    outcome: Some(outcome),
                },
            ) if staged_plan == committed_plan => {
                compare_cluster_outcome(
                    operation,
                    staged.expected_outcome,
                    outcome.generation(),
                    outcome.completed_now(),
                )?;
                SyncCheckEffectOutcome::ClusterBarrier {
                    generation: outcome.generation(),
                    completed_now: outcome.completed_now(),
                }
            }
            (
                OwnedClusterBarrierEffect::WaitRegister {
                    plan: staged_plan, ..
                },
                OwnedClusterBarrierEffect::WaitRegister {
                    plan: committed_plan,
                    outcome: Some(outcome),
                },
            ) if staged_plan == committed_plan => {
                compare_cluster_outcome(
                    operation,
                    staged.expected_outcome,
                    outcome.generation(),
                    outcome.completed_now(),
                )?;
                SyncCheckEffectOutcome::ClusterBarrier {
                    generation: outcome.generation(),
                    completed_now: outcome.completed_now(),
                }
            }
            (
                OwnedClusterBarrierEffect::WaitResume(staged_plan),
                OwnedClusterBarrierEffect::WaitResume(committed_plan),
            ) if staged_plan == committed_plan => SyncCheckEffectOutcome::Applied,
            _ => {
                return Err(EngineError::message(format!(
                    "synccheck cluster-barrier effect changed between validation and commit at {}",
                    operation.id()
                )));
            }
        };
        if state.cluster_revision != staged.revision {
            return Err(EngineError::message(format!(
                "synccheck cluster-barrier protocol changed between validation and commit at {}",
                operation.id()
            )));
        }
        state.cluster_protocol = staged
            .candidate
            .expect("cluster barrier effect has a commit candidate");
        state.cluster_revision = state.cluster_revision.checked_add(1).ok_or_else(|| {
            EngineError::message("synccheck cluster-barrier protocol revision overflow")
        })?;
        let causality = {
            let _profile_timer = ProfileTimer::new(ProfileKind::SyncApplyCausality);
            apply_cluster_barrier_causality(&mut state, &effect)
        };
        if let Err(error) = causality {
            return Err(record_causality_error(
                &mut state,
                operation.id(),
                effect.kind(),
                error,
            ));
        }
        if effect.has_unmodeled_unaligned_participation() {
            push_unique_incomplete(
                &mut state.incomplete_reasons,
                SyncCheckIncompleteReason::ClusterBarrierUnalignedUnmodeled {
                    operation: operation.id().clone(),
                    effect: effect.kind(),
                },
            );
        }
        if staged.rearrival_without_wait {
            let generation = staged
                .expected_outcome
                .expect("cluster rearrival has a strict outcome")
                .generation();
            push_unique_incomplete(
                &mut state.incomplete_reasons,
                SyncCheckIncompleteReason::ClusterBarrierRearrivalWithoutWaitUnmodeled {
                    operation: operation.id().clone(),
                    barrier_id: effect.barrier_id(),
                    generation,
                    warp_id: effect.warp_id(),
                },
            );
        }
        self.record_effect(&mut state, operation.id(), effect.kind(), record_outcome);
        Ok(())
    }

    /// Apply one `fence.mbarrier_init` over the barrier set the engine published.
    ///
    /// `published` is the engine's own coverage set. The causality tracker still
    /// selects its own set here and the two are compared, so a divergence fails
    /// loud instead of silently moving a recorded transition. Once a corpus soak
    /// has shown they never differ, the tracker-side selection can go and
    /// `published` becomes the only source.
    fn apply_mbarrier_init_fence(
        &self,
        operation: &OperationContext,
        published: &[PhysicalBarrierId],
    ) -> Result<(), EngineError> {
        let mut state = self
            .inner_for_operation(operation.id())
            .lock()
            .expect("sync-check state poisoned");
        let (fence_clock, barrier_ids) = state
            .causality
            .mbarrier_init_fence(operation.id().global_warp_id(), operation.active_mask())
            .map_err(|error| {
                EngineError::message(format!(
                    "synccheck could not record mbarrier-init fence at {}: {error}",
                    operation.id()
                ))
            })?;
        if barrier_ids != published {
            return Err(EngineError::message(format!(
                "synccheck mbarrier-init fence at {} covers {barrier_ids:?}, but the engine published {published:?}",
                operation.id()
            )));
        }
        if !barrier_ids.is_empty() {
            state
                .pending_mbarrier_init_fences
                .insert(operation.id().global_warp_id());
        }
        let changed = state
            .protocol
            .mark_init_fenced_many(&barrier_ids, Some(operation.id().clone()));
        if changed != 0 {
            state.revision = state
                .revision
                .checked_add(1)
                .ok_or_else(|| EngineError::message("synccheck protocol revision overflow"))?;
        }
        drop(state);

        if self.records_resolved_transitions() && !barrier_ids.is_empty() {
            self.transition_log()
                .register_operation(
                    operation.id().clone(),
                    ResolvedTransitionSummary::Synchronization(
                        ResolvedSynchronizationEffect::from_mbarrier_init_fence(&barrier_ids),
                    ),
                )
                .map_err(|error| {
                    EngineError::message(format!(
                        "synccheck could not record resolved mbarrier-init fence at {}: {error}",
                        operation.id()
                    ))
                })?;
            self.transition_log()
                .register_operation_clock(operation.id().clone(), fence_clock)
                .map_err(|error| {
                    EngineError::message(format!(
                        "synccheck could not record mbarrier-init fence clock at {}: {error}",
                        operation.id()
                    ))
                })?;
        }
        Ok(())
    }

    fn prepare_completion(
        state: &mut SyncCheckState,
        action: PhysicalCompletionAction,
    ) -> Result<PreparedMbarrierCompletion, EngineError> {
        let _profile = ProfileTimer::new(ProfileKind::SyncPrepareCompletion);
        let binding = state
            .completion_tokens
            .get(&action.id())
            .cloned()
            .ok_or_else(|| {
                EngineError::message(format!(
                    "synccheck observed untracked physical completion action {}",
                    action.id()
                ))
            })?;
        let token = &binding.token;
        let issue_operation = token.issue_witness().cloned().ok_or_else(|| {
            EngineError::message(format!(
                "synccheck completion action {} has no issue witness",
                action.id()
            ))
        })?;
        if token.barrier_id() != action.barrier_id()
            || token.generation() != action.generation()
            || binding.kind != action.kind()
        {
            return Err(EngineError::message(format!(
                "synccheck completion action {} identity disagrees with its strict token",
                action.id()
            )));
        }

        let candidate = state.protocol.clone();
        let staged_waits = state
            .staged
            .iter()
            .filter_map(|(operation, staged)| {
                let SyncEffectPreview::Wait(SyncCheckWaitState::Registered { generation }) =
                    &staged.preview
                else {
                    return None;
                };
                let OwnedSyncEffect::Wait { plan, .. } = &staged.effect else {
                    return None;
                };
                (plan.barrier_id() == action.barrier_id() && *generation == action.generation())
                    .then(|| (operation.clone(), *plan, *generation))
            })
            .collect::<Vec<_>>();
        for (wait_operation, plan, generation) in staged_waits {
            let already_registered = candidate
                .snapshot(plan.barrier_id())
                .waiting_warps()
                .iter()
                .any(|waiter| {
                    waiter.warp_id() == plan.warp_id() && waiter.generation() == generation
                });
            if !already_registered {
                let registered = candidate
                    .wait(
                        plan.barrier_id(),
                        plan.requested_phase(),
                        plan.warp_id(),
                        Some(wait_operation),
                    )
                    .map_err(|error| {
                        EngineError::message(format!(
                            "synccheck could not stage numeric waiter for completion action {}: {error}",
                            action.id()
                        ))
                    })?;
                if !matches!(
                    registered,
                    StrictMbarrierWaitOutcome::Registered {
                        generation: registered_generation
                    } if registered_generation == generation
                ) {
                    return Err(EngineError::message(format!(
                        "synccheck waiter for completion action {action:?} did not register generation {generation}: {registered:?}",
                    )));
                }
            }
        }
        let strict_result = match action.kind() {
            PhysicalCompletionKind::Transaction { transactions } => {
                candidate.complete_tx(token, transactions, None)
            }
            PhysicalCompletionKind::Arrival {
                warp_id,
                arrival_count,
            } => candidate.complete_tx(token, 0, None).and_then(|_| {
                candidate.arrive(
                    action.barrier_id(),
                    warp_id,
                    arrival_count,
                    Some(issue_operation.clone()),
                )
            }),
        };
        let strict_effect = match strict_result {
            Ok(effect) => effect,
            Err(error) => {
                let message = format!(
                    "synccheck rejected completion action {} from {}: {error}",
                    action.id(),
                    issue_operation
                );
                state.findings.push(SyncCheckFinding {
                    operation: issue_operation,
                    effect: SyncCheckEffectKind::MbarrierCompletion,
                    error: error.into(),
                });
                return Err(EngineError::message(message));
            }
        };
        Ok(PreparedMbarrierCompletion {
            candidate,
            strict_effect,
            issue_operation,
        })
    }

    fn before_completion(&self, action: PhysicalCompletionAction) -> Result<(), EngineError> {
        let mut state = self
            .inner_for_physical_barrier(action.barrier_id())
            .lock()
            .expect("sync-check state poisoned");
        if state.staged_completions.contains_key(&action.id()) {
            // Completion pumps may race after selecting the same published
            // action. The numeric hub serializes its actual application, so
            // the first successful callback owns the one staged candidate.
            return Ok(());
        }
        let prepared = Self::prepare_completion(&mut state, action)
            .map_err(|error| EngineError::message(format!("before completion: {error}")))?;
        let revision = state.revision;
        state.staged_completions.insert(
            action.id(),
            StagedMbarrierCompletion {
                action,
                prepared,
                revision,
            },
        );
        Ok(())
    }

    fn after_completion(&self, outcome: &PhysicalCompletionOutcome) -> Result<(), EngineError> {
        let _profile = ProfileTimer::new(ProfileKind::SyncAfterCompletion);
        let action = outcome.action();
        let mut state = self
            .inner_for_physical_barrier(action.barrier_id())
            .lock()
            .expect("sync-check state poisoned");
        let staged = state
            .staged_completions
            .remove(&action.id())
            .ok_or_else(|| {
                EngineError::message(format!(
                    "after completion: physical completion action {} was not staged",
                    action.id()
                ))
            })?;
        if staged.action != action {
            return Err(EngineError::message(format!(
                "after completion: physical completion action {} changed between validation and commit",
                action.id()
            )));
        }
        let prepared = if staged.revision == state.revision {
            staged.prepared
        } else {
            Self::prepare_completion(&mut state, action)
                .map_err(|error| EngineError::message(format!("after completion: {error}")))?
        };
        let PreparedMbarrierCompletion {
            candidate,
            strict_effect,
            issue_operation,
        } = prepared;
        let causal_token = state
            .completion_tokens
            .get(&action.id())
            .expect("prepared completion retains its causal token")
            .causal_token
            .clone();
        let strict_ready_warps = strict_effect
            .ready_waiters()
            .iter()
            .map(|waiter| waiter.warp_id())
            .collect::<Vec<_>>();
        let numeric_woken_are_typed = outcome
            .woken_warp_ids()
            .iter()
            .all(|warp_id| strict_ready_warps.contains(warp_id));
        if strict_effect.completed_generation() != outcome.completed_generation()
            || !numeric_woken_are_typed
        {
            return Err(EngineError::message(format!(
                "synccheck completion action {} disagrees with numeric outcome: strict generation {:?}, ready warps {:?}; numeric generation {:?}, woken warps {:?}",
                action.id(),
                strict_effect.completed_generation(),
                strict_ready_warps,
                outcome.completed_generation(),
                outcome.woken_warp_ids()
            )));
        }
        let next_revision = state
            .revision
            .checked_add(1)
            .ok_or_else(|| EngineError::message("synccheck protocol revision overflow"))?;
        state.protocol = candidate;
        state.revision = next_revision;
        let causal_completion = state
            .causality
            .retain_conditional_completion(
                action.barrier_id(),
                outcome.conditional_completed_generation(),
            )
            .and_then(|()| state.causality.mbarrier_complete(&causal_token));
        if let Err(error) = causal_completion {
            return Err(record_causality_error(
                &mut state,
                &issue_operation,
                SyncCheckEffectKind::MbarrierCompletion,
                error,
            ));
        }
        state.completion_tokens.remove(&action.id());
        self.record_effect(
            &mut state,
            &issue_operation,
            SyncCheckEffectKind::MbarrierCompletion,
            SyncCheckEffectOutcome::MbarrierCompletion {
                action_id: action.id(),
                barrier_id: action.barrier_id(),
                generation: action.generation(),
                transactions: action.transactions(),
                completion_kind: action.kind(),
                completed_generation: outcome.completed_generation(),
                ready_warps: strict_ready_warps.into_boxed_slice(),
            },
        );
        Ok(())
    }
}

fn fixed_sync_verifier_discards_effect(effect: OperationEffect<'_>) -> bool {
    matches!(
        effect,
        OperationEffect::PhysicalAccess(_)
            | OperationEffect::TcgenWorkIssue(_)
            | OperationEffect::TcgenWait { .. }
            | OperationEffect::AsyncGroupIssue(_)
            | OperationEffect::AsyncGroupIssueBatch(_)
            | OperationEffect::AsyncGroupCommit { .. }
            | OperationEffect::AsyncGroupWait { .. }
            | OperationEffect::ProxyAsyncFence(_)
            | OperationEffect::MbarrierInitFence { .. }
    )
}

fn apply_mbarrier_commit_causality(
    state: &mut SyncCheckState,
    operation: &OperationContext,
    effect: &OwnedSyncEffect,
    preview: &SyncEffectPreview,
) -> Result<(), SyncCausalityError> {
    match (effect, preview) {
        (OwnedSyncEffect::Invalidate(ids), SyncEffectPreview::Applied) => {
            for &id in ids.iter() {
                state.causality.mbarrier_invalidate(id);
            }
            state
                .causality
                .program_tick(operation.id().global_warp_id())?;
            Ok(())
        }
        (OwnedSyncEffect::Init(plan), SyncEffectPreview::Applied) => {
            debug_assert_eq!(plan.barrier_ids().len(), operation.active_mask().len());
            let mut seen_barriers = BTreeSet::new();
            for (&barrier_id, lane_id) in plan.barrier_ids().iter().zip(operation.active_mask()) {
                if !seen_barriers.insert(barrier_id) {
                    // PTX mbarrier.init is idempotent when multiple active
                    // lanes in one instruction name the same physical slot.
                    // The strict protocol already collapses those targets;
                    // causal clocks must do the same so one instruction
                    // contributes one initialization event.
                    continue;
                }
                let prior_generation = state
                    .causality
                    .barrier_state(barrier_id)
                    .and_then(|barrier| barrier.latest_generation());
                if let Some(prior_generation) = prior_generation {
                    state.causality.mbarrier_reinitialize(
                        barrier_id,
                        prior_generation,
                        operation.id().global_warp_id(),
                        lane_id,
                    )?;
                } else {
                    state.causality.mbarrier_init(
                        barrier_id,
                        operation.id().global_warp_id(),
                        lane_id,
                    )?;
                }
            }
            Ok(())
        }
        (OwnedSyncEffect::ExpectTx { plan, .. }, SyncEffectPreview::ExpectTx { expected }) => {
            debug_assert_eq!(plan.entries().len(), expected.len());
            for (entry, &(barrier_id, generation)) in plan.entries().iter().zip(expected.iter()) {
                debug_assert_eq!(entry.barrier_id(), barrier_id);
                state.causality.mbarrier_expect_tx(
                    barrier_id,
                    generation,
                    operation.id().global_warp_id(),
                )?;
            }
            Ok(())
        }
        (
            OwnedSyncEffect::Arrive {
                plan,
                outcome: Some(outcome),
            },
            SyncEffectPreview::Arrive { expected, .. },
        ) => {
            state.causality.retain_conditional_completion(
                plan.barrier_id(),
                outcome.conditional_completed_generation(),
            )?;
            state
                .causality
                .mbarrier_arrive(plan.barrier_id(), expected.generation(), plan.warp_id())
                .map(|_| ())
        }
        (
            OwnedSyncEffect::ArriveBatch {
                outcome: Some(outcome),
                ..
            },
            SyncEffectPreview::ArriveBatch { effects, .. },
        ) => {
            for (&(barrier_id, generation, _), outcome) in effects.iter().zip(outcome.outcomes()) {
                state.causality.retain_conditional_completion(
                    barrier_id,
                    outcome.conditional_completed_generation(),
                )?;
                state.causality.mbarrier_arrive(
                    barrier_id,
                    generation,
                    operation.id().global_warp_id(),
                )?;
            }
            Ok(())
        }
        (
            OwnedSyncEffect::Wait { plan, .. },
            SyncEffectPreview::Wait(SyncCheckWaitState::Ready {
                generation: Some(generation),
                consumed_now,
            }),
        ) => {
            apply_mbarrier_wait_causality(state, operation.id(), *plan, *generation, *consumed_now)
        }
        (
            OwnedSyncEffect::Wait { .. },
            SyncEffectPreview::Wait(SyncCheckWaitState::Ready {
                generation: None, ..
            }),
        )
        | (OwnedSyncEffect::CompletionIssue { .. }, SyncEffectPreview::CompletionIssue { .. })
        | (
            OwnedSyncEffect::CpAsyncMbarrierArrive { .. },
            SyncEffectPreview::CpAsyncMbarrierArrive { .. },
        )
        | (OwnedSyncEffect::TcgenCommitIssue { .. }, SyncEffectPreview::TcgenCommitIssue { .. }) => {
            Ok(())
        }
        (
            OwnedSyncEffect::Wait { .. },
            SyncEffectPreview::Wait(SyncCheckWaitState::Registered { .. }),
        ) => {
            unreachable!("registered mbarrier waits do not have an immediate commit candidate")
        }
        _ => unreachable!("strict mbarrier preview must match its committed effect"),
    }
}

fn apply_mbarrier_wait_causality(
    state: &mut SyncCheckState,
    operation: &DynamicOpId,
    plan: PhysicalMbarrierWaitPlan,
    generation: u64,
    claim_consumption: bool,
) -> Result<(), SyncCausalityError> {
    let already_consumed = state
        .causality
        .barrier_state(plan.barrier_id())
        .and_then(|barrier| barrier.generation(generation))
        .and_then(|generation| generation.consumption_clock())
        .is_some();
    if claim_consumption && !already_consumed {
        state.causality.mbarrier_wait_consume_at(
            plan.barrier_id(),
            generation,
            plan.warp_id(),
            operation,
        )?;
    } else {
        state.causality.mbarrier_wait_acquire_at(
            plan.barrier_id(),
            generation,
            plan.warp_id(),
            operation,
        )?;
    }
    Ok(())
}

fn record_causality_error(
    state: &mut SyncCheckState,
    operation: &DynamicOpId,
    effect: SyncCheckEffectKind,
    error: SyncCausalityError,
) -> EngineError {
    let message = format!(
        "synccheck rejected causal {} at {}: {error}",
        effect.name(),
        operation
    );
    state.findings.push(SyncCheckFinding {
        operation: operation.clone(),
        effect,
        error: error.into(),
    });
    EngineError::message(message)
}

fn merge_causal_payload<B: Ord + Copy>(
    payloads: &mut BTreeMap<(B, u64), SyncClockPayload>,
    key: (B, u64),
    payload: SyncClockPayload,
) -> Result<(), SyncCausalityError> {
    match payloads.entry(key) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(payload);
        }
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            entry.get_mut().merge(&payload)?;
        }
    }
    retire_named_barrier_generations(payloads, key.0, key.1);
    Ok(())
}

fn apply_named_barrier_causality(
    state: &mut SyncCheckState,
    effect: &OwnedNamedBarrierEffect,
) -> Result<(), SyncCausalityError> {
    match effect {
        OwnedNamedBarrierEffect::Arrive {
            plan,
            outcome: Some(outcome),
        } => {
            let payload = state.causality.sync_release(plan.warp_id())?;
            merge_causal_payload(
                &mut state.named_causal_payloads,
                (plan.barrier_id(), outcome.generation()),
                payload,
            )
        }
        OwnedNamedBarrierEffect::Register {
            plan,
            outcome: Some(outcome),
        } => {
            let payload = state.causality.sync_release(plan.warp_id())?;
            merge_causal_payload(
                &mut state.named_causal_payloads,
                (plan.barrier_id(), outcome.generation()),
                payload,
            )
        }
        OwnedNamedBarrierEffect::Resume(plan) => {
            let payload = state
                .named_causal_payloads
                .get(&(plan.barrier_id(), plan.generation()))
                .ok_or_else(|| SyncCausalityError::ReleasePayloadRetired {
                    barrier: format!("named {:?}", plan.barrier_id()),
                    generation: plan.generation(),
                })?
                .clone();
            state
                .causality
                .sync_acquire(plan.warp_id(), &payload)
                .map(|_| ())
        }
        OwnedNamedBarrierEffect::Arrive { outcome: None, .. }
        | OwnedNamedBarrierEffect::Register { outcome: None, .. } => {
            unreachable!("only committed named-barrier effects update causality")
        }
    }
}

fn apply_cluster_barrier_causality(
    state: &mut SyncCheckState,
    effect: &OwnedClusterBarrierEffect,
) -> Result<(), SyncCausalityError> {
    match effect {
        OwnedClusterBarrierEffect::Arrive {
            plan,
            outcome: Some(outcome),
        } => {
            let publishes_init_fence = state.pending_mbarrier_init_fences.remove(&plan.warp_id());
            if !plan.publishes_memory() && !publishes_init_fence {
                return Ok(());
            }
            let payload = state.causality.sync_release(plan.warp_id())?;
            merge_causal_payload(
                &mut state.cluster_causal_payloads,
                (plan.barrier_id(), outcome.generation()),
                payload,
            )
        }
        OwnedClusterBarrierEffect::WaitResume(plan) if plan.plan().acquires_memory() => {
            let Some(payload) = state
                .cluster_causal_payloads
                .get(&(plan.plan().barrier_id(), plan.generation()))
                .cloned()
            else {
                return Ok(());
            };
            state
                .causality
                .sync_acquire(plan.plan().warp_id(), &payload)
                .map(|_| ())
        }
        OwnedClusterBarrierEffect::Arrive { outcome: None, .. }
        | OwnedClusterBarrierEffect::WaitRegister { outcome: None, .. } => {
            unreachable!("only committed cluster-barrier effects update causality")
        }
        OwnedClusterBarrierEffect::WaitRegister { .. }
        | OwnedClusterBarrierEffect::WaitResume(_) => Ok(()),
    }
}

/// Native synchronization-analysis policy for the shared transpiled body.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SyncCheckMode;

impl crate::engine_mode::EngineModeImpl for SyncCheckMode {
    type LaunchState = SyncCheckLaunchState;
    type GlobalMemoryTransactionGuard<'a> = ();

    const NAME: &'static str = "synccheck";
    const OBSERVES_OPERATIONS: bool = true;

    fn observes_analysis_gap(_state: &Self::LaunchState, kind: AnalysisGapKind) -> bool {
        !matches!(
            kind,
            AnalysisGapKind::AtomicLaneSerialization
                | AnalysisGapKind::TcgenMma
                | AnalysisGapKind::TcgenShift
        )
    }

    fn resolves_async_accesses(state: &Self::LaunchState) -> bool {
        !state.uses_compact_memory_analysis()
    }

    fn elides_physical_access_resolution(state: &Self::LaunchState, kind: OperationKind) -> bool {
        state.uses_compact_memory_analysis()
            && matches!(kind, OperationKind::Load | OperationKind::Store)
    }

    fn controls_physical_access(
        state: &Self::LaunchState,
        kind: OperationKind,
        space: PhysicalAccessSpace,
    ) -> bool {
        if state.uses_compact_memory_analysis()
            && matches!(kind, OperationKind::Load | OperationKind::Store)
        {
            return false;
        }
        !(state.uses_compact_memory_analysis()
            && matches!(
                space,
                PhysicalAccessSpace::Local | PhysicalAccessSpace::Register
            ))
    }

    fn controls_physical_access_allocation(
        state: &Self::LaunchState,
        kind: OperationKind,
        space: PhysicalAccessSpace,
        allocation: Option<PhysicalAllocationId>,
    ) -> bool {
        if !Self::controls_physical_access(state, kind, space) {
            return false;
        }
        if kind == OperationKind::Load && space == PhysicalAccessSpace::Global {
            return allocation.is_none_or(|allocation| state.tracks_global_allocation(allocation));
        }
        true
    }

    fn observes_atomic_physical_access_batch(
        _state: &Self::LaunchState,
        _descriptor: PhysicalAccessDescriptor,
        _mask: WarpMask,
        _return_sync_relevant: bool,
    ) -> bool {
        false
    }

    fn before_operation(
        _state: &Self::LaunchState,
        _operation: &OperationContext,
    ) -> Result<(), EngineError> {
        Ok(())
    }

    /// Synccheck layers nothing on operation close. The `fence.mbarrier_init`
    /// work this used to carry now arrives as an `MbarrierInitFence` effect.
    fn after_operation(
        _state: &Self::LaunchState,
        _operation: &OperationContext,
    ) -> Result<(), EngineError> {
        Ok(())
    }

    fn before_effect(
        state: &Self::LaunchState,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        let _profile_timer = ProfileTimer::new(ProfileKind::SyncBeforeEffect);
        state.before_effect(operation, effect)
    }

    fn after_effect(
        state: &Self::LaunchState,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        // A warp rendezvous is actor-local: it touches no barrier resource and
        // advances no causal clock, so it records its resolved summary directly
        // and skips the generic effect+clock registration below. This is the
        // exact record the dedicated `after_warp_sync` hook used to write.
        if let OperationEffect::WarpSync(sync) = effect {
            return Self::record_warp_sync(state, operation, sync.mask());
        }
        // The engine owns the fenced-barrier set now; this arm replaces the
        // `after_operation` hook that used to derive it from private state.
        if let OperationEffect::MbarrierInitFence { barrier_ids } = effect {
            return state.apply_mbarrier_init_fence(operation, barrier_ids);
        }
        profile_count(match effect {
            OperationEffect::PhysicalAccess(_) => ProfileKind::SyncEffectMemory,
            OperationEffect::AsyncPayload(_) => ProfileKind::SyncEffectAsyncPayload,
            OperationEffect::AsyncGroupIssue(_)
            | OperationEffect::AsyncGroupIssueBatch(_)
            | OperationEffect::AsyncGroupCommit { .. }
            | OperationEffect::AsyncGroupWait { .. } => ProfileKind::SyncEffectAsyncGroup,
            OperationEffect::MbarrierInvalidate { .. }
            | OperationEffect::MbarrierInit(_)
            | OperationEffect::MbarrierInitFence { .. }
            | OperationEffect::MbarrierExpectTx { .. }
            | OperationEffect::MbarrierArrive { .. }
            | OperationEffect::MbarrierArriveBatch { .. }
            | OperationEffect::MbarrierWait { .. }
            | OperationEffect::DeclaredWordWait { .. }
            | OperationEffect::MbarrierCompletionIssue { .. }
            | OperationEffect::CpAsyncMbarrierArrive { .. } => ProfileKind::SyncEffectMbarrier,
            OperationEffect::TcgenWorkIssue(_)
            | OperationEffect::TcgenCommitIssue { .. }
            | OperationEffect::TcgenWait { .. }
            | OperationEffect::TcgenFence(_) => ProfileKind::SyncEffectTcgen,
            OperationEffect::NamedBarrierArrive { .. }
            | OperationEffect::NamedBarrierSyncRegister { .. }
            | OperationEffect::NamedBarrierSyncResume(_)
            | OperationEffect::ClusterBarrierArrive { .. }
            | OperationEffect::ClusterBarrierWaitRegister { .. }
            | OperationEffect::ClusterBarrierWaitResume(_) => ProfileKind::SyncEffectNamedCluster,
            OperationEffect::TcgenLifecycleRegister(_)
            | OperationEffect::TcgenLifecycleResume(_)
            | OperationEffect::SetmaxnregRegister(_)
            | OperationEffect::SetmaxnregResume(_) => ProfileKind::SyncEffectLifecycleSetmax,
            OperationEffect::ProxyAsyncFence(_)
            | OperationEffect::TensorMap(_)
            | OperationEffect::MemoryFence(_)
            | OperationEffect::WarpSync(_)
            | OperationEffect::AnalysisGap(_) => ProfileKind::SyncEffectOther,
        });
        {
            let _profile_timer = ProfileTimer::new(ProfileKind::SyncAfterEffect);
            state.after_effect(operation, effect)?;
        }
        if !state.records_resolved_transitions() {
            return Ok(());
        }
        // A blocking named-barrier sync records its full resolved summary at
        // registration. Resume has the same DynamicOpId and only refines that
        // operation's clock with the acquired release payload. Avoid rebuilding
        // and comparing the identical summary for every resumed barrier.
        if state.records_only_fixed_sync_transitions()
            && matches!(effect, OperationEffect::NamedBarrierSyncResume(_))
        {
            let clock = {
                let _profile_timer = ProfileTimer::new(ProfileKind::SyncTransitionClock);
                state
                    .recorded_operation_clock(operation.id(), effect)?
                    .ok_or_else(|| {
                        EngineError::message(format!(
                            "synccheck named-barrier resume at {} has no committed causal clock",
                            operation.id()
                        ))
                    })?
            };
            let registration = {
                let _profile_timer = ProfileTimer::new(ProfileKind::SyncTransitionEffect);
                state
                    .transition_log()
                    .register_operation_clock(operation.id().clone(), clock)
                    .map_err(|error| {
                        EngineError::message(format!(
                            "synccheck could not refine named-barrier resume clock at {}: {error}",
                            operation.id()
                        ))
                    })?
            };
            if registration == ResolvedTransitionRegistration::Inserted {
                return Err(EngineError::message(format!(
                    "synccheck named-barrier resume at {} has no recorded registration",
                    operation.id()
                )));
            }
            return Ok(());
        }
        if state.records_only_fixed_sync_transitions()
            && fixed_sync_verifier_discards_effect(effect)
        {
            return Ok(());
        }
        if matches!(effect, OperationEffect::AnalysisGap(gap) if !Self::observes_analysis_gap(state, gap.kind()))
        {
            return Ok(());
        }
        let clock = {
            let _profile_timer = ProfileTimer::new(ProfileKind::SyncTransitionClock);
            state.recorded_operation_clock(operation.id(), effect)?
        };
        {
            let _profile_timer = ProfileTimer::new(ProfileKind::SyncTransitionEffect);
            state
                .transition_log()
                .register_operation_effect_and_clock(operation, effect, clock)
                .map_err(|error| {
                    EngineError::message(format!(
                        "synccheck could not record resolved effect and causal clock at {}: {error}",
                        operation.id()
                    ))
                })?;
        }
        if matches!(
            effect,
            OperationEffect::MbarrierCompletionIssue { .. }
                | OperationEffect::TcgenCommitIssue { .. }
                | OperationEffect::AsyncPayload(_)
        ) {
            let _profile_timer = ProfileTimer::new(ProfileKind::SyncCompletionLookup);
            for (action_id, clock) in state.completion_source_clocks(operation.id()) {
                state
                    .transition_log()
                    .register_completion_source_clock(action_id.get(), clock)
                    .map_err(|error| {
                        EngineError::message(format!(
                            "synccheck could not record completion causal source at {}: {error}",
                            operation.id()
                        ))
                    })?;
            }
        }
        if let OperationEffect::SetmaxnregResume(resume) = effect {
            state.transition_log().record_setmaxnreg_resume(resume);
        }
        if let OperationEffect::TcgenLifecycleResume(resume) = effect {
            state
                .transition_log()
                .record_tcgen_lifecycle_resume(operation.id(), resume);
        }
        if matches!(
            effect,
            OperationEffect::MbarrierCompletionIssue { .. }
                | OperationEffect::TcgenCommitIssue { .. }
                | OperationEffect::AsyncPayload(_)
        ) {
            let _profile_timer = ProfileTimer::new(ProfileKind::SyncCompletionLookup);
            if let OperationEffect::AsyncPayload(payload) = effect {
                let completions = state.resolved_completion_issues(operation.id());
                if let Some(first) = completions.first() {
                    let completion = ResolvedCompletionEffect::new_deferred_payload(
                        first.action_id(),
                        payload.token().clone(),
                        completions.iter().map(|completion| completion.resource()),
                        payload
                            .completion_accesses()
                            .iter()
                            .map(ResolvedMemoryEffect::from_batch),
                        completions
                            .iter()
                            .map(|completion| completion.transactions())
                            .sum(),
                    );
                    state
                        .transition_log()
                        .register_completion(completion)
                        .map_err(|error| {
                            EngineError::message(format!(
                                "synccheck could not refine async payload completion at {}: {error}",
                                operation.id()
                            ))
                        })?;
                }
            } else {
                for completion in state.resolved_completion_issues(operation.id()) {
                    state
                        .transition_log()
                        .register_completion(completion)
                        .map_err(|error| {
                            EngineError::message(format!(
                                "synccheck could not record resolved completion issue at {}: {error}",
                                operation.id()
                            ))
                        })?;
                }
            }
        }
        Ok(())
    }

    fn before_completion(
        state: &Self::LaunchState,
        effect: CompletionActionEffect<'_>,
    ) -> Result<(), EngineError> {
        match effect {
            CompletionActionEffect::PhysicalMbarrier(action) => state.before_completion(*action),
            CompletionActionEffect::DeferredPayload(action) => {
                for physical_action in action.physical_actions() {
                    state.before_completion(*physical_action)?;
                }
                Ok(())
            }
            CompletionActionEffect::AsyncGroup(_) => Ok(()),
            CompletionActionEffect::Setmaxnreg(_) => Ok(()),
        }
    }

    fn after_completion(
        state: &Self::LaunchState,
        effect: CompletionEffect<'_>,
    ) -> Result<(), EngineError> {
        match effect {
            CompletionEffect::PhysicalMbarrier(outcome) => state.after_completion(outcome),
            CompletionEffect::DeferredPayload(outcome) => {
                for physical_outcome in outcome.physical_outcomes() {
                    state.after_completion(physical_outcome)?;
                }
                Ok(())
            }
            CompletionEffect::AsyncGroup(_) => Ok(()),
            CompletionEffect::Setmaxnreg(_) => Ok(()),
        }?;
        // Completion summaries only feed the fixed-sync verifier's snapshot;
        // without resolved-transition recording they would just accumulate
        // (5.5 M summaries, 3.5 GB, on MegaMoE t128_m128).
        if state.records_resolved_transitions() {
            state
                .transition_log()
                .register_completion_effect(effect)
                .map_err(|error| {
                    EngineError::message(format!(
                        "synccheck could not record resolved completion: {error}"
                    ))
                })?;
        }
        Ok(())
    }
}

impl SyncCheckMode {
    /// Record one completed same-warp lane rendezvous.
    ///
    /// Actor-local: no barrier resource, no causal clock. Kept out of the
    /// generic `after_effect` tail so the recorded summary stays exactly what
    /// the removed `after_warp_sync` hook wrote.
    fn record_warp_sync(
        state: &<Self as crate::engine_mode::EngineModeImpl>::LaunchState,
        operation: &OperationContext,
        mask: WarpMask,
    ) -> Result<(), EngineError> {
        if !state.records_resolved_transitions() {
            return Ok(());
        }
        state
            .transition_log()
            .register_operation(
                operation.id().clone(),
                ResolvedTransitionSummary::Synchronization(
                    ResolvedSynchronizationEffect::from_warp_sync(mask),
                ),
            )
            .map_err(|error| {
                EngineError::message(format!(
                    "synccheck could not record warp sync at {}: {error}",
                    operation.id()
                ))
            })?;
        Ok(())
    }
}

fn compare_cluster_outcome(
    operation: &OperationContext,
    expected: Option<StrictClusterBarrierOutcome>,
    generation: u64,
    completed_now: bool,
) -> Result<(), EngineError> {
    let expected = expected.ok_or_else(|| {
        EngineError::message(format!(
            "synccheck cluster-barrier registration at {} has no strict outcome",
            operation.id()
        ))
    })?;
    if expected.generation() != generation || expected.completed_now() != completed_now {
        return Err(EngineError::message(format!(
            "synccheck cluster-barrier registration at {} disagrees with numeric outcome: strict {:?}, numeric generation {generation}, completed_now {completed_now}",
            operation.id(),
            expected,
        )));
    }
    Ok(())
}

const fn tcgen_effect_kind(action: TcgenLifecycleAction, resumed: bool) -> SyncCheckEffectKind {
    match (action, resumed) {
        (TcgenLifecycleAction::Allocate, false) => SyncCheckEffectKind::TcgenAllocRegister,
        (TcgenLifecycleAction::Allocate, true) => SyncCheckEffectKind::TcgenAllocResume,
        (TcgenLifecycleAction::Deallocate, false) => SyncCheckEffectKind::TcgenDeallocRegister,
        (TcgenLifecycleAction::Deallocate, true) => SyncCheckEffectKind::TcgenDeallocResume,
        (TcgenLifecycleAction::Relinquish, false) => SyncCheckEffectKind::TcgenRelinquishRegister,
        (TcgenLifecycleAction::Relinquish, true) => SyncCheckEffectKind::TcgenRelinquishResume,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum OwnedSyncEffect {
    Invalidate(Box<[PhysicalBarrierId]>),
    Init(PhysicalMbarrierInitPlan),
    ExpectTx {
        plan: PhysicalMbarrierExpectTxPlan,
        outcome: Option<PhysicalMbarrierExpectTxOutcome>,
    },
    Arrive {
        plan: PhysicalMbarrierArrivePlan,
        outcome: Option<PhysicalMbarrierArrivalOutcome>,
    },
    ArriveBatch {
        plan: PhysicalMbarrierArriveBatchPlan,
        outcome: Option<PhysicalMbarrierArrivalBatchOutcome>,
    },
    Wait {
        plan: PhysicalMbarrierWaitPlan,
        outcome: Option<crate::runtime::PhysicalMbarrierWaitOutcome>,
    },
    CompletionIssue {
        plan: PhysicalMbarrierCompletionIssuePlan,
        action_ids: Option<Box<[PhysicalCompletionActionId]>>,
    },
    CpAsyncMbarrierArrive {
        plan: CpAsyncMbarrierArrivePlan,
        actions: Option<Box<[PhysicalCompletionAction]>>,
    },
    TcgenCommitIssue {
        plan: TcgenCommitIssuePlan,
        actions: Option<Box<[PhysicalCompletionAction]>>,
    },
}

impl OwnedSyncEffect {
    fn from_effect(effect: OperationEffect<'_>) -> Option<Self> {
        match effect {
            OperationEffect::MbarrierInvalidate { barrier_ids } => {
                Some(Self::Invalidate(barrier_ids.into()))
            }
            OperationEffect::MbarrierInit(plan) => Some(Self::Init(plan.clone())),
            OperationEffect::MbarrierExpectTx { plan, outcome } => Some(Self::ExpectTx {
                plan: plan.clone(),
                outcome: outcome.cloned(),
            }),
            OperationEffect::MbarrierArrive { plan, outcome } => {
                Some(Self::Arrive { plan, outcome })
            }
            OperationEffect::MbarrierArriveBatch { plan, outcome } => Some(Self::ArriveBatch {
                plan: plan.clone(),
                outcome: outcome.cloned(),
            }),
            OperationEffect::MbarrierWait { plan, outcome } => Some(Self::Wait { plan, outcome }),
            // Not a barrier effect: a declared word's wait belongs to the global
            // memory model, and synccheck models no state for it.
            OperationEffect::DeclaredWordWait { .. } => None,
            OperationEffect::MbarrierCompletionIssue { plan, action_ids } => {
                Some(Self::CompletionIssue {
                    plan: plan.clone(),
                    action_ids: action_ids.map(|ids| ids.to_vec().into_boxed_slice()),
                })
            }
            OperationEffect::CpAsyncMbarrierArrive { plan, actions, .. } => {
                Some(Self::CpAsyncMbarrierArrive {
                    plan: plan.clone(),
                    actions: actions.map(Into::into),
                })
            }
            OperationEffect::TcgenCommitIssue { plan, actions, .. } => {
                Some(Self::TcgenCommitIssue {
                    plan: plan.clone(),
                    actions: actions.map(|actions| actions.into()),
                })
            }
            OperationEffect::AsyncPayload(payload) => Some(Self::CompletionIssue {
                plan: payload.completion_plan().clone(),
                action_ids: payload
                    .completion_action_ids()
                    .map(|ids| ids.to_vec().into_boxed_slice()),
            }),
            OperationEffect::PhysicalAccess(_)
            | OperationEffect::TcgenWorkIssue(_)
            | OperationEffect::TcgenWait { .. }
            | OperationEffect::AsyncGroupIssue(_)
            | OperationEffect::AsyncGroupIssueBatch(_)
            | OperationEffect::AsyncGroupCommit { .. }
            | OperationEffect::AsyncGroupWait { .. }
            | OperationEffect::MemoryFence(_)
            | OperationEffect::ProxyAsyncFence(_)
            | OperationEffect::TensorMap(_)
            | OperationEffect::MbarrierInitFence { .. }
            | OperationEffect::WarpSync(_)
            | OperationEffect::TcgenFence(_)
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
            | OperationEffect::SetmaxnregResume(_) => None,
        }
    }

    const fn kind(&self) -> SyncCheckEffectKind {
        match self {
            Self::Invalidate(_) => SyncCheckEffectKind::MbarrierInvalidate,
            Self::Init(_) => SyncCheckEffectKind::MbarrierInit,
            Self::ExpectTx { .. } => SyncCheckEffectKind::MbarrierExpectTx,
            Self::Arrive { .. } => SyncCheckEffectKind::MbarrierArrive,
            Self::ArriveBatch { .. } => SyncCheckEffectKind::MbarrierArrive,
            Self::Wait { .. } => SyncCheckEffectKind::MbarrierWait,
            Self::CompletionIssue { .. } => SyncCheckEffectKind::MbarrierCompletionIssue,
            Self::CpAsyncMbarrierArrive { .. } => SyncCheckEffectKind::CpAsyncMbarrierArrive,
            Self::TcgenCommitIssue { .. } => SyncCheckEffectKind::TcgenCommitIssue,
        }
    }

    fn same_staged_effect(&self, committed: &Self) -> bool {
        match (self, committed) {
            (
                Self::Wait {
                    plan: staged,
                    outcome: None,
                },
                Self::Wait {
                    plan: committed,
                    outcome: Some(_),
                },
            ) => staged == committed,
            (
                Self::CompletionIssue { plan: staged, .. },
                Self::CompletionIssue {
                    plan: committed, ..
                },
            ) => staged == committed,
            (
                Self::ExpectTx {
                    plan: staged,
                    outcome: None,
                },
                Self::ExpectTx {
                    plan: committed,
                    outcome: Some(_),
                },
            ) => staged == committed,
            (
                Self::Arrive {
                    plan: staged,
                    outcome: None,
                },
                Self::Arrive {
                    plan: committed,
                    outcome: Some(_),
                },
            ) => staged == committed,
            (
                Self::ArriveBatch {
                    plan: staged,
                    outcome: None,
                },
                Self::ArriveBatch {
                    plan: committed,
                    outcome: Some(_),
                },
            ) => staged == committed,
            (
                Self::CpAsyncMbarrierArrive {
                    plan: staged,
                    actions: None,
                },
                Self::CpAsyncMbarrierArrive {
                    plan: committed,
                    actions: Some(_),
                },
            ) => staged == committed,
            (
                Self::TcgenCommitIssue {
                    plan: staged,
                    actions: None,
                },
                Self::TcgenCommitIssue {
                    plan: committed,
                    actions: Some(_),
                },
            ) => staged == committed,
            _ => self == committed,
        }
    }

    fn completion_action_ids(&self) -> Option<&[PhysicalCompletionActionId]> {
        match self {
            Self::CompletionIssue {
                action_ids: Some(action_ids),
                ..
            } => Some(action_ids),
            Self::Invalidate(_)
            | Self::Init(_)
            | Self::ExpectTx { .. }
            | Self::Arrive { .. }
            | Self::ArriveBatch { .. }
            | Self::Wait { .. }
            | Self::CpAsyncMbarrierArrive { .. }
            | Self::TcgenCommitIssue { .. }
            | Self::CompletionIssue {
                action_ids: None, ..
            } => None,
        }
    }

    fn apply(
        &self,
        protocol: &StrictMbarrierProtocol,
        operation: &DynamicOpId,
    ) -> Result<SyncEffectPreview, StrictMbarrierError> {
        let witness = Some(operation.clone());
        match self {
            Self::Invalidate(ids) => {
                protocol.invalidate_many(ids, witness)?;
                Ok(SyncEffectPreview::Applied)
            }
            Self::Init(plan) => {
                protocol.init_many(plan.barrier_ids(), plan.expected_arrivals(), witness)?;
                Ok(SyncEffectPreview::Applied)
            }
            Self::ExpectTx {
                plan,
                outcome: None,
            } => {
                let mut expected = Vec::with_capacity(plan.entries().len());
                for entry in plan.entries() {
                    protocol.expect_tx(
                        entry.barrier_id(),
                        entry.expected_transactions(),
                        witness.clone(),
                    )?;
                    let generation = protocol
                        .snapshot(entry.barrier_id())
                        .generation()
                        .expect("successful strict mbarrier.expect_tx has a generation");
                    expected.push((entry.barrier_id(), generation));
                }
                Ok(SyncEffectPreview::ExpectTx {
                    expected: expected.into_boxed_slice(),
                })
            }
            Self::ExpectTx {
                outcome: Some(_), ..
            } => unreachable!("committed mbarrier.expect_tx cannot be staged"),
            Self::Arrive {
                plan,
                outcome: None,
            } => {
                let effect = protocol.arrive_with_drop(
                    plan.barrier_id(),
                    plan.warp_id(),
                    plan.arrival_count(),
                    plan.expected_transactions(),
                    plan.is_drop(),
                    witness.clone(),
                )?;
                let snapshot = protocol.snapshot(plan.barrier_id());
                let generation = snapshot
                    .generation()
                    .expect("successful strict mbarrier arrive has a generation");
                let expected = PhysicalMbarrierArrivalOutcome::new(
                    generation,
                    effect.completed_generation() == Some(generation),
                )
                .with_pending_arrivals_before(
                    snapshot.expected_arrivals().expect("initialized barrier")
                        - snapshot.arrival_count()
                        + plan.arrival_count(),
                );
                Ok(SyncEffectPreview::Arrive { effect, expected })
            }
            Self::Arrive {
                outcome: Some(_), ..
            } => unreachable!("committed mbarrier arrival cannot be staged"),
            Self::ArriveBatch {
                plan,
                outcome: None,
            } => {
                let mut effects = Vec::with_capacity(plan.entries().len());
                let mut expected = Vec::with_capacity(plan.entries().len());
                for entry in plan.entries() {
                    let plan = entry.plan();
                    let effect = protocol.arrive_with_drop(
                        plan.barrier_id(),
                        plan.warp_id(),
                        plan.arrival_count(),
                        plan.expected_transactions(),
                        plan.is_drop(),
                        witness.clone(),
                    )?;
                    let snapshot = protocol.snapshot(plan.barrier_id());
                    let generation = snapshot
                        .generation()
                        .expect("successful strict mbarrier arrive has a generation");
                    expected.push(
                        PhysicalMbarrierArrivalOutcome::new(
                            generation,
                            effect.completed_generation() == Some(generation),
                        )
                        .with_pending_arrivals_before(
                            snapshot.expected_arrivals().expect("initialized barrier")
                                - snapshot.arrival_count()
                                + plan.arrival_count(),
                        ),
                    );
                    effects.push((plan.barrier_id(), generation, effect));
                }
                Ok(SyncEffectPreview::ArriveBatch {
                    effects: effects.into_boxed_slice(),
                    expected: expected.into_boxed_slice(),
                })
            }
            Self::ArriveBatch {
                outcome: Some(_), ..
            } => unreachable!("committed mbarrier arrival batch cannot be staged"),
            Self::Wait {
                plan,
                outcome: Some(outcome),
            } => Ok(SyncEffectPreview::Wait(
                protocol
                    .acquire_completed(plan.barrier_id(), outcome.completed_generation(), witness)?
                    .into(),
            )),
            Self::Wait {
                plan,
                outcome: None,
            } => Ok(SyncEffectPreview::Wait(
                protocol
                    .wait(
                        plan.barrier_id(),
                        plan.requested_phase(),
                        plan.warp_id(),
                        witness,
                    )?
                    .into(),
            )),
            Self::CompletionIssue { plan, .. } => {
                let tokens = plan
                    .completions()
                    .iter()
                    .map(|&(barrier_id, transactions)| {
                        (transactions != 0)
                            .then(|| protocol.capture_completion(barrier_id, witness.clone()))
                            .transpose()
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(SyncEffectPreview::CompletionIssue {
                    tokens: tokens.into_boxed_slice(),
                })
            }
            Self::CpAsyncMbarrierArrive { plan, .. } => {
                // The plain spelling raises the current phase's pending arrival
                // count before deferring its arrive-on; `.noinc` reports no
                // increases, so this loop is empty for it.
                for (barrier_id, increase) in plan.pending_increases() {
                    protocol.increase_pending_arrivals(barrier_id, increase, witness.clone())?;
                }
                let tokens = plan
                    .targets()
                    .map(|(_, barrier_id)| protocol.capture_completion(barrier_id, witness.clone()))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(SyncEffectPreview::CpAsyncMbarrierArrive {
                    tokens: tokens.into_boxed_slice(),
                })
            }
            Self::TcgenCommitIssue { plan, .. } => {
                let tokens = plan
                    .barrier_ids()
                    .iter()
                    .map(|&barrier_id| protocol.capture_completion(barrier_id, witness.clone()))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(SyncEffectPreview::TcgenCommitIssue {
                    tokens: tokens.into_boxed_slice(),
                })
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum OwnedNamedBarrierEffect {
    Arrive {
        plan: NamedBarrierArrivePlan,
        outcome: Option<NamedBarrierArrivalOutcome>,
    },
    Register {
        plan: NamedBarrierSyncPlan,
        outcome: Option<NamedBarrierSyncRegistrationOutcome>,
    },
    Resume(NamedBarrierSyncResumePlan),
}

impl OwnedNamedBarrierEffect {
    fn from_effect(effect: OperationEffect<'_>) -> Option<Self> {
        match effect {
            OperationEffect::NamedBarrierArrive { plan, outcome } => {
                Some(Self::Arrive { plan, outcome })
            }
            OperationEffect::NamedBarrierSyncRegister { plan, outcome } => {
                Some(Self::Register { plan, outcome })
            }
            OperationEffect::NamedBarrierSyncResume(plan) => Some(Self::Resume(plan)),
            OperationEffect::PhysicalAccess(_)
            | OperationEffect::TcgenWorkIssue(_)
            | OperationEffect::TcgenWait { .. }
            | OperationEffect::AnalysisGap(_)
            | OperationEffect::AsyncPayload(_)
            | OperationEffect::AsyncGroupIssue(_)
            | OperationEffect::AsyncGroupIssueBatch(_)
            | OperationEffect::AsyncGroupCommit { .. }
            | OperationEffect::CpAsyncMbarrierArrive { .. }
            | OperationEffect::AsyncGroupWait { .. }
            | OperationEffect::MemoryFence(_)
            | OperationEffect::ProxyAsyncFence(_)
            | OperationEffect::TensorMap(_)
            | OperationEffect::MbarrierInitFence { .. }
            | OperationEffect::WarpSync(_)
            | OperationEffect::TcgenFence(_)
            | OperationEffect::MbarrierInvalidate { .. }
            | OperationEffect::MbarrierInit(_)
            | OperationEffect::MbarrierExpectTx { .. }
            | OperationEffect::MbarrierArrive { .. }
            | OperationEffect::MbarrierArriveBatch { .. }
            | OperationEffect::MbarrierWait { .. }
            | OperationEffect::DeclaredWordWait { .. }
            | OperationEffect::MbarrierCompletionIssue { .. }
            | OperationEffect::TcgenCommitIssue { .. }
            | OperationEffect::ClusterBarrierArrive { .. }
            | OperationEffect::ClusterBarrierWaitRegister { .. }
            | OperationEffect::ClusterBarrierWaitResume(_)
            | OperationEffect::TcgenLifecycleRegister(_)
            | OperationEffect::TcgenLifecycleResume(_)
            | OperationEffect::SetmaxnregRegister(_)
            | OperationEffect::SetmaxnregResume(_) => None,
        }
    }

    const fn kind(&self) -> SyncCheckEffectKind {
        match self {
            Self::Arrive { .. } => SyncCheckEffectKind::NamedBarrierArrive,
            Self::Register { .. } => SyncCheckEffectKind::NamedBarrierSyncRegister,
            Self::Resume(_) => SyncCheckEffectKind::NamedBarrierSyncResume,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedNamedBarrierOutcome {
    Arrive(NamedBarrierArrivalOutcome),
    Sync(NamedBarrierSyncRegistrationOutcome),
}

struct StagedNamedBarrierEffect {
    effect: OwnedNamedBarrierEffect,
    candidate: Option<StrictNamedBarrierProtocol>,
    expected_outcome: Option<ExpectedNamedBarrierOutcome>,
    revision: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum OwnedClusterBarrierEffect {
    Arrive {
        plan: ClusterBarrierArrivePlan,
        outcome: Option<ClusterBarrierArrivalOutcome>,
    },
    WaitRegister {
        plan: ClusterBarrierWaitPlan,
        outcome: Option<ClusterBarrierRegistrationOutcome>,
    },
    WaitResume(ClusterBarrierWaitResumePlan),
}

impl OwnedClusterBarrierEffect {
    fn from_effect(effect: OperationEffect<'_>) -> Option<Self> {
        match effect {
            OperationEffect::ClusterBarrierArrive { plan, outcome } => Some(Self::Arrive {
                plan: plan.clone(),
                outcome,
            }),
            OperationEffect::ClusterBarrierWaitRegister { plan, outcome } => {
                Some(Self::WaitRegister {
                    plan: plan.clone(),
                    outcome,
                })
            }
            OperationEffect::ClusterBarrierWaitResume(plan) => Some(Self::WaitResume(plan.clone())),
            _ => None,
        }
    }

    const fn kind(&self) -> SyncCheckEffectKind {
        match self {
            Self::Arrive { .. } => SyncCheckEffectKind::ClusterBarrierArrive,
            Self::WaitRegister { .. } => SyncCheckEffectKind::ClusterBarrierWaitRegister,
            Self::WaitResume(_) => SyncCheckEffectKind::ClusterBarrierWaitResume,
        }
    }

    const fn has_unmodeled_unaligned_participation(&self) -> bool {
        match self {
            Self::Arrive { plan, .. } => !plan.aligned() && !plan.arrival_mask().is_full(),
            Self::WaitRegister { plan, .. } => !plan.aligned() && !plan.arrival_mask().is_full(),
            Self::WaitResume(_) => false,
        }
    }

    const fn barrier_id(&self) -> ClusterBarrierId {
        match self {
            Self::Arrive { plan, .. } => plan.barrier_id(),
            Self::WaitRegister { plan, .. } => plan.barrier_id(),
            Self::WaitResume(plan) => plan.plan().barrier_id(),
        }
    }

    const fn warp_id(&self) -> usize {
        match self {
            Self::Arrive { plan, .. } => plan.warp_id(),
            Self::WaitRegister { plan, .. } => plan.warp_id(),
            Self::WaitResume(plan) => plan.plan().warp_id(),
        }
    }
}

struct StagedClusterBarrierEffect {
    effect: OwnedClusterBarrierEffect,
    candidate: Option<StrictClusterBarrierProtocol>,
    expected_outcome: Option<StrictClusterBarrierOutcome>,
    rearrival_without_wait: bool,
    revision: u64,
}

struct StagedSyncEffect {
    effect: OwnedSyncEffect,
    preview: SyncEffectPreview,
    candidate: Option<StrictMbarrierProtocol>,
    revision: u64,
}

struct PreparedMbarrierCompletion {
    candidate: StrictMbarrierProtocol,
    strict_effect: StrictMbarrierEffect,
    issue_operation: DynamicOpId,
}

struct StagedMbarrierCompletion {
    action: PhysicalCompletionAction,
    prepared: PreparedMbarrierCompletion,
    revision: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SyncEffectPreview {
    Applied,
    ExpectTx {
        expected: Box<[(PhysicalBarrierId, u64)]>,
    },
    Arrive {
        effect: StrictMbarrierEffect,
        expected: PhysicalMbarrierArrivalOutcome,
    },
    ArriveBatch {
        effects: Box<[(PhysicalBarrierId, u64, StrictMbarrierEffect)]>,
        expected: Box<[PhysicalMbarrierArrivalOutcome]>,
    },
    Wait(SyncCheckWaitState),
    CompletionIssue {
        tokens: Box<[Option<StrictMbarrierCompletionToken>]>,
    },
    CpAsyncMbarrierArrive {
        tokens: Box<[StrictMbarrierCompletionToken]>,
    },
    TcgenCommitIssue {
        tokens: Box<[StrictMbarrierCompletionToken]>,
    },
}

fn outcome_from_preview(
    preview: SyncEffectPreview,
    committed_wait: Option<SyncCheckWaitState>,
) -> SyncCheckEffectOutcome {
    match preview {
        SyncEffectPreview::Applied | SyncEffectPreview::ExpectTx { .. } => {
            SyncCheckEffectOutcome::Applied
        }
        SyncEffectPreview::Arrive { effect, .. } => SyncCheckEffectOutcome::MbarrierArrive {
            completed_generation: effect.completed_generation(),
            consumed_generation: effect.consumed_generation(),
            ready_warps: effect
                .ready_waiters()
                .iter()
                .map(|waiter| waiter.warp_id())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        },
        SyncEffectPreview::ArriveBatch { effects, .. } => {
            SyncCheckEffectOutcome::MbarrierArriveBatch {
                arrivals: effects
                    .into_vec()
                    .into_iter()
                    .map(
                        |(barrier_id, generation, effect)| SyncCheckMbarrierArrivalRecord {
                            barrier_id,
                            generation,
                            completed_generation: effect.completed_generation(),
                            consumed_generation: effect.consumed_generation(),
                            ready_warps: effect
                                .ready_waiters()
                                .iter()
                                .map(|waiter| waiter.warp_id())
                                .collect::<Vec<_>>()
                                .into_boxed_slice(),
                        },
                    )
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            }
        }
        SyncEffectPreview::Wait(staged) => SyncCheckEffectOutcome::MbarrierWait {
            committed: committed_wait.or_else(|| Some(staged.clone())),
            staged,
        },
        SyncEffectPreview::CompletionIssue { .. }
        | SyncEffectPreview::CpAsyncMbarrierArrive { .. }
        | SyncEffectPreview::TcgenCommitIssue { .. } => {
            unreachable!("completion issues are recorded with their committed action IDs")
        }
    }
}

fn push_unique_incomplete(
    incomplete: &mut Vec<SyncCheckIncompleteReason>,
    reason: SyncCheckIncompleteReason,
) {
    let already_present = match &reason {
        SyncCheckIncompleteReason::AnalysisGap {
            operation,
            kind,
        } => incomplete.iter().any(|existing| {
            matches!(
                existing,
                SyncCheckIncompleteReason::AnalysisGap {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::runtime::{
        plan_named_barrier_arrive, plan_physical_mbarrier_arrive, plan_physical_mbarrier_init,
        plan_physical_mbarrier_wait, run_kernel_engine_launch, run_kernel_engine_launch_report,
        ExecutionPolicy, LaunchSelection, PhysicalPtr, RuntimeBuffer,
    };
    use crate::{
        AnalysisGapEffect, AnalysisGapKind, AsyncPayloadEffect,
        BlockedOperation, ClusterBarrierId, CompletionActionEffect, ControlProvenance, CtaId,
        DynamicOpId, EngineError, ExecutionReport, LaunchTopology, OccurrenceKey, OperationContext,
        OperationEffect, OperationKind, OwnedOperationEffect, ParticipantState,
        PhysicalAccessBatch, PhysicalAccessDescriptor, PhysicalAccessKind, PhysicalAccessSpace,
        PhysicalAllocationId, PhysicalBarrierHub, PhysicalByteSpan, PhysicalMemory,
        ResolvedSyncResourceKey, ResolvedTransitionSummary, ScopeInstance, SetmaxnregAction,
        StaticOpId, StrictMbarrierError, StrictMbarrierLifecycle, StrictNamedBarrierError,
        SyncCausalityError, TcgenLifecycleAction, WarpMask, WarpValue, SETMAXNREG_WARPS_PER_GROUP,
    };

    use super::{
        SyncCheckEffectKind, SyncCheckEffectOutcome, SyncCheckIncompleteReason,
        SyncCheckLaunchState, SyncCheckMode, SyncCheckProtocolError, SyncCheckStatus,
        SyncCheckWaitState,
    };

    struct BarrierFixture {
        physical: PhysicalMemory,
        context: crate::WarpContext,
        pointer: PhysicalPtr,
    }

    #[test]
    fn topology_synccheck_uses_independent_cluster_state_shards() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let state = SyncCheckLaunchState::with_topology_transition_log_and_global_write_allocations(
            topology,
            crate::ResolvedTransitionLog::default(),
            [],
            u64::MAX,
        );
        let cluster0 = DynamicOpId::new(0, 0, 0, StaticOpId::new(70), []);
        let cluster1 = DynamicOpId::new(0, 1, 0, StaticOpId::new(71), []);
        let cluster0_inner = state.inner_for_operation(&cluster0);
        let cluster1_inner = state.inner_for_operation(&cluster1);

        assert!(!std::ptr::eq(cluster0_inner, cluster1_inner));
        let _cluster0_guard = cluster0_inner.lock().unwrap();
        assert!(cluster1_inner.try_lock().is_ok());
        assert!(std::ptr::eq(
            cluster0_inner,
            state.inner_for_physical_barrier(crate::PhysicalBarrierId::new(1, 0, 0))
        ));
        assert!(std::ptr::eq(
            cluster1_inner,
            state.inner_for_physical_barrier(crate::PhysicalBarrierId::new(2, 0, 1))
        ));
        assert!(std::ptr::eq(
            cluster0_inner,
            state.inner_for_cluster_barrier(ClusterBarrierId::new(0, 0))
        ));
        assert!(std::ptr::eq(
            cluster1_inner,
            state.inner_for_cluster_barrier(ClusterBarrierId::new(0, 1))
        ));
    }

    #[test]
    fn bounded_effect_journal_keeps_exact_counts_and_bytes() {
        let first = super::SyncCheckEffectRecord {
            operation: DynamicOpId::new(0, 0, 0, StaticOpId::new(72), []),
            effect: SyncCheckEffectKind::SetmaxnregRegister,
            outcome: SyncCheckEffectOutcome::Applied,
        };
        let second = super::SyncCheckEffectRecord {
            operation: DynamicOpId::new(0, 1, 0, StaticOpId::new(73), []),
            effect: SyncCheckEffectKind::SetmaxnregRegister,
            outcome: SyncCheckEffectOutcome::Applied,
        };
        let first_bytes = super::sync_check_effect_diagnostic_bytes(&first);
        let second_bytes = super::sync_check_effect_diagnostic_bytes(&second);
        let state = SyncCheckLaunchState::with_topology_transition_log_and_global_write_allocations(
            LaunchTopology::new(2, 1, 1).unwrap(),
            crate::ResolvedTransitionLog::default(),
            [],
            first_bytes,
        );

        {
            let mut inner = state.inner_for_operation(first.operation()).lock().unwrap();
            state.record_effect(
                &mut inner,
                first.operation(),
                first.effect(),
                first.outcome().clone(),
            );
        }
        {
            let mut inner = state
                .inner_for_operation(second.operation())
                .lock()
                .unwrap();
            state.record_effect(
                &mut inner,
                second.operation(),
                second.effect(),
                second.outcome().clone(),
            );
        }

        let result = state.result();
        assert!(!result.effect_journal_complete());
        assert!(result.effects().is_empty());
        assert_eq!(result.total_effect_count(), 2);
        assert_eq!(
            result.effect_counts().collect::<Vec<_>>(),
            [(SyncCheckEffectKind::SetmaxnregRegister, 2)]
        );
        assert_eq!(
            result.full_effect_diagnostic_bytes(),
            first_bytes.saturating_add(second_bytes)
        );
    }

    #[test]
    fn compact_analysis_skips_plain_and_async_memory_footprints() {
        let written = PhysicalAllocationId::new(7);
        let immutable = PhysicalAllocationId::new(8);
        let state = SyncCheckLaunchState::with_global_write_allocations([written]);

        assert!(state.tracks_global_allocation(written));
        assert!(!state.tracks_global_allocation(immutable));
        assert!(
            !<SyncCheckMode as crate::engine_mode::EngineModeImpl>::resolves_async_accesses(&state)
        );
        for kind in [OperationKind::Load, OperationKind::Store] {
            for space in [
                PhysicalAccessSpace::Global,
                PhysicalAccessSpace::Shared,
                PhysicalAccessSpace::Tmem,
                PhysicalAccessSpace::Local,
                PhysicalAccessSpace::Register,
            ] {
                assert!(
                    !<SyncCheckMode as crate::engine_mode::EngineModeImpl>::controls_physical_access_allocation(
                        &state,
                        kind,
                        space,
                        Some(written),
                    ),
                    "compact synccheck unexpectedly resolved {kind} in {space:?}"
                );
            }
        }
        assert!(
            <SyncCheckMode as crate::engine_mode::EngineModeImpl>::controls_physical_access(
                &state,
                OperationKind::Atomic,
                PhysicalAccessSpace::Shared,
            )
        );
    }

    impl BarrierFixture {
        fn new() -> Self {
            let topology = LaunchTopology::new(1, 1, 1).unwrap();
            let physical = PhysicalMemory::new(topology);
            let context = topology.warp_contexts().next().unwrap();
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
            Self {
                physical,
                context,
                pointer,
            }
        }

        fn operation(&self, sequence: u64, mask: WarpMask) -> OperationContext {
            self.operation_with_kind(sequence, OperationKind::Barrier, mask)
        }

        fn operation_with_kind(
            &self,
            sequence: u64,
            kind: OperationKind,
            mask: WarpMask,
        ) -> OperationContext {
            OperationContext::new(
                DynamicOpId::new(
                    0,
                    self.context.global_warp_id(),
                    sequence,
                    StaticOpId::new(100 + sequence),
                    [],
                ),
                kind,
                mask,
            )
        }
    }

    #[test]
    fn analysis_gap_is_one_typed_incomplete_per_dynamic_operation() {
        let state = SyncCheckLaunchState::new();
        let operation = OperationContext::new(
            DynamicOpId::new(0, 0, 0, StaticOpId::new(77), []),
            OperationKind::Control,
            WarpMask::from_lanes([0]).unwrap(),
        );
        let effect = OperationEffect::AnalysisGap(AnalysisGapEffect::new(
            AnalysisGapKind::ClusterBarrierUnaligned,
            0,
            0,
            None,
        ));

        <SyncCheckMode as crate::engine_mode::EngineModeImpl>::before_effect(
            &state, &operation, effect,
        )
        .unwrap();
        <SyncCheckMode as crate::engine_mode::EngineModeImpl>::after_effect(
            &state, &operation, effect,
        )
        .unwrap();
        <SyncCheckMode as crate::engine_mode::EngineModeImpl>::after_effect(
            &state, &operation, effect,
        )
        .unwrap();

        let result = state.result();
        assert_eq!(result.status(), SyncCheckStatus::Incomplete);
        assert_eq!(
            result.incomplete_reasons(),
            [SyncCheckIncompleteReason::AnalysisGap {
                operation: operation.id().clone(),
                kind: AnalysisGapKind::ClusterBarrierUnaligned,
            }]
        );
    }

    #[test]
    fn numeric_only_async_gaps_are_outside_synccheck_scope() {
        let state = SyncCheckLaunchState::new();
        for kind in [AnalysisGapKind::TcgenMma, AnalysisGapKind::TcgenShift] {
            assert!(
                !<SyncCheckMode as crate::engine_mode::EngineModeImpl>::observes_analysis_gap(
                    &state, kind
                )
            );
        }
    }

    #[test]
    fn cluster_participant_exit_adds_only_the_matching_gap() {
        let state = SyncCheckLaunchState::new();
        let barrier_id = ClusterBarrierId::new(7, 3);
        {
            let inner = state.inner.lock().expect("sync-check state poisoned");
            for (id, participants, warp) in [
                (barrier_id, [0, 1], 0),
                (ClusterBarrierId::new(7, 4), [2, 3], 2),
            ] {
                inner
                    .cluster_protocol
                    .arrive(id, &participants, warp, WarpMask::FULL, None)
                    .unwrap();
                inner
                    .cluster_protocol
                    .wait_register(id, &participants, warp, WarpMask::FULL, None)
                    .unwrap();
            }
        }
        let execution = ExecutionReport {
            stats: crate::ExecutionStats::default(),
            terminal: Err(EngineError::deadlock(
                vec![0],
                vec![BlockedOperation::new(
                    0,
                    crate::AwaitedOperation::ClusterBarrierWait,
                    OccurrenceKey::new(
                        u64::MAX - 1,
                        "barrier.cluster",
                        [0],
                        ScopeInstance::Cluster { cluster_id: 3 },
                    ),
                    Some(0),
                    ParticipantState {
                        expected: vec![0, 1],
                        arrived: vec![0],
                        missing: vec![1],
                        expected_arrival_count: None,
                        completed_arrival_count: None,
                        expected_transactions: None,
                        completed_transactions: None,
                    },
                )],
                3,
            )),
        };

        let abort_safe_result = state.result_before_aborted_execution();
        assert_eq!(abort_safe_result.status(), SyncCheckStatus::Clean);
        #[cfg(feature = "python")]
        assert!(crate::sync_check_python::reported_sync_execution_error(
            &abort_safe_result,
            execution.error(),
        )
        .is_some());
        let result = state.result_for_execution(&execution);
        assert_eq!(result.status(), SyncCheckStatus::Incomplete);
        #[cfg(feature = "python")]
        assert!(crate::sync_check_python::reported_sync_execution_error(
            &result,
            execution.error()
        )
        .is_none());
        assert_eq!(
            result.incomplete_reasons(),
            [
                SyncCheckIncompleteReason::ClusterBarrierParticipantExitUnmodeled {
                    barrier_id,
                    generation: 0,
                    missing_warps: vec![1].into_boxed_slice(),
                }
            ]
        );
    }

    fn before(
        state: &SyncCheckLaunchState,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
    ) -> Result<(), crate::EngineError> {
        <SyncCheckMode as crate::engine_mode::EngineModeImpl>::before_effect(
            state, operation, effect,
        )
    }

    fn after(
        state: &SyncCheckLaunchState,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
    ) -> Result<(), crate::EngineError> {
        <SyncCheckMode as crate::engine_mode::EngineModeImpl>::after_effect(
            state, operation, effect,
        )
    }

    fn after_init_fence(
        state: &SyncCheckLaunchState,
        fixture: &BarrierFixture,
        sequence: u64,
        barrier_ids: &[crate::PhysicalBarrierId],
    ) -> Result<(), crate::EngineError> {
        let operation =
            fixture.operation_with_kind(sequence, OperationKind::MbarrierInitFence, WarpMask::FULL);
        after(
            state,
            &operation,
            OperationEffect::MbarrierInitFence { barrier_ids },
        )
    }

    fn staged_arrive(plan: crate::runtime::PhysicalMbarrierArrivePlan) -> OperationEffect<'static> {
        OperationEffect::MbarrierArrive {
            plan,
            outcome: None,
        }
    }

    fn committed_arrive(
        plan: crate::runtime::PhysicalMbarrierArrivePlan,
        generation: u64,
        completed_now: bool,
    ) -> OperationEffect<'static> {
        OperationEffect::MbarrierArrive {
            plan,
            outcome: Some(
                crate::PhysicalMbarrierArrivalOutcome::new(generation, completed_now)
                    .with_pending_arrivals_before(plan.arrival_count()),
            ),
        }
    }

    #[test]
    fn successful_effect_callback_records_resolved_dependency_summary() {
        let fixture = BarrierFixture::new();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let operation = fixture.operation(0, mask);
        let plan =
            plan_physical_mbarrier_init(&fixture.context, &fixture.pointer, mask, 1).unwrap();
        let state = SyncCheckLaunchState::new();

        before(&state, &operation, OperationEffect::MbarrierInit(&plan)).unwrap();
        after(&state, &operation, OperationEffect::MbarrierInit(&plan)).unwrap();

        assert!(matches!(
            state.transition_log().operation_summary(operation.id()),
            Some(ResolvedTransitionSummary::Synchronization(_))
        ));
    }

    #[test]
    fn initializing_thread_fence_records_the_covered_barrier_and_allows_first_use() {
        let fixture = BarrierFixture::new();
        let state = SyncCheckLaunchState::new();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let init =
            plan_physical_mbarrier_init(&fixture.context, &fixture.pointer, mask, 1).unwrap();
        let arrive =
            plan_physical_mbarrier_arrive(&fixture.context, &fixture.pointer, mask, None, None)
                .unwrap()
                .unwrap();
        let init_operation = fixture.operation(0, mask);
        let fence_operation =
            fixture.operation_with_kind(1, OperationKind::MbarrierInitFence, mask);
        let arrive_operation = fixture.operation(2, mask);

        before(
            &state,
            &init_operation,
            OperationEffect::MbarrierInit(&init),
        )
        .unwrap();
        after(
            &state,
            &init_operation,
            OperationEffect::MbarrierInit(&init),
        )
        .unwrap();
        after(
            &state,
            &fence_operation,
            OperationEffect::MbarrierInitFence {
                barrier_ids: init.barrier_ids(),
            },
        )
        .unwrap();

        assert!(state.snapshot(arrive.barrier_id()).init_fenced());
        assert!(matches!(
            state
                .transition_log()
                .operation_summary(fence_operation.id()),
            Some(ResolvedTransitionSummary::Synchronization(summary))
                if matches!(
                    summary.details(),
                    OwnedOperationEffect::MbarrierInitFence { .. }
                )
                    && summary.resources().iter().any(|resource| {
                        resource.key()
                            == ResolvedSyncResourceKey::PhysicalMbarrier(arrive.barrier_id())
                    })
        ));

        before(&state, &arrive_operation, staged_arrive(arrive)).unwrap();
        after(&state, &arrive_operation, committed_arrive(arrive, 0, true)).unwrap();
        assert_eq!(state.result().status(), SyncCheckStatus::Clean);
    }

    #[test]
    fn compact_analysis_does_not_build_resolved_dependency_summaries() {
        fn access(operation: OperationContext, kind: PhysicalAccessKind) -> PhysicalAccessBatch {
            let descriptor =
                PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Shared, 4).unwrap();
            PhysicalAccessBatch::resolve(operation, descriptor, |_| {
                Ok::<_, std::convert::Infallible>(vec![PhysicalByteSpan::new(
                    PhysicalAllocationId::new(91),
                    0,
                    4,
                )
                .unwrap()])
            })
            .unwrap()
        }

        let state = SyncCheckLaunchState::with_global_write_allocations([]);
        let mask = WarpMask::from_lanes([0]).unwrap();
        let ordinary_operation = OperationContext::new(
            DynamicOpId::new(0, 0, 0, StaticOpId::new(201), []),
            OperationKind::Store,
            mask,
        );
        let ordinary = access(ordinary_operation.clone(), PhysicalAccessKind::Write);
        before(
            &state,
            &ordinary_operation,
            OperationEffect::PhysicalAccess(&ordinary),
        )
        .unwrap();
        after(
            &state,
            &ordinary_operation,
            OperationEffect::PhysicalAccess(&ordinary),
        )
        .unwrap();
        assert_eq!(
            state
                .transition_log()
                .operation_summary(ordinary_operation.id()),
            None
        );

        let atomic_operation = OperationContext::new(
            DynamicOpId::new(0, 0, 1, StaticOpId::new(202), []),
            OperationKind::Atomic,
            mask,
        );
        let atomic = access(
            atomic_operation.clone(),
            PhysicalAccessKind::AtomicReadModifyWrite,
        );
        before(
            &state,
            &atomic_operation,
            OperationEffect::PhysicalAccess(&atomic),
        )
        .unwrap();
        after(
            &state,
            &atomic_operation,
            OperationEffect::PhysicalAccess(&atomic),
        )
        .unwrap();
        assert_eq!(
            state
                .transition_log()
                .operation_summary(atomic_operation.id()),
            None
        );
    }

    #[test]
    fn async_payload_commit_binds_exact_strict_completion_token() {
        let fixture = BarrierFixture::new();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let state = SyncCheckLaunchState::new();
        let init_operation = fixture.operation(0, mask);
        let init =
            plan_physical_mbarrier_init(&fixture.context, &fixture.pointer, mask, 1).unwrap();
        before(
            &state,
            &init_operation,
            OperationEffect::MbarrierInit(&init),
        )
        .unwrap();
        after(
            &state,
            &init_operation,
            OperationEffect::MbarrierInit(&init),
        )
        .unwrap();
        after_init_fence(&state, &fixture, 1, init.barrier_ids()).unwrap();

        let issue_operation = OperationContext::new(
            DynamicOpId::new(
                0,
                fixture.context.global_warp_id(),
                2,
                StaticOpId::new(102),
                [],
            ),
            OperationKind::AsyncIssue,
            mask,
        );
        let descriptor = PhysicalAccessDescriptor::new(
            PhysicalAccessKind::Write,
            PhysicalAccessSpace::Shared,
            4,
        )
        .unwrap();
        let access = PhysicalAccessBatch::resolve(issue_operation.clone(), descriptor, |_| {
            Ok::<_, std::convert::Infallible>(vec![PhysicalByteSpan::new(
                PhysicalAllocationId::new(77),
                0,
                4,
            )
            .unwrap()])
        })
        .unwrap();
        let plan =
            crate::runtime::PhysicalMbarrierCompletionIssuePlan::single(init.barrier_ids()[0], 16);
        let mut payload = AsyncPayloadEffect::new(issue_operation.clone(), [access], plan).unwrap();
        before(
            &state,
            &issue_operation,
            OperationEffect::AsyncPayload(&payload),
        )
        .unwrap();

        let hub = PhysicalBarrierHub::new();
        init.apply(&hub).unwrap();
        let action_ids = payload.completion_plan().apply(&hub).unwrap();
        let action_id = action_ids[0];
        payload.bind_completion_action_ids(action_ids).unwrap();
        after(
            &state,
            &issue_operation,
            OperationEffect::AsyncPayload(&payload),
        )
        .unwrap();

        let result = state.result();
        assert_eq!(result.status(), SyncCheckStatus::Incomplete);
        let issue = result
            .effects()
            .iter()
            .find(|record| record.operation() == issue_operation.id())
            .unwrap();
        let SyncCheckEffectOutcome::MbarrierCompletionIssue { actions } = issue.outcome() else {
            panic!("async payload must retain its completion issue")
        };
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].action_id(), action_id);
        assert_eq!(actions[0].barrier_id(), init.barrier_ids()[0]);
        assert_eq!(actions[0].generation(), Some(0));
        assert!(matches!(
            result.incomplete_reasons(),
            [SyncCheckIncompleteReason::CompletionActionUnobserved {
                action_id: pending,
                barrier_id: _,
                generation: 0,
            }] if *pending == action_id
        ));
        assert!(matches!(
            state
                .transition_log()
                .operation_summary(issue_operation.id()),
            Some(ResolvedTransitionSummary::AsyncPayload(_))
        ));
        let ResolvedTransitionSummary::Completion(completion) = state
            .transition_log()
            .completion_summary(action_id.get())
            .unwrap()
        else {
            panic!("async payload action must be indexed for controlled completion")
        };
        assert_eq!(completion.resource().generation(), Some(0));

        let issuer_clock = state
            .inner
            .lock()
            .expect("sync-check state poisoned")
            .causality
            .warp_clock(fixture.context.global_warp_id())
            .unwrap()
            .clone();
        let arrive =
            plan_physical_mbarrier_arrive(&fixture.context, &fixture.pointer, mask, None, Some(16))
                .unwrap()
                .unwrap();
        let arrive_operation = fixture.operation(3, mask);
        before(&state, &arrive_operation, staged_arrive(arrive)).unwrap();
        let arrive_outcome = arrive.apply(&hub).unwrap();
        after(
            &state,
            &arrive_operation,
            OperationEffect::MbarrierArrive {
                plan: arrive,
                outcome: Some(arrive_outcome),
            },
        )
        .unwrap();

        let action = hub.pending_completion_actions()[0];
        state.before_completion(action).unwrap();
        let outcome = hub.apply_completion_detailed(action.id()).unwrap();
        state.after_completion(&outcome).unwrap();

        let state_guard = state.inner.lock().expect("sync-check state poisoned");
        let release_clock = state_guard
            .causality
            .barrier_state(action.barrier_id())
            .unwrap()
            .generation(action.generation())
            .unwrap()
            .release_payload()
            .unwrap()
            .clock();
        assert!(issuer_clock.happens_before(release_clock));
        drop(state_guard);
        assert!(state
            .result()
            .incomplete_reasons()
            .iter()
            .all(|reason| !matches!(
                reason,
                SyncCheckIncompleteReason::CompletionActionUnobserved { .. }
            )));
    }

    #[test]
    fn completion_accepts_strict_waiter_not_yet_registered_numerically() {
        let fixture = BarrierFixture::new();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let state = SyncCheckLaunchState::new();
        let hub = PhysicalBarrierHub::new();

        let init_operation = fixture.operation(0, mask);
        let init =
            plan_physical_mbarrier_init(&fixture.context, &fixture.pointer, mask, 1).unwrap();
        before(
            &state,
            &init_operation,
            OperationEffect::MbarrierInit(&init),
        )
        .unwrap();
        init.apply(&hub).unwrap();
        after(
            &state,
            &init_operation,
            OperationEffect::MbarrierInit(&init),
        )
        .unwrap();
        after_init_fence(&state, &fixture, 1, init.barrier_ids()).unwrap();

        let issue_operation = OperationContext::new(
            DynamicOpId::new(0, 0, 2, StaticOpId::new(102), []),
            OperationKind::AsyncIssue,
            mask,
        );
        let descriptor = PhysicalAccessDescriptor::new(
            PhysicalAccessKind::Write,
            PhysicalAccessSpace::Shared,
            4,
        )
        .unwrap();
        let access = PhysicalAccessBatch::resolve(issue_operation.clone(), descriptor, |_| {
            Ok::<_, std::convert::Infallible>(vec![PhysicalByteSpan::new(
                PhysicalAllocationId::new(77),
                0,
                4,
            )
            .unwrap()])
        })
        .unwrap();
        let completion_plan =
            crate::runtime::PhysicalMbarrierCompletionIssuePlan::single(init.barrier_ids()[0], 16);
        let mut payload =
            AsyncPayloadEffect::new(issue_operation.clone(), [access], completion_plan).unwrap();
        before(
            &state,
            &issue_operation,
            OperationEffect::AsyncPayload(&payload),
        )
        .unwrap();
        let action_ids = payload.completion_plan().apply(&hub).unwrap();
        payload.bind_completion_action_ids(action_ids).unwrap();
        after(
            &state,
            &issue_operation,
            OperationEffect::AsyncPayload(&payload),
        )
        .unwrap();

        let arrive =
            plan_physical_mbarrier_arrive(&fixture.context, &fixture.pointer, mask, None, Some(16))
                .unwrap()
                .unwrap();
        let arrive_operation = fixture.operation(3, mask);
        before(&state, &arrive_operation, staged_arrive(arrive)).unwrap();
        let arrive_outcome = arrive.apply(&hub).unwrap();
        after(
            &state,
            &arrive_operation,
            OperationEffect::MbarrierArrive {
                plan: arrive,
                outcome: Some(arrive_outcome),
            },
        )
        .unwrap();

        // The analysis wait is staged immediately before the numerical future
        // is first polled. A completion may occur in that gap, so it can make a
        // strict waiter ready without having a numeric waker to wake.
        let wait = plan_physical_mbarrier_wait(&fixture.context, &fixture.pointer, mask, 0)
            .unwrap()
            .unwrap();
        let wait_operation = fixture.operation(4, mask);
        before(
            &state,
            &wait_operation,
            OperationEffect::MbarrierWait {
                plan: wait,
                outcome: None,
            },
        )
        .unwrap();

        let action = hub.pending_completion_actions()[0];
        state.before_completion(action).unwrap();
        let completion = hub.apply_completion_detailed(action.id()).unwrap();
        assert!(completion.woken_warp_ids().is_empty());
        state.after_completion(&completion).unwrap();
        after(
            &state,
            &wait_operation,
            OperationEffect::MbarrierWait {
                plan: wait,
                outcome: Some(crate::runtime::PhysicalMbarrierWaitOutcome::new(Some(0))),
            },
        )
        .unwrap();

        assert_eq!(state.result().status(), SyncCheckStatus::Clean);
    }

    #[test]
    fn completion_over_delivery_is_rejected_before_numeric_state_changes() {
        let fixture = BarrierFixture::new();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let state = SyncCheckLaunchState::new();
        let hub = PhysicalBarrierHub::new();
        let init =
            plan_physical_mbarrier_init(&fixture.context, &fixture.pointer, mask, 1).unwrap();
        let init_operation = fixture.operation(0, mask);
        before(
            &state,
            &init_operation,
            OperationEffect::MbarrierInit(&init),
        )
        .unwrap();
        init.apply(&hub).unwrap();
        after(
            &state,
            &init_operation,
            OperationEffect::MbarrierInit(&init),
        )
        .unwrap();
        after_init_fence(&state, &fixture, 1, init.barrier_ids()).unwrap();

        let issue =
            crate::runtime::PhysicalMbarrierCompletionIssuePlan::single(init.barrier_ids()[0], 16);
        let issue_operation = fixture.operation(2, mask);
        before(
            &state,
            &issue_operation,
            OperationEffect::MbarrierCompletionIssue {
                plan: &issue,
                action_ids: None,
            },
        )
        .unwrap();
        let action_ids = issue.apply(&hub).unwrap();
        after(
            &state,
            &issue_operation,
            OperationEffect::MbarrierCompletionIssue {
                plan: &issue,
                action_ids: Some(&action_ids),
            },
        )
        .unwrap();

        let arrive =
            plan_physical_mbarrier_arrive(&fixture.context, &fixture.pointer, mask, None, Some(8))
                .unwrap()
                .unwrap();
        let arrive_operation = fixture.operation(3, mask);
        before(&state, &arrive_operation, staged_arrive(arrive)).unwrap();
        let arrive_outcome = arrive.apply(&hub).unwrap();
        after(
            &state,
            &arrive_operation,
            OperationEffect::MbarrierArrive {
                plan: arrive,
                outcome: Some(arrive_outcome),
            },
        )
        .unwrap();

        let action = hub.pending_completion_actions()[0];
        let before_snapshot = state.snapshot(action.barrier_id());
        let error = <SyncCheckMode as crate::engine_mode::EngineModeImpl>::before_completion(
            &state,
            CompletionActionEffect::PhysicalMbarrier(&action),
        )
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("completed 16 transaction bytes, expected 8"));
        assert_eq!(hub.pending_completion_count(), 1);
        assert_eq!(state.snapshot(action.barrier_id()), before_snapshot);
        let result = state.result();
        assert!(matches!(
            result.findings(),
            [finding]
                if finding.effect() == SyncCheckEffectKind::MbarrierCompletion
                    && matches!(
                        finding.error(),
                        SyncCheckProtocolError::Mbarrier(
                            StrictMbarrierError::TransactionOverDelivery { .. }
                        )
                    )
        ));
    }

    #[test]
    fn named_barrier_gateway_commits_registration_before_resume() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let state = Arc::new(SyncCheckLaunchState::new());
        let report = run_kernel_engine_launch_report::<SyncCheckMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            |mut warp| async move {
                let operation = warp.begin_operation(
                    warp.context(),
                    StaticOpId::new(1000),
                    OperationKind::Collective,
                    [],
                )?;
                warp.named_barrier_sync_with_alignment(
                    Some(&operation),
                    3,
                    64,
                    WarpMask::FULL,
                    true,
                )
                .await?;
                warp.finish_operation(&operation)?;
                Ok(())
            },
        );

        assert!(report.is_success(), "{:?}", report.error());
        let result = state.result();
        assert_eq!(result.status(), SyncCheckStatus::Clean);
        assert_eq!(result.effects().len(), 4);
        assert_eq!(
            result
                .effects()
                .iter()
                .filter(|effect| effect.effect() == SyncCheckEffectKind::NamedBarrierSyncRegister)
                .count(),
            2
        );
        assert_eq!(
            result
                .effects()
                .iter()
                .filter(|effect| effect.effect() == SyncCheckEffectKind::NamedBarrierSyncResume)
                .count(),
            2
        );
        assert!(state
            .named_snapshot(crate::NamedBarrierId::new(0, 3))
            .waiting_warps()
            .is_empty());
        for warp_id in 0..2 {
            let operation = DynamicOpId::new(0, warp_id, 0, StaticOpId::new(1000), []);
            let ResolvedTransitionSummary::Synchronization(summary) = state
                .transition_log()
                .operation_summary(&operation)
                .unwrap()
            else {
                panic!("bar.sync must retain one synchronization summary")
            };
            assert_eq!(summary.resources().len(), 1);
            assert!(matches!(
                summary.resources()[0].key(),
                ResolvedSyncResourceKey::NamedBarrier(_)
            ));
            assert_eq!(summary.resources()[0].generation(), Some(0));
        }
    }

    /// `setmaxnreg` publishes `SetmaxnregRegister` and then `SetmaxnregResume`
    /// under one `DynamicOpId`. The stored payload folds the resume onto the
    /// registration, which is what makes the second publication compare equal
    /// instead of conflicting. Mutation-checked: deleting the `SetmaxnregResume`
    /// arm of `canonical_sync_payload` fails this test with "resolved to
    /// conflicting semantic summaries".
    #[test]
    fn setmaxnreg_register_and_resume_record_one_folded_summary() {
        let topology = LaunchTopology::new(1, 1, SETMAXNREG_WARPS_PER_GROUP).unwrap();
        let state = Arc::new(SyncCheckLaunchState::new());
        let report = run_kernel_engine_launch_report::<SyncCheckMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            |mut warp| async move {
                let operation = warp.begin_operation(
                    warp.context(),
                    StaticOpId::new(1200),
                    OperationKind::Collective,
                    [],
                )?;
                warp.setmaxnreg::<128>(Some(&operation), false, 96).await?;
                warp.finish_operation(&operation)?;
                Ok(())
            },
        );

        assert!(report.is_success(), "{:?}", report.error());
        for warp_id in 0..SETMAXNREG_WARPS_PER_GROUP {
            let operation = DynamicOpId::new(0, warp_id, 0, StaticOpId::new(1200), []);
            let ResolvedTransitionSummary::Synchronization(summary) = state
                .transition_log()
                .operation_summary(&operation)
                .unwrap()
            else {
                panic!("setmaxnreg must retain one synchronization summary")
            };
            assert!(matches!(
                summary.details(),
                OwnedOperationEffect::SetmaxnregRegister(plan)
                    if plan.action() == SetmaxnregAction::Decrease && plan.count() == 96
            ));
        }
    }

    /// The same fold for the TCGEN lifecycle pair. Mutation-checked: deleting
    /// the `TcgenLifecycleResume` arm of `canonical_sync_payload` fails this
    /// test the same way.
    #[test]
    fn tcgen_lifecycle_register_and_resume_record_one_folded_summary() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let state = Arc::new(SyncCheckLaunchState::new());
        let report = run_kernel_engine_launch_report::<SyncCheckMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            |mut warp| async move {
                let operation = warp.begin_operation(
                    warp.context(),
                    StaticOpId::new(1300),
                    OperationKind::Lifecycle,
                    [],
                )?;
                let allocation = warp.tcgen_allocate(Some(&operation), 32, 1, false).await?;
                warp.finish_operation(&operation)?;
                let release = warp.begin_operation(
                    warp.context(),
                    StaticOpId::new(1301),
                    OperationKind::Lifecycle,
                    [],
                )?;
                warp.tcgen_deallocate(
                    Some(&release),
                    allocation.base_column,
                    allocation.columns,
                    1,
                    false,
                )
                .await?;
                warp.finish_operation(&release)?;
                let relinquish = warp.begin_operation(
                    warp.context(),
                    StaticOpId::new(1302),
                    OperationKind::Lifecycle,
                    [],
                )?;
                warp.tcgen_relinquish(Some(&relinquish), 1).await?;
                warp.finish_operation(&relinquish)?;
                Ok(())
            },
        );

        assert!(report.is_success(), "{:?}", report.error());
        let operation = DynamicOpId::new(0, 0, 0, StaticOpId::new(1300), []);
        let ResolvedTransitionSummary::Synchronization(summary) = state
            .transition_log()
            .operation_summary(&operation)
            .unwrap()
        else {
            panic!("tcgen05.alloc must retain one synchronization summary")
        };
        assert!(matches!(
            summary.details(),
            OwnedOperationEffect::TcgenLifecycleRegister(plan)
                if plan.action() == TcgenLifecycleAction::Allocate && plan.columns() == 32
        ));
    }

    #[test]
    fn cta_sync_gateway_accepts_one_static_call_site_across_warps() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let state = Arc::new(SyncCheckLaunchState::with_topology_for_strict_protocols(
            topology,
        ));
        let report = run_kernel_engine_launch_report::<SyncCheckMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            |mut warp| async move {
                let operation = warp.begin_operation(
                    warp.context(),
                    StaticOpId::new(1100),
                    OperationKind::Collective,
                    [],
                )?;
                warp.named_barrier_sync_with_alignment(
                    Some(&operation),
                    0,
                    64,
                    WarpMask::FULL,
                    true,
                )
                .await?;
                warp.finish_operation(&operation)?;
                Ok(())
            },
        );

        assert!(report.is_success(), "{:?}", report.error());
        let result = state.result();
        assert_eq!(result.status(), SyncCheckStatus::Clean);
        assert!(result.findings().is_empty());
        assert!(result.incomplete_reasons().is_empty());
        for warp_id in 0..2 {
            let operation = DynamicOpId::new(0, warp_id, 0, StaticOpId::new(1100), []);
            let ResolvedTransitionSummary::Synchronization(summary) = state
                .transition_log()
                .operation_summary(&operation)
                .unwrap()
            else {
                panic!("cuda.cta_sync must retain one synchronization summary")
            };
            assert!(matches!(
                summary.resources(),
                [resource]
                    if matches!(
                        resource.key(),
                        ResolvedSyncResourceKey::NamedBarrier(id)
                            if id.global_cta_id() == 0 && id.barrier_id() == 0
                    ) && resource.generation() == Some(0)
            ));
            assert!(matches!(
                summary.details(),
                OwnedOperationEffect::NamedBarrierSyncRegister { plan, .. }
                    if plan.aligned()
            ));
        }
    }

    #[test]
    fn cta_sync_gateway_accepts_distinct_inlined_sites_across_warps() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let state = Arc::new(SyncCheckLaunchState::with_topology_for_strict_protocols(
            topology,
        ));
        let report = run_kernel_engine_launch_report::<SyncCheckMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            |mut warp| async move {
                let source = 1200 + warp.context().warp_id_in_cta() as u64;
                let operation = warp.begin_operation(
                    warp.context(),
                    StaticOpId::new(source),
                    OperationKind::Collective,
                    [],
                )?;
                warp.named_barrier_sync_with_alignment(
                    Some(&operation),
                    0,
                    64,
                    WarpMask::FULL,
                    true,
                )
                .await?;
                Ok(())
            },
        );

        assert!(report.is_success(), "{:?}", report.error());
        let result = state.result();
        assert_eq!(result.status(), SyncCheckStatus::Clean);
        assert!(result.findings().is_empty());
    }

    #[test]
    fn cta_sync_gateway_reports_warp_dependent_absence_as_typed_error() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let state = Arc::new(SyncCheckLaunchState::with_topology_for_strict_protocols(
            topology,
        ));
        let report = run_kernel_engine_launch_report::<SyncCheckMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            |mut warp| async move {
                if warp.context().warp_id_in_cta() == 0 {
                    let operation = warp.begin_operation(
                        warp.context(),
                        StaticOpId::new(1300),
                        OperationKind::Collective,
                        [],
                    )?;
                    warp.named_barrier_sync_with_alignment(
                        Some(&operation),
                        0,
                        64,
                        WarpMask::FULL,
                        true,
                    )
                    .await?;
                }
                Ok(())
            },
        );

        assert!(!report.is_success());
        let result = state.result();
        assert_eq!(result.status(), SyncCheckStatus::Error);
        assert!(matches!(
            result.findings(),
            [finding]
                if matches!(
                    finding.error(),
                    SyncCheckProtocolError::NamedBarrier(
                        StrictNamedBarrierError::FullCtaAlignedMissingParticipants {
                            missing_warps,
                            ..
                        }
                    ) if missing_warps.as_ref() == [1]
                )
        ));
    }

    #[test]
    fn cta_sync_gateway_accepts_distinct_sites_on_explicit_barrier_zero() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let state = Arc::new(SyncCheckLaunchState::with_topology_for_strict_protocols(
            topology,
        ));
        let report = run_kernel_engine_launch_report::<SyncCheckMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            |mut warp| async move {
                let warp_id = warp.context().warp_id_in_cta();
                let operation = warp.begin_operation(
                    warp.context(),
                    StaticOpId::new(1400 + warp_id as u64),
                    OperationKind::Collective,
                    [],
                )?;
                warp.named_barrier_sync_with_alignment(
                    Some(&operation),
                    0,
                    64,
                    WarpMask::FULL,
                    true,
                )
                .await?;
                warp.finish_operation(&operation)?;
                Ok(())
            },
        );

        assert!(report.is_success(), "{:?}", report.error());
        let result = state.result();
        assert_eq!(result.status(), SyncCheckStatus::Clean);
        assert!(result.findings().is_empty());
    }

    #[test]
    fn elect_sync_named_barrier_is_rejected_before_protocol_mutation() {
        let fixture = BarrierFixture::new();
        let state = SyncCheckLaunchState::new();
        let active_mask = WarpMask::from_lanes([0]).unwrap();
        let operation = fixture.operation(0, active_mask).with_control_provenance(
            ControlProvenance::ElectSync {
                entry_mask: WarpMask::FULL,
            },
        );
        let plan = plan_named_barrier_arrive(&fixture.context, 4, 32, active_mask)
            .unwrap()
            .unwrap();

        before(
            &state,
            &operation,
            OperationEffect::NamedBarrierArrive {
                plan,
                outcome: None,
            },
        )
        .unwrap_err();

        assert_eq!(state.named_snapshot(plan.barrier_id()).generation(), None);
        let result = state.result();
        assert!(matches!(
            result.findings(),
            [finding]
                if finding.effect() == SyncCheckEffectKind::NamedBarrierArrive
                    && matches!(
                        finding.error(),
                        SyncCheckProtocolError::NamedBarrier(
                            StrictNamedBarrierError::ElectSyncParticipation { .. }
                        )
                    )
        ));
    }

    #[test]
    fn named_barrier_contract_mismatch_is_a_typed_sync_finding() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let state = Arc::new(SyncCheckLaunchState::new());
        let report = run_kernel_engine_launch_report::<SyncCheckMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            |mut warp| async move {
                let expected = if warp.context().global_warp_id() == 0 {
                    64
                } else {
                    32
                };
                let operation = warp.begin_operation(
                    warp.context(),
                    StaticOpId::new(1001),
                    OperationKind::Collective,
                    [],
                )?;
                warp.named_barrier_sync_with_alignment(
                    Some(&operation),
                    4,
                    expected,
                    WarpMask::FULL,
                    true,
                )
                .await?;
                Ok(())
            },
        );

        assert!(!report.is_success());
        let result = state.result();
        assert_eq!(result.status(), SyncCheckStatus::Error);
        assert_eq!(result.findings().len(), 1);
        assert_eq!(
            result.findings()[0].effect(),
            SyncCheckEffectKind::NamedBarrierSyncRegister
        );
        assert!(matches!(
            result.findings()[0].error(),
            SyncCheckProtocolError::NamedBarrier(StrictNamedBarrierError::ContractMismatch { .. })
        ));
    }

    #[test]
    fn use_before_init_is_a_typed_finding() {
        let fixture = BarrierFixture::new();
        let state = SyncCheckLaunchState::new();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let arrive =
            plan_physical_mbarrier_arrive(&fixture.context, &fixture.pointer, mask, None, None)
                .unwrap()
                .unwrap();
        let operation = fixture.operation(0, mask);

        let error = before(&state, &operation, staged_arrive(arrive)).unwrap_err();

        assert!(error
            .to_string()
            .contains("uninitialized physical mbarrier"));
        let result = state.result();
        assert_eq!(result.status(), SyncCheckStatus::Error);
        assert!(matches!(
            result.findings()[0].error(),
            SyncCheckProtocolError::Mbarrier(StrictMbarrierError::Uninitialized { .. })
        ));
        assert_eq!(
            state.snapshot(arrive.barrier_id()).lifecycle(),
            StrictMbarrierLifecycle::Uninitialized
        );
        assert_eq!(
            state.transition_log().operation_summary(operation.id()),
            None
        );
    }

    #[test]
    fn same_warp_use_after_init_does_not_require_an_explicit_init_fence() {
        let fixture = BarrierFixture::new();
        let state = SyncCheckLaunchState::new();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let init =
            plan_physical_mbarrier_init(&fixture.context, &fixture.pointer, mask, 1).unwrap();
        let arrive =
            plan_physical_mbarrier_arrive(&fixture.context, &fixture.pointer, mask, None, None)
                .unwrap()
                .unwrap();
        let init_operation = fixture.operation(0, mask);
        let arrive_operation = fixture.operation(1, mask);

        before(
            &state,
            &init_operation,
            OperationEffect::MbarrierInit(&init),
        )
        .unwrap();
        after(
            &state,
            &init_operation,
            OperationEffect::MbarrierInit(&init),
        )
        .unwrap();

        before(&state, &arrive_operation, staged_arrive(arrive)).unwrap();
        after(&state, &arrive_operation, committed_arrive(arrive, 0, true)).unwrap();

        assert!(state.result().findings().is_empty());
    }

    #[test]
    fn over_arrival_is_rejected_before_commit() {
        let fixture = BarrierFixture::new();
        let state = SyncCheckLaunchState::new();
        let one_lane = WarpMask::from_lanes([0]).unwrap();
        let two_lanes = WarpMask::from_lanes([0, 1]).unwrap();
        let init =
            plan_physical_mbarrier_init(&fixture.context, &fixture.pointer, one_lane, 1).unwrap();
        let init_op = fixture.operation(0, one_lane);
        before(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after_init_fence(&state, &fixture, 1, init.barrier_ids()).unwrap();
        let arrive = plan_physical_mbarrier_arrive(
            &fixture.context,
            &fixture.pointer,
            two_lanes,
            None,
            None,
        )
        .unwrap()
        .unwrap();
        let arrive_op = fixture.operation(2, two_lanes);

        before(&state, &arrive_op, staged_arrive(arrive)).unwrap_err();

        let result = state.result();
        assert!(matches!(
            result.findings()[0].error(),
            SyncCheckProtocolError::Mbarrier(StrictMbarrierError::ArrivalOverflow {
                expected: 1,
                completed: 2,
                ..
            })
        ));
        assert_eq!(state.snapshot(arrive.barrier_id()).arrival_count(), 0);
    }

    #[test]
    fn completed_unconsumed_generation_rejects_arrive_and_reinit() {
        let fixture = BarrierFixture::new();
        let state = SyncCheckLaunchState::new();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let init =
            plan_physical_mbarrier_init(&fixture.context, &fixture.pointer, mask, 1).unwrap();
        let arrive =
            plan_physical_mbarrier_arrive(&fixture.context, &fixture.pointer, mask, None, None)
                .unwrap()
                .unwrap();
        let init_op = fixture.operation(0, mask);
        let arrive_op = fixture.operation(2, mask);
        before(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after_init_fence(&state, &fixture, 1, init.barrier_ids()).unwrap();
        before(&state, &arrive_op, staged_arrive(arrive)).unwrap();
        after(&state, &arrive_op, committed_arrive(arrive, 0, true)).unwrap();

        before(&state, &fixture.operation(3, mask), staged_arrive(arrive)).unwrap_err();
        before(
            &state,
            &fixture.operation(4, mask),
            OperationEffect::MbarrierInit(&init),
        )
        .unwrap_err();

        let result = state.result();
        assert_eq!(result.findings().len(), 2);
        assert!(matches!(
            result.findings()[0].error(),
            SyncCheckProtocolError::Mbarrier(StrictMbarrierError::ArriveBeforeConsumption { .. })
        ));
        assert!(matches!(
            result.findings()[1].error(),
            SyncCheckProtocolError::Mbarrier(
                StrictMbarrierError::ReinitializeBeforeConsumption { .. }
            )
        ));
        assert_eq!(
            state.snapshot(arrive.barrier_id()).lifecycle(),
            StrictMbarrierLifecycle::CompletedUnconsumed
        );
    }

    #[test]
    fn successful_ready_wait_consumes_and_permits_reuse() {
        let fixture = BarrierFixture::new();
        let state = SyncCheckLaunchState::new();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let init =
            plan_physical_mbarrier_init(&fixture.context, &fixture.pointer, mask, 1).unwrap();
        let arrive =
            plan_physical_mbarrier_arrive(&fixture.context, &fixture.pointer, mask, None, None)
                .unwrap()
                .unwrap();
        let wait = plan_physical_mbarrier_wait(&fixture.context, &fixture.pointer, mask, 0)
            .unwrap()
            .unwrap();
        let init_op = fixture.operation(0, mask);
        let arrive_op = fixture.operation(2, mask);
        let wait_op = fixture.operation(3, mask);
        before(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after_init_fence(&state, &fixture, 1, init.barrier_ids()).unwrap();
        before(&state, &arrive_op, staged_arrive(arrive)).unwrap();
        after(&state, &arrive_op, committed_arrive(arrive, 0, true)).unwrap();
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
        assert_eq!(
            state.snapshot(arrive.barrier_id()).lifecycle(),
            StrictMbarrierLifecycle::Consumed
        );

        let reuse_op = fixture.operation(4, mask);
        before(&state, &reuse_op, staged_arrive(arrive)).unwrap();
        after(&state, &reuse_op, committed_arrive(arrive, 1, true)).unwrap();

        let snapshot = state.snapshot(arrive.barrier_id());
        assert_eq!(snapshot.generation(), Some(1));
        assert_eq!(
            snapshot.lifecycle(),
            StrictMbarrierLifecycle::CompletedUnconsumed
        );
        assert_eq!(state.result().status(), SyncCheckStatus::Clean);
    }

    #[test]
    fn warp_sync_records_an_actor_local_resolved_effect() {
        let fixture = BarrierFixture::new();
        let state = SyncCheckLaunchState::new();
        let mask = WarpMask::FULL;

        // Give the warp a causal history BEFORE the rendezvous. This is what
        // makes the clock assertion below bite: with no prior event the warp
        // clock is `None`, so the generic `after_effect` tail would record no
        // clock either and the assertion would hold no matter how the effect
        // was routed. One `mbarrier.init` is enough, and its mask must be
        // single-lane -- synccheck rejects several lanes initializing one
        // barrier.
        let init_mask = WarpMask::from_lanes([0]).unwrap();
        let init =
            plan_physical_mbarrier_init(&fixture.context, &fixture.pointer, init_mask, 1).unwrap();
        let init_operation = fixture.operation(0, init_mask);
        before(
            &state,
            &init_operation,
            OperationEffect::MbarrierInit(&init),
        )
        .unwrap();
        after(
            &state,
            &init_operation,
            OperationEffect::MbarrierInit(&init),
        )
        .unwrap();
        assert!(state
            .transition_log()
            .operation_clock(init_operation.id())
            .is_some());

        let operation = fixture.operation(1, mask);
        after(
            &state,
            &operation,
            OperationEffect::WarpSync(crate::WarpSyncEffect::new(mask)),
        )
        .unwrap();

        assert!(matches!(
            state.transition_log().operation_summary(operation.id()),
            Some(ResolvedTransitionSummary::Synchronization(summary))
                if summary.resources().is_empty()
                    && matches!(
                        summary.details(),
                        crate::OwnedOperationEffect::WarpSync(effect)
                            if effect.mask() == mask
                    )
        ));
        // The rendezvous is actor-local, so it records no causal clock even
        // though this warp now has one. Routing it through the generic
        // `after_effect` tail would attach the warp clock here.
        assert!(state
            .transition_log()
            .operation_clock(operation.id())
            .is_none());
    }

    #[test]
    fn registered_wait_rechecks_to_ready_without_false_incomplete() {
        let fixture = BarrierFixture::new();
        let state = SyncCheckLaunchState::new();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let init =
            plan_physical_mbarrier_init(&fixture.context, &fixture.pointer, mask, 1).unwrap();
        let wait = plan_physical_mbarrier_wait(&fixture.context, &fixture.pointer, mask, 0)
            .unwrap()
            .unwrap();
        let arrive =
            plan_physical_mbarrier_arrive(&fixture.context, &fixture.pointer, mask, None, None)
                .unwrap()
                .unwrap();
        let init_op = fixture.operation(0, mask);
        let wait_op = fixture.operation(2, mask);
        let arrive_op = fixture.operation(3, mask);
        before(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after_init_fence(&state, &fixture, 1, init.barrier_ids()).unwrap();
        before(
            &state,
            &wait_op,
            OperationEffect::MbarrierWait {
                plan: wait,
                outcome: None,
            },
        )
        .unwrap();
        before(&state, &arrive_op, staged_arrive(arrive)).unwrap();
        after(&state, &arrive_op, committed_arrive(arrive, 0, true)).unwrap();
        after(
            &state,
            &wait_op,
            OperationEffect::MbarrierWait {
                plan: wait,
                outcome: Some(crate::runtime::PhysicalMbarrierWaitOutcome::new(Some(0))),
            },
        )
        .unwrap();

        let result = state.result();
        assert_eq!(result.status(), SyncCheckStatus::Clean);
        assert!(result.incomplete_reasons().is_empty());
        assert_eq!(
            state.snapshot(arrive.barrier_id()).lifecycle(),
            StrictMbarrierLifecycle::Consumed
        );
        let wait_record = result.effects().last().unwrap();
        assert!(matches!(
            wait_record.outcome(),
            SyncCheckEffectOutcome::MbarrierWait {
                staged: SyncCheckWaitState::Registered { generation: 0 },
                committed: Some(SyncCheckWaitState::Ready {
                    generation: Some(0),
                    ..
                }),
            }
        ));
    }

    #[test]
    fn registered_wait_rejects_a_mismatched_numeric_completion_generation() {
        let fixture = BarrierFixture::new();
        let state = SyncCheckLaunchState::new();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let init =
            plan_physical_mbarrier_init(&fixture.context, &fixture.pointer, mask, 1).unwrap();
        let wait = plan_physical_mbarrier_wait(&fixture.context, &fixture.pointer, mask, 0)
            .unwrap()
            .unwrap();
        let arrive =
            plan_physical_mbarrier_arrive(&fixture.context, &fixture.pointer, mask, None, None)
                .unwrap()
                .unwrap();
        let init_op = fixture.operation(0, mask);
        let wait_op = fixture.operation(2, mask);
        let arrive_op = fixture.operation(3, mask);
        before(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after_init_fence(&state, &fixture, 1, init.barrier_ids()).unwrap();
        before(
            &state,
            &wait_op,
            OperationEffect::MbarrierWait {
                plan: wait,
                outcome: None,
            },
        )
        .unwrap();
        before(&state, &arrive_op, staged_arrive(arrive)).unwrap();
        after(&state, &arrive_op, committed_arrive(arrive, 0, true)).unwrap();

        let error = after(
            &state,
            &wait_op,
            OperationEffect::MbarrierWait {
                plan: wait,
                outcome: Some(crate::runtime::PhysicalMbarrierWaitOutcome::new(Some(1))),
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("disagrees on generation"));
    }

    #[test]
    fn successful_wait_without_typed_completion_fails_closed() {
        let fixture = BarrierFixture::new();
        let state = SyncCheckLaunchState::new();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let init =
            plan_physical_mbarrier_init(&fixture.context, &fixture.pointer, mask, 1).unwrap();
        let wait = plan_physical_mbarrier_wait(&fixture.context, &fixture.pointer, mask, 0)
            .unwrap()
            .unwrap();
        let init_op = fixture.operation(0, mask);
        let wait_op = fixture.operation(2, mask);
        before(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after_init_fence(&state, &fixture, 1, init.barrier_ids()).unwrap();
        before(
            &state,
            &wait_op,
            OperationEffect::MbarrierWait {
                plan: wait,
                outcome: None,
            },
        )
        .unwrap();

        // Model `after_effect` following a numeric wake produced by a
        // completion transition that has not yet been delivered to the mode.
        let error = after(
            &state,
            &wait_op,
            OperationEffect::MbarrierWait {
                plan: wait,
                outcome: Some(crate::runtime::PhysicalMbarrierWaitOutcome::new(None)),
            },
        )
        .unwrap_err();

        let result = state.result();
        assert_eq!(result.status(), SyncCheckStatus::Incomplete);
        assert!(error.to_string().contains("without a typed completion"));
        assert!(matches!(
            result.incomplete_reasons()[0],
            SyncCheckIncompleteReason::CompletionTransitionUnobserved { generation: 0, .. }
        ));
        assert_eq!(result.effects().len(), 1);
        assert_eq!(
            result.effects()[0].effect(),
            SyncCheckEffectKind::MbarrierInit
        );
        assert_eq!(
            state.snapshot(wait.barrier_id()).lifecycle(),
            StrictMbarrierLifecycle::Pending
        );
    }

    #[test]
    fn cross_warp_mbarrier_use_requires_init_happens_before() {
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
        let contexts = topology.warp_contexts().collect::<Vec<_>>();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let init = plan_physical_mbarrier_init(&contexts[0], &pointer, mask, 1).unwrap();
        let arrive = plan_physical_mbarrier_arrive(&contexts[1], &pointer, mask, None, None)
            .unwrap()
            .unwrap();
        let init_op = OperationContext::new(
            DynamicOpId::new(0, 0, 0, StaticOpId::new(178), []),
            OperationKind::Barrier,
            mask,
        );
        let fence_op = OperationContext::new(
            DynamicOpId::new(0, 0, 1, StaticOpId::new(180), []),
            OperationKind::MbarrierInitFence,
            WarpMask::FULL,
        );
        let arrive_op = OperationContext::new(
            DynamicOpId::new(0, 1, 0, StaticOpId::new(179), []),
            OperationKind::Barrier,
            mask,
        );
        let state = SyncCheckLaunchState::new();

        before(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after(
            &state,
            &fence_op,
            OperationEffect::MbarrierInitFence {
                barrier_ids: init.barrier_ids(),
            },
        )
        .unwrap();
        before(&state, &arrive_op, staged_arrive(arrive)).unwrap();
        let error = after(&state, &arrive_op, committed_arrive(arrive, 0, true)).unwrap_err();

        assert!(error.to_string().contains("does not happen before"));
        let result = state.result();
        assert_eq!(result.status(), SyncCheckStatus::Error);
        assert!(matches!(
            result.findings()[0].error(),
            SyncCheckProtocolError::Causality(SyncCausalityError::InitNotHappensBeforeUse { .. })
        ));
        assert_eq!(
            state
                .inner
                .lock()
                .expect("sync-check state poisoned")
                .causality
                .warp_count(),
            2
        );
    }

    #[test]
    fn next_mbarrier_generation_requires_consumption_return_happens_before() {
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
        let contexts = topology.warp_contexts().collect::<Vec<_>>();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let init = plan_physical_mbarrier_init(&contexts[0], &pointer, mask, 1).unwrap();
        let arrive = plan_physical_mbarrier_arrive(&contexts[0], &pointer, mask, None, None)
            .unwrap()
            .unwrap();
        let wait = plan_physical_mbarrier_wait(&contexts[1], &pointer, mask, 0)
            .unwrap()
            .unwrap();
        let state = SyncCheckLaunchState::new();
        let init_op = OperationContext::new(
            DynamicOpId::new(0, 0, 0, StaticOpId::new(174), []),
            OperationKind::Barrier,
            mask,
        );
        let fence_op = OperationContext::new(
            DynamicOpId::new(0, 0, 1, StaticOpId::new(175), []),
            OperationKind::MbarrierInitFence,
            WarpMask::FULL,
        );
        let arrive0_op = OperationContext::new(
            DynamicOpId::new(0, 0, 2, StaticOpId::new(176), []),
            OperationKind::Barrier,
            mask,
        );
        let wait_op = OperationContext::new(
            DynamicOpId::new(0, 1, 0, StaticOpId::new(177), []),
            OperationKind::Barrier,
            mask,
        );
        let arrive1_op = OperationContext::new(
            DynamicOpId::new(0, 0, 3, StaticOpId::new(178), []),
            OperationKind::Barrier,
            mask,
        );

        before(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after(&state, &init_op, OperationEffect::MbarrierInit(&init)).unwrap();
        after(
            &state,
            &fence_op,
            OperationEffect::MbarrierInitFence {
                barrier_ids: init.barrier_ids(),
            },
        )
        .unwrap();
        state
            .inner
            .lock()
            .expect("sync-check state poisoned")
            .causality
            .synchronize_warps(&[0, 1])
            .unwrap();
        before(&state, &arrive0_op, staged_arrive(arrive)).unwrap();
        after(&state, &arrive0_op, committed_arrive(arrive, 0, true)).unwrap();
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

        before(&state, &arrive1_op, staged_arrive(arrive)).unwrap();
        let error = after(&state, &arrive1_op, committed_arrive(arrive, 1, true)).unwrap_err();

        assert!(error
            .to_string()
            .contains("consumption does not happen before"));
        assert!(matches!(
            state.result().findings().last().unwrap().error(),
            SyncCheckProtocolError::Causality(
                SyncCausalityError::PriorGenerationConsumptionNotHappensBefore {
                    prior_generation: 0,
                    next_generation: 1,
                    consumption_operation: Some(operation),
                    ..
                }
            ) if operation == wait_op.id()
        ));
    }

    #[test]
    fn cross_warp_numeric_completion_is_acquired_after_init_publication() {
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
        let mask = WarpMask::from_lanes([0]).unwrap();
        let first_context = topology.warp_contexts().next().unwrap();
        let barrier_id = plan_physical_mbarrier_arrive(&first_context, &pointer, mask, None, None)
            .unwrap()
            .unwrap()
            .barrier_id();
        let state = Arc::new(SyncCheckLaunchState::new());

        let stats = run_kernel_engine_launch::<SyncCheckMode, _, _>(
            physical,
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let pointer = pointer.clone();
                async move {
                    let context = warp.context().with_active_mask(mask);
                    if warp.context().global_warp_id() == 0 {
                        let init_op = warp.begin_operation(
                            context,
                            StaticOpId::new(180),
                            OperationKind::Barrier,
                            [],
                        )?;
                        warp.mbarrier_init(Some(&init_op), &pointer, mask, &WarpValue::splat(1))?;
                        warp.finish_operation(&init_op)?;

                        let fence_op = warp.begin_operation(
                            warp.context().with_active_mask(WarpMask::FULL),
                            StaticOpId::new(184),
                            OperationKind::MbarrierInitFence,
                            [],
                        )?;
                        warp.mbarrier_init_fence(Some(&fence_op))?;
                        warp.finish_operation(&fence_op)?;
                    }
                    let sync_context = warp.context().with_active_mask(WarpMask::FULL);
                    let sync_op = warp.begin_operation(
                        sync_context,
                        StaticOpId::new(183),
                        OperationKind::Collective,
                        [],
                    )?;
                    warp.named_barrier_sync_with_alignment(
                        Some(&sync_op),
                        1,
                        64,
                        WarpMask::FULL,
                        true,
                    )
                    .await?;
                    warp.finish_operation(&sync_op)?;

                    if warp.context().global_warp_id() == 0 {
                        let wait_op = warp.begin_operation(
                            context,
                            StaticOpId::new(181),
                            OperationKind::Barrier,
                            [],
                        )?;
                        warp.mbarrier_wait(Some(&wait_op), &pointer, mask, &WarpValue::splat(0))
                            .await?;
                        warp.finish_operation(&wait_op)?;
                    } else {
                        let arrive_op = warp.begin_operation(
                            context,
                            StaticOpId::new(182),
                            OperationKind::Barrier,
                            [],
                        )?;
                        warp.mbarrier_arrive::<false, false>(
                            Some(&arrive_op),
                            &pointer,
                            mask,
                            None,
                            None,
                            None,
                            true,
                        )?;
                        warp.finish_operation(&arrive_op)?;
                    }
                    Ok(())
                }
            },
        )
        .unwrap();

        assert_eq!(stats.completed_task_count, 2);
        assert_eq!(
            state.snapshot(barrier_id).lifecycle(),
            StrictMbarrierLifecycle::Consumed
        );
        let result = state.result();
        assert_eq!(result.status(), SyncCheckStatus::Clean);
        assert!(result.effects().iter().any(|record| matches!(
            record.outcome(),
            SyncCheckEffectOutcome::MbarrierWait {
                committed: Some(SyncCheckWaitState::Ready {
                    generation: Some(0),
                    ..
                }),
                ..
            }
        )));
    }

    #[test]
    fn invalid_expected_arrivals_rejects_before_numeric_state() {
        let fixture = BarrierFixture::new();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let init =
            plan_physical_mbarrier_init(&fixture.context, &fixture.pointer, mask, 0).unwrap();
        let barrier_id = init.barrier_ids()[0];
        let state = Arc::new(SyncCheckLaunchState::new());
        let pointer = fixture.pointer.clone();

        let error = run_kernel_engine_launch::<SyncCheckMode, _, _>(
            fixture.physical,
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let pointer = pointer.clone();
                async move {
                    let operation = warp.begin_operation(
                        warp.context().with_active_mask(mask),
                        StaticOpId::new(200),
                        OperationKind::Barrier,
                        [],
                    )?;
                    warp.mbarrier_init(Some(&operation), &pointer, mask, &WarpValue::splat(0))?;
                    Ok(())
                }
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("expected arrivals in 1..="));
        assert_eq!(
            state.snapshot(barrier_id).lifecycle(),
            StrictMbarrierLifecycle::Uninitialized
        );
        assert!(matches!(
            state.result().findings()[0].error(),
            SyncCheckProtocolError::Mbarrier(StrictMbarrierError::InvalidExpectedArrivals {
                expected: 0,
                ..
            })
        ));
        assert!(state.result().incomplete_reasons().is_empty());
    }

    #[test]
    fn pre_effect_rejection_prevents_numeric_barrier_mutation() {
        let fixture = BarrierFixture::new();
        let one_lane = WarpMask::from_lanes([0]).unwrap();
        let two_lanes = WarpMask::from_lanes([0, 1]).unwrap();
        let arrive =
            plan_physical_mbarrier_arrive(&fixture.context, &fixture.pointer, one_lane, None, None)
                .unwrap()
                .unwrap();
        let barrier_id = arrive.barrier_id();
        let state = Arc::new(SyncCheckLaunchState::new());
        let pointer = fixture.pointer.clone();

        let stats = run_kernel_engine_launch::<SyncCheckMode, _, _>(
            fixture.physical,
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let pointer = pointer.clone();
                async move {
                    let init_op = warp.begin_operation(
                        warp.context().with_active_mask(one_lane),
                        StaticOpId::new(210),
                        OperationKind::Barrier,
                        [],
                    )?;
                    warp.mbarrier_init(Some(&init_op), &pointer, one_lane, &WarpValue::splat(2))?;
                    warp.finish_operation(&init_op)?;

                    let fence_op = warp.begin_operation(
                        warp.context().with_active_mask(WarpMask::FULL),
                        StaticOpId::new(213),
                        OperationKind::MbarrierInitFence,
                        [],
                    )?;
                    warp.mbarrier_init_fence(Some(&fence_op))?;
                    warp.finish_operation(&fence_op)?;

                    let complete_op = warp.begin_operation(
                        warp.context().with_active_mask(two_lanes),
                        StaticOpId::new(211),
                        OperationKind::Barrier,
                        [],
                    )?;
                    warp.mbarrier_arrive::<false, false>(
                        Some(&complete_op),
                        &pointer,
                        two_lanes,
                        None,
                        None,
                        None,
                        true,
                    )?;
                    warp.finish_operation(&complete_op)?;

                    let rejected_op = warp.begin_operation(
                        warp.context().with_active_mask(one_lane),
                        StaticOpId::new(212),
                        OperationKind::Barrier,
                        [],
                    )?;
                    let error = warp
                        .mbarrier_arrive::<false, false>(
                            Some(&rejected_op),
                            &pointer,
                            one_lane,
                            None,
                            None,
                            None,
                            true,
                        )
                        .unwrap_err();
                    assert!(error
                        .to_string()
                        .contains("before generation 0 was consumed"));

                    // If the rejected engine call had reached the numeric hub,
                    // generation one would already contain one arrival and this
                    // exact two-arrival update would overflow.
                    warp.kernel().services().mbarriers().arrive(
                        barrier_id,
                        warp.context().global_warp_id(),
                        2,
                    )?;
                    Ok(())
                }
            },
        )
        .unwrap();

        assert_eq!(stats.completed_task_count, 1);
        assert_eq!(state.result().status(), SyncCheckStatus::Error);
        assert_eq!(
            state.snapshot(barrier_id).lifecycle(),
            StrictMbarrierLifecycle::CompletedUnconsumed
        );
    }
}
