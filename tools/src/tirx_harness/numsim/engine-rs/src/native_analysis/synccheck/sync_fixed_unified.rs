//! Synccheck-owned unified fixed-program protocol state.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use crate::resolved_transition::{
    FixedSyncLogSnapshot, FixedSyncSetmaxParticipantSnapshot, FixedSyncSnapshotRecord,
};
use crate::setmaxnreg_verifier::{SetmaxnregVerifierCore, SetmaxnregVerifierRequestDisposition};
use crate::strict_cluster_barrier::StrictClusterBarrierSemanticState;
use crate::strict_mbarrier::StrictMbarrierSemanticState;
use crate::strict_named_barrier::StrictNamedBarrierSemanticState;
use crate::{
    setmaxnreg_default_register_count, ClusterBarrierId, DynamicOpId, NamedBarrierId,
    OwnedOperationEffect, PhysicalBarrierId, ResolvedSyncResourceKey,
    ResolvedSynchronizationEffect, ResolvedTransitionLog, ResolvedTransitionSummary,
    SetmaxnregAction, SetmaxnregResource, StrictClusterBarrierProtocol,
    StrictMbarrierCompletionToken, StrictMbarrierProtocol, StrictMbarrierWaitOutcome,
    StrictNamedBarrierOutcome, StrictNamedBarrierProtocol, SyncTransitionSystem,
    TcgenLifecycleAction, WarpMask, SETMAXNREG_WARPS_PER_GROUP, TMEM_COLUMN_CAPACITY,
};

// Projection staging/building is independent by operation chunk or protocol
// resource. Eight workers materially reduce large fixed-trace construction
// without approaching the numerical executor's worker fanout.
const MAX_FIXED_SYNC_BUILD_WORKERS: usize = 8;

type FixedTcgenAllocation = crate::TcgenAllocation;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FixedSyncCommandId(usize);

impl FixedSyncCommandId {
    pub const fn get(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FixedSyncCompletionId {
    issuer: DynamicOpId,
    ordinal: usize,
}

impl FixedSyncCompletionId {
    pub const fn issuer(&self) -> &DynamicOpId {
        &self.issuer
    }

    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FixedSyncTransition {
    Issue(FixedSyncCommandId),
    Complete(FixedSyncCompletionId),
    SetmaxGrant(SetmaxnregResource),
    ValidateExit,
}

#[derive(Clone, Debug)]
struct FixedSyncCommand {
    witness: DynamicOpId,
    participants: CompactSlice<usize>,
    kind: FixedSyncCommandKind,
    canonical_generation: Option<u64>,
    initial_causal_clock: Option<crate::SyncVectorClock>,
    causal_clock: Option<crate::SyncVectorClock>,
}

struct StagedFixedSyncCommand {
    command: FixedSyncCommand,
    participant_order: CompactSlice<(usize, u64)>,
}

type StagedFixedSyncProjections = HashMap<FixedSyncProjectionKey, Vec<StagedFixedSyncCommand>>;

#[derive(Clone, Debug)]
enum CompactSlice<T> {
    One([T; 1]),
    Many(Box<[T]>),
}

impl<T> CompactSlice<T> {
    const fn one(value: T) -> Self {
        Self::One([value])
    }
}

impl<T> From<Box<[T]>> for CompactSlice<T> {
    fn from(values: Box<[T]>) -> Self {
        let mut values = values.into_vec();
        if values.len() == 1 {
            Self::One([values.pop().expect("one compact value")])
        } else {
            Self::Many(values.into_boxed_slice())
        }
    }
}

impl<T> std::ops::Deref for CompactSlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::One(value) => value,
            Self::Many(values) => values,
        }
    }
}

#[derive(Clone, Debug)]
enum FixedSyncCommandKind {
    MbarrierInvalidate {
        barrier_ids: Box<[PhysicalBarrierId]>,
    },
    MbarrierInit {
        barrier_ids: Box<[PhysicalBarrierId]>,
        expected_arrivals: u64,
    },
    MbarrierInitFence {
        barrier_ids: Box<[PhysicalBarrierId]>,
    },
    MbarrierArrive {
        arrivals: Box<[(PhysicalBarrierId, u64, Option<u64>, bool)]>,
    },
    MbarrierExpectTx {
        expectations: Box<[(PhysicalBarrierId, u64)]>,
    },
    MbarrierWait {
        barrier_id: PhysicalBarrierId,
        requested_phase: u64,
        conditional: bool,
    },
    MbarrierWaitBatch {
        waits: Box<[(PhysicalBarrierId, u64, Option<u64>)]>,
        conditional: bool,
    },
    MbarrierCompletionIssue {
        completions: Box<[FixedSyncMbarrierCompletion]>,
    },
    NamedBarrierArrive {
        barrier_id: NamedBarrierId,
        expected_arrivals: u64,
        warp_id: usize,
        arrival_mask: WarpMask,
    },
    NamedBarrierSync {
        barrier_id: NamedBarrierId,
        expected_arrivals: u64,
        warp_id: usize,
        arrival_mask: WarpMask,
        aligned: bool,
    },
    ClusterBarrierArrive {
        barrier_id: ClusterBarrierId,
        warp_id: usize,
        arrival_mask: WarpMask,
        participant_warps: Box<[usize]>,
    },
    ClusterBarrierWait {
        barrier_id: ClusterBarrierId,
        warp_id: usize,
        arrival_mask: WarpMask,
        participant_warps: Box<[usize]>,
    },
    Setmax {
        request: SetmaxnregResource,
        action: SetmaxnregAction,
        target_count: i64,
    },
    TcgenLifecycle(FixedSyncTcgenRequest),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum FixedSyncMbarrierCompletionKind {
    Transaction {
        transactions: u64,
    },
    Arrival {
        warp_id: usize,
        arrival_count: u64,
        /// Pending arrivals this issue raises on the captured generation before
        /// its deferred arrive-on lands. `cp.async.mbarrier.arrive` without
        /// `.noinc` raises one per issuing lane; every other arrival-completion
        /// source raises none.
        pending_increase: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct FixedSyncMbarrierCompletion {
    barrier_id: PhysicalBarrierId,
    kind: FixedSyncMbarrierCompletionKind,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct FixedSyncTcgenRequest {
    kernel_index: usize,
    action: TcgenLifecycleAction,
    address: u32,
    columns: usize,
    cta_group: usize,
    participant_ctas: Box<[usize]>,
    participant_warps: Box<[usize]>,
    exclusive: bool,
    capacity: usize,
    canonical_allocation: Option<FixedTcgenAllocation>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct FixedSyncTcgenCtaState {
    allocations: Vec<FixedTcgenAllocation>,
    relinquished: bool,
    cta_group: Option<usize>,
    last_allocation_columns: Option<usize>,
}

#[derive(Clone, Debug)]
struct PendingMbarrierCompletion {
    barrier_id: PhysicalBarrierId,
    generation: u64,
    kind: FixedSyncMbarrierCompletionKind,
    token: StrictMbarrierCompletionToken,
}

impl PendingMbarrierCompletion {
    fn semantic_key(&self) -> (PhysicalBarrierId, u64, FixedSyncMbarrierCompletionKind) {
        (self.barrier_id, self.generation, self.kind)
    }
}

#[derive(Clone)]
pub struct FixedSyncState {
    warp_cursors: Box<[usize]>,
    blocked_warps: BTreeMap<usize, FixedSyncCommandId>,
    cluster_waits: BTreeMap<(ClusterBarrierId, u64, usize), FixedSyncCommandId>,
    mbarriers: StrictMbarrierProtocol,
    named_barriers: StrictNamedBarrierProtocol,
    cluster_barriers: StrictClusterBarrierProtocol,
    setmax_pools: BTreeMap<(usize, usize), SetmaxnregVerifierCore>,
    pending_completions: BTreeMap<FixedSyncCompletionId, PendingMbarrierCompletion>,
    tcgen_ctas: BTreeMap<(usize, usize), FixedSyncTcgenCtaState>,
    exit_validated: bool,
}

impl FixedSyncState {
    fn mbarrier_key(&self) -> StrictMbarrierSemanticState {
        self.mbarriers.semantic_state()
    }

    fn named_barrier_key(&self) -> StrictNamedBarrierSemanticState {
        self.named_barriers.semantic_state()
    }

    fn cluster_barrier_key(&self) -> StrictClusterBarrierSemanticState {
        self.cluster_barriers.semantic_state()
    }

    fn pending_completion_keys(
        &self,
    ) -> impl Iterator<
        Item = (
            &FixedSyncCompletionId,
            (PhysicalBarrierId, u64, FixedSyncMbarrierCompletionKind),
        ),
    > {
        self.pending_completions
            .iter()
            .map(|(id, completion)| (id, completion.semantic_key()))
    }
}

impl fmt::Debug for FixedSyncState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FixedSyncState")
            .field("warp_cursors", &self.warp_cursors)
            .field("blocked_warps", &self.blocked_warps)
            .field("cluster_waits", &self.cluster_waits)
            .field("mbarriers", &self.mbarrier_key())
            .field("named_barriers", &self.named_barrier_key())
            .field("cluster_barriers", &self.cluster_barrier_key())
            .field("setmax_pools", &self.setmax_pools)
            .field(
                "pending_completions",
                &self.pending_completion_keys().collect::<Vec<_>>(),
            )
            .field("tcgen_ctas", &self.tcgen_ctas)
            .field("exit_validated", &self.exit_validated)
            .finish()
    }
}

impl PartialEq for FixedSyncState {
    fn eq(&self, other: &Self) -> bool {
        self.warp_cursors == other.warp_cursors
            && self.blocked_warps == other.blocked_warps
            && self.cluster_waits == other.cluster_waits
            && self.setmax_pools == other.setmax_pools
            && self
                .pending_completion_keys()
                .eq(other.pending_completion_keys())
            && self.tcgen_ctas == other.tcgen_ctas
            && self.exit_validated == other.exit_validated
            && self.mbarrier_key() == other.mbarrier_key()
            && self.named_barrier_key() == other.named_barrier_key()
            && self.cluster_barrier_key() == other.cluster_barrier_key()
    }
}

impl Eq for FixedSyncState {}

impl Hash for FixedSyncState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.warp_cursors.hash(state);
        self.blocked_warps.hash(state);
        self.cluster_waits.hash(state);
        self.setmax_pools.hash(state);
        for completion in self.pending_completion_keys() {
            completion.hash(state);
        }
        self.tcgen_ctas.hash(state);
        self.exit_validated.hash(state);
        self.mbarrier_key().hash(state);
        self.named_barrier_key().hash(state);
        self.cluster_barrier_key().hash(state);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FixedSyncProtocolKind {
    Mbarrier,
    NamedBarrier,
    ClusterBarrier,
    Setmaxnreg,
    TcgenLifecycle,
    Internal,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FixedSyncProgramError {
    DisabledTransition {
        transition: FixedSyncTransition,
    },
    Protocol {
        kind: FixedSyncProtocolKind,
        operation: DynamicOpId,
        details: Box<str>,
    },
    CausalProtocol {
        kind: FixedSyncProtocolKind,
        operation: DynamicOpId,
        related_operations: Box<[DynamicOpId]>,
        details: Box<str>,
    },
    Incomplete {
        kind: FixedSyncProtocolKind,
        operation: Option<DynamicOpId>,
        details: Box<str>,
    },
}

impl FixedSyncProgramError {
    pub const fn is_incomplete(&self) -> bool {
        matches!(self, Self::Incomplete { .. })
    }

    pub const fn operation(&self) -> Option<&DynamicOpId> {
        match self {
            Self::Protocol { operation, .. } | Self::CausalProtocol { operation, .. } => {
                Some(operation)
            }
            Self::Incomplete { operation, .. } => operation.as_ref(),
            Self::DisabledTransition { .. } => None,
        }
    }

    pub const fn protocol_kind(&self) -> Option<FixedSyncProtocolKind> {
        match self {
            Self::Protocol { kind, .. }
            | Self::CausalProtocol { kind, .. }
            | Self::Incomplete { kind, .. } => Some(*kind),
            Self::DisabledTransition { .. } => None,
        }
    }

    pub fn related_operations(&self) -> &[DynamicOpId] {
        match self {
            Self::CausalProtocol {
                related_operations, ..
            } => related_operations,
            Self::DisabledTransition { .. } | Self::Protocol { .. } | Self::Incomplete { .. } => {
                &[]
            }
        }
    }
}

impl fmt::Display for FixedSyncProgramError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DisabledTransition { transition } => {
                write!(
                    formatter,
                    "fixed synchronization transition {transition:?} is disabled"
                )
            }
            Self::Protocol {
                kind,
                operation,
                details,
            }
            | Self::CausalProtocol {
                kind,
                operation,
                details,
                ..
            } => write!(
                formatter,
                "fixed {kind:?} protocol rejected {operation}: {details}"
            ),
            Self::Incomplete {
                kind,
                operation,
                details,
            } => {
                write!(formatter, "fixed {kind:?} verification is incomplete")?;
                if let Some(operation) = operation {
                    write!(formatter, " at {operation}")?;
                }
                write!(formatter, ": {details}")
            }
        }
    }
}

impl Error for FixedSyncProgramError {}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FixedSyncDeadlock {
    protocol_domain: Box<str>,
    protocol_state: Box<str>,
    unfinished_warps: Box<[usize]>,
    blocked_warps: Box<[(usize, FixedSyncCommandId)]>,
    pending_setmaxnreg: Box<[SetmaxnregResource]>,
    unready_heads: Box<[Box<str>]>,
}

impl FixedSyncDeadlock {
    pub fn protocol_domain(&self) -> &str {
        &self.protocol_domain
    }

    pub fn protocol_state(&self) -> &str {
        &self.protocol_state
    }

    pub fn unfinished_warps(&self) -> &[usize] {
        &self.unfinished_warps
    }

    pub fn blocked_warps(&self) -> &[(usize, FixedSyncCommandId)] {
        &self.blocked_warps
    }

    pub fn pending_setmaxnreg(&self) -> &[SetmaxnregResource] {
        &self.pending_setmaxnreg
    }

    pub fn unready_heads(&self) -> &[Box<str>] {
        &self.unready_heads
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FixedSyncProgramBuildError {
    UnsupportedOperation {
        operation: DynamicOpId,
        details: Box<str>,
    },
    AmbiguousEvidence {
        operation: DynamicOpId,
        details: Box<str>,
    },
    InvalidCommand {
        operation: DynamicOpId,
        details: Box<str>,
    },
    InvalidSetmaxRequest {
        request: SetmaxnregResource,
        details: Box<str>,
    },
    InvalidTcgenCollective {
        operation: DynamicOpId,
        details: Box<str>,
    },
    InvalidInitialState {
        details: Box<str>,
    },
}

impl fmt::Display for FixedSyncProgramBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedOperation { operation, details }
            | Self::AmbiguousEvidence { operation, details }
            | Self::InvalidCommand { operation, details }
            | Self::InvalidTcgenCollective { operation, details } => {
                write!(
                    formatter,
                    "cannot build fixed synchronization command for {operation}: {details}"
                )
            }
            Self::InvalidSetmaxRequest { request, details } => {
                write!(
                    formatter,
                    "cannot build fixed setmax request {request:?}: {details}"
                )
            }
            Self::InvalidInitialState { details } => {
                write!(
                    formatter,
                    "invalid fixed synchronization initial state: {details}"
                )
            }
        }
    }
}

impl Error for FixedSyncProgramBuildError {}

pub struct FixedSyncProgram {
    commands: Box<[FixedSyncCommand]>,
    warp_ids: Box<[usize]>,
    warp_indices: BTreeMap<usize, usize>,
    warp_programs: Box<[Box<[FixedSyncCommandId]>]>,
    setmax_commands: BTreeMap<SetmaxnregResource, FixedSyncCommandId>,
    projection_key: Option<FixedSyncProjectionKey>,
    causal_predecessors: Box<[Box<[FixedSyncCommandId]>]>,
    warp_command_positions: HashMap<(usize, FixedSyncCommandId), usize>,
    initial_setmax_pools: BTreeMap<(usize, usize), SetmaxnregVerifierCore>,
    initial_tcgen_ctas: BTreeMap<(usize, usize), FixedSyncTcgenCtaState>,
    conditional_mbarrier_completions:
        BTreeMap<PhysicalBarrierId, BTreeMap<DynamicOpId, BTreeSet<u64>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum FixedSyncProjectionKey {
    Mbarrier(PhysicalBarrierId),
    NamedBarrier(NamedBarrierId),
    ClusterBarrier(ClusterBarrierId),
    SetmaxnregPool {
        kernel_index: usize,
        global_cta_id: usize,
    },
    TcgenCtaComponent {
        kernel_index: usize,
        anchor_global_cta_id: usize,
    },
}

struct NamedCausalGeneration {
    expected_arrivals: u64,
    arrival_count: u64,
    participant_masks: BTreeMap<(usize, NamedCausalContributionKind), u32>,
    release: crate::SyncClockPayload,
    first_operation: DynamicOpId,
    contributions: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum NamedCausalContributionKind {
    Arrive,
    Sync,
}

struct ClusterCausalGeneration {
    participant_warps: Box<[usize]>,
    arrivals: BTreeMap<usize, usize>,
    waits: BTreeMap<usize, usize>,
}

#[derive(Default)]
struct MbarrierCausalGeneration {
    arrival_count: u64,
    /// Extra arrivals this generation must receive on top of the barrier's
    /// `mbarrier.init` expectation.
    pending_arrival_increments: u64,
    expected_transactions: u64,
    completed_transactions: u64,
    mutations: Vec<usize>,
    waits: Vec<usize>,
}

fn mbarrier_incomplete(
    operation: Option<&DynamicOpId>,
    details: impl Into<Box<str>>,
) -> FixedSyncProgramError {
    FixedSyncProgramError::Incomplete {
        kind: FixedSyncProtocolKind::Mbarrier,
        operation: operation.cloned(),
        details: details.into(),
    }
}

fn mbarrier_protocol(
    operation: &DynamicOpId,
    details: impl Into<Box<str>>,
) -> FixedSyncProgramError {
    FixedSyncProgramError::Protocol {
        kind: FixedSyncProtocolKind::Mbarrier,
        operation: operation.clone(),
        details: details.into(),
    }
}

fn checked_mbarrier_count(
    current: u64,
    contribution: u64,
    operation: &DynamicOpId,
    barrier: PhysicalBarrierId,
    generation: u64,
    field: &str,
) -> Result<u64, FixedSyncProgramError> {
    current.checked_add(contribution).ok_or_else(|| {
        mbarrier_protocol(
            operation,
            format!("async mbarrier {barrier:?} generation {generation} {field} count overflows"),
        )
    })
}

fn mbarrier_mutation_requires_consumption(command: &FixedSyncCommand) -> bool {
    match &command.kind {
        FixedSyncCommandKind::MbarrierCompletionIssue { completions } => {
            completions.iter().any(|completion| {
                matches!(
                    completion.kind,
                    FixedSyncMbarrierCompletionKind::Arrival { .. }
                )
            })
        }
        _ => true,
    }
}

type FixedSyncOperationSnapshot<'a> = (
    &'a DynamicOpId,
    &'a ResolvedTransitionSummary,
    Option<&'a crate::SyncVectorClock>,
    Option<&'a crate::SyncVectorClock>,
);

impl FixedSyncProgram {
    pub(crate) fn protocol_projections_from_transition_log(
        transitions: &ResolvedTransitionLog,
        max_workers: usize,
    ) -> Result<Vec<Self>, FixedSyncProgramBuildError> {
        transitions.with_fixed_sync_snapshot(|snapshot| {
            Self::protocol_projections_from_snapshot(snapshot, max_workers)
        })
    }

    fn protocol_projections_from_snapshot(
        snapshot: FixedSyncLogSnapshot<'_>,
        max_workers: usize,
    ) -> Result<Vec<Self>, FixedSyncProgramBuildError> {
        let profile_started = Instant::now();
        let operations = snapshot.operations_with_clocks().collect::<Vec<_>>();
        let operations_collected = Instant::now();
        let mut collective_commands = Vec::<FixedSyncCommand>::new();
        let mut collective_operation_commands = HashMap::<DynamicOpId, FixedSyncCommandId>::new();
        let initial_setmax_pools = build_setmax_commands(
            &snapshot,
            &operations,
            &mut collective_commands,
            &mut collective_operation_commands,
        )?;
        let setmax_built = Instant::now();
        let initial_tcgen_ctas = build_tcgen_commands(
            &snapshot,
            &operations,
            &mut collective_commands,
            &mut collective_operation_commands,
        )?;
        let tcgen_built = Instant::now();
        let tcgen_component_anchors = tcgen_component_anchors(&collective_commands);
        let mbarrier_component_anchors = mbarrier_component_anchors(&operations);
        let anchors_built = Instant::now();

        let mut collective_participant_order =
            vec![Vec::<(usize, u64)>::new(); collective_commands.len()];
        for &(operation, _, initial_clock, clock) in &operations {
            let Some(command_id) = collective_operation_commands.get(operation).copied() else {
                continue;
            };
            let command = &mut collective_commands[command_id.get()];
            if let Some(initial_clock) = initial_clock {
                merge_command_clock(&mut command.initial_causal_clock, initial_clock);
            }
            if let Some(clock) = clock {
                merge_command_clock(&mut command.causal_clock, clock);
            }
            collective_participant_order[command_id.get()]
                .push((operation.global_warp_id(), operation.per_warp_sequence()));
        }

        let mut projected = StagedFixedSyncProjections::new();
        for (command, participant_order) in collective_commands
            .into_iter()
            .zip(collective_participant_order)
        {
            debug_assert_eq!(participant_order.len(), command.participants.len());
            stage_fixed_sync_command(
                command,
                participant_order.into_boxed_slice().into(),
                &mbarrier_component_anchors,
                &tcgen_component_anchors,
                &mut projected,
            );
        }
        let collectives_staged = Instant::now();

        let worker_count = max_workers
            .max(1)
            .min(MAX_FIXED_SYNC_BUILD_WORKERS)
            .min(operations.len());
        if worker_count <= 1 {
            merge_staged_fixed_sync_projections(
                &mut projected,
                stage_noncollective_operations(
                    &operations,
                    &collective_operation_commands,
                    &mbarrier_component_anchors,
                    &tcgen_component_anchors,
                    &snapshot.completion_mbarrier_generations,
                )?,
            );
        } else {
            let chunk_len = operations.len().div_ceil(worker_count);
            let chunks = std::thread::scope(|scope| {
                let handles = operations
                    .chunks(chunk_len)
                    .map(|operations| {
                        let collective_operation_commands = &collective_operation_commands;
                        let mbarrier_component_anchors = &mbarrier_component_anchors;
                        let tcgen_component_anchors = &tcgen_component_anchors;
                        let completion_mbarrier_generations =
                            &snapshot.completion_mbarrier_generations;
                        scope.spawn(move || {
                            stage_noncollective_operations(
                                operations,
                                collective_operation_commands,
                                mbarrier_component_anchors,
                                tcgen_component_anchors,
                                completion_mbarrier_generations,
                            )
                        })
                    })
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|handle| {
                        handle
                            .join()
                            .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
                    })
                    .collect::<Vec<_>>()
            });
            for chunk in chunks {
                merge_staged_fixed_sync_projections(&mut projected, chunk?);
            }
        }
        let commands_staged = Instant::now();
        let projection_count = projected.len();
        let mut projections = Self::build_staged_projections(
            projected,
            max_workers,
            &initial_setmax_pools,
            &initial_tcgen_ctas,
        );
        for projection in &mut projections {
            let mut barriers = BTreeSet::new();
            for command in &projection.commands {
                match &command.kind {
                    FixedSyncCommandKind::MbarrierWait {
                        barrier_id,
                        conditional: true,
                        ..
                    } => {
                        barriers.insert(*barrier_id);
                    }
                    FixedSyncCommandKind::MbarrierWaitBatch {
                        waits,
                        conditional: true,
                    } => {
                        barriers.extend(waits.iter().map(|wait| wait.0));
                    }
                    _ => {}
                }
            }
            if barriers.is_empty() {
                continue;
            }
            let issuers = projection
                .commands
                .iter()
                .map(|command| (&command.witness, command))
                .collect::<HashMap<_, _>>();
            for barrier in barriers {
                let inits = projection.commands.iter().filter(|command| {
                    matches!(&command.kind, FixedSyncCommandKind::MbarrierInit { barrier_ids, .. }
                        if barrier_ids.contains(&barrier))
                }).collect::<Vec<_>>();
                let mut lifetimes = inits
                    .iter()
                    .map(|init| (init.witness.clone(), BTreeSet::new()))
                    .collect::<BTreeMap<_, _>>();
                for (issuer, generations) in snapshot
                    .conditional_mbarrier_completions
                    .get(&barrier)
                    .into_iter()
                    .flatten()
                {
                    let command = issuers.get(issuer).ok_or_else(|| FixedSyncProgramBuildError::InvalidInitialState {
                        details: format!("conditional completion issuer {issuer} is absent from its barrier projection").into(),
                    })?;
                    let latest = latest_causal_mbarrier_init(command.initial_causal_clock.as_ref(), &inits)
                    .ok_or_else(|| FixedSyncProgramBuildError::UnsupportedOperation {
                        operation: issuer.clone(),
                        details: "conditional completion has no unique causally preceding mbarrier initialization".into(),
                    })?;
                    lifetimes
                        .get_mut(&latest.witness)
                        .expect("indexed init")
                        .extend(generations);
                }
                projection
                    .conditional_mbarrier_completions
                    .insert(barrier, lifetimes);
            }
        }
        let projections_built = Instant::now();
        if std::env::var_os("NUMSIM_FIXED_SYNC_PROFILE").is_some() {
            eprintln!(
                "fixed-sync-direct-projections-profile: collect={:.6}s setmax={:.6}s tcgen={:.6}s anchors={:.6}s collectives={:.6}s commands={:.6}s building={:.6}s total={:.6}s operations={} projections={projection_count}",
                operations_collected
                    .duration_since(profile_started)
                    .as_secs_f64(),
                setmax_built
                    .duration_since(operations_collected)
                    .as_secs_f64(),
                tcgen_built.duration_since(setmax_built).as_secs_f64(),
                anchors_built.duration_since(tcgen_built).as_secs_f64(),
                collectives_staged
                    .duration_since(anchors_built)
                    .as_secs_f64(),
                commands_staged
                    .duration_since(collectives_staged)
                    .as_secs_f64(),
                projections_built
                    .duration_since(commands_staged)
                    .as_secs_f64(),
                projections_built
                    .duration_since(profile_started)
                    .as_secs_f64(),
                operations.len(),
            );
        }
        Ok(projections)
    }

    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    pub(crate) fn mbarrier_state_search_fingerprint(&self) -> Option<u64> {
        let FixedSyncProjectionKey::Mbarrier(projected_barrier) = self.projection_key? else {
            return None;
        };
        if !self.setmax_commands.is_empty()
            || !self.initial_setmax_pools.is_empty()
            || !self.initial_tcgen_ctas.is_empty()
        {
            return None;
        }

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.commands.len().hash(&mut hasher);
        for command in self.commands.iter() {
            self.warp_indices
                .get(&command.witness.global_warp_id())?
                .hash(&mut hasher);
            command.participants.len().hash(&mut hasher);
            for warp_id in command.participants.iter() {
                self.warp_indices.get(warp_id)?.hash(&mut hasher);
            }
            match &command.kind {
                FixedSyncCommandKind::MbarrierInit {
                    barrier_ids,
                    expected_arrivals,
                } if barrier_ids.as_ref() == [projected_barrier] => {
                    0_u8.hash(&mut hasher);
                    expected_arrivals.hash(&mut hasher);
                    self.conditional_mbarrier_completions
                        .get(&projected_barrier)
                        .and_then(|lifetimes| lifetimes.get(&command.witness))
                        .hash(&mut hasher);
                }
                FixedSyncCommandKind::MbarrierInitFence { barrier_ids }
                    if barrier_ids.as_ref() == [projected_barrier] =>
                {
                    4_u8.hash(&mut hasher);
                }
                FixedSyncCommandKind::MbarrierArrive { arrivals }
                    if arrivals
                        .iter()
                        .all(|(barrier_id, _, _, _)| *barrier_id == projected_barrier) =>
                {
                    1_u8.hash(&mut hasher);
                    arrivals.len().hash(&mut hasher);
                    for (_, count, transactions, drop) in arrivals.iter() {
                        count.hash(&mut hasher);
                        transactions.hash(&mut hasher);
                        drop.hash(&mut hasher);
                    }
                }
                FixedSyncCommandKind::MbarrierExpectTx { expectations }
                    if expectations
                        .iter()
                        .all(|(barrier_id, _)| *barrier_id == projected_barrier) =>
                {
                    5_u8.hash(&mut hasher);
                    expectations.len().hash(&mut hasher);
                    for (_, transactions) in expectations.iter() {
                        transactions.hash(&mut hasher);
                    }
                }
                FixedSyncCommandKind::MbarrierWait {
                    barrier_id,
                    requested_phase,
                    conditional,
                } if *barrier_id == projected_barrier => {
                    2_u8.hash(&mut hasher);
                    requested_phase.hash(&mut hasher);
                    conditional.hash(&mut hasher);
                }
                FixedSyncCommandKind::MbarrierWaitBatch { waits, conditional } => {
                    6_u8.hash(&mut hasher);
                    waits.hash(&mut hasher);
                    conditional.hash(&mut hasher);
                }
                FixedSyncCommandKind::MbarrierCompletionIssue { completions }
                    if completions
                        .iter()
                        .all(|completion| completion.barrier_id == projected_barrier) =>
                {
                    3_u8.hash(&mut hasher);
                    completions.len().hash(&mut hasher);
                    for completion in completions.iter() {
                        match completion.kind {
                            FixedSyncMbarrierCompletionKind::Transaction { transactions } => {
                                0_u8.hash(&mut hasher);
                                transactions.hash(&mut hasher);
                            }
                            FixedSyncMbarrierCompletionKind::Arrival {
                                warp_id,
                                arrival_count,
                                pending_increase,
                            } => {
                                1_u8.hash(&mut hasher);
                                self.warp_indices.get(&warp_id)?.hash(&mut hasher);
                                arrival_count.hash(&mut hasher);
                                pending_increase.hash(&mut hasher);
                            }
                        }
                    }
                }
                _ => return None,
            }
        }
        self.warp_programs.hash(&mut hasher);
        self.causal_predecessors.hash(&mut hasher);
        Some(hasher.finish())
    }

    pub(crate) fn has_equivalent_mbarrier_state_search(&self, other: &Self) -> bool {
        let (
            Some(FixedSyncProjectionKey::Mbarrier(left_barrier)),
            Some(FixedSyncProjectionKey::Mbarrier(right_barrier)),
        ) = (self.projection_key, other.projection_key)
        else {
            return false;
        };
        if !self.setmax_commands.is_empty()
            || !self.initial_setmax_pools.is_empty()
            || !self.initial_tcgen_ctas.is_empty()
            || !other.setmax_commands.is_empty()
            || !other.initial_setmax_pools.is_empty()
            || !other.initial_tcgen_ctas.is_empty()
            || self.warp_programs != other.warp_programs
            || self.causal_predecessors != other.causal_predecessors
            || self.commands.len() != other.commands.len()
        {
            return false;
        }
        self.commands
            .iter()
            .zip(other.commands.iter())
            .all(|(left, right)| {
                equivalent_mbarrier_state_search_command(
                    self,
                    left_barrier,
                    left,
                    other,
                    right_barrier,
                    right,
                )
            })
    }

    /// Verify one named-barrier projection directly from its generation and
    /// vector-clock certificate.
    ///
    /// Contributions within a generation commute once masks are disjoint and
    /// their total is exact.  Requiring the joined release clock of generation
    /// `g` to happen-before every contribution to `g + 1` prevents a legal
    /// schedule from assigning that contribution to the prior generation.
    ///
    /// If a generation contains an aligned blocking sync, every blocking sync
    /// contribution must be aligned. Contributions from one warp must share a
    /// static TIR origin; different warps may reach equivalent aligned
    /// barriers through distinct inlined call sites. Arrive contributions are
    /// independent, and generations whose blocking syncs are all unaligned do
    /// not require one static origin. Runtime loop iterations at one TIR call
    /// site share one origin.
    pub(crate) fn verify_named_barrier_causally(
        &self,
    ) -> Option<Result<(), FixedSyncProgramError>> {
        let FixedSyncProjectionKey::NamedBarrier(projected_barrier) = self.projection_key? else {
            return None;
        };
        let mut generations = BTreeMap::<u64, NamedCausalGeneration>::new();
        for (command_index, command) in self.commands.iter().enumerate() {
            let (barrier_id, expected_arrivals, warp_id, arrival_mask, contribution_kind) =
                match &command.kind {
                    FixedSyncCommandKind::NamedBarrierArrive {
                        barrier_id,
                        expected_arrivals,
                        warp_id,
                        arrival_mask,
                    } => (
                        *barrier_id,
                        *expected_arrivals,
                        *warp_id,
                        *arrival_mask,
                        NamedCausalContributionKind::Arrive,
                    ),
                    FixedSyncCommandKind::NamedBarrierSync {
                        barrier_id,
                        expected_arrivals,
                        warp_id,
                        arrival_mask,
                        ..
                    } => (
                        *barrier_id,
                        *expected_arrivals,
                        *warp_id,
                        *arrival_mask,
                        NamedCausalContributionKind::Sync,
                    ),
                    _ => {
                        return Some(Err(FixedSyncProgramError::Incomplete {
                            kind: FixedSyncProtocolKind::NamedBarrier,
                            operation: Some(command.witness.clone()),
                            details: "named-barrier projection contains another protocol".into(),
                        }));
                    }
                };
            if barrier_id != projected_barrier {
                return Some(Err(FixedSyncProgramError::Incomplete {
                    kind: FixedSyncProtocolKind::NamedBarrier,
                    operation: Some(command.witness.clone()),
                    details: format!(
                        "named-barrier projection for {projected_barrier:?} contains {barrier_id:?}"
                    )
                    .into_boxed_str(),
                }));
            }
            let Some(generation) = command.canonical_generation else {
                return Some(Err(FixedSyncProgramError::Incomplete {
                    kind: FixedSyncProtocolKind::NamedBarrier,
                    operation: Some(command.witness.clone()),
                    details: "named-barrier contribution has no committed generation".into(),
                }));
            };
            let Some(release_clock) = command.initial_causal_clock.clone() else {
                return Some(Err(FixedSyncProgramError::Incomplete {
                    kind: FixedSyncProtocolKind::NamedBarrier,
                    operation: Some(command.witness.clone()),
                    details: "named-barrier contribution has no release clock".into(),
                }));
            };
            let generation_state =
                generations
                    .entry(generation)
                    .or_insert_with(|| NamedCausalGeneration {
                        expected_arrivals,
                        arrival_count: 0,
                        participant_masks: BTreeMap::new(),
                        release: crate::SyncClockPayload::from_clock(release_clock.clone()),
                        first_operation: command.witness.clone(),
                        contributions: Vec::new(),
                    });
            if generation_state.expected_arrivals != expected_arrivals {
                return Some(Err(FixedSyncProgramError::Protocol {
                    kind: FixedSyncProtocolKind::NamedBarrier,
                    operation: command.witness.clone(),
                    details: format!(
                        "named barrier {barrier_id:?} generation {generation} changes expected arrivals from {} to {expected_arrivals}; first contribution {}",
                        generation_state.expected_arrivals,
                        generation_state.first_operation,
                    )
                    .into_boxed_str(),
                }));
            }
            let prior_mask = generation_state
                .participant_masks
                .entry((warp_id, contribution_kind))
                .or_default();
            let overlap = *prior_mask & arrival_mask.bits();
            if overlap != 0 {
                return Some(Err(FixedSyncProgramError::Protocol {
                    kind: FixedSyncProtocolKind::NamedBarrier,
                    operation: command.witness.clone(),
                    details: format!(
                        "named barrier {barrier_id:?} generation {generation} has duplicate warp {warp_id} lanes {overlap:#010x}; first contribution {}",
                        generation_state.first_operation,
                    )
                    .into_boxed_str(),
                }));
            }
            *prior_mask |= arrival_mask.bits();
            generation_state.arrival_count = match generation_state
                .arrival_count
                .checked_add(arrival_mask.len() as u64)
            {
                Some(count) if count <= expected_arrivals => count,
                Some(count) => {
                    return Some(Err(FixedSyncProgramError::Protocol {
                        kind: FixedSyncProtocolKind::NamedBarrier,
                        operation: command.witness.clone(),
                        details: format!(
                            "named barrier {barrier_id:?} generation {generation} over-arrives: {count} > {expected_arrivals}"
                        )
                        .into_boxed_str(),
                    }));
                }
                None => {
                    return Some(Err(FixedSyncProgramError::Protocol {
                        kind: FixedSyncProtocolKind::NamedBarrier,
                        operation: command.witness.clone(),
                        details: format!(
                            "named barrier {barrier_id:?} generation {generation} arrival count overflows"
                        )
                        .into_boxed_str(),
                    }));
                }
            };
            if command.witness != generation_state.first_operation {
                generation_state
                    .release
                    .merge(&crate::SyncClockPayload::from_clock(release_clock))
                    .expect("named-barrier release clocks share one launch domain");
            }
            generation_state.contributions.push(command_index);
        }

        let mut prior: Option<(u64, crate::SyncClockPayload, DynamicOpId)> = None;
        for (&generation, state) in &generations {
            let expected_generation = prior
                .as_ref()
                .map_or(0, |(prior_generation, _, _)| prior_generation + 1);
            if generation != expected_generation {
                return Some(Err(FixedSyncProgramError::Incomplete {
                    kind: FixedSyncProtocolKind::NamedBarrier,
                    operation: Some(state.first_operation.clone()),
                    details: format!(
                        "named barrier {projected_barrier:?} skips generation {expected_generation} before {generation}"
                    )
                    .into_boxed_str(),
                }));
            }
            if state.arrival_count != state.expected_arrivals {
                return Some(Err(FixedSyncProgramError::Protocol {
                    kind: FixedSyncProtocolKind::NamedBarrier,
                    operation: state.first_operation.clone(),
                    details: format!(
                        "named barrier {projected_barrier:?} generation {generation} has {} of {} required arrivals",
                        state.arrival_count, state.expected_arrivals,
                    )
                    .into_boxed_str(),
                }));
            }
            let aligned_anchor = state.contributions.iter().find_map(|&command_index| {
                let command = &self.commands[command_index];
                matches!(
                    &command.kind,
                    FixedSyncCommandKind::NamedBarrierSync { aligned: true, .. }
                )
                .then_some(command)
            });
            if let Some(anchor) = aligned_anchor {
                let mut aligned_origins_by_warp = BTreeMap::<usize, &DynamicOpId>::new();
                for &command_index in &state.contributions {
                    let command = &self.commands[command_index];
                    let FixedSyncCommandKind::NamedBarrierSync {
                        warp_id, aligned, ..
                    } = &command.kind
                    else {
                        continue;
                    };
                    if !*aligned {
                        return Some(Err(FixedSyncProgramError::Protocol {
                            kind: FixedSyncProtocolKind::NamedBarrier,
                            operation: command.witness.clone(),
                            details: format!(
                                "named-barrier generation {generation} on {projected_barrier:?} mixes aligned and unaligned blocking syncs: aligned anchor {}, conflicting contribution {}",
                                anchor.witness,
                                command.witness,
                            )
                            .into_boxed_str(),
                        }));
                    }
                    if let Some(previous) =
                        aligned_origins_by_warp.insert(*warp_id, &command.witness)
                    {
                        if !previous.same_static_instruction(&command.witness) {
                            return Some(Err(FixedSyncProgramError::Protocol {
                                kind: FixedSyncProtocolKind::NamedBarrier,
                                operation: command.witness.clone(),
                                details: format!(
                                    "named-barrier generation {generation} on {projected_barrier:?} has divergent aligned blocking sync sites within warp {warp_id}: first {}, conflicting contribution {}",
                                    previous,
                                    command.witness,
                                )
                                .into_boxed_str(),
                            }));
                        }
                    }
                }
            }
            if let Some((prior_generation, prior_release, prior_operation)) = &prior {
                for &command_index in &state.contributions {
                    let command = &self.commands[command_index];
                    let Some(clock) = &command.initial_causal_clock else {
                        unreachable!("all named-barrier commands were checked for clocks")
                    };
                    if !prior_release.clock().happens_before(clock) {
                        return Some(Err(FixedSyncProgramError::Protocol {
                            kind: FixedSyncProtocolKind::NamedBarrier,
                            operation: command.witness.clone(),
                            details: format!(
                                "named barrier {projected_barrier:?} generation {generation} contribution {} is not happens-before ordered after generation {prior_generation} release witnessed by {prior_operation}; prior release {:?}, contribution {:?}",
                                command.witness,
                                prior_release.clock(),
                                clock,
                            )
                            .into_boxed_str(),
                        }));
                    }
                }
            }
            prior = Some((
                generation,
                state.release.clone(),
                state.first_operation.clone(),
            ));
        }
        Some(Ok(()))
    }

    /// Verify one cluster-barrier projection from the generations and per-warp
    /// programs committed by the concrete launch.
    ///
    /// Every generation must contain exactly one full-warp arrival from every
    /// declared participant.  Reuse is safe only when each participant's next
    /// arrival is program-ordered after that participant consumed the prior
    /// generation.  Since a wait blocks the issuing warp until all prior
    /// arrivals exist, the certificate rules out both early arrival and
    /// generation reassignment without enumerating arrival permutations.
    pub(crate) fn verify_cluster_barrier_causally(
        &self,
    ) -> Option<Result<(), FixedSyncProgramError>> {
        let FixedSyncProjectionKey::ClusterBarrier(projected_barrier) = self.projection_key? else {
            return None;
        };

        let mut contract: Option<Box<[usize]>> = None;
        let mut generations = BTreeMap::<u64, ClusterCausalGeneration>::new();
        for (index, command) in self.commands.iter().enumerate() {
            let (barrier_id, warp_id, arrival_mask, participant_warps, is_arrival, is_wait) =
                match &command.kind {
                    FixedSyncCommandKind::ClusterBarrierArrive {
                        barrier_id,
                        warp_id,
                        arrival_mask,
                        participant_warps,
                    } => (
                        *barrier_id,
                        *warp_id,
                        *arrival_mask,
                        participant_warps,
                        true,
                        false,
                    ),
                    FixedSyncCommandKind::ClusterBarrierWait {
                        barrier_id,
                        warp_id,
                        arrival_mask,
                        participant_warps,
                    } => (
                        *barrier_id,
                        *warp_id,
                        *arrival_mask,
                        participant_warps,
                        false,
                        true,
                    ),
                    _ => {
                        return Some(Err(FixedSyncProgramError::Incomplete {
                            kind: FixedSyncProtocolKind::ClusterBarrier,
                            operation: Some(command.witness.clone()),
                            details: "cluster-barrier projection contains another protocol".into(),
                        }));
                    }
                };
            if barrier_id != projected_barrier {
                return Some(Err(FixedSyncProgramError::Incomplete {
                    kind: FixedSyncProtocolKind::ClusterBarrier,
                    operation: Some(command.witness.clone()),
                    details: format!(
                        "cluster-barrier projection for {projected_barrier:?} contains {barrier_id:?}"
                    )
                    .into_boxed_str(),
                }));
            }
            if !arrival_mask.is_full() {
                return Some(Err(FixedSyncProgramError::Incomplete {
                    kind: FixedSyncProtocolKind::ClusterBarrier,
                    operation: Some(command.witness.clone()),
                    details: format!(
                        "cluster barrier {projected_barrier:?} contains partial-warp participation {arrival_mask:?}"
                    )
                    .into_boxed_str(),
                }));
            }
            if !participant_warps.contains(&warp_id) {
                return Some(Err(FixedSyncProgramError::Protocol {
                    kind: FixedSyncProtocolKind::ClusterBarrier,
                    operation: command.witness.clone(),
                    details: format!(
                        "cluster barrier {projected_barrier:?} warp {warp_id} is absent from participant contract {participant_warps:?}"
                    )
                    .into_boxed_str(),
                }));
            }
            if let Some(expected) = &contract {
                if expected.as_ref() != participant_warps.as_ref() {
                    return Some(Err(FixedSyncProgramError::Protocol {
                        kind: FixedSyncProtocolKind::ClusterBarrier,
                        operation: command.witness.clone(),
                        details: format!(
                            "cluster barrier {projected_barrier:?} participant contract changes from {expected:?} to {participant_warps:?}"
                        )
                        .into_boxed_str(),
                    }));
                }
            } else {
                let unique = participant_warps.iter().copied().collect::<BTreeSet<_>>();
                if unique.len() != participant_warps.len() {
                    return Some(Err(FixedSyncProgramError::Protocol {
                        kind: FixedSyncProtocolKind::ClusterBarrier,
                        operation: command.witness.clone(),
                        details: format!(
                            "cluster barrier {projected_barrier:?} participant contract contains duplicates: {participant_warps:?}"
                        )
                        .into_boxed_str(),
                    }));
                }
                contract = Some(participant_warps.clone());
            }

            let Some(generation) = command.canonical_generation else {
                return Some(Err(FixedSyncProgramError::Incomplete {
                    kind: FixedSyncProtocolKind::ClusterBarrier,
                    operation: Some(command.witness.clone()),
                    details: "cluster-barrier command has no committed generation".into(),
                }));
            };
            let generation_state =
                generations
                    .entry(generation)
                    .or_insert_with(|| ClusterCausalGeneration {
                        participant_warps: participant_warps.clone(),
                        arrivals: BTreeMap::new(),
                        waits: BTreeMap::new(),
                    });
            if generation_state.participant_warps.as_ref() != participant_warps.as_ref() {
                unreachable!("the cluster participant contract was checked above");
            }
            if is_arrival && generation_state.arrivals.insert(warp_id, index).is_some() {
                return Some(Err(FixedSyncProgramError::Protocol {
                    kind: FixedSyncProtocolKind::ClusterBarrier,
                    operation: command.witness.clone(),
                    details: format!(
                        "cluster barrier {projected_barrier:?} generation {generation} has duplicate arrival from warp {warp_id}"
                    )
                    .into_boxed_str(),
                }));
            }
            if is_wait && generation_state.waits.insert(warp_id, index).is_some() {
                return Some(Err(FixedSyncProgramError::Protocol {
                    kind: FixedSyncProtocolKind::ClusterBarrier,
                    operation: command.witness.clone(),
                    details: format!(
                        "cluster barrier {projected_barrier:?} generation {generation} has duplicate wait from warp {warp_id}"
                    )
                    .into_boxed_str(),
                }));
            }
        }

        let Some(contract) = contract else {
            return Some(Ok(()));
        };
        let mut expected_generation = 0_u64;
        for (&generation, state) in &generations {
            if generation != expected_generation {
                return Some(Err(FixedSyncProgramError::Incomplete {
                    kind: FixedSyncProtocolKind::ClusterBarrier,
                    operation: state
                        .arrivals
                        .values()
                        .next()
                        .map(|index| self.commands[*index].witness.clone()),
                    details: format!(
                        "cluster barrier {projected_barrier:?} skips generation {expected_generation} before {generation}"
                    )
                    .into_boxed_str(),
                }));
            }
            let missing = contract
                .iter()
                .copied()
                .filter(|warp_id| !state.arrivals.contains_key(warp_id))
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Some(Err(FixedSyncProgramError::Incomplete {
                    kind: FixedSyncProtocolKind::ClusterBarrier,
                    operation: state
                        .arrivals
                        .values()
                        .next()
                        .map(|index| self.commands[*index].witness.clone()),
                    details: format!(
                        "cluster barrier {projected_barrier:?} generation {generation} is missing participant warps {missing:?}; exit-aware membership is not modeled"
                    )
                    .into_boxed_str(),
                }));
            }

            for (&warp_id, &wait_index) in &state.waits {
                let Some(&arrival_index) = state.arrivals.get(&warp_id) else {
                    return Some(Err(FixedSyncProgramError::Protocol {
                        kind: FixedSyncProtocolKind::ClusterBarrier,
                        operation: self.commands[wait_index].witness.clone(),
                        details: format!(
                            "cluster barrier {projected_barrier:?} generation {generation} warp {warp_id} waits without arriving"
                        )
                        .into_boxed_str(),
                    }));
                };
                if arrival_index != wait_index {
                    let arrival = &self.commands[arrival_index];
                    let wait = &self.commands[wait_index];
                    let arrival_position =
                        self.warp_command_positions[&(warp_id, FixedSyncCommandId(arrival_index))];
                    let wait_position =
                        self.warp_command_positions[&(warp_id, FixedSyncCommandId(wait_index))];
                    if arrival_position >= wait_position {
                        return Some(Err(FixedSyncProgramError::CausalProtocol {
                            kind: FixedSyncProtocolKind::ClusterBarrier,
                            operation: wait.witness.clone(),
                            related_operations: Box::new([arrival.witness.clone()]),
                            details: format!(
                                "cluster barrier {projected_barrier:?} generation {generation} wait {} is not ordered after warp {warp_id} arrival {}",
                                wait.witness, arrival.witness,
                            )
                            .into_boxed_str(),
                        }));
                    }
                }
            }

            if generation > 0 {
                let prior = &generations[&(generation - 1)];
                for (&warp_id, &arrival_index) in &state.arrivals {
                    let Some(&prior_wait_index) = prior.waits.get(&warp_id) else {
                        return Some(Err(FixedSyncProgramError::Incomplete {
                            kind: FixedSyncProtocolKind::ClusterBarrier,
                            operation: Some(self.commands[arrival_index].witness.clone()),
                            details: format!(
                                "cluster barrier {projected_barrier:?} warp {warp_id} re-arrives for generation {generation} without consuming generation {}",
                                generation - 1,
                            )
                            .into_boxed_str(),
                        }));
                    };
                    let prior_wait = &self.commands[prior_wait_index];
                    let arrival = &self.commands[arrival_index];
                    let prior_wait_position = self.warp_command_positions
                        [&(warp_id, FixedSyncCommandId(prior_wait_index))];
                    let arrival_position =
                        self.warp_command_positions[&(warp_id, FixedSyncCommandId(arrival_index))];
                    if prior_wait_position >= arrival_position {
                        return Some(Err(FixedSyncProgramError::CausalProtocol {
                            kind: FixedSyncProtocolKind::ClusterBarrier,
                            operation: arrival.witness.clone(),
                            related_operations: Box::new([prior_wait.witness.clone()]),
                            details: format!(
                                "cluster barrier {projected_barrier:?} generation {generation} arrival {} can run before warp {warp_id} consumes generation {}; prior wait {}",
                                arrival.witness,
                                generation - 1,
                                prior_wait.witness,
                            )
                            .into_boxed_str(),
                        }));
                    }
                }
            }
            expected_generation = expected_generation.saturating_add(1);
        }
        Some(Ok(()))
    }

    /// Verify one mbarrier projection from its committed generations and
    /// vector-clock certificate, including deferred transaction and arrival
    /// completions.
    pub(crate) fn verify_mbarrier_causally(&self) -> Option<Result<(), FixedSyncProgramError>> {
        let FixedSyncProjectionKey::Mbarrier(projected_barrier) = self.projection_key? else {
            return None;
        };
        if self.commands.iter().any(|command| {
            matches!(
                command.kind,
                FixedSyncCommandKind::MbarrierWaitBatch { .. }
                    | FixedSyncCommandKind::MbarrierInvalidate { .. }
            )
        }) {
            return None;
        }

        let init_commands = self
            .commands
            .iter()
            .enumerate()
            .filter_map(|(index, command)| {
                matches!(command.kind, FixedSyncCommandKind::MbarrierInit { .. }).then_some(index)
            })
            .collect::<Vec<_>>();
        let [init_index] = init_commands.as_slice() else {
            let operation = init_commands
                .get(1)
                .copied()
                .or_else(|| (!self.commands.is_empty()).then_some(0))
                .and_then(|index| self.commands.get(index))
                .map(|command| command.witness.clone());
            return Some(Err(FixedSyncProgramError::Incomplete {
                kind: FixedSyncProtocolKind::Mbarrier,
                operation,
                details: format!(
                    "mbarrier {projected_barrier:?} projection has {} init commands",
                    init_commands.len(),
                )
                .into_boxed_str(),
            }));
        };
        let init = &self.commands[*init_index];
        let FixedSyncCommandKind::MbarrierInit {
            barrier_ids,
            expected_arrivals,
        } = &init.kind
        else {
            unreachable!("init command was selected by kind")
        };
        if barrier_ids.as_ref() != [projected_barrier] {
            return Some(Err(FixedSyncProgramError::Incomplete {
                kind: FixedSyncProtocolKind::Mbarrier,
                operation: Some(init.witness.clone()),
                details: format!(
                    "mbarrier projection for {projected_barrier:?} has init targets {barrier_ids:?}"
                )
                .into_boxed_str(),
            }));
        }
        let Some(init_clock) = &init.causal_clock else {
            return Some(Err(FixedSyncProgramError::Incomplete {
                kind: FixedSyncProtocolKind::Mbarrier,
                operation: Some(init.witness.clone()),
                details: "mbarrier init has no committed causal clock".into(),
            }));
        };

        for command in self.commands.iter() {
            let FixedSyncCommandKind::MbarrierInitFence { barrier_ids } = &command.kind else {
                continue;
            };
            if barrier_ids.as_ref() != [projected_barrier] {
                return Some(Err(FixedSyncProgramError::Incomplete {
                    kind: FixedSyncProtocolKind::Mbarrier,
                    operation: Some(command.witness.clone()),
                    details: format!(
                        "mbarrier projection for {projected_barrier:?} has init-fence targets {barrier_ids:?}"
                    )
                    .into_boxed_str(),
                }));
            }
            let Some(fence_clock) = &command.causal_clock else {
                return Some(Err(FixedSyncProgramError::Incomplete {
                    kind: FixedSyncProtocolKind::Mbarrier,
                    operation: Some(command.witness.clone()),
                    details: "mbarrier init fence has no committed causal clock".into(),
                }));
            };
            if !strict_happens_before(init_clock, fence_clock) {
                return Some(Err(FixedSyncProgramError::Protocol {
                    kind: FixedSyncProtocolKind::Mbarrier,
                    operation: command.witness.clone(),
                    details: format!(
                        "mbarrier {projected_barrier:?} init fence {} is not happens-before ordered after init {}; init {:?}, fence {:?}",
                        command.witness, init.witness, init_clock, fence_clock,
                    )
                    .into_boxed_str(),
                }));
            }
        }

        let mut generations = BTreeMap::<u64, MbarrierCausalGeneration>::new();
        for (index, command) in self.commands.iter().enumerate() {
            if index == *init_index
                || matches!(command.kind, FixedSyncCommandKind::MbarrierInitFence { .. })
            {
                continue;
            }
            let Some(initial_clock) = &command.initial_causal_clock else {
                return Some(Err(FixedSyncProgramError::Incomplete {
                    kind: FixedSyncProtocolKind::Mbarrier,
                    operation: Some(command.witness.clone()),
                    details: "mbarrier command has no initial causal clock".into(),
                }));
            };
            if !strict_happens_before(init_clock, initial_clock) {
                return Some(Err(FixedSyncProgramError::Protocol {
                    kind: FixedSyncProtocolKind::Mbarrier,
                    operation: command.witness.clone(),
                    details: format!(
                        "mbarrier {projected_barrier:?} use {} is not happens-before ordered after init {}; init {:?}, use {:?}",
                        command.witness, init.witness, init_clock, initial_clock,
                    )
                    .into_boxed_str(),
                }));
            }
            let Some(_use_clock) = &command.causal_clock else {
                return Some(Err(FixedSyncProgramError::Incomplete {
                    kind: FixedSyncProtocolKind::Mbarrier,
                    operation: Some(command.witness.clone()),
                    details: "mbarrier command has no committed causal clock".into(),
                }));
            };
            match &command.kind {
                FixedSyncCommandKind::MbarrierArrive { arrivals } => {
                    // This certificate assumes a constant phase expectation.
                    // Drops use the existing exact protocol-state verifier.
                    if arrivals.iter().any(|arrival| arrival.3) {
                        return None;
                    }
                    let Some(generation) = command.canonical_generation else {
                        return Some(Err(FixedSyncProgramError::Incomplete {
                            kind: FixedSyncProtocolKind::Mbarrier,
                            operation: Some(command.witness.clone()),
                            details: "mbarrier arrival has no committed generation".into(),
                        }));
                    };
                    let generation_state = generations.entry(generation).or_default();
                    for &(barrier_id, arrival_count, transactions, _) in arrivals.iter() {
                        if barrier_id != projected_barrier {
                            return Some(Err(FixedSyncProgramError::Incomplete {
                                kind: FixedSyncProtocolKind::Mbarrier,
                                operation: Some(command.witness.clone()),
                                details: format!(
                                    "mbarrier projection {projected_barrier:?} contains arrival for {barrier_id:?}"
                                )
                                .into_boxed_str(),
                            }));
                        }
                        generation_state.arrival_count = match checked_mbarrier_count(
                            generation_state.arrival_count,
                            arrival_count,
                            &command.witness,
                            projected_barrier,
                            generation,
                            "arrival",
                        ) {
                            Ok(count) => count,
                            Err(error) => return Some(Err(error)),
                        };
                        generation_state.expected_transactions = match checked_mbarrier_count(
                            generation_state.expected_transactions,
                            transactions.unwrap_or(0),
                            &command.witness,
                            projected_barrier,
                            generation,
                            "expected transaction",
                        ) {
                            Ok(count) => count,
                            Err(error) => return Some(Err(error)),
                        };
                    }
                    generation_state.mutations.push(index);
                }
                FixedSyncCommandKind::MbarrierExpectTx { expectations } => {
                    let Some(generation) = command.canonical_generation else {
                        return Some(Err(mbarrier_incomplete(
                            Some(&command.witness),
                            "mbarrier.expect_tx has no committed generation",
                        )));
                    };
                    let generation_state = generations.entry(generation).or_default();
                    for &(barrier_id, transactions) in expectations.iter() {
                        if barrier_id != projected_barrier {
                            return Some(Err(mbarrier_incomplete(
                                Some(&command.witness),
                                format!(
                                    "mbarrier projection {projected_barrier:?} contains expect_tx for {barrier_id:?}"
                                ),
                            )));
                        }
                        generation_state.expected_transactions = match checked_mbarrier_count(
                            generation_state.expected_transactions,
                            transactions,
                            &command.witness,
                            projected_barrier,
                            generation,
                            "expected transaction",
                        ) {
                            Ok(count) => count,
                            Err(error) => return Some(Err(error)),
                        };
                    }
                    generation_state.mutations.push(index);
                }
                FixedSyncCommandKind::MbarrierCompletionIssue { completions } => {
                    let Some(generation) = command.canonical_generation else {
                        return Some(Err(mbarrier_incomplete(
                            Some(&command.witness),
                            "mbarrier completion issue has no committed generation",
                        )));
                    };
                    let generation_state = generations.entry(generation).or_default();
                    for completion in completions.iter() {
                        if completion.barrier_id != projected_barrier {
                            return Some(Err(mbarrier_incomplete(
                                Some(&command.witness),
                                format!(
                                    "mbarrier projection {projected_barrier:?} contains completion for {:?}",
                                    completion.barrier_id,
                                ),
                            )));
                        }
                        if let FixedSyncMbarrierCompletionKind::Arrival {
                            pending_increase, ..
                        } = completion.kind
                        {
                            generation_state.pending_arrival_increments =
                                match checked_mbarrier_count(
                                    generation_state.pending_arrival_increments,
                                    pending_increase,
                                    &command.witness,
                                    projected_barrier,
                                    generation,
                                    "pending arrival",
                                ) {
                                    Ok(count) => count,
                                    Err(error) => return Some(Err(error)),
                                };
                        }
                        let (field, current, contribution) = match completion.kind {
                            FixedSyncMbarrierCompletionKind::Transaction { transactions } => (
                                "completed transaction",
                                &mut generation_state.completed_transactions,
                                transactions,
                            ),
                            FixedSyncMbarrierCompletionKind::Arrival { arrival_count, .. } => (
                                "deferred arrival",
                                &mut generation_state.arrival_count,
                                arrival_count,
                            ),
                        };
                        *current = match checked_mbarrier_count(
                            *current,
                            contribution,
                            &command.witness,
                            projected_barrier,
                            generation,
                            field,
                        ) {
                            Ok(count) => count,
                            Err(error) => return Some(Err(error)),
                        };
                    }
                    generation_state.mutations.push(index);
                }
                FixedSyncCommandKind::MbarrierWait {
                    barrier_id,
                    requested_phase,
                    conditional,
                } => {
                    if *barrier_id != projected_barrier {
                        return Some(Err(FixedSyncProgramError::Incomplete {
                            kind: FixedSyncProtocolKind::Mbarrier,
                            operation: Some(command.witness.clone()),
                            details: format!(
                                "mbarrier projection {projected_barrier:?} contains wait for {barrier_id:?}"
                            )
                            .into_boxed_str(),
                        }));
                    }
                    let Some(generation) = command.canonical_generation else {
                        if *requested_phase == 1 {
                            continue;
                        }
                        return Some(Err(FixedSyncProgramError::Incomplete {
                            kind: FixedSyncProtocolKind::Mbarrier,
                            operation: Some(command.witness.clone()),
                            details: format!(
                                "mbarrier phase {requested_phase} wait has no committed generation"
                            )
                            .into_boxed_str(),
                        }));
                    };
                    // Conditional parity is mapped to this exact primary
                    // completion by the numeric owner, not by primary parity.
                    if !conditional && *requested_phase != (generation & 1) {
                        return Some(Err(FixedSyncProgramError::Protocol {
                            kind: FixedSyncProtocolKind::Mbarrier,
                            operation: command.witness.clone(),
                            details: format!(
                                "mbarrier {projected_barrier:?} wait {} requests phase {requested_phase}, but completed generation {generation} has phase {}",
                                command.witness,
                                generation & 1,
                            )
                            .into_boxed_str(),
                        }));
                    }
                    generations.entry(generation).or_default().waits.push(index);
                }
                FixedSyncCommandKind::MbarrierInit { .. } => {
                    unreachable!("all additional init commands were rejected")
                }
                FixedSyncCommandKind::MbarrierInitFence { .. } => {
                    unreachable!("init fences were handled before use validation")
                }
                _ => {
                    return Some(Err(FixedSyncProgramError::Incomplete {
                        kind: FixedSyncProtocolKind::Mbarrier,
                        operation: Some(command.witness.clone()),
                        details: "mbarrier projection contains another protocol".into(),
                    }));
                }
            }
        }

        let mut expected_generation = 0_u64;
        for (&generation, state) in &generations {
            if generation != expected_generation {
                return Some(Err(FixedSyncProgramError::Incomplete {
                    kind: FixedSyncProtocolKind::Mbarrier,
                    operation: state
                        .mutations
                        .first()
                        .or_else(|| state.waits.first())
                        .map(|index| self.commands[*index].witness.clone()),
                    details: format!(
                        "mbarrier {projected_barrier:?} skips generation {expected_generation} before {generation}"
                    )
                    .into_boxed_str(),
                }));
            }
            let has_next_generation = generations.contains_key(&(generation + 1));
            let requires_completion = has_next_generation || !state.waits.is_empty();
            // A raised pending count belongs to this generation only, so the
            // required arrival total is the init expectation plus whatever this
            // generation raised.
            let required_arrivals =
                expected_arrivals.saturating_add(state.pending_arrival_increments);
            let invalid_counts = state.arrival_count > required_arrivals
                || (state.arrival_count == required_arrivals
                    && state.completed_transactions > state.expected_transactions)
                || (requires_completion
                    && (state.arrival_count != required_arrivals
                        || state.completed_transactions != state.expected_transactions));
            if invalid_counts {
                let operation = state.mutations.first().map_or_else(
                    || init.witness.clone(),
                    |index| self.commands[*index].witness.clone(),
                );
                return Some(Err(FixedSyncProgramError::Protocol {
                    kind: FixedSyncProtocolKind::Mbarrier,
                    operation,
                    details: format!(
                        "mbarrier {projected_barrier:?} generation {generation} has {}/{} arrivals and {}/{} completed/expected transaction bytes",
                        state.arrival_count,
                        required_arrivals,
                        state.completed_transactions,
                        state.expected_transactions,
                    )
                    .into_boxed_str(),
                }));
            }
            if has_next_generation && state.waits.is_empty() {
                let operation = state.mutations.first().map_or_else(
                    || init.witness.clone(),
                    |index| self.commands[*index].witness.clone(),
                );
                return Some(Err(FixedSyncProgramError::Protocol {
                    kind: FixedSyncProtocolKind::Mbarrier,
                    operation,
                    details: format!(
                        "mbarrier {projected_barrier:?} advances past generation {generation} without a consuming wait"
                    )
                    .into_boxed_str(),
                }));
            }

            if generation > 0 {
                let prior = &generations[&(generation - 1)];
                for &mutation_index in state
                    .mutations
                    .iter()
                    .filter(|&&index| mbarrier_mutation_requires_consumption(&self.commands[index]))
                {
                    let mutation = &self.commands[mutation_index];
                    let mutation_clock = mutation
                        .initial_causal_clock
                        .as_ref()
                        .expect("all mbarrier commands were checked for clocks");
                    let ordered_after_consumption = prior.waits.iter().any(|wait_index| {
                        self.commands[*wait_index]
                            .causal_clock
                            .as_ref()
                            .is_some_and(|clock| strict_happens_before(clock, mutation_clock))
                    });
                    if !ordered_after_consumption {
                        let Some(&prior_wait_index) = prior.waits.first() else {
                            return Some(Err(mbarrier_protocol(
                                &mutation.witness,
                                format!(
                                    "mbarrier {projected_barrier:?} generation {generation} mutation has no prior consuming wait"
                                ),
                            )));
                        };
                        let prior_wait = &self.commands[prior_wait_index];
                        return Some(Err(FixedSyncProgramError::CausalProtocol {
                            kind: FixedSyncProtocolKind::Mbarrier,
                            operation: mutation.witness.clone(),
                            related_operations: Box::new([prior_wait.witness.clone()]),
                            details: format!(
                                "mbarrier {projected_barrier:?} generation {generation} mutation {} can run before generation {} is consumed; prior wait {}, mutation clock {:?}",
                                mutation.witness,
                                generation - 1,
                                prior_wait.witness,
                                mutation_clock,
                            )
                            .into_boxed_str(),
                        }));
                    }
                }
            }

            for &wait_index in &state.waits {
                let wait = &self.commands[wait_index];
                let next_generation = match wait.kind {
                    FixedSyncCommandKind::MbarrierWait {
                        conditional: true,
                        requested_phase,
                        ..
                    } => match self.conditional_wait_successor(
                        projected_barrier,
                        Some(generation),
                        requested_phase,
                        &init.witness,
                        &wait.witness,
                    ) {
                        Ok(next) => next,
                        Err(error) => return Some(Err(error)),
                    },
                    _ => Some(generation + 1),
                };
                let Some((next_generation, next)) = next_generation.and_then(|generation| {
                    generations.get(&generation).map(|next| (generation, next))
                }) else {
                    continue;
                };
                // A parity wait remains valid until the following generation
                // of its own phase type completes. A primary completion with
                // a failed report does not overtake a conditional wait.
                // It need not precede that generation's initial
                // arrive/expect-tx: a later, required transaction completion
                // may carry the causal hand-off that prevents a phase lap.
                let Some(&first_next_mutation) = next.mutations.first() else {
                    return Some(Err(mbarrier_incomplete(
                        state
                            .mutations
                            .first()
                            .map(|index| &self.commands[*index].witness),
                        format!(
                            "mbarrier {projected_barrier:?} generation {} has no generation-defining mutation",
                            next_generation,
                        ),
                    )));
                };
                let wait_clock = wait
                    .initial_causal_clock
                    .as_ref()
                    .expect("all mbarrier commands were checked for clocks");
                let precedes_next_completion_prerequisite =
                    next.mutations.iter().any(|mutation_index| {
                        self.commands[*mutation_index]
                            .initial_causal_clock
                            .as_ref()
                            .is_some_and(|clock| strict_happens_before(wait_clock, clock))
                    });
                if !precedes_next_completion_prerequisite {
                    let next_mutation = &self.commands[first_next_mutation];
                    return Some(Err(FixedSyncProgramError::CausalProtocol {
                            kind: FixedSyncProtocolKind::Mbarrier,
                            operation: wait.witness.clone(),
                            related_operations: Box::new([next_mutation.witness.clone()]),
                            details: format!(
                                "mbarrier {projected_barrier:?} generation {generation} wait {} can be overtaken by generation {} completion; next mutation {}",
                                wait.witness,
                                next_generation,
                                next_mutation.witness,
                            )
                            .into_boxed_str(),
                        }));
                }
            }
            expected_generation = expected_generation.saturating_add(1);
        }
        Some(Ok(()))
    }

    fn build_staged_projections(
        projected: StagedFixedSyncProjections,
        max_workers: usize,
        initial_setmax_pools: &BTreeMap<(usize, usize), SetmaxnregVerifierCore>,
        initial_tcgen_ctas: &BTreeMap<(usize, usize), FixedSyncTcgenCtaState>,
    ) -> Vec<Self> {
        let projection_count = projected.len();
        let mut projected = projected.into_iter().collect::<Vec<_>>();
        projected.sort_unstable_by_key(|(key, _)| *key);
        let worker_count = max_workers
            .max(1)
            .min(MAX_FIXED_SYNC_BUILD_WORKERS)
            .min(projected.len());
        if worker_count <= 1 {
            projected
                .into_iter()
                .map(|(key, commands)| {
                    Self::build_projection(key, commands, initial_setmax_pools, initial_tcgen_ctas)
                })
                .collect()
        } else {
            let chunk_len = projected.len().div_ceil(worker_count);
            std::thread::scope(|scope| {
                let handles = projected
                    .chunks_mut(chunk_len)
                    .map(|chunk| {
                        scope.spawn(move || {
                            chunk
                                .iter_mut()
                                .map(|(key, commands)| {
                                    Self::build_projection(
                                        *key,
                                        std::mem::take(commands),
                                        initial_setmax_pools,
                                        initial_tcgen_ctas,
                                    )
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect::<Vec<_>>();
                let mut projections = Vec::with_capacity(projection_count);
                for handle in handles {
                    projections.extend(
                        handle
                            .join()
                            .unwrap_or_else(|payload| std::panic::resume_unwind(payload)),
                    );
                }
                projections
            })
        }
    }

    fn build_projection(
        key: FixedSyncProjectionKey,
        commands: Vec<StagedFixedSyncCommand>,
        base_initial_setmax_pools: &BTreeMap<(usize, usize), SetmaxnregVerifierCore>,
        base_initial_tcgen_ctas: &BTreeMap<(usize, usize), FixedSyncTcgenCtaState>,
    ) -> Self {
        let uses_linear_causal_certificate = match key {
            FixedSyncProjectionKey::NamedBarrier(_) | FixedSyncProjectionKey::ClusterBarrier(_) => {
                true
            }
            FixedSyncProjectionKey::Mbarrier(_) => {
                !commands.iter().any(|staged| match &staged.command.kind {
                    FixedSyncCommandKind::MbarrierWaitBatch { .. }
                    | FixedSyncCommandKind::MbarrierInvalidate { .. } => true,
                    FixedSyncCommandKind::MbarrierCompletionIssue { .. } => true,
                    FixedSyncCommandKind::MbarrierExpectTx { .. } => true,
                    FixedSyncCommandKind::MbarrierArrive { arrivals } => arrivals
                        .iter()
                        .any(|(_, _, transactions, drop)| transactions.unwrap_or(0) != 0 || *drop),
                    _ => false,
                })
            }
            FixedSyncProjectionKey::SetmaxnregPool { .. }
            | FixedSyncProjectionKey::TcgenCtaComponent { .. } => false,
        };
        if uses_linear_causal_certificate {
            let warp_command_positions =
                if matches!(key, FixedSyncProjectionKey::ClusterBarrier(_)) {
                    let mut programs = BTreeMap::<usize, Vec<(u64, FixedSyncCommandId)>>::new();
                    for (command_index, staged) in commands.iter().enumerate() {
                        let command_id = FixedSyncCommandId(command_index);
                        for &(warp_id, position) in staged.participant_order.iter() {
                            programs
                                .entry(warp_id)
                                .or_default()
                                .push((position, command_id));
                        }
                    }
                    programs
                        .into_iter()
                        .flat_map(|(warp_id, mut program)| {
                            program.sort_unstable_by_key(|(position, _)| *position);
                            program.into_iter().enumerate().map(
                                move |(position, (_, command_id))| {
                                    ((warp_id, command_id), position)
                                },
                            )
                        })
                        .collect()
                } else {
                    HashMap::new()
                };
            return Self {
                commands: commands
                    .into_iter()
                    .map(|staged| staged.command)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                warp_ids: Box::new([]),
                warp_indices: BTreeMap::new(),
                warp_programs: Box::new([]),
                setmax_commands: BTreeMap::new(),
                projection_key: Some(key),
                causal_predecessors: Box::new([]),
                warp_command_positions,
                initial_setmax_pools: BTreeMap::new(),
                initial_tcgen_ctas: BTreeMap::new(),
                conditional_mbarrier_completions: BTreeMap::new(),
            };
        }

        let mut projected_warp_programs = BTreeMap::<usize, Vec<(u64, FixedSyncCommandId)>>::new();
        for (new_index, staged) in commands.iter().enumerate() {
            let new_id = FixedSyncCommandId(new_index);
            for &(warp_id, position) in staged.participant_order.iter() {
                projected_warp_programs
                    .entry(warp_id)
                    .or_default()
                    .push((position, new_id));
            }
        }
        let commands = commands
            .into_iter()
            .map(|staged| staged.command)
            .collect::<Vec<_>>();
        let warp_programs = projected_warp_programs
            .into_iter()
            .map(|(warp_id, mut program)| {
                program.sort_unstable_by_key(|(position, _)| *position);
                (
                    warp_id,
                    program
                        .into_iter()
                        .map(|(_, command)| command)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                )
            })
            .collect::<Vec<_>>();
        let warp_ids = warp_programs
            .iter()
            .map(|(warp_id, _)| *warp_id)
            .collect::<Vec<_>>();
        let warp_indices = warp_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(index, warp_id)| (warp_id, index))
            .collect();
        let warp_programs = warp_programs
            .into_iter()
            .map(|(_, program)| program)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let setmax_commands = commands
            .iter()
            .enumerate()
            .filter_map(|(index, command)| match command.kind {
                FixedSyncCommandKind::Setmax { request, .. } => {
                    Some((request, FixedSyncCommandId(index)))
                }
                _ => None,
            })
            .collect();
        let initial_setmax_pools = match key {
            FixedSyncProjectionKey::SetmaxnregPool {
                kernel_index,
                global_cta_id,
            } => base_initial_setmax_pools
                .get(&(kernel_index, global_cta_id))
                .cloned()
                .map(|pool| BTreeMap::from([((kernel_index, global_cta_id), pool)]))
                .unwrap_or_default(),
            _ => BTreeMap::new(),
        };
        let tcgen_component_ctas = commands
            .iter()
            .filter_map(|command| match &command.kind {
                FixedSyncCommandKind::TcgenLifecycle(request) => {
                    Some(request.participant_ctas.iter().copied())
                }
                _ => None,
            })
            .flatten()
            .collect::<BTreeSet<_>>();
        let initial_tcgen_ctas = match key {
            FixedSyncProjectionKey::TcgenCtaComponent { kernel_index, .. } => {
                base_initial_tcgen_ctas
                    .iter()
                    .filter(|((candidate_kernel, global_cta_id), _)| {
                        *candidate_kernel == kernel_index
                            && tcgen_component_ctas.contains(global_cta_id)
                    })
                    .map(|(key, state)| (*key, state.clone()))
                    .collect()
            }
            _ => BTreeMap::new(),
        };
        let causal_predecessors = build_causal_predecessors(&commands);
        let warp_command_positions = build_warp_command_positions(&warp_ids, &warp_programs);
        let projection = Self {
            commands: commands.into_boxed_slice(),
            warp_ids: warp_ids.into_boxed_slice(),
            warp_indices,
            warp_programs,
            setmax_commands,
            projection_key: Some(key),
            causal_predecessors,
            warp_command_positions,
            initial_setmax_pools,
            initial_tcgen_ctas,
            conditional_mbarrier_completions: BTreeMap::new(),
        };
        projection
            .validate_collective_placement()
            .expect("a protocol projection preserves collective placement");
        projection
    }

    fn validate_collective_placement(&self) -> Result<(), FixedSyncProgramBuildError> {
        for (&warp_id, program) in self.warp_ids.iter().zip(self.warp_programs.iter()) {
            let mut seen = HashSet::with_capacity(program.len());
            for &command_id in program {
                let Some(command) = self.commands.get(command_id.get()) else {
                    return Err(FixedSyncProgramBuildError::InvalidInitialState {
                        details: format!(
                            "warp {warp_id} contains out-of-range command {command_id:?}"
                        )
                        .into_boxed_str(),
                    });
                };
                if !seen.insert(command_id) {
                    return Err(FixedSyncProgramBuildError::InvalidCommand {
                        operation: command.witness.clone(),
                        details: format!(
                            "participant warp {warp_id} contains collective command {command_id:?} more than once"
                        )
                        .into_boxed_str(),
                    });
                }
                if !command.participants.contains(&warp_id) {
                    return Err(FixedSyncProgramBuildError::InvalidCommand {
                        operation: command.witness.clone(),
                        details: format!(
                            "warp {warp_id} contains collective command {command_id:?} without participating"
                        )
                        .into_boxed_str(),
                    });
                }
            }
        }
        for (command_index, command) in self.commands.iter().enumerate() {
            let command_id = FixedSyncCommandId(command_index);
            for &warp_id in command.participants.iter() {
                let Some(warp_index) = self.warp_indices.get(&warp_id).copied() else {
                    return Err(FixedSyncProgramBuildError::InvalidCommand {
                        operation: command.witness.clone(),
                        details: format!("participant warp {warp_id} has no fixed program")
                            .into_boxed_str(),
                    });
                };
                let position = self
                    .warp_command_positions
                    .get(&(warp_id, command_id))
                    .copied();
                if position.is_none_or(|position| {
                    self.warp_programs[warp_index].get(position) != Some(&command_id)
                }) {
                    return Err(FixedSyncProgramBuildError::InvalidCommand {
                        operation: command.witness.clone(),
                        details: format!(
                            "participant warp {warp_id} does not contain collective command {command_id:?}"
                        )
                        .into_boxed_str(),
                    });
                }
            }
        }
        Ok(())
    }
}

fn build_warp_command_positions(
    warp_ids: &[usize],
    warp_programs: &[Box<[FixedSyncCommandId]>],
) -> HashMap<(usize, FixedSyncCommandId), usize> {
    warp_ids
        .iter()
        .copied()
        .zip(warp_programs.iter())
        .flat_map(|(warp_id, program)| {
            program
                .iter()
                .copied()
                .enumerate()
                .map(move |(position, command)| ((warp_id, command), position))
        })
        .collect()
}

fn equivalent_mbarrier_state_search_command(
    left_program: &FixedSyncProgram,
    left_barrier: PhysicalBarrierId,
    left: &FixedSyncCommand,
    right_program: &FixedSyncProgram,
    right_barrier: PhysicalBarrierId,
    right: &FixedSyncCommand,
) -> bool {
    if left_program
        .warp_indices
        .get(&left.witness.global_warp_id())
        != right_program
            .warp_indices
            .get(&right.witness.global_warp_id())
        || left.participants.len() != right.participants.len()
        || !left.participants.iter().zip(right.participants.iter()).all(
            |(left_warp, right_warp)| {
                left_program.warp_indices.get(left_warp)
                    == right_program.warp_indices.get(right_warp)
            },
        )
    {
        return false;
    }

    match (&left.kind, &right.kind) {
        (
            FixedSyncCommandKind::MbarrierInit {
                barrier_ids: left_barriers,
                expected_arrivals: left_expected,
            },
            FixedSyncCommandKind::MbarrierInit {
                barrier_ids: right_barriers,
                expected_arrivals: right_expected,
            },
        ) => {
            left_barriers.as_ref() == [left_barrier]
                && right_barriers.as_ref() == [right_barrier]
                && left_expected == right_expected
                && left_program
                    .conditional_mbarrier_completions
                    .get(&left_barrier)
                    .and_then(|lifetimes| lifetimes.get(&left.witness))
                    == right_program
                        .conditional_mbarrier_completions
                        .get(&right_barrier)
                        .and_then(|lifetimes| lifetimes.get(&right.witness))
        }
        (
            FixedSyncCommandKind::MbarrierInitFence {
                barrier_ids: left_barriers,
            },
            FixedSyncCommandKind::MbarrierInitFence {
                barrier_ids: right_barriers,
            },
        ) => left_barriers.as_ref() == [left_barrier] && right_barriers.as_ref() == [right_barrier],
        (
            FixedSyncCommandKind::MbarrierArrive {
                arrivals: left_arrivals,
            },
            FixedSyncCommandKind::MbarrierArrive {
                arrivals: right_arrivals,
            },
        ) => {
            left_arrivals.len() == right_arrivals.len()
                && left_arrivals.iter().zip(right_arrivals.iter()).all(
                    |(
                        (left_id, left_count, left_transactions, left_drop),
                        (right_id, right_count, right_transactions, right_drop),
                    )| {
                        *left_id == left_barrier
                            && *right_id == right_barrier
                            && left_count == right_count
                            && left_transactions == right_transactions
                            && left_drop == right_drop
                    },
                )
        }
        (
            FixedSyncCommandKind::MbarrierExpectTx {
                expectations: left_expectations,
            },
            FixedSyncCommandKind::MbarrierExpectTx {
                expectations: right_expectations,
            },
        ) => {
            left_expectations.len() == right_expectations.len()
                && left_expectations.iter().zip(right_expectations.iter()).all(
                    |((left_id, left_tx), (right_id, right_tx))| {
                        *left_id == left_barrier
                            && *right_id == right_barrier
                            && left_tx == right_tx
                    },
                )
        }
        (
            FixedSyncCommandKind::MbarrierWait {
                barrier_id: left_id,
                requested_phase: left_phase,
                conditional: left_conditional,
            },
            FixedSyncCommandKind::MbarrierWait {
                barrier_id: right_id,
                requested_phase: right_phase,
                conditional: right_conditional,
            },
        ) => {
            *left_id == left_barrier
                && *right_id == right_barrier
                && left_phase == right_phase
                && left_conditional == right_conditional
        }
        (
            FixedSyncCommandKind::MbarrierWaitBatch {
                waits: left,
                conditional: left_conditional,
            },
            FixedSyncCommandKind::MbarrierWaitBatch {
                waits: right,
                conditional: right_conditional,
            },
        ) => left == right && left_conditional == right_conditional,
        (
            FixedSyncCommandKind::MbarrierCompletionIssue {
                completions: left_completions,
            },
            FixedSyncCommandKind::MbarrierCompletionIssue {
                completions: right_completions,
            },
        ) => {
            left_completions.len() == right_completions.len()
                && left_completions.iter().zip(right_completions.iter()).all(
                    |(left_completion, right_completion)| {
                        if left_completion.barrier_id != left_barrier
                            || right_completion.barrier_id != right_barrier
                        {
                            return false;
                        }
                        match (left_completion.kind, right_completion.kind) {
                            (
                                FixedSyncMbarrierCompletionKind::Transaction {
                                    transactions: left_transactions,
                                },
                                FixedSyncMbarrierCompletionKind::Transaction {
                                    transactions: right_transactions,
                                },
                            ) => left_transactions == right_transactions,
                            (
                                FixedSyncMbarrierCompletionKind::Arrival {
                                    warp_id: left_warp,
                                    arrival_count: left_count,
                                    pending_increase: left_pending,
                                },
                                FixedSyncMbarrierCompletionKind::Arrival {
                                    warp_id: right_warp,
                                    arrival_count: right_count,
                                    pending_increase: right_pending,
                                },
                            ) => {
                                left_count == right_count
                                    && left_pending == right_pending
                                    && left_program.warp_indices.get(&left_warp)
                                        == right_program.warp_indices.get(&right_warp)
                            }
                            _ => false,
                        }
                    },
                )
        }
        _ => false,
    }
}

fn strict_happens_before(left: &crate::SyncVectorClock, right: &crate::SyncVectorClock) -> bool {
    left.happens_before(right) && !right.happens_before(left)
}

fn latest_causal_mbarrier_init<'a>(
    issue_clock: Option<&crate::SyncVectorClock>,
    inits: &[&'a FixedSyncCommand],
) -> Option<&'a FixedSyncCommand> {
    let precedes_issue = |init: &&FixedSyncCommand| {
        init.causal_clock
            .as_ref()
            .zip(issue_clock)
            .is_some_and(|(init_clock, issue_clock)| strict_happens_before(init_clock, issue_clock))
    };
    inits
        .iter()
        .copied()
        .filter(precedes_issue)
        .find(|candidate| {
            inits.iter().copied().filter(precedes_issue).all(|prior| {
                prior.witness == candidate.witness
                    || prior
                        .causal_clock
                        .as_ref()
                        .zip(candidate.initial_causal_clock.as_ref())
                        .is_some_and(|(prior_clock, next_clock)| {
                            strict_happens_before(prior_clock, next_clock)
                        })
            })
        })
}

/// Map every TCGEN-owning CTA to the lowest-numbered CTA in its connected
/// lifecycle component. TCGEN state is local to one CTA, except that a
/// two-CTA lifecycle operation atomically touches both participant CTAs.
/// Grouping by connected components therefore preserves every possible
/// protocol interaction without multiplying independent CTA state machines.
fn tcgen_component_anchors(commands: &[FixedSyncCommand]) -> BTreeMap<(usize, usize), usize> {
    let mut adjacency = BTreeMap::<(usize, usize), BTreeSet<usize>>::new();
    for command in commands {
        let FixedSyncCommandKind::TcgenLifecycle(request) = &command.kind else {
            continue;
        };
        for &global_cta_id in request.participant_ctas.iter() {
            adjacency
                .entry((request.kernel_index, global_cta_id))
                .or_default()
                .extend(request.participant_ctas.iter().copied());
        }
    }

    let mut anchors = BTreeMap::new();
    for &(kernel_index, first_global_cta_id) in adjacency.keys() {
        if anchors.contains_key(&(kernel_index, first_global_cta_id)) {
            continue;
        }
        let anchor_global_cta_id = first_global_cta_id;
        let mut pending = vec![first_global_cta_id];
        while let Some(global_cta_id) = pending.pop() {
            if anchors.contains_key(&(kernel_index, global_cta_id)) {
                continue;
            }
            anchors.insert((kernel_index, global_cta_id), anchor_global_cta_id);
            if let Some(neighbors) = adjacency.get(&(kernel_index, global_cta_id)) {
                pending.extend(
                    neighbors
                        .iter()
                        .copied()
                        .filter(|neighbor| !anchors.contains_key(&(kernel_index, *neighbor))),
                );
            }
        }
    }
    anchors
}

/// Map barriers joined by one warp-level wait instruction to a shared
/// projection anchor. A wait spanning several barrier/phase targets is one
/// atomic blocking command, so those barriers cannot be verified as independent
/// state machines.
fn mbarrier_component_anchors(
    operations: &[FixedSyncOperationSnapshot<'_>],
) -> BTreeMap<PhysicalBarrierId, PhysicalBarrierId> {
    let mut adjacency = BTreeMap::<PhysicalBarrierId, BTreeSet<PhysicalBarrierId>>::new();
    for &(_, summary, _, _) in operations {
        let Some(sync) = synchronization_summary(summary) else {
            continue;
        };
        if !matches!(sync.details(), OwnedOperationEffect::MbarrierWait { .. }) {
            continue;
        }
        let barriers = sync
            .mbarrier_wait_requests()
            .iter()
            .map(|request| request.0)
            .collect::<BTreeSet<_>>();
        let Some(&first) = barriers.first() else {
            continue;
        };
        if barriers.len() == 1 {
            continue;
        }
        for &barrier in &barriers {
            adjacency.entry(first).or_default().insert(barrier);
            adjacency.entry(barrier).or_default().insert(first);
        }
    }

    let mut anchors = BTreeMap::new();
    for &start in adjacency.keys() {
        if anchors.contains_key(&start) {
            continue;
        }
        let mut component = BTreeSet::new();
        let mut pending = vec![start];
        while let Some(barrier) = pending.pop() {
            if !component.insert(barrier) {
                continue;
            }
            if let Some(neighbors) = adjacency.get(&barrier) {
                pending.extend(neighbors.iter().copied());
            }
        }
        let anchor = *component
            .first()
            .expect("mbarrier component contains its seed");
        anchors.extend(component.into_iter().map(|barrier| (barrier, anchor)));
    }
    anchors
}

fn mbarrier_component_anchor(
    barrier_id: PhysicalBarrierId,
    anchors: &BTreeMap<PhysicalBarrierId, PhysicalBarrierId>,
) -> PhysicalBarrierId {
    anchors.get(&barrier_id).copied().unwrap_or(barrier_id)
}

fn build_causal_predecessors(commands: &[FixedSyncCommand]) -> Box<[Box<[FixedSyncCommandId]>]> {
    commands
        .iter()
        .enumerate()
        .map(|(right_index, right)| {
            // Readiness is defined at command issue. A blocking wait's final
            // clock includes the operation that wakes it; using that acquired
            // clock here would require the wake-up to complete before the wait
            // can even register.
            let Some(right_clock) = &right.initial_causal_clock else {
                return Box::new([]) as Box<[FixedSyncCommandId]>;
            };
            commands
                .iter()
                .enumerate()
                .filter_map(|(left_index, left)| {
                    let left_clock = left.causal_clock.as_ref()?;
                    (left_index != right_index && strict_happens_before(left_clock, right_clock))
                        .then_some(FixedSyncCommandId(left_index))
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn stage_noncollective_operations(
    operations: &[FixedSyncOperationSnapshot<'_>],
    collective_operation_commands: &HashMap<DynamicOpId, FixedSyncCommandId>,
    mbarrier_component_anchors: &BTreeMap<PhysicalBarrierId, PhysicalBarrierId>,
    tcgen_component_anchors: &BTreeMap<(usize, usize), usize>,
    completion_mbarrier_generations: &BTreeMap<u64, Box<[(PhysicalBarrierId, u64)]>>,
) -> Result<StagedFixedSyncProjections, FixedSyncProgramBuildError> {
    let mut projected = StagedFixedSyncProjections::new();
    for &(operation, summary, initial_clock, clock) in operations {
        if collective_operation_commands.contains_key(operation) {
            continue;
        }
        let Some(kind) = command_from_summary(operation, summary)? else {
            continue;
        };
        let warp_id = operation.global_warp_id();
        stage_fixed_sync_command(
            FixedSyncCommand {
                witness: operation.clone(),
                participants: CompactSlice::one(warp_id),
                kind,
                canonical_generation: canonical_protocol_generation(
                    summary,
                    completion_mbarrier_generations,
                ),
                initial_causal_clock: initial_clock.cloned(),
                causal_clock: clock.cloned(),
            },
            CompactSlice::one((warp_id, operation.per_warp_sequence())),
            mbarrier_component_anchors,
            tcgen_component_anchors,
            &mut projected,
        );
    }
    Ok(projected)
}

fn merge_staged_fixed_sync_projections(
    target: &mut StagedFixedSyncProjections,
    source: StagedFixedSyncProjections,
) {
    for (key, mut commands) in source {
        target.entry(key).or_default().append(&mut commands);
    }
}

fn stage_fixed_sync_command(
    command: FixedSyncCommand,
    participant_order: CompactSlice<(usize, u64)>,
    mbarrier_component_anchors: &BTreeMap<PhysicalBarrierId, PhysicalBarrierId>,
    tcgen_component_anchors: &BTreeMap<(usize, usize), usize>,
    projected: &mut StagedFixedSyncProjections,
) {
    let FixedSyncCommand {
        witness,
        participants,
        kind,
        canonical_generation,
        initial_causal_clock,
        causal_clock,
    } = command;
    let mut remaining = projected_command_kind_count(&kind, mbarrier_component_anchors);
    let mut metadata = Some((
        witness,
        participants,
        participant_order,
        initial_causal_clock,
        causal_clock,
    ));
    for_each_projected_command_kind(
        &kind,
        mbarrier_component_anchors,
        tcgen_component_anchors,
        |key, projected_kind| {
            remaining -= 1;
            let (witness, participants, participant_order, initial_causal_clock, causal_clock) =
                if remaining == 0 {
                    metadata
                        .take()
                        .expect("the final projection owns command metadata")
                } else {
                    let (
                        witness,
                        participants,
                        participant_order,
                        initial_causal_clock,
                        causal_clock,
                    ) = metadata
                        .as_ref()
                        .expect("non-final projections clone command metadata");
                    (
                        witness.clone(),
                        participants.clone(),
                        participant_order.clone(),
                        initial_causal_clock.clone(),
                        causal_clock.clone(),
                    )
                };
            projected
                .entry(key)
                .or_default()
                .push(StagedFixedSyncCommand {
                    command: FixedSyncCommand {
                        witness,
                        participants,
                        kind: projected_kind,
                        canonical_generation,
                        initial_causal_clock,
                        causal_clock,
                    },
                    participant_order,
                });
        },
    );
    debug_assert_eq!(remaining, 0);
}

fn projected_command_kind_count(
    kind: &FixedSyncCommandKind,
    mbarrier_component_anchors: &BTreeMap<PhysicalBarrierId, PhysicalBarrierId>,
) -> usize {
    match kind {
        FixedSyncCommandKind::MbarrierInit { barrier_ids, .. }
        | FixedSyncCommandKind::MbarrierInvalidate { barrier_ids }
        | FixedSyncCommandKind::MbarrierInitFence { barrier_ids } => barrier_ids
            .iter()
            .map(|barrier_id| mbarrier_component_anchor(*barrier_id, mbarrier_component_anchors))
            .collect::<BTreeSet<_>>()
            .len(),
        FixedSyncCommandKind::MbarrierArrive { arrivals } => arrivals
            .iter()
            .map(|arrival| mbarrier_component_anchor(arrival.0, mbarrier_component_anchors))
            .collect::<BTreeSet<_>>()
            .len(),
        FixedSyncCommandKind::MbarrierExpectTx { expectations } => expectations
            .iter()
            .map(|expectation| mbarrier_component_anchor(expectation.0, mbarrier_component_anchors))
            .collect::<BTreeSet<_>>()
            .len(),
        FixedSyncCommandKind::MbarrierCompletionIssue { completions } => completions
            .iter()
            .map(|completion| {
                mbarrier_component_anchor(completion.barrier_id, mbarrier_component_anchors)
            })
            .collect::<BTreeSet<_>>()
            .len(),
        FixedSyncCommandKind::MbarrierWaitBatch { .. } => 1,
        FixedSyncCommandKind::MbarrierWait { .. }
        | FixedSyncCommandKind::NamedBarrierArrive { .. }
        | FixedSyncCommandKind::NamedBarrierSync { .. }
        | FixedSyncCommandKind::ClusterBarrierArrive { .. }
        | FixedSyncCommandKind::ClusterBarrierWait { .. }
        | FixedSyncCommandKind::Setmax { .. }
        | FixedSyncCommandKind::TcgenLifecycle(_) => 1,
    }
}

fn for_each_projected_command_kind(
    kind: &FixedSyncCommandKind,
    mbarrier_component_anchors: &BTreeMap<PhysicalBarrierId, PhysicalBarrierId>,
    tcgen_component_anchors: &BTreeMap<(usize, usize), usize>,
    mut emit: impl FnMut(FixedSyncProjectionKey, FixedSyncCommandKind),
) {
    match kind {
        FixedSyncCommandKind::MbarrierInit {
            barrier_ids,
            expected_arrivals,
        } => {
            let mut grouped = BTreeMap::<PhysicalBarrierId, Vec<PhysicalBarrierId>>::new();
            for barrier_id in barrier_ids.iter().copied() {
                grouped
                    .entry(mbarrier_component_anchor(
                        barrier_id,
                        mbarrier_component_anchors,
                    ))
                    .or_default()
                    .push(barrier_id);
            }
            for (anchor, barrier_ids) in grouped {
                emit(
                    FixedSyncProjectionKey::Mbarrier(anchor),
                    FixedSyncCommandKind::MbarrierInit {
                        barrier_ids: barrier_ids.into_boxed_slice(),
                        expected_arrivals: *expected_arrivals,
                    },
                );
            }
        }
        FixedSyncCommandKind::MbarrierInitFence { barrier_ids }
        | FixedSyncCommandKind::MbarrierInvalidate { barrier_ids } => {
            let mut grouped = BTreeMap::<PhysicalBarrierId, Vec<PhysicalBarrierId>>::new();
            for barrier_id in barrier_ids.iter().copied() {
                grouped
                    .entry(mbarrier_component_anchor(
                        barrier_id,
                        mbarrier_component_anchors,
                    ))
                    .or_default()
                    .push(barrier_id);
            }
            for (anchor, barrier_ids) in grouped {
                emit(
                    FixedSyncProjectionKey::Mbarrier(anchor),
                    if matches!(kind, FixedSyncCommandKind::MbarrierInvalidate { .. }) {
                        FixedSyncCommandKind::MbarrierInvalidate {
                            barrier_ids: barrier_ids.into_boxed_slice(),
                        }
                    } else {
                        FixedSyncCommandKind::MbarrierInitFence {
                            barrier_ids: barrier_ids.into_boxed_slice(),
                        }
                    },
                );
            }
        }
        FixedSyncCommandKind::MbarrierArrive { arrivals } => {
            let mut grouped = BTreeMap::<
                PhysicalBarrierId,
                Vec<(PhysicalBarrierId, u64, Option<u64>, bool)>,
            >::new();
            for arrival in arrivals.iter().copied() {
                grouped
                    .entry(mbarrier_component_anchor(
                        arrival.0,
                        mbarrier_component_anchors,
                    ))
                    .or_default()
                    .push(arrival);
            }
            for (anchor, arrivals) in grouped {
                emit(
                    FixedSyncProjectionKey::Mbarrier(anchor),
                    FixedSyncCommandKind::MbarrierArrive {
                        arrivals: arrivals.into_boxed_slice(),
                    },
                );
            }
        }
        FixedSyncCommandKind::MbarrierExpectTx { expectations } => {
            let mut grouped =
                BTreeMap::<PhysicalBarrierId, BTreeMap<PhysicalBarrierId, u64>>::new();
            for &(barrier_id, transactions) in expectations.iter() {
                let anchor = mbarrier_component_anchor(barrier_id, mbarrier_component_anchors);
                let total = grouped
                    .entry(anchor)
                    .or_default()
                    .entry(barrier_id)
                    .or_default();
                *total = total
                    .checked_add(transactions)
                    .expect("validated mbarrier expectation plan must fit u64");
            }
            for (anchor, expectations) in grouped {
                emit(
                    FixedSyncProjectionKey::Mbarrier(anchor),
                    FixedSyncCommandKind::MbarrierExpectTx {
                        expectations: expectations
                            .into_iter()
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                    },
                );
            }
        }
        FixedSyncCommandKind::MbarrierWait {
            barrier_id,
            requested_phase,
            conditional,
        } => emit(
            FixedSyncProjectionKey::Mbarrier(mbarrier_component_anchor(
                *barrier_id,
                mbarrier_component_anchors,
            )),
            FixedSyncCommandKind::MbarrierWait {
                barrier_id: *barrier_id,
                requested_phase: *requested_phase,
                conditional: *conditional,
            },
        ),
        FixedSyncCommandKind::MbarrierWaitBatch { waits, conditional } => {
            let anchor = mbarrier_component_anchor(waits[0].0, mbarrier_component_anchors);
            debug_assert!(waits.iter().all(|wait| {
                mbarrier_component_anchor(wait.0, mbarrier_component_anchors) == anchor
            }));
            emit(
                FixedSyncProjectionKey::Mbarrier(anchor),
                FixedSyncCommandKind::MbarrierWaitBatch {
                    waits: waits.clone(),
                    conditional: *conditional,
                },
            );
        }
        FixedSyncCommandKind::MbarrierCompletionIssue { completions } => {
            let mut grouped =
                BTreeMap::<PhysicalBarrierId, Vec<FixedSyncMbarrierCompletion>>::new();
            for completion in completions.iter().copied() {
                grouped
                    .entry(mbarrier_component_anchor(
                        completion.barrier_id,
                        mbarrier_component_anchors,
                    ))
                    .or_default()
                    .push(completion);
            }
            for (anchor, completions) in grouped {
                emit(
                    FixedSyncProjectionKey::Mbarrier(anchor),
                    FixedSyncCommandKind::MbarrierCompletionIssue {
                        completions: completions.into_boxed_slice(),
                    },
                );
            }
        }
        FixedSyncCommandKind::NamedBarrierArrive {
            barrier_id,
            expected_arrivals,
            warp_id,
            arrival_mask,
        } => emit(
            FixedSyncProjectionKey::NamedBarrier(*barrier_id),
            FixedSyncCommandKind::NamedBarrierArrive {
                barrier_id: *barrier_id,
                expected_arrivals: *expected_arrivals,
                warp_id: *warp_id,
                arrival_mask: *arrival_mask,
            },
        ),
        FixedSyncCommandKind::NamedBarrierSync {
            barrier_id,
            expected_arrivals,
            warp_id,
            arrival_mask,
            aligned,
        } => emit(
            FixedSyncProjectionKey::NamedBarrier(*barrier_id),
            FixedSyncCommandKind::NamedBarrierSync {
                barrier_id: *barrier_id,
                expected_arrivals: *expected_arrivals,
                warp_id: *warp_id,
                arrival_mask: *arrival_mask,
                aligned: *aligned,
            },
        ),
        FixedSyncCommandKind::ClusterBarrierArrive {
            barrier_id,
            warp_id,
            arrival_mask,
            participant_warps,
        } => emit(
            FixedSyncProjectionKey::ClusterBarrier(*barrier_id),
            FixedSyncCommandKind::ClusterBarrierArrive {
                barrier_id: *barrier_id,
                warp_id: *warp_id,
                arrival_mask: *arrival_mask,
                participant_warps: participant_warps.clone(),
            },
        ),
        FixedSyncCommandKind::ClusterBarrierWait {
            barrier_id,
            warp_id,
            arrival_mask,
            participant_warps,
        } => emit(
            FixedSyncProjectionKey::ClusterBarrier(*barrier_id),
            FixedSyncCommandKind::ClusterBarrierWait {
                barrier_id: *barrier_id,
                warp_id: *warp_id,
                arrival_mask: *arrival_mask,
                participant_warps: participant_warps.clone(),
            },
        ),
        FixedSyncCommandKind::Setmax {
            request,
            action,
            target_count,
        } => emit(
            FixedSyncProjectionKey::SetmaxnregPool {
                kernel_index: request.kernel_index(),
                global_cta_id: request.global_cta_id(),
            },
            FixedSyncCommandKind::Setmax {
                request: *request,
                action: *action,
                target_count: *target_count,
            },
        ),
        FixedSyncCommandKind::TcgenLifecycle(request) => {
            let anchor_global_cta_id = request
                .participant_ctas
                .first()
                .and_then(|global_cta_id| {
                    tcgen_component_anchors.get(&(request.kernel_index, *global_cta_id))
                })
                .copied()
                .expect("a TCGEN lifecycle command has a component anchor");
            debug_assert!(request.participant_ctas.iter().all(|global_cta_id| {
                tcgen_component_anchors.get(&(request.kernel_index, *global_cta_id))
                    == Some(&anchor_global_cta_id)
            }));
            emit(
                FixedSyncProjectionKey::TcgenCtaComponent {
                    kernel_index: request.kernel_index,
                    anchor_global_cta_id,
                },
                FixedSyncCommandKind::TcgenLifecycle(request.clone()),
            );
        }
    }
}

fn synchronization_summary(
    summary: &ResolvedTransitionSummary,
) -> Option<&ResolvedSynchronizationEffect> {
    match summary {
        ResolvedTransitionSummary::Synchronization(sync) => Some(sync),
        ResolvedTransitionSummary::AsyncPayload(payload) => Some(payload.synchronization()),
        _ => None,
    }
}

fn canonical_protocol_generation(
    summary: &ResolvedTransitionSummary,
    completion_mbarrier_generations: &BTreeMap<u64, Box<[(PhysicalBarrierId, u64)]>>,
) -> Option<u64> {
    let synchronization = synchronization_summary(summary)?;
    let mut generations = synchronization
        .resources()
        .iter()
        .filter_map(|resource| match resource.key() {
            ResolvedSyncResourceKey::PhysicalMbarrier(_)
            | ResolvedSyncResourceKey::NamedBarrier(_)
            | ResolvedSyncResourceKey::ClusterBarrier(_) => resource.generation(),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    if generations.is_empty() {
        match synchronization.details() {
            OwnedOperationEffect::MbarrierCompletionIssue {
                plan,
                action_ids: Some(action_ids),
            } => {
                for (&(barrier_id, transactions), &action_id) in
                    plan.completions().iter().zip(action_ids.iter())
                {
                    if transactions == 0 {
                        continue;
                    }
                    generations.extend(
                        completion_mbarrier_generations
                            .get(&action_id.get())
                            .into_iter()
                            .flat_map(|targets| targets.iter())
                            .filter_map(|&(target, generation)| {
                                (target == barrier_id).then_some(generation)
                            }),
                    );
                }
            }
            OwnedOperationEffect::TcgenCommitIssue {
                actions: Some(actions),
                ..
            } => {
                generations.extend(actions.iter().map(|action| action.generation()));
            }
            // `cp.async.mbarrier.arrive` enqueues arrival completions, not
            // transaction completions, so it names its generations the same way
            // `tcgen05.commit` does: through the actions it just enqueued.
            OwnedOperationEffect::CpAsyncMbarrierArrive {
                actions: Some(actions),
                ..
            } => {
                generations.extend(actions.iter().map(|action| action.generation()));
            }
            _ => {}
        }
    }
    (generations.len() == 1)
        .then(|| generations.first().copied())
        .flatten()
}

fn merge_command_clock(
    destination: &mut Option<crate::SyncVectorClock>,
    clock: &crate::SyncVectorClock,
) {
    match destination {
        Some(current) => {
            let mut joined = crate::SyncClockPayload::from_clock(current.clone());
            joined
                .merge(&crate::SyncClockPayload::from_clock(clock.clone()))
                .expect("recorded operation clocks share one launch domain");
            *current = joined.clock().clone();
        }
        None => *destination = Some(clock.clone()),
    }
}

fn physical_resources(sync: &ResolvedSynchronizationEffect) -> Vec<PhysicalBarrierId> {
    sync.resources()
        .iter()
        .filter_map(|resource| match resource.key() {
            ResolvedSyncResourceKey::PhysicalMbarrier(barrier_id) => Some(barrier_id),
            _ => None,
        })
        .collect()
}

fn one_named_resource(
    operation: &DynamicOpId,
    sync: &ResolvedSynchronizationEffect,
) -> Result<NamedBarrierId, FixedSyncProgramBuildError> {
    let mut resources = sync
        .resources()
        .iter()
        .filter_map(|resource| match resource.key() {
            ResolvedSyncResourceKey::NamedBarrier(barrier_id) => Some(barrier_id),
            _ => None,
        });
    match (resources.next(), resources.next()) {
        (Some(barrier_id), None) => Ok(barrier_id),
        _ => Err(FixedSyncProgramBuildError::InvalidCommand {
            operation: operation.clone(),
            details: format!(
                "expected one named-barrier resource, got {:?}",
                sync.resources()
            )
            .into_boxed_str(),
        }),
    }
}

fn one_cluster_resource(
    operation: &DynamicOpId,
    sync: &ResolvedSynchronizationEffect,
) -> Result<ClusterBarrierId, FixedSyncProgramBuildError> {
    let mut resources = sync
        .resources()
        .iter()
        .filter_map(|resource| match resource.key() {
            ResolvedSyncResourceKey::ClusterBarrier(barrier_id) => Some(barrier_id),
            _ => None,
        });
    match (resources.next(), resources.next()) {
        (Some(barrier_id), None) => Ok(barrier_id),
        _ => Err(FixedSyncProgramBuildError::InvalidCommand {
            operation: operation.clone(),
            details: format!(
                "expected one cluster-barrier resource, got {:?}",
                sync.resources()
            )
            .into_boxed_str(),
        }),
    }
}

fn one_physical_resource(
    operation: &DynamicOpId,
    sync: &ResolvedSynchronizationEffect,
) -> Result<PhysicalBarrierId, FixedSyncProgramBuildError> {
    let mut resources = sync
        .resources()
        .iter()
        .filter_map(|resource| match resource.key() {
            ResolvedSyncResourceKey::PhysicalMbarrier(barrier_id) => Some(barrier_id),
            _ => None,
        });
    match (resources.next(), resources.next()) {
        (Some(barrier_id), None) => Ok(barrier_id),
        _ => Err(FixedSyncProgramBuildError::InvalidCommand {
            operation: operation.clone(),
            details: format!(
                "expected one physical mbarrier resource, got {:?}",
                sync.resources()
            )
            .into_boxed_str(),
        }),
    }
}

/// Both `cuda.cluster_arrive` and the arrive half of `cuda.cluster_sync`
/// use the canonical split-arrival plan.
fn cluster_arrive_command(
    operation: &DynamicOpId,
    sync: &ResolvedSynchronizationEffect,
    warp_id: usize,
    arrival_mask: WarpMask,
    participant_warps: &[usize],
    aligned: bool,
) -> Result<FixedSyncCommandKind, FixedSyncProgramBuildError> {
    if !aligned && !arrival_mask.is_full() {
        return Err(FixedSyncProgramBuildError::UnsupportedOperation {
            operation: operation.clone(),
            details: "partial-warp unaligned cluster-barrier semantics are not fixed-state modeled"
                .into(),
        });
    }
    Ok(FixedSyncCommandKind::ClusterBarrierArrive {
        barrier_id: one_cluster_resource(operation, sync)?,
        warp_id,
        arrival_mask,
        participant_warps: participant_warps.into(),
    })
}

fn command_from_summary(
    operation: &DynamicOpId,
    summary: &ResolvedTransitionSummary,
) -> Result<Option<FixedSyncCommandKind>, FixedSyncProgramBuildError> {
    let sync = match summary {
        ResolvedTransitionSummary::Memory(_) | ResolvedTransitionSummary::Completion(_) => {
            return Ok(None);
        }
        ResolvedTransitionSummary::AnalysisGap(gap) => {
            return Err(FixedSyncProgramBuildError::UnsupportedOperation {
                operation: operation.clone(),
                details: format!(
                    "analysis gap {:?} on resolved resources {:?}",
                    gap.kind(),
                    gap.resources()
                )
                .into_boxed_str(),
            });
        }
        ResolvedTransitionSummary::Unknown { reason } => {
            return Err(FixedSyncProgramBuildError::UnsupportedOperation {
                operation: operation.clone(),
                details: format!("unknown resolved effect: {reason}").into_boxed_str(),
            });
        }
        ResolvedTransitionSummary::Unsupported { effect_name } => {
            return Err(FixedSyncProgramBuildError::UnsupportedOperation {
                operation: operation.clone(),
                details: format!("unsupported resolved effect: {effect_name}").into_boxed_str(),
            });
        }
        ResolvedTransitionSummary::Synchronization(sync) => sync,
        ResolvedTransitionSummary::AsyncPayload(payload) => payload.synchronization(),
    };
    let kind = match sync.details() {
        OwnedOperationEffect::MbarrierInvalidate { .. } => {
            FixedSyncCommandKind::MbarrierInvalidate {
                barrier_ids: physical_resources(sync).into_boxed_slice(),
            }
        }
        OwnedOperationEffect::MbarrierInit(plan) => FixedSyncCommandKind::MbarrierInit {
            barrier_ids: physical_resources(sync).into_boxed_slice(),
            expected_arrivals: plan.expected_arrivals(),
        },
        OwnedOperationEffect::MbarrierInitFence { .. } => FixedSyncCommandKind::MbarrierInitFence {
            barrier_ids: physical_resources(sync).into_boxed_slice(),
        },
        OwnedOperationEffect::MbarrierArrive { plan, .. } => FixedSyncCommandKind::MbarrierArrive {
            arrivals: Box::new([(
                one_physical_resource(operation, sync)?,
                plan.arrival_count(),
                plan.expected_transactions(),
                plan.is_drop(),
            )]),
        },
        OwnedOperationEffect::MbarrierArriveBatch { plan, .. } => {
            FixedSyncCommandKind::MbarrierArrive {
                arrivals: plan
                    .entries()
                    .iter()
                    .map(|entry| {
                        let arrival = entry.plan();
                        (
                            arrival.barrier_id(),
                            arrival.arrival_count(),
                            arrival.expected_transactions(),
                            arrival.is_drop(),
                        )
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            }
        }
        OwnedOperationEffect::MbarrierExpectTx { plan, .. } => {
            FixedSyncCommandKind::MbarrierExpectTx {
                expectations: plan
                    .entries()
                    .iter()
                    .map(|entry| (entry.barrier_id(), entry.expected_transactions()))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            }
        }
        // A declared word's wait carries no fixed-synchronization command: it
        // names a memory address, not a barrier resource.
        OwnedOperationEffect::DeclaredWordWait { .. } => return Ok(None),
        OwnedOperationEffect::MbarrierWait { plan, .. } => {
            let waits = sync.mbarrier_wait_requests();
            if waits.len() == 1 {
                FixedSyncCommandKind::MbarrierWait {
                    barrier_id: waits[0].0,
                    requested_phase: waits[0].1,
                    conditional: plan.is_conditional(),
                }
            } else {
                FixedSyncCommandKind::MbarrierWaitBatch {
                    waits: waits.into(),
                    conditional: plan.is_conditional(),
                }
            }
        }
        OwnedOperationEffect::MbarrierCompletionIssue { plan, .. } => {
            FixedSyncCommandKind::MbarrierCompletionIssue {
                completions: plan
                    .completions()
                    .iter()
                    .map(|&(barrier_id, transactions)| FixedSyncMbarrierCompletion {
                        barrier_id,
                        kind: FixedSyncMbarrierCompletionKind::Transaction { transactions },
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            }
        }
        OwnedOperationEffect::CpAsyncMbarrierArrive { plan, .. } => {
            FixedSyncCommandKind::MbarrierCompletionIssue {
                completions: plan
                    .targets()
                    .map(|(_, barrier_id)| FixedSyncMbarrierCompletion {
                        barrier_id,
                        kind: FixedSyncMbarrierCompletionKind::Arrival {
                            warp_id: plan.warp_id(),
                            arrival_count: 1,
                            pending_increase: u64::from(plan.increments_pending()),
                        },
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            }
        }
        OwnedOperationEffect::TcgenCommitIssue { plan, .. } => {
            FixedSyncCommandKind::MbarrierCompletionIssue {
                completions: plan
                    .barrier_ids()
                    .iter()
                    .copied()
                    .map(|barrier_id| FixedSyncMbarrierCompletion {
                        barrier_id,
                        kind: FixedSyncMbarrierCompletionKind::Arrival {
                            warp_id: operation.global_warp_id(),
                            arrival_count: plan.arrival_count(),
                            pending_increase: 0,
                        },
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            }
        }
        OwnedOperationEffect::NamedBarrierArrive { plan, .. } => {
            FixedSyncCommandKind::NamedBarrierArrive {
                barrier_id: one_named_resource(operation, sync)?,
                expected_arrivals: plan.expected_arrivals(),
                warp_id: plan.warp_id(),
                arrival_mask: plan.arrival_mask(),
            }
        }
        OwnedOperationEffect::NamedBarrierSyncRegister { plan, .. } => {
            FixedSyncCommandKind::NamedBarrierSync {
                barrier_id: one_named_resource(operation, sync)?,
                expected_arrivals: plan.expected_arrivals(),
                warp_id: plan.warp_id(),
                arrival_mask: plan.arrival_mask(),
                aligned: plan.aligned(),
            }
        }
        OwnedOperationEffect::ClusterBarrierArrive { plan, .. } => cluster_arrive_command(
            operation,
            sync,
            plan.warp_id(),
            plan.arrival_mask(),
            plan.participant_warps(),
            plan.aligned(),
        )?,
        OwnedOperationEffect::ClusterBarrierWaitRegister { plan, .. } => {
            if !plan.aligned() && !plan.arrival_mask().is_full() {
                return Err(FixedSyncProgramBuildError::UnsupportedOperation {
                    operation: operation.clone(),
                    details: "partial-warp unaligned cluster-barrier semantics are not fixed-state modeled"
                        .into(),
                });
            }
            FixedSyncCommandKind::ClusterBarrierWait {
                barrier_id: one_cluster_resource(operation, sync)?,
                warp_id: plan.warp_id(),
                arrival_mask: plan.arrival_mask(),
                participant_warps: plan.participant_warps().into(),
            }
        }
        OwnedOperationEffect::TcgenLifecycleRegister(_)
        | OwnedOperationEffect::SetmaxnregRegister(_) => {
            return Err(FixedSyncProgramBuildError::InvalidCommand {
                operation: operation.clone(),
                details: "collective command was not aggregated".into(),
            });
        }
        OwnedOperationEffect::TcgenWorkIssue(_)
        | OwnedOperationEffect::TcgenWait { .. }
        | OwnedOperationEffect::AsyncGroupIssue(_)
        | OwnedOperationEffect::AsyncGroupIssueBatch(_)
        | OwnedOperationEffect::AsyncGroupCommit { .. }
        | OwnedOperationEffect::AsyncGroupWait { .. }
        | OwnedOperationEffect::MemoryFence(_)
        | OwnedOperationEffect::ProxyAsyncFence(_)
        | OwnedOperationEffect::TensorMap(_)
        | OwnedOperationEffect::TcgenFence(_)
        | OwnedOperationEffect::WarpSync(_) => return Ok(None),
        // Resume spellings are folded onto their registration at storage time,
        // and the three non-synchronization effects never reach a
        // `Synchronization` summary. Listed rather than caught by a wildcard so
        // a new effect kind still has to be classified here.
        OwnedOperationEffect::NamedBarrierSyncResume(_)
        | OwnedOperationEffect::ClusterBarrierWaitResume(_)
        | OwnedOperationEffect::TcgenLifecycleResume(_)
        | OwnedOperationEffect::SetmaxnregResume(_)
        | OwnedOperationEffect::PhysicalAccess(_)
        | OwnedOperationEffect::AsyncPayload(_)
        | OwnedOperationEffect::AnalysisGap(_) => {
            return Err(FixedSyncProgramBuildError::InvalidCommand {
                operation: operation.clone(),
                details: "effect is not a stored synchronization payload".into(),
            });
        }
    };
    Ok(Some(kind))
}

fn resolved_setmax_request(
    sync: &ResolvedSynchronizationEffect,
) -> Option<(SetmaxnregResource, SetmaxnregAction, i64)> {
    let OwnedOperationEffect::SetmaxnregRegister(plan) = sync.details() else {
        return None;
    };
    let resource = sync
        .resources()
        .iter()
        .find_map(|resource| match resource.key() {
            ResolvedSyncResourceKey::Setmaxnreg {
                kernel_index,
                global_cta_id,
                warpgroup_id,
                ordinal,
            } => Some(SetmaxnregResource::new(
                kernel_index,
                global_cta_id,
                warpgroup_id,
                ordinal,
            )),
            _ => None,
        })?;
    Some((resource, plan.action(), plan.count()))
}

fn build_setmax_commands(
    snapshot: &FixedSyncLogSnapshot<'_>,
    operations: &[FixedSyncOperationSnapshot<'_>],
    commands: &mut Vec<FixedSyncCommand>,
    operation_commands: &mut HashMap<DynamicOpId, FixedSyncCommandId>,
) -> Result<BTreeMap<(usize, usize), SetmaxnregVerifierCore>, FixedSyncProgramBuildError> {
    let mut groups = BTreeMap::<
        SetmaxnregResource,
        Vec<(DynamicOpId, FixedSyncSetmaxParticipantSnapshot)>,
    >::new();
    for &(operation, summary, _, _) in operations {
        let Some(sync) = synchronization_summary(summary) else {
            continue;
        };
        let Some((request, action, count)) = resolved_setmax_request(sync) else {
            continue;
        };
        let participant = match snapshot.setmax_participants.get(operation) {
            Some(FixedSyncSnapshotRecord::Known(participant)) => participant.clone(),
            Some(FixedSyncSnapshotRecord::Ambiguous) => {
                return Err(FixedSyncProgramBuildError::AmbiguousEvidence {
                    operation: operation.clone(),
                    details: "setmax participant context is ambiguous".into(),
                });
            }
            None => {
                return Err(FixedSyncProgramBuildError::InvalidSetmaxRequest {
                    request,
                    details: format!("missing participant context for {operation}")
                        .into_boxed_str(),
                });
            }
        };
        if participant.request != request
            || participant.action != action
            || participant.count != count
            || participant.plan_global_warp_id != operation.global_warp_id()
            || participant.plan_global_cta_id != request.global_cta_id()
            || participant.plan_warp_id_in_cta / SETMAXNREG_WARPS_PER_GROUP
                != request.warpgroup_id()
            || participant.operation_mask != WarpMask::FULL
            || participant.plan_mask != WarpMask::FULL
        {
            return Err(FixedSyncProgramBuildError::InvalidSetmaxRequest {
                request,
                details: format!("participant metadata disagrees with {operation}")
                    .into_boxed_str(),
            });
        }
        groups
            .entry(request)
            .or_default()
            .push((operation.clone(), participant));
    }

    let mut pool_warpgroup_counts = BTreeMap::<(usize, usize), usize>::new();
    let mut pool_first_counts = BTreeMap::<(usize, usize, usize), i64>::new();
    let mut pool_ordinals = BTreeMap::<(usize, usize, usize), Vec<u64>>::new();
    for (request, participants) in &groups {
        if participants.len() != SETMAXNREG_WARPS_PER_GROUP {
            return Err(FixedSyncProgramBuildError::InvalidSetmaxRequest {
                request: *request,
                details: format!(
                    "expected {SETMAXNREG_WARPS_PER_GROUP} participant warps, got {}",
                    participants.len()
                )
                .into_boxed_str(),
            });
        }
        let resolution = match snapshot.setmax_resolutions.get(request) {
            Some(FixedSyncSnapshotRecord::Known(resolution)) => *resolution,
            Some(FixedSyncSnapshotRecord::Ambiguous) => {
                return Err(FixedSyncProgramBuildError::InvalidSetmaxRequest {
                    request: *request,
                    details: "request resolution is ambiguous".into(),
                });
            }
            None => {
                return Err(FixedSyncProgramBuildError::InvalidSetmaxRequest {
                    request: *request,
                    details: "request resolution is missing".into(),
                });
            }
        };
        let first = &participants[0].1;
        if resolution.request != *request
            || resolution.action != first.action
            || resolution.target_count != first.count
            || resolution.warpgroup_count == 0
        {
            return Err(FixedSyncProgramBuildError::InvalidSetmaxRequest {
                request: *request,
                details: "request resolution disagrees with participant command".into(),
            });
        }
        let expected_warps_per_cta = first.plan_warps_per_cta;
        if resolution.warpgroup_count != expected_warps_per_cta.div_ceil(SETMAXNREG_WARPS_PER_GROUP)
            || participants
                .iter()
                .any(|(_, participant)| participant.plan_warps_per_cta != expected_warps_per_cta)
        {
            return Err(FixedSyncProgramBuildError::InvalidSetmaxRequest {
                request: *request,
                details: "participant launch shape disagrees with request resolution".into(),
            });
        }
        let first_warp_in_cta = request
            .warpgroup_id()
            .checked_mul(SETMAXNREG_WARPS_PER_GROUP)
            .ok_or_else(|| FixedSyncProgramBuildError::InvalidSetmaxRequest {
                request: *request,
                details: "participant warp range overflow".into(),
            })?;
        let expected_global_base = request
            .global_cta_id()
            .checked_mul(expected_warps_per_cta)
            .and_then(|base| base.checked_add(first_warp_in_cta))
            .ok_or_else(|| FixedSyncProgramBuildError::InvalidSetmaxRequest {
                request: *request,
                details: "global participant warp range overflow".into(),
            })?;
        let expected = (expected_global_base..expected_global_base + SETMAXNREG_WARPS_PER_GROUP)
            .collect::<BTreeSet<_>>();
        let actual = participants
            .iter()
            .map(|(operation, _)| operation.global_warp_id())
            .collect::<BTreeSet<_>>();
        if actual != expected {
            return Err(FixedSyncProgramBuildError::InvalidSetmaxRequest {
                request: *request,
                details: format!("participant warps are {actual:?}, expected {expected:?}")
                    .into_boxed_str(),
            });
        }
        let pool = (request.kernel_index(), request.global_cta_id());
        match pool_warpgroup_counts.insert(pool, resolution.warpgroup_count) {
            Some(previous) if previous != resolution.warpgroup_count => {
                return Err(FixedSyncProgramBuildError::InvalidSetmaxRequest {
                    request: *request,
                    details: format!(
                        "pool warpgroup count changed from {previous} to {}",
                        resolution.warpgroup_count
                    )
                    .into_boxed_str(),
                });
            }
            _ => {}
        }
        if request.ordinal() == 0 {
            pool_first_counts.insert(
                (
                    request.kernel_index(),
                    request.global_cta_id(),
                    request.warpgroup_id(),
                ),
                resolution.current_count_before,
            );
        }
        pool_ordinals
            .entry((
                request.kernel_index(),
                request.global_cta_id(),
                request.warpgroup_id(),
            ))
            .or_default()
            .push(request.ordinal());
    }
    for ((kernel, cta, warpgroup), ordinals) in &mut pool_ordinals {
        ordinals.sort_unstable();
        for (expected, actual) in ordinals.iter().copied().enumerate() {
            if actual != expected as u64 {
                return Err(FixedSyncProgramBuildError::InvalidSetmaxRequest {
                    request: SetmaxnregResource::new(*kernel, *cta, *warpgroup, actual),
                    details: format!("non-contiguous request ordinal; expected {expected}")
                        .into_boxed_str(),
                });
            }
        }
    }

    for (request, participants) in groups {
        let command_id = FixedSyncCommandId(commands.len());
        let participant_warps = participants
            .iter()
            .map(|(operation, _)| operation.global_warp_id())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let first = &participants[0];
        commands.push(FixedSyncCommand {
            witness: first.0.clone(),
            participants: participant_warps.into(),
            kind: FixedSyncCommandKind::Setmax {
                request,
                action: first.1.action,
                target_count: first.1.count,
            },
            canonical_generation: None,
            initial_causal_clock: None,
            causal_clock: None,
        });
        for (operation, _) in participants {
            operation_commands.insert(operation, command_id);
        }
    }

    let mut pools = BTreeMap::new();
    for ((kernel, cta), warpgroup_count) in pool_warpgroup_counts {
        let default_count = setmaxnreg_default_register_count(warpgroup_count);
        let mut counts = vec![default_count; warpgroup_count];
        for (warpgroup, count) in counts.iter_mut().enumerate() {
            if let Some(first) = pool_first_counts.get(&(kernel, cta, warpgroup)) {
                *count = *first;
            }
        }
        let pool = SetmaxnregVerifierCore::from_parts(kernel, cta, 0, counts).map_err(|error| {
            FixedSyncProgramBuildError::InvalidInitialState {
                details: error.to_string().into_boxed_str(),
            }
        })?;
        pools.insert((kernel, cta), pool);
    }
    Ok(pools)
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TcgenCollectiveKey {
    kernel_index: usize,
    action: TcgenLifecycleAction,
    static_op_id: u64,
    loop_iteration_path: Box<[i64]>,
    address: u32,
    columns: usize,
    cta_group: usize,
    participant_ctas: Box<[usize]>,
    participant_warps: Box<[usize]>,
    exclusive: bool,
    capacity: usize,
}

fn build_tcgen_commands(
    snapshot: &FixedSyncLogSnapshot<'_>,
    operations: &[FixedSyncOperationSnapshot<'_>],
    commands: &mut Vec<FixedSyncCommand>,
    operation_commands: &mut HashMap<DynamicOpId, FixedSyncCommandId>,
) -> Result<BTreeMap<(usize, usize), FixedSyncTcgenCtaState>, FixedSyncProgramBuildError> {
    let mut groups = BTreeMap::<TcgenCollectiveKey, Vec<DynamicOpId>>::new();
    for &(operation, summary, _, _) in operations {
        let Some(sync) = synchronization_summary(summary) else {
            continue;
        };
        let OwnedOperationEffect::TcgenLifecycleRegister(plan) = sync.details() else {
            continue;
        };
        groups
            .entry(TcgenCollectiveKey {
                kernel_index: operation.kernel_index(),
                action: plan.action(),
                static_op_id: plan.static_op_id(),
                loop_iteration_path: plan.loop_iteration_path().into(),
                address: plan.address(),
                columns: plan.columns(),
                cta_group: plan.cta_group(),
                participant_ctas: plan.participant_ctas().into(),
                participant_warps: plan.participant_warps().into(),
                exclusive: plan.exclusive(),
                capacity: plan.capacity(),
            })
            .or_default()
            .push(operation.clone());
    }

    let mut ctas = BTreeMap::new();
    for (key, operations) in groups {
        let witness = operations[0].clone();
        let actual = operations
            .iter()
            .map(DynamicOpId::global_warp_id)
            .collect::<BTreeSet<_>>();
        let expected = key
            .participant_warps
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if operations.len() != key.participant_warps.len() || actual != expected {
            return Err(FixedSyncProgramBuildError::InvalidTcgenCollective {
                operation: witness,
                details: format!("participant warps are {actual:?}, expected {expected:?}")
                    .into_boxed_str(),
            });
        }
        if key.participant_ctas.len() != key.cta_group
            || key.participant_warps.len() != key.cta_group
            || !matches!(key.cta_group, 1 | 2)
        {
            return Err(FixedSyncProgramBuildError::InvalidTcgenCollective {
                operation: witness,
                details: "TCGEN CTA-group participant contract is invalid".into(),
            });
        }
        let mut allocation = None;
        for operation in &operations {
            let resolved = match snapshot.tcgen_allocations.get(operation) {
                Some(FixedSyncSnapshotRecord::Known(allocation)) => *allocation,
                Some(FixedSyncSnapshotRecord::Ambiguous) => {
                    return Err(FixedSyncProgramBuildError::AmbiguousEvidence {
                        operation: operation.clone(),
                        details: "TCGEN lifecycle result is ambiguous".into(),
                    });
                }
                None => {
                    return Err(FixedSyncProgramBuildError::InvalidTcgenCollective {
                        operation: operation.clone(),
                        details: "TCGEN lifecycle resume result is missing".into(),
                    });
                }
            };
            let resolved = resolved.map(|(base_column, columns)| FixedTcgenAllocation {
                base_column,
                columns,
            });
            match allocation {
                Some(previous) if previous != resolved => {
                    return Err(FixedSyncProgramBuildError::InvalidTcgenCollective {
                        operation: operation.clone(),
                        details: "TCGEN collective participants observed different results".into(),
                    });
                }
                None => allocation = Some(resolved),
                Some(_) => {}
            }
        }
        let canonical_allocation = allocation.flatten();
        if key.action == TcgenLifecycleAction::Allocate && canonical_allocation.is_none() {
            return Err(FixedSyncProgramBuildError::InvalidTcgenCollective {
                operation: witness,
                details: "TCGEN allocate has no canonical allocation result".into(),
            });
        }
        if key.action != TcgenLifecycleAction::Allocate && canonical_allocation.is_some() {
            return Err(FixedSyncProgramBuildError::InvalidTcgenCollective {
                operation: witness,
                details: "non-allocate TCGEN lifecycle returned an allocation".into(),
            });
        }
        for &cta in key.participant_ctas.iter() {
            ctas.entry((key.kernel_index, cta)).or_default();
        }
        let command_id = FixedSyncCommandId(commands.len());
        commands.push(FixedSyncCommand {
            witness: operations[0].clone(),
            participants: key.participant_warps.clone().into(),
            kind: FixedSyncCommandKind::TcgenLifecycle(FixedSyncTcgenRequest {
                kernel_index: key.kernel_index,
                action: key.action,
                address: key.address,
                columns: key.columns,
                cta_group: key.cta_group,
                participant_ctas: key.participant_ctas,
                participant_warps: key.participant_warps,
                exclusive: key.exclusive,
                capacity: key.capacity,
                canonical_allocation,
            }),
            canonical_generation: None,
            initial_causal_clock: None,
            causal_clock: None,
        });
        for operation in operations {
            operation_commands.insert(operation, command_id);
        }
    }
    Ok(ctas)
}

impl FixedSyncProgram {
    pub fn command_witness(&self, command: FixedSyncCommandId) -> Option<&DynamicOpId> {
        self.commands
            .get(command.get())
            .map(|command| &command.witness)
    }

    pub fn first_operation(&self) -> Option<&DynamicOpId> {
        self.commands.first().map(|command| &command.witness)
    }

    pub(crate) fn transition_operation(
        &self,
        transition: &FixedSyncTransition,
    ) -> Option<DynamicOpId> {
        match transition {
            FixedSyncTransition::Issue(command) => self.command_witness(*command).cloned(),
            FixedSyncTransition::Complete(completion) => Some(completion.issuer().clone()),
            FixedSyncTransition::SetmaxGrant(request) => {
                self.command_witness_for_setmax(*request).cloned()
            }
            FixedSyncTransition::ValidateExit => self.first_operation().cloned(),
        }
    }

    pub(crate) fn transition_description(&self, transition: &FixedSyncTransition) -> Box<str> {
        match transition {
            FixedSyncTransition::Issue(command_id) => self
                .commands
                .get(command_id.get())
                .map_or_else(
                    || format!("unknown command {command_id:?}"),
                    |command| format!("{:?}", command.kind),
                )
                .into_boxed_str(),
            FixedSyncTransition::Complete(completion) => {
                format!("mbarrier completion ordinal {}", completion.ordinal()).into_boxed_str()
            }
            FixedSyncTransition::SetmaxGrant(request) => {
                format!("setmaxnreg grant {request:?}").into_boxed_str()
            }
            FixedSyncTransition::ValidateExit => "validate protocol quiescence".into(),
        }
    }

    pub fn command_witness_for_setmax(&self, request: SetmaxnregResource) -> Option<&DynamicOpId> {
        self.setmax_commands
            .get(&request)
            .and_then(|command| self.command_witness(*command))
    }

    fn head_command(&self, state: &FixedSyncState, warp_id: usize) -> Option<FixedSyncCommandId> {
        let warp_index = *self.warp_indices.get(&warp_id)?;
        self.warp_programs[warp_index]
            .get(state.warp_cursors[warp_index])
            .copied()
    }

    fn all_warps_finished(&self, state: &FixedSyncState) -> bool {
        state
            .warp_cursors
            .iter()
            .zip(self.warp_programs.iter())
            .all(|(cursor, program)| *cursor == program.len())
    }

    fn command_completed(&self, state: &FixedSyncState, command_id: FixedSyncCommandId) -> bool {
        self.commands.get(command_id.get()).is_some_and(|command| {
            command.participants.iter().copied().all(|warp_id| {
                let Some(warp_index) = self.warp_indices.get(&warp_id).copied() else {
                    return false;
                };
                let Some(position) = self
                    .warp_command_positions
                    .get(&(warp_id, command_id))
                    .copied()
                else {
                    return false;
                };
                state.warp_cursors[warp_index] > position
            })
        })
    }

    fn command_ready(&self, state: &FixedSyncState, command_id: FixedSyncCommandId) -> bool {
        let Some(command) = self.commands.get(command_id.get()) else {
            return false;
        };
        if !self.causal_predecessors[command_id.get()]
            .iter()
            .copied()
            .all(|predecessor| self.command_completed(state, predecessor))
        {
            return false;
        }
        command.participants.iter().copied().all(|warp_id| {
            !state.blocked_warps.contains_key(&warp_id)
                && self.head_command(state, warp_id) == Some(command_id)
        })
    }

    fn issue_ready(&self, state: &FixedSyncState, command_id: FixedSyncCommandId) -> bool {
        if !self.command_ready(state, command_id) {
            return false;
        }
        match &self.commands[command_id.get()].kind {
            FixedSyncCommandKind::MbarrierWait {
                barrier_id,
                conditional: true,
                ..
            } => Self::conditional_acquire_ready(
                state,
                *barrier_id,
                self.commands[command_id.get()].canonical_generation,
            ),
            FixedSyncCommandKind::MbarrierWaitBatch {
                waits,
                conditional: true,
            } => waits.iter().all(|&(barrier, _, generation)| {
                Self::conditional_acquire_ready(state, barrier, generation)
            }),
            FixedSyncCommandKind::Setmax { request, .. } => state
                .setmax_pools
                .get(&(request.kernel_index(), request.global_cta_id()))
                .is_some_and(|pool| !pool.warpgroup_has_pending_increase(request.warpgroup_id())),
            FixedSyncCommandKind::TcgenLifecycle(request)
                if request.action == TcgenLifecycleAction::Deallocate =>
            {
                self.tcgen_deallocation_ready(state, request)
            }
            _ => true,
        }
    }

    /// A recorded successful query acquires its observed completion. Schedules
    /// before that completion would take a different (false) query path, which
    /// is outside this fixed execution. Still expose invalid initialization to
    /// the strict protocol instead of silently treating it as a blocked wait.
    fn conditional_acquire_ready(
        state: &FixedSyncState,
        barrier: PhysicalBarrierId,
        generation: Option<u64>,
    ) -> bool {
        let snapshot = state.mbarriers.snapshot(barrier);
        snapshot.generation().is_none() || generation <= snapshot.last_completed_generation()
    }

    fn conditional_wait_successor(
        &self,
        barrier: PhysicalBarrierId,
        generation: Option<u64>,
        requested_phase: u64,
        initialization: &DynamicOpId,
        witness: &DynamicOpId,
    ) -> Result<Option<u64>, FixedSyncProgramError> {
        let completed = self
            .conditional_mbarrier_completions
            .get(&barrier)
            .and_then(|lifetimes| lifetimes.get(initialization));
        let phase = match generation {
            None => 1,
            Some(generation) => completed
                .and_then(|completed| completed.iter().position(|&candidate| candidate == generation))
                .map(|ordinal| ordinal as u64 & 1)
                .ok_or_else(|| mbarrier_incomplete(Some(witness), format!(
                    "mbarrier {barrier:?} has no recorded conditional completion at generation {generation}"
                )))?,
        };
        if requested_phase != phase {
            return Err(mbarrier_protocol(witness, format!(
                "mbarrier {barrier:?} conditional wait requests phase {requested_phase}, but its completion has conditional phase {phase}"
            )));
        }
        Ok(completed.and_then(|completed| {
            completed
                .iter()
                .copied()
                .find(|&candidate| Some(candidate) > generation)
        }))
    }

    fn validate_conditional_acquire(
        &self,
        state: &FixedSyncState,
        barrier: PhysicalBarrierId,
        generation: Option<u64>,
        requested_phase: u64,
        witness: &DynamicOpId,
        issue_clock: Option<&crate::SyncVectorClock>,
    ) -> Result<(), FixedSyncProgramError> {
        let snapshot = state.mbarriers.snapshot(barrier);
        let Some(initialization) = snapshot.init_witness() else {
            // Leave uninitialized-barrier diagnosis to the strict acquire.
            return Ok(());
        };
        if self
            .conditional_mbarrier_completions
            .get(&barrier)
            .is_some_and(|lifetimes| lifetimes.len() > 1)
        {
            let inits = self
                .commands
                .iter()
                .filter(|command| {
                    matches!(&command.kind, FixedSyncCommandKind::MbarrierInit { barrier_ids, .. }
                    if barrier_ids.contains(&barrier))
                })
                .collect::<Vec<_>>();
            let recorded_init =
                latest_causal_mbarrier_init(issue_clock, &inits).ok_or_else(|| {
                    mbarrier_incomplete(
                        Some(witness),
                        "conditional wait has no unique recorded initialization",
                    )
                })?;
            if &recorded_init.witness != initialization {
                return Err(mbarrier_protocol(
                    witness,
                    "conditional wait was overtaken by mbarrier reinitialization",
                ));
            }
        }
        let successor = self.conditional_wait_successor(
            barrier,
            generation,
            requested_phase,
            initialization,
            witness,
        )?;
        if successor
            .is_some_and(|successor| Some(successor) <= snapshot.last_completed_generation())
        {
            return Err(mbarrier_protocol(witness, format!(
                "mbarrier {barrier:?} conditional wait for generation {generation:?} was overtaken by conditional completion {successor:?}"
            )));
        }
        Ok(())
    }

    fn tcgen_deallocation_ready(
        &self,
        state: &FixedSyncState,
        request: &FixedSyncTcgenRequest,
    ) -> bool {
        let allocation = FixedTcgenAllocation {
            base_column: request.address,
            columns: request.columns,
        };
        if request.participant_ctas.iter().all(|cta| {
            state
                .tcgen_ctas
                .get(&(request.kernel_index, *cta))
                .is_some_and(|snapshot| snapshot.allocations.contains(&allocation))
        }) {
            return true;
        }

        // An allocation/deallocation pair has an implicit resource dependency
        // even when the synchronization-resource projection contains no
        // explicit cross-warp causal edge between its issuing warps.  Delay the
        // deallocation while that matching allocation can still run.  If no
        // such allocation exists (or it already ran and the interval is still
        // absent), leave the command enabled so issue_tcgen reports the real
        // malformed lifecycle instead of turning it into a deadlock.
        !self.commands.iter().enumerate().any(|(index, command)| {
            let FixedSyncCommandKind::TcgenLifecycle(candidate) = &command.kind else {
                return false;
            };
            candidate.action == TcgenLifecycleAction::Allocate
                && candidate.kernel_index == request.kernel_index
                && candidate.participant_ctas == request.participant_ctas
                && candidate.canonical_allocation == Some(allocation)
                && !self.command_completed(state, FixedSyncCommandId(index))
        })
    }

    fn internal_error(
        &self,
        command_id: FixedSyncCommandId,
        details: impl Into<Box<str>>,
    ) -> FixedSyncProgramError {
        let operation = self.commands[command_id.get()].witness.clone();
        FixedSyncProgramError::Protocol {
            kind: FixedSyncProtocolKind::Internal,
            operation,
            details: details.into(),
        }
    }

    fn protocol_error(
        &self,
        command_id: FixedSyncCommandId,
        kind: FixedSyncProtocolKind,
        error: impl fmt::Display,
    ) -> FixedSyncProgramError {
        FixedSyncProgramError::Protocol {
            kind,
            operation: self.commands[command_id.get()].witness.clone(),
            details: error.to_string().into_boxed_str(),
        }
    }

    fn incomplete_error(
        &self,
        command_id: Option<FixedSyncCommandId>,
        kind: FixedSyncProtocolKind,
        details: impl Into<Box<str>>,
    ) -> FixedSyncProgramError {
        FixedSyncProgramError::Incomplete {
            kind,
            operation: command_id.map(|id| self.commands[id.get()].witness.clone()),
            details: details.into(),
        }
    }

    fn advance_warp(
        &self,
        state: &mut FixedSyncState,
        warp_id: usize,
        command_id: FixedSyncCommandId,
    ) -> Result<(), FixedSyncProgramError> {
        let Some(warp_index) = self.warp_indices.get(&warp_id).copied() else {
            return Err(self.internal_error(
                command_id,
                format!("warp {warp_id} has no fixed-program cursor"),
            ));
        };
        if self.warp_programs[warp_index].get(state.warp_cursors[warp_index]) != Some(&command_id) {
            return Err(self.internal_error(
                command_id,
                format!("warp {warp_id} cursor does not point at {command_id:?}"),
            ));
        }
        state.warp_cursors[warp_index] += 1;
        Ok(())
    }

    fn advance_command(
        &self,
        state: &mut FixedSyncState,
        command_id: FixedSyncCommandId,
    ) -> Result<(), FixedSyncProgramError> {
        for warp_id in self.commands[command_id.get()].participants.iter().copied() {
            self.advance_warp(state, warp_id, command_id)?;
        }
        Ok(())
    }

    fn block_warp(
        &self,
        state: &mut FixedSyncState,
        warp_id: usize,
        command_id: FixedSyncCommandId,
    ) -> Result<(), FixedSyncProgramError> {
        if self.head_command(state, warp_id) != Some(command_id) {
            return Err(self.internal_error(
                command_id,
                format!("warp {warp_id} cannot block away from its command head"),
            ));
        }
        if let Some(previous) = state.blocked_warps.insert(warp_id, command_id) {
            return Err(self.internal_error(
                command_id,
                format!("warp {warp_id} was already blocked on {previous:?}"),
            ));
        }
        Ok(())
    }

    fn wake_blocked_warp(
        &self,
        state: &mut FixedSyncState,
        warp_id: usize,
        command_id: FixedSyncCommandId,
    ) -> Result<(), FixedSyncProgramError> {
        match state.blocked_warps.remove(&warp_id) {
            Some(actual) if actual == command_id => self.advance_warp(state, warp_id, command_id),
            Some(actual) => Err(self.internal_error(
                command_id,
                format!("warp {warp_id} was blocked on {actual:?}, not {command_id:?}"),
            )),
            None => Err(self.internal_error(
                command_id,
                format!("warp {warp_id} became ready without a blocked command"),
            )),
        }
    }

    fn wake_mbarrier_waiters(
        &self,
        state: &mut FixedSyncState,
        waiters: &[crate::StrictMbarrierWaiter],
        release_command: FixedSyncCommandId,
    ) -> Result<(), FixedSyncProgramError> {
        for waiter in waiters {
            let warp_id = waiter.warp_id();
            let Some(command_id) = state.blocked_warps.get(&warp_id).copied() else {
                return Err(self.internal_error(
                    release_command,
                    format!("mbarrier made unregistered warp {warp_id} ready"),
                ));
            };
            if !matches!(
                self.commands[command_id.get()].kind,
                FixedSyncCommandKind::MbarrierWait { .. }
                    | FixedSyncCommandKind::MbarrierWaitBatch { .. }
            ) {
                return Err(self.internal_error(
                    release_command,
                    format!("mbarrier waiter warp {warp_id} is blocked on a different protocol"),
                ));
            }
            if let FixedSyncCommandKind::MbarrierWaitBatch {
                waits,
                conditional: false,
            } = &self.commands[command_id.get()].kind
            {
                let all_ready = waits.iter().all(|&(barrier_id, requested_phase, _)| {
                    state.mbarriers.snapshot(barrier_id).last_completed_phase() == requested_phase
                });
                if !all_ready {
                    continue;
                }
            }
            self.wake_blocked_warp(state, warp_id, command_id)?;
        }
        Ok(())
    }

    fn wake_named_waiters(
        &self,
        state: &mut FixedSyncState,
        barrier_id: NamedBarrierId,
        waiters: &[crate::StrictNamedBarrierWaiter],
        release_command: FixedSyncCommandId,
    ) -> Result<(), FixedSyncProgramError> {
        for waiter in waiters {
            state
                .named_barriers
                .resume(
                    barrier_id,
                    waiter.warp_id(),
                    waiter.arrival_mask(),
                    waiter.generation(),
                    Some(self.commands[release_command.get()].witness.clone()),
                )
                .map_err(|error| {
                    self.protocol_error(release_command, FixedSyncProtocolKind::NamedBarrier, error)
                })?;
            let warp_id = waiter.warp_id();
            let Some(command_id) = state.blocked_warps.get(&warp_id).copied() else {
                return Err(self.internal_error(
                    release_command,
                    format!("named barrier made unregistered warp {warp_id} ready"),
                ));
            };
            if !matches!(
                self.commands[command_id.get()].kind,
                FixedSyncCommandKind::NamedBarrierSync { .. }
            ) {
                return Err(self.internal_error(
                    release_command,
                    format!("named-barrier waiter warp {warp_id} is blocked on another protocol"),
                ));
            }
            self.wake_blocked_warp(state, warp_id, command_id)?;
        }
        Ok(())
    }

    fn wake_cluster_waiters(
        &self,
        state: &mut FixedSyncState,
        barrier_id: ClusterBarrierId,
        generation: u64,
        release_command: FixedSyncCommandId,
    ) -> Result<(), FixedSyncProgramError> {
        let ready = state
            .cluster_waits
            .iter()
            .filter_map(|(&(barrier, waiter_generation, warp_id), &command_id)| {
                (barrier == barrier_id && waiter_generation == generation)
                    .then_some((warp_id, command_id))
            })
            .collect::<Vec<_>>();
        for (warp_id, command_id) in ready {
            state
                .cluster_barriers
                .resume(
                    barrier_id,
                    generation,
                    warp_id,
                    Some(self.commands[release_command.get()].witness.clone()),
                )
                .map_err(|error| {
                    self.protocol_error(
                        release_command,
                        FixedSyncProtocolKind::ClusterBarrier,
                        error,
                    )
                })?;
            state
                .cluster_waits
                .remove(&(barrier_id, generation, warp_id));
            self.wake_blocked_warp(state, warp_id, command_id)?;
        }
        Ok(())
    }

    fn issue_command(
        &self,
        state: &FixedSyncState,
        command_id: FixedSyncCommandId,
    ) -> Result<FixedSyncState, FixedSyncProgramError> {
        if state.exit_validated || !self.issue_ready(state, command_id) {
            return Err(FixedSyncProgramError::DisabledTransition {
                transition: FixedSyncTransition::Issue(command_id),
            });
        }
        let command = self.commands[command_id.get()].clone();
        let mut next = state.clone();
        match command.kind {
            FixedSyncCommandKind::MbarrierInvalidate { barrier_ids } => {
                next.mbarriers
                    .invalidate_many(&barrier_ids, Some(command.witness.clone()))
                    .map_err(|error| {
                        self.protocol_error(command_id, FixedSyncProtocolKind::Mbarrier, error)
                    })?;
                self.advance_command(&mut next, command_id)?;
            }
            FixedSyncCommandKind::MbarrierInit {
                barrier_ids,
                expected_arrivals,
            } => {
                next.mbarriers
                    .init_many(
                        &barrier_ids,
                        expected_arrivals,
                        Some(command.witness.clone()),
                    )
                    .map_err(|error| {
                        self.protocol_error(command_id, FixedSyncProtocolKind::Mbarrier, error)
                    })?;
                self.advance_command(&mut next, command_id)?;
            }
            FixedSyncCommandKind::MbarrierInitFence { barrier_ids } => {
                next.mbarriers
                    .mark_init_fenced_many(&barrier_ids, Some(command.witness.clone()));
                self.advance_command(&mut next, command_id)?;
            }
            FixedSyncCommandKind::MbarrierArrive { arrivals } => {
                for (barrier_id, arrival_count, expected_transactions, drop) in
                    arrivals.iter().copied()
                {
                    let effect = next
                        .mbarriers
                        .arrive_with_drop(
                            barrier_id,
                            command.witness.global_warp_id(),
                            arrival_count,
                            expected_transactions,
                            drop,
                            Some(command.witness.clone()),
                        )
                        .map_err(|error| {
                            self.protocol_error(command_id, FixedSyncProtocolKind::Mbarrier, error)
                        })?;
                    self.wake_mbarrier_waiters(&mut next, effect.ready_waiters(), command_id)?;
                }
                self.advance_command(&mut next, command_id)?;
            }
            FixedSyncCommandKind::MbarrierExpectTx { expectations } => {
                for (barrier_id, transactions) in expectations.iter().copied() {
                    next.mbarriers
                        .expect_tx(barrier_id, transactions, Some(command.witness.clone()))
                        .map_err(|error| {
                            self.protocol_error(command_id, FixedSyncProtocolKind::Mbarrier, error)
                        })?;
                }
                self.advance_command(&mut next, command_id)?;
            }
            FixedSyncCommandKind::MbarrierWait {
                barrier_id,
                requested_phase,
                conditional,
            } => match (if conditional {
                self.validate_conditional_acquire(
                    &next,
                    barrier_id,
                    command.canonical_generation,
                    requested_phase,
                    &command.witness,
                    command.initial_causal_clock.as_ref(),
                )?;
                next.mbarriers.acquire_completed(
                    barrier_id,
                    command.canonical_generation,
                    Some(command.witness.clone()),
                )
            } else {
                next.mbarriers.wait(
                    barrier_id,
                    requested_phase,
                    command.witness.global_warp_id(),
                    Some(command.witness.clone()),
                )
            })
            .map_err(|error| {
                self.protocol_error(command_id, FixedSyncProtocolKind::Mbarrier, error)
            })? {
                StrictMbarrierWaitOutcome::Ready { .. } => {
                    self.advance_command(&mut next, command_id)?;
                }
                StrictMbarrierWaitOutcome::Registered { .. } => {
                    self.block_warp(&mut next, command.witness.global_warp_id(), command_id)?;
                }
            },
            FixedSyncCommandKind::MbarrierWaitBatch { waits, conditional } => {
                let mut pending = false;
                for (barrier_id, requested_phase, generation) in waits.iter().copied() {
                    match (if conditional {
                        self.validate_conditional_acquire(
                            &next,
                            barrier_id,
                            generation,
                            requested_phase,
                            &command.witness,
                            command.initial_causal_clock.as_ref(),
                        )?;
                        next.mbarriers.acquire_completed(
                            barrier_id,
                            generation,
                            Some(command.witness.clone()),
                        )
                    } else {
                        next.mbarriers.wait(
                            barrier_id,
                            requested_phase,
                            command.witness.global_warp_id(),
                            Some(command.witness.clone()),
                        )
                    })
                    .map_err(|error| {
                        self.protocol_error(command_id, FixedSyncProtocolKind::Mbarrier, error)
                    })? {
                        StrictMbarrierWaitOutcome::Ready { .. } => {}
                        StrictMbarrierWaitOutcome::Registered { .. } => pending = true,
                    }
                }
                if pending {
                    self.block_warp(&mut next, command.witness.global_warp_id(), command_id)?;
                } else {
                    self.advance_command(&mut next, command_id)?;
                }
            }
            FixedSyncCommandKind::MbarrierCompletionIssue { completions } => {
                for (ordinal, completion_spec) in completions.iter().copied().enumerate() {
                    // The pending-count raise is part of issuing the deferred
                    // arrival, so it precedes the generation capture exactly as
                    // it does on the numeric path.
                    if let FixedSyncMbarrierCompletionKind::Arrival {
                        pending_increase, ..
                    } = completion_spec.kind
                    {
                        next.mbarriers
                            .increase_pending_arrivals(
                                completion_spec.barrier_id,
                                pending_increase,
                                Some(command.witness.clone()),
                            )
                            .map_err(|error| {
                                self.protocol_error(
                                    command_id,
                                    FixedSyncProtocolKind::Mbarrier,
                                    error,
                                )
                            })?;
                    }
                    let token = next
                        .mbarriers
                        .capture_completion(
                            completion_spec.barrier_id,
                            Some(command.witness.clone()),
                        )
                        .map_err(|error| {
                            self.protocol_error(command_id, FixedSyncProtocolKind::Mbarrier, error)
                        })?;
                    let completion_id = FixedSyncCompletionId {
                        issuer: command.witness.clone(),
                        ordinal,
                    };
                    let completion = PendingMbarrierCompletion {
                        barrier_id: completion_spec.barrier_id,
                        generation: token.generation(),
                        kind: completion_spec.kind,
                        token,
                    };
                    if next
                        .pending_completions
                        .insert(completion_id, completion)
                        .is_some()
                    {
                        return Err(self.internal_error(
                            command_id,
                            "duplicate stable mbarrier completion identity",
                        ));
                    }
                }
                self.advance_command(&mut next, command_id)?;
            }
            FixedSyncCommandKind::NamedBarrierArrive {
                barrier_id,
                expected_arrivals,
                warp_id,
                arrival_mask,
            } => {
                let outcome = next
                    .named_barriers
                    .arrive(
                        barrier_id,
                        expected_arrivals,
                        warp_id,
                        arrival_mask,
                        Some(command.witness.clone()),
                    )
                    .map_err(|error| {
                        self.protocol_error(command_id, FixedSyncProtocolKind::NamedBarrier, error)
                    })?;
                self.advance_command(&mut next, command_id)?;
                let StrictNamedBarrierOutcome::Arrived { ready_waiters, .. } = outcome else {
                    return Err(self.internal_error(
                        command_id,
                        "named-barrier arrive returned a non-arrive outcome",
                    ));
                };
                self.wake_named_waiters(&mut next, barrier_id, &ready_waiters, command_id)?;
            }
            FixedSyncCommandKind::NamedBarrierSync {
                barrier_id,
                expected_arrivals,
                warp_id,
                arrival_mask,
                aligned,
            } => {
                let outcome = next
                    .named_barriers
                    .sync_with_alignment(
                        barrier_id,
                        expected_arrivals,
                        warp_id,
                        arrival_mask,
                        aligned,
                        Some(command.witness.clone()),
                    )
                    .map_err(|error| {
                        self.protocol_error(command_id, FixedSyncProtocolKind::NamedBarrier, error)
                    })?;
                self.block_warp(&mut next, warp_id, command_id)?;
                match outcome {
                    StrictNamedBarrierOutcome::Registered { .. } => {}
                    StrictNamedBarrierOutcome::Ready { ready_waiters, .. } => {
                        self.wake_named_waiters(&mut next, barrier_id, &ready_waiters, command_id)?;
                    }
                    StrictNamedBarrierOutcome::Arrived { .. } => {
                        return Err(self.internal_error(
                            command_id,
                            "named-barrier sync returned an arrive outcome",
                        ));
                    }
                }
            }
            FixedSyncCommandKind::ClusterBarrierArrive {
                barrier_id,
                warp_id,
                arrival_mask,
                participant_warps,
            } => {
                let outcome = next
                    .cluster_barriers
                    .arrive(
                        barrier_id,
                        &participant_warps,
                        warp_id,
                        arrival_mask,
                        Some(command.witness.clone()),
                    )
                    .map_err(|error| {
                        self.protocol_error(
                            command_id,
                            FixedSyncProtocolKind::ClusterBarrier,
                            error,
                        )
                    })?;
                if outcome.rearrival_without_wait() {
                    return Err(self.incomplete_error(
                        Some(command_id),
                        FixedSyncProtocolKind::ClusterBarrier,
                        "cluster generation reuse without a modeled wait requires exit-aware semantics",
                    ));
                }
                self.advance_command(&mut next, command_id)?;
                if outcome.completed_now() {
                    self.wake_cluster_waiters(
                        &mut next,
                        barrier_id,
                        outcome.generation(),
                        command_id,
                    )?;
                }
            }
            FixedSyncCommandKind::ClusterBarrierWait {
                barrier_id,
                warp_id,
                arrival_mask,
                participant_warps,
            } => {
                let outcome = next
                    .cluster_barriers
                    .wait_register(
                        barrier_id,
                        &participant_warps,
                        warp_id,
                        arrival_mask,
                        Some(command.witness.clone()),
                    )
                    .map_err(|error| {
                        self.protocol_error(
                            command_id,
                            FixedSyncProtocolKind::ClusterBarrier,
                            error,
                        )
                    })?;
                self.block_warp(&mut next, warp_id, command_id)?;
                next.cluster_waits
                    .insert((barrier_id, outcome.generation(), warp_id), command_id);
                if outcome.completed_now() {
                    self.wake_cluster_waiters(
                        &mut next,
                        barrier_id,
                        outcome.generation(),
                        command_id,
                    )?;
                }
            }
            FixedSyncCommandKind::Setmax {
                request,
                action,
                target_count,
            } => {
                let Some(pool) = next
                    .setmax_pools
                    .get_mut(&(request.kernel_index(), request.global_cta_id()))
                else {
                    return Err(self.internal_error(command_id, "setmax pool is missing"));
                };
                let outcome =
                    pool.apply_request(request, action, target_count)
                        .map_err(|error| {
                            self.protocol_error(
                                command_id,
                                FixedSyncProtocolKind::Setmaxnreg,
                                error,
                            )
                        })?;
                match outcome.disposition() {
                    SetmaxnregVerifierRequestDisposition::IncreasePending { .. } => {}
                    SetmaxnregVerifierRequestDisposition::DecreaseApplied { .. }
                    | SetmaxnregVerifierRequestDisposition::IncreaseImmediate { .. } => {
                        self.advance_command(&mut next, command_id)?;
                    }
                }
            }
            FixedSyncCommandKind::TcgenLifecycle(request) => {
                self.issue_tcgen(&mut next, command_id, &request)?;
            }
        }
        Ok(next)
    }

    fn issue_tcgen(
        &self,
        state: &mut FixedSyncState,
        command_id: FixedSyncCommandId,
        request: &FixedSyncTcgenRequest,
    ) -> Result<(), FixedSyncProgramError> {
        if !matches!(request.cta_group, 1 | 2)
            || request.participant_ctas.len() != request.cta_group
            || request.participant_warps.len() != request.cta_group
        {
            return Err(self.protocol_error(
                command_id,
                FixedSyncProtocolKind::TcgenLifecycle,
                "invalid TCGEN CTA-group participant contract",
            ));
        }
        if matches!(
            request.action,
            TcgenLifecycleAction::Allocate | TcgenLifecycleAction::Deallocate
        ) && !crate::tcgen::valid_columns(request.columns, request.exclusive, request.capacity)
        {
            return Err(self.protocol_error(
                command_id,
                FixedSyncProtocolKind::TcgenLifecycle,
                format!("invalid TCGEN allocation width {}", request.columns),
            ));
        }
        let cta_keys = request
            .participant_ctas
            .iter()
            .copied()
            .map(|cta| (request.kernel_index, cta))
            .collect::<Vec<_>>();
        for &key in &cta_keys {
            let Some(snapshot) = state.tcgen_ctas.get(&key) else {
                return Err(
                    self.internal_error(command_id, format!("TCGEN CTA {key:?} is missing"))
                );
            };
            if let Some(previous) = snapshot.cta_group {
                if previous != request.cta_group {
                    return Err(self.protocol_error(
                        command_id,
                        FixedSyncProtocolKind::TcgenLifecycle,
                        format!(
                            "TCGEN CTA {} changed CTA group from {previous} to {}",
                            key.1, request.cta_group
                        ),
                    ));
                }
            }
            match request.action {
                TcgenLifecycleAction::Allocate => {
                    if snapshot.relinquished {
                        return Err(self.protocol_error(
                            command_id,
                            FixedSyncProtocolKind::TcgenLifecycle,
                            format!("TCGEN CTA {} allocated after relinquish", key.1),
                        ));
                    }
                    if let Some(previous) = snapshot.last_allocation_columns {
                        if request.columns > previous {
                            return Err(self.protocol_error(
                                command_id,
                                FixedSyncProtocolKind::TcgenLifecycle,
                                format!(
                                    "TCGEN CTA {} increased allocation size from {previous} to {}",
                                    key.1, request.columns
                                ),
                            ));
                        }
                    }
                }
                TcgenLifecycleAction::Deallocate => {
                    if !snapshot.allocations.iter().any(|allocation| {
                        allocation.base_column() == request.address
                            && allocation.columns() == request.columns
                    }) {
                        return Err(self.protocol_error(
                            command_id,
                            FixedSyncProtocolKind::TcgenLifecycle,
                            format!(
                                "TCGEN CTA {} deallocated absent interval [{}, {})",
                                key.1,
                                request.address,
                                request.address as usize + request.columns
                            ),
                        ));
                    }
                }
                TcgenLifecycleAction::Relinquish => {}
            }
        }

        let allocation = if request.action == TcgenLifecycleAction::Allocate {
            let actual = allocate_common_tcgen_interval(state, request).ok_or_else(|| {
                self.protocol_error(
                    command_id,
                    FixedSyncProtocolKind::TcgenLifecycle,
                    format!(
                        "no common TCGEN interval is available for {} columns",
                        request.columns
                    ),
                )
            })?;
            let Some(expected) = request.canonical_allocation else {
                return Err(self.internal_error(
                    command_id,
                    "TCGEN allocation lacks a canonical native result",
                ));
            };
            if actual != expected {
                return Err(self.protocol_error(
                    command_id,
                    FixedSyncProtocolKind::TcgenLifecycle,
                    format!(
                        "TCGEN allocation result changed from {expected:?} to {actual:?} under this order"
                    ),
                ));
            }
            Some(actual)
        } else {
            if request.canonical_allocation.is_some() {
                return Err(self.internal_error(
                    command_id,
                    "non-allocate TCGEN request has a canonical allocation result",
                ));
            }
            None
        };

        for key in cta_keys {
            let snapshot = state
                .tcgen_ctas
                .get_mut(&key)
                .expect("validated TCGEN CTA key remains present");
            snapshot.cta_group = Some(request.cta_group);
            match request.action {
                TcgenLifecycleAction::Allocate => {
                    snapshot
                        .allocations
                        .push(allocation.expect("allocate action produced an allocation"));
                    snapshot
                        .allocations
                        .sort_unstable_by_key(|allocation| allocation.base_column());
                    snapshot.last_allocation_columns = Some(request.columns);
                }
                TcgenLifecycleAction::Deallocate => snapshot.allocations.retain(|allocation| {
                    allocation.base_column() != request.address
                        || allocation.columns() != request.columns
                }),
                TcgenLifecycleAction::Relinquish => snapshot.relinquished = true,
            }
        }
        self.advance_command(state, command_id)
    }

    fn complete_mbarrier(
        &self,
        state: &FixedSyncState,
        completion_id: &FixedSyncCompletionId,
    ) -> Result<FixedSyncState, FixedSyncProgramError> {
        if state.exit_validated {
            return Err(FixedSyncProgramError::DisabledTransition {
                transition: FixedSyncTransition::Complete(completion_id.clone()),
            });
        }
        let Some(completion) = state.pending_completions.get(completion_id).cloned() else {
            return Err(FixedSyncProgramError::DisabledTransition {
                transition: FixedSyncTransition::Complete(completion_id.clone()),
            });
        };
        let mut next = state.clone();
        let effect = match completion.kind {
            FixedSyncMbarrierCompletionKind::Transaction { transactions } => {
                next.mbarriers.complete_tx(
                    &completion.token,
                    transactions,
                    Some(completion_id.issuer.clone()),
                )
            }
            FixedSyncMbarrierCompletionKind::Arrival {
                warp_id,
                arrival_count,
                // The raise was applied when this completion was issued.
                pending_increase: _,
            } => next
                .mbarriers
                .complete_tx(&completion.token, 0, None)
                .and_then(|_| {
                    next.mbarriers.arrive(
                        completion.barrier_id,
                        warp_id,
                        arrival_count,
                        Some(completion_id.issuer.clone()),
                    )
                }),
        }
        .map_err(|error| FixedSyncProgramError::Protocol {
            kind: FixedSyncProtocolKind::Mbarrier,
            operation: completion_id.issuer.clone(),
            details: error.to_string().into_boxed_str(),
        })?;
        next.pending_completions.remove(completion_id);
        let release_command = self
            .commands
            .iter()
            .position(|command| command.witness == completion_id.issuer)
            .map(FixedSyncCommandId)
            .expect("pending completion issuer has a fixed command");
        self.wake_mbarrier_waiters(&mut next, effect.ready_waiters(), release_command)?;
        Ok(next)
    }

    fn grant_setmax(
        &self,
        state: &FixedSyncState,
        request: SetmaxnregResource,
    ) -> Result<FixedSyncState, FixedSyncProgramError> {
        let Some(command_id) = self.setmax_commands.get(&request).copied() else {
            return Err(FixedSyncProgramError::DisabledTransition {
                transition: FixedSyncTransition::SetmaxGrant(request),
            });
        };
        if state.exit_validated || !self.command_ready(state, command_id) {
            return Err(FixedSyncProgramError::DisabledTransition {
                transition: FixedSyncTransition::SetmaxGrant(request),
            });
        }
        let mut next = state.clone();
        let Some(pool) = next
            .setmax_pools
            .get_mut(&(request.kernel_index(), request.global_cta_id()))
        else {
            return Err(self.internal_error(command_id, "setmax pool is missing"));
        };
        pool.apply_grant(request).map_err(|error| {
            self.protocol_error(command_id, FixedSyncProtocolKind::Setmaxnreg, error)
        })?;
        self.advance_command(&mut next, command_id)?;
        Ok(next)
    }

    fn validate_exit(
        &self,
        state: &FixedSyncState,
    ) -> Result<FixedSyncState, FixedSyncProgramError> {
        if state.exit_validated
            || !self.all_warps_finished(state)
            || !state.pending_completions.is_empty()
            || state.setmax_pools.values().any(|pool| !pool.is_quiescent())
        {
            return Err(FixedSyncProgramError::DisabledTransition {
                transition: FixedSyncTransition::ValidateExit,
            });
        }
        // The full-CTA missing-participant diagnosis belongs to the live
        // protocol, which owns the launch topology.  A state-searched
        // projection never holds a named-barrier slot, so it is not repeated
        // here.
        if let Some(incomplete) = state
            .cluster_barriers
            .incomplete_generations()
            .into_iter()
            .next()
        {
            return Err(FixedSyncProgramError::Incomplete {
                kind: FixedSyncProtocolKind::ClusterBarrier,
                operation: incomplete.witness().cloned(),
                details: format!(
                    "cluster barrier {:?} generation {} is missing warps {:?}; exit-aware membership is not modeled",
                    incomplete.barrier_id(),
                    incomplete.generation(),
                    incomplete.missing_warps()
                )
                .into_boxed_str(),
            });
        }
        let live_allocations = state
            .tcgen_ctas
            .iter()
            .filter(|(_, cta)| !cta.allocations.is_empty())
            .map(|(key, cta)| (*key, cta.allocations.clone()))
            .collect::<Vec<_>>();
        if !live_allocations.is_empty() {
            return Err(FixedSyncProgramError::Protocol {
                kind: FixedSyncProtocolKind::TcgenLifecycle,
                operation: self
                    .first_operation()
                    .cloned()
                    .unwrap_or_else(empty_operation_witness),
                details: format!("live TCGEN allocations remain at exit: {live_allocations:?}")
                    .into_boxed_str(),
            });
        }
        let mut next = state.clone();
        next.exit_validated = true;
        Ok(next)
    }
}

impl SyncTransitionSystem for FixedSyncProgram {
    type State = FixedSyncState;
    type Transition = FixedSyncTransition;
    type Error = FixedSyncProgramError;
    type Deadlock = FixedSyncDeadlock;

    fn initial_state(&self) -> Self::State {
        FixedSyncState {
            warp_cursors: vec![0; self.warp_programs.len()].into_boxed_slice(),
            blocked_warps: BTreeMap::new(),
            cluster_waits: BTreeMap::new(),
            mbarriers: StrictMbarrierProtocol::new(),
            // A state-searched projection never holds a named-barrier slot
            // (those projections always terminate in the causal certificate),
            // so this protocol needs no launch topology.
            named_barriers: StrictNamedBarrierProtocol::new(None),
            cluster_barriers: StrictClusterBarrierProtocol::new(),
            setmax_pools: self.initial_setmax_pools.clone(),
            pending_completions: BTreeMap::new(),
            tcgen_ctas: self.initial_tcgen_ctas.clone(),
            exit_validated: false,
        }
    }

    fn enabled_transitions(&self, state: &Self::State) -> Vec<Self::Transition> {
        if state.exit_validated {
            return Vec::new();
        }
        let heads = self
            .warp_ids
            .iter()
            .filter_map(|warp_id| self.head_command(state, *warp_id))
            .collect::<BTreeSet<_>>();
        let mut enabled = heads
            .into_iter()
            .filter(|command| self.issue_ready(state, *command))
            .map(FixedSyncTransition::Issue)
            .collect::<Vec<_>>();
        enabled.extend(
            state
                .pending_completions
                .keys()
                .cloned()
                .map(FixedSyncTransition::Complete),
        );
        for pool in state.setmax_pools.values() {
            enabled.extend(
                pool.enabled_grants()
                    .into_iter()
                    .filter(|request| {
                        self.setmax_commands
                            .get(request)
                            .is_some_and(|command| self.command_ready(state, *command))
                    })
                    .map(FixedSyncTransition::SetmaxGrant),
            );
        }
        if enabled.is_empty()
            && self.all_warps_finished(state)
            && state.pending_completions.is_empty()
            && state
                .setmax_pools
                .values()
                .all(SetmaxnregVerifierCore::is_quiescent)
        {
            enabled.push(FixedSyncTransition::ValidateExit);
        }
        enabled
    }

    fn step(
        &self,
        state: &Self::State,
        transition: &Self::Transition,
    ) -> Result<Self::State, Self::Error> {
        match transition {
            FixedSyncTransition::Issue(command) => self.issue_command(state, *command),
            FixedSyncTransition::Complete(completion) => self.complete_mbarrier(state, completion),
            FixedSyncTransition::SetmaxGrant(request) => self.grant_setmax(state, *request),
            FixedSyncTransition::ValidateExit => self.validate_exit(state),
        }
    }

    fn is_complete(&self, state: &Self::State) -> bool {
        state.exit_validated
    }

    fn describe_deadlock(&self, state: &Self::State) -> Self::Deadlock {
        let unfinished_warps = self
            .warp_ids
            .iter()
            .copied()
            .filter(|warp_id| {
                self.warp_indices.get(warp_id).is_some_and(|warp_index| {
                    state.warp_cursors[*warp_index] < self.warp_programs[*warp_index].len()
                })
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let blocked_warps = state
            .blocked_warps
            .iter()
            .map(|(&warp, &command)| (warp, command))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let pending_setmaxnreg = state
            .setmax_pools
            .values()
            .flat_map(|pool| pool.pending_increases().keys().copied())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let protocol_domain = self.projection_key.map_or_else(
            || "global".into(),
            |key| format!("{key:?}").into_boxed_str(),
        );
        let protocol_state = match self.projection_key {
            Some(FixedSyncProjectionKey::Mbarrier(_)) => format!("{:?}", state.mbarrier_key()),
            Some(FixedSyncProjectionKey::NamedBarrier(_)) => {
                format!("{:?}", state.named_barrier_key())
            }
            Some(FixedSyncProjectionKey::ClusterBarrier(_)) => {
                format!("{:?}", state.cluster_barrier_key())
            }
            Some(FixedSyncProjectionKey::SetmaxnregPool { .. }) => {
                format!("{:?}", state.setmax_pools)
            }
            Some(FixedSyncProjectionKey::TcgenCtaComponent { .. }) => {
                format!("{:?}", state.tcgen_ctas)
            }
            None => format!("{state:?}"),
        }
        .into_boxed_str();
        let unready_heads = unfinished_warps
            .iter()
            .filter_map(|warp_id| {
                let command_id = self.head_command(state, *warp_id)?;
                let command = self.commands.get(command_id.get())?;
                let unmet_predecessors = self.causal_predecessors[command_id.get()]
                    .iter()
                    .copied()
                    .filter(|predecessor| !self.command_completed(state, *predecessor))
                    .map(|predecessor| {
                        let predecessor_command = &self.commands[predecessor.get()];
                        format!(
                            "{predecessor:?} ({}, {:?})",
                            predecessor_command.witness, predecessor_command.kind,
                        )
                    })
                    .collect::<Vec<_>>();
                Some(
                    format!(
                        "warp {warp_id}: head {command_id:?} ({}, {:?}), blocked={}, unmet causal predecessors={unmet_predecessors:?}",
                        command.witness,
                        command.kind,
                        state.blocked_warps.contains_key(warp_id),
                    )
                    .into_boxed_str(),
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        FixedSyncDeadlock {
            protocol_domain,
            protocol_state,
            unfinished_warps,
            blocked_warps,
            pending_setmaxnreg,
            unready_heads,
        }
    }

    fn persistent_transition(
        &self,
        state: &Self::State,
        enabled: &[Self::Transition],
    ) -> Option<Self::Transition> {
        // A terminal transaction completion only supplies bytes to its captured
        // generation. If no future instruction can mutate that barrier, it can
        // move ahead of unrelated work and matching consuming waits. It does not
        // impose an order on that work or combine the wait's barrier targets.
        enabled.iter().find_map(|transition| {
            let FixedSyncTransition::Complete(id) = transition else {
                return None;
            };
            let completion = state.pending_completions.get(id)?;
            let barrier = completion.barrier_id;
            if self.conditional_mbarrier_completions.contains_key(&barrier)
                || state.pending_completions.values().any(|other| {
                    other.barrier_id == barrier
                        && (!matches!(
                            other.kind,
                            FixedSyncMbarrierCompletionKind::Transaction { .. }
                        ) || other.generation != completion.generation)
                })
            {
                return None;
            }
            let phase = completion.generation & 1;
            let conflicts = self.commands.iter().enumerate().any(|(index, command)| {
                if self.command_completed(state, FixedSyncCommandId(index)) {
                    return false;
                }
                match &command.kind {
                    FixedSyncCommandKind::MbarrierInit { barrier_ids, .. }
                    | FixedSyncCommandKind::MbarrierInitFence { barrier_ids }
                    | FixedSyncCommandKind::MbarrierInvalidate { barrier_ids } => {
                        barrier_ids.contains(&barrier)
                    }
                    FixedSyncCommandKind::MbarrierArrive { arrivals } => {
                        arrivals.iter().any(|a| a.0 == barrier)
                    }
                    FixedSyncCommandKind::MbarrierExpectTx { expectations } => {
                        expectations.iter().any(|e| e.0 == barrier)
                    }
                    FixedSyncCommandKind::MbarrierCompletionIssue { completions } => {
                        completions.iter().any(|c| c.barrier_id == barrier)
                    }
                    FixedSyncCommandKind::MbarrierWait {
                        barrier_id,
                        requested_phase,
                        conditional,
                    } => {
                        *barrier_id == barrier
                            && (*conditional
                                || *requested_phase != phase
                                || command
                                    .canonical_generation
                                    .is_some_and(|g| g != completion.generation))
                    }
                    FixedSyncCommandKind::MbarrierWaitBatch { waits, conditional } => {
                        waits.iter().any(|&(b, p, generation)| {
                            b == barrier
                                && (*conditional
                                    || p != phase
                                    || generation.is_some_and(|g| g != completion.generation))
                        })
                    }
                    FixedSyncCommandKind::NamedBarrierArrive { .. }
                    | FixedSyncCommandKind::NamedBarrierSync { .. }
                    | FixedSyncCommandKind::ClusterBarrierArrive { .. }
                    | FixedSyncCommandKind::ClusterBarrierWait { .. }
                    | FixedSyncCommandKind::Setmax { .. }
                    | FixedSyncCommandKind::TcgenLifecycle(_) => false,
                }
            });
            (!conflicts && self.step(state, transition).is_ok()).then(|| transition.clone())
        })
    }

    fn strong_diamond(
        &self,
        state: &Self::State,
        left: &Self::Transition,
        right: &Self::Transition,
    ) -> bool {
        if left == right {
            return false;
        }
        let (Ok(after_left), Ok(after_right)) = (self.step(state, left), self.step(state, right))
        else {
            return false;
        };
        let enabled_before = self
            .enabled_transitions(state)
            .into_iter()
            .collect::<BTreeSet<_>>();
        if !enabled_before.contains(left) || !enabled_before.contains(right) {
            return false;
        }
        let mut expected_after_left = enabled_before.clone();
        expected_after_left.remove(left);
        let mut expected_after_right = enabled_before;
        expected_after_right.remove(right);
        if self
            .enabled_transitions(&after_left)
            .into_iter()
            .collect::<BTreeSet<_>>()
            != expected_after_left
            || self
                .enabled_transitions(&after_right)
                .into_iter()
                .collect::<BTreeSet<_>>()
                != expected_after_right
        {
            return false;
        }
        match (self.step(&after_left, right), self.step(&after_right, left)) {
            (Ok(left_then_right), Ok(right_then_left)) => left_then_right == right_then_left,
            _ => false,
        }
    }

    fn commutes(
        &self,
        state: &Self::State,
        left: &Self::Transition,
        right: &Self::Transition,
    ) -> bool {
        if left == right {
            return false;
        }
        let (Ok(after_left), Ok(after_right)) = (self.step(state, left), self.step(state, right))
        else {
            return false;
        };
        match (self.step(&after_left, right), self.step(&after_right, left)) {
            (Ok(left_then_right), Ok(right_then_left)) => left_then_right == right_then_left,
            _ => false,
        }
    }

    fn successors_commute(
        &self,
        _state: &Self::State,
        left: &Self::Transition,
        after_left: &Self::State,
        right: &Self::Transition,
        after_right: &Self::State,
    ) -> bool {
        if left == right {
            return false;
        }
        match (self.step(after_left, right), self.step(after_right, left)) {
            (Ok(left_then_right), Ok(right_then_left)) => left_then_right == right_then_left,
            _ => false,
        }
    }
}

fn allocate_common_tcgen_interval(
    state: &FixedSyncState,
    request: &FixedSyncTcgenRequest,
) -> Option<FixedTcgenAllocation> {
    crate::tcgen::allocation_interval(
        request.columns,
        request.capacity,
        request
            .participant_ctas
            .iter()
            .filter_map(|cta| state.tcgen_ctas.get(&(request.kernel_index, *cta)))
            .flat_map(|snapshot| snapshot.allocations.iter().copied()),
    )
}

fn empty_operation_witness() -> DynamicOpId {
    DynamicOpId::new(0, 0, 0, crate::StaticOpId::new(0), [])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        explore_sync_states, LoopFrame, StaticOpId, SyncStateFailure, SyncStateSearchLimits,
        SyncStateSearchOptions,
    };

    fn operation(warp_id: usize, sequence: u64) -> DynamicOpId {
        DynamicOpId::new(
            0,
            warp_id,
            sequence,
            StaticOpId::new(100 + warp_id as u64 * 10 + sequence),
            [],
        )
    }

    fn direct_program(
        commands: Vec<FixedSyncCommand>,
        warp_programs: impl IntoIterator<Item = (usize, Vec<usize>)>,
        setmax_pools: BTreeMap<(usize, usize), SetmaxnregVerifierCore>,
        tcgen_ctas: BTreeMap<(usize, usize), FixedSyncTcgenCtaState>,
    ) -> FixedSyncProgram {
        let mut warp_programs = warp_programs.into_iter().collect::<Vec<_>>();
        warp_programs.sort_unstable_by_key(|(warp_id, _)| *warp_id);
        let warp_ids = warp_programs
            .iter()
            .map(|(warp_id, _)| *warp_id)
            .collect::<Vec<_>>();
        let warp_indices = warp_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(index, warp_id)| (warp_id, index))
            .collect();
        let warp_programs = warp_programs
            .into_iter()
            .map(|(_, commands)| {
                commands
                    .into_iter()
                    .map(FixedSyncCommandId)
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let setmax_commands = commands
            .iter()
            .enumerate()
            .filter_map(|(index, command)| match command.kind {
                FixedSyncCommandKind::Setmax { request, .. } => {
                    Some((request, FixedSyncCommandId(index)))
                }
                _ => None,
            })
            .collect();
        let causal_predecessors = (0..commands.len())
            .map(|_| Box::new([]) as Box<[FixedSyncCommandId]>)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let warp_command_positions = build_warp_command_positions(&warp_ids, &warp_programs);
        let model = FixedSyncProgram {
            commands: commands.into_boxed_slice(),
            warp_ids: warp_ids.into_boxed_slice(),
            warp_indices,
            warp_programs,
            setmax_commands,
            projection_key: None,
            causal_predecessors,
            warp_command_positions,
            initial_setmax_pools: setmax_pools,
            initial_tcgen_ctas: tcgen_ctas,
            conditional_mbarrier_completions: BTreeMap::new(),
        };
        model.validate_collective_placement().unwrap();
        model
    }

    fn command(
        witness: DynamicOpId,
        participants: impl IntoIterator<Item = usize>,
        kind: FixedSyncCommandKind,
    ) -> FixedSyncCommand {
        FixedSyncCommand {
            witness,
            participants: participants
                .into_iter()
                .collect::<Vec<_>>()
                .into_boxed_slice()
                .into(),
            kind,
            canonical_generation: None,
            initial_causal_clock: None,
            causal_clock: None,
        }
    }

    fn clock(components: &[u64]) -> crate::SyncVectorClock {
        let mut clock = crate::SyncVectorClock::zero(components.len());
        for (warp_id, &component) in components.iter().enumerate() {
            for _ in 0..component {
                clock.tick(warp_id).unwrap();
            }
        }
        clock
    }

    fn causal_command(
        witness: DynamicOpId,
        kind: FixedSyncCommandKind,
        generation: Option<u64>,
        initial_clock: &[u64],
        final_clock: &[u64],
    ) -> FixedSyncCommand {
        FixedSyncCommand {
            participants: CompactSlice::one(witness.global_warp_id()),
            witness,
            kind,
            canonical_generation: generation,
            initial_causal_clock: Some(clock(initial_clock)),
            causal_clock: Some(clock(final_clock)),
        }
    }

    fn explore(
        model: &FixedSyncProgram,
    ) -> crate::SyncStateSearchResult<FixedSyncTransition, FixedSyncProgramError, FixedSyncDeadlock>
    {
        explore_sync_states(
            model,
            SyncStateSearchLimits::default(),
            SyncStateSearchOptions::default(),
        )
    }

    fn explore_reduced(
        model: &FixedSyncProgram,
    ) -> crate::SyncStateSearchResult<FixedSyncTransition, FixedSyncProgramError, FixedSyncDeadlock>
    {
        explore_sync_states(
            model,
            SyncStateSearchLimits::default(),
            SyncStateSearchOptions {
                stop_on_first_failure: true,
                reduce_all_strong_diamonds: true,
                reduce_sleep_sets: true,
            },
        )
    }

    #[test]
    fn named_barrier_waiters_share_one_fixed_state_machine() {
        let barrier_id = NamedBarrierId::new(0, 3);
        let commands = vec![
            command(
                origin_operation(0, 0, 100),
                [0],
                FixedSyncCommandKind::NamedBarrierSync {
                    barrier_id,
                    expected_arrivals: 64,
                    warp_id: 0,
                    arrival_mask: WarpMask::FULL,
                    aligned: true,
                },
            ),
            command(
                origin_operation(1, 0, 100),
                [1],
                FixedSyncCommandKind::NamedBarrierSync {
                    barrier_id,
                    expected_arrivals: 64,
                    warp_id: 1,
                    arrival_mask: WarpMask::FULL,
                    aligned: true,
                },
            ),
        ];
        let model = direct_program(
            commands,
            [(0, vec![0]), (1, vec![1])],
            BTreeMap::new(),
            BTreeMap::new(),
        );

        let result = explore(&model);
        assert!(result.proves_clean(), "{result:#?}");
    }

    #[test]
    fn repeated_named_barrier_generations_do_not_enumerate_arrival_permutations() {
        let barrier_id = NamedBarrierId::new(0, 3);
        let warp_count = 4;
        let generation_count = 3;
        let mut commands = Vec::new();
        let mut programs = (0..warp_count)
            .map(|warp_id| (warp_id, Vec::new()))
            .collect::<BTreeMap<_, _>>();
        for generation in 0..generation_count {
            for warp_id in 0..warp_count {
                let command_id = commands.len();
                commands.push(command(
                    origin_operation(warp_id, generation as u64, 100),
                    [warp_id],
                    FixedSyncCommandKind::NamedBarrierSync {
                        barrier_id,
                        expected_arrivals: (warp_count * 32) as u64,
                        warp_id,
                        arrival_mask: WarpMask::FULL,
                        aligned: true,
                    },
                ));
                programs.get_mut(&warp_id).unwrap().push(command_id);
            }
        }
        let model = direct_program(commands, programs, BTreeMap::new(), BTreeMap::new());

        let result = explore_reduced(&model);
        assert!(result.proves_clean(), "{result:#?}");
        assert!(
            result.visited_states() <= generation_count * warp_count * 4,
            "sleep-set POR visited {} states",
            result.visited_states(),
        );
    }

    fn origin_operation(warp_id: usize, sequence: u64, source: u64) -> DynamicOpId {
        DynamicOpId::new(0, warp_id, sequence, StaticOpId::new(source), [])
    }

    /// Per warp, `Some(aligned)` builds a blocking sync contribution and
    /// `None` builds an arrive.
    fn named_barrier_causal_model<const WARPS: usize>(
        sources: [u64; WARPS],
        sync_aligned: [Option<bool>; WARPS],
        expected_arrivals: u64,
    ) -> FixedSyncProgram {
        let barrier_id = NamedBarrierId::new(0, 3);
        let commands = sources
            .into_iter()
            .zip(sync_aligned)
            .enumerate()
            .map(|(warp_id, (source, sync_aligned))| {
                let kind = match sync_aligned {
                    Some(aligned) => FixedSyncCommandKind::NamedBarrierSync {
                        barrier_id,
                        expected_arrivals,
                        warp_id,
                        arrival_mask: WarpMask::FULL,
                        aligned,
                    },
                    None => FixedSyncCommandKind::NamedBarrierArrive {
                        barrier_id,
                        expected_arrivals,
                        warp_id,
                        arrival_mask: WarpMask::FULL,
                    },
                };
                let mut initial = [0; WARPS];
                initial[warp_id] = 1;
                let mut release = [0; WARPS];
                release[warp_id] = 2;
                causal_command(
                    origin_operation(warp_id, 0, source),
                    kind,
                    Some(0),
                    &initial,
                    &release,
                )
            })
            .collect();
        let mut model = direct_program(
            commands,
            (0..WARPS).map(|warp_id| (warp_id, vec![warp_id])),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        model.projection_key = Some(FixedSyncProjectionKey::NamedBarrier(barrier_id));
        model
    }

    #[test]
    fn named_barrier_causal_certificate_accepts_distinct_sites_across_warps() {
        let model = named_barrier_causal_model([100, 101], [Some(true); 2], 64);

        assert_eq!(model.verify_named_barrier_causally(), Some(Ok(())));
    }

    #[test]
    fn named_barrier_causal_certificate_accepts_one_static_site_across_loop_iterations() {
        let mut model = named_barrier_causal_model([100, 100], [Some(true); 2], 64);
        model.commands[0].witness = DynamicOpId::new(
            0,
            0,
            0,
            StaticOpId::new(100),
            [LoopFrame::new(StaticOpId::new(90), 0)],
        );
        model.commands[1].witness = DynamicOpId::new(
            0,
            1,
            0,
            StaticOpId::new(100),
            [LoopFrame::new(StaticOpId::new(90), 1)],
        );

        assert_eq!(model.verify_named_barrier_causally(), Some(Ok(())));
    }

    #[test]
    fn named_barrier_origin_check_does_not_require_arrive_and_sync_to_share_origin() {
        let model = named_barrier_causal_model([100, 101], [None, Some(true)], 64);

        assert_eq!(model.verify_named_barrier_causally(), Some(Ok(())));
    }

    #[test]
    fn named_barrier_origin_check_does_not_compare_arrive_origins() {
        let model = named_barrier_causal_model([100, 101], [None, None], 64);

        assert_eq!(model.verify_named_barrier_causally(), Some(Ok(())));
    }

    #[test]
    fn named_barrier_origin_check_does_not_apply_to_unaligned_sync() {
        let model = named_barrier_causal_model([100, 101], [Some(false); 2], 64);

        assert_eq!(model.verify_named_barrier_causally(), Some(Ok(())));
    }

    #[test]
    fn named_barrier_origin_check_rejects_unaligned_then_aligned_sync() {
        let model = named_barrier_causal_model([100, 101], [Some(false), Some(true)], 64);

        assert!(matches!(
            model.verify_named_barrier_causally(),
            Some(Err(FixedSyncProgramError::Protocol { .. }))
        ));
    }

    #[test]
    fn named_barrier_origin_check_rejects_aligned_then_unaligned_sync() {
        let model = named_barrier_causal_model([100, 101], [Some(true), Some(false)], 64);

        assert!(matches!(
            model.verify_named_barrier_causally(),
            Some(Err(FixedSyncProgramError::Protocol { .. }))
        ));
    }

    #[test]
    fn named_barrier_origin_check_accepts_distinct_sync_sites_despite_arrive_contributions() {
        let model = named_barrier_causal_model([100, 101, 102], [None, Some(true), Some(true)], 96);

        assert_eq!(model.verify_named_barrier_causally(), Some(Ok(())));
    }

    #[test]
    fn named_barrier_causal_certificate_counts_arrive_then_sync_from_same_lanes() {
        let barrier_id = NamedBarrierId::new(0, 14);
        let commands = vec![
            causal_command(
                operation(0, 0),
                FixedSyncCommandKind::NamedBarrierArrive {
                    barrier_id,
                    expected_arrivals: 64,
                    warp_id: 0,
                    arrival_mask: WarpMask::FULL,
                },
                Some(0),
                &[0],
                &[1],
            ),
            causal_command(
                operation(0, 1),
                FixedSyncCommandKind::NamedBarrierSync {
                    barrier_id,
                    expected_arrivals: 64,
                    warp_id: 0,
                    arrival_mask: WarpMask::FULL,
                    aligned: true,
                },
                Some(0),
                &[1],
                &[2],
            ),
        ];
        let mut model = direct_program(
            commands,
            [(0, vec![0, 1])],
            BTreeMap::new(),
            BTreeMap::new(),
        );
        model.projection_key = Some(FixedSyncProjectionKey::NamedBarrier(barrier_id));

        assert!(matches!(
            model.verify_named_barrier_causally(),
            Some(Ok(()))
        ));
    }

    #[test]
    fn cross_protocol_cycle_is_a_deadlock_not_a_false_clean() {
        let named = NamedBarrierId::new(0, 2);
        let cluster = ClusterBarrierId::new(0, 0);
        let participants = Box::new([0, 1]);
        let commands = vec![
            command(
                operation(0, 0),
                [0],
                FixedSyncCommandKind::NamedBarrierSync {
                    barrier_id: named,
                    expected_arrivals: 64,
                    warp_id: 0,
                    arrival_mask: WarpMask::FULL,
                    aligned: true,
                },
            ),
            command(
                operation(0, 1),
                [0],
                FixedSyncCommandKind::ClusterBarrierArrive {
                    barrier_id: cluster,
                    warp_id: 0,
                    arrival_mask: WarpMask::FULL,
                    participant_warps: participants.clone(),
                },
            ),
            command(
                operation(0, 2),
                [0],
                FixedSyncCommandKind::ClusterBarrierWait {
                    barrier_id: cluster,
                    warp_id: 0,
                    arrival_mask: WarpMask::FULL,
                    participant_warps: participants.clone(),
                },
            ),
            command(
                operation(1, 0),
                [1],
                FixedSyncCommandKind::ClusterBarrierArrive {
                    barrier_id: cluster,
                    warp_id: 1,
                    arrival_mask: WarpMask::FULL,
                    participant_warps: participants.clone(),
                },
            ),
            command(
                operation(1, 1),
                [1],
                FixedSyncCommandKind::ClusterBarrierWait {
                    barrier_id: cluster,
                    warp_id: 1,
                    arrival_mask: WarpMask::FULL,
                    participant_warps: participants,
                },
            ),
            command(
                operation(1, 2),
                [1],
                FixedSyncCommandKind::NamedBarrierSync {
                    barrier_id: named,
                    expected_arrivals: 64,
                    warp_id: 1,
                    arrival_mask: WarpMask::FULL,
                    aligned: true,
                },
            ),
        ];
        let model = direct_program(
            commands,
            [(0, vec![0, 1, 2]), (1, vec![3, 4, 5])],
            BTreeMap::new(),
            BTreeMap::new(),
        );
        let result = explore(&model);

        assert!(result.failures().iter().any(|failure| matches!(
            failure,
            SyncStateFailure::Deadlock { deadlock, .. }
                if deadlock.unfinished_warps() == [0, 1]
        )));
    }

    #[test]
    fn unordered_mbarrier_init_and_arrive_exposes_the_bad_order() {
        let barrier_id = PhysicalBarrierId::new(1, 0, 0);
        let commands = vec![
            command(
                operation(0, 0),
                [0],
                FixedSyncCommandKind::MbarrierArrive {
                    arrivals: Box::new([(barrier_id, 1, None, false)]),
                },
            ),
            command(
                operation(1, 0),
                [1],
                FixedSyncCommandKind::MbarrierInit {
                    barrier_ids: Box::new([barrier_id]),
                    expected_arrivals: 1,
                },
            ),
        ];
        let model = direct_program(
            commands,
            [(0, vec![0]), (1, vec![1])],
            BTreeMap::new(),
            BTreeMap::new(),
        );
        let result = explore(&model);

        assert!(result.failures().iter().any(|failure| matches!(
            failure,
            SyncStateFailure::Error {
                error: FixedSyncProgramError::Protocol {
                    kind: FixedSyncProtocolKind::Mbarrier,
                    ..
                },
                ..
            }
        )));
    }

    #[test]
    fn counted_mbarrier_causal_certificate_accepts_ordered_generations() {
        let barrier_id = PhysicalBarrierId::new(8, 0, 0);
        let commands = vec![
            causal_command(
                operation(0, 0),
                FixedSyncCommandKind::MbarrierInit {
                    barrier_ids: Box::new([barrier_id]),
                    expected_arrivals: 1,
                },
                None,
                &[0, 0],
                &[1, 0],
            ),
            causal_command(
                operation(0, 1),
                FixedSyncCommandKind::MbarrierInitFence {
                    barrier_ids: Box::new([barrier_id]),
                },
                None,
                &[1, 0],
                &[2, 0],
            ),
            causal_command(
                operation(1, 0),
                FixedSyncCommandKind::MbarrierArrive {
                    arrivals: Box::new([(barrier_id, 1, None, false)]),
                },
                Some(0),
                &[2, 1],
                &[2, 2],
            ),
            causal_command(
                operation(0, 2),
                FixedSyncCommandKind::MbarrierWait {
                    barrier_id,
                    requested_phase: 0,
                    conditional: false,
                },
                Some(0),
                &[2, 2],
                &[3, 2],
            ),
            causal_command(
                operation(1, 1),
                FixedSyncCommandKind::MbarrierWait {
                    barrier_id,
                    requested_phase: 0,
                    conditional: false,
                },
                Some(0),
                &[3, 3],
                &[3, 4],
            ),
            causal_command(
                operation(0, 3),
                FixedSyncCommandKind::MbarrierArrive {
                    arrivals: Box::new([(barrier_id, 1, None, false)]),
                },
                Some(1),
                &[4, 4],
                &[5, 4],
            ),
            causal_command(
                operation(1, 2),
                FixedSyncCommandKind::MbarrierWait {
                    barrier_id,
                    requested_phase: 1,
                    conditional: false,
                },
                Some(1),
                &[5, 5],
                &[5, 6],
            ),
        ];
        let mut model = direct_program(
            commands,
            [(0, vec![0, 1, 3, 5]), (1, vec![2, 4, 6])],
            BTreeMap::new(),
            BTreeMap::new(),
        );
        model.projection_key = Some(FixedSyncProjectionKey::Mbarrier(barrier_id));

        assert_eq!(model.verify_mbarrier_causally(), Some(Ok(())));
    }

    #[test]
    fn counted_mbarrier_causal_certificate_accepts_ordered_init_without_explicit_fence() {
        let barrier_id = PhysicalBarrierId::new(11, 0, 0);
        let commands = vec![
            causal_command(
                operation(0, 0),
                FixedSyncCommandKind::MbarrierInit {
                    barrier_ids: Box::new([barrier_id]),
                    expected_arrivals: 1,
                },
                None,
                &[0],
                &[1],
            ),
            causal_command(
                operation(0, 1),
                FixedSyncCommandKind::MbarrierArrive {
                    arrivals: Box::new([(barrier_id, 1, None, false)]),
                },
                Some(0),
                &[2],
                &[3],
            ),
        ];
        let mut model = direct_program(
            commands,
            [(0, vec![0, 1])],
            BTreeMap::new(),
            BTreeMap::new(),
        );
        model.projection_key = Some(FixedSyncProjectionKey::Mbarrier(barrier_id));

        assert_eq!(model.verify_mbarrier_causally(), Some(Ok(())));
    }

    #[test]
    fn counted_mbarrier_causal_certificate_reports_phase_lap() {
        let barrier_id = PhysicalBarrierId::new(9, 0, 0);
        let slow_wait = operation(1, 1);
        let next_arrival = operation(0, 3);
        let commands = vec![
            causal_command(
                operation(0, 0),
                FixedSyncCommandKind::MbarrierInit {
                    barrier_ids: Box::new([barrier_id]),
                    expected_arrivals: 1,
                },
                None,
                &[0, 0],
                &[1, 0],
            ),
            causal_command(
                operation(0, 1),
                FixedSyncCommandKind::MbarrierInitFence {
                    barrier_ids: Box::new([barrier_id]),
                },
                None,
                &[1, 0],
                &[2, 0],
            ),
            causal_command(
                operation(1, 0),
                FixedSyncCommandKind::MbarrierArrive {
                    arrivals: Box::new([(barrier_id, 1, None, false)]),
                },
                Some(0),
                &[2, 1],
                &[2, 2],
            ),
            causal_command(
                operation(0, 2),
                FixedSyncCommandKind::MbarrierWait {
                    barrier_id,
                    requested_phase: 0,
                    conditional: false,
                },
                Some(0),
                &[2, 2],
                &[3, 2],
            ),
            causal_command(
                slow_wait.clone(),
                FixedSyncCommandKind::MbarrierWait {
                    barrier_id,
                    requested_phase: 0,
                    conditional: false,
                },
                Some(0),
                &[2, 3],
                &[3, 4],
            ),
            causal_command(
                next_arrival.clone(),
                FixedSyncCommandKind::MbarrierArrive {
                    arrivals: Box::new([(barrier_id, 1, None, false)]),
                },
                Some(1),
                &[4, 2],
                &[5, 2],
            ),
            causal_command(
                operation(0, 4),
                FixedSyncCommandKind::MbarrierWait {
                    barrier_id,
                    requested_phase: 1,
                    conditional: false,
                },
                Some(1),
                &[6, 2],
                &[7, 2],
            ),
        ];
        let mut model = direct_program(
            commands,
            [(0, vec![0, 1, 3, 5, 6]), (1, vec![2, 4])],
            BTreeMap::new(),
            BTreeMap::new(),
        );
        model.projection_key = Some(FixedSyncProjectionKey::Mbarrier(barrier_id));

        let Some(Err(FixedSyncProgramError::CausalProtocol {
            operation,
            related_operations,
            details,
            ..
        })) = model.verify_mbarrier_causally()
        else {
            panic!("phase lap must be rejected");
        };
        assert_eq!(operation, slow_wait);
        assert_eq!(related_operations.as_ref(), [next_arrival.clone()]);
        assert!(details.contains("can be overtaken"), "{details}");
        assert!(details.contains(&next_arrival.to_string()), "{details}");
    }

    #[test]
    fn conditional_wait_lifetime_ends_at_conditional_not_primary_completion() {
        let barrier = PhysicalBarrierId::new(91, 0, 0);
        let mut commands = Vec::new();
        let mut push = |kind, generation| {
            let index = commands.len();
            commands.push(causal_command(
                operation(0, index as u64),
                kind,
                generation,
                &[index as u64 * 2],
                &[index as u64 * 2 + 1],
            ));
        };
        push(
            FixedSyncCommandKind::MbarrierInit {
                barrier_ids: Box::new([barrier]),
                expected_arrivals: 1,
            },
            None,
        );
        for generation in 0..5 {
            push(
                FixedSyncCommandKind::MbarrierArrive {
                    arrivals: Box::new([(barrier, 1, None, false)]),
                },
                Some(generation),
            );
            push(
                FixedSyncCommandKind::MbarrierWait {
                    barrier_id: barrier,
                    requested_phase: generation & 1,
                    conditional: false,
                },
                Some(generation),
            );
            if generation == 2 {
                push(
                    FixedSyncCommandKind::MbarrierWait {
                        barrier_id: barrier,
                        requested_phase: 0,
                        conditional: true,
                    },
                    Some(1),
                );
            }
        }
        let order = (0..commands.len()).collect::<Vec<_>>();
        let mut model = direct_program(commands, [(0, order)], BTreeMap::new(), BTreeMap::new());
        model.projection_key = Some(FixedSyncProjectionKey::Mbarrier(barrier));
        model.conditional_mbarrier_completions.insert(
            barrier,
            BTreeMap::from([(operation(0, 0), BTreeSet::from([1, 4]))]),
        );
        assert_eq!(model.verify_mbarrier_causally(), Some(Ok(())));
        assert!(explore(&model).proves_clean());
        let fingerprint = model.mbarrier_state_search_fingerprint();
        let mut renamed = direct_program(
            model.commands.to_vec(),
            [(0, (0..model.commands.len()).collect())],
            BTreeMap::new(),
            BTreeMap::new(),
        );
        renamed.projection_key = model.projection_key;
        renamed.commands[0].witness = origin_operation(0, 0, 999);
        renamed.conditional_mbarrier_completions.insert(
            barrier,
            BTreeMap::from([(renamed.commands[0].witness.clone(), BTreeSet::from([1, 4]))]),
        );
        assert_eq!(renamed.mbarrier_state_search_fingerprint(), fingerprint);
        assert!(model.has_equivalent_mbarrier_state_search(&renamed));

        // A successful conditional completion at primary generation 2 really
        // does overtake the old acquire; failed reports at 2 and 3 did not.
        model
            .conditional_mbarrier_completions
            .get_mut(&barrier)
            .unwrap()
            .get_mut(&operation(0, 0))
            .unwrap()
            .insert(2);
        assert_ne!(model.mbarrier_state_search_fingerprint(), fingerprint);
        assert!(!model.has_equivalent_mbarrier_state_search(&renamed));
        // A new lifetime may reuse exactly the same primary generations.
        // An old recorded wait must not acquire that new lifetime's witness.
        let reinit = operation(0, 100);
        let mut commands = renamed.commands.into_vec();
        commands.push(causal_command(
            reinit.clone(),
            FixedSyncCommandKind::MbarrierInit {
                barrier_ids: Box::new([barrier]),
                expected_arrivals: 1,
            },
            None,
            &[100],
            &[101],
        ));
        renamed.commands = commands.into_boxed_slice();
        renamed
            .conditional_mbarrier_completions
            .get_mut(&barrier)
            .unwrap()
            .insert(reinit.clone(), BTreeSet::from([1, 4]));
        let state = renamed.initial_state();
        state.mbarriers.init(barrier, 1, Some(reinit)).unwrap();
        let old_wait = &renamed.commands[7];
        let error = renamed
            .validate_conditional_acquire(
                &state,
                barrier,
                Some(1),
                0,
                &old_wait.witness,
                old_wait.initial_causal_clock.as_ref(),
            )
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("overtaken by mbarrier reinitialization"));
        renamed
            .validate_conditional_acquire(
                &state,
                barrier,
                None,
                1,
                &operation(0, 101),
                Some(&clock(&[102])),
            )
            .unwrap();
        assert!(matches!(
            model.verify_mbarrier_causally(),
            Some(Err(FixedSyncProgramError::CausalProtocol { .. }))
        ));
        assert!(!explore(&model).proves_clean());

        // Keeping the same operand as a primary parity wait is still invalid.
        model.commands[7].kind = FixedSyncCommandKind::MbarrierWait {
            barrier_id: barrier,
            requested_phase: 0,
            conditional: false,
        };
        assert!(matches!(
            model.verify_mbarrier_causally(),
            Some(Err(FixedSyncProgramError::Protocol { .. }))
        ));
    }

    #[test]
    fn counted_mbarrier_causal_certificate_allows_unconsumed_terminal_generation() {
        let barrier_id = PhysicalBarrierId::new(10, 0, 0);
        let commands = vec![
            causal_command(
                operation(0, 0),
                FixedSyncCommandKind::MbarrierInit {
                    barrier_ids: Box::new([barrier_id]),
                    expected_arrivals: 2,
                },
                None,
                &[0],
                &[1],
            ),
            causal_command(
                operation(0, 1),
                FixedSyncCommandKind::MbarrierInitFence {
                    barrier_ids: Box::new([barrier_id]),
                },
                None,
                &[1],
                &[2],
            ),
            causal_command(
                operation(0, 2),
                FixedSyncCommandKind::MbarrierArrive {
                    arrivals: Box::new([(barrier_id, 1, None, false)]),
                },
                Some(0),
                &[2],
                &[3],
            ),
        ];
        let mut model = direct_program(
            commands,
            [(0, vec![0, 1, 2])],
            BTreeMap::new(),
            BTreeMap::new(),
        );
        model.projection_key = Some(FixedSyncProjectionKey::Mbarrier(barrier_id));

        assert_eq!(model.verify_mbarrier_causally(), Some(Ok(())));
    }

    #[test]
    fn strong_diamond_rejects_a_pair_that_exposes_another_command() {
        let commands = vec![
            command(
                operation(0, 0),
                [0],
                FixedSyncCommandKind::MbarrierInit {
                    barrier_ids: Box::new([PhysicalBarrierId::new(3, 0, 0)]),
                    expected_arrivals: 1,
                },
            ),
            command(
                operation(0, 1),
                [0],
                FixedSyncCommandKind::MbarrierInit {
                    barrier_ids: Box::new([PhysicalBarrierId::new(3, 8, 0)]),
                    expected_arrivals: 1,
                },
            ),
            command(
                operation(1, 0),
                [1],
                FixedSyncCommandKind::MbarrierInit {
                    barrier_ids: Box::new([PhysicalBarrierId::new(3, 16, 0)]),
                    expected_arrivals: 1,
                },
            ),
        ];
        let model = direct_program(
            commands,
            [(0, vec![0, 1]), (1, vec![2])],
            BTreeMap::new(),
            BTreeMap::new(),
        );
        let initial = model.initial_state();

        assert!(!model.strong_diamond(
            &initial,
            &FixedSyncTransition::Issue(FixedSyncCommandId(0)),
            &FixedSyncTransition::Issue(FixedSyncCommandId(2)),
        ));
    }

    #[test]
    fn mbarrier_wait_and_semantic_completion_orders_converge() {
        let barrier_id = PhysicalBarrierId::new(2, 0, 0);
        let commands = vec![
            command(
                operation(0, 0),
                [0],
                FixedSyncCommandKind::MbarrierInit {
                    barrier_ids: Box::new([barrier_id]),
                    expected_arrivals: 1,
                },
            ),
            command(
                operation(0, 1),
                [0],
                FixedSyncCommandKind::MbarrierInitFence {
                    barrier_ids: Box::new([barrier_id]),
                },
            ),
            command(
                operation(0, 2),
                [0],
                FixedSyncCommandKind::MbarrierArrive {
                    arrivals: Box::new([(barrier_id, 1, Some(32), false)]),
                },
            ),
            command(
                operation(0, 3),
                [0],
                FixedSyncCommandKind::MbarrierCompletionIssue {
                    completions: Box::new([FixedSyncMbarrierCompletion {
                        barrier_id,
                        kind: FixedSyncMbarrierCompletionKind::Transaction { transactions: 32 },
                    }]),
                },
            ),
            command(
                operation(0, 4),
                [0],
                FixedSyncCommandKind::MbarrierWait {
                    barrier_id,
                    requested_phase: 0,
                    conditional: false,
                },
            ),
        ];
        let model = direct_program(
            commands,
            [(0, vec![0, 1, 2, 3, 4])],
            BTreeMap::new(),
            BTreeMap::new(),
        );
        let result = explore(&model);

        assert!(result.proves_clean());
    }

    fn independent_mbarrier_program(count: usize) -> FixedSyncProgram {
        let barriers = (0..count)
            .map(|lane| PhysicalBarrierId::new(2, lane * 8, 0))
            .collect::<Vec<_>>();
        let mut kinds = vec![
            FixedSyncCommandKind::MbarrierInit {
                barrier_ids: barriers.clone().into_boxed_slice(),
                expected_arrivals: 1,
            },
            FixedSyncCommandKind::MbarrierInitFence {
                barrier_ids: barriers.clone().into_boxed_slice(),
            },
            FixedSyncCommandKind::MbarrierArrive {
                arrivals: barriers.iter().map(|&b| (b, 1, Some(16), false)).collect(),
            },
        ];
        for &barrier_id in &barriers {
            kinds.push(FixedSyncCommandKind::MbarrierCompletionIssue {
                completions: Box::new([FixedSyncMbarrierCompletion {
                    barrier_id,
                    kind: FixedSyncMbarrierCompletionKind::Transaction { transactions: 16 },
                }]),
            });
        }
        kinds.push(FixedSyncCommandKind::MbarrierWaitBatch {
            waits: barriers.iter().map(|&b| (b, 0, None)).collect(),
            conditional: false,
        });
        let commands = kinds
            .into_iter()
            .enumerate()
            .map(|(i, kind)| command(operation(0, i as u64), [0], kind))
            .collect::<Vec<_>>();
        let order = (0..commands.len()).collect::<Vec<_>>();
        direct_program(commands, [(0, order)], BTreeMap::new(), BTreeMap::new())
    }

    #[test]
    fn terminal_completion_reduction_matches_exhaustive_protocol_outcomes() {
        for bytes in [15, 16, 17] {
            for phase in [0, 1] {
                let mut model = independent_mbarrier_program(2);
                let FixedSyncCommandKind::MbarrierCompletionIssue { completions } =
                    &mut model.commands[3].kind
                else {
                    unreachable!()
                };
                completions[0].kind = FixedSyncMbarrierCompletionKind::Transaction {
                    transactions: bytes,
                };
                let FixedSyncCommandKind::MbarrierWaitBatch { waits, .. } =
                    &mut model.commands[5].kind
                else {
                    unreachable!()
                };
                waits[0].1 = phase;
                let exhaustive = explore(&model);
                let reduced = explore_reduced(&model);
                assert_eq!(
                    reduced.proves_clean(),
                    exhaustive.proves_clean(),
                    "bytes={bytes} phase={phase}"
                );
                // Waiting on the previous phase need not consume this
                // generation's bytes. Only phase 0 demands the full transfer.
                if phase == 0 {
                    assert_eq!(reduced.proves_clean(), bytes == 16);
                }
                if reduced.proves_clean() {
                    assert_eq!(reduced.complete_states(), exhaustive.complete_states());
                } else {
                    assert!(!reduced.failures().is_empty(), "{reduced:#?}");
                }
            }
        }
    }

    #[test]
    fn terminal_completion_proof_rejects_future_barrier_dependencies() {
        let barrier = PhysicalBarrierId::new(2, 0, 0);
        let mutations = [
            FixedSyncCommandKind::MbarrierInit {
                barrier_ids: Box::new([barrier]),
                expected_arrivals: 1,
            },
            FixedSyncCommandKind::MbarrierInvalidate {
                barrier_ids: Box::new([barrier]),
            },
            FixedSyncCommandKind::MbarrierExpectTx {
                expectations: Box::new([(barrier, 16)]),
            },
            FixedSyncCommandKind::MbarrierArrive {
                arrivals: Box::new([(barrier, 1, None, false)]),
            },
            FixedSyncCommandKind::MbarrierWait {
                barrier_id: barrier,
                requested_phase: 1,
                conditional: false,
            },
            FixedSyncCommandKind::MbarrierWait {
                barrier_id: barrier,
                requested_phase: 0,
                conditional: true,
            },
        ];
        for mutation in mutations {
            let mut model = independent_mbarrier_program(1);
            let mut state = model.initial_state();
            for index in 0..4 {
                state = model
                    .step(
                        &state,
                        &FixedSyncTransition::Issue(FixedSyncCommandId(index)),
                    )
                    .unwrap();
            }
            assert!(model
                .persistent_transition(&state, &model.enabled_transitions(&state))
                .is_some());
            model.commands[4].kind = mutation;
            assert!(model
                .persistent_transition(&state, &model.enabled_transitions(&state))
                .is_none());
        }
    }

    #[test]
    fn mbarrier_wait_batch_stays_blocked_until_every_barrier_is_ready() {
        let first = PhysicalBarrierId::new(2, 0, 0);
        let second = PhysicalBarrierId::new(2, 8, 0);
        let completion_operation = operation(0, 3);
        let commands = vec![
            command(
                operation(0, 0),
                [0],
                FixedSyncCommandKind::MbarrierInit {
                    barrier_ids: Box::new([first, second]),
                    expected_arrivals: 1,
                },
            ),
            command(
                operation(0, 1),
                [0],
                FixedSyncCommandKind::MbarrierInitFence {
                    barrier_ids: Box::new([first, second]),
                },
            ),
            command(
                operation(0, 2),
                [0],
                FixedSyncCommandKind::MbarrierArrive {
                    arrivals: Box::new([(first, 1, Some(32), false), (second, 1, Some(32), false)]),
                },
            ),
            command(
                completion_operation.clone(),
                [0],
                FixedSyncCommandKind::MbarrierCompletionIssue {
                    completions: Box::new([
                        FixedSyncMbarrierCompletion {
                            barrier_id: first,
                            kind: FixedSyncMbarrierCompletionKind::Transaction { transactions: 32 },
                        },
                        FixedSyncMbarrierCompletion {
                            barrier_id: second,
                            kind: FixedSyncMbarrierCompletionKind::Transaction { transactions: 32 },
                        },
                    ]),
                },
            ),
            command(
                operation(0, 4),
                [0],
                FixedSyncCommandKind::MbarrierWaitBatch {
                    waits: Box::new([(first, 0, None), (second, 0, None)]),
                    conditional: false,
                },
            ),
        ];
        let mbarrier_anchors = BTreeMap::from([(first, first), (second, first)]);
        let mut staged = StagedFixedSyncProjections::new();
        for command in commands {
            let participant_order = CompactSlice::one((
                command.witness.global_warp_id(),
                command.witness.per_warp_sequence(),
            ));
            stage_fixed_sync_command(
                command,
                participant_order,
                &mbarrier_anchors,
                &BTreeMap::new(),
                &mut staged,
            );
        }
        let mut projections = FixedSyncProgram::build_staged_projections(
            staged,
            1,
            &BTreeMap::new(),
            &BTreeMap::new(),
        );
        assert_eq!(projections.len(), 1);
        let model = projections.pop().unwrap();
        assert!(matches!(
            model.commands[4].kind,
            FixedSyncCommandKind::MbarrierWaitBatch { .. }
        ));
        let mut state = model.initial_state();
        for command_id in 0..5 {
            state = model
                .step(
                    &state,
                    &FixedSyncTransition::Issue(FixedSyncCommandId(command_id)),
                )
                .unwrap();
        }
        assert_eq!(state.blocked_warps.get(&0), Some(&FixedSyncCommandId(4)));

        state = model
            .step(
                &state,
                &FixedSyncTransition::Complete(FixedSyncCompletionId {
                    issuer: completion_operation.clone(),
                    ordinal: 0,
                }),
            )
            .unwrap();
        assert_eq!(state.blocked_warps.get(&0), Some(&FixedSyncCommandId(4)));
        assert_eq!(state.warp_cursors[0], 4);

        state = model
            .step(
                &state,
                &FixedSyncTransition::Complete(FixedSyncCompletionId {
                    issuer: completion_operation,
                    ordinal: 1,
                }),
            )
            .unwrap();
        assert!(!state.blocked_warps.contains_key(&0));
        assert_eq!(state.warp_cursors[0], 5);
    }

    #[test]
    fn tma_completion_with_many_waiters_stays_below_fixed_state_budget() {
        let barrier_id = PhysicalBarrierId::new(2, 0, 0);
        let warp_count = 16;
        let mut commands = vec![
            command(
                operation(0, 0),
                [0],
                FixedSyncCommandKind::MbarrierInit {
                    barrier_ids: Box::new([barrier_id]),
                    expected_arrivals: 1,
                },
            ),
            command(
                operation(0, 1),
                [0],
                FixedSyncCommandKind::MbarrierInitFence {
                    barrier_ids: Box::new([barrier_id]),
                },
            ),
            command(
                operation(0, 2),
                [0],
                FixedSyncCommandKind::MbarrierArrive {
                    arrivals: Box::new([(barrier_id, 1, Some(32), false)]),
                },
            ),
            command(
                operation(0, 3),
                [0],
                FixedSyncCommandKind::MbarrierCompletionIssue {
                    completions: Box::new([FixedSyncMbarrierCompletion {
                        barrier_id,
                        kind: FixedSyncMbarrierCompletionKind::Transaction { transactions: 32 },
                    }]),
                },
            ),
        ];
        let mut programs = (0..warp_count)
            .map(|warp_id| (warp_id, Vec::new()))
            .collect::<BTreeMap<_, _>>();
        programs.get_mut(&0).unwrap().extend([0, 1, 2, 3]);
        for warp_id in 0..warp_count {
            let command_id = commands.len();
            commands.push(command(
                operation(warp_id, programs[&warp_id].len() as u64),
                [warp_id],
                FixedSyncCommandKind::MbarrierWait {
                    barrier_id,
                    requested_phase: 0,
                    conditional: false,
                },
            ));
            programs.get_mut(&warp_id).unwrap().push(command_id);
        }
        let mut model = direct_program(commands, programs, BTreeMap::new(), BTreeMap::new());
        for predecessors in model.causal_predecessors.iter_mut().skip(4) {
            *predecessors = Box::new([FixedSyncCommandId(3)]);
        }

        let result = explore_sync_states(
            &model,
            SyncStateSearchLimits {
                max_states: 64,
                max_transitions: 256,
            },
            SyncStateSearchOptions {
                stop_on_first_failure: true,
                reduce_all_strong_diamonds: true,
                reduce_sleep_sets: true,
            },
        );

        assert!(result.proves_clean(), "{result:#?}");
        assert!(
            result.visited_states() < 64,
            "strong-diamond reduction visited {} states",
            result.visited_states(),
        );
    }

    #[test]
    fn deferred_arrival_completion_wakes_registered_mbarrier_waiter() {
        let barrier_id = PhysicalBarrierId::new(2, 0, 0);
        let commands = vec![
            command(
                operation(0, 0),
                [0],
                FixedSyncCommandKind::MbarrierInit {
                    barrier_ids: Box::new([barrier_id]),
                    expected_arrivals: 1,
                },
            ),
            command(
                operation(0, 1),
                [0],
                FixedSyncCommandKind::MbarrierInitFence {
                    barrier_ids: Box::new([barrier_id]),
                },
            ),
            command(
                operation(0, 2),
                [0],
                FixedSyncCommandKind::MbarrierCompletionIssue {
                    completions: Box::new([FixedSyncMbarrierCompletion {
                        barrier_id,
                        kind: FixedSyncMbarrierCompletionKind::Arrival {
                            warp_id: 0,
                            arrival_count: 1,
                            pending_increase: 0,
                        },
                    }]),
                },
            ),
            command(
                operation(0, 3),
                [0],
                FixedSyncCommandKind::MbarrierWait {
                    barrier_id,
                    requested_phase: 0,
                    conditional: false,
                },
            ),
        ];
        let model = direct_program(
            commands,
            [(0, vec![0, 1, 2, 3])],
            BTreeMap::new(),
            BTreeMap::new(),
        );
        let result = explore(&model);

        assert!(result.proves_clean());
    }

    /// `cp.async.mbarrier.arrive` without `.noinc` raises the current phase's
    /// pending count, so an `mbarrier.init(1)` barrier needs the ordinary
    /// arrival *and* the deferred arrive-on before a phase-0 wait can pass.
    /// Without the raise, the ordinary arrival alone would complete generation 0
    /// and the deferred arrival would over-arrive.
    #[test]
    fn pending_increase_makes_a_deferred_arrival_the_second_required_arrival() {
        let barrier_id = PhysicalBarrierId::new(3, 0, 0);
        let issue = |pending_increase: u64| {
            command(
                operation(0, 2),
                [0],
                FixedSyncCommandKind::MbarrierCompletionIssue {
                    completions: Box::new([FixedSyncMbarrierCompletion {
                        barrier_id,
                        kind: FixedSyncMbarrierCompletionKind::Arrival {
                            warp_id: 0,
                            arrival_count: 1,
                            pending_increase,
                        },
                    }]),
                },
            )
        };
        let program = |pending_increase: u64| {
            direct_program(
                vec![
                    command(
                        operation(0, 0),
                        [0],
                        FixedSyncCommandKind::MbarrierInit {
                            barrier_ids: Box::new([barrier_id]),
                            expected_arrivals: 1,
                        },
                    ),
                    command(
                        operation(0, 1),
                        [0],
                        FixedSyncCommandKind::MbarrierInitFence {
                            barrier_ids: Box::new([barrier_id]),
                        },
                    ),
                    issue(pending_increase),
                    command(
                        operation(0, 3),
                        [0],
                        FixedSyncCommandKind::MbarrierArrive {
                            arrivals: Box::new([(barrier_id, 1, None, false)]),
                        },
                    ),
                    command(
                        operation(0, 4),
                        [0],
                        FixedSyncCommandKind::MbarrierWait {
                            barrier_id,
                            requested_phase: 0,
                            conditional: false,
                        },
                    ),
                ],
                [(0, vec![0, 1, 2, 3, 4])],
                BTreeMap::new(),
                BTreeMap::new(),
            )
        };

        // With the raise the program is exactly balanced.
        assert!(explore(&program(1)).proves_clean());
        // Positive control: without it the ordinary arrival already completes
        // the generation, so the deferred arrive-on over-arrives.
        let unraised = explore(&program(0));
        assert!(!unraised.proves_clean(), "{unraised:#?}");
    }

    #[test]
    fn tcgen_protocol_projections_split_independent_ctas() {
        let mut commands = Vec::new();
        let mut tcgen_ctas = BTreeMap::new();
        for global_cta_id in 0..16 {
            let warp_id = global_cta_id;
            let allocate = FixedSyncTcgenRequest {
                kernel_index: 0,
                action: TcgenLifecycleAction::Allocate,
                address: 0,
                columns: 128,
                cta_group: 1,
                participant_ctas: Box::new([global_cta_id]),
                exclusive: false,
                capacity: TMEM_COLUMN_CAPACITY,
                participant_warps: Box::new([warp_id]),
                canonical_allocation: Some(FixedTcgenAllocation {
                    base_column: 0,
                    columns: 128,
                }),
            };
            let deallocate = FixedSyncTcgenRequest {
                action: TcgenLifecycleAction::Deallocate,
                canonical_allocation: None,
                ..allocate.clone()
            };
            let first_command = commands.len();
            commands.push(command(
                operation(warp_id, 0),
                [warp_id],
                FixedSyncCommandKind::TcgenLifecycle(allocate),
            ));
            commands.push(command(
                operation(warp_id, 1),
                [warp_id],
                FixedSyncCommandKind::TcgenLifecycle(deallocate),
            ));
            debug_assert_eq!(commands.len(), first_command + 2);
            tcgen_ctas.insert((0, global_cta_id), FixedSyncTcgenCtaState::default());
        }
        let anchors = tcgen_component_anchors(&commands);
        let mut staged = StagedFixedSyncProjections::new();
        for command in commands {
            let participant_order = CompactSlice::one((
                command.witness.global_warp_id(),
                command.witness.per_warp_sequence(),
            ));
            stage_fixed_sync_command(
                command,
                participant_order,
                &BTreeMap::new(),
                &anchors,
                &mut staged,
            );
        }
        let projections =
            FixedSyncProgram::build_staged_projections(staged, 1, &BTreeMap::new(), &tcgen_ctas);

        assert_eq!(projections.len(), 16);
        for (global_cta_id, projection) in projections.iter().enumerate() {
            assert_eq!(
                projection.projection_key,
                Some(FixedSyncProjectionKey::TcgenCtaComponent {
                    kernel_index: 0,
                    anchor_global_cta_id: global_cta_id,
                })
            );
            assert_eq!(projection.command_count(), 2);
            assert_eq!(projection.initial_tcgen_ctas.len(), 1);
            let result = explore_reduced(projection);
            assert!(result.proves_clean());
            assert_eq!(result.visited_states(), 4);
        }
    }

    #[test]
    fn tcgen_projection_components_join_overlapping_cta_groups() {
        let request = |participant_ctas: Box<[usize]>, warp_id: usize| {
            FixedSyncCommandKind::TcgenLifecycle(FixedSyncTcgenRequest {
                kernel_index: 0,
                action: TcgenLifecycleAction::Allocate,
                address: 0,
                columns: 128,
                cta_group: participant_ctas.len(),
                exclusive: false,
                capacity: TMEM_COLUMN_CAPACITY,
                participant_ctas,
                participant_warps: Box::new([warp_id]),
                canonical_allocation: Some(FixedTcgenAllocation {
                    base_column: 0,
                    columns: 128,
                }),
            })
        };
        let commands = vec![
            command(operation(0, 0), [0], request(Box::new([0, 1]), 0)),
            command(operation(1, 0), [1], request(Box::new([1, 2]), 1)),
            command(operation(2, 0), [2], request(Box::new([4]), 2)),
        ];

        let anchors = tcgen_component_anchors(&commands);

        assert_eq!(anchors.get(&(0, 0)), Some(&0));
        assert_eq!(anchors.get(&(0, 1)), Some(&0));
        assert_eq!(anchors.get(&(0, 2)), Some(&0));
        assert_eq!(anchors.get(&(0, 4)), Some(&4));
    }

    #[test]
    fn tcgen_canonical_allocation_detects_order_dependent_address() {
        let allocate0 = FixedSyncTcgenRequest {
            kernel_index: 0,
            action: TcgenLifecycleAction::Allocate,
            address: 0,
            columns: 128,
            cta_group: 1,
            participant_ctas: Box::new([0]),
            exclusive: false,
            capacity: TMEM_COLUMN_CAPACITY,
            participant_warps: Box::new([0]),
            canonical_allocation: Some(FixedTcgenAllocation {
                base_column: 0,
                columns: 128,
            }),
        };
        let allocate1 = FixedSyncTcgenRequest {
            participant_warps: Box::new([1]),
            canonical_allocation: Some(FixedTcgenAllocation {
                base_column: 128,
                columns: 128,
            }),
            ..allocate0.clone()
        };
        let deallocate0 = FixedSyncTcgenRequest {
            action: TcgenLifecycleAction::Deallocate,
            address: 0,
            canonical_allocation: None,
            ..allocate0.clone()
        };
        let deallocate1 = FixedSyncTcgenRequest {
            action: TcgenLifecycleAction::Deallocate,
            address: 128,
            canonical_allocation: None,
            ..allocate1.clone()
        };
        let commands = vec![
            command(
                operation(0, 0),
                [0],
                FixedSyncCommandKind::TcgenLifecycle(allocate0),
            ),
            command(
                operation(0, 1),
                [0],
                FixedSyncCommandKind::TcgenLifecycle(deallocate0),
            ),
            command(
                operation(1, 0),
                [1],
                FixedSyncCommandKind::TcgenLifecycle(allocate1),
            ),
            command(
                operation(1, 1),
                [1],
                FixedSyncCommandKind::TcgenLifecycle(deallocate1),
            ),
        ];
        let model = direct_program(
            commands,
            [(0, vec![0, 1]), (1, vec![2, 3])],
            BTreeMap::new(),
            BTreeMap::from([((0, 0), FixedSyncTcgenCtaState::default())]),
        );
        let result = explore(&model);

        assert!(result.failures().iter().any(|failure| matches!(
            failure,
            SyncStateFailure::Error {
                error: FixedSyncProgramError::Protocol {
                    kind: FixedSyncProtocolKind::TcgenLifecycle,
                    ..
                },
                ..
            }
        )));
    }

    #[test]
    fn one_warpgroup_can_execute_multiple_setmax_requests() {
        let increase = SetmaxnregResource::new(0, 0, 0, 0);
        let decrease = SetmaxnregResource::new(0, 0, 0, 1);
        let commands = vec![
            command(
                operation(0, 0),
                0..4,
                FixedSyncCommandKind::Setmax {
                    request: increase,
                    action: SetmaxnregAction::Increase,
                    target_count: 160,
                },
            ),
            command(
                operation(0, 1),
                0..4,
                FixedSyncCommandKind::Setmax {
                    request: decrease,
                    action: SetmaxnregAction::Decrease,
                    target_count: 128,
                },
            ),
        ];
        let pool = SetmaxnregVerifierCore::from_parts(0, 0, 384, vec![128]).unwrap();
        let model = direct_program(
            commands,
            (0..4).map(|warp| (warp, vec![0, 1])),
            BTreeMap::from([((0, 0), pool)]),
            BTreeMap::new(),
        );

        assert!(explore(&model).proves_clean());
    }
}
