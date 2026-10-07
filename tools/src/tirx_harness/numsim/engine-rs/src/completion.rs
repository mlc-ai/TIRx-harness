use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use crate::{AsyncGroupCompletionActionId, DynamicOpId, PhysicalCompletionActionId, WarpContext};

/// Opaque operation text retained only for diagnostics.
///
/// This type deliberately has no comparison, hashing, dereference, string
/// accessor, or formatting implementation. Runtime semantic code can carry a
/// label and can hand it to the typed error sinks below, but cannot inspect it.
#[derive(Clone)]
pub struct DiagnosticLabel {
    value: String,
}

impl DiagnosticLabel {
    pub(crate) fn new(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
        }
    }

    pub(crate) fn engine_error(&self, detail: fmt::Arguments<'_>) -> crate::EngineError {
        crate::EngineError::message(format!("{}{detail}", self.value))
    }

    pub(crate) fn out_of_bounds(&self, detail: fmt::Arguments<'_>) -> crate::EngineError {
        crate::EngineError::out_of_bounds(format!("{}{detail}", self.value))
    }

    pub(crate) fn wrap_engine_error(&self, error: crate::EngineError) -> crate::EngineError {
        error.with_context(format_args!("{}: ", self.value))
    }

    pub(crate) fn warp_collective_divergence(
        &self,
        active_mask: crate::WarpMask,
    ) -> crate::EngineError {
        crate::EngineError::warp_collective_divergence(self.value.clone(), active_mask.bits())
    }

    pub(crate) fn warp_collective_divergence_with_detail(
        &self,
        active_mask: crate::WarpMask,
        detail: fmt::Arguments<'_>,
    ) -> crate::EngineError {
        self.warp_collective_divergence(active_mask)
            .with_context(format_args!("{}{detail}: ", self.value))
    }

    pub(crate) fn collective_publication_error(
        &self,
        key: OccurrenceKey,
        detail: fmt::Arguments<'_>,
    ) -> SynchronizationError {
        SynchronizationError::CollectivePublication {
            key,
            message: format!("{}{detail}", self.value),
        }
    }
}

/// A concrete synchronization domain within one launch.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScopeInstance {
    Global,
    Cluster {
        cluster_id: usize,
    },
    Cta {
        global_cta_id: usize,
    },
    WarpGroup {
        global_cta_id: usize,
        warpgroup_id: usize,
    },
    Warp {
        global_warp_id: usize,
    },
    Custom {
        domain_id: u64,
    },
}

impl fmt::Display for ScopeInstance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Global => f.write_str("global"),
            Self::Cluster { cluster_id } => write!(f, "cluster[{cluster_id}]"),
            Self::Cta { global_cta_id } => write!(f, "cta[{global_cta_id}]"),
            Self::WarpGroup {
                global_cta_id,
                warpgroup_id,
            } => write!(f, "cta[{global_cta_id}].warpgroup[{warpgroup_id}]"),
            Self::Warp { global_warp_id } => write!(f, "warp[{global_warp_id}]"),
            Self::Custom { domain_id } => write!(f, "custom[{domain_id}]"),
        }
    }
}

/// Runtime identity for one dynamic occurrence of a synchronization operation.
///
/// The numeric ID identifies the static source site. The complete native loop
/// iteration path distinguishes runtime invocations of that site. `operation`
/// is diagnostic metadata and deliberately does not participate in identity.
#[derive(Clone)]
pub struct OccurrenceKey {
    static_op_id: u64,
    operation: DiagnosticLabel,
    scope: ScopeInstance,
    loop_iteration_path: Vec<i64>,
}

impl fmt::Debug for OccurrenceKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OccurrenceKey")
            .field("static_op_id", &self.static_op_id)
            .field("scope", &self.scope)
            .field("loop_iteration_path", &self.loop_iteration_path)
            .finish_non_exhaustive()
    }
}

impl PartialEq for OccurrenceKey {
    fn eq(&self, other: &Self) -> bool {
        self.static_op_id == other.static_op_id
            && self.loop_iteration_path == other.loop_iteration_path
            && self.scope == other.scope
    }
}

impl Eq for OccurrenceKey {}

impl PartialOrd for OccurrenceKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OccurrenceKey {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.static_op_id, &self.loop_iteration_path, &self.scope).cmp(&(
            other.static_op_id,
            &other.loop_iteration_path,
            &other.scope,
        ))
    }
}

impl Hash for OccurrenceKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.static_op_id.hash(state);
        self.loop_iteration_path.hash(state);
        self.scope.hash(state);
    }
}

impl OccurrenceKey {
    pub fn new(
        static_op_id: u64,
        operation: impl Into<String>,
        loop_iteration_path: impl IntoIterator<Item = i64>,
        scope: ScopeInstance,
    ) -> Self {
        Self::from_label(
            static_op_id,
            DiagnosticLabel::new(operation),
            loop_iteration_path,
            scope,
        )
    }

    pub(crate) fn from_label(
        static_op_id: u64,
        operation: DiagnosticLabel,
        loop_iteration_path: impl IntoIterator<Item = i64>,
        scope: ScopeInstance,
    ) -> Self {
        Self {
            static_op_id,
            operation,
            scope,
            loop_iteration_path: loop_iteration_path.into_iter().collect(),
        }
    }

    pub fn single(
        static_op_id: u64,
        operation: impl Into<String>,
        occurrence: i64,
        scope: ScopeInstance,
    ) -> Self {
        Self::from_label(
            static_op_id,
            DiagnosticLabel::new(operation),
            [occurrence],
            scope,
        )
    }

    pub fn for_warp(
        static_op_id: u64,
        operation: impl Into<String>,
        loop_iteration_path: impl IntoIterator<Item = i64>,
        context: WarpContext,
    ) -> Self {
        Self::from_label(
            static_op_id,
            DiagnosticLabel::new(operation),
            loop_iteration_path,
            ScopeInstance::Warp {
                global_warp_id: context.global_warp_id(),
            },
        )
    }

    pub fn for_cta(
        static_op_id: u64,
        operation: impl Into<String>,
        loop_iteration_path: impl IntoIterator<Item = i64>,
        context: WarpContext,
    ) -> Self {
        Self::from_label(
            static_op_id,
            DiagnosticLabel::new(operation),
            loop_iteration_path,
            ScopeInstance::Cta {
                global_cta_id: context.global_cta_id(),
            },
        )
    }

    pub fn for_cluster(
        static_op_id: u64,
        operation: impl Into<String>,
        loop_iteration_path: impl IntoIterator<Item = i64>,
        context: WarpContext,
    ) -> Self {
        Self::from_label(
            static_op_id,
            DiagnosticLabel::new(operation),
            loop_iteration_path,
            ScopeInstance::Cluster {
                cluster_id: context.cluster_id(),
            },
        )
    }

    pub(crate) fn for_cluster_labeled(
        static_op_id: u64,
        operation: DiagnosticLabel,
        loop_iteration_path: impl IntoIterator<Item = i64>,
        context: WarpContext,
    ) -> Self {
        Self::from_label(
            static_op_id,
            operation,
            loop_iteration_path,
            ScopeInstance::Cluster {
                cluster_id: context.cluster_id(),
            },
        )
    }

    pub const fn static_op_id(&self) -> u64 {
        self.static_op_id
    }

    pub const fn scope(&self) -> &ScopeInstance {
        &self.scope
    }

    pub fn loop_iteration_path(&self) -> &[i64] {
        &self.loop_iteration_path
    }

    pub(crate) fn write_diagnostic(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_occurrence(formatter, self)
    }
}

impl OccurrenceKey {
    pub(crate) fn completion_not_quiescent_error(
        &self,
        source_name: &'static str,
        detail: fmt::Arguments<'_>,
    ) -> SynchronizationError {
        SynchronizationError::CompletionSourceNotQuiescent {
            source_name,
            details: format!(
                "{}[op={}]@{}#path={:?}{detail}",
                self.operation.value, self.static_op_id, self.scope, self.loop_iteration_path
            ),
        }
    }
}

fn write_occurrence(formatter: &mut fmt::Formatter<'_>, key: &OccurrenceKey) -> fmt::Result {
    write!(
        formatter,
        "{}[op={}]@{}#path={:?}",
        key.operation.value, key.static_op_id, key.scope, key.loop_iteration_path
    )
}

/// Deterministically ordered set of global warp IDs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParticipantSet {
    warp_ids: BTreeSet<usize>,
}

impl ParticipantSet {
    pub fn new(warp_ids: impl IntoIterator<Item = usize>) -> Result<Self, SynchronizationError> {
        let warp_ids = warp_ids.into_iter().collect::<BTreeSet<_>>();
        if warp_ids.is_empty() {
            return Err(SynchronizationError::EmptyParticipantSet);
        }
        Ok(Self { warp_ids })
    }

    pub fn singleton(warp_id: usize) -> Self {
        Self {
            warp_ids: BTreeSet::from([warp_id]),
        }
    }

    pub fn contains(&self, warp_id: usize) -> bool {
        self.warp_ids.contains(&warp_id)
    }

    pub fn len(&self) -> usize {
        self.warp_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.warp_ids.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.warp_ids.iter().copied()
    }

    pub fn missing_from(&self, arrived: &BTreeSet<usize>) -> Vec<usize> {
        self.warp_ids.difference(arrived).copied().collect()
    }
}

/// The immutable participant contract attached to an occurrence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParticipantContract {
    scope: ScopeInstance,
    participants: ParticipantSet,
}

impl ParticipantContract {
    pub fn explicit(scope: ScopeInstance, participants: ParticipantSet) -> Self {
        Self {
            scope,
            participants,
        }
    }

    pub fn warp(context: WarpContext) -> Self {
        Self::explicit(
            ScopeInstance::Warp {
                global_warp_id: context.global_warp_id(),
            },
            ParticipantSet::singleton(context.global_warp_id()),
        )
    }

    pub fn cta(context: WarpContext) -> Self {
        let topology = context.topology();
        let global_cta_id = context.global_cta_id();
        let first_warp = global_cta_id * topology.warps_per_cta();
        let participants = ParticipantSet::new(first_warp..first_warp + topology.warps_per_cta())
            .expect("a valid topology has at least one warp per CTA");
        Self::explicit(ScopeInstance::Cta { global_cta_id }, participants)
    }

    pub fn cluster(context: WarpContext) -> Self {
        let topology = context.topology();
        let cluster_id = context.cluster_id();
        let warps_per_cluster = topology.ctas_per_cluster() * topology.warps_per_cta();
        let first_warp = cluster_id * warps_per_cluster;
        let participants = ParticipantSet::new(first_warp..first_warp + warps_per_cluster)
            .expect("a valid topology has at least one warp per cluster");
        Self::explicit(ScopeInstance::Cluster { cluster_id }, participants)
    }

    pub fn grid(context: WarpContext) -> Self {
        let topology = context.topology();
        let participants = ParticipantSet::new(0..topology.warp_count())
            .expect("a valid topology has at least one warp");
        Self::explicit(ScopeInstance::Global, participants)
    }

    pub fn warpgroup(
        context: WarpContext,
        warps_per_group: usize,
    ) -> Result<Self, SynchronizationError> {
        if warps_per_group == 0 {
            return Err(SynchronizationError::InvalidWarpGroupWidth);
        }
        let topology = context.topology();
        let warpgroup_id = context.warp_id_in_cta() / warps_per_group;
        let local_first = warpgroup_id * warps_per_group;
        let local_end = local_first
            .saturating_add(warps_per_group)
            .min(topology.warps_per_cta());
        let global_cta_id = context.global_cta_id();
        let cta_first = global_cta_id * topology.warps_per_cta();
        let participants = ParticipantSet::new(cta_first + local_first..cta_first + local_end)?;
        Ok(Self::explicit(
            ScopeInstance::WarpGroup {
                global_cta_id,
                warpgroup_id,
            },
            participants,
        ))
    }

    pub const fn scope(&self) -> &ScopeInstance {
        &self.scope
    }

    pub const fn participants(&self) -> &ParticipantSet {
        &self.participants
    }

    pub fn validate_key_and_participant(
        &self,
        key: &OccurrenceKey,
        warp_id: usize,
    ) -> Result<(), SynchronizationError> {
        if key.scope() != self.scope() {
            return Err(SynchronizationError::KeyScopeMismatch {
                key: key.clone(),
                contract_scope: self.scope.clone(),
            });
        }
        if !self.participants.contains(warp_id) {
            return Err(SynchronizationError::NotAParticipant {
                key: key.clone(),
                warp_id,
                participants: self.participants.iter().collect(),
            });
        }
        Ok(())
    }
}

/// Participant and transaction counters included in deadlock diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParticipantState {
    pub expected: Vec<usize>,
    pub arrived: Vec<usize>,
    pub missing: Vec<usize>,
    pub expected_arrival_count: Option<u64>,
    pub completed_arrival_count: Option<u64>,
    pub expected_transactions: Option<u64>,
    pub completed_transactions: Option<u64>,
}

impl ParticipantState {
    pub fn new(
        contract: &ParticipantContract,
        arrived: &BTreeSet<usize>,
        expected_transactions: Option<u64>,
        completed_transactions: Option<u64>,
    ) -> Self {
        Self {
            expected: contract.participants().iter().collect(),
            arrived: arrived.iter().copied().collect(),
            missing: contract.participants().missing_from(arrived),
            expected_arrival_count: None,
            completed_arrival_count: None,
            expected_transactions,
            completed_transactions,
        }
    }

    pub fn counted(
        arrived_warps: impl IntoIterator<Item = usize>,
        expected_arrival_count: u64,
        completed_arrival_count: u64,
        expected_transactions: Option<u64>,
        completed_transactions: Option<u64>,
    ) -> Self {
        Self {
            expected: Vec::new(),
            arrived: arrived_warps.into_iter().collect(),
            missing: Vec::new(),
            expected_arrival_count: Some(expected_arrival_count),
            completed_arrival_count: Some(completed_arrival_count),
            expected_transactions,
            completed_transactions,
        }
    }
}

impl fmt::Display for ParticipantState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "arrived={:?}, missing={:?}, expected={:?}",
            self.arrived, self.missing, self.expected
        )?;
        if let (Some(completed), Some(expected)) =
            (self.completed_arrival_count, self.expected_arrival_count)
        {
            write!(f, ", arrival_count={completed}/{expected}")?;
        }
        if let (Some(completed), Some(expected)) =
            (self.completed_transactions, self.expected_transactions)
        {
            write!(f, ", transactions={completed}/{expected}")?;
        }
        Ok(())
    }
}

/// Fixed kind of completion a warp is waiting for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum AwaitedOperation {
    PhysicalMbarrierTryWait,
    NamedBarrierSync,
    ClusterBarrierWait,
    CollectivePublish,
    Setmaxnreg,
    SetmaxnregPool,
    CpAsyncWaitGroup,
    CpAsyncWaitGroupRead,
    BulkWaitGroup,
    BulkWaitGroupRead,
    /// Recorded at park time rather than reconstructed: the warp is waiting for
    /// a peer to release an overlapping atomic-RMW linearization reservation.
    AtomicLinearization,
    /// Recorded at park time rather than reconstructed: a native `while`
    /// quantum saw no semantic progress and is waiting for any warp to publish
    /// some.
    SemanticProgress,
    /// A wait on a declared synchronization word: the predicate does not hold
    /// on any value the word has held that this actor could still be released
    /// by.
    DeclaredWordWait,
}

impl fmt::Display for AwaitedOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::PhysicalMbarrierTryWait => "mbarrier.try_wait",
            Self::NamedBarrierSync => "bar.sync",
            Self::ClusterBarrierWait => "barrier.cluster.wait",
            Self::CollectivePublish => "collective.publish",
            Self::Setmaxnreg => "setmaxnreg",
            Self::SetmaxnregPool => "setmaxnreg.pool",
            Self::CpAsyncWaitGroup => "cp.async.wait_group",
            Self::CpAsyncWaitGroupRead => "cp.async.wait_group.read",
            Self::BulkWaitGroup => "cp.async.bulk.wait_group",
            Self::BulkWaitGroupRead => "cp.async.bulk.wait_group.read",
            Self::AtomicLinearization => "atom.linearize",
            Self::SemanticProgress => "engine.semantic_progress",
            Self::DeclaredWordWait => "wait_until",
        })
    }
}

/// One warp-level wait recorded by an engine-owned synchronization object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockedOperation {
    pub warp_id: usize,
    awaited_operation: AwaitedOperation,
    pub key: OccurrenceKey,
    pub phase: Option<u64>,
    pub participant_state: ParticipantState,
    operation: Option<DynamicOpId>,
}

impl BlockedOperation {
    pub(crate) const fn new(
        warp_id: usize,
        awaited_operation: AwaitedOperation,
        key: OccurrenceKey,
        phase: Option<u64>,
        participant_state: ParticipantState,
    ) -> Self {
        Self {
            warp_id,
            awaited_operation,
            key,
            phase,
            participant_state,
            operation: None,
        }
    }

    pub(crate) fn diagnostic_cmp(&self, other: &Self) -> Ordering {
        (self.warp_id, &self.key, self.phase, self.awaited_operation).cmp(&(
            other.warp_id,
            &other.key,
            other.phase,
            other.awaited_operation,
        ))
    }

    pub(crate) const fn awaited_operation(&self) -> AwaitedOperation {
        self.awaited_operation
    }

    pub(crate) fn with_awaited_operation(mut self, awaited_operation: AwaitedOperation) -> Self {
        self.awaited_operation = awaited_operation;
        self
    }

    pub(crate) fn with_operation(mut self, operation: Option<DynamicOpId>) -> Self {
        self.operation = operation;
        self
    }

    pub(crate) const fn operation(&self) -> Option<&DynamicOpId> {
        self.operation.as_ref()
    }
}

impl fmt::Display for BlockedOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "warp {} awaits {} key ",
            self.warp_id, self.awaited_operation
        )?;
        write_occurrence(f, &self.key)?;
        if let Some(phase) = self.phase {
            write!(f, " phase {phase}")?;
        }
        write!(f, "; {}", self.participant_state)
    }
}

/// Progress made by one completion-source pump invocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompletionProgress {
    pub completed_operations: usize,
    pub woken_warps: usize,
}

impl CompletionProgress {
    pub const fn made_progress(self) -> bool {
        self.completed_operations != 0 || self.woken_warps != 0
    }
}

/// Completion actions become pump-visible only after the issuing mode has
/// committed its matching post-effect.
///
/// Numeric hubs enqueue actions before `EngineMode::after_effect` runs.  With
/// several ordinary executor workers, another worker may pump completions in
/// that small interval.  Keeping publication separate from numeric enqueueing
/// preserves the operation's before/numeric/after transaction without
/// serializing unrelated clusters.
#[derive(Debug, Default)]
pub(crate) struct CompletionPublicationRegistry {
    state: Mutex<CompletionPublicationState>,
}

#[derive(Debug, Default)]
struct CompletionPublicationState {
    physical: BTreeSet<u64>,
    async_groups: BTreeSet<u64>,
}

impl CompletionPublicationRegistry {
    pub(crate) fn publish_physical(
        &self,
        action_ids: impl IntoIterator<Item = PhysicalCompletionActionId>,
    ) {
        let mut state = self
            .state
            .lock()
            .expect("completion publication registry poisoned");
        state
            .physical
            .extend(action_ids.into_iter().map(PhysicalCompletionActionId::get));
    }

    pub(crate) fn publish_async_groups(
        &self,
        action_ids: impl IntoIterator<Item = AsyncGroupCompletionActionId>,
    ) {
        let mut state = self
            .state
            .lock()
            .expect("completion publication registry poisoned");
        state.async_groups.extend(
            action_ids
                .into_iter()
                .map(AsyncGroupCompletionActionId::get),
        );
    }

    pub(crate) fn physical_is_published(&self, action_id: PhysicalCompletionActionId) -> bool {
        self.state
            .lock()
            .expect("completion publication registry poisoned")
            .physical
            .contains(&action_id.get())
    }

    pub(crate) fn async_group_is_published(&self, action_id: AsyncGroupCompletionActionId) -> bool {
        self.state
            .lock()
            .expect("completion publication registry poisoned")
            .async_groups
            .contains(&action_id.get())
    }
}

/// Engine object that can advance deferred operations and describe its waits.
pub trait CompletionSource: Send + Sync {
    fn source_name(&self) -> &'static str;

    fn pump(&self) -> Result<CompletionProgress, SynchronizationError>;

    fn blocked_operations(&self) -> Vec<BlockedOperation>;

    /// Reject launch exit while this source still owns deferred or incomplete work.
    fn validate_quiescent(&self) -> Result<(), SynchronizationError>;
}

/// Work performed while draining every registered completion source to a fixed point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompletionDrainStats {
    pub pump_count: usize,
    pub completed_operation_count: usize,
    pub woken_warp_count: usize,
}

#[derive(Debug)]
pub struct CompletionRegistryError {
    source_name: &'static str,
    source: Box<SynchronizationError>,
}

impl CompletionRegistryError {
    fn new(source_name: &'static str, source: SynchronizationError) -> Self {
        Self {
            source_name,
            source: Box::new(source),
        }
    }

    pub const fn source_name(&self) -> &'static str {
        self.source_name
    }

    pub fn into_source(self) -> Box<SynchronizationError> {
        self.source
    }
}

impl fmt::Display for CompletionRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} completion source failed: {}",
            self.source_name, self.source
        )
    }
}

impl Error for CompletionRegistryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source.as_ref())
    }
}

/// Stable-order collection of completion sources owned by one engine launch.
#[derive(Default)]
pub struct CompletionRegistry {
    sources: Vec<Arc<dyn CompletionSource>>,
}

impl CompletionRegistry {
    pub const fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    pub fn register<T>(&mut self, source: Arc<T>)
    where
        T: CompletionSource + 'static,
    {
        self.sources.push(source);
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub(crate) fn sources(&self) -> &[Arc<dyn CompletionSource>] {
        &self.sources
    }

    /// Pump all sources in stable registration order until a complete round makes no progress.
    pub fn drain_to_stable(&self) -> Result<CompletionDrainStats, CompletionRegistryError> {
        let mut totals = CompletionDrainStats::default();
        loop {
            let mut round_made_progress = false;
            for source in &self.sources {
                totals.pump_count += 1;
                let progress = source
                    .pump()
                    .map_err(|error| CompletionRegistryError::new(source.source_name(), error))?;
                totals.completed_operation_count += progress.completed_operations;
                totals.woken_warp_count += progress.woken_warps;
                round_made_progress |= progress.made_progress();
            }
            if !round_made_progress {
                return Ok(totals);
            }
        }
    }

    pub fn validate_quiescent(&self) -> Result<(), CompletionRegistryError> {
        for source in &self.sources {
            source
                .validate_quiescent()
                .map_err(|error| CompletionRegistryError::new(source.source_name(), error))?;
        }
        Ok(())
    }

    pub fn drain_to_stable_and_validate(
        &self,
    ) -> Result<CompletionDrainStats, CompletionRegistryError> {
        let stats = self.drain_to_stable()?;
        self.validate_quiescent()?;
        Ok(stats)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterBarrierOperation {
    Arrive,
    Wait,
    UnalignedWaitUnsupported,
}

impl fmt::Display for ClusterBarrierOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Arrive => "barrier.cluster.arrive",
            Self::Wait => "barrier.cluster.wait",
            Self::UnalignedWaitUnsupported => {
                "unaligned barrier.cluster.wait is not representable by the warp-unit engine"
            }
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
// This type remains public only because legacy public wait futures expose it as
// an inferred `Future::Output`.  It is no longer re-exported from `abi::*`, so
// generated artifacts cannot name or match its variants.  Keep checker-only
// payload types at crate visibility rather than widening them to satisfy a
// surface the artifact does not use.
#[allow(private_interfaces)]
pub enum SynchronizationError {
    EmptyParticipantSet,
    InvalidWarpGroupWidth,
    KeyScopeMismatch {
        key: OccurrenceKey,
        contract_scope: ScopeInstance,
    },
    NotAParticipant {
        key: OccurrenceKey,
        warp_id: usize,
        participants: Vec<usize>,
    },
    ContractMismatch {
        key: OccurrenceKey,
    },
    PhaseOutOfOrder {
        key: OccurrenceKey,
        current_phase: u64,
        requested_phase: u64,
    },
    PhaseAlreadyArmed {
        key: OccurrenceKey,
        phase: u64,
    },
    PhaseNotArmed {
        key: OccurrenceKey,
        phase: u64,
    },
    PhaseAlreadyComplete {
        key: OccurrenceKey,
        phase: u64,
    },
    DuplicateArrival {
        key: OccurrenceKey,
        phase: u64,
        warp_id: usize,
    },
    DuplicateContribution {
        key: OccurrenceKey,
        warp_id: usize,
    },
    DuplicateWaiter {
        key: OccurrenceKey,
        phase: Option<u64>,
        warp_id: usize,
    },
    UndefinedOccurrence {
        key: OccurrenceKey,
    },
    TransactionOverflow {
        key: OccurrenceKey,
        phase: u64,
        expected: u64,
        completed: u64,
    },
    InvalidBarrierArrivalCount {
        key: OccurrenceKey,
        count: u64,
    },
    DuplicateMbarrierArrivalTarget {
        key: OccurrenceKey,
    },
    MbarrierLocalArriveRemoteAddress {
        issuer_global_cta_id: usize,
        target_global_cta_id: usize,
    },
    InvalidBarrierPhase {
        key: OccurrenceKey,
        phase: u64,
    },
    InvalidMbarrierStateToken {
        key: OccurrenceKey,
        token_generation: u64,
        current_generation: u64,
    },
    BarrierUninitialized {
        key: OccurrenceKey,
    },
    BarrierReinitializedWhileWaiting {
        key: OccurrenceKey,
        waiting_warps: Vec<usize>,
    },
    BarrierReinitializedWhileActive {
        key: OccurrenceKey,
        generation: u64,
        details: String,
    },
    BarrierReinitializedWithoutInvalidation {
        key: OccurrenceKey,
    },
    BarrierArrivalOverflow {
        key: OccurrenceKey,
        phase: u64,
        expected: u64,
        completed: u64,
    },
    PartialWarpSynchronization {
        operation: ClusterBarrierOperation,
        warp_id: usize,
        active_mask: u32,
    },
    ClusterBarrierContextMismatch {
        operation: ClusterBarrierOperation,
        cluster_id: usize,
        warp_id: usize,
    },
    ClusterBarrierWaitBeforeArrival {
        cluster_id: usize,
        warp_id: usize,
        phase: u64,
    },
    ClusterBarrierRearrivalWithoutWait {
        barrier_id: crate::ClusterBarrierId,
        warp_id: usize,
        generation: u64,
    },
    ClusterBarrierPhaseOverflow {
        cluster_id: usize,
        warp_id: usize,
    },
    TcgenLifecycle(crate::tcgen::TcgenLifecycleError),
    Setmaxnreg(crate::setmaxnreg::SetmaxnregError),
    CollectivePublication {
        key: OccurrenceKey,
        message: String,
    },
    CompletionSourceOperationFailed {
        source_name: &'static str,
        details: String,
    },
    CompletionSourceNotQuiescent {
        source_name: &'static str,
        details: String,
    },
}

impl fmt::Display for SynchronizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyParticipantSet => f.write_str("participant set must not be empty"),
            Self::InvalidWarpGroupWidth => f.write_str("warpgroup width must be non-zero"),
            Self::KeyScopeMismatch {
                key,
                contract_scope,
            } => {
                f.write_str("synchronization key ")?;
                write_occurrence(f, key)?;
                write!(f, " has a different scope from contract {contract_scope}")
            }
            Self::NotAParticipant {
                key,
                warp_id,
                participants,
            } => {
                write!(f, "warp {warp_id} is not a participant in ")?;
                write_occurrence(f, key)?;
                write!(f, "; expected {participants:?}")
            }
            Self::ContractMismatch { key } => {
                f.write_str("participant contract changed for occurrence ")?;
                write_occurrence(f, key)
            }
            Self::PhaseOutOfOrder {
                key,
                current_phase,
                requested_phase,
            } => {
                f.write_str("barrier ")?;
                write_occurrence(f, key)?;
                write!(
                    f,
                    " is at phase {current_phase}, requested phase {requested_phase}"
                )
            }
            Self::PhaseAlreadyArmed { key, phase } => {
                f.write_str("barrier ")?;
                write_occurrence(f, key)?;
                write!(f, " phase {phase} is already armed")
            }
            Self::PhaseNotArmed { key, phase } => {
                f.write_str("barrier ")?;
                write_occurrence(f, key)?;
                write!(f, " phase {phase} has not been armed")
            }
            Self::PhaseAlreadyComplete { key, phase } => {
                f.write_str("barrier ")?;
                write_occurrence(f, key)?;
                write!(f, " phase {phase} is already complete")
            }
            Self::DuplicateArrival {
                key,
                phase,
                warp_id,
            } => {
                write!(f, "warp {warp_id} arrived more than once at barrier ")?;
                write_occurrence(f, key)?;
                write!(f, " phase {phase}")
            }
            Self::DuplicateContribution { key, warp_id } => {
                write!(f, "warp {warp_id} contributed more than once to collective ")?;
                write_occurrence(f, key)
            }
            Self::DuplicateWaiter {
                key,
                phase,
                warp_id,
            } => {
                write!(f, "warp {warp_id} registered multiple waits for ")?;
                write_occurrence(f, key)?;
                if let Some(phase) = phase {
                    write!(f, " phase {phase}")?;
                }
                Ok(())
            }
            Self::UndefinedOccurrence { key } => {
                f.write_str("synchronization occurrence ")?;
                write_occurrence(f, key)?;
                f.write_str(" is undefined")
            }
            Self::TransactionOverflow {
                key,
                phase,
                expected,
                completed,
            } => {
                f.write_str("barrier ")?;
                write_occurrence(f, key)?;
                write!(
                    f,
                    " phase {phase} completed {completed} transactions, expected {expected}"
                )
            }
            Self::InvalidBarrierArrivalCount { key, count } => {
                f.write_str("barrier ")?;
                write_occurrence(f, key)?;
                write!(f, " requires a non-zero arrival count, got {count}")
            }
            Self::DuplicateMbarrierArrivalTarget { key } => {
                f.write_str("one mbarrier arrival batch repeated target barrier ")?;
                write_occurrence(f, key)
            }
            Self::MbarrierLocalArriveRemoteAddress {
                issuer_global_cta_id,
                target_global_cta_id,
            } => write!(
                f,
                "local-form mbarrier.arrive from global CTA {issuer_global_cta_id} used an address mapped to remote global CTA {target_global_cta_id}; use the cluster form with an explicit target CTA"
            ),
            Self::InvalidBarrierPhase { key, phase } => {
                f.write_str("barrier ")?;
                write_occurrence(f, key)?;
                write!(f, " phase must be 0 or 1, got {phase}")
            }
            Self::InvalidMbarrierStateToken {
                key,
                token_generation,
                current_generation,
            } => {
                f.write_str("barrier ")?;
                write_occurrence(f, key)?;
                write!(
                    f,
                    " state token names generation {token_generation}, but the current generation is {current_generation}; a token may name only the current or immediately preceding generation"
                )
            }
            Self::BarrierUninitialized { key } => {
                f.write_str("barrier ")?;
                write_occurrence(f, key)?;
                f.write_str(" was used before mbarrier.init")
            }
            Self::BarrierReinitializedWhileWaiting { key, waiting_warps } => {
                f.write_str("barrier ")?;
                write_occurrence(f, key)?;
                write!(
                    f,
                    " was reinitialized while warps {waiting_warps:?} were waiting"
                )
            }
            Self::BarrierReinitializedWhileActive {
                key,
                generation,
                details,
            } => {
                f.write_str("barrier ")?;
                write_occurrence(f, key)?;
                write!(
                    f,
                    " generation {generation} was reinitialized while active: {details}"
                )
            }
            Self::BarrierReinitializedWithoutInvalidation { key } => {
                f.write_str("barrier ")?;
                write_occurrence(f, key)?;
                f.write_str(" was reinitialized without mbarrier.inval")
            }
            Self::BarrierArrivalOverflow {
                key,
                phase,
                expected,
                completed,
            } => {
                f.write_str("barrier ")?;
                write_occurrence(f, key)?;
                write!(
                    f,
                    " phase {phase} received {completed} lane arrivals, expected {expected}"
                )
            }
            Self::PartialWarpSynchronization {
                operation,
                warp_id,
                active_mask,
            } => write!(
                f,
                "{operation} requires all {} lanes for warp {warp_id}, got mask 0x{active_mask:08x}",
                crate::WARP_SIZE,
            ),
            Self::ClusterBarrierContextMismatch {
                operation,
                cluster_id,
                warp_id,
            } => write!(
                f,
                "{operation} received warp {warp_id} in cluster {cluster_id} from a different launch topology"
            ),
            Self::ClusterBarrierWaitBeforeArrival {
                cluster_id,
                warp_id,
                phase,
            } => write!(
                f,
                "warp {warp_id} in cluster {cluster_id} waited for cluster-barrier phase {phase} before arriving"
            ),
            Self::ClusterBarrierRearrivalWithoutWait {
                barrier_id,
                warp_id,
                generation,
            } => write!(
                f,
                "warp {warp_id} arrived at cluster barrier {barrier_id:?} generation {generation} without consuming its prior arrival with a wait"
            ),
            Self::ClusterBarrierPhaseOverflow {
                cluster_id,
                warp_id,
            } => write!(
                f,
                "warp {warp_id} in cluster {cluster_id} exhausted the cluster-barrier phase counter"
            ),
            Self::TcgenLifecycle(error) => error.fmt(f),
            Self::Setmaxnreg(error) => error.fmt(f),
            Self::CollectivePublication { key, message } => {
                f.write_str("collective ")?;
                write_occurrence(f, key)?;
                write!(f, " publication failed: {message}")
            }
            Self::CompletionSourceOperationFailed {
                source_name,
                details,
            } => write!(f, "completion source {source_name} operation failed: {details}"),
            Self::CompletionSourceNotQuiescent {
                source_name,
                details,
            } => write!(
                f,
                "completion source {source_name} is not quiescent at kernel exit: {details}"
            ),
        }
    }
}

impl Error for SynchronizationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LaunchTopology;

    #[test]
    fn topology_contracts_are_deterministic() {
        let topology = LaunchTopology::new(2, 2, 6).unwrap();
        let context = topology.warp_contexts().nth(10).unwrap();

        assert_eq!(
            ParticipantContract::warp(context)
                .participants()
                .iter()
                .collect::<Vec<_>>(),
            vec![10]
        );
        assert_eq!(
            ParticipantContract::cta(context)
                .participants()
                .iter()
                .collect::<Vec<_>>(),
            vec![6, 7, 8, 9, 10, 11]
        );
        assert_eq!(
            ParticipantContract::cluster(context)
                .participants()
                .iter()
                .collect::<Vec<_>>(),
            (0..12).collect::<Vec<_>>()
        );

        let warpgroup = ParticipantContract::warpgroup(context, 4).unwrap();
        assert_eq!(
            warpgroup.scope(),
            &ScopeInstance::WarpGroup {
                global_cta_id: 1,
                warpgroup_id: 1
            }
        );
        assert_eq!(
            warpgroup.participants().iter().collect::<Vec<_>>(),
            vec![10, 11]
        );
    }

    #[test]
    fn dynamic_occurrences_sort_by_scope_and_counter() {
        let first = OccurrenceKey::new(41, "site", [3, 7], ScopeInstance::Cta { global_cta_id: 2 });
        let second =
            OccurrenceKey::new(41, "site", [3, 8], ScopeInstance::Cta { global_cta_id: 2 });
        assert!(first < second);
        assert_eq!(first.static_op_id(), 41);
        assert_eq!(first.loop_iteration_path(), &[3, 7]);
        assert_eq!(
            SynchronizationError::UndefinedOccurrence { key: first.clone() }.to_string(),
            "synchronization occurrence site[op=41]@cta[2]#path=[3, 7] is undefined"
        );

        let relabeled = OccurrenceKey::new(
            41,
            "diagnostic-label-changed",
            [3, 7],
            ScopeInstance::Cta { global_cta_id: 2 },
        );
        assert_eq!(first, relabeled);
    }
}
