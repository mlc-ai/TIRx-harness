use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};

use crate::{
    BlockedOperation, CollectiveHub, CollectiveWait, CompletionProgress, CompletionSource,
    LaunchTopology, OccurrenceKey, ParticipantContract, ParticipantSet, ScopeInstance,
    SynchronizationError, WarpContext,
};

pub const TMEM_COLUMN_CAPACITY: usize = 512;
const TMEM_ALLOCATION_GRANULARITY: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TcgenLifecycleAction {
    Allocate,
    Deallocate,
    Relinquish,
}

impl TcgenLifecycleAction {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Allocate => "tcgen05.alloc",
            Self::Deallocate => "tcgen05.dealloc",
            Self::Relinquish => "tcgen05.relinquish_alloc_permit",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum TcgenLifecycleErrorKind {
    InvalidColumns,
    InvalidCtaGroup,
    PartialWarpParticipation,
    MissingPeerCta,
    CollectiveArgumentMismatch,
    LifecycleStateMissing,
    CtaGroupMismatch,
    AllocAfterRelinquish,
    AllocationSizeIncrease,
    DeallocationMismatch,
    AllocationUnavailable,
    LiveAllocationsAtExit,
}

impl TcgenLifecycleErrorKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::InvalidColumns => "tcgen_invalid_columns",
            Self::InvalidCtaGroup => "tcgen_invalid_cta_group",
            Self::PartialWarpParticipation => "tcgen_partial_warp_participation",
            Self::MissingPeerCta => "tcgen_missing_peer_cta",
            Self::CollectiveArgumentMismatch => "tcgen_collective_argument_mismatch",
            Self::LifecycleStateMissing => "tcgen_lifecycle_state_missing",
            Self::CtaGroupMismatch => "tcgen_cta_group_mismatch",
            Self::AllocAfterRelinquish => "tcgen_alloc_after_relinquish",
            Self::AllocationSizeIncrease => "tcgen_allocation_size_increase",
            Self::DeallocationMismatch => "tcgen_deallocation_mismatch",
            Self::AllocationUnavailable => "tcgen_allocation_unavailable",
            Self::LiveAllocationsAtExit => "tcgen_live_allocations_at_exit",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TcgenLifecycleError {
    kind: TcgenLifecycleErrorKind,
    key: OccurrenceKey,
    message: String,
}

impl TcgenLifecycleError {
    pub(crate) fn new(
        kind: TcgenLifecycleErrorKind,
        key: OccurrenceKey,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            key,
            message: message.into(),
        }
    }

    pub const fn kind(&self) -> TcgenLifecycleErrorKind {
        self.kind
    }

    pub const fn key(&self) -> &OccurrenceKey {
        &self.key
    }
}

impl fmt::Display for TcgenLifecycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at ", self.kind.name())?;
        self.key.write_diagnostic(f)?;
        write!(f, ": {}", self.message)
    }
}

impl Error for TcgenLifecycleError {}

fn lifecycle_error(
    kind: TcgenLifecycleErrorKind,
    key: OccurrenceKey,
    message: impl Into<String>,
) -> SynchronizationError {
    SynchronizationError::TcgenLifecycle(TcgenLifecycleError::new(kind, key, message))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TcgenAllocation {
    pub base_column: u32,
    pub columns: usize,
}

impl TcgenAllocation {
    pub const fn base_column(self) -> u32 {
        self.base_column
    }
    pub const fn columns(self) -> usize {
        self.columns
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TmemAccessMode {
    Static,
    Dynamic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TmemAccessErrorKind {
    RangeOverflow,
    OutOfBounds,
    InvalidTarget,
    LifecycleStateMissing,
    OutsideLiveAllocation,
}

impl TmemAccessErrorKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::RangeOverflow => "tmem_access_range_overflow",
            Self::OutOfBounds => "tmem_access_out_of_bounds",
            Self::InvalidTarget => "tmem_access_invalid_target",
            Self::LifecycleStateMissing => "tmem_access_lifecycle_state_missing",
            Self::OutsideLiveAllocation => "tmem_access_outside_live_allocation",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TmemAccessError {
    kind: TmemAccessErrorKind,
    message: String,
}

impl TmemAccessError {
    fn new(kind: TmemAccessErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub const fn kind(&self) -> TmemAccessErrorKind {
        self.kind
    }
}

impl fmt::Display for TmemAccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for TmemAccessError {}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TcgenCtaSnapshot {
    pub allocations: Vec<TcgenAllocation>,
    pub relinquished: bool,
    pub cta_group: Option<usize>,
    pub last_allocation_columns: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TcgenContribution {
    action: TcgenLifecycleAction,
    global_cta_id: usize,
    address: u32,
    columns: usize,
    cta_group: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TcgenLifecycleResult {
    pub action: TcgenLifecycleAction,
    pub allocation: Option<TcgenAllocation>,
}

type LifecycleCollective = CollectiveHub<TcgenContribution, TcgenLifecycleResult>;

// Sharded per CTA: access validation on the tcgen05.ld hot path locks only
// the target CTA, while lifecycle mutations lock every contributing CTA in
// index order for cross-CTA atomicity.
type LifecycleState = Arc<Vec<Mutex<TcgenCtaSnapshot>>>;

pub struct TcgenLifecycleHub {
    topology: LaunchTopology,
    state: LifecycleState,
    collective: Arc<LifecycleCollective>,
    columns: usize,
    // Bumped after every published lifecycle change, so memoized TMEM
    // footprint geometry can tell whether the lifecycle validation it was
    // resolved under still describes the live allocations.
    generation: Arc<AtomicU64>,
}

impl TcgenLifecycleHub {
    pub(crate) fn new(topology: LaunchTopology) -> Self {
        Self::with_column_capacity(topology, TMEM_COLUMN_CAPACITY)
    }

    pub(crate) fn with_column_capacity(topology: LaunchTopology, columns: usize) -> Self {
        let state: LifecycleState = Arc::new(
            (0..topology.cta_count())
                .map(|_| Mutex::new(TcgenCtaSnapshot::default()))
                .collect(),
        );
        let publisher_state = Arc::clone(&state);
        let generation = Arc::new(AtomicU64::new(0));
        let publisher_generation = Arc::clone(&generation);
        let collective = Arc::new(CollectiveHub::new(move |contributions| {
            let result = publish_collective(&publisher_state, columns, contributions);
            publisher_generation.fetch_add(1, AtomicOrdering::Release);
            result
        }));
        Self {
            topology,
            state,
            collective,
            columns,
            generation,
        }
    }

    pub(crate) const fn column_capacity(&self) -> usize {
        self.columns
    }

    /// Number of published lifecycle changes so far; any TMEM access
    /// validation resolved under an older generation may be stale.
    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(AtomicOrdering::Acquire)
    }

    pub(crate) fn allocate(
        self: &Arc<Self>,
        static_op_id: u64,
        loop_iteration_path: impl IntoIterator<Item = i64>,
        context: WarpContext,
        columns: usize,
        cta_group: usize,
    ) -> Result<TcgenLifecycleWait, SynchronizationError> {
        validate_columns(columns, true, self.columns, || {
            OccurrenceKey::for_cta(static_op_id, "tcgen05.columns", [], context)
        })?;
        self.begin(
            static_op_id,
            loop_iteration_path,
            context,
            TcgenContribution {
                action: TcgenLifecycleAction::Allocate,
                global_cta_id: context.global_cta_id(),
                address: 0,
                columns,
                cta_group,
            },
            cta_group,
        )
    }

    pub(crate) fn deallocate(
        self: &Arc<Self>,
        static_op_id: u64,
        loop_iteration_path: impl IntoIterator<Item = i64>,
        context: WarpContext,
        address: u32,
        columns: usize,
        cta_group: usize,
    ) -> Result<TcgenLifecycleWait, SynchronizationError> {
        validate_columns(columns, true, self.columns, || {
            OccurrenceKey::for_cta(static_op_id, "tcgen05.columns", [], context)
        })?;
        self.begin(
            static_op_id,
            loop_iteration_path,
            context,
            TcgenContribution {
                action: TcgenLifecycleAction::Deallocate,
                global_cta_id: context.global_cta_id(),
                address,
                columns,
                cta_group,
            },
            cta_group,
        )
    }

    pub(crate) fn relinquish(
        self: &Arc<Self>,
        static_op_id: u64,
        loop_iteration_path: impl IntoIterator<Item = i64>,
        context: WarpContext,
        cta_group: usize,
    ) -> Result<TcgenLifecycleWait, SynchronizationError> {
        self.begin(
            static_op_id,
            loop_iteration_path,
            context,
            TcgenContribution {
                action: TcgenLifecycleAction::Relinquish,
                global_cta_id: context.global_cta_id(),
                address: 0,
                columns: 0,
                cta_group,
            },
            cta_group,
        )
    }

    pub(crate) fn validate_access(
        &self,
        context: WarpContext,
        column: usize,
    ) -> Result<(), SynchronizationError> {
        self.validate_access_range_at_cta(
            context,
            context.cta_id_in_cluster(),
            column,
            1,
            TmemAccessMode::Dynamic,
        )
    }

    pub(crate) fn validate_access_range_at_cta(
        &self,
        context: WarpContext,
        target_cta_id_in_cluster: usize,
        column: usize,
        columns: usize,
        mode: TmemAccessMode,
    ) -> Result<(), SynchronizationError> {
        let key = OccurrenceKey::for_cta(0, "tcgen05.access", [], context);
        self.validate_access_range_at_cta_exact(
            context,
            target_cta_id_in_cluster,
            column,
            columns,
            mode,
        )
        .map_err(|error| SynchronizationError::CollectivePublication {
            key,
            message: error.to_string(),
        })
    }

    /// Validate one column range against a per-target snapshot cached in
    /// `snapshots`, so an instruction with many ranges takes the lifecycle
    /// lock once per target CTA. Semantics are those of
    /// [`Self::validate_access_range_at_cta_exact`]; the snapshot is taken at
    /// the first range of each target.
    pub(crate) fn validate_access_range_at_cta_exact_cached(
        &self,
        context: WarpContext,
        target_cta_id_in_cluster: usize,
        column: usize,
        columns: usize,
        mode: TmemAccessMode,
        snapshots: &mut Vec<(usize, usize, TcgenCtaSnapshot)>,
    ) -> Result<(), TmemAccessError> {
        let end = column.checked_add(columns).ok_or_else(|| {
            TmemAccessError::new(
                TmemAccessErrorKind::RangeOverflow,
                "TMEM access column range overflow",
            )
        })?;
        if columns == 0 || end > TMEM_COLUMN_CAPACITY {
            return Err(TmemAccessError::new(
                TmemAccessErrorKind::OutOfBounds,
                format!(
                    "TMEM access columns [{column}, {end}) are outside [0, {TMEM_COLUMN_CAPACITY})"
                ),
            ));
        }
        if mode == TmemAccessMode::Static {
            return Ok(());
        }
        let position = match snapshots
            .iter()
            .position(|(target, _, _)| *target == target_cta_id_in_cluster)
        {
            Some(position) => position,
            None => {
                let target = crate::CtaId::new(
                    self.topology,
                    context.cluster_id(),
                    target_cta_id_in_cluster,
                )
                .map_err(|error| {
                    TmemAccessError::new(TmemAccessErrorKind::InvalidTarget, error.to_string())
                })?;
                let target_global = target.global_cta_id(self.topology).map_err(|error| {
                    TmemAccessError::new(TmemAccessErrorKind::InvalidTarget, error.to_string())
                })?;
                let state = self
                    .state
                    .get(target_global)
                    .ok_or_else(|| {
                        TmemAccessError::new(
                            TmemAccessErrorKind::LifecycleStateMissing,
                            format!("CTA {target_global} lifecycle state is missing"),
                        )
                    })?
                    .lock()
                    .expect("tcgen lifecycle mutex poisoned")
                    .clone();
                snapshots.push((target_cta_id_in_cluster, target_global, state));
                snapshots.len() - 1
            }
        };
        let (_, target_global, snapshot) = &snapshots[position];
        let covered = snapshot.allocations.iter().any(|allocation| {
            let start = allocation.base_column as usize;
            let allocation_end = start + allocation.columns;
            column >= start && end <= allocation_end
        });
        if !covered {
            return Err(TmemAccessError::new(
                TmemAccessErrorKind::OutsideLiveAllocation,
                format!(
                    "dynamic TMEM access columns [{column}, {end}) are not covered by any live allocation in CTA {target_global}; live allocations are {:?}, relinquished={}, last allocation columns={:?}",
                    snapshot.allocations,
                    snapshot.relinquished,
                    snapshot.last_allocation_columns,
                ),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_access_range_at_cta_exact(
        &self,
        context: WarpContext,
        target_cta_id_in_cluster: usize,
        column: usize,
        columns: usize,
        mode: TmemAccessMode,
    ) -> Result<(), TmemAccessError> {
        let end = column.checked_add(columns).ok_or_else(|| {
            TmemAccessError::new(
                TmemAccessErrorKind::RangeOverflow,
                "TMEM access column range overflow",
            )
        })?;
        if columns == 0 || end > self.columns {
            return Err(TmemAccessError::new(
                TmemAccessErrorKind::OutOfBounds,
                format!(
                    "TMEM access columns [{column}, {end}) are outside [0, {})",
                    self.columns
                ),
            ));
        }
        if mode == TmemAccessMode::Static {
            return Ok(());
        }
        let target = crate::CtaId::new(
            self.topology,
            context.cluster_id(),
            target_cta_id_in_cluster,
        )
        .map_err(|error| {
            TmemAccessError::new(TmemAccessErrorKind::InvalidTarget, error.to_string())
        })?;
        let target_global = target.global_cta_id(self.topology).map_err(|error| {
            TmemAccessError::new(TmemAccessErrorKind::InvalidTarget, error.to_string())
        })?;
        let snapshot = self
            .state
            .get(target_global)
            .ok_or_else(|| {
                TmemAccessError::new(
                    TmemAccessErrorKind::LifecycleStateMissing,
                    format!("CTA {target_global} lifecycle state is missing"),
                )
            })?
            .lock()
            .expect("tcgen lifecycle mutex poisoned");
        let covered = snapshot.allocations.iter().any(|allocation| {
            let start = allocation.base_column as usize;
            let allocation_end = start + allocation.columns;
            column >= start && end <= allocation_end
        });
        if !covered {
            return Err(TmemAccessError::new(
                TmemAccessErrorKind::OutsideLiveAllocation,
                format!(
                    "dynamic TMEM access columns [{column}, {end}) are not covered by any live allocation in CTA {target_global}; live allocations are {:?}, relinquished={}, last allocation columns={:?}",
                    snapshot.allocations,
                    snapshot.relinquished,
                    snapshot.last_allocation_columns,
                ),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_no_live_allocations(&self) -> Result<(), SynchronizationError> {
        let leaked = self
            .state
            .iter()
            .enumerate()
            .filter_map(|(cta, snapshot)| {
                let snapshot = snapshot.lock().expect("tcgen lifecycle mutex poisoned");
                (!snapshot.allocations.is_empty()).then(|| (cta, snapshot.allocations.clone()))
            })
            .collect::<Vec<_>>();
        if leaked.is_empty() {
            return Ok(());
        }
        let context = self
            .topology
            .warp_contexts()
            .find(|context| context.global_cta_id() == leaked[0].0)
            .expect("every CTA has at least one warp context");
        Err(lifecycle_error(
            TcgenLifecycleErrorKind::LiveAllocationsAtExit,
            OccurrenceKey::for_cta(0, "tcgen05.kernel_exit", [], context),
            format!("kernel exited with live TMEM allocations: {leaked:?}"),
        ))
    }

    pub(crate) fn snapshot(&self, global_cta_id: usize) -> Option<TcgenCtaSnapshot> {
        self.state.get(global_cta_id).map(|snapshot| {
            snapshot
                .lock()
                .expect("tcgen lifecycle mutex poisoned")
                .clone()
        })
    }

    fn begin(
        self: &Arc<Self>,
        static_op_id: u64,
        loop_iteration_path: impl IntoIterator<Item = i64>,
        context: WarpContext,
        contribution: TcgenContribution,
        cta_group: usize,
    ) -> Result<TcgenLifecycleWait, SynchronizationError> {
        let path = loop_iteration_path.into_iter().collect::<Vec<_>>();
        match cta_group {
            1 => {
                let key = OccurrenceKey::for_cta(
                    static_op_id,
                    contribution.action.label(),
                    path,
                    context,
                );
                Ok(TcgenLifecycleWait {
                    inner: TcgenLifecycleWaitInner::Immediate {
                        state: Arc::clone(&self.state),
                        capacity: self.columns,
                        key,
                        warp_id: context.global_warp_id(),
                        contribution: Some(contribution),
                    },
                })
            }
            2 => {
                let operation = crate::DiagnosticLabel::new(contribution.action.label());
                let (key, contract) =
                    paired_contract(self.topology, static_op_id, operation, path, context)?;
                let wait = self.collective.collect(
                    key,
                    contract,
                    context.global_warp_id(),
                    contribution,
                )?;
                Ok(TcgenLifecycleWait {
                    inner: TcgenLifecycleWaitInner::Collective(wait),
                })
            }
            _ => Err(lifecycle_error(
                TcgenLifecycleErrorKind::InvalidCtaGroup,
                OccurrenceKey::for_cta(static_op_id, contribution.action.label(), path, context),
                format!("tcgen05 cta_group must be 1 or 2, got {cta_group}"),
            )),
        }
    }
}

impl CompletionSource for TcgenLifecycleHub {
    fn source_name(&self) -> &'static str {
        "tcgen05-lifecycle"
    }

    fn pump(&self) -> Result<CompletionProgress, SynchronizationError> {
        self.collective.pump()
    }

    fn blocked_operations(&self) -> Vec<BlockedOperation> {
        self.collective.blocked_operations()
    }

    fn validate_quiescent(&self) -> Result<(), SynchronizationError> {
        self.collective.validate_quiescent()?;
        self.validate_no_live_allocations()
    }
}

pub struct TcgenLifecycleWait {
    inner: TcgenLifecycleWaitInner,
}

enum TcgenLifecycleWaitInner {
    Immediate {
        state: LifecycleState,
        capacity: usize,
        key: OccurrenceKey,
        warp_id: usize,
        contribution: Option<TcgenContribution>,
    },
    Collective(CollectiveWait<TcgenContribution, TcgenLifecycleResult>),
}

impl Future for TcgenLifecycleWait {
    type Output = Result<Arc<TcgenLifecycleResult>, SynchronizationError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        match &mut self.inner {
            TcgenLifecycleWaitInner::Immediate {
                state,
                capacity,
                key,
                warp_id,
                contribution,
            } => {
                let contribution = contribution
                    .take()
                    .expect("an immediate tcgen lifecycle Future is polled once");
                Poll::Ready(
                    apply_contributions(
                        state,
                        *capacity,
                        key,
                        BTreeMap::from([(*warp_id, contribution)]),
                    )
                    .map(Arc::new),
                )
            }
            TcgenLifecycleWaitInner::Collective(wait) => Pin::new(wait).poll(context),
        }
    }
}

pub(crate) fn validate_columns(
    columns: usize,
    exclusive: bool,
    capacity: usize,
    key: impl FnOnce() -> OccurrenceKey,
) -> Result<(), SynchronizationError> {
    if valid_columns(columns, exclusive, capacity) {
        return Ok(());
    }
    let rule = if exclusive {
        "multiple of 32"
    } else {
        "power of two"
    };
    let capacity = if exclusive {
        capacity
    } else {
        capacity.min(TMEM_COLUMN_CAPACITY)
    };
    Err(lifecycle_error(
        TcgenLifecycleErrorKind::InvalidColumns,
        key(),
        format!("tcgen05 column count must be a {rule} in 32..={capacity}, got {columns}"),
    ))
}

// The qualifier changes legal operand sizes, not allocation ownership or scheduling.
pub(crate) fn valid_columns(columns: usize, exclusive: bool, capacity: usize) -> bool {
    (32..=capacity).contains(&columns)
        && if exclusive {
            columns % TMEM_ALLOCATION_GRANULARITY == 0
        } else {
            columns <= TMEM_COLUMN_CAPACITY && columns.is_power_of_two()
        }
}

/// Shared CTA-local interval rule; SM placement and exclusivity are not modeled.
pub(crate) fn allocation_interval(
    columns: usize,
    capacity: usize,
    allocations: impl Iterator<Item = TcgenAllocation> + Clone,
) -> Option<TcgenAllocation> {
    if !valid_columns(columns, true, capacity) {
        return None;
    }
    (0..=capacity - columns)
        .step_by(TMEM_ALLOCATION_GRANULARITY)
        .find(|&base| {
            allocations.clone().all(|allocation| {
                base + columns <= allocation.base_column as usize
                    || base >= allocation.base_column as usize + allocation.columns
            })
        })
        .map(|base| TcgenAllocation {
            base_column: base as u32,
            columns,
        })
}

fn paired_contract(
    topology: LaunchTopology,
    static_op_id: u64,
    operation: crate::DiagnosticLabel,
    path: Vec<i64>,
    context: WarpContext,
) -> Result<(OccurrenceKey, ParticipantContract), SynchronizationError> {
    let local_cta = context.cta_id_in_cluster();
    let pair_base = local_cta & !1;
    let peer = pair_base + 1;
    if peer >= topology.ctas_per_cluster() {
        return Err(lifecycle_error(
            TcgenLifecycleErrorKind::MissingPeerCta,
            OccurrenceKey::for_cluster_labeled(static_op_id, operation, path, context),
            format!(
                "tcgen05 cta_group=2 has no peer for CTA {} in cluster size {}",
                local_cta,
                topology.ctas_per_cluster()
            ),
        ));
    }
    let cluster_base = context.cluster_id() * topology.ctas_per_cluster();
    let warp_in_cta = context.warp_id_in_cta();
    let participants = ParticipantSet::new(
        [pair_base, peer].map(|cta| (cluster_base + cta) * topology.warps_per_cta() + warp_in_cta),
    )?;
    let domain_id = (cluster_base + pair_base) as u64;
    let scope = ScopeInstance::Custom { domain_id };
    let contract = ParticipantContract::explicit(scope.clone(), participants);
    let key = OccurrenceKey::from_label(static_op_id, operation, path, scope);
    Ok((key, contract))
}

fn publish_collective(
    state: &LifecycleState,
    capacity: usize,
    contributions: BTreeMap<usize, TcgenContribution>,
) -> Result<TcgenLifecycleResult, SynchronizationError> {
    let first = contributions
        .values()
        .next()
        .copied()
        .expect("a collective publishes only after receiving contributions");
    let scope = ScopeInstance::Custom {
        domain_id: first.global_cta_id as u64,
    };
    let key = OccurrenceKey::new(0, first.action.label(), [], scope);
    for contribution in contributions.values() {
        if contribution.action != first.action
            || contribution.columns != first.columns
            || contribution.address != first.address
            || contribution.cta_group != first.cta_group
        {
            return Err(lifecycle_error(
                TcgenLifecycleErrorKind::CollectiveArgumentMismatch,
                key,
                "paired CTAs supplied different tcgen05 lifecycle arguments",
            ));
        }
    }
    apply_contributions(state, capacity, &key, contributions)
}

fn apply_contributions(
    state: &LifecycleState,
    capacity: usize,
    key: &OccurrenceKey,
    contributions: BTreeMap<usize, TcgenContribution>,
) -> Result<TcgenLifecycleResult, SynchronizationError> {
    let first = contributions
        .values()
        .next()
        .copied()
        .expect("tcgen lifecycle action has a contributor");
    // Lock every contributing CTA in ascending index order so paired
    // lifecycle actions stay atomic across their CTAs without a global lock.
    let mut cta_ids = contributions
        .values()
        .map(|contribution| contribution.global_cta_id)
        .collect::<Vec<_>>();
    cta_ids.sort_unstable();
    cta_ids.dedup();
    let mut guards: Vec<(usize, MutexGuard<'_, TcgenCtaSnapshot>)> =
        Vec::with_capacity(cta_ids.len());
    for cta_id in cta_ids {
        let snapshot = state.get(cta_id).ok_or_else(|| {
            lifecycle_error(
                TcgenLifecycleErrorKind::LifecycleStateMissing,
                key.clone(),
                format!("CTA {cta_id} lifecycle state is missing"),
            )
        })?;
        guards.push((
            cta_id,
            snapshot.lock().expect("tcgen lifecycle mutex poisoned"),
        ));
    }
    fn locked_snapshot<'a>(
        guards: &'a [(usize, MutexGuard<'_, TcgenCtaSnapshot>)],
        cta_id: usize,
    ) -> &'a TcgenCtaSnapshot {
        &guards
            .iter()
            .find(|(id, _)| *id == cta_id)
            .expect("every contributing CTA is locked")
            .1
    }
    for contribution in contributions.values() {
        let snapshot = locked_snapshot(&guards, contribution.global_cta_id);
        if snapshot
            .cta_group
            .is_some_and(|cta_group| cta_group != contribution.cta_group)
        {
            return Err(lifecycle_error(
                TcgenLifecycleErrorKind::CtaGroupMismatch,
                key.clone(),
                format!(
                    "CTA {} mixed tcgen05 cta_group={} with prior cta_group={}",
                    contribution.global_cta_id,
                    contribution.cta_group,
                    snapshot.cta_group.expect("checked as present")
                ),
            ));
        }
        match contribution.action {
            TcgenLifecycleAction::Allocate => {
                if snapshot.relinquished {
                    return Err(lifecycle_error(
                        TcgenLifecycleErrorKind::AllocAfterRelinquish,
                        key.clone(),
                        format!(
                            "CTA {} allocated TMEM after relinquishing its permit",
                            contribution.global_cta_id
                        ),
                    ));
                }
                if snapshot
                    .last_allocation_columns
                    .is_some_and(|previous| contribution.columns > previous)
                {
                    return Err(lifecycle_error(
                        TcgenLifecycleErrorKind::AllocationSizeIncrease,
                        key.clone(),
                        format!(
                            "CTA {} increased tcgen05 allocation size from {} to {} columns",
                            contribution.global_cta_id,
                            snapshot
                                .last_allocation_columns
                                .expect("checked as present"),
                            contribution.columns
                        ),
                    ));
                }
            }
            TcgenLifecycleAction::Deallocate => {
                if !snapshot.allocations.iter().any(|allocation| {
                    contribution.address == allocation.base_column
                        && contribution.columns == allocation.columns
                }) {
                    return Err(lifecycle_error(
                        TcgenLifecycleErrorKind::DeallocationMismatch,
                        key.clone(),
                        format!(
                            "CTA {} deallocation (base={}, columns={}) does not match any live allocation {:?}",
                            contribution.global_cta_id,
                            contribution.address,
                            contribution.columns,
                            snapshot.allocations,
                        ),
                    ));
                }
            }
            TcgenLifecycleAction::Relinquish => {}
        }
    }

    let allocation = match first.action {
        TcgenLifecycleAction::Allocate => Some(allocate_common_interval(
            &guards
                .iter()
                .map(|(_, snapshot)| &**snapshot)
                .collect::<Vec<_>>(),
            first.columns,
            capacity,
            key,
        )?),
        TcgenLifecycleAction::Deallocate | TcgenLifecycleAction::Relinquish => None,
    };
    for contribution in contributions.values() {
        let snapshot = &mut *guards
            .iter_mut()
            .find(|(id, _)| *id == contribution.global_cta_id)
            .expect("every contributing CTA is locked")
            .1;
        snapshot.cta_group = Some(contribution.cta_group);
        match contribution.action {
            TcgenLifecycleAction::Allocate => {
                let allocation = allocation.expect("allocate action produced an allocation");
                snapshot.allocations.push(allocation);
                snapshot
                    .allocations
                    .sort_unstable_by_key(|allocation| allocation.base_column);
                snapshot.last_allocation_columns = Some(contribution.columns);
            }
            TcgenLifecycleAction::Deallocate => snapshot.allocations.retain(|allocation| {
                allocation.base_column != contribution.address
                    || allocation.columns != contribution.columns
            }),
            TcgenLifecycleAction::Relinquish => snapshot.relinquished = true,
        }
    }
    Ok(TcgenLifecycleResult {
        action: first.action,
        allocation,
    })
}

fn allocate_common_interval(
    snapshots: &[&TcgenCtaSnapshot],
    columns: usize,
    capacity: usize,
    key: &OccurrenceKey,
) -> Result<TcgenAllocation, SynchronizationError> {
    allocation_interval(columns, capacity, snapshots.iter().flat_map(|s| s.allocations.iter().copied()))
    .ok_or_else(|| lifecycle_error(
        TcgenLifecycleErrorKind::AllocationUnavailable,
        key.clone(),
        format!(
            "tcgen05 allocation of {columns} columns has no common free interval in the participating CTA(s)"
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompletionRegistry, EngineError, Executor, WarpTask};

    fn context(topology: LaunchTopology, cta: usize, warp: usize) -> WarpContext {
        topology
            .warp_contexts()
            .find(|ctx| ctx.global_cta_id() == cta && ctx.warp_id_in_cta() == warp)
            .unwrap()
    }

    #[test]
    fn single_cta_lifecycle_tracks_allocation_and_relinquish() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let hub = Arc::new(TcgenLifecycleHub::new(topology));
        let ctx = context(topology, 0, 0);

        let task_hub = Arc::clone(&hub);
        let task = WarpTask::new(0, async move {
            let allocation = task_hub.allocate(1, [], ctx, 128, 1)?.await?;
            assert_eq!(allocation.allocation.unwrap().base_column, 0);
            task_hub.validate_access(ctx, 127)?;
            task_hub.relinquish(2, [], ctx, 1)?.await?;
            task_hub.deallocate(3, [], ctx, 0, 128, 1)?.await?;
            Ok(())
        });
        Executor::default().run([task]).unwrap();

        assert_eq!(
            hub.snapshot(0),
            Some(TcgenCtaSnapshot {
                allocations: vec![],
                relinquished: true,
                cta_group: Some(1),
                last_allocation_columns: Some(128),
            })
        );
        let task_hub = Arc::clone(&hub);
        let error = Executor::default()
            .run([WarpTask::new(0, async move {
                task_hub.allocate(4, [], ctx, 64, 1)?.await?;
                Ok(())
            })])
            .unwrap_err();
        let crate::EngineErrorKind::WarpFailed { source, .. } = error.kind() else {
            panic!("expected failed lifecycle warp")
        };
        let crate::EngineErrorKind::Synchronization(error) = source.kind() else {
            panic!("expected synchronization error")
        };
        let SynchronizationError::TcgenLifecycle(error) = error.as_ref() else {
            panic!("expected typed lifecycle error")
        };
        assert_eq!(error.kind(), TcgenLifecycleErrorKind::AllocAfterRelinquish);
    }

    #[test]
    fn single_cta_lifecycle_mutates_only_when_awaited() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let hub = Arc::new(TcgenLifecycleHub::new(topology));
        let ctx = context(topology, 0, 0);
        let wait = hub.allocate(1, [], ctx, 32, 1).unwrap();

        assert!(hub.snapshot(0).unwrap().allocations.is_empty());
        Executor::default()
            .run([WarpTask::new(0, async move {
                wait.await?;
                Ok(())
            })])
            .unwrap();
        assert_eq!(hub.snapshot(0).unwrap().allocations.len(), 1);
    }

    #[test]
    fn paired_ctas_block_until_matching_peer_arrives() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let hub = Arc::new(TcgenLifecycleHub::new(topology));
        let mut completions = CompletionRegistry::new();
        completions.register(Arc::clone(&hub));
        let tasks = [0, 1].map(|cta| {
            let hub = Arc::clone(&hub);
            let ctx = context(topology, cta, 0);
            WarpTask::new(ctx.global_warp_id(), async move {
                hub.allocate(7, [0], ctx, 256, 2).unwrap().await?;
                Ok(())
            })
        });
        let stats = Executor::default()
            .run_with_completions(tasks, &completions)
            .unwrap();
        assert_eq!(stats.completed_task_count, 2);
        assert_eq!(hub.snapshot(0).unwrap().allocations[0].columns, 256);
        assert_eq!(hub.snapshot(1).unwrap().allocations[0].columns, 256);
    }

    #[test]
    fn missing_peer_has_typed_deadlock_diagnostics() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let hub = Arc::new(TcgenLifecycleHub::new(topology));
        let mut completions = CompletionRegistry::new();
        completions.register(Arc::clone(&hub));
        let ctx = context(topology, 0, 0);
        let task = {
            let hub = Arc::clone(&hub);
            WarpTask::new(0, async move {
                hub.allocate(9, [], ctx, 64, 2).unwrap().await?;
                Ok(())
            })
        };
        let error = Executor::default()
            .run_with_completions([task], &completions)
            .unwrap_err();
        let crate::EngineErrorKind::Deadlock {
            blocked_operations, ..
        } = error.kind()
        else {
            panic!("expected deadlock")
        };
        assert_eq!(blocked_operations.len(), 1);
        assert!(blocked_operations[0]
            .to_string()
            .contains("collective.publish"));
    }

    #[test]
    fn allocator_uses_nonzero_bases_and_reuses_released_intervals() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let hub = Arc::new(TcgenLifecycleHub::new(topology));
        let ctx = context(topology, 0, 0);
        let task_hub = Arc::clone(&hub);
        let task = WarpTask::new(0, async move {
            let first = task_hub.allocate(1, [], ctx, 128, 1)?.await?;
            let second = task_hub.allocate(2, [], ctx, 64, 1)?.await?;
            assert_eq!(first.allocation.unwrap().base_column, 0);
            assert_eq!(second.allocation.unwrap().base_column, 128);
            task_hub.validate_access_range_at_cta(ctx, 0, 128, 64, TmemAccessMode::Dynamic)?;
            task_hub.deallocate(3, [], ctx, 0, 128, 1)?.await?;
            let reused = task_hub.allocate(4, [], ctx, 32, 1)?.await?;
            assert_eq!(reused.allocation.unwrap().base_column, 0);
            task_hub.deallocate(5, [], ctx, 128, 64, 1)?.await?;
            task_hub.deallocate(6, [], ctx, 0, 32, 1)?.await?;
            Ok(())
        });
        Executor::default().run([task]).unwrap();
        hub.validate_no_live_allocations().unwrap();
    }

    #[test]
    fn paired_allocations_choose_a_common_nonzero_base() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let hub = Arc::new(TcgenLifecycleHub::new(topology));
        let mut completions = CompletionRegistry::new();
        completions.register(Arc::clone(&hub));
        let tasks = [0, 1].map(|cta| {
            let hub = Arc::clone(&hub);
            let ctx = context(topology, cta, 0);
            WarpTask::new(ctx.global_warp_id(), async move {
                let first = hub.allocate(1, [], ctx, 128, 2)?.await?;
                let second = hub.allocate(2, [], ctx, 64, 2)?.await?;
                assert_eq!(first.allocation.unwrap().base_column, 0);
                assert_eq!(second.allocation.unwrap().base_column, 128);
                hub.deallocate(3, [], ctx, 0, 128, 2)?.await?;
                hub.deallocate(4, [], ctx, 128, 64, 2)?.await?;
                Ok(())
            })
        });
        Executor::default()
            .run_with_completions(tasks, &completions)
            .unwrap();
        hub.validate_no_live_allocations().unwrap();
    }

    #[test]
    fn dynamic_access_requires_a_live_covering_allocation() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let hub = Arc::new(TcgenLifecycleHub::new(topology));
        let ctx = context(topology, 0, 0);
        assert!(hub.validate_access(ctx, 0).is_err());

        let task_hub = Arc::clone(&hub);
        let task = WarpTask::new(0, async move {
            task_hub.allocate(1, [], ctx, 64, 1)?.await?;
            task_hub.validate_access_range_at_cta(ctx, 0, 0, 64, TmemAccessMode::Dynamic)?;
            assert!(task_hub
                .validate_access_range_at_cta(ctx, 0, 63, 2, TmemAccessMode::Dynamic)
                .is_err());
            task_hub.relinquish(2, [], ctx, 1)?.await?;
            task_hub.deallocate(3, [], ctx, 0, 64, 1)?.await?;
            let error = task_hub
                .validate_access_range_at_cta_exact(
                    ctx,
                    ctx.cta_id_in_cluster(),
                    0,
                    1,
                    TmemAccessMode::Dynamic,
                )
                .unwrap_err();
            assert_eq!(error.kind(), TmemAccessErrorKind::OutsideLiveAllocation);
            let message = error.to_string();
            assert!(message.contains("live allocations are []"), "{message}");
            assert!(message.contains("relinquished=true"), "{message}");
            assert!(
                message.contains("last allocation columns=Some(64)"),
                "{message}"
            );
            task_hub.validate_access_range_at_cta(ctx, 0, 0, 1, TmemAccessMode::Static)?;
            Ok(())
        });
        Executor::default().run([task]).unwrap();
    }

    #[test]
    fn live_allocations_are_reported_at_kernel_exit() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let hub = Arc::new(TcgenLifecycleHub::new(topology));
        let ctx = context(topology, 0, 0);
        let task_hub = Arc::clone(&hub);
        Executor::default()
            .run([WarpTask::new(0, async move {
                task_hub.allocate(1, [], ctx, 32, 1)?.await?;
                Ok(())
            })])
            .unwrap();
        assert!(hub.validate_no_live_allocations().is_err());
    }
}
