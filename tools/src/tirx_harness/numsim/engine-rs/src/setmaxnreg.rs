use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use crate::completion_action_id::{pair, CompletionActionNamespace};
use crate::{
    BlockedOperation, CollectiveHub, CollectiveWait, CompletionProgress, CompletionSource,
    DynamicOpId, EngineError, LaunchTopology, NamedBarrierId, OccurrenceKey, OperationContext,
    OperationKind, ParticipantContract, SynchronizationError, WarpContext, WarpMask,
};

pub const SETMAXNREG_WARPS_PER_GROUP: usize = 4;
pub const SETMAXNREG_CTA_REGISTER_POOL: i64 = 512;
pub const SETMAXNREG_MIN_COUNT: i64 = 24;
pub const SETMAXNREG_MAX_COUNT: i64 = 256;
pub const SETMAXNREG_COUNT_GRANULARITY: i64 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SetmaxnregAction {
    Increase,
    Decrease,
}

impl SetmaxnregAction {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Increase => "setmaxnreg.inc",
            Self::Decrease => "setmaxnreg.dec",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SetmaxnregResource {
    kernel_index: usize,
    global_cta_id: usize,
    warpgroup_id: usize,
    ordinal: u64,
}

impl SetmaxnregResource {
    pub const fn new(
        kernel_index: usize,
        global_cta_id: usize,
        warpgroup_id: usize,
        ordinal: u64,
    ) -> Self {
        Self {
            kernel_index,
            global_cta_id,
            warpgroup_id,
            ordinal,
        }
    }

    pub const fn kernel_index(self) -> usize {
        self.kernel_index
    }

    pub const fn global_cta_id(self) -> usize {
        self.global_cta_id
    }

    pub const fn warpgroup_id(self) -> usize {
        self.warpgroup_id
    }

    pub const fn ordinal(self) -> u64 {
        self.ordinal
    }
}

impl fmt::Display for SetmaxnregResource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "kernel:{}/cta:{}/warpgroup:{}/setmaxnreg:{}",
            self.kernel_index, self.global_cta_id, self.warpgroup_id, self.ordinal
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum SetmaxnregErrorKind {
    InvalidCount,
    InvalidDirection,
    RegisterOversubscription,
    PartialWarpParticipation,
    IncompleteWarpgroup,
    ContextMismatch,
    SequenceOverflow,
    SequenceMismatch,
    Divergence,
    MissingWarpgroupSync,
}

impl SetmaxnregErrorKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::InvalidCount => "setmaxnreg_invalid_count",
            Self::InvalidDirection => "setmaxnreg_invalid_direction",
            Self::RegisterOversubscription => "register_oversubscription",
            Self::PartialWarpParticipation => "setmaxnreg_partial_warp_participation",
            Self::IncompleteWarpgroup => "setmaxnreg_incomplete_warpgroup",
            Self::ContextMismatch => "setmaxnreg_context_mismatch",
            Self::SequenceOverflow => "setmaxnreg_sequence_overflow",
            Self::SequenceMismatch => "setmaxnreg_sequence_mismatch",
            Self::Divergence => "setmaxnreg_divergence",
            Self::MissingWarpgroupSync => "setmaxnreg_missing_warpgroup_sync",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SetmaxnregError {
    kind: SetmaxnregErrorKind,
    resource: SetmaxnregResource,
    message: String,
    prior_witness: Option<Box<DynamicOpId>>,
    witness: Option<Box<DynamicOpId>>,
}

impl SetmaxnregError {
    fn new(
        kind: SetmaxnregErrorKind,
        resource: SetmaxnregResource,
        message: impl Into<String>,
        prior_witness: Option<DynamicOpId>,
        witness: Option<DynamicOpId>,
    ) -> Self {
        Self {
            kind,
            resource,
            message: message.into(),
            prior_witness: prior_witness.map(Box::new),
            witness: witness.map(Box::new),
        }
    }

    pub const fn kind(&self) -> SetmaxnregErrorKind {
        self.kind
    }

    pub const fn resource(&self) -> SetmaxnregResource {
        self.resource
    }

    pub(crate) fn prior_witness(&self) -> Option<&DynamicOpId> {
        self.prior_witness.as_deref()
    }

    pub(crate) fn witness(&self) -> Option<&DynamicOpId> {
        self.witness.as_deref()
    }
}

impl fmt::Display for SetmaxnregError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at {}: {}",
            self.kind.name(),
            self.resource,
            self.message
        )
    }
}

impl Error for SetmaxnregError {}

fn protocol_error(
    kind: SetmaxnregErrorKind,
    resource: SetmaxnregResource,
    message: impl Into<String>,
    prior_witness: Option<DynamicOpId>,
    witness: Option<DynamicOpId>,
) -> SynchronizationError {
    SynchronizationError::Setmaxnreg(SetmaxnregError::new(
        kind,
        resource,
        message,
        prior_witness,
        witness,
    ))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetmaxnregPlan {
    resource: SetmaxnregResource,
    context: WarpContext,
    action: SetmaxnregAction,
    count: i64,
    witness: Option<Box<DynamicOpId>>,
}

impl SetmaxnregPlan {
    pub const fn resource(&self) -> SetmaxnregResource {
        self.resource
    }

    pub const fn context(&self) -> WarpContext {
        self.context
    }

    pub const fn action(&self) -> SetmaxnregAction {
        self.action
    }

    pub const fn count(&self) -> i64 {
        self.count
    }

    pub fn witness(&self) -> Option<&DynamicOpId> {
        self.witness.as_deref()
    }

    pub fn register(
        &self,
        hub: &Arc<SetmaxnregHub>,
    ) -> Result<SetmaxnregRegistration, SynchronizationError> {
        if self.context.topology() != hub.topology {
            return Err(protocol_error(
                SetmaxnregErrorKind::ContextMismatch,
                self.resource,
                "setmaxnreg plan belongs to a different launch topology",
                None,
                self.witness.as_deref().cloned(),
            ));
        }
        let contract = ParticipantContract::warpgroup(self.context, SETMAXNREG_WARPS_PER_GROUP)?;
        let key = OccurrenceKey::new(
            self.resource.ordinal,
            "setmaxnreg",
            std::iter::empty::<i64>(),
            contract.scope().clone(),
        );
        let contribution = SetmaxnregContribution {
            resource: self.resource,
            context: self.context,
            action: self.action,
            count: self.count,
            witness: self.witness.clone(),
        };
        let wait =
            hub.collective
                .collect(key, contract, self.context.global_warp_id(), contribution)?;
        Ok(SetmaxnregRegistration {
            plan: self.clone(),
            hub: Arc::clone(hub),
            wait,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SetmaxnregContribution {
    resource: SetmaxnregResource,
    context: WarpContext,
    action: SetmaxnregAction,
    count: i64,
    witness: Option<Box<DynamicOpId>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetmaxnregReleaseProvenance {
    release_resource: SetmaxnregResource,
    release_operations: Box<[DynamicOpId]>,
}

impl SetmaxnregReleaseProvenance {
    pub(crate) fn new(
        release_resource: SetmaxnregResource,
        release_operations: impl IntoIterator<Item = DynamicOpId>,
    ) -> Self {
        Self {
            release_resource,
            release_operations: release_operations
                .into_iter()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    pub const fn release_resource(&self) -> SetmaxnregResource {
        self.release_resource
    }

    pub fn release_operations(&self) -> &[DynamicOpId] {
        &self.release_operations
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetmaxnregAvailabilityProvenance {
    releases: Box<[SetmaxnregReleaseProvenance]>,
}

impl SetmaxnregAvailabilityProvenance {
    pub(crate) fn from_releases(
        releases: impl IntoIterator<Item = SetmaxnregReleaseProvenance>,
    ) -> Self {
        Self {
            releases: releases.into_iter().collect::<Vec<_>>().into_boxed_slice(),
        }
    }

    pub fn releases(&self) -> &[SetmaxnregReleaseProvenance] {
        &self.releases
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SetmaxnregCompletionActionId(u64);

impl SetmaxnregCompletionActionId {
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Derive the scheduler identity of one pending grant from its resource
    /// tuple. Like the async-group derivation this is a pure function, so a
    /// waiter can recompute the ID without a side table.
    pub(crate) fn from_resource(
        resource: SetmaxnregResource,
    ) -> Result<Self, SynchronizationError> {
        let launch = pair(
            resource.kernel_index as u128,
            resource.global_cta_id as u128,
        );
        let request = pair(resource.warpgroup_id as u128, resource.ordinal as u128);
        let identity = launch.and_then(|launch| request.and_then(|request| pair(launch, request)));
        let Some(id) =
            identity.and_then(|identity| CompletionActionNamespace::Setmaxnreg.tag(identity))
        else {
            return Err(SynchronizationError::CompletionSourceOperationFailed {
                source_name: "setmaxnreg",
                details: format!("completion identity overflow for {resource}"),
            });
        };
        Ok(Self(id))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetmaxnregBudgetDisposition {
    DecreaseApplied,
    IncreaseImmediate,
    IncreasePending {
        action_id: SetmaxnregCompletionActionId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetmaxnregCompletionAction {
    id: SetmaxnregCompletionActionId,
    resource: SetmaxnregResource,
    target_count: i64,
    required_count: i64,
    available_count_before: i64,
    availability_provenance: SetmaxnregAvailabilityProvenance,
}

impl SetmaxnregCompletionAction {
    pub const fn id(&self) -> SetmaxnregCompletionActionId {
        self.id
    }

    pub const fn resource(&self) -> SetmaxnregResource {
        self.resource
    }

    pub const fn target_count(&self) -> i64 {
        self.target_count
    }

    pub const fn required_count(&self) -> i64 {
        self.required_count
    }

    pub const fn available_count_before(&self) -> i64 {
        self.available_count_before
    }

    pub const fn availability_provenance(&self) -> &SetmaxnregAvailabilityProvenance {
        &self.availability_provenance
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetmaxnregCompletionOutcome {
    action: SetmaxnregCompletionAction,
    woken_warp_ids: Box<[usize]>,
}

impl SetmaxnregCompletionOutcome {
    pub const fn action(&self) -> &SetmaxnregCompletionAction {
        &self.action
    }

    pub fn woken_warp_ids(&self) -> &[usize] {
        &self.woken_warp_ids
    }

    pub fn progress(&self) -> CompletionProgress {
        CompletionProgress {
            completed_operations: 1,
            woken_warps: self.woken_warp_ids.len(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetmaxnregOutcome {
    resource: SetmaxnregResource,
    action: SetmaxnregAction,
    count: i64,
    current_count_before: i64,
    available_count_before: i64,
    budget_disposition: SetmaxnregBudgetDisposition,
    availability_provenance: Option<SetmaxnregAvailabilityProvenance>,
}

impl SetmaxnregOutcome {
    pub const fn resource(&self) -> SetmaxnregResource {
        self.resource
    }

    pub const fn action(&self) -> SetmaxnregAction {
        self.action
    }

    pub const fn count(&self) -> i64 {
        self.count
    }

    pub const fn current_count_before(&self) -> i64 {
        self.current_count_before
    }

    pub const fn available_count_before(&self) -> i64 {
        self.available_count_before
    }

    pub const fn budget_disposition(&self) -> SetmaxnregBudgetDisposition {
        self.budget_disposition
    }

    pub const fn availability_provenance(&self) -> Option<&SetmaxnregAvailabilityProvenance> {
        self.availability_provenance.as_ref()
    }
}

type SetmaxnregCollective = CollectiveHub<SetmaxnregContribution, SetmaxnregOutcome>;
type SetmaxnregCollectiveWait = CollectiveWait<SetmaxnregContribution, SetmaxnregOutcome>;

struct SetmaxnregWarpgroupState {
    current_count: i64,
    completed_syncs: BTreeSet<(NamedBarrierId, u64)>,
    pending_sync_participants: BTreeMap<(NamedBarrierId, u64), BTreeSet<usize>>,
    sync_epoch: u64,
    last_setmax_sync_epoch: Option<u64>,
    last_completed_ordinal: Option<u64>,
}

impl SetmaxnregWarpgroupState {
    fn new(current_count: i64) -> Self {
        Self {
            current_count,
            completed_syncs: BTreeSet::new(),
            pending_sync_participants: BTreeMap::new(),
            sync_epoch: 0,
            last_setmax_sync_epoch: None,
            last_completed_ordinal: None,
        }
    }
}

struct PendingSetmaxnregIncrease {
    action_id: SetmaxnregCompletionActionId,
    target_count: i64,
    required_count: i64,
    key: OccurrenceKey,
    contract: ParticipantContract,
    waiters: BTreeMap<usize, Waker>,
    waiter_operations: BTreeMap<usize, DynamicOpId>,
}

enum SetmaxnregBudgetOperation {
    Pending(PendingSetmaxnregIncrease),
    Completed {
        availability_provenance: Option<SetmaxnregAvailabilityProvenance>,
    },
}

struct SetmaxnregCtaState {
    available_count: i64,
    unattributed_available_count: i64,
    release_lots: VecDeque<AvailableSetmaxnregRelease>,
    operations: BTreeMap<SetmaxnregResource, SetmaxnregBudgetOperation>,
}

struct AvailableSetmaxnregRelease {
    remaining_count: i64,
    provenance: SetmaxnregReleaseProvenance,
}

struct SetmaxnregState {
    default_count: i64,
    calling_initial_count: Option<i64>,
    next_ordinal_by_warp: Vec<u64>,
    warpgroups: BTreeMap<(usize, usize), SetmaxnregWarpgroupState>,
    ctas: BTreeMap<usize, SetmaxnregCtaState>,
}

pub struct SetmaxnregHub {
    topology: LaunchTopology,
    state: Arc<Mutex<SetmaxnregState>>,
    collective: Arc<SetmaxnregCollective>,
}

impl SetmaxnregHub {
    pub fn new(topology: LaunchTopology) -> Self {
        let warpgroup_count = topology
            .warps_per_cta()
            .div_ceil(SETMAXNREG_WARPS_PER_GROUP);
        let default_count = setmaxnreg_default_register_count(warpgroup_count);
        // The CTA pool contains registers released by `setmaxnreg.dec`; the
        // register-file remainder left by allocation granularity is not part
        // of the launch allocation and cannot satisfy an increase.
        let initial_available = 0;
        let warpgroups = (0..topology.cta_count())
            .flat_map(|global_cta_id| {
                (0..warpgroup_count).map(move |warpgroup_id| {
                    (
                        (global_cta_id, warpgroup_id),
                        SetmaxnregWarpgroupState::new(default_count),
                    )
                })
            })
            .collect();
        let ctas = (0..topology.cta_count())
            .map(|global_cta_id| {
                (
                    global_cta_id,
                    SetmaxnregCtaState {
                        available_count: initial_available,
                        unattributed_available_count: initial_available,
                        release_lots: VecDeque::new(),
                        operations: BTreeMap::new(),
                    },
                )
            })
            .collect();
        let state = Arc::new(Mutex::new(SetmaxnregState {
            default_count,
            calling_initial_count: None,
            next_ordinal_by_warp: vec![0; topology.warp_count()],
            warpgroups,
            ctas,
        }));
        let publisher_state = Arc::clone(&state);
        let collective = Arc::new(CollectiveHub::new(move |contributions| {
            publish_setmaxnreg(&publisher_state, contributions)
        }));
        Self {
            topology,
            state,
            collective,
        }
    }

    /// Install the source/compiler-derived initial count before the first
    /// executed setmaxnreg operation. The generated operation supplies this
    /// fact because it is not derivable from launch topology or instruction
    /// operands alone.
    pub(crate) fn configure_calling_initial_count(&self, count: i64) -> Result<(), EngineError> {
        if !(SETMAXNREG_MIN_COUNT..=SETMAXNREG_MAX_COUNT).contains(&count)
            || count % SETMAXNREG_COUNT_GRANULARITY != 0
        {
            return Err(EngineError::message(format!(
                "setmaxnreg caller initial count must satisfy the PTX count contract, got {count}"
            )));
        }
        let mut state = self.state.lock().expect("setmaxnreg hub mutex poisoned");
        if count > state.default_count {
            return Err(EngineError::message(format!(
                "setmaxnreg caller initial count {count} exceeds topology default {}",
                state.default_count
            )));
        }
        if let Some(configured) = state.calling_initial_count {
            return if configured == count {
                Ok(())
            } else {
                Err(EngineError::message(format!(
                    "setmaxnreg caller initial count changed from {configured} to {count}"
                )))
            };
        }
        if state
            .next_ordinal_by_warp
            .iter()
            .any(|ordinal| *ordinal != 0)
            || state
                .ctas
                .values()
                .any(|cta| !cta.operations.is_empty() || !cta.release_lots.is_empty())
        {
            return Err(EngineError::message(
                "setmaxnreg caller initial count arrived after protocol execution began",
            ));
        }

        // Runtime pool transitions use the common compiler/source launch
        // count. At quiescence, WGs with no executed call are still charged
        // the exact topology default rather than this optimistic caller count.
        for warpgroup in state.warpgroups.values_mut() {
            warpgroup.current_count = count;
        }
        for cta in state.ctas.values_mut() {
            cta.available_count = 0;
            cta.unattributed_available_count = 0;
        }
        state.calling_initial_count = Some(count);
        Ok(())
    }

    pub const fn topology(&self) -> LaunchTopology {
        self.topology
    }

    pub fn plan(
        &self,
        kernel_index: usize,
        operation: Option<&OperationContext>,
        context: WarpContext,
        action: SetmaxnregAction,
        count: i64,
    ) -> Result<SetmaxnregPlan, SynchronizationError> {
        let warpgroup_id = context.warp_id_in_cta() / SETMAXNREG_WARPS_PER_GROUP;
        let provisional =
            SetmaxnregResource::new(kernel_index, context.global_cta_id(), warpgroup_id, 0);
        if context.topology() != self.topology
            || operation.is_some_and(|operation| {
                operation.id().kernel_index() != kernel_index
                    || operation.id().global_warp_id() != context.global_warp_id()
                    || operation.kind() != OperationKind::Collective
                    || operation.active_mask() != context.active_mask()
            })
        {
            return Err(protocol_error(
                SetmaxnregErrorKind::ContextMismatch,
                provisional,
                "setmaxnreg operation and warp context do not describe the same collective",
                None,
                operation.map(|operation| operation.id().clone()),
            ));
        }
        if context.active_mask() != WarpMask::FULL {
            return Err(protocol_error(
                SetmaxnregErrorKind::PartialWarpParticipation,
                provisional,
                format!(
                    "setmaxnreg.sync requires all 32 lanes, got mask 0x{:08x}",
                    context.active_mask().bits()
                ),
                None,
                operation.map(|operation| operation.id().clone()),
            ));
        }
        if !(SETMAXNREG_MIN_COUNT..=SETMAXNREG_MAX_COUNT).contains(&count)
            || count % SETMAXNREG_COUNT_GRANULARITY != 0
        {
            return Err(protocol_error(
                SetmaxnregErrorKind::InvalidCount,
                provisional,
                format!(
                    "setmaxnreg register count must be in {SETMAXNREG_MIN_COUNT}..={SETMAXNREG_MAX_COUNT} and a multiple of {SETMAXNREG_COUNT_GRANULARITY}, got {count}"
                ),
                None,
                operation.map(|operation| operation.id().clone()),
            ));
        }
        let local_first = warpgroup_id * SETMAXNREG_WARPS_PER_GROUP;
        if local_first + SETMAXNREG_WARPS_PER_GROUP > self.topology.warps_per_cta() {
            return Err(protocol_error(
                SetmaxnregErrorKind::IncompleteWarpgroup,
                provisional,
                format!(
                    "setmaxnreg requires four warps in warpgroup {warpgroup_id}, but the CTA has {} warps",
                    self.topology.warps_per_cta()
                ),
                None,
                operation.map(|operation| operation.id().clone()),
            ));
        }

        let ordinal = {
            let mut state = self.state.lock().expect("setmaxnreg hub mutex poisoned");
            let next = &mut state.next_ordinal_by_warp[context.global_warp_id()];
            let ordinal = *next;
            *next = next.checked_add(1).ok_or_else(|| {
                protocol_error(
                    SetmaxnregErrorKind::SequenceOverflow,
                    SetmaxnregResource::new(
                        kernel_index,
                        context.global_cta_id(),
                        warpgroup_id,
                        ordinal,
                    ),
                    "setmaxnreg dynamic sequence ordinal overflow",
                    None,
                    operation.map(|operation| operation.id().clone()),
                )
            })?;
            ordinal
        };
        Ok(SetmaxnregPlan {
            resource: SetmaxnregResource::new(
                kernel_index,
                context.global_cta_id(),
                warpgroup_id,
                ordinal,
            ),
            context,
            action,
            count,
            witness: operation.map(|operation| Box::new(operation.id().clone())),
        })
    }

    /// Record one completed explicit warpgroup synchronization. A named
    /// barrier generation may complete with warps from different warpgroups,
    /// so it credits this warpgroup only after its exact four full warps report
    /// the same generation.
    pub fn record_warpgroup_sync(
        &self,
        context: WarpContext,
        barrier_id: NamedBarrierId,
        generation: u64,
        arrival_mask: WarpMask,
    ) -> Result<bool, SynchronizationError> {
        let warpgroup_id = context.warp_id_in_cta() / SETMAXNREG_WARPS_PER_GROUP;
        let resource = SetmaxnregResource::new(0, context.global_cta_id(), warpgroup_id, 0);
        if context.topology() != self.topology
            || barrier_id.global_cta_id() != context.global_cta_id()
        {
            return Err(protocol_error(
                SetmaxnregErrorKind::ContextMismatch,
                resource,
                "warpgroup synchronization belongs to a different CTA or launch topology",
                None,
                None,
            ));
        }
        if arrival_mask != WarpMask::FULL {
            return Ok(false);
        }
        let contract = ParticipantContract::warpgroup(context, SETMAXNREG_WARPS_PER_GROUP)?;
        let expected = contract.participants().iter().collect::<BTreeSet<_>>();
        let mut state = self.state.lock().expect("setmaxnreg hub mutex poisoned");
        let warpgroup = state
            .warpgroups
            .get_mut(&(context.global_cta_id(), warpgroup_id))
            .expect("launch initialization creates every CTA warpgroup");
        let sync = (barrier_id, generation);
        if warpgroup.completed_syncs.contains(&sync) {
            return Ok(true);
        }
        let participants = warpgroup.pending_sync_participants.entry(sync).or_default();
        participants.insert(context.global_warp_id());
        let completed = *participants == expected;
        if completed {
            warpgroup.pending_sync_participants.remove(&sync);
            warpgroup.completed_syncs.insert(sync);
            warpgroup.sync_epoch = warpgroup.sync_epoch.checked_add(1).ok_or_else(|| {
                protocol_error(
                    SetmaxnregErrorKind::SequenceOverflow,
                    resource,
                    "setmaxnreg warpgroup synchronization epoch overflow",
                    None,
                    None,
                )
            })?;
        }
        Ok(completed)
    }

    /// Return every individually grantable blocked increase. Ordinary execution pumps
    /// the lowest stable action ID.
    pub fn pending_completion_actions(&self) -> Vec<SetmaxnregCompletionAction> {
        let state = self.state.lock().expect("setmaxnreg hub mutex poisoned");
        let mut actions = state
            .ctas
            .values()
            .flat_map(|cta| {
                cta.operations
                    .iter()
                    .filter_map(move |(resource, operation)| {
                        let SetmaxnregBudgetOperation::Pending(pending) = operation else {
                            return None;
                        };
                        (pending.required_count <= cta.available_count).then(|| {
                            let availability_provenance =
                                preview_available_provenance(cta, pending.required_count).expect(
                                    "a formerly blocked increase is enabled by a register release",
                                );
                            SetmaxnregCompletionAction {
                                id: pending.action_id,
                                resource: *resource,
                                target_count: pending.target_count,
                                required_count: pending.required_count,
                                available_count_before: cta.available_count,
                                availability_provenance,
                            }
                        })
                    })
            })
            .collect::<Vec<_>>();
        actions.sort_by_key(SetmaxnregCompletionAction::id);
        actions
    }

    pub fn pending_completion_action_ids(&self) -> Vec<SetmaxnregCompletionActionId> {
        self.pending_completion_actions()
            .into_iter()
            .map(|action| action.id())
            .collect()
    }

    /// Advance only the warpgroup collective registration phase.
    ///
    /// Analysis completion sources need to wrap each register-pool grant with
    /// mode callbacks, so they cannot call the generic `CompletionSource`
    /// implementation that may also apply a grant directly.
    pub(crate) fn pump_collective_only(&self) -> Result<CompletionProgress, SynchronizationError> {
        self.collective.pump()
    }

    pub fn apply_completion_detailed(
        &self,
        action_id: SetmaxnregCompletionActionId,
    ) -> Result<SetmaxnregCompletionOutcome, SynchronizationError> {
        self.apply_completion_detailed_with_outcome(action_id, |_| Ok(()))
    }

    /// Apply one register-pool grant, publish its mode-visible outcome, and
    /// only then wake the warps whose blocked increase became complete.
    pub fn apply_completion_detailed_with_outcome(
        &self,
        action_id: SetmaxnregCompletionActionId,
        publish_before_wake: impl FnOnce(
            &SetmaxnregCompletionOutcome,
        ) -> Result<(), SynchronizationError>,
    ) -> Result<SetmaxnregCompletionOutcome, SynchronizationError> {
        let mut state = self.state.lock().expect("setmaxnreg hub mutex poisoned");
        let matches = state
            .ctas
            .iter()
            .flat_map(|(&global_cta_id, cta)| {
                cta.operations
                    .iter()
                    .filter_map(move |(&resource, operation)| {
                        matches!(
                            operation,
                            SetmaxnregBudgetOperation::Pending(pending)
                                if pending.action_id == action_id
                        )
                        .then_some((global_cta_id, resource))
                    })
            })
            .collect::<Vec<_>>();
        let [(global_cta_id, resource)] = matches.as_slice() else {
            return Err(SynchronizationError::CompletionSourceOperationFailed {
                source_name: "setmaxnreg",
                details: format!(
                    "completion action {} is not a unique pending request",
                    action_id.get(),
                ),
            });
        };
        let global_cta_id = *global_cta_id;
        let resource = *resource;
        let pending = {
            let cta = state
                .ctas
                .get_mut(&global_cta_id)
                .expect("selected CTA register pool remains present");
            let (pending_action_id, target_count, required_count) = match cta
                .operations
                .get(&resource)
                .expect("selected setmaxnreg request remains present")
            {
                SetmaxnregBudgetOperation::Pending(pending) => (
                    pending.action_id,
                    pending.target_count,
                    pending.required_count,
                ),
                SetmaxnregBudgetOperation::Completed { .. } => {
                    unreachable!("selected setmaxnreg request must remain pending")
                }
            };
            if required_count > cta.available_count {
                return Err(SynchronizationError::CompletionSourceOperationFailed {
                    source_name: "setmaxnreg",
                    details: format!(
                        "completion action {} requires {} registers but CTA {} has {}",
                        action_id.get(),
                        required_count,
                        global_cta_id,
                        cta.available_count,
                    ),
                });
            }
            let availability_provenance = preview_available_provenance(cta, required_count)
                .ok_or_else(|| SynchronizationError::CompletionSourceOperationFailed {
                    source_name: "setmaxnreg",
                    details: format!(
                        "completion action {} has no register-release provenance",
                        action_id.get(),
                    ),
                })?;
            let action = SetmaxnregCompletionAction {
                id: pending_action_id,
                resource,
                target_count,
                required_count,
                available_count_before: cta.available_count,
                availability_provenance: availability_provenance.clone(),
            };
            let operation = cta
                .operations
                .get_mut(&resource)
                .expect("selected setmaxnreg request remains present");
            let SetmaxnregBudgetOperation::Pending(pending) = std::mem::replace(
                operation,
                SetmaxnregBudgetOperation::Completed {
                    availability_provenance: Some(availability_provenance),
                },
            ) else {
                unreachable!("selected setmaxnreg request must remain pending");
            };
            let consumed_provenance = consume_available_registers(cta, required_count)
                .expect("enabled blocked increase consumes release-backed capacity");
            debug_assert_eq!(consumed_provenance, action.availability_provenance);
            (action, pending)
        };
        let (action, pending) = pending;
        state
            .warpgroups
            .get_mut(&(resource.global_cta_id, resource.warpgroup_id))
            .expect("launch initialization creates every CTA warpgroup")
            .current_count = pending.target_count;
        let woken_warp_ids = pending.waiters.keys().copied().collect::<Vec<_>>();
        let wakers = pending.waiters.into_values().collect::<Vec<_>>();
        drop(state);
        let outcome = SetmaxnregCompletionOutcome {
            action,
            woken_warp_ids: woken_warp_ids.into_boxed_slice(),
        };
        publish_before_wake(&outcome)?;
        for waker in wakers {
            waker.wake();
        }
        Ok(outcome)
    }
}

pub(crate) fn setmaxnreg_default_register_count(warpgroup_count: usize) -> i64 {
    let count = SETMAXNREG_CTA_REGISTER_POOL
        / i64::try_from(warpgroup_count).expect("warpgroup count fits i64");
    count / SETMAXNREG_COUNT_GRANULARITY * SETMAXNREG_COUNT_GRANULARITY
}

fn publish_setmaxnreg(
    state: &Arc<Mutex<SetmaxnregState>>,
    contributions: BTreeMap<usize, SetmaxnregContribution>,
) -> Result<SetmaxnregOutcome, SynchronizationError> {
    let mut values = contributions.into_values();
    let first =
        values
            .next()
            .ok_or_else(|| SynchronizationError::CompletionSourceOperationFailed {
                source_name: "setmaxnreg",
                details: "setmaxnreg collective has no contributions".to_string(),
            })?;
    let mut participant_operations = first
        .witness
        .iter()
        .map(|witness| witness.as_ref().clone())
        .collect::<Vec<_>>();
    for contribution in values {
        if contribution.resource != first.resource
            || contribution.action != first.action
            || contribution.count != first.count
        {
            return Err(protocol_error(
                SetmaxnregErrorKind::Divergence,
                first.resource,
                format!(
                    "warpgroup disagrees at dynamic ordinal {}: prior {} {} registers, current {} {} registers",
                    first.resource.ordinal,
                    first.action.label(),
                    first.count,
                    contribution.action.label(),
                    contribution.count,
                ),
                first.witness.as_deref().cloned(),
                contribution.witness.as_deref().cloned(),
            ));
        }
        participant_operations.extend(
            contribution
                .witness
                .iter()
                .map(|witness| witness.as_ref().clone()),
        );
    }

    let mut state = state.lock().expect("setmaxnreg hub mutex poisoned");
    let warpgroup_key = (first.resource.global_cta_id, first.resource.warpgroup_id);
    let current_count = {
        let warpgroup = state
            .warpgroups
            .get(&warpgroup_key)
            .expect("launch initialization creates every CTA warpgroup");
        match warpgroup.last_completed_ordinal {
            None if first.resource.ordinal != 0 => {
                return Err(protocol_error(
                    SetmaxnregErrorKind::SequenceMismatch,
                    first.resource,
                    format!(
                        "first completed setmaxnreg ordinal is {}, expected 0",
                        first.resource.ordinal
                    ),
                    None,
                    first.witness.as_deref().cloned(),
                ));
            }
            Some(previous) => {
                let expected = previous.checked_add(1).ok_or_else(|| {
                    protocol_error(
                        SetmaxnregErrorKind::SequenceOverflow,
                        first.resource,
                        "setmaxnreg completed ordinal overflow",
                        None,
                        first.witness.as_deref().cloned(),
                    )
                })?;
                if first.resource.ordinal != expected {
                    return Err(protocol_error(
                        SetmaxnregErrorKind::SequenceMismatch,
                        first.resource,
                        format!(
                            "setmaxnreg completed ordinal {}, expected {expected}",
                            first.resource.ordinal
                        ),
                        None,
                        first.witness.as_deref().cloned(),
                    ));
                }
                if warpgroup.last_setmax_sync_epoch == Some(warpgroup.sync_epoch) {
                    return Err(protocol_error(
                        SetmaxnregErrorKind::MissingWarpgroupSync,
                        first.resource,
                        "all warps in the warpgroup must synchronize explicitly before a subsequent setmaxnreg instruction",
                        None,
                        first.witness.as_deref().cloned(),
                    ));
                }
            }
            None => {}
        }
        warpgroup.current_count
    };

    let direction_is_valid = match first.action {
        SetmaxnregAction::Increase => first.count >= current_count,
        SetmaxnregAction::Decrease => first.count <= current_count,
    };
    if !direction_is_valid {
        return Err(protocol_error(
            SetmaxnregErrorKind::InvalidDirection,
            first.resource,
            format!(
                "{} target {} is invalid for the current register count {current_count}",
                first.action.label(),
                first.count,
            ),
            None,
            first.witness.as_deref().cloned(),
        ));
    }

    {
        let warpgroup = state
            .warpgroups
            .get_mut(&warpgroup_key)
            .expect("launch initialization creates every CTA warpgroup");
        warpgroup.last_completed_ordinal = Some(first.resource.ordinal);
        warpgroup.last_setmax_sync_epoch = Some(warpgroup.sync_epoch);
    }

    let contract = ParticipantContract::warpgroup(first.context, SETMAXNREG_WARPS_PER_GROUP)?;
    let key = OccurrenceKey::new(
        first.resource.ordinal,
        "setmaxnreg.pool",
        std::iter::empty::<i64>(),
        contract.scope().clone(),
    );
    let available_count_before = state
        .ctas
        .get(&first.resource.global_cta_id)
        .expect("launch initialization creates every CTA register pool")
        .available_count;
    let (availability_provenance, budget_disposition) = match first.action {
        SetmaxnregAction::Decrease => {
            let released_count = current_count - first.count;
            state
                .warpgroups
                .get_mut(&warpgroup_key)
                .expect("launch initialization creates every CTA warpgroup")
                .current_count = first.count;
            let cta = state
                .ctas
                .get_mut(&first.resource.global_cta_id)
                .expect("launch initialization creates every CTA register pool");
            cta.available_count += released_count;
            if released_count != 0 {
                cta.release_lots.push_back(AvailableSetmaxnregRelease {
                    remaining_count: released_count,
                    provenance: SetmaxnregReleaseProvenance::new(
                        first.resource,
                        participant_operations,
                    ),
                });
            }
            cta.operations.insert(
                first.resource,
                SetmaxnregBudgetOperation::Completed {
                    availability_provenance: None,
                },
            );
            (None, SetmaxnregBudgetDisposition::DecreaseApplied)
        }
        SetmaxnregAction::Increase => {
            let required_count = first.count - current_count;
            let cta = state
                .ctas
                .get_mut(&first.resource.global_cta_id)
                .expect("launch initialization creates every CTA register pool");
            if required_count <= cta.available_count {
                let availability_provenance = consume_available_registers(cta, required_count);
                cta.operations.insert(
                    first.resource,
                    SetmaxnregBudgetOperation::Completed {
                        availability_provenance: availability_provenance.clone(),
                    },
                );
                state
                    .warpgroups
                    .get_mut(&warpgroup_key)
                    .expect("launch initialization creates every CTA warpgroup")
                    .current_count = first.count;
                (
                    availability_provenance,
                    SetmaxnregBudgetDisposition::IncreaseImmediate,
                )
            } else {
                let action_id = SetmaxnregCompletionActionId::from_resource(first.resource)?;
                cta.operations.insert(
                    first.resource,
                    SetmaxnregBudgetOperation::Pending(PendingSetmaxnregIncrease {
                        action_id,
                        target_count: first.count,
                        required_count,
                        key,
                        contract,
                        waiters: BTreeMap::new(),
                        waiter_operations: BTreeMap::new(),
                    }),
                );
                (
                    None,
                    SetmaxnregBudgetDisposition::IncreasePending { action_id },
                )
            }
        }
    };
    drop(state);

    Ok(SetmaxnregOutcome {
        resource: first.resource,
        action: first.action,
        count: first.count,
        current_count_before: current_count,
        available_count_before,
        budget_disposition,
        availability_provenance,
    })
}

fn preview_available_provenance(
    cta: &SetmaxnregCtaState,
    count: i64,
) -> Option<SetmaxnregAvailabilityProvenance> {
    debug_assert!(count <= cta.available_count);
    let mut remaining = count.saturating_sub(cta.unattributed_available_count);
    let mut releases = Vec::new();
    for lot in &cta.release_lots {
        if remaining == 0 {
            break;
        }
        let consumed = remaining.min(lot.remaining_count);
        if consumed != 0 {
            releases.push(lot.provenance.clone());
            remaining -= consumed;
        }
    }
    debug_assert_eq!(remaining, 0);
    (!releases.is_empty()).then(|| SetmaxnregAvailabilityProvenance::from_releases(releases))
}

fn consume_available_registers(
    cta: &mut SetmaxnregCtaState,
    count: i64,
) -> Option<SetmaxnregAvailabilityProvenance> {
    debug_assert!(count <= cta.available_count);
    cta.available_count -= count;
    let unattributed = count.min(cta.unattributed_available_count);
    cta.unattributed_available_count -= unattributed;
    let mut remaining = count - unattributed;
    let mut releases = Vec::new();
    while remaining != 0 {
        let lot = cta
            .release_lots
            .front_mut()
            .expect("available release-backed registers retain a provenance lot");
        let consumed = remaining.min(lot.remaining_count);
        releases.push(lot.provenance.clone());
        lot.remaining_count -= consumed;
        remaining -= consumed;
        if lot.remaining_count == 0 {
            cta.release_lots.pop_front();
        }
    }
    (!releases.is_empty()).then(|| SetmaxnregAvailabilityProvenance::from_releases(releases))
}

pub struct SetmaxnregRegistration {
    plan: SetmaxnregPlan,
    hub: Arc<SetmaxnregHub>,
    wait: SetmaxnregCollectiveWait,
}

impl SetmaxnregRegistration {
    pub const fn plan(&self) -> &SetmaxnregPlan {
        &self.plan
    }

    pub async fn resume(self) -> Result<SetmaxnregResumePlan, SynchronizationError> {
        let Self { plan, hub, wait } = self;
        let mut outcome = wait.await?.as_ref().clone();
        outcome.availability_provenance = SetmaxnregBudgetWait::new(
            hub,
            outcome.resource(),
            plan.context().global_warp_id(),
            plan.witness().cloned(),
        )
        .await?;
        Ok(SetmaxnregResumePlan { plan, outcome })
    }
}

struct SetmaxnregBudgetWait {
    hub: Arc<SetmaxnregHub>,
    resource: SetmaxnregResource,
    warp_id: usize,
    operation: Option<DynamicOpId>,
    registered: bool,
    finished: bool,
}

impl SetmaxnregBudgetWait {
    fn new(
        hub: Arc<SetmaxnregHub>,
        resource: SetmaxnregResource,
        warp_id: usize,
        operation: Option<DynamicOpId>,
    ) -> Self {
        Self {
            hub,
            resource,
            warp_id,
            operation,
            registered: false,
            finished: false,
        }
    }
}

impl Future for SetmaxnregBudgetWait {
    type Output = Result<Option<SetmaxnregAvailabilityProvenance>, SynchronizationError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut state = this
            .hub
            .state
            .lock()
            .expect("setmaxnreg hub mutex poisoned");
        let Some(operation) = state
            .ctas
            .get_mut(&this.resource.global_cta_id)
            .and_then(|cta| cta.operations.get_mut(&this.resource))
        else {
            this.finished = true;
            return Poll::Ready(Err(SynchronizationError::CompletionSourceOperationFailed {
                source_name: "setmaxnreg",
                details: format!("missing CTA register-pool request for {}", this.resource),
            }));
        };
        match operation {
            SetmaxnregBudgetOperation::Completed {
                availability_provenance,
            } => {
                this.registered = false;
                this.finished = true;
                Poll::Ready(Ok(availability_provenance.clone()))
            }
            SetmaxnregBudgetOperation::Pending(pending) => {
                match pending.waiters.get_mut(&this.warp_id) {
                    Some(waker) if this.registered => {
                        if !waker.will_wake(context.waker()) {
                            *waker = context.waker().clone();
                        }
                    }
                    Some(_) => {
                        this.finished = true;
                        return Poll::Ready(Err(SynchronizationError::DuplicateWaiter {
                            key: pending.key.clone(),
                            phase: None,
                            warp_id: this.warp_id,
                        }));
                    }
                    None => {
                        pending
                            .waiters
                            .insert(this.warp_id, context.waker().clone());
                        if let Some(operation) = &this.operation {
                            pending
                                .waiter_operations
                                .insert(this.warp_id, operation.clone());
                        }
                    }
                }
                this.registered = true;
                Poll::Pending
            }
        }
    }
}

impl Drop for SetmaxnregBudgetWait {
    fn drop(&mut self) {
        if !self.registered || self.finished {
            return;
        }
        let mut state = self
            .hub
            .state
            .lock()
            .expect("setmaxnreg hub mutex poisoned");
        if let Some(SetmaxnregBudgetOperation::Pending(pending)) = state
            .ctas
            .get_mut(&self.resource.global_cta_id)
            .and_then(|cta| cta.operations.get_mut(&self.resource))
        {
            pending.waiters.remove(&self.warp_id);
            pending.waiter_operations.remove(&self.warp_id);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetmaxnregResumePlan {
    plan: SetmaxnregPlan,
    outcome: SetmaxnregOutcome,
}

impl SetmaxnregResumePlan {
    pub const fn plan(&self) -> &SetmaxnregPlan {
        &self.plan
    }

    pub const fn outcome(&self) -> &SetmaxnregOutcome {
        &self.outcome
    }
}

impl CompletionSource for SetmaxnregHub {
    fn source_name(&self) -> &'static str {
        "setmaxnreg"
    }

    fn pump(&self) -> Result<CompletionProgress, SynchronizationError> {
        let collective = self.pump_collective_only()?;
        if collective.made_progress() {
            return Ok(collective);
        }
        let Some(action) = self.pending_completion_actions().into_iter().next() else {
            return Ok(CompletionProgress::default());
        };
        self.apply_completion_detailed(action.id())
            .map(|outcome| outcome.progress())
    }

    fn blocked_operations(&self) -> Vec<BlockedOperation> {
        let mut blocked = self
            .collective
            .blocked_operations()
            .into_iter()
            .map(|blocked| blocked.with_awaited_operation(crate::AwaitedOperation::Setmaxnreg))
            .collect::<Vec<_>>();
        let state = self.state.lock().expect("setmaxnreg hub mutex poisoned");
        for cta in state.ctas.values() {
            for operation in cta.operations.values() {
                let SetmaxnregBudgetOperation::Pending(pending) = operation else {
                    continue;
                };
                let arrived = pending
                    .contract
                    .participants()
                    .iter()
                    .collect::<BTreeSet<_>>();
                let participant_state =
                    crate::ParticipantState::new(&pending.contract, &arrived, None, None);
                for warp_id in pending.waiters.keys().copied() {
                    blocked.push(
                        BlockedOperation::new(
                            warp_id,
                            crate::AwaitedOperation::SetmaxnregPool,
                            pending.key.clone(),
                            None,
                            participant_state.clone(),
                        )
                        .with_operation(pending.waiter_operations.get(&warp_id).cloned()),
                    );
                }
            }
        }
        blocked.sort_by(|lhs, rhs| lhs.diagnostic_cmp(rhs));
        blocked
    }

    fn validate_quiescent(&self) -> Result<(), SynchronizationError> {
        self.collective.validate_quiescent()?;
        let state = self.state.lock().expect("setmaxnreg hub mutex poisoned");
        for (global_cta_id, cta) in &state.ctas {
            for operation in cta.operations.values() {
                if let SetmaxnregBudgetOperation::Pending(pending) = operation {
                    return Err(pending.key.completion_not_quiescent_error(
                        self.source_name(),
                        format_args!(
                            " is waiting for {} registers: current={}, target={}, available={}",
                            pending.required_count,
                            pending.target_count - pending.required_count,
                            pending.target_count,
                            cta.available_count,
                        ),
                    ));
                }
            }
            let per_wg = state
                .warpgroups
                .iter()
                .filter_map(|((cta_id, warpgroup_id), warpgroup)| {
                    (*cta_id == *global_cta_id).then_some((
                        *warpgroup_id,
                        if warpgroup.last_completed_ordinal.is_some() {
                            warpgroup.current_count
                        } else {
                            state.default_count
                        },
                    ))
                })
                .collect::<BTreeMap<_, _>>();
            let launch_allocation = state
                .warpgroups
                .iter()
                .filter_map(|((cta_id, _warpgroup_id), warpgroup)| {
                    (*cta_id == *global_cta_id).then_some(
                        if warpgroup.last_completed_ordinal.is_some() {
                            state.calling_initial_count.unwrap_or(state.default_count)
                        } else {
                            state.default_count
                        },
                    )
                })
                .sum::<i64>();
            let allocated_count = per_wg.values().sum::<i64>();
            if allocated_count > launch_allocation {
                let resource = cta
                    .operations
                    .keys()
                    .next()
                    .copied()
                    .unwrap_or_else(|| SetmaxnregResource::new(0, *global_cta_id, 0, 0));
                return Err(protocol_error(
                    SetmaxnregErrorKind::RegisterOversubscription,
                    resource,
                    format!(
                        "CTA register allocation {allocated_count} exceeds the {launch_allocation}-register launch allocation: {per_wg:?}",
                    ),
                    None,
                    None,
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "setmaxnreg_tests.rs"]
mod tests;
