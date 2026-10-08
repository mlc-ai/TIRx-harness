use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use crate::completion_action_id::{pair, CompletionActionNamespace};
use crate::memory::publish_deferred_global_writes;
use crate::{
    AsyncTokenId, AwaitedOperation, BlockedOperation, CompletionProgress, CompletionSource,
    DeferredGlobalWrite, EngineError, LaunchTopology, MemoryAccessSemantics, OccurrenceKey,
    OperationContext, ParticipantState, PhysicalAccessBatch, PhysicalAccessKind,
    PhysicalAccessSpace, PhysicalCompletionAction, ProfileKind, ProfileTimer, ScopeInstance,
    SynchronizationError, WarpContext, WarpMask, WARP_SIZE,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AsyncGroupDomain {
    CpAsync,
    Bulk,
    /// Register-to-global release operations commit implicitly at issue.
    /// PTX bulk/classic waits cannot observe this independent completion domain.
    Release,
}

impl AsyncGroupDomain {
    const ALL: [Self; 3] = [Self::CpAsync, Self::Bulk, Self::Release];
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::CpAsync => "cp.async",
            Self::Bulk => "cp.async.bulk",
            Self::Release => "async.release",
        }
    }
}

/// Stable scheduler identity of one per-lane async batch. Arrive-on may
/// schedule a batch before an explicit commit closes its cp.async group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AsyncGroupId {
    global_warp_id: usize,
    lane: u8,
    domain: AsyncGroupDomain,
    ordinal: u64,
}

impl AsyncGroupId {
    pub const fn global_warp_id(self) -> usize {
        self.global_warp_id
    }

    pub const fn lane(self) -> usize {
        self.lane as usize
    }

    pub const fn domain(self) -> AsyncGroupDomain {
        self.domain
    }

    pub const fn ordinal(self) -> u64 {
        self.ordinal
    }
}

/// Scheduler identity for one async-group milestone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AsyncGroupCompletionActionId(u64);

impl AsyncGroupCompletionActionId {
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Derive the scheduler identity of one milestone from the group tuple.
    ///
    /// This is a pure function, not an allocation: a waiter recomputes the ID it
    /// needs instead of consulting a side table, which is why the IDs must not
    /// depend on commit order (see
    /// `tests::completion_action_ids_do_not_depend_on_cross_warp_commit_order`).
    fn from_group(
        group: AsyncGroupId,
        milestone: AsyncGroupMilestone,
    ) -> Result<Self, EngineError> {
        let domain = match group.domain() {
            AsyncGroupDomain::CpAsync => 0_u128,
            AsyncGroupDomain::Bulk => 1_u128,
            AsyncGroupDomain::Release => 2_u128,
        };
        let milestone = match milestone {
            AsyncGroupMilestone::SourceReadComplete => 0_u128,
            AsyncGroupMilestone::FullComplete => 1_u128,
        };
        let lane_domain = pair(group.lane() as u128, domain)
            .and_then(|value| pair(value, milestone))
            .ok_or_else(|| EngineError::message("async-group completion identity overflow"))?;
        let warp_group = pair(group.global_warp_id() as u128, group.ordinal() as u128)
            .and_then(|value| pair(value, lane_domain))
            .ok_or_else(|| EngineError::message("async-group completion identity overflow"))?;
        CompletionActionNamespace::AsyncGroup
            .tag(warp_group)
            .map(Self)
            .ok_or_else(|| {
                EngineError::message("async-group completion identity exceeds scheduler namespace")
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AsyncGroupMilestone {
    SourceReadComplete,
    FullComplete,
}

/// Exact source and destination footprints owned by one async-group token.
/// No-payload operations retain empty footprint slices while using the same
/// issue, commit, completion, and wait lifecycle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsyncGroupIssueEffect {
    token: AsyncTokenId,
    operation: OperationContext,
    domain: AsyncGroupDomain,
    source_accesses: Arc<[PhysicalAccessBatch]>,
    destination_accesses: Arc<[PhysicalAccessBatch]>,
}

impl AsyncGroupIssueEffect {
    pub fn new(
        operation: OperationContext,
        domain: AsyncGroupDomain,
        source_accesses: impl IntoIterator<Item = PhysicalAccessBatch>,
        destination_accesses: impl IntoIterator<Item = PhysicalAccessBatch>,
    ) -> Result<Self, EngineError> {
        Self::new_with_token_ordinal(operation, domain, 0, source_accesses, destination_accesses)
    }

    fn new_with_token_ordinal(
        operation: OperationContext,
        domain: AsyncGroupDomain,
        token_ordinal: u32,
        source_accesses: impl IntoIterator<Item = PhysicalAccessBatch>,
        destination_accesses: impl IntoIterator<Item = PhysicalAccessBatch>,
    ) -> Result<Self, EngineError> {
        let source_accesses = source_accesses
            .into_iter()
            .map(|batch| mark_async_access(domain, batch))
            .collect::<Vec<_>>();
        let destination_accesses = destination_accesses
            .into_iter()
            .map(|batch| mark_async_access(domain, batch))
            .collect::<Vec<_>>();
        validate_issue_accesses(&operation, &source_accesses, |kind| {
            kind == PhysicalAccessKind::Read
        })?;
        validate_issue_accesses(&operation, &destination_accesses, |kind| {
            matches!(
                kind,
                PhysicalAccessKind::Write | PhysicalAccessKind::AtomicReadModifyWrite
            )
        })?;
        Ok(Self {
            token: AsyncTokenId::new(operation.id().clone(), token_ordinal),
            operation,
            domain,
            source_accesses: source_accesses.into(),
            destination_accesses: destination_accesses.into(),
        })
    }

    pub const fn token(&self) -> &AsyncTokenId {
        &self.token
    }

    pub const fn operation(&self) -> &OperationContext {
        &self.operation
    }

    pub const fn domain(&self) -> AsyncGroupDomain {
        self.domain
    }

    pub fn source_accesses(&self) -> &[PhysicalAccessBatch] {
        self.source_accesses.as_ref()
    }

    pub fn destination_accesses(&self) -> &[PhysicalAccessBatch] {
        self.destination_accesses.as_ref()
    }
}

/// One warp instruction containing one independent async-group token for every
/// participating lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsyncGroupIssueBatchEffect {
    operation: OperationContext,
    domain: AsyncGroupDomain,
    members: Arc<[AsyncGroupIssueEffect]>,
}

impl AsyncGroupIssueBatchEffect {
    pub fn new(
        operation: OperationContext,
        domain: AsyncGroupDomain,
        lane_accesses: impl IntoIterator<
            Item = (usize, Vec<PhysicalAccessBatch>, Vec<PhysicalAccessBatch>),
        >,
    ) -> Result<Self, EngineError> {
        let mut members = Vec::new();
        let mut member_mask = WarpMask::EMPTY;
        for (lane, source_accesses, destination_accesses) in lane_accesses {
            let lane_mask = WarpMask::from_lanes([lane])
                .map_err(|error| EngineError::message(error.to_string()))?;
            if !operation.active_mask().contains(lane) {
                return Err(EngineError::message(format!(
                    "async-group batch lane {lane} is not active at {}",
                    operation.id()
                )));
            }
            if member_mask.contains(lane) {
                return Err(EngineError::message(format!(
                    "async-group batch repeats lane {lane} at {}",
                    operation.id()
                )));
            }
            member_mask |= lane_mask;
            let lane_operation = operation.clone().with_active_mask(lane_mask);
            members.push(AsyncGroupIssueEffect::new_with_token_ordinal(
                lane_operation,
                domain,
                u32::try_from(lane).expect("warp lane fits in u32"),
                source_accesses,
                destination_accesses,
            )?);
        }
        if member_mask != operation.active_mask() {
            return Err(EngineError::message(format!(
                "async-group batch lane mask {:#010x} does not match operation mask {:#010x} at {}",
                member_mask.bits(),
                operation.active_mask().bits(),
                operation.id()
            )));
        }
        Ok(Self {
            operation,
            domain,
            members: members.into(),
        })
    }

    pub const fn operation(&self) -> &OperationContext {
        &self.operation
    }

    pub const fn domain(&self) -> AsyncGroupDomain {
        self.domain
    }

    pub fn members(&self) -> &[AsyncGroupIssueEffect] {
        &self.members
    }
}

fn mark_async_access(domain: AsyncGroupDomain, batch: PhysicalAccessBatch) -> PhysicalAccessBatch {
    if matches!(
        batch.descriptor().space(),
        PhysicalAccessSpace::Global | PhysicalAccessSpace::Shared
    ) {
        let semantics = match domain {
            AsyncGroupDomain::CpAsync => MemoryAccessSemantics::async_generic(),
            AsyncGroupDomain::Bulk
                if batch.descriptor().memory_semantics().order().is_strong()
                    && batch.descriptor().memory_semantics().proxy()
                        == crate::MemoryProxy::Async =>
            {
                batch.descriptor().memory_semantics()
            }
            AsyncGroupDomain::Bulk
                if batch.descriptor().space() == PhysicalAccessSpace::Global
                    && batch.descriptor().kind() == PhysicalAccessKind::AtomicReadModifyWrite =>
            {
                batch.descriptor().memory_semantics()
            }
            AsyncGroupDomain::Bulk => MemoryAccessSemantics::async_proxy(),
            AsyncGroupDomain::Release => batch.descriptor().memory_semantics(),
        };
        batch.with_memory_semantics(semantics)
    } else {
        batch
    }
}

fn validate_issue_accesses<F>(
    operation: &OperationContext,
    accesses: &[PhysicalAccessBatch],
    accepts_kind: F,
) -> Result<(), EngineError>
where
    F: Fn(PhysicalAccessKind) -> bool,
{
    for (index, access) in accesses.iter().enumerate() {
        if access.operation() != operation {
            return Err(EngineError::message(format!(
                "async-group access batch {index} belongs to {}, expected {}",
                access.operation().id(),
                operation.id()
            )));
        }
        if !accepts_kind(access.descriptor().kind()) {
            return Err(EngineError::message(format!(
                "async-group access batch {index} has invalid {} access kind",
                access.descriptor().kind()
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsyncGroupCommitPlan {
    global_warp_id: usize,
    domain: AsyncGroupDomain,
    lanes: Box<[u8]>,
}

impl AsyncGroupCommitPlan {
    pub const fn global_warp_id(&self) -> usize {
        self.global_warp_id
    }

    pub const fn domain(&self) -> AsyncGroupDomain {
        self.domain
    }

    pub fn lanes(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.lanes.iter().map(|lane| usize::from(*lane))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsyncGroupCommittedGroup {
    id: AsyncGroupId,
    members: Arc<[AsyncGroupIssueEffect]>,
    source_read_action_id: AsyncGroupCompletionActionId,
    full_action_id: AsyncGroupCompletionActionId,
    full_physical_actions: Arc<[PhysicalCompletionAction]>,
}

impl AsyncGroupCommittedGroup {
    pub const fn id(&self) -> AsyncGroupId {
        self.id
    }

    pub fn members(&self) -> &[AsyncGroupIssueEffect] {
        &self.members
    }

    pub const fn source_read_action_id(&self) -> AsyncGroupCompletionActionId {
        self.source_read_action_id
    }

    pub const fn full_action_id(&self) -> AsyncGroupCompletionActionId {
        self.full_action_id
    }

    pub fn source_read_action(&self) -> AsyncGroupCompletionAction {
        AsyncGroupCompletionAction {
            id: self.source_read_action_id,
            group_id: self.id,
            milestone: AsyncGroupMilestone::SourceReadComplete,
            members: Arc::clone(&self.members),
            physical_actions: Arc::from([]),
        }
    }

    pub fn full_action(&self) -> AsyncGroupCompletionAction {
        AsyncGroupCompletionAction {
            id: self.full_action_id,
            group_id: self.id,
            milestone: AsyncGroupMilestone::FullComplete,
            members: Arc::clone(&self.members),
            physical_actions: Arc::clone(&self.full_physical_actions),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsyncGroupCommitOutcome {
    plan: AsyncGroupCommitPlan,
    groups: Box<[AsyncGroupCommittedGroup]>,
    immediately_ready_physical_actions: Box<[PhysicalCompletionAction]>,
}

impl AsyncGroupCommitOutcome {
    pub const fn plan(&self) -> &AsyncGroupCommitPlan {
        &self.plan
    }

    pub fn groups(&self) -> &[AsyncGroupCommittedGroup] {
        &self.groups
    }

    /// Deferred arrivals whose lane had no prior async work to wait for.
    pub fn immediately_ready_physical_actions(&self) -> &[PhysicalCompletionAction] {
        &self.immediately_ready_physical_actions
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsyncGroupWaitPlan {
    global_warp_id: usize,
    domain: AsyncGroupDomain,
    lanes: Box<[u8]>,
    pending_groups: usize,
    read_only: bool,
}

impl AsyncGroupWaitPlan {
    pub const fn global_warp_id(&self) -> usize {
        self.global_warp_id
    }

    pub const fn domain(&self) -> AsyncGroupDomain {
        self.domain
    }

    pub fn lanes(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.lanes.iter().map(|lane| usize::from(*lane))
    }

    pub const fn pending_groups(&self) -> usize {
        self.pending_groups
    }

    pub const fn read_only(&self) -> bool {
        self.read_only
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsyncGroupWaitedGroup {
    id: AsyncGroupId,
    milestone: AsyncGroupMilestone,
    members: Arc<[AsyncGroupIssueEffect]>,
}

impl AsyncGroupWaitedGroup {
    pub const fn id(&self) -> AsyncGroupId {
        self.id
    }

    pub const fn milestone(&self) -> AsyncGroupMilestone {
        self.milestone
    }

    pub fn members(&self) -> &[AsyncGroupIssueEffect] {
        &self.members
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsyncGroupWaitOutcome {
    plan: AsyncGroupWaitPlan,
    groups: Box<[AsyncGroupWaitedGroup]>,
}

impl AsyncGroupWaitOutcome {
    pub const fn plan(&self) -> &AsyncGroupWaitPlan {
        &self.plan
    }

    pub fn groups(&self) -> &[AsyncGroupWaitedGroup] {
        &self.groups
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsyncGroupCompletionAction {
    id: AsyncGroupCompletionActionId,
    group_id: AsyncGroupId,
    milestone: AsyncGroupMilestone,
    members: Arc<[AsyncGroupIssueEffect]>,
    physical_actions: Arc<[PhysicalCompletionAction]>,
}

impl AsyncGroupCompletionAction {
    pub const fn id(&self) -> AsyncGroupCompletionActionId {
        self.id
    }

    pub const fn group_id(&self) -> AsyncGroupId {
        self.group_id
    }

    pub const fn milestone(&self) -> AsyncGroupMilestone {
        self.milestone
    }

    pub fn members(&self) -> &[AsyncGroupIssueEffect] {
        &self.members
    }

    /// Physical arrive-on completions released by this full milestone.
    pub fn physical_actions(&self) -> &[PhysicalCompletionAction] {
        &self.physical_actions
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsyncGroupCompletionOutcome {
    action: AsyncGroupCompletionAction,
    progress: CompletionProgress,
}

struct AsyncGroupWaitRegistration {
    id: u64,
    plan: AsyncGroupWaitPlan,
    operation: OperationContext,
    waker: Mutex<Option<Waker>>,
}

impl AsyncGroupWaitRegistration {
    fn update_waker(&self, waker: &Waker) {
        *self.waker.lock().expect("async-group wait waker poisoned") = Some(waker.clone());
    }

    fn wake(&self) {
        if let Some(waker) = self
            .waker
            .lock()
            .expect("async-group wait waker poisoned")
            .take()
        {
            waker.wake();
        }
    }
}

pub struct AsyncGroupWaitFuture {
    hub: Arc<AsyncGroupHub>,
    registration: Arc<AsyncGroupWaitRegistration>,
    registered: bool,
    completed: bool,
}

impl Future for AsyncGroupWaitFuture {
    type Output = Result<AsyncGroupWaitOutcome, EngineError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.completed {
            return Poll::Ready(Err(EngineError::message(
                "async-group wait future was polled after completion",
            )));
        }
        if self.hub.wait_ready(&self.registration.plan)? {
            let outcome = self.hub.complete_wait(self.registration.plan.clone())?;
            self.hub.unregister_waiter(
                self.registration.plan.global_warp_id(),
                self.registration.id,
            );
            self.registered = false;
            self.completed = true;
            return Poll::Ready(Ok(outcome));
        }
        self.registration.update_waker(context.waker());
        if !self.registered {
            self.hub.register_waiter(Arc::clone(&self.registration))?;
            self.registered = true;
        }
        if self.hub.wait_ready(&self.registration.plan)? {
            self.registration.wake();
        }
        Poll::Pending
    }
}

impl Drop for AsyncGroupWaitFuture {
    fn drop(&mut self) {
        if self.registered {
            self.hub.unregister_waiter(
                self.registration.plan.global_warp_id(),
                self.registration.id,
            );
        }
    }
}

impl AsyncGroupCompletionOutcome {
    pub const fn action(&self) -> &AsyncGroupCompletionAction {
        &self.action
    }

    pub const fn progress(&self) -> CompletionProgress {
        self.progress
    }
}

#[derive(Debug)]
struct OpenIssue {
    effect: Option<AsyncGroupIssueEffect>,
    writes: Vec<DeferredGlobalWrite>,
}

#[derive(Debug, Default)]
struct DomainState {
    next_group_ordinal: u64,
    open_issues: Vec<OpenIssue>,
    committed_groups: VecDeque<CommittedGroup>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupCompletion {
    Pending,
    ReadsComplete,
    FullyComplete,
}

#[derive(Debug)]
struct CommittedGroup {
    id: AsyncGroupId,
    issues: Vec<OpenIssue>,
    members: Arc<[AsyncGroupIssueEffect]>,
    completion: GroupCompletion,
    // Execution batches between explicit commit boundaries belong to one
    // PTX group. Arrive-on schedules copies but does not close their group.
    closes_group: bool,
    source_read_action_id: AsyncGroupCompletionActionId,
    full_action_id: AsyncGroupCompletionActionId,
    full_physical_actions: Arc<[PhysicalCompletionAction]>,
}

impl CommittedGroup {
    fn action(&self, milestone: AsyncGroupMilestone) -> AsyncGroupCompletionAction {
        AsyncGroupCompletionAction {
            id: match milestone {
                AsyncGroupMilestone::SourceReadComplete => self.source_read_action_id,
                AsyncGroupMilestone::FullComplete => self.full_action_id,
            },
            group_id: self.id,
            milestone,
            members: Arc::clone(&self.members),
            physical_actions: match milestone {
                AsyncGroupMilestone::SourceReadComplete => Arc::from([]),
                AsyncGroupMilestone::FullComplete => Arc::clone(&self.full_physical_actions),
            },
        }
    }

    fn committed(&self) -> AsyncGroupCommittedGroup {
        AsyncGroupCommittedGroup {
            id: self.id,
            members: Arc::clone(&self.members),
            source_read_action_id: self.source_read_action_id,
            full_action_id: self.full_action_id,
            full_physical_actions: Arc::clone(&self.full_physical_actions),
        }
    }

    fn waited(&self, milestone: AsyncGroupMilestone) -> AsyncGroupWaitedGroup {
        AsyncGroupWaitedGroup {
            id: self.id,
            milestone,
            members: Arc::clone(&self.members),
        }
    }
}

impl DomainState {
    fn issue(&mut self, effect: Option<AsyncGroupIssueEffect>, writes: Vec<DeferredGlobalWrite>) {
        self.open_issues.push(OpenIssue { effect, writes });
    }

    fn next_group_id(
        &self,
        global_warp_id: usize,
        lane: usize,
        domain: AsyncGroupDomain,
    ) -> AsyncGroupId {
        AsyncGroupId {
            global_warp_id,
            lane: u8::try_from(lane).expect("warp lane fits in u8"),
            domain,
            ordinal: self.next_group_ordinal,
        }
    }

    fn validate_commit_identity(
        &self,
        global_warp_id: usize,
        lane: usize,
        domain: AsyncGroupDomain,
    ) -> Result<(), EngineError> {
        self.next_group_ordinal
            .checked_add(1)
            .ok_or_else(|| EngineError::message("async-group ordinal overflow"))?;
        let id = self.next_group_id(global_warp_id, lane, domain);
        AsyncGroupCompletionActionId::from_group(id, AsyncGroupMilestone::SourceReadComplete)?;
        AsyncGroupCompletionActionId::from_group(id, AsyncGroupMilestone::FullComplete)?;
        Ok(())
    }

    fn commit(
        &mut self,
        global_warp_id: usize,
        lane: usize,
        domain: AsyncGroupDomain,
        full_physical_actions: Arc<[PhysicalCompletionAction]>,
    ) -> Result<(Option<AsyncGroupCommittedGroup>, bool), EngineError> {
        let id = self.next_group_id(global_warp_id, lane, domain);
        self.next_group_ordinal = self
            .next_group_ordinal
            .checked_add(1)
            .expect("async-group commit identity was prevalidated");
        let source_read_action_id =
            AsyncGroupCompletionActionId::from_group(id, AsyncGroupMilestone::SourceReadComplete)?;
        let full_action_id =
            AsyncGroupCompletionActionId::from_group(id, AsyncGroupMilestone::FullComplete)?;
        let mut issues = std::mem::take(&mut self.open_issues);
        let is_empty = issues.is_empty();
        let members = issues
            .iter_mut()
            .filter_map(|issue| issue.effect.take())
            .collect::<Vec<_>>()
            .into();
        let group = CommittedGroup {
            id,
            issues,
            members,
            completion: if is_empty {
                GroupCompletion::FullyComplete
            } else {
                GroupCompletion::Pending
            },
            closes_group: full_physical_actions.is_empty(),
            source_read_action_id,
            full_action_id,
            full_physical_actions,
        };
        let outcome = (!is_empty).then(|| group.committed());
        self.committed_groups.push_back(group);
        Ok((outcome, is_empty))
    }

    fn pending_read_count(&self) -> usize {
        self.committed_groups
            .iter()
            .filter(|group| group.completion == GroupCompletion::Pending)
            .count()
    }

    fn read_complete_count(&self) -> usize {
        self.committed_groups
            .iter()
            .filter(|group| group.completion == GroupCompletion::ReadsComplete)
            .count()
    }

    fn pending_full_count(&self) -> usize {
        self.committed_groups
            .iter()
            .filter(|group| group.completion != GroupCompletion::FullyComplete)
            .count()
    }

    fn pending_issue_count(&self) -> usize {
        self.committed_groups
            .iter()
            .filter(|group| group.completion != GroupCompletion::FullyComplete)
            .map(|group| group.issues.len())
            .sum()
    }

    fn enabled_actions(&self) -> Vec<AsyncGroupCompletionAction> {
        self.enabled_action_ids()
            .into_iter()
            .map(|action_id| {
                let group = self
                    .committed_groups
                    .iter()
                    .find(|group| {
                        group.source_read_action_id == action_id
                            || group.full_action_id == action_id
                    })
                    .expect("enabled async-group action retains its committed group");
                let milestone = if group.source_read_action_id == action_id {
                    AsyncGroupMilestone::SourceReadComplete
                } else {
                    AsyncGroupMilestone::FullComplete
                };
                group.action(milestone)
            })
            .collect()
    }

    fn enabled_action_ids(&self) -> Vec<AsyncGroupCompletionActionId> {
        let mut action_ids = Vec::with_capacity(2);
        if let Some(group) = self
            .committed_groups
            .iter()
            .find(|group| group.completion != GroupCompletion::FullyComplete)
        {
            action_ids.push(match group.completion {
                GroupCompletion::Pending => group.source_read_action_id,
                GroupCompletion::ReadsComplete => group.full_action_id,
                GroupCompletion::FullyComplete => unreachable!(),
            });
        }
        if let Some(group) = self
            .committed_groups
            .iter()
            .find(|group| group.completion == GroupCompletion::Pending)
        {
            if !action_ids.contains(&group.source_read_action_id) {
                action_ids.push(group.source_read_action_id);
            }
        }
        action_ids
    }

    fn apply_enabled_action(
        &mut self,
        action_id: AsyncGroupCompletionActionId,
    ) -> Result<(AsyncGroupCompletionAction, usize), EngineError> {
        let index = {
            let _profile = ProfileTimer::new(ProfileKind::AsyncEnabledValidation);
            let index = self
                .committed_groups
                .iter()
                .position(|group| {
                    group.source_read_action_id == action_id || group.full_action_id == action_id
                })
                .ok_or_else(|| {
                    EngineError::message(format!(
                        "unknown async-group completion action {}",
                        action_id.get()
                    ))
                })?;
            let enabled = self.enabled_action_ids();
            if !enabled.contains(&action_id) {
                return Err(EngineError::message(format!(
                    "async-group completion action {} is blocked by FIFO milestone order; enabled actions are {:?}",
                    action_id.get(),
                    enabled
                        .iter()
                        .map(|action_id| action_id.get())
                        .collect::<Vec<_>>()
                )));
            }
            index
        };
        let group = &mut self.committed_groups[index];
        let action = match group.completion {
            GroupCompletion::Pending => {
                group.completion = GroupCompletion::ReadsComplete;
                let _profile = ProfileTimer::new(ProfileKind::AsyncActionMetadata);
                group.action(AsyncGroupMilestone::SourceReadComplete)
            }
            GroupCompletion::ReadsComplete => {
                {
                    let _profile = ProfileTimer::new(ProfileKind::AsyncGlobalPublish);
                    publish_deferred_global_writes(
                        group.issues.iter().flat_map(|issue| issue.writes.iter()),
                    )?;
                }
                // Keep the published payloads owned by the committed group until
                // its normal full-wait retirement.  Eagerly dropping every tiny
                // payload here serialized allocator and Arc teardown on the
                // completion pump even though the completion state already
                // prevents a second publish.
                group.completion = GroupCompletion::FullyComplete;
                let _profile = ProfileTimer::new(ProfileKind::AsyncActionMetadata);
                group.action(AsyncGroupMilestone::FullComplete)
            }
            GroupCompletion::FullyComplete => {
                return Err(EngineError::message(format!(
                    "async-group completion action {} already completed",
                    action_id.get()
                )));
            }
        };
        let completed_operations = group.issues.len();
        // Release operations have no explicit wait that could retire a group.
        // The returned action owns its analysis metadata; no issuer acquires it.
        if action.group_id().domain() == AsyncGroupDomain::Release
            && action.milestone() == AsyncGroupMilestone::FullComplete
        {
            self.committed_groups.remove(index);
        }
        Ok((action, completed_operations))
    }

    fn has_enabled_action(&self) -> bool {
        self.committed_groups
            .iter()
            .any(|group| group.completion != GroupCompletion::FullyComplete)
    }

    fn wait_prefix_len(&self, mut pending_groups: usize) -> usize {
        // Ignore a trailing uncommitted suffix and count explicit boundaries,
        // not the number of execution batches produced by arrive-on.
        for (index, group) in self.committed_groups.iter().enumerate().rev() {
            if group.closes_group {
                if pending_groups == 0 {
                    return index + 1;
                }
                pending_groups -= 1;
            }
        }
        0
    }

    fn required_wait_action(
        &self,
        pending_groups: usize,
        read_only: bool,
    ) -> Option<AsyncGroupCompletionAction> {
        let target_count = self.wait_prefix_len(pending_groups);
        self.committed_groups
            .iter()
            .take(target_count)
            .find_map(|group| match (read_only, group.completion) {
                (_, GroupCompletion::Pending) => {
                    Some(group.action(AsyncGroupMilestone::SourceReadComplete))
                }
                (false, GroupCompletion::ReadsComplete) => {
                    Some(group.action(AsyncGroupMilestone::FullComplete))
                }
                _ => None,
            })
    }

    fn validate_wait_completion(
        &self,
        pending_groups: usize,
        read_only: bool,
    ) -> Result<(), EngineError> {
        let target_count = self.wait_prefix_len(pending_groups);
        for group in self.committed_groups.iter().take(target_count) {
            let ready = if read_only {
                group.completion != GroupCompletion::Pending
            } else {
                group.completion == GroupCompletion::FullyComplete
            };
            if !ready {
                return Err(EngineError::message(format!(
                    "async-group wait completed before group {:?} reached its required milestone",
                    group.id
                )));
            }
        }
        Ok(())
    }

    fn commit_wait_completion(
        &mut self,
        pending_groups: usize,
        read_only: bool,
    ) -> Vec<AsyncGroupWaitedGroup> {
        let target_count = self.wait_prefix_len(pending_groups);
        let milestone = if read_only {
            AsyncGroupMilestone::SourceReadComplete
        } else {
            AsyncGroupMilestone::FullComplete
        };
        let waited = self
            .committed_groups
            .iter()
            .take(target_count)
            .map(|group| group.waited(milestone))
            .collect::<Vec<_>>();
        if read_only {
            let mut index = 0;
            self.committed_groups.retain(|group| {
                let in_waited_prefix = index < target_count;
                index += 1;
                !(in_waited_prefix
                    && group.issues.is_empty()
                    && group.completion == GroupCompletion::FullyComplete)
            });
        } else {
            self.committed_groups.drain(..target_count);
        }
        waited
    }
}

#[derive(Debug, Default)]
struct LaneState {
    cp_async: DomainState,
    bulk: DomainState,
    release: DomainState,
}

impl LaneState {
    fn domain_mut(&mut self, domain: AsyncGroupDomain) -> &mut DomainState {
        match domain {
            AsyncGroupDomain::CpAsync => &mut self.cp_async,
            AsyncGroupDomain::Bulk => &mut self.bulk,
            AsyncGroupDomain::Release => &mut self.release,
        }
    }

    fn domain(&self, domain: AsyncGroupDomain) -> &DomainState {
        match domain {
            AsyncGroupDomain::CpAsync => &self.cp_async,
            AsyncGroupDomain::Bulk => &self.bulk,
            AsyncGroupDomain::Release => &self.release,
        }
    }
}

#[derive(Debug)]
struct WarpState {
    lanes: [LaneState; WARP_SIZE],
}

impl Default for WarpState {
    fn default() -> Self {
        Self {
            lanes: std::array::from_fn(|_| LaneState::default()),
        }
    }
}

/// Per-lane PTX async-group protocol state, sharded by warp for concurrent launches.
pub struct AsyncGroupHub {
    warps: Vec<Mutex<WarpState>>,
    active_completion_warps: Box<[AtomicBool]>,
    waiters: Vec<Mutex<BTreeMap<u64, Arc<AsyncGroupWaitRegistration>>>>,
    next_waiter_id: AtomicU64,
}

impl AsyncGroupHub {
    pub(crate) fn new(topology: LaunchTopology) -> Self {
        let warp_count = topology.warp_count();
        Self {
            warps: (0..warp_count)
                .map(|_| Mutex::new(WarpState::default()))
                .collect(),
            active_completion_warps: (0..warp_count).map(|_| AtomicBool::new(false)).collect(),
            waiters: (0..warp_count)
                .map(|_| Mutex::new(BTreeMap::new()))
                .collect(),
            next_waiter_id: AtomicU64::new(0),
        }
    }

    fn warp_has_enabled_action(state: &WarpState) -> bool {
        state.lanes.iter().any(|lane| {
            lane.cp_async.has_enabled_action()
                || lane.bulk.has_enabled_action()
                || lane.release.has_enabled_action()
        })
    }

    fn state(&self, global_warp_id: usize) -> Result<&Mutex<WarpState>, EngineError> {
        self.warps.get(global_warp_id).ok_or_else(|| {
            EngineError::message(format!(
                "async-group warp {global_warp_id} is outside the launch topology"
            ))
        })
    }

    fn validate_mask(
        context: &WarpContext,
        mask: WarpMask,
        operation: &crate::DiagnosticLabel,
    ) -> Result<(), EngineError> {
        let inactive = mask - context.active_mask();
        if inactive.is_empty() {
            return Ok(());
        }
        Err(operation.engine_error(format_args!(
            " mask contains inactive lanes {:?}",
            inactive.iter().collect::<Vec<_>>()
        )))
    }

    fn validate_wait(
        context: &WarpContext,
        domain: AsyncGroupDomain,
        mask: WarpMask,
        pending_groups: i64,
        read_only: bool,
    ) -> Result<usize, EngineError> {
        if domain == AsyncGroupDomain::Release {
            return Err(EngineError::message(
                "async.release has no PTX group-wait instruction",
            ));
        }
        Self::validate_mask(
            context,
            mask,
            &crate::DiagnosticLabel::new("async-group wait"),
        )?;
        if read_only && domain == AsyncGroupDomain::CpAsync {
            return Err(EngineError::message(
                "cp.async.wait_group does not support the bulk .read modifier",
            ));
        }
        usize::try_from(pending_groups).map_err(|_| {
            EngineError::message(format!(
                "async-group pending count must fit usize, got {pending_groups}"
            ))
        })
    }

    pub(crate) fn issue(
        &self,
        context: &WarpContext,
        domain: AsyncGroupDomain,
        mask: WarpMask,
    ) -> Result<(), EngineError> {
        Self::validate_mask(context, mask, &crate::DiagnosticLabel::new(domain.name()))?;
        let mut state = self
            .state(context.global_warp_id())?
            .lock()
            .map_err(|_| EngineError::message("async-group warp state lock is poisoned"))?;
        for lane in mask {
            state.lanes[lane].domain_mut(domain).issue(None, Vec::new());
        }
        Ok(())
    }

    pub(crate) fn issue_unmodeled(
        &self,
        context: &WarpContext,
        domain: AsyncGroupDomain,
        mask: WarpMask,
        writes: Vec<DeferredGlobalWrite>,
    ) -> Result<(), EngineError> {
        if domain == AsyncGroupDomain::CpAsync {
            if !writes.is_empty() {
                return Err(EngineError::message(
                    "cp.async issue cannot carry deferred global writes",
                ));
            }
            return self.issue(context, domain, mask);
        }
        Self::validate_mask(
            context,
            mask,
            &crate::DiagnosticLabel::new("cp.async.bulk write issue"),
        )?;
        if mask.is_empty() {
            return Err(EngineError::message(
                "cp.async.bulk write issue has no active lane",
            ));
        }
        if mask.len() != 1 && writes.len() != mask.len() {
            return Err(EngineError::message(format!(
                "cp.async.bulk write count {} does not match {} issuing lanes",
                writes.len(),
                mask.len(),
            )));
        }
        let mut state = self
            .state(context.global_warp_id())?
            .lock()
            .map_err(|_| EngineError::message("async-group warp state lock is poisoned"))?;
        if mask.len() == 1 {
            let lane = mask
                .first_active()
                .expect("non-empty single-lane mask has a first lane");
            state.lanes[lane].domain_mut(domain).issue(None, writes);
            return Ok(());
        }
        for (lane, write) in mask.into_iter().zip(writes) {
            state.lanes[lane]
                .domain_mut(domain)
                .issue(None, vec![write]);
        }
        Ok(())
    }

    pub(crate) fn issue_exact(
        &self,
        context: &WarpContext,
        effect: AsyncGroupIssueEffect,
        writes: Vec<DeferredGlobalWrite>,
    ) -> Result<(), EngineError> {
        self.validate_exact_issue(context, &effect)?;
        let lane = effect
            .operation()
            .active_mask()
            .first_active()
            .expect("validated exact async-group issue has one active lane");
        let mut state = self
            .state(context.global_warp_id())?
            .lock()
            .map_err(|_| EngineError::message("async-group warp state lock is poisoned"))?;
        state.lanes[lane].bulk.issue(Some(effect), writes);
        Ok(())
    }

    pub(crate) fn issue_exact_batch(
        &self,
        context: &WarpContext,
        effect: &AsyncGroupIssueBatchEffect,
        writes: Vec<DeferredGlobalWrite>,
    ) -> Result<(), EngineError> {
        if (effect.domain() == AsyncGroupDomain::Release && writes.len() != effect.members().len())
            || (effect.domain() != AsyncGroupDomain::Release && !writes.is_empty())
        {
            return Err(EngineError::message(
                "async batch deferred writes disagree with its domain/members",
            ));
        }
        let mut writes = writes.into_iter();
        if effect.operation().id().global_warp_id() != context.global_warp_id() {
            return Err(EngineError::message(format!(
                "async-group batch operation warp {} does not match runtime warp {}",
                effect.operation().id().global_warp_id(),
                context.global_warp_id()
            )));
        }
        Self::validate_mask(
            context,
            effect.operation().active_mask(),
            &crate::DiagnosticLabel::new(format!("exact {} issue", effect.domain().name())),
        )?;
        let mut state = self
            .state(context.global_warp_id())?
            .lock()
            .map_err(|_| EngineError::message("async-group warp state lock is poisoned"))?;
        for member in effect.members() {
            let lane = member
                .operation()
                .active_mask()
                .first_active()
                .ok_or_else(|| {
                    EngineError::message("async-group batch member has no active lane")
                })?;
            state.lanes[lane]
                .domain_mut(effect.domain())
                .issue(Some(member.clone()), writes.next().into_iter().collect());
        }
        Ok(())
    }

    pub(crate) fn validate_exact_issue(
        &self,
        context: &WarpContext,
        effect: &AsyncGroupIssueEffect,
    ) -> Result<(), EngineError> {
        self.validate_exact_issue_participation(context, effect.operation(), effect.domain())
    }

    pub(crate) fn validate_exact_issue_participation(
        &self,
        context: &WarpContext,
        operation: &OperationContext,
        domain: AsyncGroupDomain,
    ) -> Result<(), EngineError> {
        if domain != AsyncGroupDomain::Bulk {
            return Err(EngineError::message(
                "exact S2G issue currently requires the cp.async.bulk domain",
            ));
        }
        if operation.id().global_warp_id() != context.global_warp_id() {
            return Err(EngineError::message(format!(
                "async-group issue operation warp {} does not match runtime warp {}",
                operation.id().global_warp_id(),
                context.global_warp_id()
            )));
        }
        let mask = operation.active_mask();
        Self::validate_mask(
            context,
            mask,
            &crate::DiagnosticLabel::new("exact cp.async.bulk write issue"),
        )?;
        if mask.len() != 1 {
            return Err(EngineError::async_group_issuer_violation(
                operation.id().to_string(),
                1,
                mask.bits(),
            ));
        }
        Ok(())
    }

    pub(crate) fn commit_plan(
        &self,
        context: &WarpContext,
        domain: AsyncGroupDomain,
        mask: WarpMask,
    ) -> Result<AsyncGroupCommitPlan, EngineError> {
        Self::validate_mask(
            context,
            mask,
            &crate::DiagnosticLabel::new("async-group commit"),
        )?;
        Ok(AsyncGroupCommitPlan {
            global_warp_id: context.global_warp_id(),
            domain,
            lanes: mask
                .iter()
                .map(|lane| u8::try_from(lane).expect("warp lane fits in u8"))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
    }

    pub(crate) fn commit_detailed(
        &self,
        plan: AsyncGroupCommitPlan,
    ) -> Result<AsyncGroupCommitOutcome, EngineError> {
        self.commit_detailed_with_physical_arrivals(plan, &[], |_| Ok(()))
    }

    /// Commit one per-lane group and gate each matching physical arrive-on
    /// action behind that lane's full-completion milestone.
    pub(crate) fn commit_detailed_with_physical_arrivals(
        &self,
        plan: AsyncGroupCommitPlan,
        physical_actions: &[PhysicalCompletionAction],
        publish: impl FnOnce(&AsyncGroupCommitOutcome) -> Result<(), EngineError>,
    ) -> Result<AsyncGroupCommitOutcome, EngineError> {
        let lanes = plan.lanes().collect::<Vec<_>>();
        if !physical_actions.is_empty() && physical_actions.len() != lanes.len() {
            return Err(EngineError::message(format!(
                "async-group commit has {} active lanes but {} physical arrive-on actions",
                lanes.len(),
                physical_actions.len()
            )));
        }
        for action in physical_actions {
            if action.arrival() != Some((plan.global_warp_id(), 1)) {
                return Err(EngineError::message(format!(
                    "async-group physical action {} is not a one-lane arrival from warp {}",
                    action.id(),
                    plan.global_warp_id()
                )));
            }
        }
        let mut state = self
            .state(plan.global_warp_id())?
            .lock()
            .map_err(|_| EngineError::message("async-group warp state lock is poisoned"))?;
        let mut groups = Vec::new();
        let mut immediately_ready_physical_actions = Vec::new();
        for &lane in &lanes {
            state.lanes[lane]
                .domain(plan.domain())
                .validate_commit_identity(plan.global_warp_id(), lane, plan.domain())?;
        }
        for (index, &lane) in lanes.iter().enumerate() {
            let lane_actions: Arc<[PhysicalCompletionAction]> = physical_actions
                .get(index)
                .copied()
                .into_iter()
                .collect::<Vec<_>>()
                .into();
            let domain = state.lanes[lane].domain_mut(plan.domain());
            if !lane_actions.is_empty() && domain.open_issues.is_empty() {
                // Arrive-on observes prior copies, not just uncommitted ones.
                // Full completion is FIFO, so the last unfinished group gates
                // the whole pending prefix. Do not manufacture an empty group:
                // only an explicit commit changes wait_group's group count.
                if let Some(group) = domain
                    .committed_groups
                    .iter_mut()
                    .rev()
                    .find(|group| group.completion != GroupCompletion::FullyComplete)
                {
                    group.full_physical_actions = group
                        .full_physical_actions
                        .iter()
                        .chain(lane_actions.iter())
                        .copied()
                        .collect::<Vec<_>>()
                        .into();
                } else {
                    immediately_ready_physical_actions.extend(lane_actions.iter().copied());
                }
                continue;
            }
            let (group, is_empty) = domain.commit(
                plan.global_warp_id(),
                lane,
                plan.domain(),
                Arc::clone(&lane_actions),
            )?;
            if let Some(group) = group {
                groups.push(group);
            }
            if is_empty {
                immediately_ready_physical_actions.extend(lane_actions.iter().copied());
            }
        }
        if Self::warp_has_enabled_action(&state) {
            self.active_completion_warps[plan.global_warp_id()].store(true, Ordering::Release);
        }
        let outcome = AsyncGroupCommitOutcome {
            plan,
            groups: groups.into_boxed_slice(),
            immediately_ready_physical_actions: immediately_ready_physical_actions
                .into_boxed_slice(),
        };
        // A previously committed group is already scheduler-visible. Register
        // the newly attached arrivals before completion can take this lock.
        publish(&outcome)?;
        Ok(outcome)
    }

    pub(crate) fn commit(
        &self,
        context: &WarpContext,
        domain: AsyncGroupDomain,
        mask: WarpMask,
    ) -> Result<(), EngineError> {
        let plan = self.commit_plan(context, domain, mask)?;
        self.commit_detailed(plan)?;
        Ok(())
    }

    /// Make issued bulk operations eligible for the kernel-exit completion drain.
    ///
    /// `cp.async.bulk.commit_group` only groups prior operations for an explicit
    /// wait; it does not issue them.  A kernel may therefore exit with a final
    /// uncommitted S2G group when its shared source is never reused.  The launch
    /// boundary still has to retire those already-issued operations and publish
    /// their observable global writes.  This internal commit creates no
    /// instruction effect or happens-before edge.
    pub(crate) fn commit_open_bulk_groups_for_exit(
        &self,
    ) -> Result<Vec<AsyncGroupCommittedGroup>, EngineError> {
        let mut groups = Vec::new();
        for (global_warp_id, state) in self.warps.iter().enumerate() {
            let mut state = state
                .lock()
                .map_err(|_| EngineError::message("async-group warp state lock is poisoned"))?;
            for lane in 0..WARP_SIZE {
                let bulk = &state.lanes[lane].bulk;
                if !bulk.open_issues.is_empty() {
                    bulk.validate_commit_identity(global_warp_id, lane, AsyncGroupDomain::Bulk)?;
                }
            }
            for lane in 0..WARP_SIZE {
                let bulk = &mut state.lanes[lane].bulk;
                if bulk.open_issues.is_empty() {
                    continue;
                }
                let (group, is_empty) =
                    bulk.commit(global_warp_id, lane, AsyncGroupDomain::Bulk, Arc::from([]))?;
                debug_assert!(!is_empty);
                groups.push(group.expect("a non-empty exit group has completion metadata"));
            }
            if Self::warp_has_enabled_action(&state) {
                self.active_completion_warps[global_warp_id].store(true, Ordering::Release);
            }
        }
        Ok(groups)
    }

    pub(crate) fn wait_plan(
        &self,
        context: &WarpContext,
        domain: AsyncGroupDomain,
        mask: WarpMask,
        pending_groups: i64,
        read_only: bool,
    ) -> Result<AsyncGroupWaitPlan, EngineError> {
        let pending_groups = Self::validate_wait(context, domain, mask, pending_groups, read_only)?;
        Ok(AsyncGroupWaitPlan {
            global_warp_id: context.global_warp_id(),
            domain,
            lanes: mask
                .iter()
                .map(|lane| u8::try_from(lane).expect("warp lane fits in u8"))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            pending_groups,
            read_only,
        })
    }

    pub(crate) fn wait(
        self: &Arc<Self>,
        plan: AsyncGroupWaitPlan,
        operation: OperationContext,
    ) -> Result<AsyncGroupWaitFuture, EngineError> {
        if plan.global_warp_id() != operation.id().global_warp_id() {
            return Err(EngineError::message(format!(
                "async-group wait plan warp {} does not match operation warp {}",
                plan.global_warp_id(),
                operation.id().global_warp_id()
            )));
        }
        let id = self
            .next_waiter_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |next| {
                next.checked_add(1)
            })
            .map_err(|_| EngineError::message("async-group waiter ID overflow"))?;
        Ok(AsyncGroupWaitFuture {
            hub: Arc::clone(self),
            registration: Arc::new(AsyncGroupWaitRegistration {
                id,
                plan,
                operation,
                waker: Mutex::new(None),
            }),
            registered: false,
            completed: false,
        })
    }

    fn register_waiter(
        &self,
        registration: Arc<AsyncGroupWaitRegistration>,
    ) -> Result<(), EngineError> {
        let global_warp_id = registration.plan.global_warp_id();
        let mut waiters = self
            .waiters
            .get(global_warp_id)
            .ok_or_else(|| {
                EngineError::message(format!(
                    "async-group waiter warp {global_warp_id} is outside the launch topology"
                ))
            })?
            .lock()
            .map_err(|_| EngineError::message("async-group waiter registry is poisoned"))?;
        if waiters.insert(registration.id, registration).is_some() {
            return Err(EngineError::message("duplicate async-group waiter ID"));
        }
        Ok(())
    }

    fn unregister_waiter(&self, global_warp_id: usize, waiter_id: u64) {
        self.waiters[global_warp_id]
            .lock()
            .expect("async-group waiter registry poisoned")
            .remove(&waiter_id);
    }

    fn wait_ready(&self, plan: &AsyncGroupWaitPlan) -> Result<bool, EngineError> {
        Ok(self.next_wait_action(plan)?.is_none())
    }

    /// Return waiters whose predicate may have changed after one action.
    ///
    /// Async-group state and wait plans are warp-local. An action for one warp
    /// cannot make any other warp's wait ready, so checking the launch-wide
    /// waiter registry after every milestone is both redundant and quadratic
    /// on large launches.
    fn ready_waiters(
        &self,
        global_warp_id: usize,
    ) -> Result<Vec<Arc<AsyncGroupWaitRegistration>>, EngineError> {
        let waiters = self
            .waiters
            .get(global_warp_id)
            .ok_or_else(|| {
                EngineError::message(format!(
                    "async-group waiter warp {global_warp_id} is outside the launch topology"
                ))
            })?
            .lock()
            .map_err(|_| EngineError::message("async-group waiter registry is poisoned"))?
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let mut ready = Vec::new();
        for waiter in waiters {
            if self.wait_ready(&waiter.plan)? {
                ready.push(waiter);
            }
        }
        Ok(ready)
    }

    pub(crate) fn next_wait_action(
        &self,
        plan: &AsyncGroupWaitPlan,
    ) -> Result<Option<AsyncGroupCompletionAction>, EngineError> {
        let state = self
            .state(plan.global_warp_id())?
            .lock()
            .map_err(|_| EngineError::message("async-group warp state lock is poisoned"))?;
        Ok(plan.lanes().find_map(|lane| {
            state.lanes[lane]
                .domain(plan.domain())
                .required_wait_action(plan.pending_groups(), plan.read_only())
        }))
    }

    pub(crate) fn complete_wait(
        &self,
        plan: AsyncGroupWaitPlan,
    ) -> Result<AsyncGroupWaitOutcome, EngineError> {
        let mut state = self
            .state(plan.global_warp_id())?
            .lock()
            .map_err(|_| EngineError::message("async-group warp state lock is poisoned"))?;
        let mut groups = Vec::new();
        for lane in plan.lanes() {
            state.lanes[lane]
                .domain(plan.domain())
                .validate_wait_completion(plan.pending_groups(), plan.read_only())?;
        }
        for lane in plan.lanes() {
            groups.extend(
                state.lanes[lane]
                    .domain_mut(plan.domain())
                    .commit_wait_completion(plan.pending_groups(), plan.read_only()),
            );
        }
        Ok(AsyncGroupWaitOutcome {
            plan,
            groups: groups.into_boxed_slice(),
        })
    }

    pub(crate) fn wait_group(
        &self,
        context: &WarpContext,
        domain: AsyncGroupDomain,
        mask: WarpMask,
        pending_groups: i64,
        read_only: bool,
    ) -> Result<(), EngineError> {
        let plan = self.wait_plan(context, domain, mask, pending_groups, read_only)?;
        while self.apply_next_wait_completion(&plan)?.is_some() {}
        self.complete_wait(plan)?;
        Ok(())
    }

    fn apply_next_wait_completion(
        &self,
        plan: &AsyncGroupWaitPlan,
    ) -> Result<Option<AsyncGroupCompletionOutcome>, EngineError> {
        let mut state = self
            .state(plan.global_warp_id())?
            .lock()
            .map_err(|_| EngineError::message("async-group warp state lock is poisoned"))?;
        for lane in plan.lanes() {
            let domain_state = state.lanes[lane].domain_mut(plan.domain());
            let Some(action) =
                domain_state.required_wait_action(plan.pending_groups(), plan.read_only())
            else {
                continue;
            };
            let (action, completed_operations) = domain_state.apply_enabled_action(action.id())?;
            self.active_completion_warps[plan.global_warp_id()]
                .store(Self::warp_has_enabled_action(&state), Ordering::Release);
            drop(state);
            let ready_waiters = self.ready_waiters(plan.global_warp_id())?;
            let outcome = AsyncGroupCompletionOutcome {
                progress: CompletionProgress {
                    completed_operations,
                    woken_warps: ready_waiters.len(),
                },
                action,
            };
            for waiter in ready_waiters {
                waiter.wake();
            }
            return Ok(Some(outcome));
        }
        Ok(None)
    }

    pub(crate) fn pending_completion_actions(&self) -> Vec<AsyncGroupCompletionAction> {
        let mut actions = Vec::new();
        for (warp_id, state) in self.warps.iter().enumerate() {
            if !self.active_completion_warps[warp_id].load(Ordering::Acquire) {
                continue;
            }
            let state = state.lock().expect("async-group warp state lock poisoned");
            for lane in &state.lanes {
                for domain in AsyncGroupDomain::ALL {
                    actions.extend(lane.domain(domain).enabled_actions());
                }
            }
        }
        actions.sort_by_key(|action| action.id());
        actions
    }

    /// Return enabled completion IDs without cloning their issue metadata.
    pub(crate) fn pending_completion_action_ids(&self) -> Vec<AsyncGroupCompletionActionId> {
        let mut action_ids = Vec::new();
        for (warp_id, state) in self.warps.iter().enumerate() {
            if !self.active_completion_warps[warp_id].load(Ordering::Acquire) {
                continue;
            }
            let state = state.lock().expect("async-group warp state lock poisoned");
            for lane in &state.lanes {
                for domain in AsyncGroupDomain::ALL {
                    action_ids.extend(lane.domain(domain).enabled_action_ids());
                }
            }
        }
        action_ids.sort_unstable();
        action_ids
    }

    /// Apply a completion snapshot using its exact owning warp/lane/domain.
    ///
    /// The completion pump already obtained this action from
    /// `pending_completion_actions`; retaining its coordinates avoids
    /// rescanning every warp to rediscover the owner from the opaque scheduler
    /// ID. `apply_enabled_action` still validates FIFO milestone enablement
    /// against live state.
    pub(crate) fn apply_completion_action_detailed_with_outcome(
        &self,
        pending: &AsyncGroupCompletionAction,
        publish_before_wake: impl FnOnce(&AsyncGroupCompletionOutcome) -> Result<(), EngineError>,
    ) -> Result<AsyncGroupCompletionOutcome, EngineError> {
        let group_id = pending.group_id();
        let global_warp_id = group_id.global_warp_id();
        let mut state = self
            .state(global_warp_id)?
            .lock()
            .map_err(|_| EngineError::message("async-group warp state lock is poisoned"))?;
        let domain_state = state.lanes[group_id.lane()].domain_mut(group_id.domain());
        let (action, completed_operations) = {
            let _profile = ProfileTimer::new(ProfileKind::AsyncDomainApply);
            domain_state.apply_enabled_action(pending.id())?
        };
        if action.group_id() != group_id
            || action.milestone() != pending.milestone()
            || action.members() != pending.members()
        {
            return Err(EngineError::message(format!(
                "async-group completion action {} metadata drifted after it was enabled",
                pending.id().get()
            )));
        }
        // Publish the mode-visible completion while the numeric milestone is
        // still protected by the owning warp lock. A waiter may already be
        // scheduled, so delaying clock/access publication until after unlock
        // would expose the completed group too early.
        let mut outcome = AsyncGroupCompletionOutcome {
            progress: CompletionProgress {
                completed_operations,
                woken_warps: 0,
            },
            action,
        };
        {
            let _profile = ProfileTimer::new(ProfileKind::AsyncPublishOutcome);
            publish_before_wake(&outcome)?;
        }
        self.active_completion_warps[global_warp_id]
            .store(Self::warp_has_enabled_action(&state), Ordering::Release);
        drop(state);
        let ready_waiters = {
            let _profile = ProfileTimer::new(ProfileKind::AsyncReadyWaiters);
            self.ready_waiters(global_warp_id)?
        };
        outcome.progress.woken_warps = ready_waiters.len();
        for waiter in ready_waiters {
            waiter.wake();
        }
        Ok(outcome)
    }

    fn apply_first_enabled_completion(
        &self,
    ) -> Result<Option<AsyncGroupCompletionOutcome>, EngineError> {
        for (warp_id, state) in self.warps.iter().enumerate() {
            if !self.active_completion_warps[warp_id].load(Ordering::Acquire) {
                continue;
            }
            let mut state = state
                .lock()
                .map_err(|_| EngineError::message("async-group warp state lock is poisoned"))?;
            for lane in 0..WARP_SIZE {
                for domain in AsyncGroupDomain::ALL {
                    let domain_state = state.lanes[lane].domain_mut(domain);
                    let Some(action_id) = domain_state.enabled_action_ids().into_iter().next()
                    else {
                        continue;
                    };
                    let (action, completed_operations) =
                        domain_state.apply_enabled_action(action_id)?;
                    self.active_completion_warps[warp_id]
                        .store(Self::warp_has_enabled_action(&state), Ordering::Release);
                    drop(state);
                    let ready_waiters = self.ready_waiters(warp_id)?;
                    let outcome = AsyncGroupCompletionOutcome {
                        progress: CompletionProgress {
                            completed_operations,
                            woken_warps: ready_waiters.len(),
                        },
                        action,
                    };
                    for waiter in ready_waiters {
                        waiter.wake();
                    }
                    return Ok(Some(outcome));
                }
            }
        }
        Ok(None)
    }

    fn quiescence_details(&self) -> Result<Vec<String>, String> {
        let mut details = Vec::new();
        for (warp_id, state) in self.warps.iter().enumerate() {
            let state = state
                .lock()
                .map_err(|_| "async-group warp state lock is poisoned".to_string())?;
            for (lane, lane_state) in state.lanes.iter().enumerate() {
                for domain in AsyncGroupDomain::ALL {
                    let domain_state = lane_state.domain(domain);
                    if !domain_state.open_issues.is_empty() {
                        details.push(format!(
                            "warp {warp_id} lane {lane} {} has {} uncommitted issue(s)",
                            domain.name(),
                            domain_state.open_issues.len()
                        ));
                    }
                    if domain_state.pending_full_count() != 0 {
                        details.push(format!(
                            "warp {warp_id} lane {lane} {} has {} execution batch(es) / {} issue(s) pending full completion ({} batch(es) still reading source, {} batch(es) with source reads complete)",
                            domain.name(),
                            domain_state.pending_full_count(),
                            domain_state.pending_issue_count(),
                            domain_state.pending_read_count(),
                            domain_state.read_complete_count(),
                        ));
                    }
                }
            }
        }
        Ok(details)
    }
}


impl CompletionSource for AsyncGroupHub {
    fn source_name(&self) -> &'static str {
        "async_groups"
    }

    fn pump(&self) -> Result<CompletionProgress, SynchronizationError> {
        let budget = self.pending_completion_action_ids().len();
        let mut progress = CompletionProgress::default();
        for _ in 0..budget {
            let Some(outcome) = self.apply_first_enabled_completion().map_err(|error| {
                SynchronizationError::CompletionSourceOperationFailed {
                    source_name: self.source_name(),
                    details: format!("async-group completion failed: {error}"),
                }
            })?
            else {
                break;
            };
            progress.completed_operations += outcome.progress().completed_operations;
            progress.woken_warps += outcome.progress().woken_warps;
        }
        Ok(progress)
    }

    fn blocked_operations(&self) -> Vec<BlockedOperation> {
        self.waiters
            .iter()
            .flat_map(|waiters| {
                waiters
                    .lock()
                    .expect("async-group waiter registry poisoned")
                    .values()
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .map(|waiter| {
                let operation = &waiter.operation;
                let loop_path = operation
                    .id()
                    .loop_frames()
                    .iter()
                    .map(|frame| i64::try_from(frame.iteration_ordinal()).unwrap_or(i64::MAX));
                let awaited_operation = match (waiter.plan.domain(), waiter.plan.read_only()) {
                    (AsyncGroupDomain::CpAsync, false) => AwaitedOperation::CpAsyncWaitGroup,
                    (AsyncGroupDomain::CpAsync, true) => AwaitedOperation::CpAsyncWaitGroupRead,
                    (AsyncGroupDomain::Bulk, false) => AwaitedOperation::BulkWaitGroup,
                    (AsyncGroupDomain::Bulk, true) => AwaitedOperation::BulkWaitGroupRead,
                    (AsyncGroupDomain::Release, _) => unreachable!("release waits are rejected"),
                };
                BlockedOperation::new(
                    operation.id().global_warp_id(),
                    awaited_operation,
                    OccurrenceKey::new(
                        operation.id().source_op_id().get(),
                        "async-group wait",
                        loop_path,
                        ScopeInstance::Warp {
                            global_warp_id: operation.id().global_warp_id(),
                        },
                    ),
                    None,
                    ParticipantState::counted([operation.id().global_warp_id()], 1, 0, None, None),
                )
            })
            .collect()
    }

    fn validate_quiescent(&self) -> Result<(), SynchronizationError> {
        let details = self.quiescence_details().map_err(|details| {
            SynchronizationError::CompletionSourceNotQuiescent {
                source_name: self.source_name(),
                details,
            }
        })?;
        if details.is_empty() {
            return Ok(());
        }
        Err(SynchronizationError::CompletionSourceNotQuiescent {
            source_name: self.source_name(),
            details: details.join("; "),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::Ordering;
    use std::sync::{mpsc, Arc};
    use std::thread;
    use std::time::Duration;

    use super::{
        AsyncGroupDomain, AsyncGroupHub, AsyncGroupIssueBatchEffect, AsyncGroupIssueEffect,
        AsyncGroupMilestone, GroupCompletion,
    };
    use crate::{
        CompletionSource, DynamicOpId, GlobalMemory, LaunchTopology, MemoryAccessClass,
        MemoryOrder, MemoryProxy, MemoryScope, OperationContext, OperationKind,
        PhysicalAccessBatch, PhysicalAccessDescriptor, PhysicalAccessKind, PhysicalAccessSpace,
        PhysicalAllocationId, PhysicalByteSpan, StaticOpId, WarpMask,
    };

    fn context() -> crate::WarpContext {
        LaunchTopology::new(1, 1, 1)
            .unwrap()
            .warp_contexts()
            .next()
            .unwrap()
            .with_active_mask(WarpMask::from_lanes([0, 1]).unwrap())
    }

    fn access_batch(
        operation: &OperationContext,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        allocation: u64,
    ) -> PhysicalAccessBatch {
        let descriptor = PhysicalAccessDescriptor::new(kind, space, 4).unwrap();
        PhysicalAccessBatch::resolve(operation.clone(), descriptor, |lane| {
            Ok::<_, std::convert::Infallible>(vec![PhysicalByteSpan::new(
                PhysicalAllocationId::new(allocation),
                lane.lane() * 4,
                4,
            )
            .unwrap()])
        })
        .unwrap()
    }

    #[test]
    fn only_bulk_async_groups_use_the_async_memory_proxy() {
        let operation = OperationContext::new(
            DynamicOpId::new(0, 0, 0, StaticOpId::new(1), []),
            OperationKind::AsyncIssue,
            context().active_mask(),
        );
        let source = access_batch(
            &operation,
            PhysicalAccessKind::Read,
            PhysicalAccessSpace::Global,
            1,
        );
        let destination = access_batch(
            &operation,
            PhysicalAccessKind::Write,
            PhysicalAccessSpace::Shared,
            2,
        );

        for (domain, expected_proxy) in [
            (AsyncGroupDomain::CpAsync, MemoryProxy::Generic),
            (AsyncGroupDomain::Bulk, MemoryProxy::Async),
        ] {
            let effect = AsyncGroupIssueEffect::new(
                operation.clone(),
                domain,
                [source.clone()],
                [destination.clone()],
            )
            .unwrap();
            assert!(effect
                .source_accesses()
                .iter()
                .chain(effect.destination_accesses())
                .all(|batch| batch.descriptor().memory_semantics().proxy() == expected_proxy));
        }
    }

    #[test]
    fn bulk_reduction_preserves_the_planner_scope() {
        let operation = OperationContext::new(
            DynamicOpId::new(0, 0, 0, StaticOpId::new(2), []),
            OperationKind::AsyncIssue,
            context().active_mask(),
        );
        let source = access_batch(
            &operation,
            PhysicalAccessKind::Read,
            PhysicalAccessSpace::Shared,
            1,
        );
        let destination = access_batch(
            &operation,
            PhysicalAccessKind::AtomicReadModifyWrite,
            PhysicalAccessSpace::Global,
            2,
        )
        .with_memory_semantics(crate::MemoryAccessSemantics::async_reduction());

        let effect =
            AsyncGroupIssueEffect::new(operation, AsyncGroupDomain::Bulk, [source], [destination])
                .unwrap();
        let semantics = effect.destination_accesses()[0]
            .descriptor()
            .memory_semantics();
        assert_eq!(semantics.class(), MemoryAccessClass::Reduction);
        assert_eq!(semantics.order(), MemoryOrder::Relaxed);
        assert_eq!(semantics.scope(), Some(MemoryScope::Gpu));
        assert_eq!(semantics.proxy(), MemoryProxy::Async);
    }

    #[test]
    fn issue_commit_wait_reaches_quiescence() {
        let context = context();
        let hub = AsyncGroupHub::new(context.topology());
        let mask = context.active_mask();
        hub.issue(&context, AsyncGroupDomain::CpAsync, mask)
            .unwrap();
        hub.commit(&context, AsyncGroupDomain::CpAsync, mask)
            .unwrap();
        hub.wait_group(&context, AsyncGroupDomain::CpAsync, mask, 0, false)
            .unwrap();
        hub.validate_quiescent().unwrap();
    }

    #[test]
    fn uncommitted_and_unwaited_groups_fail_quiescence() {
        let context = context();
        let mask = context.active_mask();

        let uncommitted = AsyncGroupHub::new(context.topology());
        uncommitted
            .issue(&context, AsyncGroupDomain::CpAsync, mask)
            .unwrap();
        assert!(uncommitted
            .validate_quiescent()
            .unwrap_err()
            .to_string()
            .contains("uncommitted issue"));

        let pending = AsyncGroupHub::new(context.topology());
        pending
            .issue(&context, AsyncGroupDomain::Bulk, mask)
            .unwrap();
        pending
            .commit(&context, AsyncGroupDomain::Bulk, mask)
            .unwrap();
        assert!(pending
            .validate_quiescent()
            .unwrap_err()
            .to_string()
            .contains("pending full completion"));
    }

    #[test]
    fn wait_group_keeps_only_the_requested_recent_groups() {
        let context = context();
        let hub = AsyncGroupHub::new(context.topology());
        let mask = context.active_mask();
        for _ in 0..3 {
            hub.issue(&context, AsyncGroupDomain::Bulk, mask).unwrap();
            hub.commit(&context, AsyncGroupDomain::Bulk, mask).unwrap();
        }
        hub.wait_group(&context, AsyncGroupDomain::Bulk, mask, 1, true)
            .unwrap();
        let error = hub.validate_quiescent().unwrap_err().to_string();
        assert!(error.contains("3 execution batch(es) / 3 issue(s) pending full completion"));
        assert!(error.contains("1 batch(es) still reading source"));
        assert!(error.contains("2 batch(es) with source reads complete"));
        hub.wait_group(&context, AsyncGroupDomain::Bulk, mask, 0, false)
            .unwrap();
        hub.validate_quiescent().unwrap();
    }

    #[test]
    fn bulk_read_wait_releases_source_reads_without_full_completion() {
        let context = context();
        let hub = AsyncGroupHub::new(context.topology());
        let mask = context.active_mask();
        hub.issue(&context, AsyncGroupDomain::Bulk, mask).unwrap();
        hub.commit(&context, AsyncGroupDomain::Bulk, mask).unwrap();

        hub.wait_group(&context, AsyncGroupDomain::Bulk, mask, 0, true)
            .unwrap();

        let state = hub.state(context.global_warp_id()).unwrap().lock().unwrap();
        for lane in mask {
            let bulk = state.lanes[lane].domain(AsyncGroupDomain::Bulk);
            assert_eq!(bulk.pending_read_count(), 0);
            assert_eq!(bulk.read_complete_count(), 1);
            assert_eq!(bulk.committed_groups.len(), 1);
        }
        drop(state);
        let error = hub.validate_quiescent().unwrap_err().to_string();
        assert!(error.contains("source reads complete"));

        let progress = hub.pump().unwrap();
        assert_eq!(progress.completed_operations, 2);
        hub.validate_quiescent().unwrap();
    }

    #[test]
    fn wait_cannot_observe_completion_before_mode_publication() {
        let context = context();
        let lane = WarpMask::from_lanes([0]).unwrap();
        let hub = Arc::new(AsyncGroupHub::new(context.topology()));
        hub.issue(&context, AsyncGroupDomain::Bulk, lane).unwrap();
        hub.commit(&context, AsyncGroupDomain::Bulk, lane).unwrap();
        let wait_plan = hub
            .wait_plan(&context, AsyncGroupDomain::Bulk, lane, 0, true)
            .unwrap();
        let action = hub.pending_completion_actions().remove(0);

        let (publication_entered_tx, publication_entered_rx) = mpsc::channel();
        let (release_publication_tx, release_publication_rx) = mpsc::channel();
        let completion_hub = Arc::clone(&hub);
        let completion = thread::spawn(move || {
            completion_hub.apply_completion_action_detailed_with_outcome(&action, |_| {
                publication_entered_tx.send(()).unwrap();
                release_publication_rx.recv().unwrap();
                Ok(())
            })
        });
        publication_entered_rx.recv().unwrap();

        let (wait_started_tx, wait_started_rx) = mpsc::channel();
        let (wait_finished_tx, wait_finished_rx) = mpsc::channel();
        let wait_hub = Arc::clone(&hub);
        let waiter = thread::spawn(move || {
            wait_started_tx.send(()).unwrap();
            wait_finished_tx
                .send(wait_hub.complete_wait(wait_plan).is_ok())
                .unwrap();
        });
        wait_started_rx.recv().unwrap();
        assert!(wait_finished_rx
            .recv_timeout(Duration::from_millis(50))
            .is_err());

        release_publication_tx.send(()).unwrap();
        completion.join().unwrap().unwrap();
        assert!(wait_finished_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap());
        waiter.join().unwrap();
    }

    #[test]
    fn empty_commit_and_empty_wait_are_trivially_complete() {
        let context = context();
        let hub = AsyncGroupHub::new(context.topology());
        let mask = context.active_mask();
        hub.commit(&context, AsyncGroupDomain::Bulk, mask).unwrap();
        hub.wait_group(&context, AsyncGroupDomain::Bulk, mask, 0, true)
            .unwrap();
        let state = hub.state(context.global_warp_id()).unwrap().lock().unwrap();
        for lane in mask {
            assert!(state.lanes[lane].bulk.committed_groups.is_empty());
        }
        drop(state);
        hub.validate_quiescent().unwrap();
    }

    #[test]
    fn issue_membership_is_lane_local_across_a_full_mask_commit() {
        let context = context();
        let hub = AsyncGroupHub::new(context.topology());
        let issuing_lane = WarpMask::from_lanes([0]).unwrap();
        let full_mask = context.active_mask();

        hub.issue(&context, AsyncGroupDomain::CpAsync, issuing_lane)
            .unwrap();
        let outcome = hub
            .commit_detailed(
                hub.commit_plan(&context, AsyncGroupDomain::CpAsync, full_mask)
                    .unwrap(),
            )
            .unwrap();

        assert_eq!(outcome.groups().len(), 1);
        assert_eq!(outcome.groups()[0].id().lane(), 0);
        let state = hub.state(context.global_warp_id()).unwrap().lock().unwrap();
        assert_eq!(state.lanes[0].cp_async.committed_groups[0].issues.len(), 1);
        assert!(state.lanes[1].cp_async.committed_groups[0]
            .issues
            .is_empty());
        drop(state);

        hub.wait_group(&context, AsyncGroupDomain::CpAsync, full_mask, 0, false)
            .unwrap();
        hub.validate_quiescent().unwrap();
    }

    #[test]
    fn exact_bulk_batch_commits_one_lane_local_token_per_issuing_thread() {
        let context = context();
        let mask = context.active_mask();
        let operation = OperationContext::new(
            DynamicOpId::new(0, 0, 0, StaticOpId::new(3), []),
            OperationKind::AsyncIssue,
            mask,
        );
        let effect = AsyncGroupIssueBatchEffect::new(
            operation,
            AsyncGroupDomain::Bulk,
            mask.into_iter().map(|lane| (lane, Vec::new(), Vec::new())),
        )
        .unwrap();
        let hub = AsyncGroupHub::new(context.topology());

        hub.issue_exact_batch(&context, &effect, Vec::new())
            .unwrap();
        let outcome = hub
            .commit_detailed(
                hub.commit_plan(&context, AsyncGroupDomain::Bulk, mask)
                    .unwrap(),
            )
            .unwrap();

        assert_eq!(outcome.groups().len(), mask.len());
        for group in outcome.groups() {
            assert_eq!(group.members().len(), 1);
            assert_eq!(group.members()[0].domain(), AsyncGroupDomain::Bulk);
            assert_eq!(
                group.members()[0].operation().active_mask(),
                WarpMask::from_lanes([group.id().lane()]).unwrap()
            );
        }
        hub.wait_group(&context, AsyncGroupDomain::Bulk, mask, 0, false)
            .unwrap();
        hub.validate_quiescent().unwrap();
    }

    #[test]
    fn large_wait_operands_select_the_fifo_prefix() {
        let context = context();
        let hub = AsyncGroupHub::new(context.topology());
        let mask = context.active_mask();
        for _ in 0..80 {
            hub.issue(&context, AsyncGroupDomain::Bulk, mask).unwrap();
            hub.commit(&context, AsyncGroupDomain::Bulk, mask).unwrap();
        }
        for pending in [i64::MAX, 255, 64] {
            hub.wait_group(&context, AsyncGroupDomain::Bulk, mask, pending, false)
                .unwrap();
            let state = hub.state(context.global_warp_id()).unwrap().lock().unwrap();
            for lane in mask {
                assert_eq!(
                    state.lanes[lane].bulk.committed_groups.len(),
                    80.min(pending as usize)
                );
            }
        }
        assert!(hub
            .wait_group(&context, AsyncGroupDomain::Bulk, mask, -1, false)
            .is_err());
        hub.wait_group(&context, AsyncGroupDomain::Bulk, mask, 0, false)
            .unwrap();
        hub.validate_quiescent().unwrap();
    }

    #[test]
    fn committed_bulk_group_advances_to_full_completion_during_exit_drain() {
        let context = context();
        let hub = AsyncGroupHub::new(context.topology());
        let lane = WarpMask::from_lanes([0]).unwrap();

        hub.issue(&context, AsyncGroupDomain::Bulk, lane).unwrap();
        hub.commit(&context, AsyncGroupDomain::Bulk, lane).unwrap();

        assert!(hub.validate_quiescent().is_err());
        assert_eq!(
            hub.pending_completion_action_ids(),
            hub.pending_completion_actions()
                .into_iter()
                .map(|action| action.id())
                .collect::<Vec<_>>()
        );
        assert_eq!(hub.pump().unwrap().completed_operations, 1);
        let reads_complete = hub.validate_quiescent().unwrap_err().to_string();
        assert!(reads_complete.contains("1 batch(es) with source reads complete"));
        assert_eq!(
            hub.pending_completion_action_ids(),
            hub.pending_completion_actions()
                .into_iter()
                .map(|action| action.id())
                .collect::<Vec<_>>()
        );
        assert_eq!(hub.pump().unwrap().completed_operations, 1);
        assert!(hub.pending_completion_action_ids().is_empty());
        hub.validate_quiescent().unwrap();
    }

    #[test]
    fn completion_pump_tracks_and_reactivates_only_active_warps() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let contexts = topology.warp_contexts().collect::<Vec<_>>();
        let lane = WarpMask::from_lanes([0]).unwrap();
        let hub = AsyncGroupHub::new(topology);

        assert!(hub
            .active_completion_warps
            .iter()
            .all(|active| !active.load(Ordering::Acquire)));
        for context in &contexts {
            hub.issue(context, AsyncGroupDomain::CpAsync, lane).unwrap();
            hub.commit(context, AsyncGroupDomain::CpAsync, lane)
                .unwrap();
        }
        assert!(hub
            .active_completion_warps
            .iter()
            .all(|active| active.load(Ordering::Acquire)));

        let first = hub.apply_first_enabled_completion().unwrap().unwrap();
        assert_eq!(first.action().group_id().global_warp_id(), 0);
        assert!(hub.active_completion_warps[0].load(Ordering::Acquire));
        assert!(hub.active_completion_warps[1].load(Ordering::Acquire));
        let first_full = hub.apply_first_enabled_completion().unwrap().unwrap();
        assert_eq!(first_full.action().group_id().global_warp_id(), 0);
        assert!(!hub.active_completion_warps[0].load(Ordering::Acquire));
        assert!(hub.active_completion_warps[1].load(Ordering::Acquire));

        let second = hub.apply_first_enabled_completion().unwrap().unwrap();
        assert_eq!(second.action().group_id().global_warp_id(), 1);
        assert!(hub.active_completion_warps[1].load(Ordering::Acquire));
        let second_full = hub.apply_first_enabled_completion().unwrap().unwrap();
        assert_eq!(second_full.action().group_id().global_warp_id(), 1);
        assert!(hub
            .active_completion_warps
            .iter()
            .all(|active| !active.load(Ordering::Acquire)));
        assert!(hub.apply_first_enabled_completion().unwrap().is_none());

        for context in &contexts {
            hub.wait_group(context, AsyncGroupDomain::CpAsync, lane, 0, false)
                .unwrap();
        }
        hub.issue(&contexts[1], AsyncGroupDomain::CpAsync, lane)
            .unwrap();
        hub.commit(&contexts[1], AsyncGroupDomain::CpAsync, lane)
            .unwrap();
        assert!(!hub.active_completion_warps[0].load(Ordering::Acquire));
        assert!(hub.active_completion_warps[1].load(Ordering::Acquire));
        assert_eq!(hub.pump().unwrap().completed_operations, 1);
        assert!(hub.active_completion_warps[1].load(Ordering::Acquire));
        assert_eq!(hub.pump().unwrap().completed_operations, 1);
        assert!(!hub.active_completion_warps[1].load(Ordering::Acquire));
        hub.wait_group(&contexts[1], AsyncGroupDomain::CpAsync, lane, 0, false)
            .unwrap();
        hub.validate_quiescent().unwrap();
    }

    #[test]
    fn completion_action_ids_do_not_depend_on_cross_warp_commit_order() {
        fn action_ids(order: [usize; 2]) -> BTreeMap<usize, (u64, u64)> {
            let topology = LaunchTopology::new(1, 1, 2).unwrap();
            let contexts = topology.warp_contexts().collect::<Vec<_>>();
            let lane = WarpMask::from_lanes([0]).unwrap();
            let hub = AsyncGroupHub::new(topology);
            for context in &contexts {
                hub.issue(context, AsyncGroupDomain::Bulk, lane).unwrap();
            }
            order
                .into_iter()
                .map(|warp_id| {
                    let plan = hub
                        .commit_plan(&contexts[warp_id], AsyncGroupDomain::Bulk, lane)
                        .unwrap();
                    let outcome = hub.commit_detailed(plan).unwrap();
                    let group = &outcome.groups()[0];
                    (
                        warp_id,
                        (
                            group.source_read_action_id().get(),
                            group.full_action_id().get(),
                        ),
                    )
                })
                .collect()
        }

        assert_eq!(action_ids([0, 1]), action_ids([1, 0]));
    }

    #[test]
    fn multi_lane_commit_and_wait_validate_before_mutating_any_lane() {
        let context = context();
        let mask = context.active_mask();

        let commit_hub = AsyncGroupHub::new(context.topology());
        commit_hub
            .issue(&context, AsyncGroupDomain::CpAsync, mask)
            .unwrap();
        {
            let mut state = commit_hub
                .state(context.global_warp_id())
                .unwrap()
                .lock()
                .unwrap();
            state.lanes[1].cp_async.next_group_ordinal = u64::MAX;
        }
        let plan = commit_hub
            .commit_plan(&context, AsyncGroupDomain::CpAsync, mask)
            .unwrap();
        assert!(commit_hub.commit_detailed(plan).is_err());
        let state = commit_hub
            .state(context.global_warp_id())
            .unwrap()
            .lock()
            .unwrap();
        assert_eq!(state.lanes[0].cp_async.open_issues.len(), 1);
        assert!(state.lanes[0].cp_async.committed_groups.is_empty());
        drop(state);

        let wait_hub = AsyncGroupHub::new(context.topology());
        wait_hub
            .issue(&context, AsyncGroupDomain::Bulk, mask)
            .unwrap();
        let commit = wait_hub
            .commit_detailed(
                wait_hub
                    .commit_plan(&context, AsyncGroupDomain::Bulk, mask)
                    .unwrap(),
            )
            .unwrap();
        let lane_zero = &commit.groups()[0];
        for action_id in [
            lane_zero.source_read_action_id(),
            lane_zero.full_action_id(),
        ] {
            let action = wait_hub
                .pending_completion_actions()
                .into_iter()
                .find(|action| action.id() == action_id)
                .expect("committed test action is enabled");
            wait_hub
                .apply_completion_action_detailed_with_outcome(&action, |_| Ok(()))
                .unwrap();
        }
        let wait = wait_hub
            .wait_plan(&context, AsyncGroupDomain::Bulk, mask, 0, false)
            .unwrap();
        assert!(wait_hub.complete_wait(wait).is_err());
        let state = wait_hub
            .state(context.global_warp_id())
            .unwrap()
            .lock()
            .unwrap();
        assert_eq!(state.lanes[0].bulk.committed_groups.len(), 1);
        assert_eq!(state.lanes[1].bulk.committed_groups.len(), 1);
    }

    #[test]
    fn empty_group_counts_toward_wait_depth_without_blocking_completion() {
        let context = context();
        let lane = WarpMask::from_lanes([0]).unwrap();
        let hub = AsyncGroupHub::new(context.topology());
        hub.issue(&context, AsyncGroupDomain::Bulk, lane).unwrap();
        hub.commit(&context, AsyncGroupDomain::Bulk, lane).unwrap();
        hub.commit(&context, AsyncGroupDomain::Bulk, lane).unwrap();

        hub.wait_group(&context, AsyncGroupDomain::Bulk, lane, 1, false)
            .unwrap();
        let state = hub.state(context.global_warp_id()).unwrap().lock().unwrap();
        let remaining = &state.lanes[0].bulk.committed_groups;
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id.ordinal(), 1);
        assert!(remaining[0].issues.is_empty());
        drop(state);

        hub.wait_group(&context, AsyncGroupDomain::Bulk, lane, 0, true)
            .unwrap();
        hub.validate_quiescent().unwrap();
    }

    #[test]
    fn full_completion_publishes_one_group_across_multiple_allocations() {
        let context = context();
        let lane = WarpMask::from_lanes([0]).unwrap();
        let memory = GlobalMemory::new();
        let first = memory.allocate_zeroed(4).unwrap();
        let second = memory.allocate_zeroed(4).unwrap();
        let first_view = memory.full_view(first).unwrap();
        let second_view = memory.full_view(second).unwrap();
        let writes = vec![
            memory
                .defer_write_bytes(&first_view, 0, vec![1, 2, 3, 4])
                .unwrap(),
            memory
                .defer_write_bytes(&second_view, 0, vec![5, 6, 7, 8])
                .unwrap(),
        ];
        let hub = AsyncGroupHub::new(context.topology());
        hub.issue_unmodeled(&context, AsyncGroupDomain::Bulk, lane, writes)
            .unwrap();
        hub.commit(&context, AsyncGroupDomain::Bulk, lane).unwrap();

        let read = hub.pending_completion_actions().remove(0);
        hub.apply_completion_action_detailed_with_outcome(&read, |_| Ok(()))
            .unwrap();
        assert_eq!(memory.read_bytes(&first_view, 0, 4).unwrap(), [0; 4]);
        assert_eq!(memory.read_bytes(&second_view, 0, 4).unwrap(), [0; 4]);

        let full = hub.pending_completion_actions().remove(0);
        hub.apply_completion_action_detailed_with_outcome(&full, |_| Ok(()))
            .unwrap();
        assert_eq!(memory.read_bytes(&first_view, 0, 4).unwrap(), [1, 2, 3, 4]);
        assert_eq!(memory.read_bytes(&second_view, 0, 4).unwrap(), [5, 6, 7, 8]);

        let state = hub.state(context.global_warp_id()).unwrap().lock().unwrap();
        let completed = &state.lanes[0].bulk.committed_groups[0];
        assert_eq!(completed.completion, GroupCompletion::FullyComplete);
        assert_eq!(completed.issues[0].writes.len(), 2);
        drop(state);

        hub.wait_group(&context, AsyncGroupDomain::Bulk, lane, 0, false)
            .unwrap();
        let state = hub.state(context.global_warp_id()).unwrap().lock().unwrap();
        assert!(state.lanes[0].bulk.committed_groups.is_empty());
    }
}
