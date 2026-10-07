use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::{
    ClusterBarrierArrivalOutcome, ClusterBarrierHub, ClusterBarrierId, DiagnosticLabel,
    DynamicOpId, EngineError, LaunchTopology, NamedBarrierArrivalOutcome, NamedBarrierHub,
    NamedBarrierId, PhysicalBarrierHub, PhysicalBarrierId, PhysicalCompletionAction,
    MemoryAccessSemantics, PhysicalByteSpan, PhysicalCompletionActionId,
    PhysicalMbarrierArrivalOutcome,
    SynchronizationError, WarpContext,
    WarpMask, WarpValue, WARP_SIZE,
};

use super::PhysicalPtr;

const MAX_MBARRIER_COUNT: u64 = (1_u64 << 20) - 1;

fn one_lane(lane: usize) -> WarpMask {
    WarpMask::from_bits(1_u32 << lane)
}

fn expected_arrival_count(value: i64) -> Result<u64, EngineError> {
    let value = u64::try_from(value)
        .map_err(|_| EngineError::message("negative mbarrier arrival count"))?;
    if !(1..=MAX_MBARRIER_COUNT).contains(&value) {
        return Err(EngineError::message(format!(
            "mbarrier arrival count must be in 1..={MAX_MBARRIER_COUNT}, got {value}"
        )));
    }
    Ok(value)
}

fn transaction_count(value: i64) -> Result<u64, EngineError> {
    let value = u64::try_from(value)
        .map_err(|_| EngineError::message("negative mbarrier transaction count"))?;
    if value > MAX_MBARRIER_COUNT {
        return Err(EngineError::message(format!(
            "mbarrier transaction count must be in 0..={MAX_MBARRIER_COUNT}, got {value}"
        )));
    }
    Ok(value)
}

pub fn require_full_warp_sync(mask: WarpMask, operation: &str) -> Result<(), EngineError> {
    require_full_warp_sync_labeled(mask, &DiagnosticLabel::new(operation))
}

pub(crate) fn require_full_warp_sync_labeled(
    mask: WarpMask,
    operation: &DiagnosticLabel,
) -> Result<(), EngineError> {
    if mask != WarpMask::FULL {
        return Err(operation.warp_collective_divergence(mask));
    }
    Ok(())
}

/// Syntactic and effective ordering carried by `barrier.cluster.arrive`.
///
/// NOTE (P3 sync-subset dissolution): the Release/Relaxed distinction currently
/// has **zero readers** in the fixed-state command path. The retired
/// `ResolvedSynchronizationDetails` projection copied `arrival_semantics` /
/// `wait_semantics` / `publishes_memory` / `acquires_memory` into every stored
/// cluster-barrier summary, but `command_from_summary` never consulted them —
/// `FixedSyncCommandKind::ClusterBarrierArrive` carries no ordering field. Now
/// that the summary stores the plan itself, the data survives verbatim and the
/// accessors are simply unread, which is why `dead_code` reports them and
/// `DefaultRelease` as never constructed. Kept, not deleted: a fixed-state
/// model that distinguishes relaxed from release arrivals is the intended
/// consumer, and it can read them straight off the stored plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ClusterBarrierArrivalSemantics {
    /// No PTX semantic qualifier. PTX defines this as release.
    DefaultRelease,
    Release,
    Relaxed,
}

impl ClusterBarrierArrivalSemantics {
    pub const fn publishes_memory(self) -> bool {
        !matches!(self, Self::Relaxed)
    }
}

/// Syntactic and effective ordering carried by `barrier.cluster.wait`.
///
/// Same standing as [`ClusterBarrierArrivalSemantics`]: unread by the
/// fixed-state command path since the P3 sync-subset dissolution, retained in
/// the stored plan for a future consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ClusterBarrierWaitSemantics {
    /// No PTX acquire qualifier. PTX defines this as acquire.
    DefaultAcquire,
    Acquire,
}

impl ClusterBarrierWaitSemantics {
    pub const fn acquires_memory(self) -> bool {
        true
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ClusterBarrierContract {
    barrier_id: ClusterBarrierId,
    topology: LaunchTopology,
    warp_id: usize,
    arrival_mask: WarpMask,
    participant_warps: Box<[usize]>,
}

impl ClusterBarrierContract {
    fn new(kernel_index: usize, context: &WarpContext) -> Result<Self, EngineError> {
        let cluster_id = context.cluster_id();
        let participant_warps = context
            .topology()
            .cluster_warp_range(cluster_id)
            .ok_or_else(|| EngineError::message("cluster barrier cluster ID is outside topology"))?
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Self {
            barrier_id: ClusterBarrierId::new(kernel_index, cluster_id),
            topology: context.topology(),
            warp_id: context.global_warp_id(),
            arrival_mask: context.active_mask(),
            participant_warps,
        })
    }
}

macro_rules! cluster_contract_accessors {
    () => {
        pub const fn barrier_id(&self) -> ClusterBarrierId {
            self.contract.barrier_id
        }

        pub const fn topology(&self) -> LaunchTopology {
            self.contract.topology
        }

        pub const fn warp_id(&self) -> usize {
            self.contract.warp_id
        }

        pub const fn arrival_mask(&self) -> WarpMask {
            self.contract.arrival_mask
        }

        pub fn participant_warps(&self) -> &[usize] {
            &self.contract.participant_warps
        }

        pub fn participant_count(&self) -> usize {
            self.contract.participant_warps.len()
        }
    };
}

/// Fully resolved nonblocking cluster-barrier arrival.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClusterBarrierArrivePlan {
    contract: ClusterBarrierContract,
    arrival_semantics: ClusterBarrierArrivalSemantics,
    aligned: bool,
}

impl ClusterBarrierArrivePlan {
    cluster_contract_accessors!();

    pub const fn arrival_semantics(&self) -> ClusterBarrierArrivalSemantics {
        self.arrival_semantics
    }

    pub const fn publishes_memory(&self) -> bool {
        self.arrival_semantics.publishes_memory()
    }

    pub const fn aligned(&self) -> bool {
        self.aligned
    }

    pub fn apply(
        &self,
        barriers: &ClusterBarrierHub,
    ) -> Result<ClusterBarrierArrivalOutcome, EngineError> {
        barriers
            .arrive_resolved_with_alignment(
                self.topology(),
                self.barrier_id().cluster_id(),
                self.warp_id(),
                self.arrival_mask(),
                self.aligned,
            )
            .map_err(Into::into)
    }
}

/// Fully resolved split cluster-barrier wait.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClusterBarrierWaitPlan {
    contract: ClusterBarrierContract,
    wait_semantics: ClusterBarrierWaitSemantics,
    aligned: bool,
}

impl ClusterBarrierWaitPlan {
    cluster_contract_accessors!();

    pub const fn wait_semantics(&self) -> ClusterBarrierWaitSemantics {
        self.wait_semantics
    }

    pub const fn acquires_memory(&self) -> bool {
        self.wait_semantics.acquires_memory()
    }

    pub const fn aligned(&self) -> bool {
        self.aligned
    }

    pub fn register(
        &self,
        barriers: &Arc<ClusterBarrierHub>,
    ) -> Result<ClusterBarrierWaitRegistration, EngineError> {
        let wait = barriers.wait_resolved_with_alignment(
            self.topology(),
            self.barrier_id().cluster_id(),
            self.warp_id(),
            self.arrival_mask(),
            self.aligned,
        )?;
        let outcome = ClusterBarrierRegistrationOutcome::new(
            wait.generation(),
            wait.completed_at_registration(),
        );
        Ok(ClusterBarrierWaitRegistration {
            plan: self.clone(),
            outcome,
            wait,
        })
    }
}

/// Exact result of registering a split cluster-barrier wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterBarrierRegistrationOutcome {
    generation: u64,
    completed_now: bool,
}

impl ClusterBarrierRegistrationOutcome {
    pub const fn new(generation: u64, completed_now: bool) -> Self {
        Self {
            generation,
            completed_now,
        }
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }

    pub const fn completed_now(self) -> bool {
        self.completed_now
    }

    pub const fn released_generation(self) -> Option<u64> {
        if self.completed_now {
            Some(self.generation)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClusterBarrierWaitResumePlan {
    plan: ClusterBarrierWaitPlan,
    generation: u64,
}

impl ClusterBarrierWaitResumePlan {
    pub const fn plan(&self) -> &ClusterBarrierWaitPlan {
        &self.plan
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn consumed_generation(&self) -> u64 {
        self.generation
    }
}

pub struct ClusterBarrierWaitRegistration {
    plan: ClusterBarrierWaitPlan,
    outcome: ClusterBarrierRegistrationOutcome,
    wait: crate::ClusterBarrierWait,
}

impl ClusterBarrierWaitRegistration {
    pub const fn outcome(&self) -> ClusterBarrierRegistrationOutcome {
        self.outcome
    }

    pub async fn resume(self) -> Result<ClusterBarrierWaitResumePlan, EngineError> {
        self.wait.await?;
        Ok(ClusterBarrierWaitResumePlan {
            plan: self.plan,
            generation: self.outcome.generation(),
        })
    }
}

pub fn plan_cluster_barrier_arrive(
    kernel_index: usize,
    context: &WarpContext,
    arrival_semantics: ClusterBarrierArrivalSemantics,
    aligned: bool,
) -> Result<ClusterBarrierArrivePlan, EngineError> {
    Ok(ClusterBarrierArrivePlan {
        contract: ClusterBarrierContract::new(kernel_index, context)?,
        arrival_semantics,
        aligned,
    })
}

pub fn plan_cluster_barrier_wait(
    kernel_index: usize,
    context: &WarpContext,
    wait_semantics: ClusterBarrierWaitSemantics,
    aligned: bool,
) -> Result<ClusterBarrierWaitPlan, EngineError> {
    Ok(ClusterBarrierWaitPlan {
        contract: ClusterBarrierContract::new(kernel_index, context)?,
        wait_semantics,
        aligned,
    })
}

/// Fully resolved lane targets for one warp-level `mbarrier.init` operation.
///
/// The target list preserves one entry per active lane.  Protocol layers
/// collapse repeated physical slots because `mbarrier.init` is idempotent for
/// same-address lanes, while still validating distinct lane-varying targets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalMbarrierInitPlan {
    barrier_ids: Box<[PhysicalBarrierId]>,
    expected_arrivals: u64,
    layout_v1: bool,
}

impl PhysicalMbarrierInitPlan {
    pub(crate) fn with_layout(mut self, layout_v1: bool) -> Self {
        self.layout_v1 = layout_v1;
        self
    }
    pub fn barrier_ids(&self) -> &[PhysicalBarrierId] {
        &self.barrier_ids
    }

    pub const fn expected_arrivals(&self) -> u64 {
        self.expected_arrivals
    }

    pub fn is_empty(&self) -> bool {
        self.barrier_ids.is_empty()
    }

    pub fn apply(&self, mbarriers: &PhysicalBarrierHub) -> Result<(), EngineError> {
        let limit = crate::hardware_barriers::mbarrier_arrival_limit(self.layout_v1);
        if !(1..=limit).contains(&self.expected_arrivals) {
            return Err(EngineError::message(format!(
                "mbarrier arrival count must be in 1..={limit}, got {}",
                self.expected_arrivals
            )));
        }
        mbarriers
            .init_many_layout(&self.barrier_ids, self.expected_arrivals, self.layout_v1)
            .map_err(Into::into)
    }
}

/// Fully resolved target and counts for one warp-level mbarrier arrival.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalMbarrierArrivePlan {
    barrier_id: PhysicalBarrierId,
    warp_id: usize,
    arrival_count: u64,
    expected_transactions: Option<u64>,
    drop: bool,
    release: bool,
}

/// One target in a lane-varying warp-level mbarrier arrival.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalMbarrierArriveBatchEntry {
    plan: PhysicalMbarrierArrivePlan,
    arrival_mask: WarpMask,
}

impl PhysicalMbarrierArriveBatchEntry {
    pub const fn plan(self) -> PhysicalMbarrierArrivePlan {
        self.plan
    }

    pub const fn arrival_mask(self) -> WarpMask {
        self.arrival_mask
    }
}

/// Fully resolved target set for one lane-varying warp-level arrival.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalMbarrierArriveBatchPlan {
    entries: Box<[PhysicalMbarrierArriveBatchEntry]>,
}

/// One physical target in a standalone `mbarrier.expect_tx` operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalMbarrierExpectTxEntry {
    barrier_id: PhysicalBarrierId,
    expected_transactions: u64,
}

impl PhysicalMbarrierExpectTxEntry {
    pub const fn barrier_id(self) -> PhysicalBarrierId {
        self.barrier_id
    }

    pub const fn expected_transactions(self) -> u64 {
        self.expected_transactions
    }
}

/// Fully resolved target set for one standalone `mbarrier.expect_tx`.
///
/// Lanes naming the same physical barrier are aggregated into one entry. Both
/// numeric execution and checker modes consume this plan so target resolution
/// and transaction-count validation have one owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalMbarrierExpectTxPlan {
    entries: Box<[PhysicalMbarrierExpectTxEntry]>,
}

/// Numeric generations selected atomically by one standalone expectation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalMbarrierExpectTxOutcome {
    generations: Box<[(PhysicalBarrierId, u64)]>,
}

impl PhysicalMbarrierExpectTxOutcome {
    pub fn generations(&self) -> &[(PhysicalBarrierId, u64)] {
        &self.generations
    }
}

impl PhysicalMbarrierExpectTxPlan {
    pub fn entries(&self) -> &[PhysicalMbarrierExpectTxEntry] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn apply(
        &self,
        mbarriers: &PhysicalBarrierHub,
    ) -> Result<PhysicalMbarrierExpectTxOutcome, EngineError> {
        let expectations = self
            .entries
            .iter()
            .map(|entry| (entry.barrier_id(), entry.expected_transactions()))
            .collect::<Vec<_>>();
        let generations = mbarriers.expect_tx_many(&expectations)?;
        Ok(PhysicalMbarrierExpectTxOutcome { generations })
    }
}

impl PhysicalMbarrierArriveBatchPlan {
    pub(crate) fn with_semantics(mut self, drop: bool, release: bool) -> Self {
        for entry in &mut self.entries {
            entry.plan.drop = drop;
            entry.plan.release = release;
        }
        self
    }

    pub fn entries(&self) -> &[PhysicalMbarrierArriveBatchEntry] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn apply(
        &self,
        mbarriers: &PhysicalBarrierHub,
    ) -> Result<PhysicalMbarrierArrivalBatchOutcome, EngineError> {
        self.apply_with_outcome(mbarriers, |_| Ok(()))
    }

    pub fn apply_with_outcome(
        &self,
        mbarriers: &PhysicalBarrierHub,
        publish_before_wake: impl FnOnce(
            &PhysicalMbarrierArrivalBatchOutcome,
        ) -> Result<(), EngineError>,
    ) -> Result<PhysicalMbarrierArrivalBatchOutcome, EngineError> {
        if self.entries.iter().any(|entry| entry.plan.drop) {
            mbarriers.drop_expected_arrivals(
                self.entries
                    .iter()
                    .filter(|entry| entry.plan.drop)
                    .map(|entry| (entry.plan.barrier_id(), entry.plan.arrival_count())),
            )?;
        }
        let arrivals = self
            .entries
            .iter()
            .map(|entry| {
                let plan = entry.plan();
                (
                    plan.barrier_id(),
                    plan.warp_id(),
                    plan.arrival_count(),
                    plan.expected_transactions().unwrap_or(0),
                )
            })
            .collect::<Vec<_>>();
        let outcomes = mbarriers
            .arrive_many_with_outcomes(&arrivals, |outcomes| {
                publish_before_wake(&PhysicalMbarrierArrivalBatchOutcome::new(outcomes.to_vec()))
            })
            .map_err(|error| EngineError::message(error.to_string()))?;
        Ok(PhysicalMbarrierArrivalBatchOutcome { outcomes })
    }
}

/// Numeric outcomes aligned one-for-one with a batch arrival plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalMbarrierArrivalBatchOutcome {
    outcomes: Box<[PhysicalMbarrierArrivalOutcome]>,
}

impl PhysicalMbarrierArrivalBatchOutcome {
    pub fn new(outcomes: impl Into<Box<[PhysicalMbarrierArrivalOutcome]>>) -> Self {
        Self {
            outcomes: outcomes.into(),
        }
    }

    pub fn outcomes(&self) -> &[PhysicalMbarrierArrivalOutcome] {
        &self.outcomes
    }
}

impl PhysicalMbarrierArrivePlan {
    /// Ordinary memory publication; execution ordering and counter updates
    /// also occur for relaxed arrivals.
    pub const fn is_release(self) -> bool {
        self.release
    }

    pub const fn is_drop(self) -> bool {
        self.drop
    }

    pub const fn barrier_id(self) -> PhysicalBarrierId {
        self.barrier_id
    }

    pub const fn warp_id(self) -> usize {
        self.warp_id
    }

    pub const fn arrival_count(self) -> u64 {
        self.arrival_count
    }

    pub const fn expected_transactions(self) -> Option<u64> {
        self.expected_transactions
    }

    pub fn apply(
        self,
        mbarriers: &PhysicalBarrierHub,
    ) -> Result<PhysicalMbarrierArrivalOutcome, EngineError> {
        self.apply_with_outcome(mbarriers, |_| Ok(()))
    }

    pub fn apply_with_outcome(
        self,
        mbarriers: &PhysicalBarrierHub,
        publish_before_wake: impl FnOnce(&PhysicalMbarrierArrivalOutcome) -> Result<(), EngineError>,
    ) -> Result<PhysicalMbarrierArrivalOutcome, EngineError> {
        if self.drop {
            mbarriers.drop_expected_arrivals([(self.barrier_id, self.arrival_count)])?;
        }
        Ok(match self.expected_transactions {
            Some(transactions) => mbarriers
                .arrive_expect_tx_with_outcome(
                    self.barrier_id,
                    self.warp_id,
                    self.arrival_count,
                    transactions,
                    publish_before_wake,
                )
                .map_err(|error| EngineError::message(error.to_string()))?,
            None => mbarriers
                .arrive_with_outcome(
                    self.barrier_id,
                    self.warp_id,
                    self.arrival_count,
                    publish_before_wake,
                )
                .map_err(|error| EngineError::message(error.to_string()))?,
        })
    }
}

/// Fully resolved target and phase for one warp-level mbarrier wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalMbarrierWaitPlan {
    barrier_id: PhysicalBarrierId,
    warp_id: usize,
    requested_phase: u64,
    conditional: bool,
    acquire: bool,
}

/// One lane's resolved wait on a declared synchronization word.
///
/// `accepted` is the position, in the word's write history, of the first write
/// whose value the wait's predicate accepts. Choosing it in the instruction
/// rather than in the checker keeps the predicate where it was written -- it
/// reads thread-local scalars the checker has no access to -- while the policy
/// stays in the checker: it is handed a position in its own history and the
/// rule "the earliest legal exit" is what reading that position means.
///
/// `None` with `satisfied_on_entry` says the wait left on a value it already
/// held, which needs no new edge. `None` without it says no value the protocol
/// ever wrote satisfies the predicate, so this exit was not caused by the
/// protocol at all (API §3 step 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeclaredWordWaitPlan {
    span: PhysicalByteSpan,
    warp_id: usize,
    lane: usize,
    accepted: Option<usize>,
    /// The wait's predicate already held on a write this actor had observed,
    /// so it left without any new arrival releasing it.
    satisfied_on_entry: bool,
    /// The predicate accepts the value this word was launched with, so no
    /// publication had to happen for the wait to leave.
    ///
    /// The predicate can only run where the generated code runs, so the answer
    /// is computed there and travels as a decision, the way `accepted` and
    /// `satisfied_on_entry` already do. `None` means the launch value could not
    /// be read -- a host-mapped allocation -- and the checker must then keep
    /// treating an unexplained exit as unexplained.
    satisfied_by_launch_value: Option<bool>,
    semantics: MemoryAccessSemantics,
}

impl DeclaredWordWaitPlan {
    pub const fn new(
        span: PhysicalByteSpan,
        warp_id: usize,
        lane: usize,
        accepted: Option<usize>,
        satisfied_on_entry: bool,
        satisfied_by_launch_value: Option<bool>,
        semantics: MemoryAccessSemantics,
    ) -> Self {
        Self {
            span,
            warp_id,
            lane,
            accepted,
            satisfied_on_entry,
            satisfied_by_launch_value,
            semantics,
        }
    }

    pub const fn span(self) -> PhysicalByteSpan {
        self.span
    }

    pub const fn warp_id(self) -> usize {
        self.warp_id
    }

    pub const fn lane(self) -> usize {
        self.lane
    }

    pub const fn accepted(self) -> Option<usize> {
        self.accepted
    }

    pub const fn satisfied_by_launch_value(self) -> Option<bool> {
        self.satisfied_by_launch_value
    }

    pub const fn satisfied_on_entry(self) -> bool {
        self.satisfied_on_entry
    }

    /// The wait's own memory semantics. `acquire` states that it synchronizes
    /// with the writes that made its predicate hold; `relaxed` states that it
    /// does not, and builds no edge. The scope travels with it, because an
    /// edge is only acquirable by a scope that covers its release.
    pub const fn semantics(self) -> MemoryAccessSemantics {
        self.semantics
    }
}

/// One uniform target/phase group in a lane-varying warp-level mbarrier wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalMbarrierWaitLanePlan {
    plan: PhysicalMbarrierWaitPlan,
    wait_mask: WarpMask,
}

impl PhysicalMbarrierWaitLanePlan {
    pub const fn plan(self) -> PhysicalMbarrierWaitPlan {
        self.plan
    }

    pub const fn wait_mask(self) -> WarpMask {
        self.wait_mask
    }
}

/// Lane groups for one warp-level mbarrier wait.
///
/// The overwhelmingly common warp-uniform case stays inline and allocation
/// free.  Heap storage is used only after two lanes resolve to different
/// barrier/phase requests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhysicalMbarrierWaitLanePlans {
    Empty,
    Uniform(PhysicalMbarrierWaitLanePlan),
    LaneVarying(Box<[PhysicalMbarrierWaitLanePlan]>),
}

impl PhysicalMbarrierWaitLanePlans {
    pub fn as_slice(&self) -> &[PhysicalMbarrierWaitLanePlan] {
        match self {
            Self::Empty => &[],
            Self::Uniform(plan) => std::slice::from_ref(plan),
            Self::LaneVarying(plans) => plans,
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, PhysicalMbarrierWaitLanePlan> {
        self.as_slice().iter()
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    pub fn is_uniform(&self) -> bool {
        matches!(self, Self::Uniform(_))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalMbarrierWaitOutcome {
    completed_generation: Option<u64>,
}

impl PhysicalMbarrierWaitOutcome {
    pub const fn new(completed_generation: Option<u64>) -> Self {
        Self {
            completed_generation,
        }
    }

    pub const fn completed_generation(self) -> Option<u64> {
        self.completed_generation
    }
}

impl PhysicalMbarrierWaitPlan {
    pub const fn barrier_id(self) -> PhysicalBarrierId {
        self.barrier_id
    }

    pub const fn warp_id(self) -> usize {
        self.warp_id
    }

    pub const fn requested_phase(self) -> u64 {
        self.requested_phase
    }

    pub const fn is_conditional(self) -> bool {
        self.conditional
    }

    pub const fn has_acquire(self) -> bool {
        self.acquire
    }

    pub(crate) const fn with_acquire(mut self, acquire: bool) -> Self {
        self.acquire = acquire;
        self
    }

    pub(crate) const fn with_conditional_phase(mut self, conditional: bool) -> Self {
        self.conditional = conditional;
        self
    }

    pub async fn apply(
        self,
        mbarriers: &Arc<PhysicalBarrierHub>,
    ) -> Result<PhysicalMbarrierWaitOutcome, EngineError> {
        self.apply_with_optional_operation(mbarriers, None).await
    }

    pub(crate) async fn apply_with_operation(
        self,
        mbarriers: &Arc<PhysicalBarrierHub>,
        operation: Option<&DynamicOpId>,
    ) -> Result<PhysicalMbarrierWaitOutcome, EngineError> {
        self.apply_with_optional_operation(mbarriers, operation.cloned())
            .await
    }

    async fn apply_with_optional_operation(
        self,
        mbarriers: &Arc<PhysicalBarrierHub>,
        operation: Option<DynamicOpId>,
    ) -> Result<PhysicalMbarrierWaitOutcome, EngineError> {
        if self.conditional {
            return Err(EngineError::message(
                "conditional waits use the nonblocking query path",
            ));
        }
        let completed_generation = mbarriers
            .wait_with_operation(
                self.barrier_id,
                self.requested_phase,
                self.warp_id,
                operation,
            )?
            .await?;
        Ok(PhysicalMbarrierWaitOutcome::new(completed_generation))
    }
}

/// Fully resolved physical mbarrier targets for one asynchronous payload issue.
///
/// Target resolution is intentionally separate from transaction-byte
/// accounting so multicast pointers can be validated before the payload
/// runtime mutates memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalMbarrierCompletionTargets {
    barrier_ids: Box<[PhysicalBarrierId]>,
}

impl PhysicalMbarrierCompletionTargets {
    pub fn single(barrier_id: PhysicalBarrierId) -> Self {
        Self {
            barrier_ids: Box::new([barrier_id]),
        }
    }

    pub fn from_barrier_ids(barrier_ids: impl IntoIterator<Item = PhysicalBarrierId>) -> Self {
        Self {
            barrier_ids: barrier_ids
                .into_iter()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    pub fn barrier_ids(&self) -> &[PhysicalBarrierId] {
        &self.barrier_ids
    }

    pub fn is_empty(&self) -> bool {
        self.barrier_ids.is_empty()
    }

    pub fn issue_plan(&self, transactions_per_target: u64) -> PhysicalMbarrierCompletionIssuePlan {
        PhysicalMbarrierCompletionIssuePlan {
            counter_only: false,
            completions: self
                .barrier_ids
                .iter()
                .copied()
                .map(|barrier_id| (barrier_id, transactions_per_target))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
}

/// Fully resolved barrier identities and byte credits for one async issue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalMbarrierCompletionIssuePlan {
    completions: Box<[(PhysicalBarrierId, u64)]>,
    counter_only: bool,
}

/// One deferred `cp.async.mbarrier.arrive` contribution per active lane.
///
/// Keeping lane entries separate is observable: each arrive-on becomes ready
/// only after that same lane's prior `cp.async` operations are complete.
///
/// `increments_pending` distinguishes the two PTX spellings. The plain form
/// first raises the *current phase's* pending arrival count by one per lane and
/// then discharges it from the deferred arrive-on; `.noinc` skips the raise and
/// relies on a pending count some other instruction already installed. Both
/// spellings share this plan because the deferred half is identical.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpAsyncMbarrierArrivePlan {
    targets: Box<[(u8, PhysicalBarrierId)]>,
    warp_id: usize,
    increments_pending: bool,
}

impl CpAsyncMbarrierArrivePlan {
    pub fn targets(&self) -> impl ExactSizeIterator<Item = (usize, PhysicalBarrierId)> + '_ {
        self.targets
            .iter()
            .map(|&(lane, barrier_id)| (usize::from(lane), barrier_id))
    }

    pub const fn warp_id(&self) -> usize {
        self.warp_id
    }

    /// Whether this spelling raises the current phase's pending arrival count.
    pub const fn increments_pending(&self) -> bool {
        self.increments_pending
    }

    /// The per-barrier pending-count increase this plan installs alongside its
    /// deferred arrive-on.
    ///
    /// Lane entries are aggregated per physical barrier because the hub applies
    /// one transactional increase per target.
    pub fn pending_increases(&self) -> Vec<(PhysicalBarrierId, u64)> {
        if !self.increments_pending {
            return Vec::new();
        }
        let mut increases: BTreeMap<PhysicalBarrierId, u64> = BTreeMap::new();
        for &(_, barrier_id) in &self.targets {
            *increases.entry(barrier_id).or_insert(0) += 1;
        }
        increases.into_iter().collect()
    }

    /// Install the pending-count raise and enqueue one deferred arrive-on per
    /// lane.
    ///
    /// The arrivals are enqueued first. Both halves are one instruction, so a
    /// failure in either must leave the barrier untouched; enqueueing first
    /// means the raise --- which cannot fail once its own transactional
    /// validation passes --- is the last mutation, and an enqueue error cannot
    /// strand a raise that no arrive-on will ever discharge.
    pub fn apply(
        &self,
        mbarriers: &PhysicalBarrierHub,
    ) -> Result<Box<[PhysicalCompletionAction]>, EngineError> {
        let completions = self
            .targets
            .iter()
            .map(|&(_, barrier_id)| (barrier_id, self.warp_id, 1))
            .collect::<Vec<_>>();
        let actions = mbarriers.enqueue_arrival_completions(&completions)?;
        mbarriers.increase_pending_arrivals_many(&self.pending_increases())?;
        Ok(actions)
    }
}

impl PhysicalMbarrierCompletionIssuePlan {
    /// Direct PTX complete_tx credits bytes but does not describe a data copy.
    pub(crate) fn counter_only(mut self) -> Self {
        self.counter_only = true;
        self
    }

    pub(crate) const fn is_counter_only(&self) -> bool {
        self.counter_only
    }

    pub fn single(barrier_id: PhysicalBarrierId, transactions: u64) -> Self {
        PhysicalMbarrierCompletionTargets::single(barrier_id).issue_plan(transactions)
    }

    pub(crate) fn from_completions(
        completions: impl IntoIterator<Item = (PhysicalBarrierId, u64)>,
    ) -> Self {
        Self {
            completions: completions
                .into_iter()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            counter_only: false,
        }
    }

    pub fn completions(&self) -> &[(PhysicalBarrierId, u64)] {
        &self.completions
    }

    pub fn apply(
        &self,
        mbarriers: &PhysicalBarrierHub,
    ) -> Result<Box<[PhysicalCompletionActionId]>, EngineError> {
        mbarriers
            .enqueue_transaction_completions(&self.completions)
            .map_err(Into::into)
    }

    /// Like [`Self::apply`], but `register` runs under the physical barrier
    /// lock before the queued actions become visible; see
    /// [`PhysicalBarrierHub::enqueue_transaction_completions_with`].
    pub fn apply_with<R>(
        &self,
        mbarriers: &PhysicalBarrierHub,
        register: impl FnOnce(
            &[PhysicalCompletionActionId],
            &[PhysicalCompletionAction],
        ) -> Result<R, EngineError>,
    ) -> Result<(Box<[PhysicalCompletionActionId]>, R), EngineError> {
        mbarriers.enqueue_transaction_completions_with(&self.completions, register)
    }

    /// Complete a payload that NumSim has already executed at issue time.
    pub(crate) fn complete_numeric(
        &self,
        mbarriers: &PhysicalBarrierHub,
        delivered_bytes: u64,
    ) -> Result<(), EngineError> {
        let has_completion = self
            .completions
            .iter()
            .any(|&(_, transactions)| transactions != 0);
        if !has_completion && delivered_bytes != 0 {
            return Err(EngineError::message(format!(
                "eager async payload executed {delivered_bytes} destination bytes without an mbarrier completion"
            )));
        }
        mbarriers.complete_transactions_immediately(&self.completions)?;
        Ok(())
    }
}

/// Fully resolved deferred arrive-on batch produced by one `tcgen05.commit`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcgenCommitIssuePlan {
    barrier_ids: Box<[PhysicalBarrierId]>,
    warp_id: usize,
    arrival_count: u64,
}

impl TcgenCommitIssuePlan {
    pub fn barrier_ids(&self) -> &[PhysicalBarrierId] {
        &self.barrier_ids
    }

    pub const fn warp_id(&self) -> usize {
        self.warp_id
    }

    pub const fn arrival_count(&self) -> u64 {
        self.arrival_count
    }

    pub fn apply(
        &self,
        mbarriers: &PhysicalBarrierHub,
    ) -> Result<Box<[PhysicalCompletionAction]>, EngineError> {
        let completions = self
            .barrier_ids
            .iter()
            .copied()
            .map(|barrier_id| (barrier_id, self.warp_id, self.arrival_count))
            .collect::<Vec<_>>();
        mbarriers
            .enqueue_arrival_completions(&completions)
            .map_err(Into::into)
    }

    pub(crate) fn complete_numeric(
        &self,
        mbarriers: &PhysicalBarrierHub,
    ) -> Result<(), EngineError> {
        let completions = self
            .barrier_ids
            .iter()
            .copied()
            .map(|barrier_id| (barrier_id, self.warp_id, self.arrival_count, 0))
            .collect::<Vec<_>>();
        mbarriers.arrive_many(&completions)?;
        Ok(())
    }
}

/// Fully resolved CTA-local nonblocking `bar.arrive` contribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamedBarrierArrivePlan {
    barrier_id: NamedBarrierId,
    expected_arrivals: u64,
    warp_id: usize,
    arrival_mask: WarpMask,
}

impl NamedBarrierArrivePlan {
    pub const fn barrier_id(self) -> NamedBarrierId {
        self.barrier_id
    }

    pub const fn expected_arrivals(self) -> u64 {
        self.expected_arrivals
    }

    pub const fn warp_id(self) -> usize {
        self.warp_id
    }

    pub const fn arrival_count(self) -> u64 {
        self.arrival_mask.len() as u64
    }

    pub const fn arrival_mask(self) -> WarpMask {
        self.arrival_mask
    }

    pub fn apply(
        self,
        barriers: &Arc<NamedBarrierHub>,
    ) -> Result<NamedBarrierArrivalOutcome, EngineError> {
        barriers
            .arrive(
                self.barrier_id,
                self.expected_arrivals,
                self.warp_id,
                self.arrival_mask,
            )
            .map_err(Into::into)
    }
}

/// Fully resolved CTA-local blocking named-barrier contribution and wait contract.
///
/// `aligned` records the PTX mnemonic form: `bar.sync` is the aligned
/// `barrier.sync.aligned`, while the bare `barrier.sync` mnemonic permits
/// divergent entry. Every other distinction the engine or checkers need is
/// re-derived from the contract (barrier id, expected arrivals versus the CTA
/// thread count) and the occurrence identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamedBarrierSyncPlan {
    barrier_id: NamedBarrierId,
    expected_arrivals: u64,
    warp_id: usize,
    arrival_mask: WarpMask,
    aligned: bool,
}

impl NamedBarrierSyncPlan {
    pub const fn barrier_id(self) -> NamedBarrierId {
        self.barrier_id
    }

    pub const fn expected_arrivals(self) -> u64 {
        self.expected_arrivals
    }

    pub const fn warp_id(self) -> usize {
        self.warp_id
    }

    pub const fn arrival_count(self) -> u64 {
        self.arrival_mask.len() as u64
    }

    pub const fn arrival_mask(self) -> WarpMask {
        self.arrival_mask
    }

    pub const fn aligned(self) -> bool {
        self.aligned
    }

    pub fn register(
        self,
        barriers: &Arc<NamedBarrierHub>,
    ) -> Result<NamedBarrierSyncRegistration, EngineError> {
        let wait = barriers.register_sync(
            self.barrier_id,
            self.expected_arrivals,
            self.warp_id,
            self.arrival_mask,
        )?;
        let outcome =
            NamedBarrierSyncRegistrationOutcome::new(wait.generation(), wait.completed_now());
        Ok(NamedBarrierSyncRegistration {
            plan: self,
            outcome,
            wait,
        })
    }

}

/// Exact numeric outcome produced when a named-barrier contribution is registered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamedBarrierSyncRegistrationOutcome {
    generation: u64,
    completed_now: bool,
}

impl NamedBarrierSyncRegistrationOutcome {
    pub const fn new(generation: u64, completed_now: bool) -> Self {
        Self {
            generation,
            completed_now,
        }
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }

    pub const fn completed_now(self) -> bool {
        self.completed_now
    }
}

/// Exact post-unblock identity for one registered `bar.sync` waiter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamedBarrierSyncResumePlan {
    plan: NamedBarrierSyncPlan,
    generation: u64,
}

impl NamedBarrierSyncResumePlan {
    pub const fn plan(self) -> NamedBarrierSyncPlan {
        self.plan
    }

    pub const fn barrier_id(self) -> NamedBarrierId {
        self.plan.barrier_id()
    }

    pub const fn expected_arrivals(self) -> u64 {
        self.plan.expected_arrivals()
    }

    pub const fn warp_id(self) -> usize {
        self.plan.warp_id()
    }

    pub const fn arrival_mask(self) -> WarpMask {
        self.plan.arrival_mask()
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }
}

/// Registered contribution plus the future that resumes after its generation completes.
pub struct NamedBarrierSyncRegistration {
    plan: NamedBarrierSyncPlan,
    outcome: NamedBarrierSyncRegistrationOutcome,
    wait: crate::NamedBarrierWait,
}

impl NamedBarrierSyncRegistration {
    pub const fn plan(&self) -> NamedBarrierSyncPlan {
        self.plan
    }

    pub const fn outcome(&self) -> NamedBarrierSyncRegistrationOutcome {
        self.outcome
    }

    pub fn accumulated_arrival_mask(&self) -> Result<WarpMask, EngineError> {
        self.wait.accumulated_arrival_mask().map_err(Into::into)
    }

    pub async fn resume(self) -> Result<NamedBarrierSyncResumePlan, EngineError> {
        let resume = NamedBarrierSyncResumePlan {
            plan: self.plan,
            generation: self.outcome.generation(),
        };
        self.wait.await?;
        Ok(resume)
    }

    pub async fn resume_recombined(
        self,
        arrival_mask: WarpMask,
    ) -> Result<NamedBarrierSyncResumePlan, EngineError> {
        let accumulated = self.accumulated_arrival_mask()?;
        if accumulated != arrival_mask {
            return Err(EngineError::message(format!(
                "unaligned named-barrier resume mask {:#010x} does not match this warp's accumulated contribution {:#010x}",
                arrival_mask.bits(),
                accumulated.bits(),
            )));
        }
        let mut plan = self.plan;
        plan.arrival_mask = arrival_mask;
        let resume = NamedBarrierSyncResumePlan {
            plan,
            generation: self.outcome.generation(),
        };
        self.wait.await?;
        Ok(resume)
    }
}

pub fn plan_named_barrier_sync(
    context: &WarpContext,
    barrier_id: i64,
    expected_arrivals: i64,
    mask: WarpMask,
) -> Result<Option<NamedBarrierSyncPlan>, EngineError> {
    plan_named_barrier_sync_with_alignment(context, barrier_id, expected_arrivals, mask, true)
}

pub fn plan_named_barrier_sync_with_alignment(
    context: &WarpContext,
    barrier_id: i64,
    expected_arrivals: i64,
    mask: WarpMask,
    aligned: bool,
) -> Result<Option<NamedBarrierSyncPlan>, EngineError> {
    if mask.is_empty() {
        return Ok(None);
    }
    let (barrier_id, expected_arrivals) =
        validate_named_barrier_operands(barrier_id, expected_arrivals)?;
    Ok(Some(NamedBarrierSyncPlan {
        barrier_id: NamedBarrierId::new(context.global_cta_id(), barrier_id),
        expected_arrivals,
        warp_id: context.global_warp_id(),
        arrival_mask: mask,
        aligned,
    }))
}

pub(crate) fn plan_internal_warpgroup_sync(
    context: &WarpContext,
    static_op_id: u64,
    warps_per_group: usize,
    mask: WarpMask,
) -> Result<Option<NamedBarrierSyncPlan>, EngineError> {
    if mask.is_empty() {
        return Ok(None);
    }
    if warps_per_group == 0 {
        return Err(EngineError::message(
            "internal warpgroup rendezvous requires at least one warp",
        ));
    }
    let warpgroup_id = context.warp_id_in_cta() / warps_per_group;
    let expected_arrivals = u64::try_from(warps_per_group)
        .ok()
        .and_then(|warps| warps.checked_mul(crate::WARP_SIZE as u64))
        .ok_or_else(|| EngineError::message("internal warpgroup rendezvous size overflow"))?;
    Ok(Some(NamedBarrierSyncPlan {
        barrier_id: NamedBarrierId::internal_warpgroup(
            context.global_cta_id(),
            8,
            static_op_id,
            warpgroup_id,
        ),
        expected_arrivals,
        warp_id: context.global_warp_id(),
        arrival_mask: mask,
        aligned: true,
    }))
}

pub fn plan_named_barrier_arrive(
    context: &WarpContext,
    barrier_id: i64,
    expected_arrivals: i64,
    mask: WarpMask,
) -> Result<Option<NamedBarrierArrivePlan>, EngineError> {
    if mask.is_empty() {
        return Ok(None);
    }
    let (barrier_id, expected_arrivals) =
        validate_named_barrier_operands(barrier_id, expected_arrivals)?;
    Ok(Some(NamedBarrierArrivePlan {
        barrier_id: NamedBarrierId::new(context.global_cta_id(), barrier_id),
        expected_arrivals,
        warp_id: context.global_warp_id(),
        arrival_mask: mask,
    }))
}

fn validate_named_barrier_operands(
    barrier_id: i64,
    expected_arrivals: i64,
) -> Result<(u32, u64), EngineError> {
    let barrier_id = u32::try_from(barrier_id).map_err(|_| {
        EngineError::message(format!(
            "named barrier id must be in 0..15, got {barrier_id}"
        ))
    })?;
    if barrier_id >= 16 {
        return Err(EngineError::message(format!(
            "named barrier id must be in 0..15, got {barrier_id}"
        )));
    }
    let expected_arrivals = u64::try_from(expected_arrivals).map_err(|_| {
        EngineError::message(format!(
            "named barrier arrival count must be a positive multiple of {WARP_SIZE}, got {expected_arrivals}"
        ))
    })?;
    if expected_arrivals == 0 || expected_arrivals % WARP_SIZE as u64 != 0 {
        return Err(EngineError::message(format!(
            "named barrier arrival count must be a positive multiple of {WARP_SIZE}, got {expected_arrivals}"
        )));
    }
    Ok((barrier_id, expected_arrivals))
}

pub fn plan_physical_mbarrier_init(
    context: &WarpContext,
    pointer: &PhysicalPtr,
    mask: WarpMask,
    expected: i64,
) -> Result<PhysicalMbarrierInitPlan, EngineError> {
    let mut barrier_ids = Vec::new();
    for lane in mask {
        let lane_mask = one_lane(lane);
        let barrier_id = pointer.resolve_shared_barrier_write(context, lane_mask, None)?;
        barrier_ids.push(barrier_id);
    }
    // Keep invalid runtime counts in the resolved plan so analysis modes can
    // report the typed protocol violation before numeric mutation. Numeric
    // execution validates the same range in `PhysicalMbarrierInitPlan::apply`.
    let expected_arrivals = if barrier_ids.is_empty() {
        0
    } else {
        u64::try_from(expected).unwrap_or(0)
    };
    Ok(PhysicalMbarrierInitPlan {
        barrier_ids: barrier_ids.into_boxed_slice(),
        expected_arrivals,
        layout_v1: false,
    })
}

pub fn plan_physical_mbarrier_arrive(
    context: &WarpContext,
    pointer: &PhysicalPtr,
    mask: WarpMask,
    target_cta: Option<i64>,
    transactions: Option<i64>,
) -> Result<Option<PhysicalMbarrierArrivePlan>, EngineError> {
    if mask.is_empty() {
        return Ok(None);
    }
    let is_remote_form = target_cta.is_some();
    let target_cta = target_cta
        .map(|value| {
            usize::try_from(value).map_err(|_| EngineError::message("negative remote CTA id"))
        })
        .transpose()?;
    let barrier_id = pointer.resolve_shared_barrier(context, mask, target_cta)?;
    validate_mbarrier_arrive_address_space(context, barrier_id, is_remote_form)?;
    let expected_transactions = transactions.map(transaction_count).transpose()?;
    Ok(Some(PhysicalMbarrierArrivePlan {
        barrier_id,
        warp_id: context.global_warp_id(),
        arrival_count: mask.len() as u64,
        expected_transactions,
        drop: false,
        release: true,
    }))
}

pub fn plan_physical_mbarrier_arrive_lanes(
    context: &WarpContext,
    pointer: &PhysicalPtr,
    mask: WarpMask,
    target_ctas: Option<&WarpValue<i64>>,
    arrival_counts: Option<&WarpValue<i64>>,
    transactions: Option<&WarpValue<i64>>,
) -> Result<PhysicalMbarrierArriveBatchPlan, EngineError> {
    let is_remote_form = target_ctas.is_some();
    let mut arrivals: BTreeMap<PhysicalBarrierId, (WarpMask, u64, u64)> = BTreeMap::new();
    for lane in mask {
        let target_cta = target_ctas
            .map(|values| {
                usize::try_from(values[lane])
                    .map_err(|_| EngineError::message("negative remote CTA id"))
            })
            .transpose()?;
        let barrier_id = pointer.resolve_shared_barrier(context, one_lane(lane), target_cta)?;
        validate_mbarrier_arrive_address_space(context, barrier_id, is_remote_form)?;
        let lane_transactions = transactions
            .map(|values| transaction_count(values[lane]))
            .transpose()?
            .unwrap_or(0);
        let lane_arrivals = arrival_counts
            .map(|values| expected_arrival_count(values[lane]))
            .transpose()?
            .unwrap_or(1);
        let (arrival_mask, arrival_count, expected_transactions) = arrivals
            .entry(barrier_id)
            .or_insert((WarpMask::EMPTY, 0, 0));
        *arrival_mask = arrival_mask.union(one_lane(lane));
        *arrival_count = arrival_count
            .checked_add(lane_arrivals)
            .ok_or_else(|| EngineError::message("mbarrier arrival count overflow"))?;
        *expected_transactions = expected_transactions
            .checked_add(lane_transactions)
            .ok_or_else(|| EngineError::message("mbarrier transaction count overflow"))?;
        if *expected_transactions > MAX_MBARRIER_COUNT {
            return Err(EngineError::message(format!(
                "mbarrier transaction count must be in 0..={MAX_MBARRIER_COUNT}, got {expected_transactions}"
            )));
        }
    }
    Ok(PhysicalMbarrierArriveBatchPlan {
        entries: arrivals
            .into_iter()
            .map(
                |(barrier_id, (arrival_mask, arrival_count, expected_transactions))| {
                    PhysicalMbarrierArriveBatchEntry {
                        plan: PhysicalMbarrierArrivePlan {
                            barrier_id,
                            warp_id: context.global_warp_id(),
                            arrival_count,
                            expected_transactions: transactions
                                .is_some()
                                .then_some(expected_transactions),
                            drop: false,
                            release: true,
                        },
                        arrival_mask,
                    }
                },
            )
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    })
}

pub fn plan_cp_async_mbarrier_arrive(
    context: &WarpContext,
    pointer: &PhysicalPtr,
    mask: WarpMask,
    increments_pending: bool,
) -> Result<CpAsyncMbarrierArrivePlan, EngineError> {
    let mut targets = Vec::with_capacity(mask.len());
    for lane in mask {
        let barrier_id = pointer.resolve_shared_barrier(context, one_lane(lane), None)?;
        validate_mbarrier_arrive_address_space(context, barrier_id, false)?;
        targets.push((
            u8::try_from(lane).expect("warp lane fits in u8"),
            barrier_id,
        ));
    }
    Ok(CpAsyncMbarrierArrivePlan {
        targets: targets.into_boxed_slice(),
        warp_id: context.global_warp_id(),
        increments_pending,
    })
}

fn validate_mbarrier_arrive_address_space(
    context: &WarpContext,
    barrier_id: PhysicalBarrierId,
    is_remote_form: bool,
) -> Result<(), EngineError> {
    let issuer_global_cta_id = context.global_cta_id();
    let target_global_cta_id = barrier_id.target_global_cta_id();
    if !is_remote_form && target_global_cta_id != issuer_global_cta_id {
        return Err(SynchronizationError::MbarrierLocalArriveRemoteAddress {
            issuer_global_cta_id,
            target_global_cta_id,
        }
        .into());
    }
    Ok(())
}

pub fn plan_physical_mbarrier_wait(
    context: &WarpContext,
    pointer: &PhysicalPtr,
    mask: WarpMask,
    phase: i64,
) -> Result<Option<PhysicalMbarrierWaitPlan>, EngineError> {
    if mask.is_empty() {
        return Ok(None);
    }
    let barrier_id = pointer.resolve_shared_barrier_read(context, mask, None)?;
    let requested_phase =
        u64::try_from(phase).map_err(|_| EngineError::message("negative mbarrier phase"))?;
    Ok(Some(PhysicalMbarrierWaitPlan {
        barrier_id,
        warp_id: context.global_warp_id(),
        requested_phase,
        conditional: false,
        acquire: true,
    }))
}

pub fn plan_physical_mbarrier_wait_lanes(
    context: &WarpContext,
    pointer: &PhysicalPtr,
    mask: WarpMask,
    phases: &WarpValue<i64>,
) -> Result<PhysicalMbarrierWaitLanePlans, EngineError> {
    let mut uniform_request = None;
    let mut lane_varying_requests = None::<BTreeMap<(PhysicalBarrierId, u64), WarpMask>>;
    for lane in mask {
        let barrier_id = pointer.resolve_shared_barrier_read(context, one_lane(lane), None)?;
        let requested_phase = u64::try_from(phases[lane])
            .map_err(|_| EngineError::message("negative mbarrier phase"))?;
        let request = (barrier_id, requested_phase);
        if let Some(requests) = lane_varying_requests.as_mut() {
            let wait_mask = requests.entry(request).or_insert(WarpMask::EMPTY);
            *wait_mask = wait_mask.union(one_lane(lane));
            continue;
        }
        match uniform_request.as_mut() {
            None => uniform_request = Some((request, one_lane(lane))),
            Some((uniform, wait_mask)) if *uniform == request => {
                *wait_mask = wait_mask.union(one_lane(lane));
            }
            Some((uniform, wait_mask)) => {
                let mut requests = BTreeMap::new();
                requests.insert(*uniform, *wait_mask);
                requests.insert(request, one_lane(lane));
                lane_varying_requests = Some(requests);
            }
        }
    }

    let make_plan = |((barrier_id, requested_phase), wait_mask)| PhysicalMbarrierWaitLanePlan {
        plan: PhysicalMbarrierWaitPlan {
            barrier_id,
            warp_id: context.global_warp_id(),
            requested_phase,
            conditional: false,
            acquire: true,
        },
        wait_mask,
    };
    if let Some(requests) = lane_varying_requests {
        return Ok(PhysicalMbarrierWaitLanePlans::LaneVarying(
            requests
                .into_iter()
                .map(make_plan)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ));
    }
    Ok(match uniform_request {
        Some(request) => PhysicalMbarrierWaitLanePlans::Uniform(make_plan(request)),
        None => PhysicalMbarrierWaitLanePlans::Empty,
    })
}

/// Resolve one warp-uniform state-token query.
///
/// The returned wait plan carries the token's parity only for the existing
/// checker effect vocabulary; the exact generation remains separate and is
/// validated by `PhysicalBarrierHub::query_many_with` before the effect
/// is published.
pub fn plan_physical_mbarrier_state_wait_lanes(
    context: &WarpContext,
    pointer: &PhysicalPtr,
    mask: WarpMask,
    states: &WarpValue<u64>,
) -> Result<Option<(PhysicalMbarrierWaitPlan, u64)>, EngineError> {
    let mut request = None;
    for lane in mask {
        let barrier_id = pointer.resolve_shared_barrier_read(context, one_lane(lane), None)?;
        let state = crate::hardware_barriers::mbarrier_state_generation(states[lane]);
        let lane_request = (barrier_id, state);
        if request.is_some_and(|existing| existing != lane_request) {
            return Err(EngineError::message(
                "one warp mbarrier state query cannot use different barrier/state requests",
            ));
        }
        request = Some(lane_request);
    }
    Ok(request.map(|(barrier_id, state)| {
        (
            PhysicalMbarrierWaitPlan {
                barrier_id,
                warp_id: context.global_warp_id(),
                requested_phase: state & 1,
                conditional: false,
                acquire: true,
            },
            state,
        )
    }))
}

pub fn plan_tcgen_commit_issue(
    context: &WarpContext,
    pointer: &PhysicalPtr,
    issue_mask: WarpMask,
    multicast_cta_masks: Option<&WarpValue<i64>>,
) -> Result<Option<TcgenCommitIssuePlan>, EngineError> {
    if issue_mask.is_empty() {
        return Ok(None);
    }
    if issue_mask.len() != 1 {
        return Err(EngineError::message(format!(
            "tcgen05.commit requires one issuing lane, got {}",
            issue_mask.len()
        )));
    }
    let issuing_lane = issue_mask
        .first_active()
        .expect("a single-lane issue mask has one active lane");
    let barrier_ids = if let Some(cta_masks) = multicast_cta_masks {
        let cta_mask = u64::try_from(cta_masks[issuing_lane])
            .map_err(|_| EngineError::message("tcgen05.commit cta_mask must be nonnegative"))?;
        if cta_mask > u32::MAX as u64 {
            return Err(EngineError::message(format!(
                "tcgen05.commit cta_mask {cta_mask:#x} exceeds the 32-bit PTX operand"
            )));
        }
        let ctas_per_cluster = context.topology().ctas_per_cluster();
        let invalid_bits = if ctas_per_cluster >= 64 {
            0
        } else {
            cta_mask >> ctas_per_cluster
        };
        if invalid_bits != 0 {
            return Err(EngineError::message(format!(
                "tcgen05.commit cta_mask {cta_mask:#x} names a CTA outside cluster size {ctas_per_cluster}"
            )));
        }
        (0..ctas_per_cluster)
            .filter(|&target_cta| cta_mask & (1_u64 << target_cta) != 0)
            .map(|target_cta| {
                pointer.resolve_shared_barrier_multicast(context, issue_mask, target_cta)
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![pointer.resolve_shared_barrier(context, issue_mask, None)?]
    };
    Ok(Some(TcgenCommitIssuePlan {
        barrier_ids: barrier_ids.into_boxed_slice(),
        warp_id: context.global_warp_id(),
        arrival_count: 1,
    }))
}

pub fn initialize_physical_mbarriers(
    context: &WarpContext,
    pointer: &PhysicalPtr,
    mask: WarpMask,
    mbarriers: &PhysicalBarrierHub,
    expected: &WarpValue<i64>,
    layout_v1: bool,
) -> Result<(), EngineError> {
    let mut mbarrier_init_ids: BTreeMap<PhysicalBarrierId, (u64, usize)> = BTreeMap::new();
    for lane in mask {
        let lane_mask = one_lane(lane);
        let barrier_id = pointer.resolve_shared_barrier_write(context, lane_mask, None)?;
        let lane_expected = expected_arrival_count(expected[lane])?;
        match mbarrier_init_ids.get_mut(&barrier_id) {
            Some((prior_expected, lane_count)) => {
                if *prior_expected != lane_expected {
                    return Err(EngineError::message(format!(
                        "mbarrier.init thread_count differs for lanes targeting the same barrier: {prior_expected} vs {lane_expected}"
                    )));
                }
                *lane_count += 1;
            }
            None => {
                mbarrier_init_ids.insert(barrier_id, (lane_expected, 1));
            }
        }
    }
    if mbarrier_init_ids.len() != 1 && mbarrier_init_ids.len() != mask.len() {
        return Err(EngineError::message(
            "mbarrier.init pointer must be warp-uniform or one-to-one across active lanes",
        ));
    }
    for (barrier_id, (lane_expected, _)) in mbarrier_init_ids {
        mbarriers.init_numeric_layout(barrier_id, lane_expected, layout_v1)?;
    }
    Ok(())
}

/// Resolve and aggregate one lane-wise standalone `mbarrier.expect_tx`.
pub fn plan_physical_mbarrier_expect_tx(
    context: &WarpContext,
    pointer: &PhysicalPtr,
    mask: WarpMask,
    transactions: &WarpValue<i64>,
) -> Result<PhysicalMbarrierExpectTxPlan, EngineError> {
    let mut by_barrier = BTreeMap::<PhysicalBarrierId, u64>::new();
    for lane in mask {
        let barrier_id = pointer.resolve_shared_barrier(context, one_lane(lane), None)?;
        let lane_transactions = transaction_count(transactions[lane])?;
        let total = by_barrier.entry(barrier_id).or_default();
        *total = total
            .checked_add(lane_transactions)
            .ok_or_else(|| EngineError::message("mbarrier transaction count overflow"))?;
        if *total > MAX_MBARRIER_COUNT {
            return Err(EngineError::message(format!(
                "mbarrier transaction count must be in 0..={MAX_MBARRIER_COUNT}, got {total}"
            )));
        }
    }
    Ok(PhysicalMbarrierExpectTxPlan {
        entries: by_barrier
            .into_iter()
            .map(
                |(barrier_id, expected_transactions)| PhysicalMbarrierExpectTxEntry {
                    barrier_id,
                    expected_transactions,
                },
            )
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    })
}

pub async fn wait_physical_mbarrier(
    context: &WarpContext,
    pointer: &PhysicalPtr,
    mask: WarpMask,
    phases: &WarpValue<i64>,
    mbarriers: &Arc<PhysicalBarrierHub>,
) -> Result<(), EngineError> {
    wait_physical_mbarrier_impl(context, pointer, mask, phases, mbarriers).await
}

async fn wait_physical_mbarrier_impl(
    context: &WarpContext,
    pointer: &PhysicalPtr,
    mask: WarpMask,
    phases: &WarpValue<i64>,
    mbarriers: &Arc<PhysicalBarrierHub>,
) -> Result<(), EngineError> {
    if mask.is_empty() {
        return Ok(());
    }
    let mut requests = BTreeSet::new();
    for lane in mask {
        let barrier = pointer.resolve_shared_barrier_read(context, one_lane(lane), None)?;
        let phase = u64::try_from(phases[lane])
            .map_err(|_| EngineError::message("negative mbarrier phase"))?;
        requests.insert((barrier, phase));
    }

    let requests = requests.into_iter().collect::<Vec<_>>();
    let initially_ready = mbarriers.test_wait_many(&requests)?;
    let pending = requests
        .iter()
        .copied()
        .zip(initially_ready.iter().copied())
        .filter_map(|(request, ready)| (!ready).then_some(request))
        .collect::<Vec<_>>();
    if pending.is_empty() {
        return Ok(());
    }
    if pending.len() != requests.len() || pending.len() != 1 {
        return Err(EngineError::message(
            "one warp mbarrier.try_wait cannot suspend lanes with different readiness or barrier/phase requests",
        ));
    }
    let (barrier, phase) = pending[0];
    mbarriers
        .wait(barrier, phase, context.global_warp_id())?
        .await
        .map_err(EngineError::from)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        CtaId, EngineError, GlobalMemory, LaunchTopology, PhysicalBarrierHub, PhysicalBarrierId,
        PhysicalMemory, SynchronizationError, WarpContext, WarpMask, WarpValue,
    };

    use super::{
        plan_named_barrier_arrive, plan_named_barrier_sync, plan_named_barrier_sync_with_alignment,
        plan_physical_mbarrier_arrive, plan_physical_mbarrier_arrive_lanes,
        plan_physical_mbarrier_init, plan_physical_mbarrier_wait,
        plan_physical_mbarrier_wait_lanes, plan_tcgen_commit_issue,
        PhysicalMbarrierCompletionTargets, PhysicalPtr,
    };
    use crate::runtime::RuntimeBuffer;

    fn shared_pointer(indices: WarpValue<i64>) -> (PhysicalMemory, WarpContext, PhysicalPtr) {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let physical = PhysicalMemory::new(topology);
        let owner = CtaId::new(topology, 0, 0).unwrap();
        let allocation = physical.shared().allocate_cta_zeroed(owner, 256).unwrap();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Shared {
                allocations: Arc::new(vec![allocation]),
                byte_offset: 0,
                byte_len: 256,
                backing_byte_len: 256,
                virtual_base: 0,
            },
            indices,
            8,
        );
        (physical, context, pointer)
    }

    fn mapped_shared_pointer(
        issuer_global_cta_id: usize,
        ranks: WarpValue<i64>,
    ) -> (PhysicalMemory, WarpContext, PhysicalPtr) {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let context = topology
            .warp_contexts()
            .find(|context| context.global_cta_id() == issuer_global_cta_id)
            .unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocations = (0..2)
            .map(|cta_id| {
                physical
                    .shared()
                    .allocate_cta_zeroed(CtaId::new(topology, 0, cta_id).unwrap(), 256)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let local_pointer = PhysicalPtr::new(
            RuntimeBuffer::Shared {
                allocations: Arc::new(allocations),
                byte_offset: 0,
                byte_len: 256,
                backing_byte_len: 256,
                virtual_base: 0,
            },
            WarpValue::splat(0_i64),
            8,
        );
        let pointer = local_pointer
            .map_shared_rank(&context, &ranks, WarpMask::FULL)
            .unwrap();
        (physical, context, pointer)
    }

    #[test]
    fn mbarrier_plans_resolve_complete_lane_targets_before_apply() {
        let indices = WarpValue::from_fn(|lane| lane as i64);
        let (_physical, context, pointer) = shared_pointer(indices);
        let mask = WarpMask::from_lanes([0, 1]).unwrap();

        let plan = plan_physical_mbarrier_init(&context, &pointer, mask, 2).unwrap();

        assert_eq!(plan.barrier_ids().len(), 2);
        assert_eq!(plan.barrier_ids()[0].byte_offset(), 0);
        assert_eq!(plan.barrier_ids()[1].byte_offset(), 8);
        assert_eq!(plan.expected_arrivals(), 2);
        let hub = PhysicalBarrierHub::new();
        plan.apply(&hub).unwrap();
        hub.arrive(plan.barrier_ids()[0], 0, 2).unwrap();
        hub.arrive(plan.barrier_ids()[1], 0, 2).unwrap();
    }

    #[test]
    fn mbarrier_wait_lanes_group_equal_requests_and_preserve_masks() {
        let indices = WarpValue::from_fn(|lane| match lane {
            0 | 2 | 3 => 0_i64,
            1 => 1_i64,
            _ => 0_i64,
        });
        let (_physical, context, pointer) = shared_pointer(indices);
        let mask = WarpMask::from_lanes([0, 1, 2, 3]).unwrap();
        let phases = WarpValue::from_fn(|lane| if lane == 3 { 1_i64 } else { 0_i64 });

        let plans = plan_physical_mbarrier_wait_lanes(&context, &pointer, mask, &phases).unwrap();
        let plans = plans.as_slice();

        assert_eq!(plans.len(), 3);
        assert_eq!(plans[0].plan().barrier_id().byte_offset(), 0);
        assert_eq!(plans[0].plan().requested_phase(), 0);
        assert_eq!(plans[0].wait_mask(), WarpMask::from_lanes([0, 2]).unwrap());
        assert_eq!(plans[1].plan().barrier_id().byte_offset(), 0);
        assert_eq!(plans[1].plan().requested_phase(), 1);
        assert_eq!(plans[1].wait_mask(), WarpMask::from_lanes([3]).unwrap());
        assert_eq!(plans[2].plan().barrier_id().byte_offset(), 8);
        assert_eq!(plans[2].plan().requested_phase(), 0);
        assert_eq!(plans[2].wait_mask(), WarpMask::from_lanes([1]).unwrap());
        assert!(
            plan_physical_mbarrier_wait_lanes(&context, &pointer, WarpMask::EMPTY, &phases,)
                .unwrap()
                .is_empty()
        );

        let (_physical, context, pointer) = shared_pointer(WarpValue::splat(0_i64));
        let uniform = plan_physical_mbarrier_wait_lanes(
            &context,
            &pointer,
            WarpMask::from_lanes([0, 1, 2, 3]).unwrap(),
            &WarpValue::splat(0_i64),
        )
        .unwrap();
        assert!(uniform.is_uniform());
        assert_eq!(uniform.len(), 1);
    }

    #[test]
    fn completion_issue_plan_batch_failure_does_not_enqueue_a_prefix_or_reserve_ids() {
        let first = PhysicalBarrierId::new(1, 0, 0);
        let second = PhysicalBarrierId::new(2, 0, 0);
        let hub = PhysicalBarrierHub::new();
        hub.init(first, 1).unwrap();
        let targets = PhysicalMbarrierCompletionTargets::from_barrier_ids([first, second]);
        let plan = targets.issue_plan(64);

        let error = plan.apply(&hub).unwrap_err();

        assert!(matches!(
            error.kind(),
            crate::EngineErrorKind::Synchronization(source)
                if matches!(source.as_ref(), SynchronizationError::BarrierUninitialized { .. })
        ));
        assert_eq!(hub.pending_completion_count(), 0);
        hub.init(second, 1).unwrap();
        let action_ids = plan.apply(&hub).unwrap();
        assert_eq!(
            action_ids.iter().map(|id| id.get()).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(hub.pending_completion_count(), 2);
    }

    #[test]
    fn named_barrier_plan_captures_exact_cta_warp_mask_and_contract() {
        let topology = LaunchTopology::new(1, 2, 2).unwrap();
        let context = topology.warp_contexts().nth(3).unwrap();
        let mask = WarpMask::from_lanes([0, 3, 7]).unwrap();
        let plan = plan_named_barrier_sync(&context, 9, 96, mask)
            .unwrap()
            .unwrap();

        assert_eq!(plan.barrier_id().global_cta_id(), 1);
        assert_eq!(plan.barrier_id().barrier_id(), 9);
        assert_eq!(plan.expected_arrivals(), 96);
        assert_eq!(plan.warp_id(), 3);
        assert_eq!(plan.arrival_mask(), mask);
        assert!(plan.aligned());

        let unaligned_plan =
            plan_named_barrier_sync_with_alignment(&context, 0, 64, WarpMask::FULL, false)
                .unwrap()
                .unwrap();
        assert_eq!(unaligned_plan.barrier_id().barrier_id(), 0);
        assert!(!unaligned_plan.aligned());
        assert_eq!(plan.arrival_count(), 3);
        assert_eq!(
            plan_named_barrier_sync(&context, 9, 96, WarpMask::EMPTY).unwrap(),
            None
        );
        assert!(plan_named_barrier_sync(&context, -1, 96, mask).is_err());
        assert!(plan_named_barrier_sync(&context, 9, 0, mask).is_err());
        assert!(plan_named_barrier_sync(&context, 16, 96, mask).is_err());
        assert!(plan_named_barrier_sync(&context, 9, 48, mask).is_err());

        let arrive = plan_named_barrier_arrive(&context, 9, 96, mask)
            .unwrap()
            .unwrap();
        assert_eq!(arrive.barrier_id(), plan.barrier_id());
        assert_eq!(arrive.expected_arrivals(), plan.expected_arrivals());
        assert_eq!(arrive.warp_id(), plan.warp_id());
        assert_eq!(arrive.arrival_mask(), plan.arrival_mask());
        assert_eq!(
            plan_named_barrier_arrive(&context, 9, 96, WarpMask::EMPTY).unwrap(),
            None
        );
        assert!(plan_named_barrier_arrive(&context, 16, 96, mask).is_err());
        assert!(plan_named_barrier_arrive(&context, 9, 48, mask).is_err());
    }

    #[test]
    fn mbarrier_init_plan_accepts_duplicate_lane_targets() {
        let indices = WarpValue::from_fn(|lane| if lane < 2 { 0 } else { 1 });
        let (_physical, context, pointer) = shared_pointer(indices);
        let mask = WarpMask::from_lanes([0, 1, 2]).unwrap();

        let plan = plan_physical_mbarrier_init(&context, &pointer, mask, 3).unwrap();

        assert_eq!(plan.barrier_ids().len(), 3);
        assert_eq!(plan.barrier_ids()[0], plan.barrier_ids()[1]);
        assert_ne!(plan.barrier_ids()[1], plan.barrier_ids()[2]);
        let hub = PhysicalBarrierHub::new();
        plan.apply(&hub).unwrap();
        hub.arrive(plan.barrier_ids()[0], 0, 1).unwrap();
        hub.arrive(plan.barrier_ids()[2], 0, 1).unwrap();
    }

    #[test]
    fn mbarrier_arrive_and_wait_plans_capture_concrete_protocol_arguments() {
        let (_physical, context, pointer) = shared_pointer(WarpValue::splat(2_i64));
        let mask = WarpMask::from_lanes([3, 7]).unwrap();

        let arrive = plan_physical_mbarrier_arrive(&context, &pointer, mask, None, Some(64))
            .unwrap()
            .unwrap();
        assert_eq!(arrive.barrier_id().byte_offset(), 16);
        assert_eq!(arrive.warp_id(), context.global_warp_id());
        assert_eq!(arrive.arrival_count(), 2);
        assert_eq!(arrive.expected_transactions(), Some(64));
        assert!(arrive.is_release());

        for release in [false, true] {
            for drop in [false, true] {
                let batch = plan_physical_mbarrier_arrive_lanes(
                    &context,
                    &pointer,
                    mask,
                    None,
                    None,
                    Some(&WarpValue::splat(32)),
                )
                .unwrap()
                .with_semantics(drop, release);
                let target = batch.entries()[0].plan();
                assert_eq!(target.is_release(), release);
                assert_eq!(target.is_drop(), drop);
                assert_eq!(target.barrier_id(), arrive.barrier_id());
                assert_eq!(target.arrival_count(), arrive.arrival_count());
                assert_eq!(
                    target.expected_transactions(),
                    arrive.expected_transactions()
                );
            }
        }

        let wait = plan_physical_mbarrier_wait(&context, &pointer, mask, 1)
            .unwrap()
            .unwrap();
        assert_eq!(wait.barrier_id(), arrive.barrier_id());
        assert_eq!(wait.warp_id(), context.global_warp_id());
        assert_eq!(wait.requested_phase(), 1);
    }

    #[test]
    fn scalar_mbarrier_arrive_expect_tx_rejects_mapped_remote_address() {
        let (_physical, context, pointer) = mapped_shared_pointer(1, WarpValue::splat(0_i64));
        let mask = WarpMask::from_lanes([0]).unwrap();

        let error =
            plan_physical_mbarrier_arrive(&context, &pointer, mask, None, Some(16)).unwrap_err();

        assert!(matches!(
            error.kind(),
            crate::EngineErrorKind::Synchronization(source)
                if matches!(
                    source.as_ref(),
                    SynchronizationError::MbarrierLocalArriveRemoteAddress {
                        issuer_global_cta_id: 1,
                        target_global_cta_id: 0,
                    }
                )
        ));
    }

    #[test]
    fn lane_mixed_local_mbarrier_arrive_rejects_mapped_remote_address() {
        let ranks = WarpValue::from_fn(|lane| if lane == 0 { 1 } else { 0 });
        let (_physical, context, pointer) = mapped_shared_pointer(1, ranks);
        let mask = WarpMask::from_lanes([0, 1]).unwrap();

        let error = plan_physical_mbarrier_arrive_lanes(&context, &pointer, mask, None, None, None)
            .unwrap_err();

        assert!(matches!(
            error.kind(),
            crate::EngineErrorKind::Synchronization(source)
                if matches!(
                    source.as_ref(),
                    SynchronizationError::MbarrierLocalArriveRemoteAddress {
                        issuer_global_cta_id: 1,
                        target_global_cta_id: 0,
                    }
                )
        ));
    }

    #[test]
    fn empty_mbarrier_plans_preserve_no_effect_semantics() {
        let (_physical, context, pointer) = shared_pointer(WarpValue::splat(999_i64));

        let init = plan_physical_mbarrier_init(&context, &pointer, WarpMask::EMPTY, -1).unwrap();
        assert!(init.is_empty());
        assert_eq!(init.expected_arrivals(), 0);
        assert!(plan_physical_mbarrier_arrive(
            &context,
            &pointer,
            WarpMask::EMPTY,
            Some(-1),
            Some(-1),
        )
        .unwrap()
        .is_none());
        assert!(
            plan_physical_mbarrier_wait(&context, &pointer, WarpMask::EMPTY, -1)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn tcgen_commit_validates_issue_and_cta_masks_before_barrier_resolution() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let global = GlobalMemory::new();
        let allocation = global.allocate_zeroed(8).unwrap();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Global(global.full_view(allocation).unwrap()),
            WarpValue::splat(0_i64),
            8,
        );
        let one_lane = WarpMask::from_lanes([0]).unwrap();
        let two_lanes = WarpMask::from_lanes([0, 1]).unwrap();

        let zero = WarpValue::splat(0_i64);
        let negative = WarpValue::splat(-1_i64);
        let too_wide = WarpValue::splat(0x1_0000_0000_i64);
        let outside_cluster = WarpValue::splat(0b100_i64);

        assert!(
            plan_tcgen_commit_issue(&context, &pointer, WarpMask::EMPTY, None)
                .unwrap()
                .is_none()
        );
        assert!(plan_tcgen_commit_issue(&context, &pointer, two_lanes, None)
            .unwrap_err()
            .to_string()
            .contains("requires one issuing lane"));
        assert!(
            plan_tcgen_commit_issue(&context, &pointer, one_lane, Some(&negative))
                .unwrap_err()
                .to_string()
                .contains("must be nonnegative")
        );
        assert!(
            plan_tcgen_commit_issue(&context, &pointer, one_lane, Some(&too_wide))
                .unwrap_err()
                .to_string()
                .contains("exceeds the 32-bit PTX operand")
        );
        assert!(
            plan_tcgen_commit_issue(&context, &pointer, one_lane, Some(&outside_cluster))
                .unwrap_err()
                .to_string()
                .contains("outside cluster size 2")
        );
        let empty = plan_tcgen_commit_issue(&context, &pointer, one_lane, Some(&zero))
            .unwrap()
            .unwrap();
        assert!(empty.barrier_ids().is_empty());
        let mbarriers = PhysicalBarrierHub::new();
        assert!(empty.apply(&mbarriers).unwrap().is_empty());
        assert_eq!(mbarriers.pending_completion_count(), 0);
    }

    #[test]
    fn tcgen_multicast_plan_resolves_and_enqueues_every_target_transactionally() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocations = (0..2)
            .map(|global_cta_id| {
                physical
                    .shared()
                    .allocate_cta_zeroed(CtaId::new(topology, 0, global_cta_id).unwrap(), 8)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Shared {
                allocations: Arc::new(allocations),
                byte_offset: 0,
                byte_len: 8,
                backing_byte_len: 8,
                virtual_base: 0,
            },
            WarpValue::splat(0_i64),
            8,
        );
        let one_lane = WarpMask::from_lanes([0]).unwrap();
        let cta_masks = WarpValue::splat(0b11_i64);
        let plan = plan_tcgen_commit_issue(&context, &pointer, one_lane, Some(&cta_masks))
            .unwrap()
            .unwrap();

        assert_eq!(
            plan.barrier_ids()
                .iter()
                .map(|barrier_id| barrier_id.target_global_cta_id())
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        let mbarriers = PhysicalBarrierHub::new();
        mbarriers.init(plan.barrier_ids()[0], 1).unwrap();
        assert!(plan.apply(&mbarriers).is_err());
        assert_eq!(mbarriers.pending_completion_count(), 0);

        mbarriers.init(plan.barrier_ids()[1], 1).unwrap();
        let actions = plan.apply(&mbarriers).unwrap();
        assert_eq!(actions.len(), 2);
        assert_eq!(
            actions
                .iter()
                .map(|action| action.id().get())
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(actions
            .iter()
            .all(|action| { action.arrival() == Some((context.global_warp_id(), 1)) }));
    }
}
