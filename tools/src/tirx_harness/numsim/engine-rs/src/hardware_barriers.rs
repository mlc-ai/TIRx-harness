use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

use crate::completion_action_id::CompletionActionNamespace;
use crate::{
    BlockedOperation, CompletionProgress, CompletionSource, DynamicOpId, EngineError,
    OccurrenceKey, ParticipantState, ScopeInstance, SynchronizationError, WarpMask, WARP_SIZE,
};

pub const MAX_MBARRIER_EXPECTED_ARRIVALS: u64 = (1 << 20) - 1;
pub const MAX_MBARRIER_TRANSACTIONS: u64 = (1 << 20) - 1;

pub(crate) const fn mbarrier_arrival_limit(layout_v1: bool) -> u64 {
    if layout_v1 {
        (1 << 9) - 1
    } else {
        MAX_MBARRIER_EXPECTED_ARRIVALS
    }
}

/// Identity of one physical shared-memory mbarrier slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalBarrierId {
    allocation_id: u64,
    byte_offset: usize,
    target_global_cta_id: usize,
}

impl PhysicalBarrierId {
    pub const fn new(allocation_id: u64, byte_offset: usize, target_global_cta_id: usize) -> Self {
        Self {
            allocation_id,
            byte_offset,
            target_global_cta_id,
        }
    }

    pub const fn allocation_id(self) -> u64 {
        self.allocation_id
    }

    pub const fn byte_offset(self) -> usize {
        self.byte_offset
    }

    pub const fn target_global_cta_id(self) -> usize {
        self.target_global_cta_id
    }

    fn occurrence_key(self) -> OccurrenceKey {
        let offset = i64::try_from(self.byte_offset).unwrap_or(i64::MAX);
        OccurrenceKey::new(
            self.allocation_id,
            format!(
                "mbarrier[allocation#{}+{}]",
                self.allocation_id, self.byte_offset
            ),
            [offset],
            ScopeInstance::Cta {
                global_cta_id: self.target_global_cta_id,
            },
        )
    }
}

/// Engine-owned state for physical SMEM mbarriers.
///
/// One generated Future represents one warp, so arrivals are weighted by the
/// number of active lanes that execute the intrinsic. The hardware parity bit
/// starts at one: a wait for phase one passes immediately after init, while a
/// wait for phase zero blocks until the first generation completes.
pub struct PhysicalBarrierHub {
    // Publication gates, striped by barrier identity. A gate serializes the
    // commit, the analysis publication, and the wake of one barrier's
    // transitions against that barrier's waiters and queries, so no waiter
    // resumes before the outcome it depends on is published. Transitions of
    // different barriers carry no mutual order, so they proceed in parallel
    // on the completion pump instead of queueing on one hub-wide lock.
    publication_gates: Box<[Mutex<()>]>,
    // Barrier state, striped exactly like the gates: a transition of one
    // barrier locks only its stripe, so arrivals, polls and completions of
    // unrelated barriers no longer serialize on one hub-wide mutex.
    shards: Box<[Mutex<PhysicalBarrierState>]>,
    // Completion IDs stay launch-wide monotonic, so `pending_completion_actions`
    // still yields the exact scheduler queue order across stripes.
    next_completion_action_id: AtomicU64,
    // The stripe a queued action lives in is fixed by its barrier when it is
    // enqueued; this index resolves an action ID to that barrier without
    // scanning every stripe. It is taken on its own, or inside the stripe
    // locks of the enqueue that publishes the action — never the other way
    // round.
    action_barriers: Mutex<BTreeMap<PhysicalCompletionActionId, PhysicalBarrierId>>,
}

const PUBLICATION_GATE_STRIPES: usize = 1024;

impl Default for PhysicalBarrierHub {
    fn default() -> Self {
        Self {
            publication_gates: (0..PUBLICATION_GATE_STRIPES)
                .map(|_| Mutex::new(()))
                .collect(),
            shards: (0..PUBLICATION_GATE_STRIPES)
                .map(|_| Mutex::default())
                .collect(),
            next_completion_action_id: AtomicU64::new(0),
            action_barriers: Mutex::default(),
        }
    }
}

/// The stripes one hub operation holds, in stripe order.
struct LockedShards<'a> {
    guards: Vec<(usize, MutexGuard<'a, PhysicalBarrierState>)>,
}

impl LockedShards<'_> {
    fn position(&self, id: PhysicalBarrierId) -> usize {
        let stripe = PhysicalBarrierHub::publication_stripe(id);
        self.guards
            .binary_search_by_key(&stripe, |(stripe, _)| *stripe)
            .expect("physical barrier stripe is locked by this operation")
    }

    fn shard(&self, id: PhysicalBarrierId) -> &PhysicalBarrierState {
        &self.guards[self.position(id)].1
    }

    fn shard_mut(&mut self, id: PhysicalBarrierId) -> &mut PhysicalBarrierState {
        let position = self.position(id);
        &mut self.guards[position].1
    }

    fn entry(&self, id: PhysicalBarrierId) -> Option<&PhysicalBarrierEntry> {
        self.shard(id).entries.get(&id)
    }

    fn entry_mut(&mut self, id: PhysicalBarrierId) -> Option<&mut PhysicalBarrierEntry> {
        self.shard_mut(id).entries.get_mut(&id)
    }

    /// Merge the held stripes into one state, so a multi-barrier transaction
    /// can be staged and validated as a whole before it is installed.
    fn staged(&self) -> PhysicalBarrierState {
        let mut staged = PhysicalBarrierState::default();
        for (_, shard) in &self.guards {
            staged
                .entries
                .extend(shard.entries.iter().map(|(id, entry)| (*id, entry.clone())));
            staged.transaction_completions.extend(
                shard
                    .transaction_completions
                    .iter()
                    .map(|(action_id, action)| (*action_id, *action)),
            );
        }
        staged
    }

    /// Install a state produced by [`Self::staged`]: every held stripe takes
    /// back exactly the entries and queued actions that belong to it.
    fn install(&mut self, staged: PhysicalBarrierState) {
        for (_, shard) in &mut self.guards {
            shard.entries.clear();
            shard.transaction_completions.clear();
        }
        for (id, entry) in staged.entries {
            self.shard_mut(id).entries.insert(id, entry);
        }
        for (action_id, action) in staged.transaction_completions {
            self.shard_mut(action.barrier_id())
                .transaction_completions
                .insert(action_id, action);
        }
    }
}

impl PhysicalBarrierHub {
    fn publication_stripe(id: PhysicalBarrierId) -> usize {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        id.hash(&mut hasher);
        (hasher.finish() % PUBLICATION_GATE_STRIPES as u64) as usize
    }

    /// Hold the publication gates of `ids`, in stripe order so that any set
    /// of barriers is always locked the same way.
    fn lock_publication(
        &self,
        ids: impl IntoIterator<Item = PhysicalBarrierId>,
    ) -> Vec<std::sync::MutexGuard<'_, ()>> {
        let mut stripes = ids
            .into_iter()
            .map(Self::publication_stripe)
            .collect::<Vec<_>>();
        stripes.sort_unstable();
        stripes.dedup();
        stripes
            .into_iter()
            .map(|stripe| {
                self.publication_gates[stripe]
                    .lock()
                    .expect("physical barrier publication gate poisoned")
            })
            .collect()
    }

    /// Hold the state stripes of `ids`, in stripe order so that any set of
    /// barriers is always locked the same way.
    fn lock_shards(&self, ids: impl IntoIterator<Item = PhysicalBarrierId>) -> LockedShards<'_> {
        let mut stripes = ids
            .into_iter()
            .map(Self::publication_stripe)
            .collect::<Vec<_>>();
        stripes.sort_unstable();
        stripes.dedup();
        LockedShards {
            guards: stripes
                .into_iter()
                .map(|stripe| {
                    (
                        stripe,
                        self.shards[stripe]
                            .lock()
                            .expect("physical barrier mutex poisoned"),
                    )
                })
                .collect(),
        }
    }

    /// Visit every stripe, one at a time.
    fn for_each_shard(&self, mut visit: impl FnMut(&PhysicalBarrierState)) {
        for shard in &self.shards {
            visit(&shard.lock().expect("physical barrier mutex poisoned"));
        }
    }

    /// The barriers the queued completion actions target. An action's
    /// barrier is fixed when it is enqueued, so looking it up before taking
    /// the gate is race-free; an unknown action yields no gate and fails in
    /// the caller's own lookup.
    fn completion_barrier_ids(
        &self,
        action_ids: &[PhysicalCompletionActionId],
    ) -> Vec<PhysicalBarrierId> {
        self.indexed_actions(action_ids)
            .into_iter()
            .map(|(_, barrier_id)| barrier_id)
            .collect()
    }

    /// The known actions among `action_ids`, each with its barrier.
    fn indexed_actions(
        &self,
        action_ids: &[PhysicalCompletionActionId],
    ) -> Vec<(PhysicalCompletionActionId, PhysicalBarrierId)> {
        let index = self
            .action_barriers
            .lock()
            .expect("physical barrier action index poisoned");
        action_ids
            .iter()
            .filter_map(|action_id| {
                index
                    .get(action_id)
                    .map(|barrier_id| (*action_id, *barrier_id))
            })
            .collect()
    }

    /// Record which barrier each queued action targets. Called while the
    /// actions' stripes are held, before the actions become visible, so a
    /// reader that finds an action in its stripe can always resolve it.
    fn index_actions<'a>(&self, actions: impl IntoIterator<Item = &'a PhysicalCompletionAction>) {
        let mut index = self
            .action_barriers
            .lock()
            .expect("physical barrier action index poisoned");
        for action in actions {
            index.insert(action.id(), action.barrier_id());
        }
    }

    fn unindex_actions(&self, action_ids: impl IntoIterator<Item = PhysicalCompletionActionId>) {
        let mut index = self
            .action_barriers
            .lock()
            .expect("physical barrier action index poisoned");
        for action_id in action_ids {
            index.remove(&action_id);
        }
    }

    /// Reserve `count` consecutive completion action IDs and return the first.
    fn allocate_action_ids(&self, count: u64) -> Result<u64, SynchronizationError> {
        let first = self.next_completion_action_id.fetch_add(count, Ordering::Relaxed);
        first
            .checked_add(count)
            .filter(|end| u128::from(*end) <= CompletionActionNamespace::PhysicalBarrier.capacity())
            .map(|_| first)
            .ok_or_else(|| SynchronizationError::CompletionSourceOperationFailed {
                source_name: "physical-mbarrier",
                details: "completion action ID space exhausted".to_string(),
            })
    }
}

#[derive(Clone, Default)]
struct PhysicalBarrierState {
    entries: BTreeMap<PhysicalBarrierId, PhysicalBarrierEntry>,
    // Completion IDs are allocated monotonically, so key order is the exact
    // scheduler queue order.  Keeping the ID as the map key makes selected
    // completion lookup/removal logarithmic instead of repeatedly scanning
    // and shifting a potentially launch-wide VecDeque.
    transaction_completions: BTreeMap<PhysicalCompletionActionId, PhysicalCompletionAction>,
}

/// Which architected phase identity a nonblocking mbarrier query observes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MbarrierQuery {
    PrimaryParity,
    PrimaryState,
    ConditionalParity,
}

#[derive(Clone)]
struct PhysicalBarrierEntry {
    layout_v1: bool,
    previous_report: bool,
    expected_arrivals: u64,
    generation: u64,
    last_completed_generation: Option<u64>,
    completed_parity: u64,
    conditional_completed_parity: u64,
    conditional_completed_generation: Option<u64>,
    phase: CountedPhase,
    buffered_transactions: BTreeMap<u64, u64>,
    waiters: BTreeMap<usize, PhysicalWaiter>,
}

impl PhysicalBarrierEntry {
    /// Arrivals the current generation must receive before it can complete.
    ///
    /// This is the barrier's steady-state expectation plus whatever pending
    /// arrivals the current phase had raised on top of it.
    const fn required_arrivals(&self) -> u64 {
        self.expected_arrivals
            .saturating_add(self.phase.pending_arrival_increments)
    }
}

#[derive(Clone, Default)]
struct CountedPhase {
    report: bool,
    arrival_count: u64,
    arrived_warps: BTreeSet<usize>,
    expected_transactions: u64,
    completed_transactions: u64,
    /// Extra arrivals this generation must receive on top of
    /// `PhysicalBarrierEntry::expected_arrivals`.
    ///
    /// `cp.async.mbarrier.arrive` (without `.noinc`) raises the pending arrival
    /// count of the *current* phase only; the barrier's steady-state
    /// expectation is unchanged, so the raise is dropped when the generation
    /// ends rather than stored on the entry.
    pending_arrival_increments: u64,
    complete: bool,
}

#[derive(Clone)]
struct PhysicalWaiter {
    requested_phase: u64,
    operation: Option<DynamicOpId>,
    waker: Waker,
}

/// Stable identity assigned to one physical-barrier completion action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalCompletionActionId(u64);

impl PhysicalCompletionActionId {
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Tag a monotonic ordinal into the physical-barrier completion namespace.
    ///
    /// The ordinal *is* the identity here — `transaction_completions` keys its
    /// scheduler queue on the ID and relies on key order being issue order — so
    /// unlike the derived namespaces this one allocates. The namespace base is
    /// zero, so every value is numerically what the bare counter produced; what
    /// is new is that the upper bound is now checked rather than assumed.
    fn from_ordinal(ordinal: u64) -> Option<Self> {
        CompletionActionNamespace::PhysicalBarrier
            .tag(ordinal as u128)
            .map(Self)
    }
}

impl fmt::Display for PhysicalCompletionActionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Concrete mutation performed by one schedulable physical completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicalCompletionKind {
    Transaction { transactions: u64 },
    Arrival { warp_id: usize, arrival_count: u64 },
}

/// Immutable description of one schedulable physical-barrier completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalCompletionAction {
    action_id: PhysicalCompletionActionId,
    barrier_id: PhysicalBarrierId,
    generation: u64,
    kind: PhysicalCompletionKind,
}

/// Detailed result of one selected physical completion transition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalCompletionOutcome {
    action: PhysicalCompletionAction,
    progress: CompletionProgress,
    completed_generation: Option<u64>,
    conditional_completed_generation: Option<u64>,
    arrival_outcome: Option<PhysicalMbarrierArrivalOutcome>,
    woken_warp_ids: Box<[usize]>,
}

/// Exact result of one committed physical mbarrier arrival.
///
/// The generation and completion transition are captured while the numeric
/// hub lock is held, so analysis modes never infer them from waiter wake or
/// polling order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalMbarrierArrivalOutcome {
    generation: u64,
    completed_now: bool,
    pending_arrivals_before: u64,
    conditional_completed_generation: Option<u64>,
}

impl PhysicalMbarrierArrivalOutcome {
    pub const fn new(generation: u64, completed_now: bool) -> Self {
        Self {
            generation,
            completed_now,
            pending_arrivals_before: 0,
            conditional_completed_generation: if completed_now {
                Some(generation)
            } else {
                None
            },
        }
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }

    pub const fn completed_now(self) -> bool {
        self.completed_now
    }

    /// The physical owner's retained conditional completion, not a parity guess.
    pub(crate) const fn conditional_completed_generation(self) -> Option<u64> {
        self.conditional_completed_generation
    }

    pub(crate) const fn with_conditional_generation(mut self, generation: Option<u64>) -> Self {
        self.conditional_completed_generation = generation;
        self
    }

    /// The strict counter protocol validates these fields independently. Report
    /// phase selection remains owned by the physical barrier, not that protocol.
    pub(crate) fn same_primary_transition(self, expected: Self) -> bool {
        self.generation == expected.generation
            && self.completed_now == expected.completed_now
            && self.pending_arrivals_before == expected.pending_arrivals_before
    }

    pub(crate) const fn with_pending_arrivals_before(mut self, count: u64) -> Self {
        self.pending_arrivals_before = count;
        self
    }

    /// Retain the arrival-time count, not a later snapshot of the barrier.
    /// Lanes sharing a barrier use the engine's ascending-lane arrival order.
    pub(crate) fn write_states<const NO_COMPLETE: bool>(
        self,
        states: &mut crate::WarpValue<u64>,
        mask: WarpMask,
        counts: Option<&crate::WarpValue<i64>>,
    ) -> Result<(), EngineError> {
        if self.generation >= MBARRIER_STATE_NO_COMPLETE {
            return Err(EngineError::message(
                "mbarrier state generation exceeds token capacity",
            ));
        }
        let mut pending = self.pending_arrivals_before;
        for lane in mask {
            states[lane] = if NO_COMPLETE {
                if pending > MAX_MBARRIER_EXPECTED_ARRIVALS {
                    return Err(EngineError::message(
                        "mbarrier pending count exceeds token capacity",
                    ));
                }
                let state = self.generation | MBARRIER_STATE_NO_COMPLETE | (pending << 44);
                pending -= counts.map_or(1, |values| values[lane] as u64);
                state
            } else {
                self.generation
            };
        }
        Ok(())
    }
}

// Private opaque-token encoding: 43 generation bits, a noComplete producer
// bit, and 20 pending-count bits. This is not the hardware barrier layout.
const MBARRIER_STATE_NO_COMPLETE: u64 = 1 << 43;

pub(crate) const fn mbarrier_state_generation(state: u64) -> u64 {
    if state & MBARRIER_STATE_NO_COMPLETE != 0 {
        state & (MBARRIER_STATE_NO_COMPLETE - 1)
    } else {
        state
    }
}

pub(crate) fn mbarrier_state_pending_count(state: u64) -> Result<u32, EngineError> {
    if state & MBARRIER_STATE_NO_COMPLETE == 0 {
        return Err(EngineError::message(
            "mbarrier.pending_count requires a state from arrive/arrive_drop.noComplete",
        ));
    }
    Ok((state >> 44) as u32)
}

impl PhysicalCompletionOutcome {
    pub const fn action(&self) -> PhysicalCompletionAction {
        self.action
    }

    pub const fn progress(&self) -> CompletionProgress {
        self.progress
    }

    pub const fn completed_generation(&self) -> Option<u64> {
        self.completed_generation
    }

    pub(crate) const fn conditional_completed_generation(&self) -> Option<u64> {
        self.conditional_completed_generation
    }

    pub const fn arrival_outcome(&self) -> Option<PhysicalMbarrierArrivalOutcome> {
        self.arrival_outcome
    }

    pub fn woken_warp_ids(&self) -> &[usize] {
        &self.woken_warp_ids
    }
}

impl PhysicalCompletionAction {
    pub const fn id(self) -> PhysicalCompletionActionId {
        self.action_id
    }

    pub const fn barrier_id(self) -> PhysicalBarrierId {
        self.barrier_id
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }

    pub const fn kind(self) -> PhysicalCompletionKind {
        self.kind
    }

    pub const fn transactions(self) -> u64 {
        match self.kind {
            PhysicalCompletionKind::Transaction { transactions } => transactions,
            PhysicalCompletionKind::Arrival { .. } => 0,
        }
    }

    pub const fn arrival(self) -> Option<(usize, u64)> {
        match self.kind {
            PhysicalCompletionKind::Arrival {
                warp_id,
                arrival_count,
            } => Some((warp_id, arrival_count)),
            PhysicalCompletionKind::Transaction { .. } => None,
        }
    }
}

/// Fail-closed result of selecting one physical completion action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhysicalCompletionActionError {
    Missing {
        action_id: PhysicalCompletionActionId,
    },
    NotEnabled {
        action: PhysicalCompletionAction,
        current_generation: u64,
        current_generation_complete: bool,
    },
    ApplyFailed {
        action: PhysicalCompletionAction,
        source: SynchronizationError,
    },
}

impl PhysicalCompletionActionError {
    fn into_synchronization_error(self) -> SynchronizationError {
        match self {
            Self::ApplyFailed { source, .. } => source,
            error => SynchronizationError::CompletionSourceOperationFailed {
                source_name: "physical-mbarrier",
                details: error.to_string(),
            },
        }
    }
}

impl fmt::Display for PhysicalCompletionActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { action_id } => {
                write!(f, "physical completion action {action_id} is not pending")
            }
            Self::NotEnabled {
                action,
                current_generation,
                current_generation_complete,
            } => write!(
                f,
                "physical completion action {} targets generation {}, but barrier {:?} is at generation {} (complete={})",
                action.id(),
                action.generation(),
                action.barrier_id(),
                current_generation,
                current_generation_complete,
            ),
            Self::ApplyFailed { action, source } => {
                write!(f, "physical completion action {} failed: {source}", action.id())
            }
        }
    }
}

impl Error for PhysicalCompletionActionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ApplyFailed { source, .. } => Some(source),
            Self::Missing { .. } | Self::NotEnabled { .. } => None,
        }
    }
}

/// Engine-owned record of which mbarriers a warp has initialized but not yet
/// covered with `fence.mbarrier_init`.
///
/// This is the engine's own answer to "which barriers does this fence cover",
/// published as the `MbarrierInitFence` effect payload so a checker never has
/// to re-derive it from private causality state.
///
/// The update rule deliberately mirrors `SyncCausality`'s barrier map: there,
/// an initialization and a re-initialization both overwrite the owning
/// (warp, lane) and both reset fence eligibility, so the fence-relevant rule is
/// the same on either path and this tracker never has to tell them apart. Both
/// derivations are driven by the same `MbarrierInit` effect, and both key on
/// `PhysicalBarrierId`, so the covered set agrees in content and in order.
#[derive(Debug, Default)]
pub struct MbarrierInitFenceTracker {
    slots: Mutex<BTreeMap<PhysicalBarrierId, InitFenceSlot>>,
}

#[derive(Clone, Copy, Debug)]
struct InitFenceSlot {
    init_warp_id: usize,
    init_lane_id: usize,
    fenced: bool,
}

impl MbarrierInitFenceTracker {
    pub(crate) fn invalidate_many(&self, ids: &[PhysicalBarrierId]) {
        let mut slots = self
            .slots
            .lock()
            .expect("mbarrier init-fence tracker poisoned");
        for id in ids {
            slots.remove(id);
        }
    }

    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record one `mbarrier.init` lane target, re-arming fence eligibility.
    pub(crate) fn record_init(
        &self,
        barrier_id: PhysicalBarrierId,
        init_warp_id: usize,
        init_lane_id: usize,
    ) {
        self.slots
            .lock()
            .expect("mbarrier init-fence tracker poisoned")
            .insert(
                barrier_id,
                InitFenceSlot {
                    init_warp_id,
                    init_lane_id,
                    fenced: false,
                },
            );
    }

    /// Take the set one `fence.mbarrier_init` covers: every still-unfenced
    /// barrier this warp initialized from a lane in `mask`, in barrier-id order.
    ///
    /// Concurrency note for the announced follow-up. Today the returned set is
    /// only *compared* against synccheck's own selection, so the window between
    /// this call and the checker applying it is compare-only: any interleaving
    /// with another worker either matches or fails loud. Once the checker-side
    /// selection is retired and the published set becomes the recorded payload,
    /// that same window turns payload-affecting -- what a concurrent worker
    /// observes here decides what gets recorded, not merely whether an assertion
    /// holds. The commit that retires the checker-side selection must carry that
    /// reasoning and re-examine this window; it is not covered by the soak that
    /// justified the compare-only form.
    pub(crate) fn take_fenced(&self, warp_id: usize, mask: WarpMask) -> Vec<PhysicalBarrierId> {
        let mut slots = self
            .slots
            .lock()
            .expect("mbarrier init-fence tracker poisoned");
        let mut covered = Vec::new();
        for (&barrier_id, slot) in slots.iter_mut() {
            if slot.fenced || slot.init_warp_id != warp_id || !mask.contains(slot.init_lane_id) {
                continue;
            }
            slot.fenced = true;
            covered.push(barrier_id);
        }
        covered
    }
}

impl PhysicalBarrierHub {
    /// Copy inspection contributes to the same phase as its transaction credit.
    /// Inspection has no independently observable event: queries expose the
    /// accumulated predicate only after that primary phase completes.
    pub(crate) fn report_on(
        &self,
        id: PhysicalBarrierId,
        matched: bool,
    ) -> Result<(), EngineError> {
        let _publication = self.lock_publication([id]);
        let mut shards = self.lock_shards([id]);
        let entry =
            shards
                .entry_mut(id)
                .ok_or_else(|| SynchronizationError::BarrierUninitialized {
                    key: id.occurrence_key(),
                })?;
        if !entry.layout_v1 {
            return Err(EngineError::message(
                "copy reporting requires mbarrier layout::v1",
            ));
        }
        begin_next_generation(entry);
        entry.phase.report |= matched;
        Ok(())
    }

    pub(crate) fn check_layout(
        &self,
        id: PhysicalBarrierId,
        layout: u8,
    ) -> Result<bool, EngineError> {
        let shards = self.lock_shards([id]);
        let entry = shards
            .entry(id)
            .ok_or_else(|| SynchronizationError::BarrierUninitialized {
                key: id.occurrence_key(),
            })?;
        Ok(entry.layout_v1 == (layout == 1))
    }

    pub(crate) fn invalidate_many(&self, ids: &[PhysicalBarrierId]) -> Result<(), EngineError> {
        let _publication = self.lock_publication(ids.iter().copied());
        let mut shards = self.lock_shards(ids.iter().copied());
        let ids: BTreeSet<_> = ids.iter().copied().collect();
        for id in &ids {
            let entry = shards.entry(*id).ok_or_else(|| {
                EngineError::from(SynchronizationError::BarrierUninitialized {
                    key: id.occurrence_key(),
                })
            })?;
            // A partially arrived count can be abandoned; a queued actor or
            // async completion cannot be silently detached from its barrier.
            if !entry.waiters.is_empty()
                || shards
                    .shard(*id)
                    .transaction_completions
                    .values()
                    .any(|event| event.barrier_id == *id)
            {
                return Err(EngineError::message(format!(
                    "mbarrier.inval {id:?} has outstanding waiters or asynchronous completions"
                )));
            }
        }
        for id in ids {
            shards.shard_mut(id).entries.remove(&id);
        }
        Ok(())
    }

    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn init(
        &self,
        id: PhysicalBarrierId,
        expected_arrivals: u64,
    ) -> Result<(), SynchronizationError> {
        self.init_many(std::slice::from_ref(&id), expected_arrivals)
    }

    /// Initialize a complete warp-level mbarrier target set transactionally.
    ///
    /// `mbarrier.init` is idempotent when several active lanes in one
    /// instruction name the same physical slot.  Collapse those repeated
    /// targets before validation; lane-varying pointers that name distinct
    /// slots remain one target per lane.  Every distinct target is validated
    /// under the same hub lock before any target changes, so a failure cannot
    /// leave a prefix initialized.
    pub(crate) fn init_many(
        &self,
        ids: &[PhysicalBarrierId],
        expected_arrivals: u64,
    ) -> Result<(), SynchronizationError> {
        self.init_many_layout(ids, expected_arrivals, false)
    }

    pub(crate) fn init_many_layout(
        &self,
        ids: &[PhysicalBarrierId],
        expected_arrivals: u64,
        layout_v1: bool,
    ) -> Result<(), SynchronizationError> {
        self.init_many_impl(ids, expected_arrivals, true, layout_v1)
    }

    /// Initialize numerical state, permitting reuse of a completed slot.
    pub(crate) fn init_numeric_layout(
        &self,
        id: PhysicalBarrierId,
        expected_arrivals: u64,
        layout_v1: bool,
    ) -> Result<(), SynchronizationError> {
        self.init_many_impl(
            std::slice::from_ref(&id),
            expected_arrivals,
            false,
            layout_v1,
        )
    }

    fn init_many_impl(
        &self,
        ids: &[PhysicalBarrierId],
        expected_arrivals: u64,
        require_invalidation: bool,
        layout_v1: bool,
    ) -> Result<(), SynchronizationError> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut shards = self.lock_shards(ids.iter().copied());
        if !(1..=mbarrier_arrival_limit(layout_v1)).contains(&expected_arrivals) {
            return Err(SynchronizationError::InvalidBarrierArrivalCount {
                key: ids[0].occurrence_key(),
                count: expected_arrivals,
            });
        }
        let distinct_ids = ids.iter().copied().collect::<BTreeSet<_>>();
        for id in distinct_ids {
            let key = id.occurrence_key();
            let queued_completions = shards
                .shard(id)
                .transaction_completions
                .values()
                .filter(|event| event.barrier_id == id)
                .count();
            if let Some(entry) = shards.entry(id) {
                if !entry.waiters.is_empty() {
                    return Err(SynchronizationError::BarrierReinitializedWhileWaiting {
                        key,
                        waiting_warps: entry.waiters.keys().copied().collect(),
                    });
                }
                if queued_completions != 0
                    || physical_phase_is_active(&entry.phase)
                    || !entry.buffered_transactions.is_empty()
                {
                    return Err(SynchronizationError::BarrierReinitializedWhileActive {
                        key,
                        generation: entry.generation,
                        details: format!(
                            "queued_completions={queued_completions}, arrivals={}/{}, transactions={}/{}, complete={}, buffered_transactions={:?}",
                            entry.phase.arrival_count,
                            entry.required_arrivals(),
                            entry.phase.completed_transactions,
                            entry.phase.expected_transactions,
                            entry.phase.complete,
                            entry.buffered_transactions,
                        ),
                    });
                }
                if require_invalidation {
                    return Err(
                        SynchronizationError::BarrierReinitializedWithoutInvalidation { key },
                    );
                }
            }
        }
        for id in ids.iter().copied().collect::<BTreeSet<_>>() {
            shards.shard_mut(id).entries.insert(
                id,
                PhysicalBarrierEntry {
                    layout_v1,
                    previous_report: false,
                    expected_arrivals,
                    generation: 0,
                    last_completed_generation: None,
                    completed_parity: 1,
                    conditional_completed_parity: 1,
                    conditional_completed_generation: None,
                    phase: CountedPhase::default(),
                    buffered_transactions: BTreeMap::new(),
                    waiters: BTreeMap::new(),
                },
            );
        }
        Ok(())
    }

    pub(crate) fn arrive(
        &self,
        id: PhysicalBarrierId,
        warp_id: usize,
        arrival_count: u64,
    ) -> Result<PhysicalMbarrierArrivalOutcome, SynchronizationError> {
        self.arrive_impl(id, warp_id, arrival_count, 0)
    }

    /// Arrive and arm an exact transaction-byte expectation.
    ///
    /// The asynchronous payload operation separately calls
    /// [`Self::enqueue_transaction_completion`] with the number of bytes it
    /// actually delivered. Keeping those two events separate makes an
    /// expect/copy mismatch observable instead of crediting the expectation
    /// itself as completed work.
    pub(crate) fn arrive_expect_tx(
        &self,
        id: PhysicalBarrierId,
        warp_id: usize,
        arrival_count: u64,
        expected_transactions: u64,
    ) -> Result<PhysicalMbarrierArrivalOutcome, SynchronizationError> {
        self.arrive_impl(id, warp_id, arrival_count, expected_transactions)
    }

    /// Apply one `mbarrier.expect_tx` instruction's complete target set.
    ///
    /// Unlike `arrive_expect_tx`, this transition does not consume an arrival
    /// and does not add the issuing warp to `arrived_warps`.  The input is
    /// expected to be aggregated by physical barrier; validating every target
    /// against cloned state before publishing keeps a lane-varying instruction
    /// transactional.
    pub(crate) fn expect_tx_many(
        &self,
        expectations: &[(PhysicalBarrierId, u64)],
    ) -> Result<Box<[(PhysicalBarrierId, u64)]>, SynchronizationError> {
        if expectations.is_empty() {
            return Ok(Box::new([]));
        }
        let mut shards = self.lock_shards(expectations.iter().map(|&(id, _)| id));
        let mut candidates = BTreeMap::new();
        for &(id, transactions) in expectations {
            if candidates.contains_key(&id) {
                return Err(SynchronizationError::CompletionSourceOperationFailed {
                    source_name: "mbarrier.expect_tx",
                    details: format!("duplicate physical target {id:?}"),
                });
            }
            let mut candidate = shards.entry(id).cloned().ok_or_else(|| {
                SynchronizationError::BarrierUninitialized {
                    key: id.occurrence_key(),
                }
            })?;
            apply_physical_expect_tx_entry(id, &mut candidate, transactions)?;
            candidates.insert(id, candidate);
        }
        let generations = candidates
            .iter()
            .map(|(&barrier_id, candidate)| (barrier_id, candidate.generation))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        for (id, candidate) in candidates {
            shards.shard_mut(id).entries.insert(id, candidate);
        }
        Ok(generations)
    }

    /// Raise the current phase's pending arrival count on each named barrier.
    ///
    /// This is the immediate half of `cp.async.mbarrier.arrive` without
    /// `.noinc`; the matching arrive-on is enqueued separately against the
    /// issuing lane's `cp.async` completion. Like [`Self::expect_tx_many`] the
    /// whole target set is validated against cloned state before publication,
    /// so a lane-varying instruction cannot leave a prefix raised.
    pub(crate) fn increase_pending_arrivals_many(
        &self,
        increases: &[(PhysicalBarrierId, u64)],
    ) -> Result<(), SynchronizationError> {
        if increases.is_empty() {
            return Ok(());
        }
        let mut shards = self.lock_shards(increases.iter().map(|&(id, _)| id));
        let mut candidates = BTreeMap::new();
        for &(id, increase) in increases {
            if candidates.contains_key(&id) {
                return Err(SynchronizationError::CompletionSourceOperationFailed {
                    source_name: "cp.async.mbarrier.arrive",
                    details: format!("duplicate physical target {id:?}"),
                });
            }
            let mut candidate = shards.entry(id).cloned().ok_or_else(|| {
                SynchronizationError::BarrierUninitialized {
                    key: id.occurrence_key(),
                }
            })?;
            apply_physical_pending_arrival_increase_entry(id, &mut candidate, increase)?;
            candidates.insert(id, candidate);
        }
        for (id, candidate) in candidates {
            shards.shard_mut(id).entries.insert(id, candidate);
        }
        Ok(())
    }

    /// Apply one warp instruction's complete physical-mbarrier target set.
    ///
    /// Lane-varying remote CTA operands can resolve to several barriers. Every
    /// target is validated against one candidate snapshot before the hub state
    /// changes, so a later invalid target cannot leave an earlier arrival
    /// committed.
    pub(crate) fn arrive_many(
        &self,
        arrivals: &[(PhysicalBarrierId, usize, u64, u64)],
    ) -> Result<Box<[PhysicalMbarrierArrivalOutcome]>, SynchronizationError> {
        self.arrive_many_with_outcomes(arrivals, |_| Ok(()))
            .map_err(|error| match error {
                PhysicalArrivalCommitError::Barrier(error) => error,
                PhysicalArrivalCommitError::Publication(error) => {
                    unreachable!("empty physical arrival publication failed: {error}")
                }
            })
    }

    /// Apply a complete warp arrival transaction, publish its exact outcomes,
    /// and only then wake waiters made ready by the numeric transition.
    pub(crate) fn arrive_many_with_outcomes(
        &self,
        arrivals: &[(PhysicalBarrierId, usize, u64, u64)],
        publish_before_wake: impl FnOnce(&[PhysicalMbarrierArrivalOutcome]) -> Result<(), EngineError>,
    ) -> Result<Box<[PhysicalMbarrierArrivalOutcome]>, PhysicalArrivalCommitError> {
        let _publication = self.lock_publication(arrivals.iter().map(|&(id, _, _, _)| id));
        if arrivals.is_empty() {
            let outcomes = Vec::new().into_boxed_slice();
            publish_before_wake(&outcomes).map_err(PhysicalArrivalCommitError::Publication)?;
            return Ok(outcomes);
        }
        let mut distinct_ids = BTreeSet::new();
        for &(id, _, _, _) in arrivals {
            if !distinct_ids.insert(id) {
                return Err(PhysicalArrivalCommitError::Barrier(
                    SynchronizationError::DuplicateMbarrierArrivalTarget {
                        key: id.occurrence_key(),
                    },
                ));
            }
        }
        let (outcomes, wakers) = {
            let mut shards = self.lock_shards(arrivals.iter().map(|&(id, _, _, _)| id));
            let mut candidates = BTreeMap::new();
            let mut outcomes = Vec::with_capacity(arrivals.len());
            let mut wakers = Vec::new();
            for &(id, warp_id, arrival_count, expected_transactions) in arrivals {
                if !candidates.contains_key(&id) {
                    let entry = shards.entry(id).cloned().ok_or_else(|| {
                        PhysicalArrivalCommitError::Barrier(
                            SynchronizationError::BarrierUninitialized {
                                key: id.occurrence_key(),
                            },
                        )
                    })?;
                    candidates.insert(id, entry);
                }
                let (outcome, target_wakers) = apply_physical_arrival_entry(
                    id,
                    candidates
                        .get_mut(&id)
                        .expect("batch candidate inserted above"),
                    warp_id,
                    arrival_count,
                    expected_transactions,
                )
                .map_err(PhysicalArrivalCommitError::Barrier)?;
                outcomes.push(outcome);
                wakers.extend(target_wakers);
            }
            for (id, candidate) in candidates {
                shards.shard_mut(id).entries.insert(id, candidate);
            }
            (outcomes.into_boxed_slice(), wakers)
        };
        publish_before_wake(&outcomes).map_err(PhysicalArrivalCommitError::Publication)?;
        wake_all(wakers);
        Ok(outcomes)
    }

    /// The warp executor calls this immediately before committing an arrival,
    /// without yielding. noComplete must leave a positive pending arrival count,
    /// even when outstanding transactions would prevent phase completion.
    pub(crate) fn validate_no_complete(
        &self,
        arrivals: impl IntoIterator<Item = (PhysicalBarrierId, u64)>,
    ) -> Result<(), EngineError> {
        let arrivals = arrivals.into_iter().collect::<Vec<_>>();
        let shards = self.lock_shards(arrivals.iter().map(|(id, _)| *id));
        for (id, count) in arrivals {
            let entry = shards.entry(id).ok_or_else(|| {
                EngineError::from(SynchronizationError::BarrierUninitialized {
                    key: id.occurrence_key(),
                })
            })?;
            // Completed phases roll over lazily on the next arrival.
            let pending = if entry.phase.complete {
                entry.expected_arrivals
            } else {
                entry
                    .required_arrivals()
                    .saturating_sub(entry.phase.arrival_count)
            };
            if count >= pending {
                return Err(EngineError::message(format!(
                    "mbarrier.arrive.noComplete count {count} must be less than pending arrival count {pending}"
                )));
            }
        }
        Ok(())
    }

    /// Reduce future-phase expectations without changing the current phase's
    /// pending count. The ordinary arrival immediately following this call
    /// consumes that count and publishes the existing release/completion effect.
    pub(crate) fn drop_expected_arrivals(
        &self,
        arrivals: impl IntoIterator<Item = (PhysicalBarrierId, u64)>,
    ) -> Result<(), EngineError> {
        let arrivals = arrivals.into_iter().collect::<Vec<_>>();
        let mut shards = self.lock_shards(arrivals.iter().map(|(id, _)| *id));
        let mut candidates = BTreeMap::new();
        for (id, count) in arrivals {
            let mut entry = shards.entry(id).cloned().ok_or_else(|| {
                EngineError::from(SynchronizationError::BarrierUninitialized {
                    key: id.occurrence_key(),
                })
            })?;
            begin_next_generation(&mut entry);
            entry.expected_arrivals =
                entry.expected_arrivals.checked_sub(count).ok_or_else(|| {
                    EngineError::message("mbarrier.arrive_drop exceeds the expected arrival count")
                })?;
            // Keep required_arrivals() unchanged until the ordinary arrival.
            // This phase-only increment disappears at the next generation.
            entry.phase.pending_arrival_increments += count;
            candidates.insert(id, entry);
        }
        for (id, candidate) in candidates {
            shards.shard_mut(id).entries.insert(id, candidate);
        }
        Ok(())
    }

    fn arrive_impl(
        &self,
        id: PhysicalBarrierId,
        warp_id: usize,
        arrival_count: u64,
        expected_transactions: u64,
    ) -> Result<PhysicalMbarrierArrivalOutcome, SynchronizationError> {
        self.arrive_impl_with_outcome(
            id,
            warp_id,
            arrival_count,
            expected_transactions,
            |_| Ok(()),
        )
        .map_err(|error| match error {
            PhysicalArrivalCommitError::Barrier(error) => error,
            PhysicalArrivalCommitError::Publication(error) => {
                unreachable!("empty physical arrival publication failed: {error}")
            }
        })
    }

    pub(crate) fn arrive_with_outcome(
        &self,
        id: PhysicalBarrierId,
        warp_id: usize,
        arrival_count: u64,
        publish_before_wake: impl FnOnce(&PhysicalMbarrierArrivalOutcome) -> Result<(), EngineError>,
    ) -> Result<PhysicalMbarrierArrivalOutcome, PhysicalArrivalCommitError> {
        self.arrive_impl_with_outcome(id, warp_id, arrival_count, 0, publish_before_wake)
    }

    pub(crate) fn arrive_expect_tx_with_outcome(
        &self,
        id: PhysicalBarrierId,
        warp_id: usize,
        arrival_count: u64,
        expected_transactions: u64,
        publish_before_wake: impl FnOnce(&PhysicalMbarrierArrivalOutcome) -> Result<(), EngineError>,
    ) -> Result<PhysicalMbarrierArrivalOutcome, PhysicalArrivalCommitError> {
        self.arrive_impl_with_outcome(
            id,
            warp_id,
            arrival_count,
            expected_transactions,
            publish_before_wake,
        )
    }

    fn arrive_impl_with_outcome(
        &self,
        id: PhysicalBarrierId,
        warp_id: usize,
        arrival_count: u64,
        expected_transactions: u64,
        publish_before_wake: impl FnOnce(&PhysicalMbarrierArrivalOutcome) -> Result<(), EngineError>,
    ) -> Result<PhysicalMbarrierArrivalOutcome, PhysicalArrivalCommitError> {
        let _publication = self.lock_publication([id]);
        let (outcome, wakers) = {
            let mut shards = self.lock_shards([id]);
            apply_physical_arrival(
                shards.shard_mut(id),
                id,
                warp_id,
                arrival_count,
                expected_transactions,
            )?
        };
        publish_before_wake(&outcome).map_err(PhysicalArrivalCommitError::Publication)?;
        wake_all(wakers);
        Ok(outcome)
    }

    pub(crate) fn enqueue_transaction_completion(
        &self,
        id: PhysicalBarrierId,
        transactions: u64,
    ) -> Result<PhysicalCompletionActionId, SynchronizationError> {
        Ok(self
            .enqueue_transaction_completions(&[(id, transactions)])?
            .into_vec()
            .into_iter()
            .next()
            .expect("single completion enqueue returns one action ID"))
    }

    /// Apply issue-time numeric deliveries directly to barrier state.
    ///
    /// The numerical payload is already complete, so NumSim has no separate
    /// completion actor to schedule. A delivery for the next generation uses
    /// the same buffered-transaction state as the deferred path.
    pub(crate) fn complete_transactions_immediately(
        &self,
        completions: &[(PhysicalBarrierId, u64)],
    ) -> Result<(), SynchronizationError> {
        if completions.is_empty() {
            return Ok(());
        }
        let _publication = self.lock_publication(completions.iter().map(|&(id, _)| id));
        let wakers = {
            let mut shards = self.lock_shards(completions.iter().map(|&(id, _)| id));
            let mut planned = Vec::with_capacity(completions.len());
            for &(id, transactions) in completions {
                if transactions == 0 {
                    continue;
                }
                let entry =
                    shards
                        .entry(id)
                        .ok_or_else(|| SynchronizationError::BarrierUninitialized {
                            key: id.occurrence_key(),
                        })?;
                let generation = entry
                    .generation
                    .checked_add(u64::from(entry.phase.complete))
                    .ok_or_else(|| SynchronizationError::TransactionOverflow {
                        key: id.occurrence_key(),
                        phase: entry.generation & 1,
                        expected: u64::MAX,
                        completed: u64::MAX,
                    })?;
                planned.push((id, transactions, generation));
            }
            let mut candidates = BTreeMap::new();
            let mut all_wakers = Vec::new();
            for (id, transactions, generation) in planned {
                if !candidates.contains_key(&id) {
                    let entry = shards
                        .entry(id)
                        .cloned()
                        .expect("numeric completion target validated above");
                    candidates.insert(id, entry);
                }
                let entry = candidates
                    .get_mut(&id)
                    .expect("numeric completion candidate inserted above");
                all_wakers.extend(apply_physical_transaction_completion_entry(
                    id,
                    entry,
                    generation,
                    transactions,
                )?);
            }
            for (id, candidate) in candidates {
                shards.shard_mut(id).entries.insert(id, candidate);
            }
            all_wakers
        };
        wake_all(wakers);
        Ok(())
    }

    /// Validate and enqueue a complete multi-target delivery transaction.
    ///
    /// TMA multicast must not leave an arbitrary prefix of target completions
    /// queued when a later target is invalid. Action IDs are reserved only
    /// after every target and generation has validated under the stripe locks.
    pub(crate) fn enqueue_transaction_completions(
        &self,
        completions: &[(PhysicalBarrierId, u64)],
    ) -> Result<Box<[PhysicalCompletionActionId]>, SynchronizationError> {
        let mut shards = self.lock_shards(completions.iter().map(|&(id, _)| id));
        let validated = self.validate_transaction_completions(&shards, completions)?;
        Ok(self.queue_validated_transaction_completions(&mut shards, validated))
    }

    /// Queue transaction completions and let `register` record their owner
    /// before they become visible.
    ///
    /// `register` runs under the stripe locks with the actions that will be
    /// queued (zero-transaction entries reserve an ID but are absent, as in
    /// [`Self::queued_completion_actions`]); if it fails, nothing is queued.
    /// A deferred payload registers its token here, so the completion pump
    /// can never observe a queued action whose owner is not yet known, and
    /// the payload hub no longer has to hold its own lock across this call.
    pub(crate) fn enqueue_transaction_completions_with<R>(
        &self,
        completions: &[(PhysicalBarrierId, u64)],
        register: impl FnOnce(
            &[PhysicalCompletionActionId],
            &[PhysicalCompletionAction],
        ) -> Result<R, EngineError>,
    ) -> Result<(Box<[PhysicalCompletionActionId]>, R), EngineError> {
        let mut shards = self.lock_shards(completions.iter().map(|&(id, _)| id));
        let validated = self.validate_transaction_completions(&shards, completions)?;
        let action_ids = validated
            .iter()
            .map(|(action_id, _)| *action_id)
            .collect::<Vec<_>>();
        let actions = validated
            .iter()
            .filter_map(|(_, action)| *action)
            .collect::<Vec<_>>();
        let registered = register(&action_ids, &actions)?;
        let action_ids = self.queue_validated_transaction_completions(&mut shards, validated);
        Ok((action_ids, registered))
    }

    /// Validate `completions` against the current phases and allocate their
    /// action IDs, without queueing anything.
    #[allow(clippy::type_complexity)]
    fn validate_transaction_completions(
        &self,
        shards: &LockedShards<'_>,
        completions: &[(PhysicalBarrierId, u64)],
    ) -> Result<
        Vec<(PhysicalCompletionActionId, Option<PhysicalCompletionAction>)>,
        SynchronizationError,
    > {
        let action_count = u64::try_from(completions.len()).map_err(|_| {
            SynchronizationError::CompletionSourceOperationFailed {
                source_name: "physical-mbarrier",
                details: "completion action batch is too large".to_string(),
            }
        })?;
        let mut planned = Vec::with_capacity(completions.len());
        for &(id, transactions) in completions {
            if transactions == 0 {
                planned.push(None);
                continue;
            }
            let entry =
                shards
                    .entry(id)
                    .ok_or_else(|| SynchronizationError::BarrierUninitialized {
                        key: id.occurrence_key(),
                    })?;
            let generation = entry
                .generation
                .checked_add(u64::from(entry.phase.complete))
                .ok_or_else(|| SynchronizationError::TransactionOverflow {
                    key: id.occurrence_key(),
                    phase: entry.generation & 1,
                    expected: u64::MAX,
                    completed: u64::MAX,
                })?;
            planned.push(Some((id, transactions, generation)));
        }
        let first_action_id = self.allocate_action_ids(action_count)?;
        Ok(planned
            .into_iter()
            .enumerate()
            .map(|(index, planned)| {
                let index =
                    u64::try_from(index).expect("validated completion batch length fits u64");
                let action_id = first_action_id
                    .checked_add(index)
                    .and_then(PhysicalCompletionActionId::from_ordinal)
                    .expect("batch ID bound validated above");
                let action =
                    planned.map(|(id, transactions, generation)| PhysicalCompletionAction {
                        action_id,
                        barrier_id: id,
                        generation,
                        kind: PhysicalCompletionKind::Transaction { transactions },
                    });
                (action_id, action)
            })
            .collect())
    }

    fn queue_validated_transaction_completions(
        &self,
        shards: &mut LockedShards<'_>,
        validated: Vec<(PhysicalCompletionActionId, Option<PhysicalCompletionAction>)>,
    ) -> Box<[PhysicalCompletionActionId]> {
        self.index_actions(validated.iter().filter_map(|(_, action)| action.as_ref()));
        let mut action_ids = Vec::with_capacity(validated.len());
        for (action_id, action) in validated {
            action_ids.push(action_id);
            if let Some(action) = action {
                shards
                    .shard_mut(action.barrier_id())
                    .transaction_completions
                    .insert(action.id(), action);
            }
        }
        action_ids.into_boxed_slice()
    }

    pub(crate) fn enqueue_arrival_completion(
        &self,
        id: PhysicalBarrierId,
        warp_id: usize,
        arrival_count: u64,
    ) -> Result<PhysicalCompletionAction, SynchronizationError> {
        Ok(self
            .enqueue_arrival_completions(&[(id, warp_id, arrival_count)])?
            .into_vec()
            .into_iter()
            .next()
            .expect("single deferred arrival enqueue returns one action"))
    }

    /// Validate and enqueue one logical multi-target deferred arrival.
    ///
    /// `tcgen05.commit` may signal the same barrier offset in multiple CTAs.
    /// Resolve those physical identities before calling this method; the hub
    /// then validates every target and reserves every action ID under one lock
    /// so a failed multicast cannot publish an arbitrary prefix.
    pub(crate) fn enqueue_arrival_completions(
        &self,
        completions: &[(PhysicalBarrierId, usize, u64)],
    ) -> Result<Box<[PhysicalCompletionAction]>, SynchronizationError> {
        let mut shards = self.lock_shards(completions.iter().map(|&(id, _, _)| id));
        let action_count = u64::try_from(completions.len()).map_err(|_| {
            SynchronizationError::CompletionSourceOperationFailed {
                source_name: "physical-mbarrier",
                details: "deferred arrival batch is too large".to_string(),
            }
        })?;
        let mut planned = Vec::with_capacity(completions.len());
        for &(id, warp_id, arrival_count) in completions {
            if arrival_count == 0 {
                return Err(SynchronizationError::InvalidBarrierArrivalCount {
                    key: id.occurrence_key(),
                    count: arrival_count,
                });
            }
            let entry =
                shards
                    .entry(id)
                    .ok_or_else(|| SynchronizationError::BarrierUninitialized {
                        key: id.occurrence_key(),
                    })?;
            let generation = entry
                .generation
                .checked_add(u64::from(entry.phase.complete))
                .ok_or_else(|| SynchronizationError::CompletionSourceOperationFailed {
                    source_name: "physical-mbarrier",
                    details: "deferred arrival generation overflow".to_string(),
                })?;
            planned.push((id, warp_id, arrival_count, generation));
        }
        let first_action_id = self.allocate_action_ids(action_count)?;
        let actions = planned
            .into_iter()
            .enumerate()
            .map(|(index, (id, warp_id, arrival_count, generation))| {
                let index =
                    u64::try_from(index).expect("validated deferred arrival batch fits u64");
                let action_id = first_action_id
                    .checked_add(index)
                    .and_then(PhysicalCompletionActionId::from_ordinal)
                    .expect("deferred arrival action-ID bound validated above");
                PhysicalCompletionAction {
                    action_id,
                    barrier_id: id,
                    generation,
                    kind: PhysicalCompletionKind::Arrival {
                        warp_id,
                        arrival_count,
                    },
                }
            })
            .collect::<Vec<_>>();
        self.index_actions(actions.iter());
        for action in &actions {
            shards
                .shard_mut(action.barrier_id())
                .transaction_completions
                .insert(action.id(), *action);
        }
        Ok(actions.into_boxed_slice())
    }

    pub(crate) fn wait(
        self: &Arc<Self>,
        id: PhysicalBarrierId,
        requested_phase: u64,
        warp_id: usize,
    ) -> Result<PhysicalBarrierWait, SynchronizationError> {
        self.wait_with_operation(id, requested_phase, warp_id, None)
    }

    pub(crate) fn wait_with_operation(
        self: &Arc<Self>,
        id: PhysicalBarrierId,
        requested_phase: u64,
        warp_id: usize,
        operation: Option<DynamicOpId>,
    ) -> Result<PhysicalBarrierWait, SynchronizationError> {
        let key = id.occurrence_key();
        if requested_phase > 1 {
            return Err(SynchronizationError::InvalidBarrierPhase {
                key,
                phase: requested_phase,
            });
        }
        Ok(PhysicalBarrierWait {
            hub: Arc::clone(self),
            id,
            requested_phase,
            warp_id,
            operation,
            registered: false,
            finished: false,
        })
    }

    pub(crate) fn test_wait(
        &self,
        id: PhysicalBarrierId,
        requested_phase: u64,
    ) -> Result<bool, SynchronizationError> {
        self.test_wait_detailed(id, requested_phase)
            .map(|(ready, _)| ready)
    }

    /// Snapshot one parity predicate together with the generation whose
    /// completion made it true. Acquire-qualified nonblocking waits use the
    /// generation to publish the same causal edge as a blocking wait without
    /// changing the numeric readiness result.
    pub(crate) fn test_wait_detailed(
        &self,
        id: PhysicalBarrierId,
        requested_phase: u64,
    ) -> Result<(bool, Option<u64>), SynchronizationError> {
        let (ready, generation, _) =
            self.query_many(&[(id, requested_phase)], MbarrierQuery::PrimaryParity)?[0];
        Ok((ready, generation))
    }

    pub(crate) fn test_wait_many(
        &self,
        requests: &[(PhysicalBarrierId, u64)],
    ) -> Result<Vec<bool>, SynchronizationError> {
        Ok(self
            .query_many(requests, MbarrierQuery::PrimaryParity)?
            .into_iter()
            .map(|(ready, _, _)| ready)
            .collect())
    }

    /// Readiness, the acquire generation and its report are one snapshot.
    /// Never fetch a report after dropping the lock: another warp may already
    /// have advanced the barrier to a phase with a different payload.
    pub(crate) fn query_many(
        &self,
        requests: &[(PhysicalBarrierId, u64)],
        kind: MbarrierQuery,
    ) -> Result<Vec<(bool, Option<u64>, bool)>, SynchronizationError> {
        let _publication = self.lock_publication(requests.iter().map(|&(id, _)| id));
        self.query_many_locked(requests, kind)
    }

    /// Keep the observed phase's release payload reachable until its checker
    /// acquire is published. Numeric shard locks are dropped before analysis.
    pub(crate) fn query_many_with<R>(
        &self,
        requests: &[(PhysicalBarrierId, u64)],
        kind: MbarrierQuery,
        publish: impl FnOnce(Vec<(bool, Option<u64>, bool)>) -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        let _publication = self.lock_publication(requests.iter().map(|&(id, _)| id));
        publish(self.query_many_locked(requests, kind)?)
    }

    fn query_many_locked(
        &self,
        requests: &[(PhysicalBarrierId, u64)],
        kind: MbarrierQuery,
    ) -> Result<Vec<(bool, Option<u64>, bool)>, SynchronizationError> {
        if kind != MbarrierQuery::PrimaryState {
            for &(id, phase) in requests {
                if phase > 1 {
                    return Err(SynchronizationError::InvalidBarrierPhase {
                        key: id.occurrence_key(),
                        phase,
                    });
                }
            }
        }
        let shards = self.lock_shards(requests.iter().map(|&(id, _)| id));
        requests
            .iter()
            .map(|&(id, requested)| {
                let key = id.occurrence_key();
                let entry =
                    shards
                        .entry(id)
                        .ok_or_else(|| SynchronizationError::BarrierUninitialized {
                            key: key.clone(),
                        })?;
                let (ready, generation) = if kind == MbarrierQuery::ConditionalParity {
                    let ready = entry.conditional_completed_parity == requested;
                    (
                        ready,
                        ready
                            .then_some(entry.conditional_completed_generation)
                            .flatten(),
                    )
                } else if kind == MbarrierQuery::PrimaryParity {
                    let ready = entry.completed_parity == requested;
                    (
                        ready,
                        ready.then_some(entry.last_completed_generation).flatten(),
                    )
                } else if requested == entry.generation {
                    (
                        entry.phase.complete,
                        entry.phase.complete.then_some(requested),
                    )
                } else if requested.checked_add(1) == Some(entry.generation) {
                    (true, Some(requested))
                } else {
                    return Err(SynchronizationError::InvalidMbarrierStateToken {
                        key,
                        token_generation: requested,
                        current_generation: entry.generation,
                    });
                };
                let report = kind != MbarrierQuery::ConditionalParity
                    && ready
                    && generation.is_some_and(|generation| {
                        if generation == entry.generation {
                            entry.phase.report
                        } else {
                            entry.previous_report
                        }
                    });
                Ok((ready, generation, report))
            })
            .collect()
    }

    pub(crate) fn pending_completion_count(&self) -> usize {
        let mut count = 0;
        self.for_each_shard(|shard| count += shard.transaction_completions.len());
        count
    }

    /// Return currently applicable actions in stable queue order.
    ///
    /// Stripes are visited one at a time; the completion pump re-validates
    /// enablement when it applies an action, so the snapshot need not be
    /// atomic across barriers.
    pub(crate) fn pending_completion_actions(&self) -> Vec<PhysicalCompletionAction> {
        let mut actions = Vec::new();
        self.for_each_shard(|shard| {
            actions.extend(
                shard
                    .transaction_completions
                    .values()
                    .filter(|action| {
                        shard
                            .entries
                            .get(&action.barrier_id)
                            .is_some_and(|entry| physical_completion_is_enabled(entry, **action))
                    })
                    .copied(),
            );
        });
        actions.sort_unstable_by_key(|action| action.id());
        actions
    }

    /// Return queued actions for a previously allocated action-ID batch.
    ///
    /// Zero-transaction entries reserve an ID but do not create a queue entry,
    /// so they are intentionally absent from the returned list.
    pub(crate) fn queued_completion_actions(
        &self,
        action_ids: &[PhysicalCompletionActionId],
    ) -> Vec<PhysicalCompletionAction> {
        let indexed = self.indexed_actions(action_ids);
        let shards = self.lock_shards(indexed.iter().map(|&(_, barrier_id)| barrier_id));
        let mut actions = indexed
            .iter()
            .filter_map(|&(action_id, barrier_id)| {
                shards
                    .shard(barrier_id)
                    .transaction_completions
                    .get(&action_id)
                    .copied()
            })
            .collect::<Vec<_>>();
        actions.sort_unstable_by_key(|action| action.id());
        actions
    }

    /// Remove a batch that has not been published to the completion pump.
    ///
    /// This is rollback for the engine-private immediate `mbarrier.complete_tx`
    /// path when a mode rejects the resolved effect before numeric commit.  A
    /// published asynchronous payload must never use this method.
    pub(crate) fn discard_unpublished_transaction_completions(
        &self,
        action_ids: &[PhysicalCompletionActionId],
    ) {
        let indexed = self.indexed_actions(action_ids);
        {
            let mut shards = self.lock_shards(indexed.iter().map(|&(_, barrier_id)| barrier_id));
            for &(action_id, barrier_id) in &indexed {
                shards
                    .shard_mut(barrier_id)
                    .transaction_completions
                    .remove(&action_id);
            }
        }
        self.unindex_actions(indexed.iter().map(|&(action_id, _)| action_id));
    }

    /// Apply exactly one currently enabled completion action.
    pub(crate) fn apply_completion(
        &self,
        action_id: PhysicalCompletionActionId,
    ) -> Result<CompletionProgress, PhysicalCompletionActionError> {
        Ok(self.apply_completion_detailed(action_id)?.progress())
    }

    /// Apply one action and retain exact generation/waiter provenance for
    /// strict analysis modes.
    pub(crate) fn apply_completion_detailed(
        &self,
        action_id: PhysicalCompletionActionId,
    ) -> Result<PhysicalCompletionOutcome, PhysicalCompletionActionError> {
        self.apply_completion_detailed_with_outcome(action_id, |_| Ok(()))
            .map_err(|error| match error {
                PhysicalCompletionBatchCommitError::Barrier(error) => error,
                PhysicalCompletionBatchCommitError::Mutation(error) => {
                    unreachable!("single physical completion has no mutation: {error}")
                }
                PhysicalCompletionBatchCommitError::Publication(error) => {
                    unreachable!("empty physical completion publication failed: {error}")
                }
            })
    }

    /// Apply one completion while giving analysis the exact committed outcome
    /// before any numeric waiter is woken.
    pub(crate) fn apply_completion_detailed_with_outcome(
        &self,
        action_id: PhysicalCompletionActionId,
        publish_before_wake: impl FnOnce(&PhysicalCompletionOutcome) -> Result<(), EngineError>,
    ) -> Result<PhysicalCompletionOutcome, PhysicalCompletionBatchCommitError> {
        let barrier_ids = self.completion_barrier_ids(&[action_id]);
        let _publication = self.lock_publication(barrier_ids.iter().copied());
        let (outcome, wakers) = {
            let mut shards = self.lock_shards(barrier_ids.iter().copied());
            let Some(&barrier_id) = barrier_ids.first() else {
                return Err(PhysicalCompletionBatchCommitError::Barrier(
                    PhysicalCompletionActionError::Missing { action_id },
                ));
            };
            let state = shards.shard_mut(barrier_id);
            let action = state
                .transaction_completions
                .get(&action_id)
                .copied()
                .ok_or(PhysicalCompletionActionError::Missing { action_id })
                .map_err(PhysicalCompletionBatchCommitError::Barrier)?;
            let (previous_generation, was_complete, waiting_before) = state
                .entries
                .get(&action.barrier_id())
                .map(|entry| {
                    (
                        entry.generation,
                        entry.phase.complete,
                        entry.waiters.keys().copied().collect::<BTreeSet<_>>(),
                    )
                })
                .unwrap_or((0, false, BTreeSet::new()));
            let (progress, wakers) = apply_physical_completion(state, action_id)
                .map_err(PhysicalCompletionBatchCommitError::Barrier)?;
            let entry = state
                .entries
                .get(&action.barrier_id())
                .expect("applied physical completion retains its barrier entry");
            let waiting_after = entry.waiters.keys().copied().collect::<BTreeSet<_>>();
            let woken_warp_ids = waiting_before
                .difference(&waiting_after)
                .copied()
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let completed_generation = (entry.phase.complete
                && (!was_complete || entry.generation != previous_generation))
                .then_some(entry.generation);
            let arrival_outcome = action.arrival().map(|_| {
                PhysicalMbarrierArrivalOutcome::new(
                    action.generation(),
                    completed_generation == Some(action.generation()),
                )
                .with_conditional_generation(entry.conditional_completed_generation)
            });
            (
                PhysicalCompletionOutcome {
                    action,
                    progress,
                    completed_generation,
                    conditional_completed_generation: entry.conditional_completed_generation,
                    arrival_outcome,
                    woken_warp_ids,
                },
                wakers,
            )
        };
        self.unindex_actions([action_id]);
        publish_before_wake(&outcome).map_err(PhysicalCompletionBatchCommitError::Publication)?;
        wake_all(wakers);
        Ok(outcome)
    }

    /// Commit a completion batch, publish its exact analysis outcome, then
    /// wake numeric waiters.  The callback runs after the live numeric state is
    /// installed and before any waiter can resume on another executor worker.
    pub(crate) fn apply_completion_batch_detailed_with_outcomes(
        &self,
        action_ids: &[PhysicalCompletionActionId],
        commit_mutation: impl FnOnce() -> Result<(), EngineError>,
        publish_before_wake: impl FnOnce(&[PhysicalCompletionOutcome]) -> Result<(), EngineError>,
    ) -> Result<Box<[PhysicalCompletionOutcome]>, PhysicalCompletionBatchCommitError> {
        let barrier_ids = self.completion_barrier_ids(action_ids);
        let _publication = { self.lock_publication(barrier_ids.iter().copied()) };
        if let [action_id] = action_ids {
            // A normal, non-multicast payload has exactly one physical target.
            // Stage only that barrier entry: cloning the complete hub here
            // copies every CTA's waiter and generation trees on the serialized
            // completion-pump path.
            let (outcomes, wakers) = {
                let mut shards = { self.lock_shards(barrier_ids.iter().copied()) };
                let Some(&barrier_id) = barrier_ids.first() else {
                    return Err(PhysicalCompletionBatchCommitError::Barrier(
                        PhysicalCompletionActionError::Missing {
                            action_id: *action_id,
                        },
                    ));
                };
                let state = shards.shard_mut(barrier_id);
                let action = state
                    .transaction_completions
                    .get(action_id)
                    .copied()
                    .ok_or(PhysicalCompletionActionError::Missing {
                        action_id: *action_id,
                    })
                    .map_err(PhysicalCompletionBatchCommitError::Barrier)?;
                let entry = state
                    .entries
                    .get(&action.barrier_id())
                    .cloned()
                    .ok_or_else(|| PhysicalCompletionActionError::ApplyFailed {
                        action,
                        source: SynchronizationError::BarrierUninitialized {
                            key: action.barrier_id().occurrence_key(),
                        },
                    })
                    .map_err(PhysicalCompletionBatchCommitError::Barrier)?;
                let previous_generation = entry.generation;
                let was_complete = entry.phase.complete;
                let waiting_before = entry.waiters.keys().copied().collect::<BTreeSet<_>>();
                let mut staged = PhysicalBarrierState {
                    entries: BTreeMap::from([(action.barrier_id(), entry)]),
                    transaction_completions: BTreeMap::from([(action.id(), action)]),
                };
                let (progress, wakers) = apply_physical_completion(&mut staged, *action_id)
                    .map_err(PhysicalCompletionBatchCommitError::Barrier)?;
                let entry = staged
                    .entries
                    .remove(&action.barrier_id())
                    .expect("staged physical completion retains its barrier entry");
                let waiting_after = entry.waiters.keys().copied().collect::<BTreeSet<_>>();
                let woken_warp_ids = waiting_before
                    .difference(&waiting_after)
                    .copied()
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let completed_generation = (entry.phase.complete
                    && (!was_complete || entry.generation != previous_generation))
                    .then_some(entry.generation);
                let arrival_outcome = action.arrival().map(|_| {
                    PhysicalMbarrierArrivalOutcome::new(
                        action.generation(),
                        completed_generation == Some(action.generation()),
                    )
                    .with_conditional_generation(entry.conditional_completed_generation)
                    .with_conditional_generation(entry.conditional_completed_generation)
                });
                let outcome = PhysicalCompletionOutcome {
                    action,
                    progress,
                    completed_generation,
                    conditional_completed_generation: entry.conditional_completed_generation,
                    arrival_outcome,
                    woken_warp_ids,
                };

                commit_mutation().map_err(PhysicalCompletionBatchCommitError::Mutation)?;
                state.entries.insert(action.barrier_id(), entry);
                let removed = state
                    .transaction_completions
                    .remove(action_id)
                    .expect("validated physical completion remains queued while locked");
                debug_assert_eq!(removed, action);
                (vec![outcome].into_boxed_slice(), wakers)
            };
            self.unindex_actions([*action_id]);
            publish_before_wake(&outcomes)
                .map_err(PhysicalCompletionBatchCommitError::Publication)?;
            wake_all(wakers);
            return Ok(outcomes);
        }
        let (outcomes, wakers) = {
            let mut shards = { self.lock_shards(barrier_ids.iter().copied()) };
            let mut staged = shards.staged();
            let mut outcomes = Vec::with_capacity(action_ids.len());
            let mut all_wakers = Vec::new();
            for &action_id in action_ids {
                let action = staged
                    .transaction_completions
                    .get(&action_id)
                    .copied()
                    .ok_or(PhysicalCompletionActionError::Missing { action_id })
                    .map_err(PhysicalCompletionBatchCommitError::Barrier)?;
                let (previous_generation, was_complete, waiting_before) = staged
                    .entries
                    .get(&action.barrier_id())
                    .map(|entry| {
                        (
                            entry.generation,
                            entry.phase.complete,
                            entry.waiters.keys().copied().collect::<BTreeSet<_>>(),
                        )
                    })
                    .unwrap_or((0, false, BTreeSet::new()));
                let (progress, mut wakers) = apply_physical_completion(&mut staged, action_id)
                    .map_err(PhysicalCompletionBatchCommitError::Barrier)?;
                let entry = staged
                    .entries
                    .get(&action.barrier_id())
                    .expect("applied physical completion retains its barrier entry");
                let waiting_after = entry.waiters.keys().copied().collect::<BTreeSet<_>>();
                let woken_warp_ids = waiting_before
                    .difference(&waiting_after)
                    .copied()
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let completed_generation = (entry.phase.complete
                    && (!was_complete || entry.generation != previous_generation))
                    .then_some(entry.generation);
                let arrival_outcome = action.arrival().map(|_| {
                    PhysicalMbarrierArrivalOutcome::new(
                        action.generation(),
                        completed_generation == Some(action.generation()),
                    )
                });
                outcomes.push(PhysicalCompletionOutcome {
                    action,
                    progress,
                    completed_generation,
                    conditional_completed_generation: entry.conditional_completed_generation,
                    arrival_outcome,
                    woken_warp_ids,
                });
                all_wakers.append(&mut wakers);
            }
            commit_mutation().map_err(PhysicalCompletionBatchCommitError::Mutation)?;
            shards.install(staged);
            (outcomes.into_boxed_slice(), all_wakers)
        };
        self.unindex_actions(action_ids.iter().copied());
        publish_before_wake(&outcomes).map_err(PhysicalCompletionBatchCommitError::Publication)?;
        wake_all(wakers);
        Ok(outcomes)
    }
}

fn apply_physical_arrival(
    state: &mut PhysicalBarrierState,
    id: PhysicalBarrierId,
    warp_id: usize,
    arrival_count: u64,
    expected_transactions: u64,
) -> Result<(PhysicalMbarrierArrivalOutcome, Vec<Waker>), SynchronizationError> {
    let key = id.occurrence_key();
    let mut candidate = state
        .entries
        .get(&id)
        .cloned()
        .ok_or_else(|| SynchronizationError::BarrierUninitialized { key: key.clone() })?;
    let (outcome, wakers) = apply_physical_arrival_entry(
        id,
        &mut candidate,
        warp_id,
        arrival_count,
        expected_transactions,
    )?;
    state.entries.insert(id, candidate);
    Ok((outcome, wakers))
}

fn apply_physical_arrival_entry(
    id: PhysicalBarrierId,
    candidate: &mut PhysicalBarrierEntry,
    warp_id: usize,
    arrival_count: u64,
    expected_transactions: u64,
) -> Result<(PhysicalMbarrierArrivalOutcome, Vec<Waker>), SynchronizationError> {
    let key = id.occurrence_key();
    if arrival_count == 0 {
        return Ok((
            PhysicalMbarrierArrivalOutcome::new(candidate.generation, false)
                .with_conditional_generation(candidate.conditional_completed_generation),
            Vec::new(),
        ));
    }
    begin_next_generation(candidate);
    let phase = candidate.generation & 1;
    let required_arrivals = candidate.required_arrivals();
    let pending_arrivals_before = required_arrivals - candidate.phase.arrival_count;
    let completed = candidate
        .phase
        .arrival_count
        .checked_add(arrival_count)
        .ok_or_else(|| SynchronizationError::BarrierArrivalOverflow {
            key: key.clone(),
            phase,
            expected: required_arrivals,
            completed: u64::MAX,
        })?;
    if completed > required_arrivals {
        return Err(SynchronizationError::BarrierArrivalOverflow {
            key,
            phase,
            expected: required_arrivals,
            completed,
        });
    }
    let transactions = candidate
        .phase
        .expected_transactions
        .checked_add(expected_transactions)
        .ok_or_else(|| SynchronizationError::TransactionOverflow {
            key: key.clone(),
            phase,
            expected: u64::MAX,
            completed: u64::MAX,
        })?;
    candidate.phase.arrival_count = completed;
    candidate.phase.arrived_warps.insert(warp_id);
    candidate.phase.expected_transactions = transactions;
    if candidate.phase.arrival_count == required_arrivals
        && candidate.phase.completed_transactions > candidate.phase.expected_transactions
    {
        return Err(SynchronizationError::TransactionOverflow {
            key,
            phase,
            expected: candidate.phase.expected_transactions,
            completed: candidate.phase.completed_transactions,
        });
    }
    let wakers = complete_physical_if_ready(candidate);
    let mut outcome =
        PhysicalMbarrierArrivalOutcome::new(candidate.generation, candidate.phase.complete)
            .with_conditional_generation(candidate.conditional_completed_generation);
    outcome.pending_arrivals_before = pending_arrivals_before;
    Ok((outcome, wakers))
}

/// Raise the current generation's pending arrival count.
///
/// PTX `cp.async.mbarrier.arrive` (without `.noinc`) increments the pending
/// count of the phase the issuing thread observes, then discharges it from the
/// arrive-on it defers to its prior `cp.async` work. The raise therefore lives
/// on the phase, not on the entry: the next generation starts from the
/// barrier's `mbarrier.init` expectation again.
fn apply_physical_pending_arrival_increase_entry(
    id: PhysicalBarrierId,
    candidate: &mut PhysicalBarrierEntry,
    increase: u64,
) -> Result<(), SynchronizationError> {
    if increase == 0 {
        return Ok(());
    }
    begin_next_generation(candidate);
    let phase = candidate.generation & 1;
    let required = candidate
        .required_arrivals()
        .checked_add(increase)
        .filter(|required| *required <= mbarrier_arrival_limit(candidate.layout_v1))
        .ok_or_else(|| SynchronizationError::BarrierArrivalOverflow {
            key: id.occurrence_key(),
            phase,
            expected: mbarrier_arrival_limit(candidate.layout_v1),
            completed: candidate.required_arrivals().saturating_add(increase),
        })?;
    candidate.phase.pending_arrival_increments = required - candidate.expected_arrivals;
    Ok(())
}

fn apply_physical_expect_tx_entry(
    id: PhysicalBarrierId,
    candidate: &mut PhysicalBarrierEntry,
    expected_transactions: u64,
) -> Result<(), SynchronizationError> {
    let key = id.occurrence_key();
    if expected_transactions > MAX_MBARRIER_TRANSACTIONS {
        return Err(SynchronizationError::TransactionOverflow {
            key,
            phase: candidate.generation & 1,
            expected: MAX_MBARRIER_TRANSACTIONS,
            completed: expected_transactions,
        });
    }
    begin_next_generation(candidate);
    let phase = candidate.generation & 1;
    let total = candidate
        .phase
        .expected_transactions
        .checked_add(expected_transactions)
        .ok_or_else(|| SynchronizationError::TransactionOverflow {
            key: key.clone(),
            phase,
            expected: MAX_MBARRIER_TRANSACTIONS,
            completed: u64::MAX,
        })?;
    if total > MAX_MBARRIER_TRANSACTIONS {
        return Err(SynchronizationError::TransactionOverflow {
            key,
            phase,
            expected: MAX_MBARRIER_TRANSACTIONS,
            completed: total,
        });
    }
    candidate.phase.expected_transactions = total;
    Ok(())
}

#[derive(Debug)]
pub enum PhysicalArrivalCommitError {
    Barrier(SynchronizationError),
    Publication(EngineError),
}

impl From<SynchronizationError> for PhysicalArrivalCommitError {
    fn from(error: SynchronizationError) -> Self {
        Self::Barrier(error)
    }
}

impl fmt::Display for PhysicalArrivalCommitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Barrier(error) => error.fmt(f),
            Self::Publication(error) => write!(f, "mbarrier arrival publication failed: {error}"),
        }
    }
}

impl Error for PhysicalArrivalCommitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Barrier(error) => Some(error),
            Self::Publication(error) => Some(error),
        }
    }
}

#[derive(Debug)]
pub enum PhysicalCompletionBatchCommitError {
    Barrier(PhysicalCompletionActionError),
    Mutation(EngineError),
    Publication(EngineError),
}

impl fmt::Display for PhysicalCompletionBatchCommitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Barrier(error) => error.fmt(f),
            Self::Mutation(error) => write!(f, "deferred payload mutation failed: {error}"),
            Self::Publication(error) => write!(f, "completion publication failed: {error}"),
        }
    }
}

impl Error for PhysicalCompletionBatchCommitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Barrier(error) => Some(error),
            Self::Mutation(error) | Self::Publication(error) => Some(error),
        }
    }
}

impl CompletionSource for PhysicalBarrierHub {
    fn source_name(&self) -> &'static str {
        "physical-mbarrier"
    }

    fn pump(&self) -> Result<CompletionProgress, SynchronizationError> {
        let actions = self.pending_completion_actions();
        let mut progress = CompletionProgress::default();
        for action in actions {
            let action_progress = self
                .apply_completion(action.id())
                .map_err(PhysicalCompletionActionError::into_synchronization_error)?;
            progress.completed_operations += action_progress.completed_operations;
            progress.woken_warps += action_progress.woken_warps;
        }
        Ok(progress)
    }

    fn blocked_operations(&self) -> Vec<BlockedOperation> {
        let mut blocked = Vec::new();
        self.for_each_shard(|shard| {
            for (id, entry) in &shard.entries {
            let participant_state = ParticipantState::counted(
                entry.phase.arrived_warps.iter().copied(),
                entry.required_arrivals(),
                entry.phase.arrival_count,
                Some(entry.phase.expected_transactions),
                Some(entry.phase.completed_transactions),
            );
            for (&warp_id, waiter) in &entry.waiters {
                blocked.push(
                    BlockedOperation::new(
                        warp_id,
                        crate::AwaitedOperation::PhysicalMbarrierTryWait,
                        id.occurrence_key(),
                        Some(waiter.requested_phase),
                        participant_state.clone(),
                    )
                    .with_operation(waiter.operation.clone()),
                );
            }
            }
        });
        blocked
    }

    fn validate_quiescent(&self) -> Result<(), SynchronizationError> {
        let queued = self.pending_completion_count();
        if queued != 0 {
            return Err(SynchronizationError::CompletionSourceNotQuiescent {
                source_name: self.source_name(),
                details: format!("{queued} transaction completions remain queued"),
            });
        }
        let mut result = Ok(());
        self.for_each_shard(|shard| {
            if result.is_err() {
                return;
            }
            for (id, entry) in &shard.entries {
                let transactions_are_unresolved =
                    entry.phase.completed_transactions != entry.phase.expected_transactions;
                // A producer may reserve the next pipeline phase immediately before
                // kernel exit. Once every arrival has been issued, no waiter remains,
                // and no payload completion is queued, that reservation has no
                // observable consumer and dies with the CTA-local shared-memory
                // lifetime. A real under-delivery remains blocking whenever a waiter
                // is present (and queued/buffered completions are never ignored).
                let terminal_reservation = transactions_are_unresolved
                    && entry.phase.arrival_count >= entry.required_arrivals()
                    && entry.waiters.is_empty()
                    && entry.buffered_transactions.is_empty();
                if !entry.waiters.is_empty()
                    || (transactions_are_unresolved && !terminal_reservation)
                    || !entry.buffered_transactions.is_empty()
                {
                    result = Err(id.occurrence_key().completion_not_quiescent_error(
                        self.source_name(),
                        format_args!(
                            " generation {} is incomplete: arrivals={}/{}, transactions={}/{}, buffered_transactions={:?}, waiting_warps={:?}",
                            entry.generation,
                            entry.phase.arrival_count,
                            entry.required_arrivals(),
                            entry.phase.completed_transactions,
                            entry.phase.expected_transactions,
                            entry.buffered_transactions,
                            entry.waiters.keys().copied().collect::<Vec<_>>()
                        ),
                    ));
                    return;
                }
            }
        });
        result
    }
}

pub struct PhysicalBarrierWait {
    hub: Arc<PhysicalBarrierHub>,
    id: PhysicalBarrierId,
    requested_phase: u64,
    warp_id: usize,
    operation: Option<DynamicOpId>,
    registered: bool,
    finished: bool,
}

impl Future for PhysicalBarrierWait {
    type Output = Result<Option<u64>, SynchronizationError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let key = this.id.occurrence_key();
        let _publication = this.hub.lock_publication([this.id]);
        let mut shards = this.hub.lock_shards([this.id]);
        let entry = match shards.entry_mut(this.id) {
            Some(entry) => entry,
            None => {
                return Poll::Ready(Err(SynchronizationError::BarrierUninitialized { key }));
            }
        };
        if entry.completed_parity == this.requested_phase {
            entry.waiters.remove(&this.warp_id);
            this.registered = false;
            this.finished = true;
            return Poll::Ready(Ok(entry.last_completed_generation));
        }
        match entry.waiters.get_mut(&this.warp_id) {
            Some(waiter) if this.registered => {
                if waiter.requested_phase != this.requested_phase {
                    return Poll::Ready(Err(SynchronizationError::DuplicateWaiter {
                        key,
                        phase: Some(this.requested_phase),
                        warp_id: this.warp_id,
                    }));
                }
                if !waiter.waker.will_wake(context.waker()) {
                    waiter.waker = context.waker().clone();
                }
            }
            Some(_) => {
                return Poll::Ready(Err(SynchronizationError::DuplicateWaiter {
                    key,
                    phase: Some(this.requested_phase),
                    warp_id: this.warp_id,
                }));
            }
            None => {
                entry.waiters.insert(
                    this.warp_id,
                    PhysicalWaiter {
                        requested_phase: this.requested_phase,
                        operation: this.operation.clone(),
                        waker: context.waker().clone(),
                    },
                );
                this.registered = true;
            }
        }
        Poll::Pending
    }
}

impl Drop for PhysicalBarrierWait {
    fn drop(&mut self) {
        if !self.registered || self.finished {
            return;
        }
        let mut shards = self.hub.lock_shards([self.id]);
        if let Some(entry) = shards.entry_mut(self.id) {
            entry.waiters.remove(&self.warp_id);
        }
    }
}

fn apply_physical_completion(
    state: &mut PhysicalBarrierState,
    action_id: PhysicalCompletionActionId,
) -> Result<(CompletionProgress, Vec<Waker>), PhysicalCompletionActionError> {
    let Some(action) = state.transaction_completions.get(&action_id).copied() else {
        return Err(PhysicalCompletionActionError::Missing { action_id });
    };
    let Some(entry) = state.entries.get(&action.barrier_id()) else {
        return Err(PhysicalCompletionActionError::ApplyFailed {
            action,
            source: SynchronizationError::BarrierUninitialized {
                key: action.barrier_id().occurrence_key(),
            },
        });
    };
    if !physical_completion_is_enabled(entry, action) {
        return Err(PhysicalCompletionActionError::NotEnabled {
            action,
            current_generation: entry.generation,
            current_generation_complete: entry.phase.complete,
        });
    }

    let key = action.barrier_id().occurrence_key();
    let wakers = match action.kind() {
        PhysicalCompletionKind::Transaction { transactions } => {
            let entry = state
                .entries
                .get_mut(&action.barrier_id())
                .expect("validated physical barrier entry remains present while locked");
            apply_physical_transaction_completion_entry(
                action.barrier_id(),
                entry,
                action.generation(),
                transactions,
            )
            .map_err(|source| PhysicalCompletionActionError::ApplyFailed { action, source })?
        }
        PhysicalCompletionKind::Arrival {
            warp_id,
            arrival_count,
        } => {
            let entry = state
                .entries
                .get_mut(&action.barrier_id())
                .expect("validated physical barrier entry remains present while locked");
            begin_next_generation(entry);
            debug_assert_eq!(entry.generation, action.generation());
            let required_arrivals = entry.required_arrivals();
            let completed = entry
                .phase
                .arrival_count
                .checked_add(arrival_count)
                .ok_or_else(|| PhysicalCompletionActionError::ApplyFailed {
                    action,
                    source: SynchronizationError::BarrierArrivalOverflow {
                        key: key.clone(),
                        phase: entry.generation & 1,
                        expected: required_arrivals,
                        completed: u64::MAX,
                    },
                })?;
            if completed > required_arrivals {
                return Err(PhysicalCompletionActionError::ApplyFailed {
                    action,
                    source: SynchronizationError::BarrierArrivalOverflow {
                        key,
                        phase: entry.generation & 1,
                        expected: required_arrivals,
                        completed,
                    },
                });
            }
            entry.phase.arrival_count = completed;
            entry.phase.arrived_warps.insert(warp_id);
            if completed == required_arrivals
                && entry.phase.completed_transactions > entry.phase.expected_transactions
            {
                return Err(PhysicalCompletionActionError::ApplyFailed {
                    action,
                    source: SynchronizationError::TransactionOverflow {
                        key,
                        phase: entry.generation & 1,
                        expected: entry.phase.expected_transactions,
                        completed: entry.phase.completed_transactions,
                    },
                });
            }
            complete_physical_if_ready(entry)
        }
    };
    let removed = state
        .transaction_completions
        .remove(&action_id)
        .expect("validated physical completion action remains queued while locked");
    debug_assert_eq!(removed, action);
    Ok((
        CompletionProgress {
            completed_operations: 1,
            woken_warps: wakers.len(),
        },
        wakers,
    ))
}

fn apply_physical_transaction_completion_entry(
    id: PhysicalBarrierId,
    entry: &mut PhysicalBarrierEntry,
    generation: u64,
    transactions: u64,
) -> Result<Vec<Waker>, SynchronizationError> {
    if generation != entry.generation {
        let buffered = entry
            .buffered_transactions
            .get(&generation)
            .copied()
            .unwrap_or(0)
            .checked_add(transactions)
            .ok_or_else(|| SynchronizationError::TransactionOverflow {
                key: id.occurrence_key(),
                phase: generation & 1,
                expected: u64::MAX,
                completed: u64::MAX,
            })?;
        entry.buffered_transactions.insert(generation, buffered);
        return Ok(Vec::new());
    }
    let completed = entry
        .phase
        .completed_transactions
        .checked_add(transactions)
        .ok_or_else(|| SynchronizationError::TransactionOverflow {
            key: id.occurrence_key(),
            phase: entry.generation & 1,
            expected: entry.phase.expected_transactions,
            completed: u64::MAX,
        })?;
    if entry.phase.arrival_count == entry.required_arrivals()
        && completed > entry.phase.expected_transactions
    {
        return Err(SynchronizationError::TransactionOverflow {
            key: id.occurrence_key(),
            phase: entry.generation & 1,
            expected: entry.phase.expected_transactions,
            completed,
        });
    }
    entry.phase.completed_transactions = completed;
    Ok(complete_physical_if_ready(entry))
}

fn begin_next_generation(entry: &mut PhysicalBarrierEntry) {
    if !entry.phase.complete {
        return;
    }
    entry.previous_report = entry.phase.report;
    entry.generation += 1;
    entry.phase = CountedPhase {
        completed_transactions: entry
            .buffered_transactions
            .remove(&entry.generation)
            .unwrap_or(0),
        ..CountedPhase::default()
    };
}

fn physical_completion_is_enabled(
    entry: &PhysicalBarrierEntry,
    action: PhysicalCompletionAction,
) -> bool {
    action.generation() == entry.generation
        || (entry.phase.complete
            && entry
                .generation
                .checked_add(1)
                .is_some_and(|generation| action.generation() == generation))
}

fn physical_phase_is_active(phase: &CountedPhase) -> bool {
    !phase.complete
        && (phase.arrival_count != 0
            || phase.pending_arrival_increments != 0
            || phase.expected_transactions != 0
            || phase.completed_transactions != 0)
}

fn complete_physical_if_ready(entry: &mut PhysicalBarrierEntry) -> Vec<Waker> {
    if entry.phase.complete
        || entry.phase.arrival_count != entry.required_arrivals()
        || entry.phase.completed_transactions != entry.phase.expected_transactions
    {
        return Vec::new();
    }
    entry.phase.complete = true;
    entry.last_completed_generation = Some(entry.generation);
    entry.completed_parity = entry.generation & 1;
    // A failed report advances the primary phase but leaves the conditional
    // phase and its acquire witness intact, even across many primary phases.
    if !entry.phase.report {
        entry.conditional_completed_parity ^= 1;
        entry.conditional_completed_generation = Some(entry.generation);
    }
    let completed_parity = entry.completed_parity;
    let ready = entry
        .waiters
        .iter()
        .filter_map(|(&warp_id, waiter)| {
            (waiter.requested_phase == completed_parity).then_some(warp_id)
        })
        .collect::<Vec<_>>();
    ready
        .into_iter()
        .filter_map(|warp_id| entry.waiters.remove(&warp_id).map(|waiter| waiter.waker))
        .collect()
}

/// Identity of one CTA-local named barrier or analyzer-owned rendezvous.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum NamedBarrierNamespace {
    Hardware,
    InternalWarpgroup {
        static_op_id: u64,
        warpgroup_id: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NamedBarrierId {
    global_cta_id: usize,
    barrier_id: u32,
    namespace: NamedBarrierNamespace,
}

impl NamedBarrierId {
    pub const fn new(global_cta_id: usize, barrier_id: u32) -> Self {
        Self {
            global_cta_id,
            barrier_id,
            namespace: NamedBarrierNamespace::Hardware,
        }
    }

    pub(crate) const fn internal_warpgroup(
        global_cta_id: usize,
        barrier_id: u32,
        static_op_id: u64,
        warpgroup_id: usize,
    ) -> Self {
        Self {
            global_cta_id,
            barrier_id,
            namespace: NamedBarrierNamespace::InternalWarpgroup {
                static_op_id,
                warpgroup_id,
            },
        }
    }

    pub const fn global_cta_id(self) -> usize {
        self.global_cta_id
    }

    pub const fn barrier_id(self) -> u32 {
        self.barrier_id
    }

    fn occurrence_key(self, generation: u64) -> OccurrenceKey {
        let generation = i64::try_from(generation).unwrap_or(i64::MAX);
        match self.namespace {
            NamedBarrierNamespace::Hardware => OccurrenceKey::new(
                u64::from(self.barrier_id),
                format!("named_barrier[{}]", self.barrier_id),
                [generation],
                ScopeInstance::Cta {
                    global_cta_id: self.global_cta_id,
                },
            ),
            NamedBarrierNamespace::InternalWarpgroup {
                static_op_id,
                warpgroup_id,
            } => OccurrenceKey::new(
                static_op_id,
                "internal warpgroup rendezvous",
                [generation],
                ScopeInstance::WarpGroup {
                    global_cta_id: self.global_cta_id,
                    warpgroup_id,
                },
            ),
        }
    }
}

/// Exact outcome of one nonblocking named-barrier contribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamedBarrierArrivalOutcome {
    generation: u64,
    completed_now: bool,
}

impl NamedBarrierArrivalOutcome {
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

/// Weighted, generation-reusable CTA named barriers (`bar.arrive` / `bar.sync`).
#[derive(Default)]
pub struct NamedBarrierHub {
    publication_gate: Mutex<()>,
    state: Mutex<BTreeMap<NamedBarrierId, NamedBarrierEntry>>,
}

#[derive(Clone)]
struct NamedBarrierEntry {
    expected_arrivals: u64,
    generation: u64,
    completed_through: Option<u64>,
    phase: CountedPhase,
    // A lane may contribute once through bar.arrive and once through the
    // subsequent bar.sync in the same counted generation.  CUTLASS uses
    // exactly that 32 + 128 = 160 epilogue rendezvous.  Keep duplicate
    // detection per instruction flavor so repeated arrive or repeated sync
    // remains an error.
    contributors: BTreeMap<(usize, bool), WarpMask>,
    waiters: BTreeMap<(u64, usize, u32), Option<Waker>>,
}

impl NamedBarrierHub {
    /// Snapshot the actual participant set while a completed generation is
    /// still current. Its first reduction waiter transfers this immutable
    /// contract to the existing collective hub before yielding.
    pub(crate) fn completed_participants(
        &self,
        id: NamedBarrierId,
        generation: u64,
    ) -> Result<crate::ParticipantSet, EngineError> {
        let state = self.state.lock().expect("named barrier mutex poisoned");
        let entry = state
            .get(&id)
            .filter(|entry| entry.generation == generation && entry.phase.complete)
            .ok_or_else(|| {
                EngineError::message("named-barrier reduction lost its completed generation")
            })?;
        crate::ParticipantSet::new(entry.phase.arrived_warps.iter().copied()).map_err(Into::into)
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn sync(
        self: &Arc<Self>,
        id: NamedBarrierId,
        expected_arrivals: u64,
        warp_id: usize,
        arrival_count: u64,
    ) -> Result<NamedBarrierWait, SynchronizationError> {
        if arrival_count == 0 || arrival_count > WARP_SIZE as u64 {
            return Err(SynchronizationError::InvalidBarrierArrivalCount {
                key: id.occurrence_key(0),
                count: arrival_count,
            });
        }
        let bits = if arrival_count == WARP_SIZE as u64 {
            u32::MAX
        } else {
            (1_u32 << arrival_count) - 1
        };
        self.register_sync(id, expected_arrivals, warp_id, WarpMask::from_bits(bits))
    }

    /// Transactionally apply one exact nonblocking `bar.arrive` contribution.
    pub fn arrive(
        self: &Arc<Self>,
        id: NamedBarrierId,
        expected_arrivals: u64,
        warp_id: usize,
        arrival_mask: WarpMask,
    ) -> Result<NamedBarrierArrivalOutcome, SynchronizationError> {
        self.arrive_with_outcome(id, expected_arrivals, warp_id, arrival_mask, |_| Ok(()))
            .map_err(|error| match error {
                NamedBarrierCommitError::Barrier(error) => error,
                NamedBarrierCommitError::Publication(error) => {
                    unreachable!("empty named-barrier publication failed: {error}")
                }
            })
    }

    /// Apply one exact contribution, publish its mode-visible release, and
    /// only then wake waiters made ready by the numeric transition.
    pub(crate) fn arrive_with_outcome(
        self: &Arc<Self>,
        id: NamedBarrierId,
        expected_arrivals: u64,
        warp_id: usize,
        arrival_mask: WarpMask,
        publish_before_wake: impl FnOnce(&NamedBarrierArrivalOutcome) -> Result<(), EngineError>,
    ) -> Result<NamedBarrierArrivalOutcome, NamedBarrierCommitError> {
        let _publication = self
            .publication_gate
            .lock()
            .expect("named barrier publication gate poisoned");
        let (outcome, wakers) = self
            .register_contribution(id, expected_arrivals, warp_id, arrival_mask, false)
            .map_err(NamedBarrierCommitError::Barrier)?;
        publish_before_wake(&outcome).map_err(NamedBarrierCommitError::Publication)?;
        wake_all(wakers);
        Ok(outcome)
    }

    /// Transactionally register one exact `bar.sync` lane contribution and waiter.
    ///
    /// Registration is deliberately synchronous: analysis sees the contribution
    /// before the returned future can block, instead of inferring it from the
    /// executor's first poll of that future.
    pub fn register_sync(
        self: &Arc<Self>,
        id: NamedBarrierId,
        expected_arrivals: u64,
        warp_id: usize,
        arrival_mask: WarpMask,
    ) -> Result<NamedBarrierWait, SynchronizationError> {
        self.register_sync_with_outcome(id, expected_arrivals, warp_id, arrival_mask, |_| Ok(()))
            .map_err(|error| match error {
                NamedBarrierCommitError::Barrier(error) => error,
                NamedBarrierCommitError::Publication(error) => {
                    unreachable!("empty named-barrier publication failed: {error}")
                }
            })
    }

    /// Register one blocking contribution, publish its mode-visible release,
    /// and only then wake waiters made ready by the numeric transition.
    pub(crate) fn register_sync_with_outcome(
        self: &Arc<Self>,
        id: NamedBarrierId,
        expected_arrivals: u64,
        warp_id: usize,
        arrival_mask: WarpMask,
        publish_before_wake: impl FnOnce(&NamedBarrierArrivalOutcome) -> Result<(), EngineError>,
    ) -> Result<NamedBarrierWait, NamedBarrierCommitError> {
        let _publication = self
            .publication_gate
            .lock()
            .expect("named barrier publication gate poisoned");
        let (outcome, wakers) = self
            .register_contribution(id, expected_arrivals, warp_id, arrival_mask, true)
            .map_err(NamedBarrierCommitError::Barrier)?;
        publish_before_wake(&outcome).map_err(NamedBarrierCommitError::Publication)?;
        wake_all(wakers);
        Ok(NamedBarrierWait {
            hub: Arc::clone(self),
            id,
            warp_id,
            arrival_mask,
            generation: outcome.generation(),
            completed_now: outcome.completed_now(),
            registered: true,
            finished: false,
        })
    }

    fn register_contribution(
        &self,
        id: NamedBarrierId,
        expected_arrivals: u64,
        warp_id: usize,
        arrival_mask: WarpMask,
        register_waiter: bool,
    ) -> Result<(NamedBarrierArrivalOutcome, Vec<Waker>), SynchronizationError> {
        let key = id.occurrence_key(0);
        if expected_arrivals == 0 {
            return Err(SynchronizationError::InvalidBarrierArrivalCount {
                key,
                count: expected_arrivals,
            });
        }
        if arrival_mask.is_empty() {
            return Err(SynchronizationError::InvalidBarrierArrivalCount { key, count: 0 });
        }

        let (generation, completed_now, wakers) = {
            let mut state = self.state.lock().expect("named barrier mutex poisoned");
            let mut entry = state
                .get(&id)
                .cloned()
                .unwrap_or_else(|| NamedBarrierEntry {
                    expected_arrivals,
                    generation: 0,
                    completed_through: None,
                    phase: CountedPhase::default(),
                    contributors: BTreeMap::new(),
                    waiters: BTreeMap::new(),
                });
            if entry.phase.complete {
                entry.completed_through = Some(entry.generation);
                entry.generation = entry.generation.checked_add(1).ok_or_else(|| {
                    SynchronizationError::CompletionSourceOperationFailed {
                        source_name: "named-barrier",
                        details: format!(
                            "named barrier {id:?} cannot advance beyond generation {}",
                            entry.generation
                        ),
                    }
                })?;
                entry.expected_arrivals = expected_arrivals;
                entry.phase = CountedPhase::default();
                entry.contributors.clear();
                entry.waiters.clear();
            } else if entry.expected_arrivals != expected_arrivals {
                return Err(SynchronizationError::ContractMismatch {
                    key: id.occurrence_key(entry.generation),
                });
            }

            let generation = entry.generation;
            let key = id.occurrence_key(generation);
            let contributor_key = (warp_id, register_waiter);
            let prior_mask = entry
                .contributors
                .get(&contributor_key)
                .copied()
                .unwrap_or(WarpMask::EMPTY);
            if !prior_mask.intersection(arrival_mask).is_empty() {
                return Err(SynchronizationError::DuplicateArrival {
                    key,
                    phase: generation,
                    warp_id,
                });
            }
            let completed = entry
                .phase
                .arrival_count
                .checked_add(arrival_mask.len() as u64)
                .ok_or_else(|| SynchronizationError::BarrierArrivalOverflow {
                    key: key.clone(),
                    phase: generation,
                    expected: entry.expected_arrivals,
                    completed: u64::MAX,
                })?;
            if completed > entry.expected_arrivals {
                return Err(SynchronizationError::BarrierArrivalOverflow {
                    key,
                    phase: generation,
                    expected: entry.expected_arrivals,
                    completed,
                });
            }

            entry
                .contributors
                .insert(contributor_key, prior_mask.union(arrival_mask));
            entry.phase.arrival_count = completed;
            entry.phase.arrived_warps.insert(warp_id);
            if register_waiter {
                let waiter_key = (generation, warp_id, arrival_mask.bits());
                if entry.waiters.insert(waiter_key, None).is_some() {
                    return Err(SynchronizationError::DuplicateWaiter {
                        key: id.occurrence_key(generation),
                        phase: Some(generation),
                        warp_id,
                    });
                }
            }
            let completed_now = completed == entry.expected_arrivals;
            let wakers = if completed_now {
                entry.phase.complete = true;
                entry
                    .waiters
                    .values()
                    .filter_map(Clone::clone)
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            state.insert(id, entry);
            (generation, completed_now, wakers)
        };
        Ok((
            NamedBarrierArrivalOutcome::new(generation, completed_now),
            wakers,
        ))
    }
}

#[derive(Debug)]
pub(crate) enum NamedBarrierCommitError {
    Barrier(SynchronizationError),
    Publication(EngineError),
}

impl fmt::Display for NamedBarrierCommitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Barrier(error) => error.fmt(f),
            Self::Publication(error) => {
                write!(f, "named-barrier release publication failed: {error}")
            }
        }
    }
}

impl Error for NamedBarrierCommitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Barrier(error) => Some(error),
            Self::Publication(error) => Some(error),
        }
    }
}

impl CompletionSource for NamedBarrierHub {
    fn source_name(&self) -> &'static str {
        "named-barrier"
    }

    fn pump(&self) -> Result<CompletionProgress, SynchronizationError> {
        Ok(CompletionProgress::default())
    }

    fn blocked_operations(&self) -> Vec<BlockedOperation> {
        let state = self.state.lock().expect("named barrier mutex poisoned");
        let mut blocked = Vec::new();
        for (id, entry) in &*state {
            for &(generation, warp_id, _) in entry.waiters.keys() {
                blocked.push(BlockedOperation::new(
                    warp_id,
                    crate::AwaitedOperation::NamedBarrierSync,
                    id.occurrence_key(generation),
                    None,
                    ParticipantState::counted(
                        entry.phase.arrived_warps.iter().copied(),
                        entry.expected_arrivals,
                        entry.phase.arrival_count,
                        None,
                        None,
                    ),
                ));
            }
        }
        blocked
    }

    fn validate_quiescent(&self) -> Result<(), SynchronizationError> {
        let state = self.state.lock().expect("named barrier mutex poisoned");
        for (id, entry) in &*state {
            if !entry.phase.complete || !entry.waiters.is_empty() {
                return Err(id
                    .occurrence_key(entry.generation)
                    .completion_not_quiescent_error(
                        self.source_name(),
                        format_args!(
                            " is incomplete: arrivals={}/{}, waiting_warps={:?}",
                            entry.phase.arrival_count,
                            entry.expected_arrivals,
                            entry
                                .waiters
                                .keys()
                                .map(|(_, warp_id, _)| *warp_id)
                                .collect::<Vec<_>>()
                        ),
                    ));
            }
        }
        Ok(())
    }
}

pub struct NamedBarrierWait {
    hub: Arc<NamedBarrierHub>,
    id: NamedBarrierId,
    warp_id: usize,
    arrival_mask: WarpMask,
    generation: u64,
    completed_now: bool,
    registered: bool,
    finished: bool,
}

impl NamedBarrierWait {
    pub const fn barrier_id(&self) -> NamedBarrierId {
        self.id
    }

    pub const fn warp_id(&self) -> usize {
        self.warp_id
    }

    pub const fn arrival_mask(&self) -> WarpMask {
        self.arrival_mask
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn completed_now(&self) -> bool {
        self.completed_now
    }

    /// Return every disjoint lane contribution registered by this warp in the
    /// current unaligned-barrier generation.
    pub fn accumulated_arrival_mask(&self) -> Result<WarpMask, SynchronizationError> {
        let state = self.hub.state.lock().expect("named barrier mutex poisoned");
        let Some(entry) = state.get(&self.id) else {
            return Err(SynchronizationError::UndefinedOccurrence {
                key: self.id.occurrence_key(self.generation),
            });
        };
        if entry.generation != self.generation {
            return Err(SynchronizationError::UndefinedOccurrence {
                key: self.id.occurrence_key(self.generation),
            });
        }
        entry
            .contributors
            .get(&(self.warp_id, true))
            .copied()
            .ok_or_else(|| SynchronizationError::UndefinedOccurrence {
                key: self.id.occurrence_key(self.generation),
            })
    }
}

impl Future for NamedBarrierWait {
    type Output = Result<(), SynchronizationError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let result = {
            let mut state = this.hub.state.lock().expect("named barrier mutex poisoned");
            let Some(entry) = state.get_mut(&this.id) else {
                return Poll::Ready(Err(SynchronizationError::UndefinedOccurrence {
                    key: this.id.occurrence_key(this.generation),
                }));
            };
            let generation = this.generation;
            let key = this.id.occurrence_key(generation);
            let completed = entry.phase.complete && generation == entry.generation
                || entry
                    .completed_through
                    .is_some_and(|through| generation <= through);
            if completed {
                entry
                    .waiters
                    .remove(&(generation, this.warp_id, this.arrival_mask.bits()));
                this.registered = false;
                this.finished = true;
                Poll::Ready(Ok(()))
            } else {
                let waiter_key = (generation, this.warp_id, this.arrival_mask.bits());
                match entry.waiters.get_mut(&waiter_key) {
                    Some(waker) => {
                        if waker
                            .as_ref()
                            .map(|waker| !waker.will_wake(context.waker()))
                            .unwrap_or(true)
                        {
                            *waker = Some(context.waker().clone());
                        }
                    }
                    None => {
                        return Poll::Ready(Err(SynchronizationError::UndefinedOccurrence { key }));
                    }
                }
                Poll::Pending
            }
        };
        result
    }
}

impl Drop for NamedBarrierWait {
    fn drop(&mut self) {
        if !self.registered || self.finished {
            return;
        }
        let mut state = self.hub.state.lock().expect("named barrier mutex poisoned");
        if let Some(entry) = state.get_mut(&self.id) {
            entry
                .waiters
                .remove(&(self.generation, self.warp_id, self.arrival_mask.bits()));
        }
    }
}

fn wake_all(wakers: Vec<Waker>) {
    for waker in wakers {
        waker.wake();
    }
}

#[cfg(test)]
#[path = "hardware_barriers_tests.rs"]
mod tests;
