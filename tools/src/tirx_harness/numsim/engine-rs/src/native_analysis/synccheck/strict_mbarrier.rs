use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::{DynamicOpId, PhysicalBarrierId, MAX_MBARRIER_EXPECTED_ARRIVALS};

type SharedWitness = Option<Arc<DynamicOpId>>;

fn shared_witness(witness: Option<DynamicOpId>) -> SharedWitness {
    witness.map(Arc::new)
}

/// Strict checker-visible lifecycle of one physical mbarrier slot.
///
/// Numeric execution only needs to know whether a phase has completed. The
/// analysis modes additionally retain whether that completion has been
/// observed by a matching wait before permitting the slot to advance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StrictMbarrierLifecycle {
    Uninitialized,
    Pending,
    CompletedUnconsumed,
    Consumed,
}

/// Operation being validated when a strict protocol error is raised.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrictMbarrierOperation {
    Init,
    Invalidate,
    ExpectTx,
    Arrive,
    ArriveExpectTx,
    IncreasePendingArrivals,
    Wait,
    CaptureCompletion,
    CompleteTx,
}

impl fmt::Display for StrictMbarrierOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Init => "mbarrier.init",
            Self::Invalidate => "mbarrier.inval",
            Self::ExpectTx => "mbarrier.expect_tx",
            Self::Arrive => "mbarrier.arrive",
            Self::ArriveExpectTx => "mbarrier.arrive.expect_tx",
            Self::IncreasePendingArrivals => "cp.async.mbarrier.arrive",
            Self::Wait => "mbarrier.try_wait",
            Self::CaptureCompletion => "mbarrier completion issue",
            Self::CompleteTx => "mbarrier.complete_tx",
        })
    }
}

/// Stable identity of a generation-bound asynchronous completion token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StrictMbarrierCompletionTokenId(u64);

impl StrictMbarrierCompletionTokenId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for StrictMbarrierCompletionTokenId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Token captured when an asynchronous payload operation is issued.
///
/// The token binds the eventual transaction credit to a concrete barrier
/// generation. Applying it later never guesses a generation from the slot's
/// then-current lifecycle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrictMbarrierCompletionToken {
    id: StrictMbarrierCompletionTokenId,
    barrier_id: PhysicalBarrierId,
    generation: u64,
    issue_witness: SharedWitness,
}

impl StrictMbarrierCompletionToken {
    pub const fn id(&self) -> StrictMbarrierCompletionTokenId {
        self.id
    }

    pub const fn barrier_id(&self) -> PhysicalBarrierId {
        self.barrier_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn issue_witness(&self) -> Option<&DynamicOpId> {
        self.issue_witness.as_deref()
    }
}

/// One warp blocked on a parity wait for a concrete generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrictMbarrierWaiter {
    warp_id: usize,
    generation: u64,
    requested_phase: u64,
    witness: SharedWitness,
}

impl StrictMbarrierWaiter {
    pub const fn warp_id(&self) -> usize {
        self.warp_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn requested_phase(&self) -> u64 {
        self.requested_phase
    }

    pub fn witness(&self) -> Option<&DynamicOpId> {
        self.witness.as_deref()
    }
}

/// State changes caused by an arrive or transaction-completion event.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StrictMbarrierEffect {
    completed_generation: Option<u64>,
    consumed_generation: Option<u64>,
    ready_waiters: Vec<StrictMbarrierWaiter>,
}

impl StrictMbarrierEffect {
    pub const fn completed_generation(&self) -> Option<u64> {
        self.completed_generation
    }

    pub const fn consumed_generation(&self) -> Option<u64> {
        self.consumed_generation
    }

    pub fn ready_waiters(&self) -> &[StrictMbarrierWaiter] {
        &self.ready_waiters
    }
}

/// Result of one parity wait operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StrictMbarrierWaitOutcome {
    /// The requested parity was already available. `generation` is `None` for
    /// the vacuous phase-one success immediately after initialization.
    Ready {
        generation: Option<u64>,
        consumed_now: bool,
    },
    /// The warp is waiting for the specified current or next generation.
    Registered { generation: u64 },
}

/// Read-only state used by checker diagnostics and unit tests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrictMbarrierSnapshot {
    lifecycle: StrictMbarrierLifecycle,
    generation: Option<u64>,
    current_phase: Option<u64>,
    init_fenced: bool,
    last_completed_phase: u64,
    last_completed_generation: Option<u64>,
    expected_arrivals: Option<u64>,
    arrival_count: u64,
    arrived_warps: Vec<usize>,
    expected_transactions: u64,
    completed_transactions: u64,
    waiting_warps: Vec<StrictMbarrierWaiter>,
    buffered_transactions: Vec<(u64, u64)>,
    outstanding_completion_tokens: Vec<(StrictMbarrierCompletionTokenId, u64)>,
    init_witness: SharedWitness,
    init_fence_witness: SharedWitness,
    completion_witness: SharedWitness,
    consumption_witness: SharedWitness,
}

impl StrictMbarrierSnapshot {
    pub const fn lifecycle(&self) -> StrictMbarrierLifecycle {
        self.lifecycle
    }

    pub const fn generation(&self) -> Option<u64> {
        self.generation
    }

    pub const fn current_phase(&self) -> Option<u64> {
        self.current_phase
    }

    pub const fn init_fenced(&self) -> bool {
        self.init_fenced
    }

    pub const fn last_completed_phase(&self) -> u64 {
        self.last_completed_phase
    }

    pub const fn last_completed_generation(&self) -> Option<u64> {
        self.last_completed_generation
    }

    pub const fn expected_arrivals(&self) -> Option<u64> {
        self.expected_arrivals
    }

    pub const fn arrival_count(&self) -> u64 {
        self.arrival_count
    }

    pub const fn expected_transactions(&self) -> u64 {
        self.expected_transactions
    }

    pub const fn completed_transactions(&self) -> u64 {
        self.completed_transactions
    }

    pub fn waiting_warps(&self) -> &[StrictMbarrierWaiter] {
        &self.waiting_warps
    }

    pub fn buffered_transactions(&self) -> &[(u64, u64)] {
        &self.buffered_transactions
    }

    pub fn outstanding_completion_tokens(&self) -> &[(StrictMbarrierCompletionTokenId, u64)] {
        &self.outstanding_completion_tokens
    }

    pub fn init_witness(&self) -> Option<&DynamicOpId> {
        self.init_witness.as_deref()
    }

    pub fn init_fence_witness(&self) -> Option<&DynamicOpId> {
        self.init_fence_witness.as_deref()
    }

    pub fn completion_witness(&self) -> Option<&DynamicOpId> {
        self.completion_witness.as_deref()
    }

    pub fn consumption_witness(&self) -> Option<&DynamicOpId> {
        self.consumption_witness.as_deref()
    }
}

/// Fail-closed strict protocol error with optional source-level witnesses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StrictMbarrierError {
    InvalidateWithOutstandingWork {
        barrier_id: PhysicalBarrierId,
        witness: SharedWitness,
    },
    Uninitialized {
        barrier_id: PhysicalBarrierId,
        operation: StrictMbarrierOperation,
        witness: SharedWitness,
    },
    InvalidPhase {
        barrier_id: PhysicalBarrierId,
        phase: u64,
        witness: SharedWitness,
    },
    AcquireBeforeCompletion {
        barrier_id: PhysicalBarrierId,
        generation: u64,
        last_completed: Option<u64>,
        witness: SharedWitness,
    },
    InvalidExpectedArrivals {
        barrier_id: PhysicalBarrierId,
        expected: u64,
        witness: SharedWitness,
    },
    ReinitializeBeforeConsumption {
        barrier_id: PhysicalBarrierId,
        generation: u64,
        completion_witness: SharedWitness,
        witness: SharedWitness,
    },
    ReinitializeWhileActive {
        barrier_id: PhysicalBarrierId,
        generation: u64,
        lifecycle: StrictMbarrierLifecycle,
        waiting_warps: Vec<usize>,
        outstanding_completion_tokens: usize,
        buffered_generation_count: usize,
        witness: SharedWitness,
    },
    ReinitializeWithoutInvalidation {
        barrier_id: PhysicalBarrierId,
        generation: u64,
        lifecycle: StrictMbarrierLifecycle,
        init_witness: SharedWitness,
        witness: SharedWitness,
    },
    ArriveBeforeConsumption {
        barrier_id: PhysicalBarrierId,
        generation: u64,
        completion_witness: SharedWitness,
        witness: SharedWitness,
    },
    ExpectTxBeforeConsumption {
        barrier_id: PhysicalBarrierId,
        generation: u64,
        completion_witness: SharedWitness,
        witness: SharedWitness,
    },
    ArrivalOverflow {
        barrier_id: PhysicalBarrierId,
        generation: u64,
        expected: u64,
        completed: u64,
        witness: SharedWitness,
    },
    CounterOverflow {
        barrier_id: PhysicalBarrierId,
        generation: u64,
        counter: &'static str,
        witness: SharedWitness,
    },
    TransactionOverDelivery {
        barrier_id: PhysicalBarrierId,
        generation: u64,
        expected: u64,
        completed: u64,
        witness: SharedWitness,
    },
    DuplicateWaiter {
        barrier_id: PhysicalBarrierId,
        warp_id: usize,
        registered_generation: u64,
        witness: SharedWitness,
    },
    CompletionAfterGenerationComplete {
        barrier_id: PhysicalBarrierId,
        generation: u64,
        lifecycle: StrictMbarrierLifecycle,
        issue_witness: SharedWitness,
        witness: SharedWitness,
    },
    StaleCompletion {
        barrier_id: PhysicalBarrierId,
        captured_generation: u64,
        current_generation: u64,
        issue_witness: SharedWitness,
        witness: SharedWitness,
    },
    FutureCompletionNotBufferable {
        barrier_id: PhysicalBarrierId,
        captured_generation: u64,
        current_generation: u64,
        lifecycle: StrictMbarrierLifecycle,
        issue_witness: SharedWitness,
        witness: SharedWitness,
    },
    UnknownCompletionToken {
        token_id: StrictMbarrierCompletionTokenId,
    },
    GenerationOverflow {
        barrier_id: PhysicalBarrierId,
        generation: u64,
        operation: StrictMbarrierOperation,
        witness: SharedWitness,
    },
    CompletionTokenSpaceExhausted {
        barrier_id: PhysicalBarrierId,
        witness: SharedWitness,
    },
}

impl fmt::Display for StrictMbarrierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidateWithOutstandingWork { barrier_id, witness } => write!(f,
                "mbarrier.inval {barrier_id:?} has outstanding waiters or asynchronous completions{}", witness_suffix(witness)),
            Self::Uninitialized {
                barrier_id,
                operation,
                witness,
            } => write!(
                f,
                "{operation} used uninitialized physical mbarrier {barrier_id:?}{}",
                witness_suffix(witness)
            ),
            Self::InvalidPhase {
                barrier_id,
                phase,
                witness,
            } => write!(
                f,
                "mbarrier wait phase for {barrier_id:?} must be 0 or 1, got {phase}{}",
                witness_suffix(witness)
            ),
            Self::AcquireBeforeCompletion {
                barrier_id,
                generation,
                last_completed,
                witness,
            } => write!(
                f,
                "mbarrier acquire for {barrier_id:?} names generation {generation} beyond the last completed generation {last_completed:?}{}",
                witness_suffix(witness)
            ),
            Self::InvalidExpectedArrivals {
                barrier_id,
                expected,
                witness,
            } => write!(
                f,
                "mbarrier.init for {barrier_id:?} requires expected arrivals in 1..={MAX_MBARRIER_EXPECTED_ARRIVALS}, got {expected}{}",
                witness_suffix(witness)
            ),
            Self::ReinitializeBeforeConsumption {
                barrier_id,
                generation,
                completion_witness,
                witness,
            } => write!(
                f,
                "mbarrier.init attempted on {barrier_id:?} generation {generation} before its completion was consumed{}{}",
                prior_witness_suffix(completion_witness),
                witness_suffix(witness)
            ),
            Self::ReinitializeWhileActive {
                barrier_id,
                generation,
                lifecycle,
                waiting_warps,
                outstanding_completion_tokens,
                buffered_generation_count,
                witness,
            } => write!(
                f,
                "mbarrier.init attempted on active {barrier_id:?} generation {generation} ({lifecycle:?}, waiters={waiting_warps:?}, outstanding_tokens={outstanding_completion_tokens}, buffered_generations={buffered_generation_count}){}",
                witness_suffix(witness)
            ),
            Self::ReinitializeWithoutInvalidation {
                barrier_id,
                generation,
                lifecycle,
                init_witness,
                witness,
            } => write!(
                f,
                "mbarrier.init attempted on valid {barrier_id:?} generation {generation} ({lifecycle:?}) without mbarrier.inval{}{}",
                prior_witness_suffix(init_witness),
                witness_suffix(witness)
            ),
            Self::ArriveBeforeConsumption {
                barrier_id,
                generation,
                completion_witness,
                witness,
            } => write!(
                f,
                "next-generation arrive attempted on {barrier_id:?} before generation {generation} was consumed{}{}",
                prior_witness_suffix(completion_witness),
                witness_suffix(witness)
            ),
            Self::ExpectTxBeforeConsumption {
                barrier_id,
                generation,
                completion_witness,
                witness,
            } => write!(
                f,
                "next-generation mbarrier.expect_tx attempted on {barrier_id:?} before generation {generation} was consumed{}{}",
                prior_witness_suffix(completion_witness),
                witness_suffix(witness)
            ),
            Self::ArrivalOverflow {
                barrier_id,
                generation,
                expected,
                completed,
                witness,
            } => write!(
                f,
                "mbarrier {barrier_id:?} generation {generation} received {completed} arrivals, expected {expected}{}",
                witness_suffix(witness)
            ),
            Self::CounterOverflow {
                barrier_id,
                generation,
                counter,
                witness,
            } => write!(
                f,
                "mbarrier {barrier_id:?} generation {generation} overflowed its {counter} counter{}",
                witness_suffix(witness)
            ),
            Self::TransactionOverDelivery {
                barrier_id,
                generation,
                expected,
                completed,
                witness,
            } => write!(
                f,
                "mbarrier {barrier_id:?} generation {generation} completed {completed} transaction bytes, expected {expected}{}",
                witness_suffix(witness)
            ),
            Self::DuplicateWaiter {
                barrier_id,
                warp_id,
                registered_generation,
                witness,
            } => write!(
                f,
                "warp {warp_id} already waits on mbarrier {barrier_id:?} generation {registered_generation}{}",
                witness_suffix(witness)
            ),
            Self::CompletionAfterGenerationComplete {
                barrier_id,
                generation,
                lifecycle,
                issue_witness,
                witness,
            } => write!(
                f,
                "transaction completion targeted already-complete mbarrier {barrier_id:?} generation {generation} ({lifecycle:?}){}{}",
                prior_witness_suffix(issue_witness),
                witness_suffix(witness)
            ),
            Self::StaleCompletion {
                barrier_id,
                captured_generation,
                current_generation,
                issue_witness,
                witness,
            } => write!(
                f,
                "transaction completion for mbarrier {barrier_id:?} generation {captured_generation} is stale; current generation is {current_generation}{}{}",
                prior_witness_suffix(issue_witness),
                witness_suffix(witness)
            ),
            Self::FutureCompletionNotBufferable {
                barrier_id,
                captured_generation,
                current_generation,
                lifecycle,
                issue_witness,
                witness,
            } => write!(
                f,
                "transaction completion for future mbarrier {barrier_id:?} generation {captured_generation} cannot be buffered while generation {current_generation} is {lifecycle:?}{}{}",
                prior_witness_suffix(issue_witness),
                witness_suffix(witness)
            ),
            Self::UnknownCompletionToken { token_id } => {
                write!(f, "mbarrier completion token {token_id} is not outstanding")
            }
            Self::GenerationOverflow {
                barrier_id,
                generation,
                operation,
                witness,
            } => write!(
                f,
                "{operation} cannot advance mbarrier {barrier_id:?} beyond generation {generation}{}",
                witness_suffix(witness)
            ),
            Self::CompletionTokenSpaceExhausted {
                barrier_id,
                witness,
            } => write!(
                f,
                "completion token ID space exhausted while issuing for mbarrier {barrier_id:?}{}",
                witness_suffix(witness)
            ),
        }
    }
}

impl Error for StrictMbarrierError {}

fn witness_suffix(witness: &SharedWitness) -> String {
    witness
        .as_ref()
        .map(|witness| format!(" at {witness}"))
        .unwrap_or_default()
}

fn prior_witness_suffix(witness: &SharedWitness) -> String {
    witness
        .as_ref()
        .map(|witness| format!("; prior event: {witness}"))
        .unwrap_or_default()
}

/// Launch-wide strict mbarrier protocol tracker.
///
/// This supplements, rather than replaces, [`crate::PhysicalBarrierHub`]. The
/// numeric hub owns actual Future wakeups and payload progress; this tracker
/// owns checker-only consumption and generation-token invariants.
///
/// # Why the counter duplication here is deliberate
///
/// This tracker re-derives state the numeric hub already owns (generation,
/// arrival and transaction counts, buffered transactions, the waiter set).
/// The independent counters let synccheck cross-check this tracker's prediction
/// against the numeric hub's committed result:
///
///   * `OwnedSyncEffect::apply` (sync_check.rs) runs in `before_effect`, i.e.
///     strictly before the hub mutates, and predicts the hub's result as
///     `SyncEffectPreview::Arrive { expected: PhysicalMbarrierArrivalOutcome }`
///     from this tracker's own counters.
///   * `after_effect` compares that prediction against the hub's committed
///     outcome and hard-errors ("disagrees with numeric outcome") on any
///     disagreement.
///
/// So the prediction has information content only while `generation` and the
/// arrival/transaction counters here are advanced by this module's own rules.
/// Sourcing them from the effect stream would make the comparison
/// `actual == actual`.
///
/// Measured, 2026-08-04 (P3 barrier-unification worker): perturbing the hub's
/// generation advance (`begin_next_generation`, `+= 1` -> `+= 2`) makes the
/// cross-check fire loudly and precisely — "strict generation 1, numeric 2".
/// Re-running the same mutation with `expected` sourced from the hub's outcome
/// instead of predicted here silences the cross-check completely, and the
/// mutation then resurfaces as a *false `deadlock` verdict on correct kernels*
/// — a silent misdiagnosis in place of a diagnostic abort.
///
/// Per the §5 Q4 ruling, that is a failed positive control, so the two
/// machines stay. Deduplicating these counters and keeping the cross-check's
/// bite are mutually exclusive by construction; removing the redundancy is a
/// checker-semantics change needing its own golden-delta review, not a
/// refactor step.
#[derive(Default)]
pub struct StrictMbarrierProtocol {
    state: Mutex<ProtocolState>,
}

impl Clone for StrictMbarrierProtocol {
    fn clone(&self) -> Self {
        let state = self
            .state
            .lock()
            .expect("strict mbarrier mutex poisoned")
            .clone();
        Self {
            state: Mutex::new(state),
        }
    }
}

#[derive(Clone, Default)]
struct ProtocolState {
    slots: Arc<BTreeMap<PhysicalBarrierId, Arc<StrictSlot>>>,
    outstanding_tokens: Arc<BTreeMap<StrictMbarrierCompletionTokenId, CompletionTokenRecord>>,
    next_token_id: u64,
}

#[derive(Clone)]
struct StrictSlot {
    expected_arrivals: u64,
    generation: u64,
    current_phase: u64,
    init_fenced: bool,
    last_completed_phase: u64,
    last_completed_generation: Option<u64>,
    lifecycle: StrictMbarrierLifecycle,
    phase: StrictPhase,
    waiters: BTreeMap<(u64, usize), StrictMbarrierWaiter>,
    buffered_transactions: BTreeMap<u64, BufferedTransactions>,
    init_witness: SharedWitness,
    init_fence_witness: SharedWitness,
    completion_witness: SharedWitness,
    consumption_witness: SharedWitness,
}

#[derive(Clone, Default)]
struct StrictPhase {
    arrival_count: u64,
    arrived_warps: BTreeSet<usize>,
    expected_transactions: u64,
    completed_transactions: u64,
    /// Extra arrivals raised on this generation by `cp.async.mbarrier.arrive`
    /// without `.noinc`; discharged by that instruction's deferred arrive-on
    /// and dropped when the generation ends.
    pending_arrival_increments: u64,
}

#[derive(Clone, Default)]
struct BufferedTransactions {
    total: u64,
}

#[derive(Clone)]
struct CompletionTokenRecord {
    barrier_id: PhysicalBarrierId,
    generation: u64,
    issue_witness: SharedWitness,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StrictMbarrierSemanticState {
    slots: Box<[(PhysicalBarrierId, StrictMbarrierSlotSemanticState)]>,
    outstanding_tokens: Box<[(PhysicalBarrierId, u64)]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StrictMbarrierSlotSemanticState {
    expected_arrivals: u64,
    pending_arrival_increments: u64,
    generation: u64,
    current_phase: u64,
    init_fenced: bool,
    last_completed_phase: u64,
    last_completed_generation: Option<u64>,
    lifecycle: StrictMbarrierLifecycle,
    arrival_count: u64,
    arrived_warps: Box<[usize]>,
    expected_transactions: u64,
    completed_transactions: u64,
    waiters: Box<[(u64, usize, u64)]>,
    buffered_transactions: Box<[(u64, u64)]>,
}

impl StrictMbarrierProtocol {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn semantic_state(&self) -> StrictMbarrierSemanticState {
        let state = self.state.lock().expect("strict mbarrier mutex poisoned");
        let slots = state
            .slots
            .iter()
            .map(|(&barrier_id, slot)| {
                (
                    barrier_id,
                    StrictMbarrierSlotSemanticState {
                        expected_arrivals: slot.expected_arrivals,
                        pending_arrival_increments: slot.phase.pending_arrival_increments,
                        generation: slot.generation,
                        current_phase: slot.current_phase,
                        init_fenced: slot.init_fenced,
                        last_completed_phase: slot.last_completed_phase,
                        last_completed_generation: slot.last_completed_generation,
                        lifecycle: slot.lifecycle,
                        arrival_count: slot.phase.arrival_count,
                        arrived_warps: slot
                            .phase
                            .arrived_warps
                            .iter()
                            .copied()
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                        expected_transactions: slot.phase.expected_transactions,
                        completed_transactions: slot.phase.completed_transactions,
                        waiters: slot
                            .waiters
                            .values()
                            .map(|waiter| {
                                (waiter.generation, waiter.warp_id, waiter.requested_phase)
                            })
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                        buffered_transactions: slot
                            .buffered_transactions
                            .iter()
                            .map(|(&generation, buffered)| (generation, buffered.total))
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                    },
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut outstanding_tokens = state
            .outstanding_tokens
            .values()
            .map(|token| (token.barrier_id, token.generation))
            .collect::<Vec<_>>();
        outstanding_tokens.sort_unstable();
        StrictMbarrierSemanticState {
            slots,
            outstanding_tokens: outstanding_tokens.into_boxed_slice(),
        }
    }

    /// Initialize one physical barrier slot.
    pub fn invalidate_many(
        &self,
        ids: &[PhysicalBarrierId],
        witness: Option<DynamicOpId>,
    ) -> Result<(), StrictMbarrierError> {
        let witness = shared_witness(witness);
        let mut state = self.state.lock().expect("strict mbarrier mutex poisoned");
        let ids: BTreeSet<_> = ids.iter().copied().collect();
        for &id in &ids {
            let slot = state
                .slots
                .get(&id)
                .ok_or_else(|| StrictMbarrierError::Uninitialized {
                    barrier_id: id,
                    operation: StrictMbarrierOperation::Invalidate,
                    witness: witness.clone(),
                })?;
            if !slot.waiters.is_empty()
                || state
                    .outstanding_tokens
                    .values()
                    .any(|token| token.barrier_id == id)
            {
                return Err(StrictMbarrierError::InvalidateWithOutstandingWork {
                    barrier_id: id,
                    witness,
                });
            }
        }
        for id in ids {
            Arc::make_mut(&mut state.slots).remove(&id);
        }
        Ok(())
    }

    /// Initialize one physical barrier slot.
    pub fn init(
        &self,
        barrier_id: PhysicalBarrierId,
        expected_arrivals: u64,
        witness: Option<DynamicOpId>,
    ) -> Result<(), StrictMbarrierError> {
        self.init_many(&[barrier_id], expected_arrivals, witness)
    }

    /// Atomically initialize the lane targets resolved for one warp-level init
    /// operation. Repeated same-address lanes are idempotent and are collapsed;
    /// distinct lane-varying targets remain independently validated.
    pub fn init_many(
        &self,
        barrier_ids: &[PhysicalBarrierId],
        expected_arrivals: u64,
        witness: Option<DynamicOpId>,
    ) -> Result<(), StrictMbarrierError> {
        let witness = shared_witness(witness);
        let Some(&first_barrier_id) = barrier_ids.first() else {
            return Ok(());
        };
        if !(1..=MAX_MBARRIER_EXPECTED_ARRIVALS).contains(&expected_arrivals) {
            return Err(StrictMbarrierError::InvalidExpectedArrivals {
                barrier_id: first_barrier_id,
                expected: expected_arrivals,
                witness,
            });
        }
        let distinct_ids = barrier_ids.iter().copied().collect::<BTreeSet<_>>();
        let mut state = self.state.lock().expect("strict mbarrier mutex poisoned");
        let mut candidate = state.clone();
        for barrier_id in distinct_ids {
            candidate.init_one(barrier_id, expected_arrivals, witness.clone())?;
        }
        *state = candidate;
        Ok(())
    }

    /// Mark every already-initialized target covered by one
    /// `fence.mbarrier_init` operation.
    ///
    /// Targets that have not been initialized are deliberately ignored: a
    /// fence before init must not satisfy a later first use.
    pub fn mark_init_fenced_many(
        &self,
        barrier_ids: &[PhysicalBarrierId],
        witness: Option<DynamicOpId>,
    ) -> usize {
        let witness = shared_witness(witness);
        let mut state = self.state.lock().expect("strict mbarrier mutex poisoned");
        let slots = Arc::make_mut(&mut state.slots);
        let mut changed = 0;
        for barrier_id in barrier_ids.iter().copied() {
            let Some(slot) = slots.get_mut(&barrier_id) else {
                continue;
            };
            let slot = Arc::make_mut(slot);
            if slot.init_fenced {
                continue;
            }
            slot.init_fenced = true;
            slot.init_fence_witness = witness.clone();
            changed += 1;
        }
        changed
    }

    pub fn arrive(
        &self,
        barrier_id: PhysicalBarrierId,
        warp_id: usize,
        arrival_count: u64,
        witness: Option<DynamicOpId>,
    ) -> Result<StrictMbarrierEffect, StrictMbarrierError> {
        self.arrive_impl(
            barrier_id,
            warp_id,
            arrival_count,
            0,
            StrictMbarrierOperation::Arrive,
            shared_witness(witness),
            false,
        )
    }

    pub fn arrive_expect_tx(
        &self,
        barrier_id: PhysicalBarrierId,
        warp_id: usize,
        arrival_count: u64,
        expected_transactions: u64,
        witness: Option<DynamicOpId>,
    ) -> Result<StrictMbarrierEffect, StrictMbarrierError> {
        self.arrive_impl(
            barrier_id,
            warp_id,
            arrival_count,
            expected_transactions,
            StrictMbarrierOperation::ArriveExpectTx,
            shared_witness(witness),
            false,
        )
    }

    /// Add a transaction-byte expectation without consuming an arrival.
    ///
    /// A completed generation may advance only after a matching wait has
    /// consumed it. This is stricter than the numeric hub by design: the
    /// checker owns phase-reuse safety while independently reproducing the
    /// numeric counter transition for every legal operation.
    pub fn expect_tx(
        &self,
        barrier_id: PhysicalBarrierId,
        expected_transactions: u64,
        witness: Option<DynamicOpId>,
    ) -> Result<(), StrictMbarrierError> {
        let witness = shared_witness(witness);
        let operation = StrictMbarrierOperation::ExpectTx;
        let mut state = self.state.lock().expect("strict mbarrier mutex poisoned");
        let slot =
            state
                .slots
                .get(&barrier_id)
                .ok_or_else(|| StrictMbarrierError::Uninitialized {
                    barrier_id,
                    operation,
                    witness: witness.clone(),
                })?;
        if slot.lifecycle == StrictMbarrierLifecycle::CompletedUnconsumed {
            return Err(StrictMbarrierError::ExpectTxBeforeConsumption {
                barrier_id,
                generation: slot.generation,
                completion_witness: slot.completion_witness.clone(),
                witness,
            });
        }

        let mut candidate = slot.as_ref().clone();
        if candidate.lifecycle == StrictMbarrierLifecycle::Consumed {
            candidate.start_next_generation(barrier_id, operation, witness.as_ref())?;
        }
        debug_assert_eq!(candidate.lifecycle, StrictMbarrierLifecycle::Pending);
        candidate.phase.expected_transactions = candidate
            .phase
            .expected_transactions
            .checked_add(expected_transactions)
            .ok_or_else(|| StrictMbarrierError::CounterOverflow {
                barrier_id,
                generation: candidate.generation,
                counter: "expected transaction",
                witness: witness.clone(),
            })?;
        Arc::make_mut(&mut state.slots).insert(barrier_id, Arc::new(candidate));
        Ok(())
    }

    pub(crate) fn arrive_with_drop(
        &self,
        barrier_id: PhysicalBarrierId,
        warp_id: usize,
        arrival_count: u64,
        transactions: Option<u64>,
        drop: bool,
        witness: Option<DynamicOpId>,
    ) -> Result<StrictMbarrierEffect, StrictMbarrierError> {
        self.arrive_impl(
            barrier_id,
            warp_id,
            arrival_count,
            transactions.unwrap_or(0),
            if transactions.is_some() {
                StrictMbarrierOperation::ArriveExpectTx
            } else {
                StrictMbarrierOperation::Arrive
            },
            shared_witness(witness),
            drop,
        )
    }

    fn arrive_impl(
        &self,
        barrier_id: PhysicalBarrierId,
        warp_id: usize,
        arrival_count: u64,
        expected_transactions: u64,
        operation: StrictMbarrierOperation,
        witness: SharedWitness,
        drop: bool,
    ) -> Result<StrictMbarrierEffect, StrictMbarrierError> {
        if arrival_count == 0 {
            return Ok(StrictMbarrierEffect::default());
        }
        let mut state = self.state.lock().expect("strict mbarrier mutex poisoned");
        let slot =
            state
                .slots
                .get(&barrier_id)
                .ok_or_else(|| StrictMbarrierError::Uninitialized {
                    barrier_id,
                    operation,
                    witness: witness.clone(),
                })?;
        if slot.lifecycle == StrictMbarrierLifecycle::CompletedUnconsumed {
            return Err(StrictMbarrierError::ArriveBeforeConsumption {
                barrier_id,
                generation: slot.generation,
                completion_witness: slot.completion_witness.clone(),
                witness,
            });
        }

        let mut candidate = slot.as_ref().clone();
        if candidate.lifecycle == StrictMbarrierLifecycle::Consumed {
            candidate.start_next_generation(barrier_id, operation, witness.as_ref())?;
        }
        debug_assert_eq!(candidate.lifecycle, StrictMbarrierLifecycle::Pending);
        let generation = candidate.generation;
        if drop {
            candidate.expected_arrivals = candidate
                .expected_arrivals
                .checked_sub(arrival_count)
                .ok_or_else(|| StrictMbarrierError::ArrivalOverflow {
                    barrier_id,
                    generation,
                    expected: candidate.expected_arrivals,
                    completed: arrival_count,
                    witness: witness.clone(),
                })?;
            candidate.phase.pending_arrival_increments += arrival_count;
        }
        let completed_arrivals = candidate
            .phase
            .arrival_count
            .checked_add(arrival_count)
            .ok_or_else(|| StrictMbarrierError::CounterOverflow {
                barrier_id,
                generation,
                counter: "arrival",
                witness: witness.clone(),
            })?;
        if completed_arrivals > candidate.required_arrivals() {
            return Err(StrictMbarrierError::ArrivalOverflow {
                barrier_id,
                generation,
                expected: candidate.required_arrivals(),
                completed: completed_arrivals,
                witness,
            });
        }
        let total_expected_transactions = candidate
            .phase
            .expected_transactions
            .checked_add(expected_transactions)
            .ok_or_else(|| StrictMbarrierError::CounterOverflow {
                barrier_id,
                generation,
                counter: "expected transaction",
                witness: witness.clone(),
            })?;
        if completed_arrivals == candidate.required_arrivals()
            && candidate.phase.completed_transactions > total_expected_transactions
        {
            return Err(StrictMbarrierError::TransactionOverDelivery {
                barrier_id,
                generation,
                expected: total_expected_transactions,
                completed: candidate.phase.completed_transactions,
                witness,
            });
        }

        candidate.phase.arrival_count = completed_arrivals;
        candidate.phase.arrived_warps.insert(warp_id);
        candidate.phase.expected_transactions = total_expected_transactions;
        let effect = candidate.maybe_complete(witness);
        Arc::make_mut(&mut state.slots).insert(barrier_id, Arc::new(candidate));
        Ok(effect)
    }

    /// Raise the current generation's pending arrival count.
    ///
    /// This is the immediate half of `cp.async.mbarrier.arrive` without
    /// `.noinc`. The deferred arrive-on that discharges it is registered
    /// separately through [`Self::capture_completion`], so the strict model
    /// stays exactly one arrival short until that lane's `cp.async` work is
    /// complete.
    pub fn increase_pending_arrivals(
        &self,
        barrier_id: PhysicalBarrierId,
        increase: u64,
        witness: Option<DynamicOpId>,
    ) -> Result<(), StrictMbarrierError> {
        if increase == 0 {
            return Ok(());
        }
        let witness = shared_witness(witness);
        let operation = StrictMbarrierOperation::IncreasePendingArrivals;
        let mut state = self.state.lock().expect("strict mbarrier mutex poisoned");
        let slot =
            state
                .slots
                .get(&barrier_id)
                .ok_or_else(|| StrictMbarrierError::Uninitialized {
                    barrier_id,
                    operation,
                    witness: witness.clone(),
                })?;
        let mut candidate = slot.as_ref().clone();
        if candidate.lifecycle == StrictMbarrierLifecycle::Consumed {
            candidate.start_next_generation(barrier_id, operation, witness.as_ref())?;
        }
        let raised = candidate
            .phase
            .pending_arrival_increments
            .checked_add(increase)
            .ok_or_else(|| StrictMbarrierError::CounterOverflow {
                barrier_id,
                generation: candidate.generation,
                counter: "pending arrival",
                witness: witness.clone(),
            })?;
        candidate.phase.pending_arrival_increments = raised;
        Arc::make_mut(&mut state.slots).insert(barrier_id, Arc::new(candidate));
        Ok(())
    }

    /// Capture the generation to which a later async payload completion must
    /// contribute.
    pub fn capture_completion(
        &self,
        barrier_id: PhysicalBarrierId,
        issue_witness: Option<DynamicOpId>,
    ) -> Result<StrictMbarrierCompletionToken, StrictMbarrierError> {
        let issue_witness = shared_witness(issue_witness);
        let mut state = self.state.lock().expect("strict mbarrier mutex poisoned");
        let slot =
            state
                .slots
                .get(&barrier_id)
                .ok_or_else(|| StrictMbarrierError::Uninitialized {
                    barrier_id,
                    operation: StrictMbarrierOperation::CaptureCompletion,
                    witness: issue_witness.clone(),
                })?;
        let generation = match slot.lifecycle {
            StrictMbarrierLifecycle::Pending => slot.generation,
            StrictMbarrierLifecycle::CompletedUnconsumed | StrictMbarrierLifecycle::Consumed => {
                slot.generation.checked_add(1).ok_or_else(|| {
                    StrictMbarrierError::GenerationOverflow {
                        barrier_id,
                        generation: slot.generation,
                        operation: StrictMbarrierOperation::CaptureCompletion,
                        witness: issue_witness.clone(),
                    }
                })?
            }
            StrictMbarrierLifecycle::Uninitialized => unreachable!("stored slots are initialized"),
        };
        let token_id = StrictMbarrierCompletionTokenId(state.next_token_id);
        state.next_token_id = state.next_token_id.checked_add(1).ok_or_else(|| {
            StrictMbarrierError::CompletionTokenSpaceExhausted {
                barrier_id,
                witness: issue_witness.clone(),
            }
        })?;
        Arc::make_mut(&mut state.outstanding_tokens).insert(
            token_id,
            CompletionTokenRecord {
                barrier_id,
                generation,
                issue_witness: issue_witness.clone(),
            },
        );
        Ok(StrictMbarrierCompletionToken {
            id: token_id,
            barrier_id,
            generation,
            issue_witness,
        })
    }

    /// Apply transaction bytes using the generation captured at issue time.
    ///
    /// A completion for the immediately following generation may be buffered
    /// only after the current generation has completed. The token remains
    /// outstanding when validation fails, making the update transactional.
    pub fn complete_tx(
        &self,
        token: &StrictMbarrierCompletionToken,
        transactions: u64,
        witness: Option<DynamicOpId>,
    ) -> Result<StrictMbarrierEffect, StrictMbarrierError> {
        let witness = shared_witness(witness);
        let mut state = self.state.lock().expect("strict mbarrier mutex poisoned");
        let record = state
            .outstanding_tokens
            .get(&token.id)
            .filter(|record| {
                record.barrier_id == token.barrier_id && record.generation == token.generation
            })
            .cloned()
            .ok_or(StrictMbarrierError::UnknownCompletionToken { token_id: token.id })?;
        let slot = state.slots.get(&record.barrier_id).ok_or_else(|| {
            StrictMbarrierError::Uninitialized {
                barrier_id: record.barrier_id,
                operation: StrictMbarrierOperation::CompleteTx,
                witness: witness.clone(),
            }
        })?;
        let mut candidate = slot.as_ref().clone();
        let effect = if record.generation < candidate.generation {
            return Err(StrictMbarrierError::StaleCompletion {
                barrier_id: record.barrier_id,
                captured_generation: record.generation,
                current_generation: candidate.generation,
                issue_witness: record.issue_witness,
                witness,
            });
        } else if record.generation == candidate.generation {
            if candidate.lifecycle != StrictMbarrierLifecycle::Pending {
                return Err(StrictMbarrierError::CompletionAfterGenerationComplete {
                    barrier_id: record.barrier_id,
                    generation: record.generation,
                    lifecycle: candidate.lifecycle,
                    issue_witness: record.issue_witness,
                    witness,
                });
            }
            if transactions == 0 {
                StrictMbarrierEffect::default()
            } else {
                let completed_transactions = candidate
                    .phase
                    .completed_transactions
                    .checked_add(transactions)
                    .ok_or_else(|| StrictMbarrierError::CounterOverflow {
                        barrier_id: record.barrier_id,
                        generation: record.generation,
                        counter: "completed transaction",
                        witness: witness.clone(),
                    })?;
                if candidate.phase.arrival_count == candidate.required_arrivals()
                    && completed_transactions > candidate.phase.expected_transactions
                {
                    return Err(StrictMbarrierError::TransactionOverDelivery {
                        barrier_id: record.barrier_id,
                        generation: record.generation,
                        expected: candidate.phase.expected_transactions,
                        completed: completed_transactions,
                        witness,
                    });
                }
                candidate.phase.completed_transactions = completed_transactions;
                candidate.maybe_complete(witness.or(record.issue_witness.clone()))
            }
        } else {
            let next_generation = candidate.generation.checked_add(1).ok_or_else(|| {
                StrictMbarrierError::GenerationOverflow {
                    barrier_id: record.barrier_id,
                    generation: candidate.generation,
                    operation: StrictMbarrierOperation::CompleteTx,
                    witness: witness.clone(),
                }
            })?;
            if record.generation != next_generation
                || !matches!(
                    candidate.lifecycle,
                    StrictMbarrierLifecycle::CompletedUnconsumed
                        | StrictMbarrierLifecycle::Consumed
                )
            {
                return Err(StrictMbarrierError::FutureCompletionNotBufferable {
                    barrier_id: record.barrier_id,
                    captured_generation: record.generation,
                    current_generation: candidate.generation,
                    lifecycle: candidate.lifecycle,
                    issue_witness: record.issue_witness,
                    witness,
                });
            }
            if transactions != 0 {
                let buffer = candidate
                    .buffered_transactions
                    .entry(record.generation)
                    .or_default();
                buffer.total = buffer.total.checked_add(transactions).ok_or_else(|| {
                    StrictMbarrierError::CounterOverflow {
                        barrier_id: record.barrier_id,
                        generation: record.generation,
                        counter: "buffered transaction",
                        witness: witness.clone(),
                    }
                })?;
            }
            StrictMbarrierEffect::default()
        };
        Arc::make_mut(&mut state.slots).insert(record.barrier_id, Arc::new(candidate));
        Arc::make_mut(&mut state.outstanding_tokens).remove(&token.id);
        Ok(effect)
    }

    /// Test a parity and either consume an available completion or register a
    /// waiter for the concrete current/next generation.
    pub fn wait(
        &self,
        barrier_id: PhysicalBarrierId,
        requested_phase: u64,
        warp_id: usize,
        witness: Option<DynamicOpId>,
    ) -> Result<StrictMbarrierWaitOutcome, StrictMbarrierError> {
        let witness = shared_witness(witness);
        if requested_phase > 1 {
            return Err(StrictMbarrierError::InvalidPhase {
                barrier_id,
                phase: requested_phase,
                witness,
            });
        }
        let mut state = self.state.lock().expect("strict mbarrier mutex poisoned");
        let slot =
            state
                .slots
                .get(&barrier_id)
                .ok_or_else(|| StrictMbarrierError::Uninitialized {
                    barrier_id,
                    operation: StrictMbarrierOperation::Wait,
                    witness: witness.clone(),
                })?;
        if let Some(waiter) = slot
            .waiters
            .values()
            .find(|waiter| waiter.warp_id == warp_id)
        {
            return Err(StrictMbarrierError::DuplicateWaiter {
                barrier_id,
                warp_id,
                registered_generation: waiter.generation,
                witness,
            });
        }

        let mut candidate = slot.as_ref().clone();
        if requested_phase == candidate.last_completed_phase {
            let generation = candidate.last_completed_generation;
            let consumed_now = candidate.consume_completion(generation, witness);
            if consumed_now {
                Arc::make_mut(&mut state.slots).insert(barrier_id, Arc::new(candidate));
            }
            return Ok(StrictMbarrierWaitOutcome::Ready {
                generation,
                consumed_now,
            });
        }

        let generation = match candidate.lifecycle {
            StrictMbarrierLifecycle::Pending => candidate.generation,
            StrictMbarrierLifecycle::CompletedUnconsumed | StrictMbarrierLifecycle::Consumed => {
                candidate.generation.checked_add(1).ok_or_else(|| {
                    StrictMbarrierError::GenerationOverflow {
                        barrier_id,
                        generation: candidate.generation,
                        operation: StrictMbarrierOperation::Wait,
                        witness: witness.clone(),
                    }
                })?
            }
            StrictMbarrierLifecycle::Uninitialized => unreachable!("stored slots are initialized"),
        };
        candidate.waiters.insert(
            (generation, warp_id),
            StrictMbarrierWaiter {
                warp_id,
                generation,
                requested_phase,
                witness,
            },
        );
        Arc::make_mut(&mut state.slots).insert(barrier_id, Arc::new(candidate));
        Ok(StrictMbarrierWaitOutcome::Registered { generation })
    }

    /// Acquire the physical query's exact completion witness. The strict
    /// counter protocol still verifies completion; an older conditional
    /// completion must not consume a newer primary generation.
    pub fn acquire_completed(
        &self,
        barrier_id: PhysicalBarrierId,
        generation: Option<u64>,
        witness: Option<DynamicOpId>,
    ) -> Result<StrictMbarrierWaitOutcome, StrictMbarrierError> {
        let witness = shared_witness(witness);
        let mut state = self.state.lock().expect("strict mbarrier mutex poisoned");
        let slot =
            state
                .slots
                .get(&barrier_id)
                .ok_or_else(|| StrictMbarrierError::Uninitialized {
                    barrier_id,
                    operation: StrictMbarrierOperation::Wait,
                    witness: witness.clone(),
                })?;
        if let Some(generation) = generation {
            if Some(generation) > slot.last_completed_generation {
                return Err(StrictMbarrierError::AcquireBeforeCompletion {
                    barrier_id,
                    generation,
                    last_completed: slot.last_completed_generation,
                    witness,
                });
            }
        }
        let mut candidate = slot.as_ref().clone();
        let consumed_now = candidate.consume_completion(generation, witness);
        if consumed_now {
            Arc::make_mut(&mut state.slots).insert(barrier_id, Arc::new(candidate));
        }
        Ok(StrictMbarrierWaitOutcome::Ready {
            generation,
            consumed_now,
        })
    }

    pub fn snapshot(&self, barrier_id: PhysicalBarrierId) -> StrictMbarrierSnapshot {
        let state = self.state.lock().expect("strict mbarrier mutex poisoned");
        let outstanding_completion_tokens = state
            .outstanding_tokens
            .iter()
            .filter_map(|(&token_id, token)| {
                (token.barrier_id == barrier_id).then_some((token_id, token.generation))
            })
            .collect::<Vec<_>>();
        let Some(slot) = state.slots.get(&barrier_id) else {
            return StrictMbarrierSnapshot {
                lifecycle: StrictMbarrierLifecycle::Uninitialized,
                generation: None,
                current_phase: None,
                init_fenced: false,
                last_completed_phase: 1,
                last_completed_generation: None,
                expected_arrivals: None,
                arrival_count: 0,
                arrived_warps: Vec::new(),
                expected_transactions: 0,
                completed_transactions: 0,
                waiting_warps: Vec::new(),
                buffered_transactions: Vec::new(),
                outstanding_completion_tokens,
                init_witness: None,
                init_fence_witness: None,
                completion_witness: None,
                consumption_witness: None,
            };
        };
        StrictMbarrierSnapshot {
            lifecycle: slot.lifecycle,
            generation: Some(slot.generation),
            current_phase: Some(slot.current_phase),
            init_fenced: slot.init_fenced,
            last_completed_phase: slot.last_completed_phase,
            last_completed_generation: slot.last_completed_generation,
            expected_arrivals: Some(slot.required_arrivals()),
            arrival_count: slot.phase.arrival_count,
            arrived_warps: slot.phase.arrived_warps.iter().copied().collect(),
            expected_transactions: slot.phase.expected_transactions,
            completed_transactions: slot.phase.completed_transactions,
            waiting_warps: slot.waiters.values().cloned().collect(),
            buffered_transactions: slot
                .buffered_transactions
                .iter()
                .map(|(&generation, buffer)| (generation, buffer.total))
                .collect(),
            outstanding_completion_tokens,
            init_witness: slot.init_witness.clone(),
            init_fence_witness: slot.init_fence_witness.clone(),
            completion_witness: slot.completion_witness.clone(),
            consumption_witness: slot.consumption_witness.clone(),
        }
    }
}

impl ProtocolState {
    fn init_one(
        &mut self,
        barrier_id: PhysicalBarrierId,
        expected_arrivals: u64,
        witness: SharedWitness,
    ) -> Result<(), StrictMbarrierError> {
        if !(1..=MAX_MBARRIER_EXPECTED_ARRIVALS).contains(&expected_arrivals) {
            return Err(StrictMbarrierError::InvalidExpectedArrivals {
                barrier_id,
                expected: expected_arrivals,
                witness,
            });
        }
        let outstanding_completion_tokens = self
            .outstanding_tokens
            .values()
            .filter(|token| token.barrier_id == barrier_id)
            .count();
        let Some(existing) = self.slots.get(&barrier_id) else {
            Arc::make_mut(&mut self.slots).insert(
                barrier_id,
                Arc::new(StrictSlot::new(expected_arrivals, witness)),
            );
            return Ok(());
        };

        if existing.lifecycle == StrictMbarrierLifecycle::CompletedUnconsumed {
            return Err(StrictMbarrierError::ReinitializeBeforeConsumption {
                barrier_id,
                generation: existing.generation,
                completion_witness: existing.completion_witness.clone(),
                witness,
            });
        }
        let waiting_warps = existing
            .waiters
            .values()
            .map(StrictMbarrierWaiter::warp_id)
            .collect::<Vec<_>>();
        let active_pending = existing.lifecycle == StrictMbarrierLifecycle::Pending
            && (existing.phase.arrival_count != 0
                || existing.phase.expected_transactions != 0
                || existing.phase.completed_transactions != 0);
        if active_pending
            || !waiting_warps.is_empty()
            || outstanding_completion_tokens != 0
            || !existing.buffered_transactions.is_empty()
        {
            return Err(StrictMbarrierError::ReinitializeWhileActive {
                barrier_id,
                generation: existing.generation,
                lifecycle: existing.lifecycle,
                waiting_warps,
                outstanding_completion_tokens,
                buffered_generation_count: existing.buffered_transactions.len(),
                witness,
            });
        }
        Err(StrictMbarrierError::ReinitializeWithoutInvalidation {
            barrier_id,
            generation: existing.generation,
            lifecycle: existing.lifecycle,
            init_witness: existing.init_witness.clone(),
            witness,
        })
    }
}

impl StrictSlot {
    fn new(expected_arrivals: u64, witness: SharedWitness) -> Self {
        Self::reinitialized(0, expected_arrivals, witness)
    }

    /// Arrivals the current generation must receive before it can complete.
    const fn required_arrivals(&self) -> u64 {
        self.expected_arrivals
            .saturating_add(self.phase.pending_arrival_increments)
    }

    fn reinitialized(generation: u64, expected_arrivals: u64, witness: SharedWitness) -> Self {
        Self {
            expected_arrivals,
            generation,
            current_phase: 0,
            init_fenced: false,
            last_completed_phase: 1,
            last_completed_generation: None,
            lifecycle: StrictMbarrierLifecycle::Pending,
            phase: StrictPhase::default(),
            waiters: BTreeMap::new(),
            buffered_transactions: BTreeMap::new(),
            init_witness: witness,
            init_fence_witness: None,
            completion_witness: None,
            consumption_witness: None,
        }
    }

    fn start_next_generation(
        &mut self,
        barrier_id: PhysicalBarrierId,
        operation: StrictMbarrierOperation,
        witness: Option<&Arc<DynamicOpId>>,
    ) -> Result<(), StrictMbarrierError> {
        let next_generation = self.generation.checked_add(1).ok_or_else(|| {
            StrictMbarrierError::GenerationOverflow {
                barrier_id,
                generation: self.generation,
                operation,
                witness: witness.cloned(),
            }
        })?;
        let buffered_transactions = self
            .buffered_transactions
            .remove(&next_generation)
            .unwrap_or_default()
            .total;
        self.generation = next_generation;
        self.current_phase = self.last_completed_phase ^ 1;
        self.lifecycle = StrictMbarrierLifecycle::Pending;
        self.phase = StrictPhase {
            completed_transactions: buffered_transactions,
            ..StrictPhase::default()
        };
        self.completion_witness = None;
        self.consumption_witness = None;
        Ok(())
    }

    fn consume_completion(&mut self, generation: Option<u64>, witness: SharedWitness) -> bool {
        let consumed = self.lifecycle == StrictMbarrierLifecycle::CompletedUnconsumed
            && generation == Some(self.generation);
        if consumed {
            self.lifecycle = StrictMbarrierLifecycle::Consumed;
            self.consumption_witness = witness;
        }
        consumed
    }

    fn maybe_complete(&mut self, witness: SharedWitness) -> StrictMbarrierEffect {
        let count_met = self.phase.arrival_count == self.required_arrivals();
        if self.lifecycle != StrictMbarrierLifecycle::Pending
            || !count_met
            || self.phase.completed_transactions != self.phase.expected_transactions
        {
            return StrictMbarrierEffect::default();
        }
        self.lifecycle = StrictMbarrierLifecycle::CompletedUnconsumed;
        self.last_completed_phase = self.current_phase;
        self.last_completed_generation = Some(self.generation);
        self.completion_witness = witness;

        let ready_keys = self
            .waiters
            .iter()
            .filter_map(|(&key @ (generation, _), waiter)| {
                (generation == self.generation && waiter.requested_phase == self.current_phase)
                    .then_some(key)
            })
            .collect::<Vec<_>>();
        let ready_waiters = ready_keys
            .into_iter()
            .filter_map(|key| self.waiters.remove(&key))
            .collect::<Vec<_>>();
        let consumed_generation = if ready_waiters.is_empty() {
            None
        } else {
            self.lifecycle = StrictMbarrierLifecycle::Consumed;
            self.consumption_witness = ready_waiters
                .first()
                .and_then(|waiter| waiter.witness.clone());
            Some(self.generation)
        };
        StrictMbarrierEffect {
            completed_generation: Some(self.generation),
            consumed_generation,
            ready_waiters,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LoopFrame, StaticOpId};

    fn barrier_id() -> PhysicalBarrierId {
        PhysicalBarrierId::new(7, 16, 3)
    }

    fn second_barrier_id() -> PhysicalBarrierId {
        PhysicalBarrierId::new(7, 24, 3)
    }

    fn witness(warp_id: usize, sequence: u64) -> DynamicOpId {
        DynamicOpId::new(
            2,
            warp_id,
            sequence,
            StaticOpId::new(100 + sequence),
            [LoopFrame::new(StaticOpId::new(9), 4)],
        )
    }

    fn init(protocol: &StrictMbarrierProtocol, count: u64) {
        protocol
            .init(barrier_id(), count, Some(witness(0, 0)))
            .unwrap();
        assert_eq!(
            protocol.mark_init_fenced_many(&[barrier_id()], Some(witness(0, 1))),
            1
        );
    }

    #[test]
    fn uninitialized_slot_is_visible_and_active_operations_fail() {
        let protocol = StrictMbarrierProtocol::new();
        assert_eq!(
            protocol.snapshot(barrier_id()).lifecycle(),
            StrictMbarrierLifecycle::Uninitialized
        );
        assert!(matches!(
            protocol.arrive(barrier_id(), 0, 1, Some(witness(0, 1))),
            Err(StrictMbarrierError::Uninitialized {
                operation: StrictMbarrierOperation::Arrive,
                ..
            })
        ));
        assert!(matches!(
            protocol.wait(barrier_id(), 0, 0, Some(witness(0, 2))),
            Err(StrictMbarrierError::Uninitialized {
                operation: StrictMbarrierOperation::Wait,
                ..
            })
        ));
        assert!(matches!(
            protocol.capture_completion(barrier_id(), Some(witness(0, 3))),
            Err(StrictMbarrierError::Uninitialized {
                operation: StrictMbarrierOperation::CaptureCompletion,
                ..
            })
        ));
    }

    #[test]
    fn initialized_slot_defers_init_ordering_to_the_causality_checker() {
        let protocol = StrictMbarrierProtocol::new();
        protocol.init(barrier_id(), 1, Some(witness(0, 0))).unwrap();

        assert!(!protocol.snapshot(barrier_id()).init_fenced());
        let effect = protocol
            .arrive(barrier_id(), 0, 1, Some(witness(0, 1)))
            .unwrap();
        assert_eq!(effect.completed_generation(), Some(0));
        assert!(matches!(
            protocol.wait(barrier_id(), 0, 0, Some(witness(0, 2))),
            Ok(StrictMbarrierWaitOutcome::Ready {
                generation: Some(0),
                consumed_now: true,
            })
        ));
    }

    #[test]
    fn init_fence_only_covers_slots_initialized_before_it() {
        let protocol = StrictMbarrierProtocol::new();
        assert_eq!(
            protocol.mark_init_fenced_many(&[barrier_id()], Some(witness(0, 0))),
            0
        );
        protocol.init(barrier_id(), 1, Some(witness(0, 1))).unwrap();
        assert!(!protocol.snapshot(barrier_id()).init_fenced());

        assert_eq!(
            protocol.mark_init_fenced_many(&[barrier_id()], Some(witness(0, 2))),
            1
        );
        assert!(protocol.snapshot(barrier_id()).init_fenced());
        assert_eq!(
            protocol
                .snapshot(barrier_id())
                .init_fence_witness()
                .map(DynamicOpId::per_warp_sequence),
            Some(2)
        );
    }

    #[test]
    fn zero_lane_arrive_is_a_masked_noop() {
        let protocol = StrictMbarrierProtocol::new();
        assert_eq!(
            protocol
                .arrive(barrier_id(), 0, 0, Some(witness(0, 1)))
                .unwrap(),
            StrictMbarrierEffect::default()
        );
        assert_eq!(
            protocol.snapshot(barrier_id()).lifecycle(),
            StrictMbarrierLifecycle::Uninitialized
        );
    }

    #[test]
    fn initial_phase_one_wait_is_vacuously_ready_without_consuming_generation_zero() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        assert_eq!(
            protocol
                .wait(barrier_id(), 1, 0, Some(witness(0, 1)))
                .unwrap(),
            StrictMbarrierWaitOutcome::Ready {
                generation: None,
                consumed_now: false,
            }
        );
        let snapshot = protocol.snapshot(barrier_id());
        assert_eq!(snapshot.lifecycle(), StrictMbarrierLifecycle::Pending);
        assert_eq!(snapshot.generation(), Some(0));
        assert_eq!(snapshot.last_completed_phase(), 1);
    }

    #[test]
    fn completed_generation_must_be_consumed_before_next_arrive() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        let completion = protocol
            .arrive(barrier_id(), 1, 1, Some(witness(1, 1)))
            .unwrap();
        assert_eq!(completion.completed_generation(), Some(0));
        assert_eq!(completion.consumed_generation(), None);
        assert_eq!(
            protocol.snapshot(barrier_id()).lifecycle(),
            StrictMbarrierLifecycle::CompletedUnconsumed
        );

        let attempted = witness(1, 2);
        let error = protocol
            .arrive(barrier_id(), 1, 1, Some(attempted.clone()))
            .unwrap_err();
        assert!(matches!(
            &error,
            StrictMbarrierError::ArriveBeforeConsumption {
                generation: 0,
                witness: Some(found),
                ..
            } if found.as_ref() == &attempted
        ));

        assert_eq!(
            protocol
                .wait(barrier_id(), 0, 0, Some(witness(0, 3)))
                .unwrap(),
            StrictMbarrierWaitOutcome::Ready {
                generation: Some(0),
                consumed_now: true,
            }
        );
        assert_eq!(
            protocol.snapshot(barrier_id()).lifecycle(),
            StrictMbarrierLifecycle::Consumed
        );
        let second = protocol
            .arrive(barrier_id(), 1, 1, Some(witness(1, 4)))
            .unwrap();
        assert_eq!(second.completed_generation(), Some(1));
        let snapshot = protocol.snapshot(barrier_id());
        assert_eq!(snapshot.generation(), Some(1));
        assert_eq!(snapshot.current_phase(), Some(1));
    }

    #[test]
    fn standalone_expect_tx_arms_the_generation_without_consuming_an_arrival() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        let token = protocol
            .capture_completion(barrier_id(), Some(witness(1, 1)))
            .unwrap();

        protocol
            .expect_tx(barrier_id(), 64, Some(witness(0, 2)))
            .unwrap();
        let armed = protocol.snapshot(barrier_id());
        assert_eq!(armed.arrival_count(), 0);
        assert_eq!(armed.expected_transactions(), 64);
        assert_eq!(armed.lifecycle(), StrictMbarrierLifecycle::Pending);

        let arrival = protocol
            .arrive(barrier_id(), 0, 1, Some(witness(0, 3)))
            .unwrap();
        assert_eq!(arrival.completed_generation(), None);
        let completion = protocol
            .complete_tx(&token, 64, Some(witness(1, 4)))
            .unwrap();
        assert_eq!(completion.completed_generation(), Some(0));
    }

    #[test]
    fn standalone_expect_tx_cannot_advance_an_unconsumed_generation() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        protocol
            .arrive(barrier_id(), 0, 1, Some(witness(0, 1)))
            .unwrap();
        let before = protocol.snapshot(barrier_id());

        assert!(matches!(
            protocol.expect_tx(barrier_id(), 64, Some(witness(0, 2))),
            Err(StrictMbarrierError::ExpectTxBeforeConsumption { generation: 0, .. })
        ));
        assert_eq!(protocol.snapshot(barrier_id()), before);

        protocol
            .wait(barrier_id(), 0, 0, Some(witness(0, 3)))
            .unwrap();
        protocol
            .expect_tx(barrier_id(), 64, Some(witness(0, 4)))
            .unwrap();
        let next = protocol.snapshot(barrier_id());
        assert_eq!(next.generation(), Some(1));
        assert_eq!(next.arrival_count(), 0);
        assert_eq!(next.expected_transactions(), 64);
    }

    #[test]
    fn pre_registered_waiter_consumes_when_completion_wakes_it() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 2);
        assert_eq!(
            protocol
                .wait(barrier_id(), 0, 0, Some(witness(0, 1)))
                .unwrap(),
            StrictMbarrierWaitOutcome::Registered { generation: 0 }
        );
        protocol
            .arrive(barrier_id(), 1, 1, Some(witness(1, 2)))
            .unwrap();
        let effect = protocol
            .arrive(barrier_id(), 2, 1, Some(witness(2, 3)))
            .unwrap();
        assert_eq!(effect.completed_generation(), Some(0));
        assert_eq!(effect.consumed_generation(), Some(0));
        assert_eq!(effect.ready_waiters().len(), 1);
        assert_eq!(effect.ready_waiters()[0].warp_id(), 0);
        assert_eq!(
            protocol.snapshot(barrier_id()).lifecycle(),
            StrictMbarrierLifecycle::Consumed
        );
    }

    #[test]
    fn all_pre_registered_waiters_are_released_by_one_completion() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        for warp_id in [4, 2, 3] {
            protocol
                .wait(barrier_id(), 0, warp_id, Some(witness(warp_id, 1)))
                .unwrap();
        }
        let effect = protocol
            .arrive(barrier_id(), 1, 1, Some(witness(1, 2)))
            .unwrap();
        assert_eq!(
            effect
                .ready_waiters()
                .iter()
                .map(StrictMbarrierWaiter::warp_id)
                .collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
        assert!(protocol.snapshot(barrier_id()).waiting_warps().is_empty());
    }

    #[test]
    fn future_wait_does_not_consume_current_completion() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        protocol
            .arrive(barrier_id(), 1, 1, Some(witness(1, 1)))
            .unwrap();
        assert_eq!(
            protocol
                .wait(barrier_id(), 1, 2, Some(witness(2, 2)))
                .unwrap(),
            StrictMbarrierWaitOutcome::Registered { generation: 1 }
        );
        assert!(matches!(
            protocol.arrive(barrier_id(), 1, 1, Some(witness(1, 3))),
            Err(StrictMbarrierError::ArriveBeforeConsumption { .. })
        ));
        protocol
            .wait(barrier_id(), 0, 0, Some(witness(0, 4)))
            .unwrap();
        let effect = protocol
            .arrive(barrier_id(), 1, 1, Some(witness(1, 5)))
            .unwrap();
        assert_eq!(effect.consumed_generation(), Some(1));
        assert_eq!(effect.ready_waiters()[0].warp_id(), 2);
    }

    #[test]
    fn reinitialize_rejects_unconsumed_active_waiting_and_outstanding_states() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        protocol
            .arrive(barrier_id(), 0, 1, Some(witness(0, 1)))
            .unwrap();
        assert!(matches!(
            protocol.init(barrier_id(), 1, Some(witness(0, 2))),
            Err(StrictMbarrierError::ReinitializeBeforeConsumption { .. })
        ));

        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 2);
        protocol
            .arrive(barrier_id(), 0, 1, Some(witness(0, 1)))
            .unwrap();
        assert!(matches!(
            protocol.init(barrier_id(), 2, Some(witness(0, 2))),
            Err(StrictMbarrierError::ReinitializeWhileActive { .. })
        ));

        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        protocol
            .wait(barrier_id(), 0, 0, Some(witness(0, 1)))
            .unwrap();
        assert!(matches!(
            protocol.init(barrier_id(), 1, Some(witness(0, 2))),
            Err(StrictMbarrierError::ReinitializeWhileActive { .. })
        ));

        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        let _token = protocol
            .capture_completion(barrier_id(), Some(witness(0, 1)))
            .unwrap();
        assert!(matches!(
            protocol.init(barrier_id(), 1, Some(witness(0, 2))),
            Err(StrictMbarrierError::ReinitializeWhileActive {
                outstanding_completion_tokens: 1,
                ..
            })
        ));
    }

    #[test]
    fn same_lane_target_init_is_idempotent() {
        let protocol = StrictMbarrierProtocol::new();
        protocol
            .init_many(
                &[barrier_id(), second_barrier_id(), barrier_id()],
                1,
                Some(witness(0, 0)),
            )
            .unwrap();
        assert_eq!(protocol.snapshot(barrier_id()).generation(), Some(0));
        assert_eq!(protocol.snapshot(second_barrier_id()).generation(), Some(0));

        protocol.mark_init_fenced_many(&[barrier_id(), second_barrier_id()], Some(witness(0, 1)));
        protocol
            .arrive(second_barrier_id(), 1, 1, Some(witness(1, 2)))
            .unwrap();
        let first_before = protocol.snapshot(barrier_id());
        assert!(matches!(
            protocol.init_many(
                &[barrier_id(), second_barrier_id()],
                2,
                Some(witness(0, 4)),
            ),
            Err(StrictMbarrierError::ReinitializeWithoutInvalidation {
                barrier_id: failed,
                ..
            }) if failed == barrier_id()
        ));
        assert_eq!(protocol.snapshot(barrier_id()), first_before);
    }

    #[test]
    fn reinitialization_without_invalidation_rejects_pending_and_consumed_slots() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        assert!(matches!(
            protocol.init(barrier_id(), 2, Some(witness(0, 1))),
            Err(StrictMbarrierError::ReinitializeWithoutInvalidation {
                lifecycle: StrictMbarrierLifecycle::Pending,
                ..
            })
        ));

        protocol
            .arrive(barrier_id(), 0, 1, Some(witness(0, 2)))
            .unwrap();
        protocol
            .wait(barrier_id(), 0, 0, Some(witness(0, 3)))
            .unwrap();
        assert!(matches!(
            protocol.init(barrier_id(), 3, Some(witness(0, 4))),
            Err(StrictMbarrierError::ReinitializeWithoutInvalidation {
                lifecycle: StrictMbarrierLifecycle::Consumed,
                ..
            })
        ));
        let snapshot = protocol.snapshot(barrier_id());
        assert_eq!(snapshot.generation(), Some(0));
        assert_eq!(snapshot.expected_arrivals(), Some(1));
        assert_eq!(snapshot.lifecycle(), StrictMbarrierLifecycle::Consumed);
    }

    #[test]
    fn completion_token_credits_the_generation_captured_at_issue() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        let issue = witness(1, 1);
        let token = protocol
            .capture_completion(barrier_id(), Some(issue.clone()))
            .unwrap();
        assert_eq!(token.generation(), 0);
        assert_eq!(token.issue_witness(), Some(&issue));
        protocol
            .arrive_expect_tx(barrier_id(), 1, 1, 64, Some(witness(1, 2)))
            .unwrap();
        let effect = protocol
            .complete_tx(&token, 64, Some(witness(3, 3)))
            .unwrap();
        assert_eq!(effect.completed_generation(), Some(0));
        assert!(protocol
            .snapshot(barrier_id())
            .outstanding_completion_tokens()
            .is_empty());
        assert!(matches!(
            protocol.complete_tx(&token, 64, Some(witness(3, 4))),
            Err(StrictMbarrierError::UnknownCompletionToken { .. })
        ));
    }

    #[test]
    fn future_generation_completion_buffers_until_arrive_expect_tx() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        protocol
            .arrive(barrier_id(), 1, 1, Some(witness(1, 1)))
            .unwrap();
        let token = protocol
            .capture_completion(barrier_id(), Some(witness(1, 2)))
            .unwrap();
        assert_eq!(token.generation(), 1);
        protocol
            .complete_tx(&token, 128, Some(witness(3, 3)))
            .unwrap();
        assert_eq!(
            protocol.snapshot(barrier_id()).buffered_transactions(),
            &[(1, 128)]
        );
        protocol
            .wait(barrier_id(), 0, 0, Some(witness(0, 4)))
            .unwrap();
        protocol
            .wait(barrier_id(), 1, 2, Some(witness(2, 5)))
            .unwrap();
        let effect = protocol
            .arrive_expect_tx(barrier_id(), 1, 1, 128, Some(witness(1, 6)))
            .unwrap();
        assert_eq!(effect.completed_generation(), Some(1));
        assert_eq!(effect.consumed_generation(), Some(1));
        assert_eq!(effect.ready_waiters()[0].warp_id(), 2);
        assert!(protocol
            .snapshot(barrier_id())
            .buffered_transactions()
            .is_empty());
    }

    #[test]
    fn arrival_overflow_is_transactional() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 2);
        protocol
            .arrive_expect_tx(barrier_id(), 0, 1, 32, Some(witness(0, 1)))
            .unwrap();
        let before = protocol.snapshot(barrier_id());
        assert!(matches!(
            protocol.arrive(barrier_id(), 1, 2, Some(witness(1, 2))),
            Err(StrictMbarrierError::ArrivalOverflow {
                expected: 2,
                completed: 3,
                ..
            })
        ));
        assert_eq!(protocol.snapshot(barrier_id()), before);
    }

    #[test]
    fn expected_transaction_overflow_is_transactional() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 2);
        protocol
            .arrive_expect_tx(barrier_id(), 0, 1, u64::MAX, Some(witness(0, 1)))
            .unwrap();
        let before = protocol.snapshot(barrier_id());
        assert!(matches!(
            protocol.arrive_expect_tx(barrier_id(), 1, 1, 1, Some(witness(1, 2))),
            Err(StrictMbarrierError::CounterOverflow {
                counter: "expected transaction",
                ..
            })
        ));
        assert_eq!(protocol.snapshot(barrier_id()), before);
    }

    #[test]
    fn over_delivery_errors_only_after_arrivals_fix_the_expectation() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        let token = protocol
            .capture_completion(barrier_id(), Some(witness(1, 1)))
            .unwrap();
        protocol
            .complete_tx(&token, 128, Some(witness(3, 2)))
            .unwrap();
        let before = protocol.snapshot(barrier_id());
        assert_eq!(before.completed_transactions(), 128);
        assert!(matches!(
            protocol.arrive_expect_tx(barrier_id(), 1, 1, 64, Some(witness(1, 3))),
            Err(StrictMbarrierError::TransactionOverDelivery {
                expected: 64,
                completed: 128,
                ..
            })
        ));
        assert_eq!(protocol.snapshot(barrier_id()), before);
    }

    #[test]
    fn failed_completion_keeps_token_and_phase_unchanged() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        protocol
            .arrive_expect_tx(barrier_id(), 1, 1, 64, Some(witness(1, 1)))
            .unwrap();
        let token = protocol
            .capture_completion(barrier_id(), Some(witness(1, 2)))
            .unwrap();
        let before = protocol.snapshot(barrier_id());
        assert!(matches!(
            protocol.complete_tx(&token, 128, Some(witness(3, 3))),
            Err(StrictMbarrierError::TransactionOverDelivery {
                expected: 64,
                completed: 128,
                ..
            })
        ));
        assert_eq!(protocol.snapshot(barrier_id()), before);
    }

    #[test]
    fn undershoot_stays_pending_and_exact_match_completes() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        protocol
            .arrive_expect_tx(barrier_id(), 1, 1, 128, Some(witness(1, 1)))
            .unwrap();
        let first = protocol
            .capture_completion(barrier_id(), Some(witness(1, 2)))
            .unwrap();
        protocol
            .complete_tx(&first, 64, Some(witness(3, 3)))
            .unwrap();
        assert_eq!(
            protocol.snapshot(barrier_id()).lifecycle(),
            StrictMbarrierLifecycle::Pending
        );
        let second = protocol
            .capture_completion(barrier_id(), Some(witness(1, 4)))
            .unwrap();
        let effect = protocol
            .complete_tx(&second, 64, Some(witness(3, 5)))
            .unwrap();
        assert_eq!(effect.completed_generation(), Some(0));
    }

    #[test]
    fn completion_after_plain_arrival_completion_is_rejected() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        let token = protocol
            .capture_completion(barrier_id(), Some(witness(1, 1)))
            .unwrap();
        protocol
            .arrive(barrier_id(), 1, 1, Some(witness(1, 2)))
            .unwrap();
        assert!(matches!(
            protocol.complete_tx(&token, 64, Some(witness(3, 3))),
            Err(StrictMbarrierError::CompletionAfterGenerationComplete {
                generation: 0,
                lifecycle: StrictMbarrierLifecycle::CompletedUnconsumed,
                ..
            })
        ));
    }

    #[test]
    fn zero_byte_completion_consumes_only_the_token() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        let token = protocol
            .capture_completion(barrier_id(), Some(witness(1, 1)))
            .unwrap();
        let effect = protocol
            .complete_tx(&token, 0, Some(witness(3, 2)))
            .unwrap();
        assert_eq!(effect, StrictMbarrierEffect::default());
        let snapshot = protocol.snapshot(barrier_id());
        assert_eq!(snapshot.lifecycle(), StrictMbarrierLifecycle::Pending);
        assert!(snapshot.buffered_transactions().is_empty());
        assert!(snapshot.outstanding_completion_tokens().is_empty());

        protocol
            .arrive(barrier_id(), 1, 1, Some(witness(1, 3)))
            .unwrap();
        let future = protocol
            .capture_completion(barrier_id(), Some(witness(1, 4)))
            .unwrap();
        protocol
            .complete_tx(&future, 0, Some(witness(3, 5)))
            .unwrap();
        assert!(protocol
            .snapshot(barrier_id())
            .buffered_transactions()
            .is_empty());
    }

    #[test]
    fn stale_completion_is_rejected_after_generation_advances() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        let stale = protocol
            .capture_completion(barrier_id(), Some(witness(1, 1)))
            .unwrap();
        protocol
            .arrive(barrier_id(), 1, 1, Some(witness(1, 2)))
            .unwrap();
        protocol
            .wait(barrier_id(), 0, 0, Some(witness(0, 3)))
            .unwrap();
        protocol
            .arrive(barrier_id(), 1, 1, Some(witness(1, 4)))
            .unwrap();
        assert!(matches!(
            protocol.complete_tx(&stale, 64, Some(witness(3, 5))),
            Err(StrictMbarrierError::StaleCompletion {
                captured_generation: 0,
                current_generation: 1,
                ..
            })
        ));
    }

    #[test]
    fn already_consumed_generation_remains_ready_for_other_waiters() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        protocol
            .arrive(barrier_id(), 1, 1, Some(witness(1, 1)))
            .unwrap();
        assert_eq!(
            protocol
                .wait(barrier_id(), 0, 0, Some(witness(0, 2)))
                .unwrap(),
            StrictMbarrierWaitOutcome::Ready {
                generation: Some(0),
                consumed_now: true,
            }
        );
        assert_eq!(
            protocol
                .wait(barrier_id(), 0, 2, Some(witness(2, 3)))
                .unwrap(),
            StrictMbarrierWaitOutcome::Ready {
                generation: Some(0),
                consumed_now: false,
            }
        );
    }

    #[test]
    fn transaction_completion_wakes_and_consumes_pre_registered_waiter() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        protocol
            .wait(barrier_id(), 0, 0, Some(witness(0, 1)))
            .unwrap();
        protocol
            .arrive_expect_tx(barrier_id(), 1, 1, 64, Some(witness(1, 2)))
            .unwrap();
        let token = protocol
            .capture_completion(barrier_id(), Some(witness(1, 3)))
            .unwrap();
        let effect = protocol
            .complete_tx(&token, 64, Some(witness(3, 4)))
            .unwrap();
        assert_eq!(effect.completed_generation(), Some(0));
        assert_eq!(effect.consumed_generation(), Some(0));
        assert_eq!(effect.ready_waiters()[0].warp_id(), 0);
    }

    #[test]
    fn duplicate_waiter_is_rejected_without_mutation() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        protocol
            .wait(barrier_id(), 0, 4, Some(witness(4, 1)))
            .unwrap();
        let before = protocol.snapshot(barrier_id());
        assert!(matches!(
            protocol.wait(barrier_id(), 0, 4, Some(witness(4, 2))),
            Err(StrictMbarrierError::DuplicateWaiter {
                warp_id: 4,
                registered_generation: 0,
                ..
            })
        ));
        assert_eq!(protocol.snapshot(barrier_id()), before);
    }

    #[test]
    fn resolved_acquire_validates_completion_without_consuming_a_newer_phase() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        assert!(matches!(
            protocol.acquire_completed(barrier_id(), Some(0), None),
            Err(StrictMbarrierError::AcquireBeforeCompletion { .. })
        ));
        for generation in 0..3 {
            protocol.arrive(barrier_id(), 0, 1, None).unwrap();
            if generation > 0 {
                let before = protocol.snapshot(barrier_id());
                assert_eq!(
                    protocol
                        .acquire_completed(barrier_id(), Some(0), None)
                        .unwrap(),
                    StrictMbarrierWaitOutcome::Ready {
                        generation: Some(0),
                        consumed_now: false
                    }
                );
                assert_eq!(protocol.snapshot(barrier_id()), before);
                assert!(matches!(
                    protocol.acquire_completed(barrier_id(), Some(generation + 1), None),
                    Err(StrictMbarrierError::AcquireBeforeCompletion { .. })
                ));
                assert_eq!(protocol.snapshot(barrier_id()), before);
            }
            assert_eq!(
                protocol
                    .acquire_completed(barrier_id(), Some(generation), None)
                    .unwrap(),
                StrictMbarrierWaitOutcome::Ready {
                    generation: Some(generation),
                    consumed_now: true
                }
            );
        }
    }

    #[test]
    fn invalid_phase_is_rejected() {
        let protocol = StrictMbarrierProtocol::new();
        init(&protocol, 1);
        assert!(matches!(
            protocol.wait(barrier_id(), 2, 0, Some(witness(0, 1))),
            Err(StrictMbarrierError::InvalidPhase { phase: 2, .. })
        ));
    }

    #[test]
    fn zero_expected_arrivals_is_rejected() {
        let protocol = StrictMbarrierProtocol::new();
        assert!(matches!(
            protocol.init(barrier_id(), 0, Some(witness(0, 0))),
            Err(StrictMbarrierError::InvalidExpectedArrivals { expected: 0, .. })
        ));
        assert_eq!(
            protocol.snapshot(barrier_id()).lifecycle(),
            StrictMbarrierLifecycle::Uninitialized
        );
    }

    #[test]
    fn source_witnesses_survive_completion_and_consumption() {
        let protocol = StrictMbarrierProtocol::new();
        let init_witness = witness(0, 0);
        let completion_witness = witness(1, 1);
        let consumption_witness = witness(2, 2);
        protocol
            .init(barrier_id(), 1, Some(init_witness.clone()))
            .unwrap();
        protocol.mark_init_fenced_many(&[barrier_id()], Some(witness(0, 1)));
        protocol
            .arrive(barrier_id(), 1, 1, Some(completion_witness.clone()))
            .unwrap();
        protocol
            .wait(barrier_id(), 0, 2, Some(consumption_witness.clone()))
            .unwrap();
        let snapshot = protocol.snapshot(barrier_id());
        assert_eq!(snapshot.init_witness(), Some(&init_witness));
        assert_eq!(snapshot.completion_witness(), Some(&completion_witness));
        assert_eq!(snapshot.consumption_witness(), Some(&consumption_witness));
    }
}
