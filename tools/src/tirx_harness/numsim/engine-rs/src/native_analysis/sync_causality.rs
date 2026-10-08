//! Shared native-analysis happens-before tracking for synchronization protocols.
//!
//! This module deliberately does not model protocol counters or scheduler
//! progress. It records only causal evidence. A canonical CPU replay may call
//! two operations in sequence, but that sequence is accepted here only when a
//! vector-clock edge carries the required happens-before relation.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::{DynamicOpId, PhysicalBarrierId, WarpMask};

/// Per-warp vector clock used by the synchronization causal checker.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SyncVectorClock {
    components: Arc<[u64]>,
}

impl SyncVectorClock {
    pub fn zero(warp_count: usize) -> Self {
        Self {
            components: Arc::from(vec![0; warp_count].into_boxed_slice()),
        }
    }

    pub fn warp_count(&self) -> usize {
        self.components.len()
    }

    pub fn component(&self, warp_id: usize) -> Option<u64> {
        self.components.get(warp_id).copied()
    }

    pub fn happens_before(&self, other: &Self) -> bool {
        let warp_count = self.warp_count().max(other.warp_count());
        (0..warp_count).all(|warp_id| {
            self.component(warp_id).unwrap_or(0) <= other.component(warp_id).unwrap_or(0)
        })
    }

    pub fn concurrent_with(&self, other: &Self) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }

    pub fn tick(&mut self, warp_id: usize) -> Result<(), SyncCausalityError> {
        let warp_count = self.warp_count();
        let component = Arc::make_mut(&mut self.components).get_mut(warp_id).ok_or(
            SyncCausalityError::InvalidWarp {
                warp_id,
                warp_count,
            },
        )?;
        *component = component
            .checked_add(1)
            .ok_or(SyncCausalityError::ClockOverflow { warp_id })?;
        Ok(())
    }

    pub fn merge(&mut self, other: &Self) -> Result<(), SyncCausalityError> {
        if self.warp_count() != other.warp_count() {
            return Err(SyncCausalityError::ClockDimensionMismatch {
                expected_warps: self.warp_count(),
                actual_warps: other.warp_count(),
            });
        }
        for (current, acquired) in Arc::make_mut(&mut self.components)
            .iter_mut()
            .zip(other.components.iter())
        {
            *current = (*current).max(*acquired);
        }
        Ok(())
    }

    fn ensure_warp_count(&mut self, warp_count: usize) {
        if self.warp_count() >= warp_count {
            return;
        }
        let mut components = self.components.to_vec();
        components.resize(warp_count, 0);
        self.components = Arc::from(components.into_boxed_slice());
    }
}

/// Clock payload transferred by an ordinary or named synchronization edge.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SyncClockPayload {
    clock: SyncVectorClock,
}

impl SyncClockPayload {
    pub fn from_clock(clock: SyncVectorClock) -> Self {
        Self { clock }
    }

    pub const fn clock(&self) -> &SyncVectorClock {
        &self.clock
    }

    pub fn merge(&mut self, other: &Self) -> Result<(), SyncCausalityError> {
        let warp_count = self.clock.warp_count().max(other.clock.warp_count());
        self.clock.ensure_warp_count(warp_count);
        let mut other_clock = other.clock.clone();
        other_clock.ensure_warp_count(warp_count);
        self.clock.merge(&other_clock)
    }

    fn ensure_warp_count(&mut self, warp_count: usize) {
        self.clock.ensure_warp_count(warp_count);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MbarrierCausalUse {
    Arrive,
    ExpectTx,
    CompletionIssue,
    Wait,
    Reinitialize,
}

impl fmt::Display for MbarrierCausalUse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Arrive => "mbarrier.arrive",
            Self::ExpectTx => "mbarrier.expect_tx",
            Self::CompletionIssue => "mbarrier completion issue",
            Self::Wait => "mbarrier.wait",
            Self::Reinitialize => "mbarrier reinitialize",
        })
    }
}

/// Causal state retained for one mbarrier generation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct MbarrierGenerationCausalState {
    release_payload: Option<SyncClockPayload>,
    consumption_clock: Option<SyncVectorClock>,
    consumption_operation: Option<DynamicOpId>,
    consumption_candidates: Vec<(SyncVectorClock, Option<DynamicOpId>)>,
}

impl MbarrierGenerationCausalState {
    pub const fn release_payload(&self) -> Option<&SyncClockPayload> {
        self.release_payload.as_ref()
    }

    pub const fn consumption_clock(&self) -> Option<&SyncVectorClock> {
        self.consumption_clock.as_ref()
    }
}

/// Causal state retained for one initialized physical mbarrier slot.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MbarrierCausalState {
    epoch: u64,
    init_warp_id: usize,
    init_lane_id: usize,
    init_clock: SyncVectorClock,
    init_fence_clock: Option<SyncVectorClock>,
    generations: BTreeMap<u64, MbarrierGenerationCausalState>,
    /// Derived from the physical barrier's completion outcome. This one
    /// generation can remain observable through arbitrarily many primary phases.
    conditional_completed_generation: Option<u64>,
    /// Generations below this were retired by the window
    /// (`RETAINED_MBARRIER_GENERATIONS`); a lookup below it reports
    /// `ReleasePayloadRetired` instead of a protocol violation.
    retired_below: u64,
}

impl MbarrierCausalState {
    /// Bound primary history while retaining the conditional completion that
    /// the physical owner still exposes. At most one extra generation survives.
    fn retire_old_generations(&mut self) {
        let Some(&newest) = self.generations.keys().next_back() else {
            return;
        };
        let floor = newest.saturating_sub(RETAINED_MBARRIER_GENERATIONS - 1);
        if floor > self.retired_below {
            let pin = self.conditional_completed_generation;
            self.generations
                .retain(|&generation, _| generation >= floor || Some(generation) == pin);
            self.retired_below = floor;
        }
    }

    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    pub const fn init_clock(&self) -> &SyncVectorClock {
        &self.init_clock
    }

    pub const fn init_fence_clock(&self) -> Option<&SyncVectorClock> {
        self.init_fence_clock.as_ref()
    }

    pub fn generation(&self, generation: u64) -> Option<&MbarrierGenerationCausalState> {
        self.generations.get(&generation)
    }

    pub fn latest_generation(&self) -> Option<u64> {
        self.generations.keys().next_back().copied()
    }
}

/// Stable identity of one captured async-completion issuer clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MbarrierCompletionCausalTokenId(u64);

impl MbarrierCompletionCausalTokenId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Token whose completion contributes the clock captured at issue time.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MbarrierCompletionCausalToken {
    id: MbarrierCompletionCausalTokenId,
    barrier_id: PhysicalBarrierId,
    epoch: u64,
    generation: u64,
    issuer_clock: SyncVectorClock,
}

impl MbarrierCompletionCausalToken {
    pub const fn id(&self) -> MbarrierCompletionCausalTokenId {
        self.id
    }

    pub const fn barrier_id(&self) -> PhysicalBarrierId {
        self.barrier_id
    }

    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn issuer_clock(&self) -> &SyncVectorClock {
        &self.issuer_clock
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SyncCausalityError {
    InvalidWarp {
        warp_id: usize,
        warp_count: usize,
    },
    ClockOverflow {
        warp_id: usize,
    },
    ClockDimensionMismatch {
        expected_warps: usize,
        actual_warps: usize,
    },
    DuplicateSyncParticipant {
        warp_id: usize,
    },
    BarrierAlreadyInitialized {
        barrier_id: PhysicalBarrierId,
    },
    BarrierUninitialized {
        barrier_id: PhysicalBarrierId,
        operation: MbarrierCausalUse,
    },
    InitNotHappensBeforeUse {
        barrier_id: PhysicalBarrierId,
        epoch: u64,
        generation: u64,
        operation: MbarrierCausalUse,
        init_clock: SyncVectorClock,
        use_clock: SyncVectorClock,
    },
    PriorGenerationMissing {
        barrier_id: PhysicalBarrierId,
        epoch: u64,
        prior_generation: u64,
        operation: MbarrierCausalUse,
    },
    PriorGenerationNotConsumed {
        barrier_id: PhysicalBarrierId,
        epoch: u64,
        prior_generation: u64,
        operation: MbarrierCausalUse,
    },
    PriorGenerationConsumptionNotHappensBefore {
        barrier_id: PhysicalBarrierId,
        epoch: u64,
        prior_generation: u64,
        next_generation: u64,
        operation: MbarrierCausalUse,
        consumption_operation: Option<DynamicOpId>,
        consumption_clock: SyncVectorClock,
        use_clock: SyncVectorClock,
    },
    MissingGenerationRelease {
        barrier_id: PhysicalBarrierId,
        epoch: u64,
        generation: u64,
    },
    /// A resume asked for a release payload older than the retained
    /// generation window.
    ReleasePayloadRetired {
        barrier: String,
        generation: u64,
    },
    GenerationAlreadyConsumed {
        barrier_id: PhysicalBarrierId,
        epoch: u64,
        generation: u64,
    },
    CompletionTokenSpaceExhausted,
    UnknownCompletionToken {
        token_id: MbarrierCompletionCausalTokenId,
    },
    StaleCompletionToken {
        token_id: MbarrierCompletionCausalTokenId,
        barrier_id: PhysicalBarrierId,
        token_epoch: u64,
        current_epoch: u64,
    },
    ReinitializeGenerationMismatch {
        barrier_id: PhysicalBarrierId,
        expected_generation: Option<u64>,
        actual_generation: u64,
    },
    ReinitializeWithOutstandingCompletions {
        barrier_id: PhysicalBarrierId,
        epoch: u64,
        outstanding: usize,
    },
    EpochOverflow {
        barrier_id: PhysicalBarrierId,
        epoch: u64,
    },
}

impl fmt::Display for SyncCausalityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWarp {
                warp_id,
                warp_count,
            } => write!(f, "warp {warp_id} is outside 0..{warp_count}"),
            Self::ClockOverflow { warp_id } => {
                write!(f, "vector-clock component for warp {warp_id} overflowed")
            }
            Self::ClockDimensionMismatch {
                expected_warps,
                actual_warps,
            } => write!(
                f,
                "vector-clock dimension mismatch: expected {expected_warps}, got {actual_warps}"
            ),
            Self::DuplicateSyncParticipant { warp_id } => {
                write!(f, "synchronization participant warp {warp_id} was repeated")
            }
            Self::BarrierAlreadyInitialized { barrier_id } => {
                write!(f, "physical mbarrier {barrier_id:?} is already initialized")
            }
            Self::BarrierUninitialized {
                barrier_id,
                operation,
            } => write!(
                f,
                "{operation} used uninitialized physical mbarrier {barrier_id:?}"
            ),
            Self::InitNotHappensBeforeUse {
                barrier_id,
                epoch,
                generation,
                operation,
                ..
            } => write!(
                f,
                "mbarrier.init does not happen before {operation} on {barrier_id:?} epoch {epoch} generation {generation}"
            ),
            Self::PriorGenerationMissing {
                barrier_id,
                epoch,
                prior_generation,
                operation,
            } => write!(
                f,
                "{operation} on {barrier_id:?} epoch {epoch} has no prior generation {prior_generation}"
            ),
            Self::PriorGenerationNotConsumed {
                barrier_id,
                epoch,
                prior_generation,
                operation,
            } => write!(
                f,
                "{operation} on {barrier_id:?} epoch {epoch} started before generation {prior_generation} was consumed"
            ),
            Self::PriorGenerationConsumptionNotHappensBefore {
                barrier_id,
                epoch,
                prior_generation,
                next_generation,
                operation,
                ..
            } => write!(
                f,
                "generation {prior_generation} consumption does not happen before {operation} on {barrier_id:?} epoch {epoch} generation {next_generation}"
            ),
            Self::MissingGenerationRelease {
                barrier_id,
                epoch,
                generation,
            } => write!(
                f,
                "mbarrier.wait on {barrier_id:?} epoch {epoch} generation {generation} has no release payload"
            ),
            Self::ReleasePayloadRetired {
                barrier,
                generation,
            } => write!(
                f,
                "barrier {barrier} generation {generation} release payload was retired by the generation window"
            ),
            Self::GenerationAlreadyConsumed {
                barrier_id,
                epoch,
                generation,
            } => write!(
                f,
                "mbarrier {barrier_id:?} epoch {epoch} generation {generation} was already consumed"
            ),
            Self::CompletionTokenSpaceExhausted => {
                f.write_str("mbarrier causal completion token ID space exhausted")
            }
            Self::UnknownCompletionToken { token_id } => {
                write!(f, "unknown mbarrier causal completion token {}", token_id.get())
            }
            Self::StaleCompletionToken {
                token_id,
                barrier_id,
                token_epoch,
                current_epoch,
            } => write!(
                f,
                "mbarrier causal completion token {} targets stale {barrier_id:?} epoch {token_epoch}; current epoch is {current_epoch}",
                token_id.get()
            ),
            Self::ReinitializeGenerationMismatch {
                barrier_id,
                expected_generation,
                actual_generation,
            } => write!(
                f,
                "mbarrier reinitialize on {barrier_id:?} named generation {actual_generation}, but latest generation is {expected_generation:?}"
            ),
            Self::ReinitializeWithOutstandingCompletions {
                barrier_id,
                epoch,
                outstanding,
            } => write!(
                f,
                "mbarrier {barrier_id:?} epoch {epoch} cannot be reinitialized with {outstanding} outstanding completion token(s)"
            ),
            Self::EpochOverflow { barrier_id, epoch } => write!(
                f,
                "mbarrier {barrier_id:?} causal epoch overflowed after {epoch}"
            ),
        }
    }
}

impl Error for SyncCausalityError {}

/// Launch-local causal tracker for synchronization protocol verification.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SyncCausalityTracker {
    warp_slots: BTreeMap<usize, usize>,
    warp_clocks: Vec<SyncVectorClock>,
    barriers: BTreeMap<PhysicalBarrierId, MbarrierCausalState>,
    completion_tokens: BTreeMap<MbarrierCompletionCausalTokenId, MbarrierCompletionCausalToken>,
    next_completion_token_id: u64,
}

impl SyncCausalityTracker {
    pub fn new(warp_count: usize) -> Self {
        let zero = SyncVectorClock::zero(warp_count);
        Self {
            warp_slots: (0..warp_count).map(|warp_id| (warp_id, warp_id)).collect(),
            warp_clocks: vec![zero; warp_count],
            barriers: BTreeMap::new(),
            completion_tokens: BTreeMap::new(),
            next_completion_token_id: 0,
        }
    }

    pub const fn warp_count(&self) -> usize {
        self.warp_clocks.len()
    }

    pub fn warp_clock(&self, warp_id: usize) -> Option<&SyncVectorClock> {
        self.warp_slots
            .get(&warp_id)
            .and_then(|&slot| self.warp_clocks.get(slot))
    }

    pub fn barrier_state(&self, barrier_id: PhysicalBarrierId) -> Option<&MbarrierCausalState> {
        self.barriers.get(&barrier_id)
    }

    /// Advance one ordinary program-order event for `warp_id`.
    pub fn program_tick(&mut self, warp_id: usize) -> Result<SyncVectorClock, SyncCausalityError> {
        self.ensure_warp(warp_id);
        let next = self.next_event_clock(warp_id)?;
        let slot = self.warp_slot(warp_id)?;
        self.warp_clocks[slot] = next.clone();
        Ok(next)
    }

    /// Preview the clock of a program-order event without committing it.
    ///
    /// A blocking wait records this clock when it first registers. Its later
    /// completion refines the same operation clock with the acquired release
    /// payload, while the tracker itself advances only once.
    pub fn preview_program_tick(
        &self,
        warp_id: usize,
    ) -> Result<SyncVectorClock, SyncCausalityError> {
        self.next_event_clock(warp_id)
    }

    /// Produce a directional synchronization release payload.
    pub fn sync_release(&mut self, warp_id: usize) -> Result<SyncClockPayload, SyncCausalityError> {
        Ok(SyncClockPayload::from_clock(self.program_tick(warp_id)?))
    }

    /// Acquire a directional synchronization payload on one warp.
    pub fn sync_acquire(
        &mut self,
        warp_id: usize,
        payload: &SyncClockPayload,
    ) -> Result<SyncVectorClock, SyncCausalityError> {
        self.ensure_warp(warp_id);
        let mut next = self.current_clock(warp_id)?.clone();
        let mut acquired = payload.clock().clone();
        acquired.ensure_warp_count(self.warp_count());
        next.merge(&acquired)?;
        let slot = self.warp_slot(warp_id)?;
        next.tick(slot)?;
        self.warp_clocks[slot] = next.clone();
        Ok(next)
    }

    /// Apply an all-to-all named/ordinary barrier edge to the participants.
    pub fn synchronize_warps(
        &mut self,
        warp_ids: &[usize],
    ) -> Result<SyncClockPayload, SyncCausalityError> {
        let mut seen = BTreeSet::new();
        for &warp_id in warp_ids {
            if !seen.insert(warp_id) {
                return Err(SyncCausalityError::DuplicateSyncParticipant { warp_id });
            }
            self.ensure_warp(warp_id);
        }
        seen.clear();
        let mut joined: Option<SyncClockPayload> = None;
        for &warp_id in warp_ids {
            if !seen.insert(warp_id) {
                return Err(SyncCausalityError::DuplicateSyncParticipant { warp_id });
            }
            let event_clock = self.next_event_clock(warp_id)?;
            let payload = SyncClockPayload::from_clock(event_clock);
            if let Some(joined) = &mut joined {
                joined.merge(&payload)?;
            } else {
                joined = Some(payload);
            }
        }
        let joined = joined.unwrap_or_else(|| {
            SyncClockPayload::from_clock(SyncVectorClock::zero(self.warp_count()))
        });
        for &warp_id in warp_ids {
            let slot = self.warp_slot(warp_id)?;
            self.warp_clocks[slot] = joined.clock().clone();
        }
        Ok(joined)
    }

    /// Publish the physical owner's conditional completion before merging an
    /// arrival/completion can retire old release payloads. This creates no HB edge.
    pub(crate) fn retain_conditional_completion(
        &mut self,
        barrier_id: PhysicalBarrierId,
        generation: Option<u64>,
    ) -> Result<(), SyncCausalityError> {
        let barrier =
            self.barriers
                .get_mut(&barrier_id)
                .ok_or(SyncCausalityError::BarrierUninitialized {
                    barrier_id,
                    operation: MbarrierCausalUse::Wait,
                })?;
        let previous = std::mem::replace(&mut barrier.conditional_completed_generation, generation);
        if previous != generation {
            if let Some(previous) = previous.filter(|&value| value < barrier.retired_below) {
                barrier.generations.remove(&previous);
            }
        }
        Ok(())
    }

    /// Forget both primary history and the conditional pin on invalidation.
    pub fn mbarrier_invalidate(&mut self, barrier_id: PhysicalBarrierId) {
        self.barriers.remove(&barrier_id);
    }

    /// Record the initialization event for a physical mbarrier slot.
    pub fn mbarrier_init(
        &mut self,
        barrier_id: PhysicalBarrierId,
        warp_id: usize,
        lane_id: usize,
    ) -> Result<SyncVectorClock, SyncCausalityError> {
        self.ensure_warp(warp_id);
        if self.barriers.contains_key(&barrier_id) {
            return Err(SyncCausalityError::BarrierAlreadyInitialized { barrier_id });
        }
        let init_clock = self.next_event_clock(warp_id)?;
        self.barriers.insert(
            barrier_id,
            MbarrierCausalState {
                epoch: 0,
                init_warp_id: warp_id,
                init_lane_id: lane_id,
                init_clock: init_clock.clone(),
                init_fence_clock: None,
                generations: BTreeMap::new(),
                conditional_completed_generation: None,
                retired_below: 0,
            },
        );
        let slot = self.warp_slot(warp_id)?;
        self.warp_clocks[slot] = init_clock.clone();
        Ok(init_clock)
    }

    /// Record one `fence.mbarrier_init` event and bind it to every still-live
    /// mbarrier epoch initialized earlier by one of the issuing threads.
    ///
    /// Returning the newly covered barrier IDs lets the strict protocol and
    /// fixed-program recorder reuse exactly the same fence classification.
    pub fn mbarrier_init_fence(
        &mut self,
        warp_id: usize,
        active_mask: WarpMask,
    ) -> Result<(SyncVectorClock, Vec<PhysicalBarrierId>), SyncCausalityError> {
        let fence_clock = self.program_tick(warp_id)?;
        let mut barrier_ids = Vec::new();
        for (&barrier_id, barrier) in &mut self.barriers {
            if barrier.init_warp_id != warp_id
                || !active_mask.contains(barrier.init_lane_id)
                || barrier.init_fence_clock.is_some()
            {
                continue;
            }
            debug_assert!(barrier.init_clock.happens_before(&fence_clock));
            barrier.init_fence_clock = Some(fence_clock.clone());
            barrier_ids.push(barrier_id);
        }
        Ok((fence_clock, barrier_ids))
    }

    /// Add one synchronous arrival's event clock to a generation release.
    pub fn mbarrier_arrive(
        &mut self,
        barrier_id: PhysicalBarrierId,
        generation: u64,
        warp_id: usize,
    ) -> Result<SyncVectorClock, SyncCausalityError> {
        self.ensure_warp(warp_id);
        let event_clock = self.next_event_clock(warp_id)?;
        let merged_release = {
            let barrier =
                self.barriers
                    .get(&barrier_id)
                    .ok_or(SyncCausalityError::BarrierUninitialized {
                        barrier_id,
                        operation: MbarrierCausalUse::Arrive,
                    })?;
            Self::validate_init_happens_before(
                barrier_id,
                barrier,
                generation,
                MbarrierCausalUse::Arrive,
                &event_clock,
            )?;
            Self::validate_prior_consumption(
                barrier_id,
                barrier,
                generation,
                MbarrierCausalUse::Arrive,
                &event_clock,
            )?;
            merged_release_payload(
                barrier
                    .generations
                    .get(&generation)
                    .and_then(MbarrierGenerationCausalState::release_payload),
                event_clock.clone(),
            )?
        };
        let barrier = self
            .barriers
            .get_mut(&barrier_id)
            .expect("validated mbarrier remains initialized");
        barrier
            .generations
            .entry(generation)
            .or_default()
            .release_payload = Some(merged_release);
        barrier.retire_old_generations();
        let slot = self.warp_slot(warp_id)?;
        self.warp_clocks[slot] = event_clock.clone();
        Ok(event_clock)
    }

    /// Order one expect-only counter mutation without publishing a release.
    pub fn mbarrier_expect_tx(
        &mut self,
        barrier_id: PhysicalBarrierId,
        generation: u64,
        warp_id: usize,
    ) -> Result<SyncVectorClock, SyncCausalityError> {
        self.ensure_warp(warp_id);
        let event_clock = self.next_event_clock(warp_id)?;
        let barrier =
            self.barriers
                .get(&barrier_id)
                .ok_or(SyncCausalityError::BarrierUninitialized {
                    barrier_id,
                    operation: MbarrierCausalUse::ExpectTx,
                })?;
        Self::validate_init_happens_before(
            barrier_id,
            barrier,
            generation,
            MbarrierCausalUse::ExpectTx,
            &event_clock,
        )?;
        Self::validate_prior_consumption(
            barrier_id,
            barrier,
            generation,
            MbarrierCausalUse::ExpectTx,
            &event_clock,
        )?;
        let slot = self.warp_slot(warp_id)?;
        self.warp_clocks[slot] = event_clock.clone();
        Ok(event_clock)
    }

    /// Capture an async completion's causal source at its issuing warp.
    pub fn mbarrier_completion_issue(
        &mut self,
        barrier_id: PhysicalBarrierId,
        generation: u64,
        warp_id: usize,
    ) -> Result<MbarrierCompletionCausalToken, SyncCausalityError> {
        self.ensure_warp(warp_id);
        let issuer_clock = self.next_event_clock(warp_id)?;
        let barrier =
            self.barriers
                .get(&barrier_id)
                .ok_or(SyncCausalityError::BarrierUninitialized {
                    barrier_id,
                    operation: MbarrierCausalUse::CompletionIssue,
                })?;
        Self::validate_init_happens_before(
            barrier_id,
            barrier,
            generation,
            MbarrierCausalUse::CompletionIssue,
            &issuer_clock,
        )?;
        let token_id = MbarrierCompletionCausalTokenId(self.next_completion_token_id);
        self.next_completion_token_id = self
            .next_completion_token_id
            .checked_add(1)
            .ok_or(SyncCausalityError::CompletionTokenSpaceExhausted)?;
        let token = MbarrierCompletionCausalToken {
            id: token_id,
            barrier_id,
            epoch: barrier.epoch,
            generation,
            issuer_clock: issuer_clock.clone(),
        };
        let slot = self.warp_slot(warp_id)?;
        self.warp_clocks[slot] = issuer_clock;
        self.completion_tokens.insert(token_id, token.clone());
        Ok(token)
    }

    /// Complete an async action using only its captured issuer clock.
    pub fn mbarrier_complete(
        &mut self,
        token: &MbarrierCompletionCausalToken,
    ) -> Result<(), SyncCausalityError> {
        let recorded = self
            .completion_tokens
            .get(&token.id)
            .filter(|recorded| {
                recorded.barrier_id == token.barrier_id
                    && recorded.epoch == token.epoch
                    && recorded.generation == token.generation
            })
            .cloned()
            .ok_or(SyncCausalityError::UnknownCompletionToken { token_id: token.id })?;
        let merged_release = {
            let barrier = self.barriers.get(&recorded.barrier_id).ok_or(
                SyncCausalityError::BarrierUninitialized {
                    barrier_id: recorded.barrier_id,
                    operation: MbarrierCausalUse::CompletionIssue,
                },
            )?;
            if barrier.epoch != recorded.epoch {
                return Err(SyncCausalityError::StaleCompletionToken {
                    token_id: recorded.id,
                    barrier_id: recorded.barrier_id,
                    token_epoch: recorded.epoch,
                    current_epoch: barrier.epoch,
                });
            }
            merged_release_payload(
                barrier
                    .generations
                    .get(&recorded.generation)
                    .and_then(MbarrierGenerationCausalState::release_payload),
                recorded.issuer_clock.clone(),
            )?
        };
        let barrier = self
            .barriers
            .get_mut(&recorded.barrier_id)
            .expect("validated completion mbarrier remains initialized");
        barrier
            .generations
            .entry(recorded.generation)
            .or_default()
            .release_payload = Some(merged_release);
        barrier.retire_old_generations();
        self.completion_tokens.remove(&recorded.id);
        Ok(())
    }

    /// Acquire a completed generation without claiming its consumption edge.
    pub fn mbarrier_wait_acquire(
        &mut self,
        barrier_id: PhysicalBarrierId,
        generation: u64,
        warp_id: usize,
    ) -> Result<SyncVectorClock, SyncCausalityError> {
        self.mbarrier_wait_impl(barrier_id, generation, warp_id, false, None)
    }

    /// Acquire a completed generation and retain this wait as an alternate
    /// causal consumer for phase-reuse validation.
    pub fn mbarrier_wait_acquire_at(
        &mut self,
        barrier_id: PhysicalBarrierId,
        generation: u64,
        warp_id: usize,
        operation: &DynamicOpId,
    ) -> Result<SyncVectorClock, SyncCausalityError> {
        self.mbarrier_wait_impl(
            barrier_id,
            generation,
            warp_id,
            false,
            Some(operation.clone()),
        )
    }

    /// Acquire a completed generation and record the clock that consumed it.
    pub fn mbarrier_wait_consume(
        &mut self,
        barrier_id: PhysicalBarrierId,
        generation: u64,
        warp_id: usize,
    ) -> Result<SyncVectorClock, SyncCausalityError> {
        self.mbarrier_wait_impl(barrier_id, generation, warp_id, true, None)
    }

    /// Consume a completed generation and retain the exact wait operation as
    /// the other endpoint of any later happens-before rejection.
    pub fn mbarrier_wait_consume_at(
        &mut self,
        barrier_id: PhysicalBarrierId,
        generation: u64,
        warp_id: usize,
        operation: &DynamicOpId,
    ) -> Result<SyncVectorClock, SyncCausalityError> {
        self.mbarrier_wait_impl(
            barrier_id,
            generation,
            warp_id,
            true,
            Some(operation.clone()),
        )
    }

    /// Start a fresh initialized epoch after the named generation was consumed.
    pub fn mbarrier_reinitialize(
        &mut self,
        barrier_id: PhysicalBarrierId,
        prior_generation: u64,
        warp_id: usize,
        lane_id: usize,
    ) -> Result<SyncVectorClock, SyncCausalityError> {
        self.ensure_warp(warp_id);
        let event_clock = self.next_event_clock(warp_id)?;
        let epoch = {
            let barrier =
                self.barriers
                    .get(&barrier_id)
                    .ok_or(SyncCausalityError::BarrierUninitialized {
                        barrier_id,
                        operation: MbarrierCausalUse::Reinitialize,
                    })?;
            if barrier.latest_generation() != Some(prior_generation) {
                return Err(SyncCausalityError::ReinitializeGenerationMismatch {
                    barrier_id,
                    expected_generation: barrier.latest_generation(),
                    actual_generation: prior_generation,
                });
            }
            Self::validate_init_happens_before(
                barrier_id,
                barrier,
                prior_generation,
                MbarrierCausalUse::Reinitialize,
                &event_clock,
            )?;
            Self::validate_consumption_clock(
                barrier_id,
                barrier,
                prior_generation,
                prior_generation.saturating_add(1),
                MbarrierCausalUse::Reinitialize,
                &event_clock,
            )?;
            let outstanding = self
                .completion_tokens
                .values()
                .filter(|token| token.barrier_id == barrier_id && token.epoch == barrier.epoch)
                .count();
            if outstanding != 0 {
                return Err(SyncCausalityError::ReinitializeWithOutstandingCompletions {
                    barrier_id,
                    epoch: barrier.epoch,
                    outstanding,
                });
            }
            barrier
                .epoch
                .checked_add(1)
                .ok_or(SyncCausalityError::EpochOverflow {
                    barrier_id,
                    epoch: barrier.epoch,
                })?
        };
        self.barriers.insert(
            barrier_id,
            MbarrierCausalState {
                epoch,
                init_warp_id: warp_id,
                init_lane_id: lane_id,
                init_clock: event_clock.clone(),
                init_fence_clock: None,
                generations: BTreeMap::new(),
                conditional_completed_generation: None,
                retired_below: 0,
            },
        );
        let slot = self.warp_slot(warp_id)?;
        self.warp_clocks[slot] = event_clock.clone();
        Ok(event_clock)
    }

    fn mbarrier_wait_impl(
        &mut self,
        barrier_id: PhysicalBarrierId,
        generation: u64,
        warp_id: usize,
        consume: bool,
        consumption_operation: Option<DynamicOpId>,
    ) -> Result<SyncVectorClock, SyncCausalityError> {
        self.ensure_warp(warp_id);
        let (barrier_epoch, release) = {
            let barrier =
                self.barriers
                    .get(&barrier_id)
                    .ok_or(SyncCausalityError::BarrierUninitialized {
                        barrier_id,
                        operation: MbarrierCausalUse::Wait,
                    })?;
            if generation < barrier.retired_below
                && Some(generation) != barrier.conditional_completed_generation
            {
                return Err(SyncCausalityError::ReleasePayloadRetired {
                    barrier: format!("physical {barrier_id:?}"),
                    generation,
                });
            }
            let release = barrier
                .generations
                .get(&generation)
                .and_then(MbarrierGenerationCausalState::release_payload)
                .cloned()
                .ok_or(SyncCausalityError::MissingGenerationRelease {
                    barrier_id,
                    epoch: barrier.epoch,
                    generation,
                })?;
            (barrier.epoch, release)
        };
        let mut event_clock = self.current_clock(warp_id)?.clone();
        let mut release_clock = release.clock().clone();
        release_clock.ensure_warp_count(self.warp_count());
        event_clock.merge(&release_clock)?;
        let slot = self.warp_slot(warp_id)?;
        event_clock.tick(slot)?;
        {
            let barrier = self
                .barriers
                .get(&barrier_id)
                .expect("release payload came from an initialized mbarrier");
            Self::validate_init_happens_before(
                barrier_id,
                barrier,
                generation,
                MbarrierCausalUse::Wait,
                &event_clock,
            )?;
            if consume
                && barrier
                    .generations
                    .get(&generation)
                    .expect("release payload came from this generation")
                    .consumption_clock
                    .is_some()
            {
                return Err(SyncCausalityError::GenerationAlreadyConsumed {
                    barrier_id,
                    epoch: barrier_epoch,
                    generation,
                });
            }
        }
        let generation_state = self
            .barriers
            .get_mut(&barrier_id)
            .expect("validated wait mbarrier remains initialized")
            .generations
            .get_mut(&generation)
            .expect("release payload came from this generation");
        if consume {
            generation_state.consumption_clock = Some(event_clock.clone());
            generation_state.consumption_operation = consumption_operation.clone();
        }
        generation_state
            .consumption_candidates
            .push((event_clock.clone(), consumption_operation));
        self.warp_clocks[slot] = event_clock.clone();
        Ok(event_clock)
    }

    fn current_clock(&self, warp_id: usize) -> Result<&SyncVectorClock, SyncCausalityError> {
        let slot = self.warp_slot(warp_id)?;
        self.warp_clocks
            .get(slot)
            .ok_or(SyncCausalityError::InvalidWarp {
                warp_id,
                warp_count: self.warp_count(),
            })
    }

    fn next_event_clock(&self, warp_id: usize) -> Result<SyncVectorClock, SyncCausalityError> {
        let mut next = self.current_clock(warp_id)?.clone();
        next.tick(self.warp_slot(warp_id)?)?;
        Ok(next)
    }

    fn warp_slot(&self, warp_id: usize) -> Result<usize, SyncCausalityError> {
        self.warp_slots
            .get(&warp_id)
            .copied()
            .ok_or(SyncCausalityError::InvalidWarp {
                warp_id,
                warp_count: self.warp_count(),
            })
    }

    fn ensure_warp(&mut self, warp_id: usize) {
        if self.warp_slots.contains_key(&warp_id) {
            return;
        }
        let slot = self.warp_clocks.len();
        let warp_count = slot + 1;
        for clock in &mut self.warp_clocks {
            clock.ensure_warp_count(warp_count);
        }
        for barrier in self.barriers.values_mut() {
            barrier.init_clock.ensure_warp_count(warp_count);
            if let Some(clock) = &mut barrier.init_fence_clock {
                clock.ensure_warp_count(warp_count);
            }
            for generation in barrier.generations.values_mut() {
                if let Some(payload) = &mut generation.release_payload {
                    payload.ensure_warp_count(warp_count);
                }
                if let Some(clock) = &mut generation.consumption_clock {
                    clock.ensure_warp_count(warp_count);
                }
                for (clock, _) in &mut generation.consumption_candidates {
                    clock.ensure_warp_count(warp_count);
                }
            }
        }
        for token in self.completion_tokens.values_mut() {
            token.issuer_clock.ensure_warp_count(warp_count);
        }
        self.warp_clocks.push(SyncVectorClock::zero(warp_count));
        self.warp_slots.insert(warp_id, slot);
    }

    fn validate_init_happens_before(
        barrier_id: PhysicalBarrierId,
        barrier: &MbarrierCausalState,
        generation: u64,
        operation: MbarrierCausalUse,
        use_clock: &SyncVectorClock,
    ) -> Result<(), SyncCausalityError> {
        if barrier.init_clock.happens_before(use_clock) {
            return Ok(());
        }
        Err(SyncCausalityError::InitNotHappensBeforeUse {
            barrier_id,
            epoch: barrier.epoch,
            generation,
            operation,
            init_clock: barrier.init_clock.clone(),
            use_clock: use_clock.clone(),
        })
    }

    fn validate_prior_consumption(
        barrier_id: PhysicalBarrierId,
        barrier: &MbarrierCausalState,
        generation: u64,
        operation: MbarrierCausalUse,
        use_clock: &SyncVectorClock,
    ) -> Result<(), SyncCausalityError> {
        let Some(prior_generation) = generation.checked_sub(1) else {
            return Ok(());
        };
        Self::validate_consumption_clock(
            barrier_id,
            barrier,
            prior_generation,
            generation,
            operation,
            use_clock,
        )
    }

    fn validate_consumption_clock(
        barrier_id: PhysicalBarrierId,
        barrier: &MbarrierCausalState,
        prior_generation: u64,
        next_generation: u64,
        operation: MbarrierCausalUse,
        use_clock: &SyncVectorClock,
    ) -> Result<(), SyncCausalityError> {
        if prior_generation < barrier.retired_below {
            // The prior generation was retired by the window: its consumption
            // ordering was validated while it was current.
            return Ok(());
        }
        let prior = barrier.generations.get(&prior_generation).ok_or(
            SyncCausalityError::PriorGenerationMissing {
                barrier_id,
                epoch: barrier.epoch,
                prior_generation,
                operation,
            },
        )?;
        if prior
            .consumption_candidates
            .iter()
            .any(|(candidate, _)| candidate.happens_before(use_clock))
        {
            return Ok(());
        }
        let consumption_clock = prior.consumption_clock.as_ref().ok_or(
            SyncCausalityError::PriorGenerationNotConsumed {
                barrier_id,
                epoch: barrier.epoch,
                prior_generation,
                operation,
            },
        )?;
        if consumption_clock.happens_before(use_clock) {
            return Ok(());
        }
        Err(
            SyncCausalityError::PriorGenerationConsumptionNotHappensBefore {
                barrier_id,
                epoch: barrier.epoch,
                prior_generation,
                next_generation,
                operation,
                consumption_operation: prior.consumption_operation.clone(),
                consumption_clock: consumption_clock.clone(),
                use_clock: use_clock.clone(),
            },
        )
    }
}

fn merged_release_payload(
    current: Option<&SyncClockPayload>,
    clock: SyncVectorClock,
) -> Result<SyncClockPayload, SyncCausalityError> {
    let incoming = SyncClockPayload::from_clock(clock);
    let Some(current) = current else {
        return Ok(incoming);
    };
    let mut merged = current.clone();
    merged.merge(&incoming)?;
    Ok(merged)
}

impl Default for SyncCausalityTracker {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Named and cluster barriers resume the generation they registered for (not
/// the latest completed one), so a warp woken but not yet scheduled can still
/// ask for a generation the barrier has moved past; their tables are small
/// (a few thousand entries per launch) and keep a wider window.
pub const RETAINED_NAMED_BARRIER_GENERATIONS: u64 = 64;
/// Newest generations kept per physical mbarrier in the causality tracker,
/// matching the racecheck release-payload window.
pub const RETAINED_MBARRIER_GENERATIONS: u64 = 8;

/// Retires the generations of a named or cluster barrier's payload table.
pub fn retire_named_barrier_generations<B: Ord + Copy, V>(
    table: &mut BTreeMap<(B, u64), V>,
    barrier: B,
    newest: u64,
) -> u64 {
    retire_barrier_generations_with(table, barrier, newest, RETAINED_NAMED_BARRIER_GENERATIONS)
}

/// Drops `barrier`'s generations below `newest - window`.
pub fn retire_barrier_generations_with<B: Ord + Copy, V>(
    table: &mut BTreeMap<(B, u64), V>,
    barrier: B,
    newest: u64,
    window: u64,
) -> u64 {
    retire_barrier_generations_except(table, barrier, newest, window, None)
}

/// Retire primary history without discarding the conditional completion that
/// the physical owner still exposes. The returned floor excludes this pin.
pub(crate) fn retire_barrier_generations_except<B: Ord + Copy, V>(
    table: &mut BTreeMap<(B, u64), V>,
    barrier: B,
    newest: u64,
    window: u64,
    pin: Option<u64>,
) -> u64 {
    if window == 0 {
        return 0;
    }
    let floor = newest.saturating_sub(window);
    if floor == 0 {
        return 0;
    }
    let stale: Vec<(B, u64)> = table
        .range((barrier, 0)..(barrier, floor))
        .filter(|((_, generation), _)| Some(*generation) != pin)
        .map(|(key, _)| *key)
        .collect();
    for key in stale {
        table.remove(&key);
    }
    floor
}

#[cfg(test)]
mod tests {
    use super::*;

    fn barrier_id() -> PhysicalBarrierId {
        PhysicalBarrierId::new(7, 16, 0)
    }

    fn initialized_pair() -> SyncCausalityTracker {
        let mut tracker = SyncCausalityTracker::new(2);
        tracker.mbarrier_init(barrier_id(), 0, 0).unwrap();
        tracker.mbarrier_init_fence(0, WarpMask::FULL).unwrap();
        tracker.synchronize_warps(&[0, 1]).unwrap();
        tracker
    }

    #[test]
    fn same_warp_program_order_publishes_mbarrier_init() {
        let mut tracker = SyncCausalityTracker::new(1);
        tracker.mbarrier_init(barrier_id(), 0, 0).unwrap();
        tracker.mbarrier_arrive(barrier_id(), 0, 0).unwrap();
    }

    #[test]
    fn cta_synchronization_publishes_mbarrier_init_to_other_warps() {
        let mut tracker = SyncCausalityTracker::new(2);
        tracker.mbarrier_init(barrier_id(), 0, 0).unwrap();

        assert!(matches!(
            tracker.mbarrier_arrive(barrier_id(), 0, 1),
            Err(SyncCausalityError::InitNotHappensBeforeUse {
                operation: MbarrierCausalUse::Arrive,
                ..
            })
        ));
        tracker.synchronize_warps(&[0, 1]).unwrap();
        tracker.mbarrier_arrive(barrier_id(), 0, 1).unwrap();
    }

    #[test]
    fn init_fence_covers_only_barriers_initialized_by_issuing_threads() {
        let mut tracker = SyncCausalityTracker::new(2);
        tracker.mbarrier_init(barrier_id(), 0, 3).unwrap();
        let (_, covered) = tracker
            .mbarrier_init_fence(1, WarpMask::from_lanes([3]).unwrap())
            .unwrap();
        assert!(covered.is_empty());
        let (_, covered) = tracker
            .mbarrier_init_fence(0, WarpMask::from_lanes([4]).unwrap())
            .unwrap();
        assert!(covered.is_empty());
        assert!(tracker
            .barrier_state(barrier_id())
            .unwrap()
            .init_fence_clock()
            .is_none());

        let (_, covered) = tracker
            .mbarrier_init_fence(0, WarpMask::from_lanes([3]).unwrap())
            .unwrap();
        assert_eq!(covered, [barrier_id()]);
        tracker.mbarrier_arrive(barrier_id(), 0, 0).unwrap();
    }

    #[test]
    fn preview_program_tick_is_the_uncommitted_next_event() {
        let mut tracker = SyncCausalityTracker::new(2);
        let before = tracker.warp_clock(1).unwrap().clone();
        let preview = tracker.preview_program_tick(1).unwrap();

        assert_eq!(tracker.warp_clock(1), Some(&before));
        assert_eq!(tracker.program_tick(1).unwrap(), preview);
    }

    #[test]
    fn canonical_cpu_order_does_not_order_next_generation_arrive() {
        let mut tracker = initialized_pair();
        tracker.mbarrier_arrive(barrier_id(), 0, 0).unwrap();
        let consumption_operation = DynamicOpId::new(0, 1, 0, crate::StaticOpId::new(42), []);
        tracker
            .mbarrier_wait_consume_at(barrier_id(), 0, 1, &consumption_operation)
            .unwrap();

        let error = tracker.mbarrier_arrive(barrier_id(), 1, 0).unwrap_err();
        assert!(matches!(
            &error,
            SyncCausalityError::PriorGenerationConsumptionNotHappensBefore {
                prior_generation: 0,
                next_generation: 1,
                operation: MbarrierCausalUse::Arrive,
                consumption_operation: Some(operation),
                ..
            } if operation == &consumption_operation
        ));
    }

    #[test]
    fn conditional_completion_retains_only_its_exact_release_beyond_the_window() {
        let mut tracker = initialized_pair();
        let mut conditional_release = None;
        let mut latest_release = None;
        for generation in 0..21 {
            let pin = (generation >= 1).then_some(1);
            tracker
                .retain_conditional_completion(barrier_id(), pin)
                .unwrap();
            let release = tracker
                .mbarrier_arrive(barrier_id(), generation, 0)
                .unwrap();
            if generation == 1 {
                conditional_release = Some(release.clone());
            }
            latest_release = Some(release);
            tracker
                .mbarrier_wait_consume(barrier_id(), generation, 0)
                .unwrap();
        }
        let retained = tracker.barrier_state(barrier_id()).unwrap();
        assert_eq!(
            retained.generations.len(),
            RETAINED_MBARRIER_GENERATIONS as usize + 1
        );
        assert!(retained.generation(1).is_some());
        assert!(retained.generation(2).is_none());
        let acquired = tracker.mbarrier_wait_acquire(barrier_id(), 1, 1).unwrap();
        assert!(conditional_release.unwrap().happens_before(&acquired));
        assert!(!latest_release.unwrap().happens_before(&acquired));
        tracker
            .retain_conditional_completion(barrier_id(), Some(20))
            .unwrap();
        assert!(matches!(
            tracker.mbarrier_wait_acquire(barrier_id(), 1, 1),
            Err(SyncCausalityError::ReleasePayloadRetired { generation: 1, .. })
        ));
        assert_eq!(
            tracker
                .barrier_state(barrier_id())
                .unwrap()
                .generations
                .len(),
            RETAINED_MBARRIER_GENERATIONS as usize
        );
        tracker.mbarrier_invalidate(barrier_id());
        tracker.mbarrier_init(barrier_id(), 0, 0).unwrap();
        assert_eq!(
            tracker
                .barrier_state(barrier_id())
                .unwrap()
                .conditional_completed_generation,
            None
        );
    }

    #[test]
    fn old_generations_retire_behind_the_window_and_report_as_retired() {
        let mut tracker = initialized_pair();
        for generation in 0..20 {
            tracker.mbarrier_arrive(barrier_id(), generation, 0).unwrap();
            tracker
                .mbarrier_wait_consume(barrier_id(), generation, 1)
                .unwrap();
            tracker.synchronize_warps(&[0, 1]).unwrap();
        }
        let state = tracker.barrier_state(barrier_id()).unwrap();
        let floor = 19 - (RETAINED_MBARRIER_GENERATIONS - 1);
        assert!(state.generation(19).is_some());
        assert!(state.generation(floor).is_some());
        assert!(state.generation(floor - 1).is_none());
        // A wait below the window names the retirement, not a protocol error.
        assert!(matches!(
            tracker.mbarrier_wait_acquire(barrier_id(), 5, 0),
            Err(SyncCausalityError::ReleasePayloadRetired { generation: 5, .. })
        ));
        // The newest generation still acquires normally.
        tracker.mbarrier_wait_acquire(barrier_id(), 19, 0).unwrap();
    }

    #[test]
    fn synchronization_return_path_orders_next_generation_arrive() {
        let mut tracker = initialized_pair();
        tracker.mbarrier_arrive(barrier_id(), 0, 0).unwrap();
        let consumed = tracker.mbarrier_wait_consume(barrier_id(), 0, 1).unwrap();

        tracker.synchronize_warps(&[0, 1]).unwrap();
        let next_arrive = tracker.mbarrier_arrive(barrier_id(), 1, 0).unwrap();
        assert!(consumed.happens_before(&next_arrive));
    }

    #[test]
    fn directional_sync_payload_can_return_consumption_to_producer() {
        let mut tracker = initialized_pair();
        tracker.mbarrier_arrive(barrier_id(), 0, 0).unwrap();
        let consumed = tracker.mbarrier_wait_consume(barrier_id(), 0, 1).unwrap();

        let returned = tracker.sync_release(1).unwrap();
        tracker.sync_acquire(0, &returned).unwrap();
        let next_arrive = tracker.mbarrier_arrive(barrier_id(), 1, 0).unwrap();
        assert!(consumed.happens_before(&next_arrive));
    }

    #[test]
    fn any_causally_ordered_wait_can_guard_collective_phase_reuse() {
        let mut tracker = SyncCausalityTracker::new(4);
        tracker.mbarrier_init(barrier_id(), 0, 0).unwrap();
        tracker.mbarrier_init_fence(0, WarpMask::FULL).unwrap();
        tracker.synchronize_warps(&[0, 1, 2, 3]).unwrap();
        tracker.mbarrier_arrive(barrier_id(), 0, 0).unwrap();

        let first = DynamicOpId::new(0, 2, 0, crate::StaticOpId::new(50), []);
        tracker
            .mbarrier_wait_consume_at(barrier_id(), 0, 2, &first)
            .unwrap();
        let paired = DynamicOpId::new(0, 3, 0, crate::StaticOpId::new(51), []);
        tracker
            .mbarrier_wait_acquire_at(barrier_id(), 0, 3, &paired)
            .unwrap();

        // The canonical first consumer (warp 2) is concurrent with producer
        // warp 1.  Pairwise synchronization returns the other waiter's credit
        // to that producer, which is sufficient to make generation reuse safe.
        tracker.synchronize_warps(&[1, 3]).unwrap();
        tracker.mbarrier_arrive(barrier_id(), 1, 1).unwrap();
    }

    #[test]
    fn init_must_happen_before_cross_warp_use() {
        let mut tracker = SyncCausalityTracker::new(2);
        tracker.mbarrier_init(barrier_id(), 0, 0).unwrap();
        tracker.mbarrier_init_fence(0, WarpMask::FULL).unwrap();

        let error = tracker.mbarrier_arrive(barrier_id(), 0, 1).unwrap_err();
        assert!(matches!(
            error,
            SyncCausalityError::InitNotHappensBeforeUse {
                operation: MbarrierCausalUse::Arrive,
                ..
            }
        ));

        tracker.synchronize_warps(&[0, 1]).unwrap();
        tracker.mbarrier_arrive(barrier_id(), 0, 1).unwrap();
    }

    #[test]
    fn completion_release_inherits_only_the_captured_issuer_clock() {
        let mut tracker = initialized_pair();
        let token = tracker
            .mbarrier_completion_issue(barrier_id(), 0, 1)
            .unwrap();
        let issuer_clock = token.issuer_clock().clone();

        let later_cpu_event = tracker.program_tick(0).unwrap();
        tracker.mbarrier_complete(&token).unwrap();

        let release_clock = tracker
            .barrier_state(barrier_id())
            .unwrap()
            .generation(0)
            .unwrap()
            .release_payload()
            .unwrap()
            .clock();
        assert_eq!(release_clock, &issuer_clock);
        assert!(release_clock.component(0).unwrap() < later_cpu_event.component(0).unwrap());

        let acquired = tracker.mbarrier_wait_consume(barrier_id(), 0, 0).unwrap();
        assert!(issuer_clock.happens_before(&acquired));
    }

    #[test]
    fn reinitialize_also_requires_consumption_happens_before() {
        let mut tracker = initialized_pair();
        tracker.mbarrier_arrive(barrier_id(), 0, 0).unwrap();
        tracker.mbarrier_wait_consume(barrier_id(), 0, 1).unwrap();

        let error = tracker
            .mbarrier_reinitialize(barrier_id(), 0, 0, 0)
            .unwrap_err();
        assert!(matches!(
            error,
            SyncCausalityError::PriorGenerationConsumptionNotHappensBefore {
                operation: MbarrierCausalUse::Reinitialize,
                ..
            }
        ));
    }

    #[test]
    fn synchronization_return_path_allows_reinitialize() {
        let mut tracker = initialized_pair();
        tracker.mbarrier_arrive(barrier_id(), 0, 0).unwrap();
        tracker.mbarrier_wait_consume(barrier_id(), 0, 1).unwrap();

        tracker.synchronize_warps(&[0, 1]).unwrap();
        tracker
            .mbarrier_reinitialize(barrier_id(), 0, 0, 0)
            .unwrap();

        let barrier = tracker.barrier_state(barrier_id()).unwrap();
        assert_eq!(barrier.epoch(), 1);
        assert_eq!(barrier.latest_generation(), None);
    }

    #[test]
    fn sparse_global_warp_ids_use_dense_clocks_and_resize_protocol_payloads() {
        let mut tracker = SyncCausalityTracker::default();
        tracker.mbarrier_init(barrier_id(), 100, 0).unwrap();
        tracker.mbarrier_init_fence(100, WarpMask::FULL).unwrap();
        tracker.synchronize_warps(&[100, 900]).unwrap();
        let token = tracker
            .mbarrier_completion_issue(barrier_id(), 0, 900)
            .unwrap();

        tracker.program_tick(5_000).unwrap();

        assert_eq!(tracker.warp_count(), 3);
        assert_eq!(tracker.warp_clock(100).unwrap().warp_count(), 3);
        assert_eq!(tracker.warp_clock(900).unwrap().warp_count(), 3);
        assert_eq!(tracker.warp_clock(5_000).unwrap().warp_count(), 3);
        assert_eq!(
            tracker
                .barrier_state(barrier_id())
                .unwrap()
                .init_clock()
                .warp_count(),
            3
        );

        // The caller's token predates the resize. Identity, rather than the
        // old clock allocation, selects the tracker-owned resized token.
        tracker.mbarrier_complete(&token).unwrap();
        assert_eq!(
            tracker
                .barrier_state(barrier_id())
                .unwrap()
                .generation(0)
                .unwrap()
                .release_payload()
                .unwrap()
                .clock()
                .warp_count(),
            3
        );
    }
}
