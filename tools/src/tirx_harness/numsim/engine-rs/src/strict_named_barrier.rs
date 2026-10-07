use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::{DynamicOpId, NamedBarrierId, OperationContext, WarpMask, WARP_SIZE};

type SharedWitness = Option<Arc<DynamicOpId>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StrictNamedBarrierOperation {
    Arrive,
    Sync,
}

impl fmt::Display for StrictNamedBarrierOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Arrive => "bar.arrive",
            Self::Sync => "bar.sync",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrictNamedBarrierWaiter {
    warp_id: usize,
    generation: u64,
    arrival_mask: WarpMask,
    witness: SharedWitness,
}

impl StrictNamedBarrierWaiter {
    pub const fn warp_id(&self) -> usize {
        self.warp_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn arrival_mask(&self) -> WarpMask {
        self.arrival_mask
    }

    pub fn witness(&self) -> Option<&DynamicOpId> {
        self.witness.as_deref()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrictNamedBarrierContribution {
    warp_id: usize,
    arrival_mask: WarpMask,
    operation: StrictNamedBarrierOperation,
    /// `Some(aligned)` for a blocking sync contribution; `None` for an arrive.
    sync_aligned: Option<bool>,
    witness: SharedWitness,
}

impl StrictNamedBarrierContribution {
    pub const fn arrival_mask(&self) -> WarpMask {
        self.arrival_mask
    }

}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StrictNamedBarrierOutcome {
    Arrived {
        generation: u64,
        completed: bool,
        ready_waiters: Box<[StrictNamedBarrierWaiter]>,
    },
    Ready {
        generation: u64,
        completed_now: bool,
        ready_waiters: Box<[StrictNamedBarrierWaiter]>,
    },
    Registered {
        generation: u64,
    },
}

impl StrictNamedBarrierOutcome {
    pub const fn generation(&self) -> u64 {
        match self {
            Self::Arrived { generation, .. }
            | Self::Ready { generation, .. }
            | Self::Registered { generation } => *generation,
        }
    }

    pub const fn is_ready(&self) -> bool {
        matches!(self, Self::Ready { .. })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrictNamedBarrierSnapshot {
    generation: Option<u64>,
    completed_through: Option<u64>,
    expected_arrivals: Option<u64>,
    arrival_count: u64,
    contributors: Box<[StrictNamedBarrierContribution]>,
    waiting_warps: Box<[StrictNamedBarrierWaiter]>,
    complete: bool,
}

impl StrictNamedBarrierSnapshot {
    pub const fn generation(&self) -> Option<u64> {
        self.generation
    }

    pub const fn completed_through(&self) -> Option<u64> {
        self.completed_through
    }

    pub const fn arrival_count(&self) -> u64 {
        self.arrival_count
    }

    pub fn contributors(&self) -> &[StrictNamedBarrierContribution] {
        &self.contributors
    }

    pub fn waiting_warps(&self) -> &[StrictNamedBarrierWaiter] {
        &self.waiting_warps
    }

    pub const fn is_complete(&self) -> bool {
        self.complete
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StrictNamedBarrierError {
    ElectSyncParticipation {
        barrier_id: NamedBarrierId,
        operation: StrictNamedBarrierOperation,
        entry_mask: WarpMask,
        arrival_mask: WarpMask,
        witness: SharedWitness,
    },
    InvalidExpectedArrivals {
        barrier_id: NamedBarrierId,
        expected_arrivals: u64,
        witness: SharedWitness,
    },
    InvalidArrivalCount {
        barrier_id: NamedBarrierId,
        arrival_count: u64,
        witness: SharedWitness,
    },
    ContractMismatch {
        barrier_id: NamedBarrierId,
        generation: u64,
        expected_arrivals: u64,
        observed_arrivals: u64,
        witness: SharedWitness,
    },
    DuplicateContribution {
        barrier_id: NamedBarrierId,
        generation: u64,
        warp_id: usize,
        overlap_mask: WarpMask,
        prior_witness: SharedWitness,
        witness: SharedWitness,
    },
    ArrivalOverflow {
        barrier_id: NamedBarrierId,
        generation: u64,
        expected_arrivals: u64,
        completed_arrivals: u64,
        witness: SharedWitness,
    },
    CounterOverflow {
        barrier_id: NamedBarrierId,
        generation: u64,
        witness: SharedWitness,
    },
    GenerationOverflow {
        barrier_id: NamedBarrierId,
        generation: u64,
        witness: SharedWitness,
    },
    ResumeWithoutRegistration {
        barrier_id: NamedBarrierId,
        generation: u64,
        warp_id: usize,
        arrival_mask: WarpMask,
        witness: SharedWitness,
    },
    AlignedSyncContractMismatch {
        barrier_id: NamedBarrierId,
        generation: u64,
        aligned_witness: SharedWitness,
        witness: SharedWitness,
    },
    FullCtaAlignedMissingParticipants {
        barrier_id: NamedBarrierId,
        generation: u64,
        missing_warps: Box<[usize]>,
        witness: SharedWitness,
    },
}

impl fmt::Display for StrictNamedBarrierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ElectSyncParticipation {
                barrier_id,
                operation,
                entry_mask,
                arrival_mask,
                ..
            } => write!(
                f,
                "{operation} on named barrier {barrier_id:?} executes under elect_sync-derived divergent control: entry {entry_mask:?}, participating {arrival_mask:?}"
            ),
            Self::InvalidExpectedArrivals {
                barrier_id,
                expected_arrivals,
                ..
            } => write!(
                f,
                "named barrier {barrier_id:?} expected arrival count must be positive, got {expected_arrivals}"
            ),
            Self::InvalidArrivalCount {
                barrier_id,
                arrival_count,
                ..
            } => write!(
                f,
                "named barrier {barrier_id:?} contribution must contain at least one active lane, got {arrival_count}"
            ),
            Self::ContractMismatch {
                barrier_id,
                generation,
                expected_arrivals,
                observed_arrivals,
                ..
            } => write!(
                f,
                "named barrier {barrier_id:?} generation {generation} expected {expected_arrivals} arrivals, observed contract {observed_arrivals}"
            ),
            Self::DuplicateContribution {
                barrier_id,
                generation,
                warp_id,
                overlap_mask,
                ..
            } => write!(
                f,
                "warp {warp_id} contributed overlapping lanes {overlap_mask:?} twice to named barrier {barrier_id:?} generation {generation}"
            ),
            Self::ArrivalOverflow {
                barrier_id,
                generation,
                expected_arrivals,
                completed_arrivals,
                ..
            } => write!(
                f,
                "named barrier {barrier_id:?} generation {generation} received {completed_arrivals} arrivals, expected {expected_arrivals}"
            ),
            Self::CounterOverflow {
                barrier_id,
                generation,
                ..
            } => write!(
                f,
                "named barrier {barrier_id:?} generation {generation} arrival counter overflowed"
            ),
            Self::GenerationOverflow {
                barrier_id,
                generation,
                ..
            } => write!(
                f,
                "named barrier {barrier_id:?} cannot advance beyond generation {generation}"
            ),
            Self::ResumeWithoutRegistration {
                barrier_id,
                generation,
                warp_id,
                arrival_mask,
                ..
            } => write!(
                f,
                "warp {warp_id} resumed named barrier {barrier_id:?} generation {generation} with unregistered lanes {arrival_mask:?}"
            ),
            Self::AlignedSyncContractMismatch {
                barrier_id,
                generation,
                aligned_witness,
                witness,
            } => write!(
                f,
                "named-barrier generation {generation} on {barrier_id:?} contains an aligned blocking sync, but not all blocking sync contributions come from the same aligned static TIR site: aligned anchor {}, conflicting contribution {}",
                display_witness(aligned_witness),
                display_witness(witness),
            ),
            Self::FullCtaAlignedMissingParticipants {
                barrier_id,
                generation,
                missing_warps,
                ..
            } => write!(
                f,
                "full-CTA aligned named-barrier sync on {barrier_id:?} generation {generation} is missing contributions from CTA warps {missing_warps:?}"
            ),
        }
    }
}

impl Error for StrictNamedBarrierError {}

impl StrictNamedBarrierError {
    pub fn witness(&self) -> Option<&DynamicOpId> {
        match self {
            Self::ElectSyncParticipation { witness, .. }
            | Self::InvalidExpectedArrivals { witness, .. }
            | Self::InvalidArrivalCount { witness, .. }
            | Self::ContractMismatch { witness, .. }
            | Self::ArrivalOverflow { witness, .. }
            | Self::CounterOverflow { witness, .. }
            | Self::GenerationOverflow { witness, .. }
            | Self::ResumeWithoutRegistration { witness, .. }
            | Self::AlignedSyncContractMismatch { witness, .. }
            | Self::FullCtaAlignedMissingParticipants { witness, .. } => witness.as_deref(),
            Self::DuplicateContribution {
                witness,
                prior_witness,
                ..
            } => witness.as_deref().or(prior_witness.as_deref()),
        }
    }
}

fn display_witness(witness: &SharedWitness) -> String {
    witness
        .as_deref()
        .map(ToString::to_string)
        .unwrap_or_else(|| "<unknown>".to_string())
}

/// Enforce the aligned named-barrier participation contract before runtime
/// mutation. The exact hardware uniformity requirement remains [VERIFY].
pub(crate) fn validate_named_barrier_control(
    operation: &OperationContext,
    barrier_id: NamedBarrierId,
    kind: StrictNamedBarrierOperation,
) -> Result<(), StrictNamedBarrierError> {
    let Some(entry_mask) = operation.control_provenance().elect_sync_entry_mask() else {
        return Ok(());
    };
    let arrival_mask = operation.active_mask();
    if arrival_mask == entry_mask {
        return Ok(());
    }
    Err(StrictNamedBarrierError::ElectSyncParticipation {
        barrier_id,
        operation: kind,
        entry_mask,
        arrival_mask,
        witness: Some(Arc::new(operation.id().clone())),
    })
}

#[derive(Clone, Debug)]
struct NamedBarrierSlot {
    generation: u64,
    completed_through: Option<u64>,
    expected_arrivals: u64,
    arrival_count: u64,
    contributors: Vec<StrictNamedBarrierContribution>,
    waiters: BTreeMap<(usize, u32), StrictNamedBarrierWaiter>,
    completed_waiters: BTreeMap<u64, BTreeSet<(usize, u32)>>,
    complete: bool,
}

impl NamedBarrierSlot {
    fn new(expected_arrivals: u64) -> Self {
        Self {
            generation: 0,
            completed_through: None,
            expected_arrivals,
            arrival_count: 0,
            contributors: Vec::new(),
            waiters: BTreeMap::new(),
            completed_waiters: BTreeMap::new(),
            complete: false,
        }
    }

    fn start_next(
        &mut self,
        barrier_id: NamedBarrierId,
        expected_arrivals: u64,
        witness: &SharedWitness,
    ) -> Result<(), StrictNamedBarrierError> {
        debug_assert!(self.complete);
        self.completed_through = Some(self.generation);
        if !self.waiters.is_empty() {
            self.completed_waiters
                .insert(self.generation, self.waiters.keys().copied().collect());
        }
        self.generation = self.generation.checked_add(1).ok_or_else(|| {
            StrictNamedBarrierError::GenerationOverflow {
                barrier_id,
                generation: self.generation,
                witness: witness.clone(),
            }
        })?;
        self.expected_arrivals = expected_arrivals;
        self.arrival_count = 0;
        self.contributors.clear();
        self.waiters.clear();
        self.complete = false;
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct StrictNamedBarrierState {
    slots: BTreeMap<NamedBarrierId, NamedBarrierSlot>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StrictNamedBarrierSemanticState {
    slots: Box<[(NamedBarrierId, StrictNamedBarrierSlotSemanticState)]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StrictNamedBarrierSlotSemanticState {
    generation: u64,
    completed_through: Option<u64>,
    expected_arrivals: u64,
    arrival_count: u64,
    contributors: Box<[(usize, u32, StrictNamedBarrierOperation, Option<bool>)]>,
    waiters: Box<[(usize, u64, u32)]>,
    completed_waiters: Box<[(u64, Box<[(usize, u32)]>)]>,
    complete: bool,
}

#[derive(Default)]
pub struct StrictNamedBarrierProtocol {
    /// Full-CTA thread count from the launch topology. `None` skips every
    /// full-CTA-contract check; it is used where no topology is available
    /// (bare unit tests) or where no named-barrier slot can exist anyway
    /// (a state-searched fixed-sync projection).
    cta_thread_count: Option<u64>,
    state: Mutex<StrictNamedBarrierState>,
}

impl StrictNamedBarrierProtocol {
    pub fn new(cta_thread_count: Option<u64>) -> Self {
        Self {
            cta_thread_count,
            state: Mutex::default(),
        }
    }

    pub(crate) fn semantic_state(&self) -> StrictNamedBarrierSemanticState {
        let state = self
            .state
            .lock()
            .expect("strict named barrier mutex poisoned");
        StrictNamedBarrierSemanticState {
            slots: state
                .slots
                .iter()
                .map(|(&barrier_id, slot)| {
                    let mut contributors = slot
                        .contributors
                        .iter()
                        .map(|contribution| {
                            (
                                contribution.warp_id,
                                contribution.arrival_mask.bits(),
                                contribution.operation,
                                contribution.sync_aligned,
                            )
                        })
                        .collect::<Vec<_>>();
                    contributors.sort_unstable();
                    (
                        barrier_id,
                        StrictNamedBarrierSlotSemanticState {
                            generation: slot.generation,
                            completed_through: slot.completed_through,
                            expected_arrivals: slot.expected_arrivals,
                            arrival_count: slot.arrival_count,
                            contributors: contributors.into_boxed_slice(),
                            waiters: slot
                                .waiters
                                .values()
                                .map(|waiter| {
                                    (
                                        waiter.warp_id,
                                        waiter.generation,
                                        waiter.arrival_mask.bits(),
                                    )
                                })
                                .collect::<Vec<_>>()
                                .into_boxed_slice(),
                            completed_waiters: slot
                                .completed_waiters
                                .iter()
                                .map(|(&generation, waiters)| {
                                    (
                                        generation,
                                        waiters
                                            .iter()
                                            .copied()
                                            .collect::<Vec<_>>()
                                            .into_boxed_slice(),
                                    )
                                })
                                .collect::<Vec<_>>()
                                .into_boxed_slice(),
                            complete: slot.complete,
                        },
                    )
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    pub fn arrive(
        &self,
        barrier_id: NamedBarrierId,
        expected_arrivals: u64,
        warp_id: usize,
        arrival_mask: WarpMask,
        witness: Option<DynamicOpId>,
    ) -> Result<StrictNamedBarrierOutcome, StrictNamedBarrierError> {
        self.contribute(
            barrier_id,
            expected_arrivals,
            warp_id,
            arrival_mask,
            StrictNamedBarrierOperation::Arrive,
            None,
            witness,
        )
    }

    pub fn sync(
        &self,
        barrier_id: NamedBarrierId,
        expected_arrivals: u64,
        warp_id: usize,
        arrival_mask: WarpMask,
        witness: Option<DynamicOpId>,
    ) -> Result<StrictNamedBarrierOutcome, StrictNamedBarrierError> {
        self.sync_with_alignment(
            barrier_id,
            expected_arrivals,
            warp_id,
            arrival_mask,
            true,
            witness,
        )
    }

    pub fn sync_with_alignment(
        &self,
        barrier_id: NamedBarrierId,
        expected_arrivals: u64,
        warp_id: usize,
        arrival_mask: WarpMask,
        aligned: bool,
        witness: Option<DynamicOpId>,
    ) -> Result<StrictNamedBarrierOutcome, StrictNamedBarrierError> {
        self.contribute(
            barrier_id,
            expected_arrivals,
            warp_id,
            arrival_mask,
            StrictNamedBarrierOperation::Sync,
            Some(aligned),
            witness,
        )
    }

    fn contribute(
        &self,
        barrier_id: NamedBarrierId,
        expected_arrivals: u64,
        warp_id: usize,
        arrival_mask: WarpMask,
        operation: StrictNamedBarrierOperation,
        sync_aligned: Option<bool>,
        witness: Option<DynamicOpId>,
    ) -> Result<StrictNamedBarrierOutcome, StrictNamedBarrierError> {
        let witness = witness.map(Arc::new);
        let arrival_count = arrival_mask.len() as u64;
        if expected_arrivals == 0 {
            return Err(StrictNamedBarrierError::InvalidExpectedArrivals {
                barrier_id,
                expected_arrivals,
                witness,
            });
        }
        if arrival_count == 0 {
            return Err(StrictNamedBarrierError::InvalidArrivalCount {
                barrier_id,
                arrival_count,
                witness,
            });
        }

        let mut state = self
            .state
            .lock()
            .expect("strict named barrier mutex poisoned");
        let mut slot = state
            .slots
            .get(&barrier_id)
            .cloned()
            .unwrap_or_else(|| NamedBarrierSlot::new(expected_arrivals));
        if slot.complete {
            slot.start_next(barrier_id, expected_arrivals, &witness)?;
        } else if slot.expected_arrivals != expected_arrivals {
            return Err(StrictNamedBarrierError::ContractMismatch {
                barrier_id,
                generation: slot.generation,
                expected_arrivals: slot.expected_arrivals,
                observed_arrivals: expected_arrivals,
                witness,
            });
        }
        if let Some(prior) = slot.contributors.iter().find(|prior| {
            prior.warp_id == warp_id
                && prior.operation == operation
                && !prior.arrival_mask.intersection(arrival_mask).is_empty()
        }) {
            return Err(StrictNamedBarrierError::DuplicateContribution {
                barrier_id,
                generation: slot.generation,
                warp_id,
                overlap_mask: prior.arrival_mask.intersection(arrival_mask),
                prior_witness: prior.witness.clone(),
                witness,
            });
        }
        let completed_arrivals =
            slot.arrival_count
                .checked_add(arrival_count)
                .ok_or_else(|| StrictNamedBarrierError::CounterOverflow {
                    barrier_id,
                    generation: slot.generation,
                    witness: witness.clone(),
                })?;
        if completed_arrivals > slot.expected_arrivals {
            return Err(StrictNamedBarrierError::ArrivalOverflow {
                barrier_id,
                generation: slot.generation,
                expected_arrivals: slot.expected_arrivals,
                completed_arrivals,
                witness,
            });
        }

        let generation = slot.generation;
        slot.arrival_count = completed_arrivals;
        slot.contributors.push(StrictNamedBarrierContribution {
            warp_id,
            arrival_mask,
            operation,
            sync_aligned,
            witness: witness.clone(),
        });
        if operation == StrictNamedBarrierOperation::Sync {
            slot.waiters.insert(
                (warp_id, arrival_mask.bits()),
                StrictNamedBarrierWaiter {
                    warp_id,
                    generation,
                    arrival_mask,
                    witness: witness.clone(),
                },
            );
        }
        let completed_now = completed_arrivals == slot.expected_arrivals;
        if completed_now {
            slot.complete = true;
            self.check_completed_aligned_sync_origins(barrier_id, &slot)?;
        }
        let ready_waiters = if completed_now {
            slot.waiters.values().cloned().collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let outcome = match operation {
            StrictNamedBarrierOperation::Arrive => StrictNamedBarrierOutcome::Arrived {
                generation,
                completed: slot.complete,
                ready_waiters: ready_waiters.into_boxed_slice(),
            },
            StrictNamedBarrierOperation::Sync if slot.complete => {
                StrictNamedBarrierOutcome::Ready {
                    generation,
                    completed_now,
                    ready_waiters: ready_waiters.into_boxed_slice(),
                }
            }
            StrictNamedBarrierOperation::Sync => {
                StrictNamedBarrierOutcome::Registered { generation }
            }
        };
        state.slots.insert(barrier_id, slot);
        Ok(outcome)
    }

    /// Check the aligned blocking-sync contract at generation completion.
    ///
    /// If a generation contains an aligned blocking sync, every blocking sync
    /// contribution must be aligned. Contributions from one warp must share a
    /// static origin (kernel index and source operation; runtime loop
    /// iterations of one site share one origin), while different warps may
    /// reach equivalent barriers through distinct inlined call sites. Arrive
    /// contributions are independent, and generations whose blocking syncs
    /// are all unaligned do not require one static origin.
    fn check_completed_aligned_sync_origins(
        &self,
        barrier_id: NamedBarrierId,
        slot: &NamedBarrierSlot,
    ) -> Result<(), StrictNamedBarrierError> {
        let Some(anchor) = slot
            .contributors
            .iter()
            .find(|contribution| contribution.sync_aligned == Some(true))
        else {
            return Ok(());
        };
        let mut aligned_origins_by_warp = std::collections::BTreeMap::new();
        for contribution in slot
            .contributors
            .iter()
            .filter(|contribution| contribution.sync_aligned.is_some())
        {
            if contribution.sync_aligned != Some(true) {
                return Err(StrictNamedBarrierError::AlignedSyncContractMismatch {
                    barrier_id,
                    generation: slot.generation,
                    aligned_witness: anchor.witness.clone(),
                    witness: contribution.witness.clone(),
                });
            }
            if let Some(previous) =
                aligned_origins_by_warp.insert(contribution.warp_id, contribution.witness.clone())
            {
                let same_origin = match (previous.as_deref(), contribution.witness.as_deref()) {
                    (Some(prior), Some(current)) => prior.same_static_instruction(current),
                    _ => true,
                };
                if !same_origin {
                    return Err(StrictNamedBarrierError::AlignedSyncContractMismatch {
                        barrier_id,
                        generation: slot.generation,
                        aligned_witness: previous,
                        witness: contribution.witness.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Terminal scan for full-CTA aligned sync generations left incomplete at
    /// exit. Gated on the full-CTA contract (expected arrivals equal to the
    /// launch CTA thread count, requiring a topology) with every contribution
    /// an aligned blocking sync; the expected warp set is derived from the
    /// contract as `expected_arrivals / 32` warps starting at the CTA base.
    pub fn full_cta_aligned_nonuniform_errors(&self) -> Vec<StrictNamedBarrierError> {
        let state = self
            .state
            .lock()
            .expect("strict named barrier mutex poisoned");
        state
            .slots
            .iter()
            .filter_map(|(&barrier_id, slot)| {
                if slot.complete {
                    return None;
                }
                if self.cta_thread_count != Some(slot.expected_arrivals) {
                    return None;
                }
                if slot.contributors.is_empty()
                    || !slot
                        .contributors
                        .iter()
                        .all(|contribution| contribution.sync_aligned == Some(true))
                    || slot.expected_arrivals % WARP_SIZE as u64 != 0
                {
                    return None;
                }
                let expected_warps =
                    usize::try_from(slot.expected_arrivals / WARP_SIZE as u64).ok()?;
                let first_warp = barrier_id.global_cta_id().checked_mul(expected_warps)?;
                let contributors = slot
                    .contributors
                    .iter()
                    .map(|contribution| contribution.warp_id)
                    .collect::<BTreeSet<_>>();
                let missing_warps = (first_warp..first_warp.checked_add(expected_warps)?)
                    .filter(|warp_id| !contributors.contains(warp_id))
                    .collect::<Vec<_>>();
                (!missing_warps.is_empty()).then(|| {
                    StrictNamedBarrierError::FullCtaAlignedMissingParticipants {
                        barrier_id,
                        generation: slot.generation,
                        missing_warps: missing_warps.into_boxed_slice(),
                        witness: slot.contributors[0].witness.clone(),
                    }
                })
            })
            .collect()
    }

    pub fn waiter_state(
        &self,
        barrier_id: NamedBarrierId,
        warp_id: usize,
        generation: u64,
    ) -> bool {
        let state = self
            .state
            .lock()
            .expect("strict named barrier mutex poisoned");
        state.slots.get(&barrier_id).is_some_and(|slot| {
            slot.complete
                && slot.generation == generation
                && slot
                    .waiters
                    .keys()
                    .any(|(waiter_warp, _)| *waiter_warp == warp_id)
                || slot
                    .completed_waiters
                    .get(&generation)
                    .is_some_and(|waiters| {
                        waiters
                            .iter()
                            .any(|(waiter_warp, _)| *waiter_warp == warp_id)
                    })
        })
    }

    pub fn waiter_state_exact(
        &self,
        barrier_id: NamedBarrierId,
        warp_id: usize,
        arrival_mask: WarpMask,
        generation: u64,
    ) -> bool {
        let state = self
            .state
            .lock()
            .expect("strict named barrier mutex poisoned");
        state.slots.get(&barrier_id).is_some_and(|slot| {
            slot.complete
                && slot.generation == generation
                && slot.waiters.contains_key(&(warp_id, arrival_mask.bits()))
                || slot
                    .completed_waiters
                    .get(&generation)
                    .is_some_and(|waiters| waiters.contains(&(warp_id, arrival_mask.bits())))
        })
    }

    pub fn resume(
        &self,
        barrier_id: NamedBarrierId,
        warp_id: usize,
        arrival_mask: WarpMask,
        generation: u64,
        witness: Option<DynamicOpId>,
    ) -> Result<(), StrictNamedBarrierError> {
        self.resume_with_alignment(barrier_id, warp_id, arrival_mask, generation, true, witness)
    }

    pub fn resume_with_alignment(
        &self,
        barrier_id: NamedBarrierId,
        warp_id: usize,
        arrival_mask: WarpMask,
        generation: u64,
        aligned: bool,
        witness: Option<DynamicOpId>,
    ) -> Result<(), StrictNamedBarrierError> {
        let witness = witness.map(Arc::new);
        let waiter = (warp_id, arrival_mask.bits());
        let mut state = self
            .state
            .lock()
            .expect("strict named barrier mutex poisoned");
        let Some(mut slot) = state.slots.get(&barrier_id).cloned() else {
            return Err(StrictNamedBarrierError::ResumeWithoutRegistration {
                barrier_id,
                generation,
                warp_id,
                arrival_mask,
                witness,
            });
        };
        let allow_partition = !aligned;
        let removed = if slot.complete && slot.generation == generation {
            remove_waiter_partition(&mut slot.waiters, waiter, allow_partition)
        } else if let Some(waiters) = slot.completed_waiters.get_mut(&generation) {
            let removed = remove_completed_waiter_partition(waiters, waiter, allow_partition);
            if waiters.is_empty() {
                slot.completed_waiters.remove(&generation);
            }
            removed
        } else {
            false
        };
        if !removed {
            return Err(StrictNamedBarrierError::ResumeWithoutRegistration {
                barrier_id,
                generation,
                warp_id,
                arrival_mask,
                witness,
            });
        }
        state.slots.insert(barrier_id, slot);
        Ok(())
    }

    pub fn snapshot(&self, barrier_id: NamedBarrierId) -> StrictNamedBarrierSnapshot {
        let state = self
            .state
            .lock()
            .expect("strict named barrier mutex poisoned");
        let Some(slot) = state.slots.get(&barrier_id) else {
            return StrictNamedBarrierSnapshot {
                generation: None,
                completed_through: None,
                expected_arrivals: None,
                arrival_count: 0,
                contributors: Box::new([]),
                waiting_warps: Box::new([]),
                complete: false,
            };
        };
        StrictNamedBarrierSnapshot {
            generation: Some(slot.generation),
            completed_through: slot.completed_through,
            expected_arrivals: Some(slot.expected_arrivals),
            arrival_count: slot.arrival_count,
            contributors: slot.contributors.clone().into_boxed_slice(),
            waiting_warps: slot
                .waiters
                .values()
                .cloned()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            complete: slot.complete,
        }
    }
}

fn waiter_partition(
    mut keys: impl Iterator<Item = (usize, u32)>,
    requested: (usize, u32),
    allow_partition: bool,
) -> Option<Vec<(usize, u32)>> {
    if !allow_partition {
        return keys.find(|key| *key == requested).map(|key| vec![key]);
    }

    let (warp_id, requested_mask) = requested;
    let mut union = 0_u32;
    let matching = keys
        .filter(|&(candidate_warp, candidate_mask)| {
            candidate_warp == warp_id && candidate_mask & !requested_mask == 0
        })
        .inspect(|&(_, candidate_mask)| union |= candidate_mask)
        .collect::<Vec<_>>();
    (!matching.is_empty() && union == requested_mask).then_some(matching)
}

fn remove_waiter_partition(
    waiters: &mut BTreeMap<(usize, u32), StrictNamedBarrierWaiter>,
    requested: (usize, u32),
    allow_partition: bool,
) -> bool {
    let Some(keys) = waiter_partition(waiters.keys().copied(), requested, allow_partition) else {
        return false;
    };
    for key in keys {
        waiters.remove(&key);
    }
    true
}

fn remove_completed_waiter_partition(
    waiters: &mut BTreeSet<(usize, u32)>,
    requested: (usize, u32),
    allow_partition: bool,
) -> bool {
    let Some(keys) = waiter_partition(waiters.iter().copied(), requested, allow_partition) else {
        return false;
    };
    for key in keys {
        waiters.remove(&key);
    }
    true
}

impl Clone for StrictNamedBarrierProtocol {
    fn clone(&self) -> Self {
        let state = self
            .state
            .lock()
            .expect("strict named barrier mutex poisoned")
            .clone();
        Self {
            cta_thread_count: self.cta_thread_count,
            state: Mutex::new(state),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::{DynamicOpId, LoopFrame, StaticOpId};

    use super::*;

    fn id(cta: usize, barrier: u32) -> NamedBarrierId {
        NamedBarrierId::new(cta, barrier)
    }

    fn witness(warp: usize, sequence: u64) -> DynamicOpId {
        DynamicOpId::new(0, warp, sequence, StaticOpId::new(100 + sequence), [])
    }

    fn origin_witness(
        warp: usize,
        sequence: u64,
        source: u64,
        loops: impl Into<std::sync::Arc<[LoopFrame]>>,
    ) -> DynamicOpId {
        DynamicOpId::new(0, warp, sequence, StaticOpId::new(source), loops)
    }

    fn lanes(lanes: impl IntoIterator<Item = usize>) -> WarpMask {
        WarpMask::from_lanes(lanes).unwrap()
    }

    #[test]
    fn sync_registers_then_completion_releases_exact_waiters() {
        let protocol = StrictNamedBarrierProtocol::new(None);
        assert_eq!(
            protocol.sync(id(0, 3), 64, 0, WarpMask::FULL, Some(witness(0, 0))),
            Ok(StrictNamedBarrierOutcome::Registered { generation: 0 })
        );
        let completed = protocol
            .sync(id(0, 3), 64, 1, WarpMask::FULL, Some(witness(1, 0)))
            .unwrap();
        let StrictNamedBarrierOutcome::Ready {
            generation,
            completed_now,
            ready_waiters,
        } = completed
        else {
            panic!("last participant must complete bar.sync")
        };
        assert_eq!(generation, 0);
        assert!(completed_now);
        assert_eq!(
            ready_waiters
                .iter()
                .map(StrictNamedBarrierWaiter::warp_id)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([0, 1])
        );
        assert!(protocol.waiter_state(id(0, 3), 0, 0));
        assert!(protocol.waiter_state(id(0, 3), 1, 0));
    }

    #[test]
    fn nonblocking_arrive_and_sync_share_one_generation() {
        let protocol = StrictNamedBarrierProtocol::new(None);
        assert!(matches!(
            protocol
                .arrive(id(0, 4), 64, 0, WarpMask::FULL, Some(witness(0, 0)))
                .unwrap(),
            StrictNamedBarrierOutcome::Arrived {
                generation: 0,
                completed: false,
                ..
            }
        ));
        assert!(matches!(
            protocol
                .sync(id(0, 4), 64, 1, WarpMask::FULL, Some(witness(1, 0)))
                .unwrap(),
            StrictNamedBarrierOutcome::Ready {
                generation: 0,
                completed_now: true,
                ..
            }
        ));
    }

    #[test]
    fn the_same_lanes_may_arrive_then_sync_in_one_generation() {
        let protocol = StrictNamedBarrierProtocol::new(None);
        assert!(matches!(
            protocol
                .arrive(id(0, 14), 64, 0, WarpMask::FULL, Some(witness(0, 0)))
                .unwrap(),
            StrictNamedBarrierOutcome::Arrived {
                generation: 0,
                completed: false,
                ..
            }
        ));
        assert!(matches!(
            protocol
                .sync(id(0, 14), 64, 0, WarpMask::FULL, Some(witness(0, 1)))
                .unwrap(),
            StrictNamedBarrierOutcome::Ready {
                generation: 0,
                completed_now: true,
                ..
            }
        ));
    }

    #[test]
    fn contract_duplicate_and_over_arrival_rejections_are_transactional() {
        let protocol = StrictNamedBarrierProtocol::new(None);
        protocol
            .sync(id(0, 5), 32, 0, lanes(0..16), Some(witness(0, 0)))
            .unwrap();
        assert!(matches!(
            protocol.sync(id(0, 5), 64, 1, WarpMask::FULL, Some(witness(1, 0))),
            Err(StrictNamedBarrierError::ContractMismatch { .. })
        ));
        assert!(matches!(
            protocol.sync(id(0, 5), 32, 0, lanes([0]), Some(witness(0, 1))),
            Err(StrictNamedBarrierError::DuplicateContribution { .. })
        ));
        assert!(matches!(
            protocol.sync(id(0, 5), 32, 1, WarpMask::FULL, Some(witness(1, 0))),
            Err(StrictNamedBarrierError::ArrivalOverflow {
                expected_arrivals: 32,
                completed_arrivals: 48,
                ..
            })
        ));
        let snapshot = protocol.snapshot(id(0, 5));
        assert_eq!(snapshot.arrival_count(), 16);
        assert_eq!(snapshot.contributors().len(), 1);
        assert!(!snapshot.is_complete());
    }

    #[test]
    fn disjoint_lane_masks_from_one_warp_are_distinct_contributions() {
        let protocol = StrictNamedBarrierProtocol::new(None);
        let first = lanes(0..16);
        let second = lanes(16..32);

        assert!(matches!(
            protocol.arrive(id(0, 8), 32, 0, first, Some(witness(0, 0))),
            Ok(StrictNamedBarrierOutcome::Arrived {
                completed: false,
                ..
            })
        ));
        assert!(matches!(
            protocol.arrive(id(0, 8), 32, 0, second, Some(witness(0, 1))),
            Ok(StrictNamedBarrierOutcome::Arrived {
                completed: true,
                ..
            })
        ));

        let snapshot = protocol.snapshot(id(0, 8));
        assert_eq!(snapshot.arrival_count(), 32);
        assert_eq!(snapshot.contributors().len(), 2);
        assert_eq!(snapshot.contributors()[0].arrival_mask(), first);
        assert_eq!(snapshot.contributors()[1].arrival_mask(), second);
    }

    #[test]
    fn completed_generation_reuses_id_without_cross_cta_collision() {
        let protocol = StrictNamedBarrierProtocol::new(None);
        protocol
            .sync(id(0, 6), 32, 0, WarpMask::FULL, Some(witness(0, 0)))
            .unwrap();
        let reuse = protocol
            .sync(id(0, 6), 32, 0, WarpMask::FULL, Some(witness(0, 1)))
            .unwrap();
        assert_eq!(reuse.generation(), 1);
        assert_eq!(protocol.snapshot(id(0, 6)).completed_through(), Some(0));

        let other_cta = protocol
            .sync(id(1, 6), 32, 1, WarpMask::FULL, Some(witness(1, 0)))
            .unwrap();
        assert_eq!(other_cta.generation(), 0);
        assert_eq!(protocol.snapshot(id(1, 6)).completed_through(), None);
    }

    #[test]
    fn completed_generation_retains_exact_waiter_identities() {
        let protocol = StrictNamedBarrierProtocol::new(None);
        protocol
            .sync(id(0, 12), 32, 0, WarpMask::FULL, Some(witness(0, 0)))
            .unwrap();
        protocol
            .sync(id(0, 12), 32, 1, WarpMask::FULL, Some(witness(1, 0)))
            .unwrap();

        assert!(protocol.waiter_state_exact(id(0, 12), 0, WarpMask::FULL, 0));
        assert!(!protocol.waiter_state_exact(id(0, 12), 7, WarpMask::FULL, 0));
        assert!(!protocol.waiter_state_exact(id(0, 12), 0, lanes(0..16), 0));
        protocol
            .resume(id(0, 12), 0, WarpMask::FULL, 0, Some(witness(0, 1)))
            .unwrap();
        assert!(!protocol.waiter_state_exact(id(0, 12), 0, WarpMask::FULL, 0));
        assert!(matches!(
            protocol.resume(id(0, 12), 0, WarpMask::FULL, 0, Some(witness(0, 2))),
            Err(StrictNamedBarrierError::ResumeWithoutRegistration { .. })
        ));
    }

    #[test]
    fn unaligned_resume_recombines_disjoint_lane_registrations() {
        let protocol = StrictNamedBarrierProtocol::new(None);
        let first = lanes([0]);
        let second = lanes(1..32);
        protocol
            .sync_with_alignment(id(0, 13), 32, 0, first, false, Some(witness(0, 0)))
            .unwrap();
        protocol
            .sync_with_alignment(id(0, 13), 32, 0, second, false, Some(witness(0, 1)))
            .unwrap();

        protocol
            .resume_with_alignment(id(0, 13), 0, WarpMask::FULL, 0, false, Some(witness(0, 2)))
            .unwrap();
        assert!(protocol.snapshot(id(0, 13)).waiting_warps().is_empty());
    }

    #[test]
    fn aligned_resume_does_not_recombine_partial_waiters() {
        let protocol = StrictNamedBarrierProtocol::new(None);
        let low = lanes(0..16);
        let high = lanes(16..32);
        protocol
            .sync(id(0, 14), 32, 0, low, Some(witness(0, 0)))
            .unwrap();
        protocol
            .sync(id(0, 14), 32, 0, high, Some(origin_witness(0, 1, 100, [])))
            .unwrap();

        assert!(matches!(
            protocol.resume(id(0, 14), 0, WarpMask::FULL, 0, Some(witness(0, 2))),
            Err(StrictNamedBarrierError::ResumeWithoutRegistration { .. })
        ));
        assert!(protocol.waiter_state_exact(id(0, 14), 0, low, 0));
        assert!(protocol.waiter_state_exact(id(0, 14), 0, high, 0));
    }

    #[test]
    fn zero_contract_and_zero_contribution_are_rejected() {
        let protocol = StrictNamedBarrierProtocol::new(None);
        assert!(matches!(
            protocol.sync(id(0, 7), 0, 0, WarpMask::FULL, Some(witness(0, 0))),
            Err(StrictNamedBarrierError::InvalidExpectedArrivals { .. })
        ));
        assert!(matches!(
            protocol.sync(id(0, 7), 32, 0, WarpMask::EMPTY, Some(witness(0, 0))),
            Err(StrictNamedBarrierError::InvalidArrivalCount { .. })
        ));
        assert_eq!(protocol.snapshot(id(0, 7)).generation(), None);
    }

    #[test]
    fn full_cta_sync_accepts_the_same_static_call_site_across_warps() {
        let protocol = StrictNamedBarrierProtocol::new(Some(64));
        let loops = [LoopFrame::new(StaticOpId::new(90), 3)];
        assert!(matches!(
            protocol.sync(
                id(0, 0),
                64,
                0,
                WarpMask::FULL,
                Some(origin_witness(0, 7, 100, loops)),
            ),
            Ok(StrictNamedBarrierOutcome::Registered { generation: 0 })
        ));
        assert!(matches!(
            protocol.sync(
                id(0, 0),
                64,
                1,
                WarpMask::FULL,
                Some(origin_witness(1, 11, 100, loops)),
            ),
            Ok(StrictNamedBarrierOutcome::Ready {
                generation: 0,
                completed_now: true,
                ..
            })
        ));
        assert!(protocol.full_cta_aligned_nonuniform_errors().is_empty());
    }

    #[test]
    fn full_cta_sync_accepts_distinct_inlined_sites_across_warps_transactionally() {
        let protocol = StrictNamedBarrierProtocol::new(Some(64));
        protocol
            .sync(
                id(0, 0),
                64,
                0,
                WarpMask::FULL,
                Some(origin_witness(0, 0, 100, [])),
            )
            .unwrap();

        assert!(matches!(
            protocol.sync(
                id(0, 0),
                64,
                1,
                WarpMask::FULL,
                Some(origin_witness(1, 0, 101, [])),
            ),
            Ok(StrictNamedBarrierOutcome::Ready {
                generation: 0,
                completed_now: true,
                ..
            })
        ));
        let snapshot = protocol.snapshot(id(0, 0));
        assert_eq!(snapshot.arrival_count(), 64);
        assert_eq!(snapshot.contributors().len(), 2);
    }

    /// Positive control for an intentional detection change: loop iterations
    /// of one static site share one origin, so a full-CTA generation whose
    /// warps reach the same site at skewed iterations is accepted (it was
    /// an origin-mismatch error under the retired flavor-keyed origin check).
    #[test]
    fn full_cta_sync_accepts_skewed_loop_iterations_of_one_static_site() {
        let protocol = StrictNamedBarrierProtocol::new(Some(64));
        protocol
            .sync(
                id(0, 0),
                64,
                0,
                WarpMask::FULL,
                Some(origin_witness(
                    0,
                    0,
                    100,
                    [LoopFrame::new(StaticOpId::new(90), 0)],
                )),
            )
            .unwrap();
        assert!(matches!(
            protocol.sync(
                id(0, 0),
                64,
                1,
                WarpMask::FULL,
                Some(origin_witness(
                    1,
                    0,
                    100,
                    [LoopFrame::new(StaticOpId::new(90), 1)],
                )),
            ),
            Ok(StrictNamedBarrierOutcome::Ready {
                generation: 0,
                completed_now: true,
                ..
            })
        ));
        assert!(protocol.full_cta_aligned_nonuniform_errors().is_empty());
    }

    #[test]
    fn incomplete_full_cta_sync_generation_names_missing_warps() {
        let protocol = StrictNamedBarrierProtocol::new(Some(64));
        protocol
            .sync(
                id(1, 0),
                64,
                2,
                WarpMask::FULL,
                Some(origin_witness(2, 0, 100, [])),
            )
            .unwrap();

        assert!(matches!(
            protocol.full_cta_aligned_nonuniform_errors().as_slice(),
            [StrictNamedBarrierError::FullCtaAlignedMissingParticipants {
                missing_warps,
                ..
            }] if missing_warps.as_ref() == [3]
        ));
    }

    /// Two aligned sites reached by different lanes of one warp are still a
    /// divergent-barrier error.  Distinct warps may use separate inlined
    /// copies of the same barrier operation, but one warp cannot split its
    /// aligned participants across static sites.
    #[test]
    fn one_warp_from_two_static_sites_errors() {
        let protocol = StrictNamedBarrierProtocol::new(Some(2));
        let reduce_witness = origin_witness(0, 0, 100, []);
        let user_witness = origin_witness(1, 0, 101, []);
        protocol
            .sync(
                id(0, 0),
                2,
                0,
                WarpMask::from_bits(0b01),
                Some(reduce_witness),
            )
            .unwrap();

        assert!(matches!(
            protocol.sync(
                id(0, 0),
                2,
                0,
                WarpMask::from_bits(0b10),
                Some(user_witness),
            ),
            Err(StrictNamedBarrierError::AlignedSyncContractMismatch { .. })
        ));
    }

    #[test]
    fn distinct_warps_may_use_inlined_aligned_sites() {
        let protocol = StrictNamedBarrierProtocol::new(Some(64));
        protocol
            .sync(
                id(0, 0),
                64,
                0,
                WarpMask::FULL,
                Some(origin_witness(0, 0, 100, [])),
            )
            .unwrap();
        assert!(matches!(
            protocol.sync(
                id(0, 0),
                64,
                1,
                WarpMask::FULL,
                Some(origin_witness(1, 0, 101, [])),
            ),
            Ok(StrictNamedBarrierOutcome::Ready { generation: 0, .. })
        ));
    }

    #[test]
    fn origin_check_ignores_arrive_origins() {
        let protocol = StrictNamedBarrierProtocol::new(Some(64));
        protocol
            .arrive(
                id(0, 0),
                64,
                0,
                WarpMask::FULL,
                Some(origin_witness(0, 0, 100, [])),
            )
            .unwrap();
        assert!(matches!(
            protocol.sync(
                id(0, 0),
                64,
                1,
                WarpMask::FULL,
                Some(origin_witness(1, 0, 101, [])),
            ),
            Ok(StrictNamedBarrierOutcome::Ready { .. })
        ));
    }

    #[test]
    fn origin_check_rejects_unaligned_then_aligned_sync() {
        let protocol = StrictNamedBarrierProtocol::new(Some(64));
        protocol
            .sync_with_alignment(
                id(0, 0),
                64,
                0,
                WarpMask::FULL,
                false,
                Some(origin_witness(0, 0, 100, [])),
            )
            .unwrap();
        assert!(matches!(
            protocol.sync(
                id(0, 0),
                64,
                1,
                WarpMask::FULL,
                Some(origin_witness(1, 0, 101, [])),
            ),
            Err(StrictNamedBarrierError::AlignedSyncContractMismatch { .. })
        ));
    }

    #[test]
    fn origin_check_rejects_aligned_then_unaligned_sync() {
        let protocol = StrictNamedBarrierProtocol::new(Some(64));
        protocol
            .sync(
                id(0, 0),
                64,
                0,
                WarpMask::FULL,
                Some(origin_witness(0, 0, 100, [])),
            )
            .unwrap();
        assert!(matches!(
            protocol.sync_with_alignment(
                id(0, 0),
                64,
                1,
                WarpMask::FULL,
                false,
                Some(origin_witness(1, 0, 101, [])),
            ),
            Err(StrictNamedBarrierError::AlignedSyncContractMismatch { .. })
        ));
    }

    #[test]
    fn origin_check_accepts_distinct_sites_across_sub_cta_warps() {
        let protocol = StrictNamedBarrierProtocol::new(Some(128));
        protocol
            .sync(
                id(0, 0),
                64,
                0,
                WarpMask::FULL,
                Some(origin_witness(0, 0, 100, [])),
            )
            .unwrap();
        assert!(matches!(
            protocol.sync(
                id(0, 0),
                64,
                1,
                WarpMask::FULL,
                Some(origin_witness(1, 0, 101, [])),
            ),
            Ok(StrictNamedBarrierOutcome::Ready { .. })
        ));
        assert!(protocol.full_cta_aligned_nonuniform_errors().is_empty());
    }

    #[test]
    fn origin_check_accepts_distinct_sites_without_a_topology() {
        let protocol = StrictNamedBarrierProtocol::new(None);
        protocol
            .sync(
                id(0, 0),
                64,
                0,
                WarpMask::FULL,
                Some(origin_witness(0, 0, 100, [])),
            )
            .unwrap();
        assert!(matches!(
            protocol.sync(
                id(0, 0),
                64,
                1,
                WarpMask::FULL,
                Some(origin_witness(1, 0, 101, [])),
            ),
            Ok(StrictNamedBarrierOutcome::Ready { .. })
        ));
        assert!(protocol.full_cta_aligned_nonuniform_errors().is_empty());
    }

    #[test]
    fn origin_check_ignores_arrive_and_accepts_distinct_sync_sites_across_warps() {
        let protocol = StrictNamedBarrierProtocol::new(Some(96));
        protocol
            .arrive(
                id(0, 0),
                96,
                0,
                WarpMask::FULL,
                Some(origin_witness(0, 0, 100, [])),
            )
            .unwrap();
        protocol
            .sync(
                id(0, 0),
                96,
                1,
                WarpMask::FULL,
                Some(origin_witness(1, 0, 101, [])),
            )
            .unwrap();

        assert!(matches!(
            protocol.sync(
                id(0, 0),
                96,
                2,
                WarpMask::FULL,
                Some(origin_witness(2, 0, 102, [])),
            ),
            Ok(StrictNamedBarrierOutcome::Ready { .. })
        ));
    }
}
