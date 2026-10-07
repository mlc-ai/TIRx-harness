use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::Mutex;

use crate::{ClusterBarrierId, DynamicOpId, WarpMask};

type SharedWitness = Option<Box<DynamicOpId>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StrictClusterBarrierOutcome {
    generation: u64,
    completed_now: bool,
    rearrival_without_wait: bool,
}

impl StrictClusterBarrierOutcome {
    pub const fn new(generation: u64, completed_now: bool) -> Self {
        Self {
            generation,
            completed_now,
            rearrival_without_wait: false,
        }
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }

    pub const fn completed_now(self) -> bool {
        self.completed_now
    }

    pub const fn rearrival_without_wait(self) -> bool {
        self.rearrival_without_wait
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrictClusterBarrierIncomplete {
    barrier_id: ClusterBarrierId,
    generation: u64,
    missing_warps: Box<[usize]>,
    witness: SharedWitness,
}

impl StrictClusterBarrierIncomplete {
    pub const fn barrier_id(&self) -> ClusterBarrierId {
        self.barrier_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn missing_warps(&self) -> &[usize] {
        &self.missing_warps
    }

    pub fn witness(&self) -> Option<&DynamicOpId> {
        self.witness.as_deref()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StrictClusterBarrierError {
    PartialWarpParticipation {
        barrier_id: ClusterBarrierId,
        warp_id: usize,
        active_mask: WarpMask,
        witness: SharedWitness,
    },
    ContractMismatch {
        barrier_id: ClusterBarrierId,
        generation: u64,
        expected_participants: Box<[usize]>,
        observed_participants: Box<[usize]>,
        witness: SharedWitness,
    },
    UnexpectedParticipant {
        barrier_id: ClusterBarrierId,
        generation: u64,
        warp_id: usize,
        witness: SharedWitness,
    },
    EarlyArrival {
        barrier_id: ClusterBarrierId,
        generation: u64,
        warp_id: usize,
        prior_witness: SharedWitness,
        witness: SharedWitness,
    },
    WaitBeforeArrival {
        barrier_id: ClusterBarrierId,
        warp_id: usize,
        witness: SharedWitness,
    },
    DuplicateWait {
        barrier_id: ClusterBarrierId,
        generation: u64,
        warp_id: usize,
        witness: SharedWitness,
    },
    ResumeWithoutRegistration {
        barrier_id: ClusterBarrierId,
        generation: u64,
        warp_id: usize,
        witness: SharedWitness,
    },
    ResumeBeforeCompletion {
        barrier_id: ClusterBarrierId,
        generation: u64,
        warp_id: usize,
        witness: SharedWitness,
    },
    GenerationOverflow {
        barrier_id: ClusterBarrierId,
        generation: u64,
        witness: SharedWitness,
    },
}

impl fmt::Display for StrictClusterBarrierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PartialWarpParticipation {
                barrier_id,
                warp_id,
                active_mask,
                ..
            } => write!(
                f,
                "warp {warp_id} reached aligned cluster barrier {barrier_id:?} with partial active mask {active_mask:?}"
            ),
            Self::ContractMismatch {
                barrier_id,
                generation,
                expected_participants,
                observed_participants,
                ..
            } => write!(
                f,
                "cluster barrier {barrier_id:?} generation {generation} participant contract changed from {expected_participants:?} to {observed_participants:?}"
            ),
            Self::UnexpectedParticipant {
                barrier_id,
                generation,
                warp_id,
                ..
            } => write!(
                f,
                "warp {warp_id} is not a participant of cluster barrier {barrier_id:?} generation {generation}"
            ),
            Self::EarlyArrival {
                barrier_id,
                generation,
                warp_id,
                ..
            } => write!(
                f,
                "warp {warp_id} arrived at cluster barrier {barrier_id:?} again before generation {generation} completed"
            ),
            Self::WaitBeforeArrival {
                barrier_id,
                warp_id,
                ..
            } => write!(
                f,
                "warp {warp_id} waited on cluster barrier {barrier_id:?} before arriving"
            ),
            Self::DuplicateWait {
                barrier_id,
                generation,
                warp_id,
                ..
            } => write!(
                f,
                "warp {warp_id} registered cluster barrier {barrier_id:?} generation {generation} wait twice"
            ),
            Self::ResumeWithoutRegistration {
                barrier_id,
                generation,
                warp_id,
                ..
            } => write!(
                f,
                "warp {warp_id} resumed cluster barrier {barrier_id:?} generation {generation} without a registered wait"
            ),
            Self::ResumeBeforeCompletion {
                barrier_id,
                generation,
                warp_id,
                ..
            } => write!(
                f,
                "warp {warp_id} resumed cluster barrier {barrier_id:?} before generation {generation} completed"
            ),
            Self::GenerationOverflow {
                barrier_id,
                generation,
                ..
            } => write!(
                f,
                "cluster barrier {barrier_id:?} cannot advance beyond generation {generation}"
            ),
        }
    }
}

impl Error for StrictClusterBarrierError {}

#[derive(Clone, Debug)]
struct Arrival {
    witness: SharedWitness,
}

#[derive(Clone, Debug, Default)]
struct GenerationState {
    arrivals: BTreeMap<usize, Arrival>,
    waiters: BTreeSet<usize>,
    consumed_waiters: BTreeSet<usize>,
    complete: bool,
}

#[derive(Clone, Debug)]
struct BarrierState {
    participants: Box<[usize]>,
    current_generation: u64,
    generations: BTreeMap<u64, GenerationState>,
    last_arrival_generation: BTreeMap<usize, u64>,
}

impl BarrierState {
    fn new(participants: &[usize]) -> Self {
        Self {
            participants: participants.into(),
            current_generation: 0,
            generations: BTreeMap::new(),
            last_arrival_generation: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct ProtocolState {
    barriers: BTreeMap<ClusterBarrierId, BarrierState>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StrictClusterBarrierSemanticState {
    barriers: Box<[(ClusterBarrierId, StrictClusterBarrierStateSemanticState)]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StrictClusterBarrierStateSemanticState {
    participants: Box<[usize]>,
    current_generation: u64,
    generations: Box<[(u64, StrictClusterGenerationSemanticState)]>,
    last_arrival_generation: Box<[(usize, u64)]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StrictClusterGenerationSemanticState {
    arrivals: Box<[usize]>,
    waiters: Box<[usize]>,
    consumed_waiters: Box<[usize]>,
    complete: bool,
}

#[derive(Default)]
pub struct StrictClusterBarrierProtocol {
    state: Mutex<ProtocolState>,
}

impl StrictClusterBarrierProtocol {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn semantic_state(&self) -> StrictClusterBarrierSemanticState {
        let state = self
            .state
            .lock()
            .expect("strict cluster barrier mutex poisoned");
        StrictClusterBarrierSemanticState {
            barriers: state
                .barriers
                .iter()
                .map(|(&barrier_id, barrier)| {
                    (
                        barrier_id,
                        StrictClusterBarrierStateSemanticState {
                            participants: barrier.participants.clone(),
                            current_generation: barrier.current_generation,
                            generations: barrier
                                .generations
                                .iter()
                                .map(|(&generation, phase)| {
                                    (
                                        generation,
                                        StrictClusterGenerationSemanticState {
                                            arrivals: phase
                                                .arrivals
                                                .keys()
                                                .copied()
                                                .collect::<Vec<_>>()
                                                .into_boxed_slice(),
                                            waiters: phase
                                                .waiters
                                                .iter()
                                                .copied()
                                                .collect::<Vec<_>>()
                                                .into_boxed_slice(),
                                            consumed_waiters: phase
                                                .consumed_waiters
                                                .iter()
                                                .copied()
                                                .collect::<Vec<_>>()
                                                .into_boxed_slice(),
                                            complete: phase.complete,
                                        },
                                    )
                                })
                                .collect::<Vec<_>>()
                                .into_boxed_slice(),
                            last_arrival_generation: barrier
                                .last_arrival_generation
                                .iter()
                                .map(|(&warp_id, &generation)| (warp_id, generation))
                                .collect::<Vec<_>>()
                                .into_boxed_slice(),
                        },
                    )
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    pub fn arrive(
        &self,
        barrier_id: ClusterBarrierId,
        participant_warps: &[usize],
        warp_id: usize,
        arrival_mask: WarpMask,
        witness: Option<DynamicOpId>,
    ) -> Result<StrictClusterBarrierOutcome, StrictClusterBarrierError> {
        let witness = witness.map(Box::new);
        if arrival_mask != WarpMask::FULL {
            return Err(StrictClusterBarrierError::PartialWarpParticipation {
                barrier_id,
                warp_id,
                active_mask: arrival_mask,
                witness,
            });
        }
        let mut state = self
            .state
            .lock()
            .expect("strict cluster barrier mutex poisoned");
        let barrier = state
            .barriers
            .entry(barrier_id)
            .or_insert_with(|| BarrierState::new(participant_warps));
        let generation = barrier.current_generation;
        let rearrival_without_wait = barrier
            .last_arrival_generation
            .get(&warp_id)
            .copied()
            .filter(|prior| *prior < generation)
            .and_then(|prior| barrier.generations.get(&prior))
            .is_some_and(|prior| prior.complete && !prior.consumed_waiters.contains(&warp_id));
        if barrier.participants.as_ref() != participant_warps {
            return Err(StrictClusterBarrierError::ContractMismatch {
                barrier_id,
                generation,
                expected_participants: barrier.participants.clone(),
                observed_participants: participant_warps.into(),
                witness,
            });
        }
        if !barrier.participants.contains(&warp_id) {
            return Err(StrictClusterBarrierError::UnexpectedParticipant {
                barrier_id,
                generation,
                warp_id,
                witness,
            });
        }
        let phase = barrier.generations.entry(generation).or_default();
        if let Some(prior) = phase.arrivals.get(&warp_id) {
            return Err(StrictClusterBarrierError::EarlyArrival {
                barrier_id,
                generation,
                warp_id,
                prior_witness: prior.witness.clone(),
                witness,
            });
        }
        phase.arrivals.insert(
            warp_id,
            Arrival {
                witness: witness.clone(),
            },
        );
        barrier.last_arrival_generation.insert(warp_id, generation);
        let completed_now = phase.arrivals.len() == barrier.participants.len();
        if completed_now {
            phase.complete = true;
            barrier.current_generation = generation.checked_add(1).ok_or_else(|| {
                StrictClusterBarrierError::GenerationOverflow {
                    barrier_id,
                    generation,
                    witness,
                }
            })?;
        }
        Ok(StrictClusterBarrierOutcome {
            generation,
            completed_now,
            rearrival_without_wait,
        })
    }

    pub fn wait_register(
        &self,
        barrier_id: ClusterBarrierId,
        participant_warps: &[usize],
        warp_id: usize,
        arrival_mask: WarpMask,
        witness: Option<DynamicOpId>,
    ) -> Result<StrictClusterBarrierOutcome, StrictClusterBarrierError> {
        let witness = witness.map(Box::new);
        if arrival_mask != WarpMask::FULL {
            return Err(StrictClusterBarrierError::PartialWarpParticipation {
                barrier_id,
                warp_id,
                active_mask: arrival_mask,
                witness,
            });
        }
        let mut state = self
            .state
            .lock()
            .expect("strict cluster barrier mutex poisoned");
        let Some(barrier) = state.barriers.get_mut(&barrier_id) else {
            return Err(StrictClusterBarrierError::WaitBeforeArrival {
                barrier_id,
                warp_id,
                witness,
            });
        };
        let generation = barrier
            .last_arrival_generation
            .get(&warp_id)
            .copied()
            .ok_or_else(|| StrictClusterBarrierError::WaitBeforeArrival {
                barrier_id,
                warp_id,
                witness: witness.clone(),
            })?;
        if barrier.participants.as_ref() != participant_warps {
            return Err(StrictClusterBarrierError::ContractMismatch {
                barrier_id,
                generation,
                expected_participants: barrier.participants.clone(),
                observed_participants: participant_warps.into(),
                witness,
            });
        }
        let phase = barrier
            .generations
            .get_mut(&generation)
            .expect("last arrival names an existing generation");
        if phase.waiters.contains(&warp_id) || phase.consumed_waiters.contains(&warp_id) {
            return Err(StrictClusterBarrierError::DuplicateWait {
                barrier_id,
                generation,
                warp_id,
                witness,
            });
        }
        phase.waiters.insert(warp_id);
        Ok(StrictClusterBarrierOutcome::new(generation, phase.complete))
    }

    pub fn resume(
        &self,
        barrier_id: ClusterBarrierId,
        generation: u64,
        warp_id: usize,
        witness: Option<DynamicOpId>,
    ) -> Result<(), StrictClusterBarrierError> {
        let witness = witness.map(Box::new);
        let mut state = self
            .state
            .lock()
            .expect("strict cluster barrier mutex poisoned");
        let Some(phase) = state
            .barriers
            .get_mut(&barrier_id)
            .and_then(|barrier| barrier.generations.get_mut(&generation))
        else {
            return Err(StrictClusterBarrierError::ResumeWithoutRegistration {
                barrier_id,
                generation,
                warp_id,
                witness,
            });
        };
        if !phase.complete {
            return Err(StrictClusterBarrierError::ResumeBeforeCompletion {
                barrier_id,
                generation,
                warp_id,
                witness,
            });
        }
        if !phase.waiters.remove(&warp_id) {
            return Err(StrictClusterBarrierError::ResumeWithoutRegistration {
                barrier_id,
                generation,
                warp_id,
                witness,
            });
        }
        phase.consumed_waiters.insert(warp_id);
        Ok(())
    }

    pub fn incomplete_generations(&self) -> Vec<StrictClusterBarrierIncomplete> {
        let state = self
            .state
            .lock()
            .expect("strict cluster barrier mutex poisoned");
        let mut incomplete = Vec::new();
        for (&barrier_id, barrier) in &state.barriers {
            for (&generation, phase) in &barrier.generations {
                if phase.complete {
                    continue;
                }
                let missing_warps = barrier
                    .participants
                    .iter()
                    .copied()
                    .filter(|warp_id| !phase.arrivals.contains_key(warp_id))
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let witness = phase
                    .arrivals
                    .values()
                    .find_map(|arrival| arrival.witness.clone());
                incomplete.push(StrictClusterBarrierIncomplete {
                    barrier_id,
                    generation,
                    missing_warps,
                    witness,
                });
            }
        }
        incomplete
    }
}

impl Clone for StrictClusterBarrierProtocol {
    fn clone(&self) -> Self {
        Self {
            state: Mutex::new(
                self.state
                    .lock()
                    .expect("strict cluster barrier mutex poisoned")
                    .clone(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{StaticOpId, WarpMask};

    use super::*;

    fn witness(warp: usize, sequence: u64) -> DynamicOpId {
        DynamicOpId::new(0, warp, sequence, StaticOpId::new(10 + sequence), [])
    }

    #[test]
    fn different_source_sites_share_one_cluster_generation() {
        let protocol = StrictClusterBarrierProtocol::new();
        let id = ClusterBarrierId::new(0, 0);
        assert_eq!(
            protocol
                .arrive(id, &[0, 1], 0, WarpMask::FULL, Some(witness(0, 0)))
                .unwrap(),
            StrictClusterBarrierOutcome::new(0, false)
        );
        assert_eq!(
            protocol
                .arrive(id, &[0, 1], 1, WarpMask::FULL, Some(witness(1, 0)))
                .unwrap(),
            StrictClusterBarrierOutcome::new(0, true)
        );
    }

    #[test]
    fn second_arrival_before_completion_is_exact_error() {
        let protocol = StrictClusterBarrierProtocol::new();
        let id = ClusterBarrierId::new(0, 0);
        protocol
            .arrive(id, &[0, 1], 0, WarpMask::FULL, Some(witness(0, 0)))
            .unwrap();
        assert!(matches!(
            protocol.arrive(id, &[0, 1], 0, WarpMask::FULL, Some(witness(0, 1))),
            Err(StrictClusterBarrierError::EarlyArrival {
                generation: 0,
                warp_id: 0,
                ..
            })
        ));
    }

    #[test]
    fn aligned_barrier_rejects_partial_warp_before_mutation() {
        let protocol = StrictClusterBarrierProtocol::new();
        let id = ClusterBarrierId::new(0, 0);
        let mask = WarpMask::from_lanes(0..16).unwrap();
        assert!(matches!(
            protocol.arrive(id, &[0], 0, mask, Some(witness(0, 0))),
            Err(StrictClusterBarrierError::PartialWarpParticipation {
                warp_id: 0,
                active_mask,
                ..
            }) if active_mask == mask
        ));
        assert!(protocol.incomplete_generations().is_empty());
    }

    #[test]
    fn incomplete_generation_retains_an_arrival_witness() {
        let protocol = StrictClusterBarrierProtocol::new();
        let id = ClusterBarrierId::new(0, 0);
        let arrival = witness(0, 0);
        protocol
            .arrive(id, &[0, 1], 0, WarpMask::FULL, Some(arrival.clone()))
            .unwrap();

        let incomplete = protocol.incomplete_generations();
        assert_eq!(incomplete.len(), 1);
        assert_eq!(incomplete[0].witness(), Some(&arrival));
        assert_eq!(incomplete[0].missing_warps(), [1]);
    }

    #[test]
    fn arrive_only_rearrival_is_marked_verify_incomplete() {
        let protocol = StrictClusterBarrierProtocol::new();
        let id = ClusterBarrierId::new(0, 0);
        assert!(protocol
            .arrive(id, &[0], 0, WarpMask::FULL, Some(witness(0, 0)))
            .unwrap()
            .completed_now());
        assert!(protocol
            .arrive(id, &[0], 0, WarpMask::FULL, Some(witness(0, 1)))
            .unwrap()
            .rearrival_without_wait());
    }
}
