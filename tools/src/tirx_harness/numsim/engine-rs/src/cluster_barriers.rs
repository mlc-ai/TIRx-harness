use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use crate::{
    BlockedOperation, ClusterBarrierOperation, CompletionProgress, CompletionSource,
    LaunchTopology, OccurrenceKey, ParticipantContract, ParticipantSet, ParticipantState,
    ScopeInstance, SynchronizationError, WarpContext, WarpMask,
};

const CLUSTER_BARRIER_DIAGNOSTIC_OP_ID: u64 = u64::MAX - 1;

/// Analysis identity of the one implicit hardware barrier owned by a cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ClusterBarrierId {
    kernel_index: usize,
    cluster_id: usize,
}

impl ClusterBarrierId {
    pub const fn new(kernel_index: usize, cluster_id: usize) -> Self {
        Self {
            kernel_index,
            cluster_id,
        }
    }

    pub const fn kernel_index(self) -> usize {
        self.kernel_index
    }

    pub const fn cluster_id(self) -> usize {
        self.cluster_id
    }
}

/// Exact numeric result of one cluster-barrier arrival.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterBarrierArrivalOutcome {
    cluster_id: usize,
    generation: u64,
    completed_now: bool,
}

impl ClusterBarrierArrivalOutcome {
    pub const fn cluster_id(self) -> usize {
        self.cluster_id
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }

    pub const fn completed_now(self) -> bool {
        self.completed_now
    }
}

/// Completion returned after a warp observes one cluster-barrier phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterBarrierCompletion {
    pub cluster_id: usize,
    pub generation: u64,
}

/// Engine-owned split-phase state for `barrier.cluster.arrive/wait`.
///
/// Each cluster is independently locked. An arrive advances only the calling
/// warp's arrival phase and never blocks. A wait observes the calling warp's
/// oldest arrived-but-not-yet-waited phase; it does not count as an arrival.
pub struct ClusterBarrierHub {
    topology: LaunchTopology,
    selected_warps: Arc<BTreeSet<usize>>,
    clusters: Vec<Option<Mutex<ClusterBarrierState>>>,
}

struct ClusterBarrierState {
    contract: ParticipantContract,
    phases: BTreeMap<u64, ClusterBarrierPhase>,
    completed_through: Option<u64>,
    current_generation: u64,
    last_arrival_generation: BTreeMap<usize, u64>,
    last_waited_generation: BTreeMap<usize, u64>,
}

#[derive(Default)]
struct ClusterBarrierPhase {
    arrivals: BTreeSet<usize>,
    lane_arrivals: BTreeMap<usize, WarpMask>,
    waiters: BTreeMap<usize, Waker>,
    complete: bool,
}

impl ClusterBarrierHub {
    pub fn new(topology: LaunchTopology) -> Self {
        Self::for_launch(
            topology,
            Arc::new((0..topology.warp_count()).collect::<BTreeSet<_>>()),
        )
    }

    pub(crate) fn for_launch(
        topology: LaunchTopology,
        selected_warps: Arc<BTreeSet<usize>>,
    ) -> Self {
        let clusters = (0..topology.clusters())
            .map(|cluster_id| {
                let participant_ids = topology
                    .cluster_warp_range(cluster_id)
                    .expect("cluster ID came from the launch topology")
                    .filter(|warp_id| selected_warps.contains(warp_id))
                    .collect::<Vec<_>>();
                if participant_ids.is_empty() {
                    return None;
                }
                let participants = ParticipantSet::new(participant_ids)
                    .expect("a selected cluster has at least one participating warp");
                Some(Mutex::new(ClusterBarrierState {
                    contract: ParticipantContract::explicit(
                        ScopeInstance::Cluster { cluster_id },
                        participants,
                    ),
                    phases: BTreeMap::new(),
                    completed_through: None,
                    current_generation: 0,
                    last_arrival_generation: BTreeMap::new(),
                    last_waited_generation: BTreeMap::new(),
                }))
            })
            .collect();
        Self {
            topology,
            selected_warps,
            clusters,
        }
    }

    /// Record this warp's arrival at its next cluster-barrier phase.
    pub fn arrive(
        &self,
        context: WarpContext,
    ) -> Result<ClusterBarrierArrivalOutcome, SynchronizationError> {
        self.arrive_with_alignment(context, true)
    }

    pub fn arrive_with_alignment(
        &self,
        context: WarpContext,
        aligned: bool,
    ) -> Result<ClusterBarrierArrivalOutcome, SynchronizationError> {
        self.arrive_resolved_with_alignment(
            context.topology(),
            context.cluster_id(),
            context.global_warp_id(),
            context.active_mask(),
            aligned,
        )
    }

    pub(crate) fn arrive_resolved_with_alignment(
        &self,
        topology: LaunchTopology,
        cluster_id: usize,
        warp_id: usize,
        active_mask: WarpMask,
        aligned: bool,
    ) -> Result<ClusterBarrierArrivalOutcome, SynchronizationError> {
        let cluster = self.validate_resolved(
            topology,
            cluster_id,
            warp_id,
            ClusterBarrierOperation::Arrive,
        )?;
        Self::validate_alignment(
            warp_id,
            active_mask,
            ClusterBarrierOperation::Arrive,
            aligned,
        )?;
        let wakers;
        let generation;
        let completed_now;
        {
            let mut state = cluster.lock().expect("cluster barrier mutex poisoned");
            generation = state.current_generation;
            let expected = state.contract.participants().len();
            let warp_arrival_complete;
            completed_now = {
                let barrier_phase = state.phases.entry(generation).or_default();
                let prior_lanes = barrier_phase
                    .lane_arrivals
                    .get(&warp_id)
                    .copied()
                    .unwrap_or(WarpMask::EMPTY);
                if !prior_lanes.intersection(active_mask).is_empty() {
                    return Err(SynchronizationError::DuplicateArrival {
                        key: occurrence_key(cluster_id, generation),
                        phase: generation,
                        warp_id,
                    });
                }
                let arrived_lanes = prior_lanes.union(active_mask);
                barrier_phase.lane_arrivals.insert(warp_id, arrived_lanes);
                warp_arrival_complete = arrived_lanes.is_full();
                if warp_arrival_complete {
                    barrier_phase.arrivals.insert(warp_id);
                }
                barrier_phase.arrivals.len() == expected
            };
            if warp_arrival_complete {
                state.last_arrival_generation.insert(warp_id, generation);
            }
            if completed_now {
                state.current_generation = generation.checked_add(1).ok_or(
                    SynchronizationError::ClusterBarrierPhaseOverflow {
                        cluster_id,
                        warp_id,
                    },
                )?;
                let barrier_phase = state
                    .phases
                    .get_mut(&generation)
                    .expect("the phase was inserted above");
                barrier_phase.complete = true;
                wakers = std::mem::take(&mut barrier_phase.waiters)
                    .into_values()
                    .collect();
                advance_completed_phases(&mut state);
            } else {
                wakers = Vec::new();
            }
        }
        wake_all(wakers);
        Ok(ClusterBarrierArrivalOutcome {
            cluster_id,
            generation,
            completed_now,
        })
    }

    /// Wait for every cluster participant to arrive at this warp's oldest
    /// arrived-but-not-yet-waited phase.
    pub fn wait(
        self: &Arc<Self>,
        context: WarpContext,
    ) -> Result<ClusterBarrierWait, SynchronizationError> {
        self.wait_with_alignment(context, true)
    }

    pub fn wait_with_alignment(
        self: &Arc<Self>,
        context: WarpContext,
        aligned: bool,
    ) -> Result<ClusterBarrierWait, SynchronizationError> {
        self.wait_resolved_with_alignment(
            context.topology(),
            context.cluster_id(),
            context.global_warp_id(),
            context.active_mask(),
            aligned,
        )
    }

    pub(crate) fn wait_resolved_with_alignment(
        self: &Arc<Self>,
        topology: LaunchTopology,
        cluster_id: usize,
        warp_id: usize,
        active_mask: WarpMask,
        aligned: bool,
    ) -> Result<ClusterBarrierWait, SynchronizationError> {
        let cluster =
            self.validate_resolved(topology, cluster_id, warp_id, ClusterBarrierOperation::Wait)?;
        Self::validate_alignment(warp_id, active_mask, ClusterBarrierOperation::Wait, aligned)?;
        if !active_mask.is_full() {
            return Err(SynchronizationError::PartialWarpSynchronization {
                operation: ClusterBarrierOperation::UnalignedWaitUnsupported,
                warp_id,
                active_mask: active_mask.bits(),
            });
        }
        let state = cluster.lock().expect("cluster barrier mutex poisoned");
        let generation = state.last_arrival_generation.get(&warp_id).copied().ok_or(
            SynchronizationError::ClusterBarrierWaitBeforeArrival {
                cluster_id,
                warp_id,
                phase: state.current_generation,
            },
        )?;
        if state
            .last_waited_generation
            .get(&warp_id)
            .is_some_and(|waited| *waited >= generation)
        {
            return Err(SynchronizationError::DuplicateWaiter {
                key: occurrence_key(cluster_id, generation),
                phase: Some(generation),
                warp_id,
            });
        }
        if generation > state.current_generation {
            return Err(SynchronizationError::ClusterBarrierWaitBeforeArrival {
                cluster_id,
                warp_id,
                phase: generation,
            });
        }
        let completed_at_registration = phase_is_complete(&state, generation);
        drop(state);
        Ok(ClusterBarrierWait {
            hub: Arc::clone(self),
            cluster_id,
            warp_id,
            generation,
            completed_at_registration,
            registered: false,
            finished: false,
        })
    }

    fn validate_resolved(
        &self,
        topology: LaunchTopology,
        cluster_id: usize,
        warp_id: usize,
        operation: ClusterBarrierOperation,
    ) -> Result<&Mutex<ClusterBarrierState>, SynchronizationError> {
        let participant_range = self.topology.cluster_warp_range(cluster_id);
        if topology != self.topology
            || participant_range
                .as_ref()
                .is_none_or(|participants| !participants.contains(&warp_id))
            || !self.selected_warps.contains(&warp_id)
        {
            return Err(SynchronizationError::ClusterBarrierContextMismatch {
                operation,
                cluster_id,
                warp_id,
            });
        }
        Ok(self.clusters[cluster_id]
            .as_ref()
            .expect("a selected warp has cluster-barrier state"))
    }

    fn validate_alignment(
        warp_id: usize,
        active_mask: WarpMask,
        operation: ClusterBarrierOperation,
        aligned: bool,
    ) -> Result<(), SynchronizationError> {
        if aligned && active_mask != WarpMask::FULL {
            return Err(SynchronizationError::PartialWarpSynchronization {
                operation,
                warp_id,
                active_mask: active_mask.bits(),
            });
        }
        Ok(())
    }
}

impl CompletionSource for ClusterBarrierHub {
    fn source_name(&self) -> &'static str {
        "cluster-barrier"
    }

    fn pump(&self) -> Result<CompletionProgress, SynchronizationError> {
        Ok(CompletionProgress::default())
    }

    fn blocked_operations(&self) -> Vec<BlockedOperation> {
        let mut blocked = Vec::new();
        for (cluster_id, cluster) in self.clusters.iter().enumerate() {
            let Some(cluster) = cluster else {
                continue;
            };
            let state = cluster.lock().expect("cluster barrier mutex poisoned");
            for (phase, barrier_phase) in &state.phases {
                if barrier_phase.waiters.is_empty() {
                    continue;
                }
                let participant_state =
                    ParticipantState::new(&state.contract, &barrier_phase.arrivals, None, None);
                for warp_id in barrier_phase.waiters.keys().copied() {
                    blocked.push(BlockedOperation::new(
                        warp_id,
                        crate::AwaitedOperation::ClusterBarrierWait,
                        occurrence_key(cluster_id, *phase),
                        Some(*phase),
                        participant_state.clone(),
                    ));
                }
            }
        }
        blocked
    }

    fn validate_quiescent(&self) -> Result<(), SynchronizationError> {
        for (cluster_id, cluster) in self.clusters.iter().enumerate() {
            let Some(cluster) = cluster else {
                continue;
            };
            let state = cluster.lock().expect("cluster barrier mutex poisoned");
            if let Some((phase, barrier_phase)) = state.phases.iter().next() {
                return Err(SynchronizationError::CompletionSourceNotQuiescent {
                    source_name: self.source_name(),
                    details: format!(
                        "cluster {cluster_id} phase {phase} is incomplete: arrivals={}/{}, waiting_warps={:?}",
                        barrier_phase.arrivals.len(),
                        state.contract.participants().len(),
                        barrier_phase.waiters.keys().copied().collect::<Vec<_>>()
                    ),
                });
            }
        }
        Ok(())
    }
}

/// Future representing one warp's wait on one already-arrived phase.
pub struct ClusterBarrierWait {
    hub: Arc<ClusterBarrierHub>,
    cluster_id: usize,
    warp_id: usize,
    generation: u64,
    completed_at_registration: bool,
    registered: bool,
    finished: bool,
}

impl ClusterBarrierWait {
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn completed_at_registration(&self) -> bool {
        self.completed_at_registration
    }
}

impl Future for ClusterBarrierWait {
    type Output = Result<ClusterBarrierCompletion, SynchronizationError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if this.finished {
            return Poll::Ready(Ok(ClusterBarrierCompletion {
                cluster_id: this.cluster_id,
                generation: this.generation,
            }));
        }

        let mut state = this.hub.clusters[this.cluster_id]
            .as_ref()
            .expect("a cluster-barrier wait has selected participants")
            .lock()
            .expect("cluster barrier mutex poisoned");
        if state
            .last_waited_generation
            .get(&this.warp_id)
            .is_some_and(|waited| *waited >= this.generation)
        {
            return Poll::Ready(Err(SynchronizationError::DuplicateWaiter {
                key: occurrence_key(this.cluster_id, this.generation),
                phase: Some(this.generation),
                warp_id: this.warp_id,
            }));
        }
        if phase_is_complete(&state, this.generation) {
            state
                .last_waited_generation
                .insert(this.warp_id, this.generation);
            this.registered = false;
            this.finished = true;
            return Poll::Ready(Ok(ClusterBarrierCompletion {
                cluster_id: this.cluster_id,
                generation: this.generation,
            }));
        }

        let barrier_phase = state
            .phases
            .get_mut(&this.generation)
            .expect("a wait is only created after this warp arrives");
        match barrier_phase.waiters.get_mut(&this.warp_id) {
            Some(waker) if this.registered => {
                if !waker.will_wake(context.waker()) {
                    *waker = context.waker().clone();
                }
            }
            Some(_) => {
                return Poll::Ready(Err(SynchronizationError::DuplicateWaiter {
                    key: occurrence_key(this.cluster_id, this.generation),
                    phase: Some(this.generation),
                    warp_id: this.warp_id,
                }));
            }
            None => {
                barrier_phase
                    .waiters
                    .insert(this.warp_id, context.waker().clone());
                this.registered = true;
            }
        }
        Poll::Pending
    }
}

impl Drop for ClusterBarrierWait {
    fn drop(&mut self) {
        if !self.registered || self.finished {
            return;
        }
        let mut state = self.hub.clusters[self.cluster_id]
            .as_ref()
            .expect("a cluster-barrier wait has selected participants")
            .lock()
            .expect("cluster barrier mutex poisoned");
        if let Some(phase) = state.phases.get_mut(&self.generation) {
            phase.waiters.remove(&self.warp_id);
        }
    }
}

fn phase_is_complete(state: &ClusterBarrierState, phase: u64) -> bool {
    state
        .completed_through
        .is_some_and(|completed| phase <= completed)
        || state.phases.get(&phase).is_some_and(|entry| entry.complete)
}

fn advance_completed_phases(state: &mut ClusterBarrierState) {
    let mut next = state
        .completed_through
        .and_then(|phase| phase.checked_add(1))
        .unwrap_or(0);
    while state.phases.get(&next).is_some_and(|phase| phase.complete) {
        state.phases.remove(&next);
        state.completed_through = Some(next);
        let Some(following) = next.checked_add(1) else {
            break;
        };
        next = following;
    }
}

fn occurrence_key(cluster_id: usize, phase: u64) -> OccurrenceKey {
    OccurrenceKey::new(
        CLUSTER_BARRIER_DIAGNOSTIC_OP_ID,
        "barrier.cluster",
        [i64::try_from(phase).unwrap_or(i64::MAX)],
        ScopeInstance::Cluster { cluster_id },
    )
}

fn wake_all(wakers: Vec<Waker>) {
    for waker in wakers {
        waker.wake();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompletionRegistry, EngineError, Executor, WarpTask};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn arrive_only_participant_releases_waiter_and_wakes_it() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let contexts = topology.warp_contexts().collect::<Vec<_>>();
        let waiter_context = contexts[0];
        let arrive_only_context = contexts[1];
        let hub = Arc::new(ClusterBarrierHub::new(topology));
        let completed = Arc::new(AtomicUsize::new(0));

        let waiter = {
            let hub = Arc::clone(&hub);
            let completed = Arc::clone(&completed);
            WarpTask::new(0, async move {
                hub.arrive(waiter_context)?;
                hub.wait(waiter_context)?.await?;
                completed.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        };
        let arrive_only = {
            let hub = Arc::clone(&hub);
            WarpTask::new(1, async move {
                hub.arrive(arrive_only_context)?;
                Ok(())
            })
        };
        let mut completions = CompletionRegistry::new();
        completions.register(Arc::clone(&hub));

        let stats = Executor::default()
            .run_with_topology_and_completions([waiter, arrive_only], topology, &completions)
            .unwrap();
        assert_eq!(completed.load(Ordering::SeqCst), 1);
        assert_eq!(stats.completed_task_count, 2);
        assert_eq!(stats.poll_count, 3);
    }

    #[test]
    fn phases_repeat_without_requiring_every_participant_to_wait() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let contexts = topology.warp_contexts().collect::<Vec<_>>();
        let waiter_context = contexts[0];
        let arrive_only_context = contexts[1];
        let hub = Arc::new(ClusterBarrierHub::new(topology));
        let completed = Arc::new(AtomicUsize::new(0));

        let waiter = {
            let hub = Arc::clone(&hub);
            let completed = Arc::clone(&completed);
            WarpTask::new(0, async move {
                for phase in 0..2 {
                    assert_eq!(hub.arrive(waiter_context)?.generation(), phase);
                    assert_eq!(hub.wait(waiter_context)?.await?.generation, phase);
                    completed.fetch_add(1, Ordering::SeqCst);
                }
                Ok(())
            })
        };
        let arrive_only = {
            let hub = Arc::clone(&hub);
            WarpTask::new(1, async move {
                assert_eq!(hub.arrive(arrive_only_context)?.generation(), 0);
                assert_eq!(hub.arrive(arrive_only_context)?.generation(), 1);
                Ok(())
            })
        };
        let mut completions = CompletionRegistry::new();
        completions.register(Arc::clone(&hub));

        Executor::default()
            .run_with_topology_and_completions([waiter, arrive_only], topology, &completions)
            .unwrap();
        assert_eq!(completed.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn missing_arrival_is_reported_as_a_cluster_barrier_deadlock() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let hub = Arc::new(ClusterBarrierHub::new(topology));
        let task = {
            let hub = Arc::clone(&hub);
            WarpTask::new(0, async move {
                hub.arrive(context)?;
                hub.wait(context)?.await?;
                Ok(())
            })
        };
        let mut completions = CompletionRegistry::new();
        completions.register(Arc::clone(&hub));

        let error = Executor::default()
            .run_with_topology_and_completions([task], topology, &completions)
            .unwrap_err();
        let crate::EngineErrorKind::Deadlock {
            blocked_operations, ..
        } = error.kind()
        else {
            panic!("expected cluster barrier deadlock");
        };
        assert_eq!(blocked_operations.len(), 1);
        assert!(blocked_operations[0]
            .to_string()
            .contains("barrier.cluster.wait"));
        assert_eq!(blocked_operations[0].phase, Some(0));
        assert_eq!(blocked_operations[0].participant_state.arrived, vec![0]);
        assert_eq!(blocked_operations[0].participant_state.missing, vec![1]);
    }

    #[test]
    fn arrive_and_wait_require_a_full_warp() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology
            .warp_contexts()
            .next()
            .unwrap()
            .with_active_mask(WarpMask::from_lanes(0..31).unwrap());
        let hub = Arc::new(ClusterBarrierHub::new(topology));

        assert!(matches!(
            hub.arrive(context),
            Err(SynchronizationError::PartialWarpSynchronization {
                operation: ClusterBarrierOperation::Arrive,
                ..
            })
        ));
        assert!(matches!(
            hub.wait(context),
            Err(SynchronizationError::PartialWarpSynchronization {
                operation: ClusterBarrierOperation::Wait,
                ..
            })
        ));
    }

    #[test]
    fn unaligned_arrive_accumulates_disjoint_lane_paths() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let full = topology.warp_contexts().next().unwrap();
        let lower = full.with_active_mask(WarpMask::from_lanes(0..16).unwrap());
        let upper = full.with_active_mask(WarpMask::from_lanes(16..32).unwrap());
        let hub = Arc::new(ClusterBarrierHub::new(topology));

        assert_eq!(
            hub.arrive_with_alignment(lower, false)
                .unwrap()
                .generation(),
            0
        );
        assert_eq!(
            hub.arrive_with_alignment(upper, false)
                .unwrap()
                .generation(),
            0
        );
        assert!(hub.wait_with_alignment(full, false).is_ok());
    }

    #[test]
    fn a_warp_cannot_arrive_twice_before_the_generation_completes() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let hub = ClusterBarrierHub::new(topology);

        assert_eq!(hub.arrive(context).unwrap().generation(), 0);
        assert!(matches!(
            hub.arrive(context),
            Err(SynchronizationError::DuplicateArrival {
                phase: 0,
                warp_id: 0,
                ..
            })
        ));
    }

    #[test]
    fn launch_cluster_barrier_contract_includes_only_selected_warps() {
        let topology = LaunchTopology::new(1, 2, 2).unwrap();
        let contexts = topology.warp_contexts().take(2).collect::<Vec<_>>();
        let hub = Arc::new(ClusterBarrierHub::for_launch(
            topology,
            Arc::new(BTreeSet::from([0, 1])),
        ));
        let tasks = contexts.into_iter().map(|context| {
            let hub = Arc::clone(&hub);
            WarpTask::new(context.global_warp_id(), async move {
                hub.arrive(context)?;
                hub.wait(context)?.await?;
                Ok(())
            })
        });
        let mut completions = CompletionRegistry::new();
        completions.register(Arc::clone(&hub));

        let stats = Executor::default()
            .run_with_topology_and_completions(tasks, topology, &completions)
            .unwrap();
        assert_eq!(stats.completed_task_count, 2);
    }
}
