use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::panic::{self, AssertUnwindSafe};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, JoinHandle};

use numsim_fp_env::set_round_to_nearest;

use crate::scheduling::{current_wake_priority, PriorityReadyQueue, ReadyOrder, WakePriority};
use crate::worker_affinity::{
    WorkerAffinityGuard, WorkerAffinityPlan, WorkerAffinityPolicy, WorkerAffinityRelease,
    WORKER_THREAD_PREFIX,
};
use crate::{
    AddressSpaceError, BlockedOperation, CompletionRegistry, CompletionRegistryError,
    CompletionSource, LaunchTopology, MemoryError, OperationContext, ProfileKind, ProfileTimer,
    SynchronizationError, WARP_SIZE,
};

// Very large generated artifacts deliberately compile at opt-level=0 to keep
// cold builds tractable. Rust then preserves large fixed-size kernel state and
// clone temporaries as stack slots; real kernels have produced nested frames
// larger than the platform's usual spawned-thread stack. Native targets reserve
// this virtual range and commit pages on demand. Preserve a larger caller-set
// RUST_MIN_STACK because Builder::stack_size would otherwise override it.
const MIN_NATIVE_WORKER_STACK_BYTES: usize = 16 * 1024 * 1024;
const MAX_LOCAL_POLLS_BEFORE_PENDING_CLUSTER_CLAIM: usize = 128;

fn native_worker_stack_bytes(configured: Option<&str>) -> usize {
    configured
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
        .max(MIN_NATIVE_WORKER_STACK_BYTES)
}

fn configured_native_worker_stack_bytes() -> usize {
    let configured = std::env::var("RUST_MIN_STACK").ok();
    native_worker_stack_bytes(configured.as_deref())
}

fn pending_cluster_claim_poll_interval(worker_count: usize) -> usize {
    if worker_count <= 1 {
        // One worker has no peer that can claim a producer cluster for it.
        return 1;
    }
    // Keep a normal-ready cluster local long enough to preserve its runtime
    // and analysis working sets. Poll-recheck work claims a pending peer
    // immediately below, while this bound preserves fairness for a cluster
    // that remains normal-ready for an unusually long time.
    MAX_LOCAL_POLLS_BEFORE_PENDING_CLUSTER_CLAIM
}

pub(crate) type WarpFuture =
    Pin<Box<dyn Future<Output = Result<(), EngineError>> + Send + 'static>>;

/// Stable display prefix retained when a compatibility caller has only the
/// rendered error.
pub const OUT_OF_BOUNDS_ERROR_PREFIX: &str = "out-of-bounds access: ";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutOfBoundsError {
    details: String,
}

impl OutOfBoundsError {
    pub fn new(details: impl Into<String>) -> Self {
        Self {
            details: details.into(),
        }
    }

    pub fn details(&self) -> &str {
        &self.details
    }

    fn with_context(self, context: impl fmt::Display) -> Self {
        Self::new(format!("{context}{}", self.details))
    }
}

impl fmt::Display for OutOfBoundsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{OUT_OF_BOUNDS_ERROR_PREFIX}{}", self.details)
    }
}

impl Error for OutOfBoundsError {}

pub(crate) fn native_worker_builder(name: String) -> thread::Builder {
    thread::Builder::new()
        .name(name)
        .stack_size(configured_native_worker_stack_bytes())
}

/// A single stackless task for one concrete warp.
pub struct WarpTask {
    warp_id: usize,
    future: WarpFuture,
}

impl WarpTask {
    pub fn new<F>(warp_id: usize, future: F) -> Self
    where
        F: Future<Output = Result<(), EngineError>> + Send + 'static,
    {
        Self {
            warp_id,
            future: Box::pin(future),
        }
    }

    pub const fn warp_id(&self) -> usize {
        self.warp_id
    }

    pub(crate) fn into_parts(self) -> (usize, WarpFuture) {
        (self.warp_id, self.future)
    }
}

/// Scheduler-independent ownership and polling of warp futures.
///
/// The executor composes this core with launch-wide cluster scheduling.
pub(crate) struct WarpRuntimeCore {
    ready: Arc<PriorityReadyQueue>,
    tasks: BTreeMap<usize, Option<WarpFuture>>,
    completed_task_count: usize,
    #[cfg(feature = "profile")]
    instruction_profile: Option<crate::instruction_profile::InstructionProfile>,
}

pub(crate) enum CorePollResult {
    Completed,
    Pending { self_ready: bool },
    Failed(EngineError),
    Panicked(String),
}

impl WarpRuntimeCore {
    pub(crate) fn new(
        tasks: impl IntoIterator<Item = WarpTask>,
        ready_order: ReadyOrder,
    ) -> Result<Self, EngineError> {
        let mut futures = BTreeMap::new();
        for task in tasks {
            let (warp_id, future) = task.into_parts();
            if futures.insert(warp_id, Some(future)).is_some() {
                return Err(EngineError::duplicate_warp_id(warp_id));
            }
        }
        Ok(Self {
            ready: Arc::new(PriorityReadyQueue::new(
                futures.keys().copied(),
                ready_order,
            )),
            tasks: futures,
            completed_task_count: 0,
            #[cfg(feature = "profile")]
            instruction_profile: crate::instruction_profile::InstructionProfile::current(),
        })
    }

    pub(crate) fn ready_handle(&self) -> Arc<PriorityReadyQueue> {
        Arc::clone(&self.ready)
    }

    pub(crate) fn pop_ready_with_priority(&self) -> Option<(usize, WakePriority)> {
        self.ready.pop_with_priority()
    }

    pub(crate) fn has_ready(&self) -> bool {
        self.ready.has_ready()
    }

    pub(crate) fn highest_ready_priority(&self) -> Option<WakePriority> {
        self.ready.highest_priority()
    }

    pub(crate) fn is_pending(&self, warp_id: usize) -> bool {
        self.tasks
            .get(&warp_id)
            .is_some_and(|future| future.is_some())
    }

    pub(crate) fn poll(&mut self, warp_id: usize, waker: &Waker) -> Option<CorePollResult> {
        #[cfg(feature = "profile")]
        let _instructions =
            crate::instruction_profile::InstructionProfile::enter(self.instruction_profile.clone());
        let poll_result = {
            let future = self.tasks.get_mut(&warp_id).and_then(Option::as_mut)?;
            let mut context = Context::from_waker(waker);
            let _profile_timer = ProfileTimer::new(ProfileKind::FuturePoll);
            panic::catch_unwind(AssertUnwindSafe(|| future.as_mut().poll(&mut context)))
        };
        Some(match poll_result {
            Ok(Poll::Ready(Ok(()))) => {
                *self.tasks.get_mut(&warp_id).expect("polled task exists") = None;
                self.ready.retire(warp_id);
                self.completed_task_count += 1;
                CorePollResult::Completed
            }
            Ok(Poll::Ready(Err(error))) => CorePollResult::Failed(error),
            Ok(Poll::Pending) => CorePollResult::Pending {
                self_ready: self.ready.contains(warp_id),
            },
            Err(payload) => CorePollResult::Panicked(panic_payload_message(payload.as_ref())),
        })
    }

    pub(crate) fn task_count(&self) -> usize {
        self.tasks.len()
    }

    pub(crate) fn completed_task_count(&self) -> usize {
        self.completed_task_count
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.completed_task_count == self.tasks.len()
    }

    pub(crate) fn pending_warps(&self) -> Vec<usize> {
        self.tasks
            .iter()
            .filter_map(|(warp_id, future)| future.is_some().then_some(*warp_id))
            .collect()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionStats {
    pub task_count: usize,
    pub completed_task_count: usize,
    pub poll_count: usize,
    pub(crate) normal_poll_count: usize,
    pub(crate) poll_recheck_poll_count: usize,
    /// Stable cluster-major diagnostic order. With multiple workers this is
    /// intentionally not a wall-clock interleaving, which would require a
    /// contended global sequence counter on every warp poll.
    pub poll_order: Vec<usize>,
    pub completion_pump_count: usize,
    pub completion_operation_count: usize,
    pub worker_count: usize,
    pub scheduling_domain_count: usize,
    #[cfg(feature = "profile")]
    pub executed_instruction_variants: Vec<(u64, &'static str)>,
}

/// One ordinary-executor run, including partial statistics when execution
/// terminates with a typed engine error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionReport {
    pub stats: ExecutionStats,
    pub terminal: Result<(), EngineError>,
}

impl ExecutionReport {
    pub fn success(stats: ExecutionStats) -> Self {
        Self {
            stats,
            terminal: Ok(()),
        }
    }

    pub fn failure(stats: ExecutionStats, error: EngineError) -> Self {
        Self {
            stats,
            terminal: Err(error),
        }
    }

    pub const fn is_success(&self) -> bool {
        self.terminal.is_ok()
    }

    pub fn error(&self) -> Option<&EngineError> {
        self.terminal.as_ref().err()
    }

    pub fn into_result(self) -> Result<ExecutionStats, EngineError> {
        self.terminal.map(|()| self.stats)
    }

    pub fn cluster_barrier_participant_exit_evidence(
        &self,
    ) -> Box<[ClusterBarrierParticipantExitEvidence]> {
        self.error()
            .map(EngineError::cluster_barrier_participant_exit_evidence)
            .unwrap_or_default()
    }
}

/// Cluster-parallel executor backed by a fixed pool of worker threads.
///
/// Each cluster owns an independent FIFO of runnable warps and remains pinned
/// to one worker for the complete launch. Workers advance their clusters
/// autonomously, so progress in one cluster never waits for a polling wave in
/// another cluster. Without topology, all tasks form one serial cluster domain.
#[derive(Clone, Copy, Debug)]
pub struct Executor {
    poll_limit: Option<usize>,
    max_workers: usize,
}

impl Default for Executor {
    fn default() -> Self {
        Self {
            poll_limit: None,
            max_workers: 1,
        }
    }
}

impl Executor {
    pub const fn with_poll_limit(poll_limit: usize) -> Self {
        Self {
            poll_limit: Some(poll_limit),
            max_workers: 1,
        }
    }

    pub const fn with_max_workers(max_workers: usize) -> Self {
        Self {
            poll_limit: None,
            max_workers: if max_workers == 0 { 1 } else { max_workers },
        }
    }

    pub const fn with_max_workers_and_poll_limit(
        max_workers: usize,
        poll_limit: Option<usize>,
    ) -> Self {
        Self {
            poll_limit,
            max_workers: if max_workers == 0 { 1 } else { max_workers },
        }
    }

    pub fn run(
        &mut self,
        tasks: impl IntoIterator<Item = WarpTask>,
    ) -> Result<ExecutionStats, EngineError> {
        self.run_report(tasks).into_result()
    }

    pub fn run_report(&mut self, tasks: impl IntoIterator<Item = WarpTask>) -> ExecutionReport {
        self.run_with_completions_report(tasks, &CompletionRegistry::new())
    }

    /// Run warp tasks and invoke registered completion sources before declaring
    /// a no-ready-task state to be a deadlock.
    pub fn run_with_completions(
        &mut self,
        tasks: impl IntoIterator<Item = WarpTask>,
        completions: &CompletionRegistry,
    ) -> Result<ExecutionStats, EngineError> {
        self.run_with_completions_report(tasks, completions)
            .into_result()
    }

    pub fn run_with_completions_report(
        &mut self,
        tasks: impl IntoIterator<Item = WarpTask>,
        completions: &CompletionRegistry,
    ) -> ExecutionReport {
        self.run_internal_report(tasks, None, completions)
    }

    /// Run tasks with each cluster kept as an indivisible scheduling domain.
    pub fn run_with_topology(
        &mut self,
        tasks: impl IntoIterator<Item = WarpTask>,
        topology: LaunchTopology,
    ) -> Result<ExecutionStats, EngineError> {
        self.run_with_topology_report(tasks, topology).into_result()
    }

    pub fn run_with_topology_report(
        &mut self,
        tasks: impl IntoIterator<Item = WarpTask>,
        topology: LaunchTopology,
    ) -> ExecutionReport {
        self.run_with_topology_and_completions_report(tasks, topology, &CompletionRegistry::new())
    }

    /// Run tasks with topology-aware cluster scheduling and completion sources.
    ///
    /// A cluster and all of its CTAs remain assigned to one worker for the
    /// complete launch. Different clusters may execute concurrently.
    pub fn run_with_topology_and_completions(
        &mut self,
        tasks: impl IntoIterator<Item = WarpTask>,
        topology: LaunchTopology,
        completions: &CompletionRegistry,
    ) -> Result<ExecutionStats, EngineError> {
        self.run_with_topology_and_completions_report(tasks, topology, completions)
            .into_result()
    }

    pub fn run_with_topology_and_completions_report(
        &mut self,
        tasks: impl IntoIterator<Item = WarpTask>,
        topology: LaunchTopology,
        completions: &CompletionRegistry,
    ) -> ExecutionReport {
        self.run_internal_report(tasks, Some(topology), completions)
    }

    fn run_internal_report(
        &mut self,
        tasks: impl IntoIterator<Item = WarpTask>,
        topology: Option<LaunchTopology>,
        completions: &CompletionRegistry,
    ) -> ExecutionReport {
        #[cfg(feature = "profile")]
        let instruction_profile = crate::instruction_profile::InstructionProfile::default();
        #[cfg(feature = "profile")]
        let _instructions = crate::instruction_profile::InstructionProfile::enter(Some(
            instruction_profile.clone(),
        ));
        let mut task_ids = BTreeSet::new();
        let mut domains = BTreeMap::<usize, Vec<WarpTask>>::new();

        for task in tasks {
            let warp_id = task.warp_id;
            if !task_ids.insert(warp_id) {
                return ExecutionReport::failure(
                    ExecutionStats::default(),
                    EngineError::duplicate_warp_id(warp_id),
                );
            }
            let domain_id = match topology {
                Some(topology) => match topology.cluster_id_for_warp(warp_id) {
                    Some(cluster_id) => cluster_id,
                    None => {
                        return ExecutionReport::failure(
                            ExecutionStats::default(),
                            EngineError::warp_outside_topology(warp_id, topology.warp_count()),
                        );
                    }
                },
                None => 0,
            };
            domains.entry(domain_id).or_default().push(task);
        }

        let scheduling_domain_count = domains.len();
        let worker_count = self.max_workers.min(scheduling_domain_count);
        let mut stats = ExecutionStats {
            task_count: task_ids.len(),
            worker_count,
            scheduling_domain_count,
            ..ExecutionStats::default()
        };
        if task_ids.is_empty() {
            return ExecutionReport::success(stats);
        }
        let max_clusters_per_worker = scheduling_domain_count.div_ceil(worker_count);

        let mut clusters = domains
            .into_iter()
            .map(|(cluster_id, domain_tasks)| ClusterRuntime::new(cluster_id, domain_tasks));
        let initial_clusters = clusters.by_ref().take(worker_count).collect::<Vec<_>>();
        let pending_clusters = Arc::new(PendingClusterQueue::new(clusters));
        let completion_sources = completions.sources().to_vec();
        let control = Arc::new(RunControl::new(
            worker_count,
            completion_sources,
            self.poll_limit,
            scheduling_domain_count,
        ));
        let reports = {
            // Workers inherit the launching thread's mask; it is restored when
            // the guard drops after the pool has joined.
            let affinity = WorkerAffinityGuard::apply(&WorkerAffinityPlan::for_workers(
                &WorkerAffinityPolicy::from_env(),
                initial_clusters.len(),
            ));
            match WorkerPool::new(
                initial_clusters,
                Arc::clone(&pending_clusters),
                Arc::clone(&control),
                max_clusters_per_worker,
                affinity.as_ref().map(WorkerAffinityGuard::release),
            ) {
                Ok(pool) => pool.join(),
                Err(error) => return ExecutionReport::failure(stats, error),
            }
        };
        let unclaimed_pending_warps = pending_clusters.pending_warps();

        for report in &reports {
            stats.completed_task_count += report.completed_task_count;
            stats.poll_count += report.poll_count;
            stats.normal_poll_count += report.normal_poll_count;
            stats.poll_recheck_poll_count += report.poll_recheck_poll_count;
            stats.completion_pump_count += report.completion_pump_count;
            stats.completion_operation_count += report.completion_operation_count;
        }
        let mut cluster_poll_orders = reports
            .iter()
            .flat_map(|report| report.cluster_poll_orders.iter())
            .collect::<Vec<_>>();
        cluster_poll_orders.sort_by_key(|(cluster_id, _)| *cluster_id);
        for (_, poll_order) in cluster_poll_orders {
            stats.poll_order.extend(poll_order.iter().copied());
        }

        let terminal = match control.stop_reason() {
            Some(RunStop::Error(error)) => Err(error),
            Some(RunStop::Deadlock { blocked_operations }) => {
                let mut blocked_warps = reports
                    .iter()
                    .flat_map(|report| report.pending_warps.iter().copied())
                    .collect::<Vec<_>>();
                blocked_warps.extend(unclaimed_pending_warps.iter().copied());
                blocked_warps.sort_unstable();
                Err(EngineError::deadlock(
                    blocked_warps,
                    blocked_operations,
                    stats.poll_count,
                ))
            }
            Some(RunStop::PollLimit { limit }) => {
                let mut pending_warps = reports
                    .iter()
                    .flat_map(|report| report.pending_warps.iter().copied())
                    .collect::<Vec<_>>();
                pending_warps.extend(unclaimed_pending_warps.iter().copied());
                pending_warps.sort_unstable();
                Err(EngineError::poll_limit_exceeded(limit, pending_warps))
            }
            None => Ok(()),
        };

        #[cfg(feature = "profile")]
        {
            stats.executed_instruction_variants = instruction_profile.snapshot();
        }
        ExecutionReport { stats, terminal }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClusterBarrierParticipantExitEvidence {
    cluster_id: usize,
    generation: u64,
    exited_warps: Box<[usize]>,
}

impl ClusterBarrierParticipantExitEvidence {
    pub const fn cluster_id(&self) -> usize {
        self.cluster_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn exited_warps(&self) -> &[usize] {
        &self.exited_warps
    }
}

/// Opaque error returned across the generated-artifact boundary.
///
/// Generated code may construct the two source-owned failures exposed below,
/// but diagnostic classification and payloads remain engine implementation
/// details.  In particular, adding a checker-only failure must not add a
/// public enum variant to the artifact ABI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineError(EngineErrorKind);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EngineErrorKind {
    DuplicateWarpId {
        warp_id: usize,
    },
    WarpOutsideTopology {
        warp_id: usize,
        warp_count: usize,
    },
    Deadlock {
        blocked_warps: Vec<usize>,
        blocked_operations: Vec<BlockedOperation>,
        poll_count: usize,
    },
    PollLimitExceeded {
        limit: usize,
        pending_warps: Vec<usize>,
    },
    NativeLoopIterationLimitExceeded {
        label: String,
        limit: usize,
    },
    Context {
        context: String,
        operation: Option<OperationContext>,
        source: Box<EngineError>,
    },
    WarpFailed {
        warp_id: usize,
        source: Box<EngineError>,
    },
    WarpPanicked {
        warp_id: usize,
        message: String,
    },
    WorkerFailed {
        worker_id: usize,
        message: String,
    },
    Memory(MemoryError),
    AddressSpace(AddressSpaceError),
    Synchronization(Box<SynchronizationError>),
    CompletionFailed {
        source_name: String,
        source: Box<SynchronizationError>,
    },
    TmemAccess(crate::TmemAccessError),
    WarpCollectiveDivergence {
        operation: String,
        active_mask: u32,
    },
    AsyncGroupIssuerViolation {
        operation: String,
        expected_lane_count: usize,
        active_mask: u32,
    },
    TmaSharedAddressMisaligned {
        component_index: usize,
        byte_offset: i64,
    },
    AnalysisIncomplete {
        kind: &'static str,
    },
    OutOfBounds(OutOfBoundsError),
    Message(String),
}

impl EngineError {
    pub(crate) const fn from_kind(kind: EngineErrorKind) -> Self {
        Self(kind)
    }

    pub(crate) const fn kind(&self) -> &EngineErrorKind {
        &self.0
    }

    pub fn message(message: impl Into<String>) -> Self {
        Self::from_kind(EngineErrorKind::Message(message.into()))
    }

    pub fn native_loop_iteration_limit(label: impl Into<String>, limit: usize) -> Self {
        Self::from_kind(EngineErrorKind::NativeLoopIterationLimitExceeded {
            label: label.into(),
            limit,
        })
    }

    pub(crate) const fn duplicate_warp_id(warp_id: usize) -> Self {
        Self::from_kind(EngineErrorKind::DuplicateWarpId { warp_id })
    }

    pub(crate) const fn warp_outside_topology(warp_id: usize, warp_count: usize) -> Self {
        Self::from_kind(EngineErrorKind::WarpOutsideTopology {
            warp_id,
            warp_count,
        })
    }

    pub(crate) fn deadlock(
        blocked_warps: Vec<usize>,
        blocked_operations: Vec<BlockedOperation>,
        poll_count: usize,
    ) -> Self {
        Self::from_kind(EngineErrorKind::Deadlock {
            blocked_warps,
            blocked_operations,
            poll_count,
        })
    }

    pub(crate) fn poll_limit_exceeded(limit: usize, pending_warps: Vec<usize>) -> Self {
        Self::from_kind(EngineErrorKind::PollLimitExceeded {
            limit,
            pending_warps,
        })
    }

    pub(crate) fn warp_failed(warp_id: usize, source: Self) -> Self {
        Self::from_kind(EngineErrorKind::WarpFailed {
            warp_id,
            source: Box::new(source),
        })
    }

    pub(crate) fn warp_panicked(warp_id: usize, message: impl Into<String>) -> Self {
        Self::from_kind(EngineErrorKind::WarpPanicked {
            warp_id,
            message: message.into(),
        })
    }

    pub(crate) fn worker_failed(worker_id: usize, message: impl Into<String>) -> Self {
        Self::from_kind(EngineErrorKind::WorkerFailed {
            worker_id,
            message: message.into(),
        })
    }

    pub(crate) fn completion_failed(
        source_name: impl Into<String>,
        source: SynchronizationError,
    ) -> Self {
        Self::from_kind(EngineErrorKind::CompletionFailed {
            source_name: source_name.into(),
            source: Box::new(source),
        })
    }

    pub(crate) fn synchronization(source: SynchronizationError) -> Self {
        Self::from_kind(EngineErrorKind::Synchronization(Box::new(source)))
    }

    pub(crate) fn warp_collective_divergence(
        operation: impl Into<String>,
        active_mask: u32,
    ) -> Self {
        Self::from_kind(EngineErrorKind::WarpCollectiveDivergence {
            operation: operation.into(),
            active_mask,
        })
    }

    pub(crate) fn async_group_issuer_violation(
        operation: impl Into<String>,
        expected_lane_count: usize,
        active_mask: u32,
    ) -> Self {
        Self::from_kind(EngineErrorKind::AsyncGroupIssuerViolation {
            operation: operation.into(),
            expected_lane_count,
            active_mask,
        })
    }

    pub(crate) const fn tma_shared_address_misaligned(
        component_index: usize,
        byte_offset: i64,
    ) -> Self {
        Self::from_kind(EngineErrorKind::TmaSharedAddressMisaligned {
            component_index,
            byte_offset,
        })
    }

    pub(crate) const fn analysis_incomplete(kind: &'static str) -> Self {
        Self::from_kind(EngineErrorKind::AnalysisIncomplete { kind })
    }

    pub(crate) fn out_of_bounds(details: impl Into<String>) -> Self {
        Self::from_kind(EngineErrorKind::OutOfBounds(OutOfBoundsError::new(details)))
    }

    pub(crate) fn is_out_of_bounds(&self) -> bool {
        match self.kind() {
            EngineErrorKind::OutOfBounds(_) => true,
            EngineErrorKind::Context { source, .. }
            | EngineErrorKind::WarpFailed { source, .. } => source.is_out_of_bounds(),
            EngineErrorKind::Memory(
                MemoryError::ViewOutOfBounds { .. }
                | MemoryError::AccessOutOfBounds { .. }
                | MemoryError::OffsetOverflow,
            ) => true,
            _ => false,
        }
    }

    pub(crate) fn with_context(self, context: impl fmt::Display) -> Self {
        match self.0 {
            EngineErrorKind::OutOfBounds(error) => {
                Self::from_kind(EngineErrorKind::OutOfBounds(error.with_context(context)))
            }
            EngineErrorKind::Message(message) => Self::message(format!("{context}{message}")),
            kind => Self::from_kind(EngineErrorKind::Context {
                context: context.to_string(),
                operation: None,
                source: Box::new(Self::from_kind(kind)),
            }),
        }
    }

    pub(crate) fn with_operation_context(self, operation: &OperationContext) -> Self {
        if self.operation_context().is_some() {
            return self;
        }
        Self::from_kind(EngineErrorKind::Context {
            context: format!("{operation}: "),
            operation: Some(operation.clone()),
            source: Box::new(self),
        })
    }

    pub(crate) const fn operation_context(&self) -> Option<&OperationContext> {
        match self.kind() {
            EngineErrorKind::Context {
                operation: Some(operation),
                ..
            } => Some(operation),
            EngineErrorKind::Context { source, .. }
            | EngineErrorKind::WarpFailed { source, .. } => source.operation_context(),
            _ => None,
        }
    }

    pub(crate) fn cluster_barrier_participant_exit_evidence(
        &self,
    ) -> Box<[ClusterBarrierParticipantExitEvidence]> {
        let EngineErrorKind::Deadlock {
            blocked_warps,
            blocked_operations,
            ..
        } = self.kind()
        else {
            return Box::default();
        };
        if blocked_operations.is_empty()
            || blocked_operations.iter().any(|blocked| {
                blocked.awaited_operation() != crate::AwaitedOperation::ClusterBarrierWait
            })
            || blocked_warps.iter().any(|warp_id| {
                !blocked_operations
                    .iter()
                    .any(|blocked| blocked.warp_id == *warp_id)
            })
        {
            return Box::default();
        }

        let mut evidence = BTreeMap::<(usize, u64), BTreeSet<usize>>::new();
        for blocked in blocked_operations {
            let crate::ScopeInstance::Cluster { cluster_id } = blocked.key.scope() else {
                return Box::default();
            };
            let Some(generation) = blocked.phase else {
                return Box::default();
            };
            evidence
                .entry((*cluster_id, generation))
                .or_default()
                .extend(
                    blocked
                        .participant_state
                        .missing
                        .iter()
                        .copied()
                        .filter(|warp_id| !blocked_warps.contains(warp_id)),
                );
        }
        evidence
            .into_iter()
            .filter(|(_, exited_warps)| !exited_warps.is_empty())
            .map(
                |((cluster_id, generation), exited_warps)| ClusterBarrierParticipantExitEvidence {
                    cluster_id,
                    generation,
                    exited_warps: exited_warps.into_iter().collect(),
                },
            )
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

impl From<CompletionRegistryError> for EngineError {
    fn from(error: CompletionRegistryError) -> Self {
        Self::from_kind(EngineErrorKind::CompletionFailed {
            source_name: error.source_name().to_string(),
            source: error.into_source(),
        })
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind() {
            EngineErrorKind::DuplicateWarpId { warp_id } => {
                write!(f, "duplicate warp task ID {warp_id}")
            }
            EngineErrorKind::WarpOutsideTopology {
                warp_id,
                warp_count,
            } => write!(
                f,
                "warp task ID {warp_id} is outside launch topology with {warp_count} warps"
            ),
            EngineErrorKind::Deadlock {
                blocked_warps,
                blocked_operations,
                poll_count,
            } => {
                write!(
                    f,
                    "executor deadlock after {poll_count} polls; blocked warps: {blocked_warps:?}"
                )?;
                for blocked in blocked_operations {
                    write!(f, "; {blocked}")?;
                }
                Ok(())
            }
            EngineErrorKind::PollLimitExceeded {
                limit,
                pending_warps,
            } => write!(
                f,
                "executor poll limit {limit} exceeded; pending warps: {pending_warps:?}"
            ),
            EngineErrorKind::NativeLoopIterationLimitExceeded { label, limit } => write!(
                f,
                "generated {label} exceeded configured native loop iteration budget {limit}"
            ),
            EngineErrorKind::Context {
                context, source, ..
            } => write!(f, "{context}{source}"),
            EngineErrorKind::WarpFailed { warp_id, source } => {
                write!(f, "warp {warp_id} failed: {source}")
            }
            EngineErrorKind::WarpPanicked { warp_id, message } => {
                write!(f, "warp {warp_id} panicked while being polled: {message}")
            }
            EngineErrorKind::WorkerFailed { worker_id, message } => {
                write!(f, "executor worker {worker_id} failed: {message}")
            }
            EngineErrorKind::Memory(source) if self.is_out_of_bounds() => {
                write!(f, "{OUT_OF_BOUNDS_ERROR_PREFIX}{source}")
            }
            EngineErrorKind::Memory(source) => write!(f, "global-memory error: {source}"),
            EngineErrorKind::AddressSpace(source) => {
                write!(f, "physical address-space error: {source}")
            }
            EngineErrorKind::Synchronization(source) => {
                write!(f, "synchronization error: {source}")
            }
            EngineErrorKind::CompletionFailed {
                source_name,
                source,
            } => write!(f, "completion source {source_name} failed: {source}"),
            EngineErrorKind::TmemAccess(source) => write!(f, "TMEM access error: {source}"),
            EngineErrorKind::WarpCollectiveDivergence {
                operation,
                active_mask,
            } => write!(
                f,
                "{operation} requires all {WARP_SIZE} lanes, got mask 0x{active_mask:08x}"
            ),
            EngineErrorKind::AsyncGroupIssuerViolation {
                operation,
                expected_lane_count,
                active_mask,
            } => write!(
                f,
                "async-group issue {operation} requires {expected_lane_count} active issuing lane, got mask 0x{active_mask:08x}"
            ),
            EngineErrorKind::TmaSharedAddressMisaligned {
                component_index,
                byte_offset,
            } => write!(
                f,
                "TMA shared payload component {component_index} byte offset {byte_offset} must be 128-byte aligned"
            ),
            EngineErrorKind::AnalysisIncomplete { kind } => {
                write!(f, "{kind} requires an unmodeled analysis contract")
            }
            EngineErrorKind::OutOfBounds(source) => source.fmt(f),
            EngineErrorKind::Message(message) => f.write_str(message),
        }
    }
}

impl Error for EngineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self.kind() {
            EngineErrorKind::Context { source, .. }
            | EngineErrorKind::WarpFailed { source, .. } => Some(source.as_ref()),
            EngineErrorKind::Memory(source) => Some(source),
            EngineErrorKind::AddressSpace(source) => Some(source),
            EngineErrorKind::Synchronization(source) => Some(source.as_ref()),
            EngineErrorKind::CompletionFailed { source, .. } => Some(source.as_ref()),
            EngineErrorKind::TmemAccess(source) => Some(source),
            EngineErrorKind::OutOfBounds(source) => Some(source),
            _ => None,
        }
    }
}

/// Stable public diagnostic kind for a typed engine failure.
///
/// Wrapper variants preserve the innermost typed cause. The match is
/// intentionally exhaustive so adding a runtime error cannot silently fall
/// back to a rendered-string classification.
pub(crate) fn engine_error_kind(error: &EngineError) -> &'static str {
    if error.is_out_of_bounds() {
        return "oob";
    }
    match error.kind() {
        EngineErrorKind::DuplicateWarpId { .. } => "duplicate_warp_id",
        EngineErrorKind::WarpOutsideTopology { .. } => "warp_outside_topology",
        EngineErrorKind::Deadlock {
            blocked_operations, ..
        } if blocked_operations.iter().any(|blocked| {
            blocked.awaited_operation() == crate::AwaitedOperation::SetmaxnregPool
        }) =>
        {
            "setmaxnreg_pool_deadlock"
        }
        EngineErrorKind::Deadlock {
            blocked_operations, ..
        } if blocked_operations
            .iter()
            .any(|blocked| blocked.awaited_operation() == crate::AwaitedOperation::Setmaxnreg) =>
        {
            "setmaxnreg_divergence"
        }
        EngineErrorKind::Deadlock { .. } => "deadlock",
        EngineErrorKind::PollLimitExceeded { .. } => "poll_limit",
        EngineErrorKind::NativeLoopIterationLimitExceeded { .. } => "native_loop_iteration_limit",
        EngineErrorKind::Context { source, .. } => engine_error_kind(source),
        EngineErrorKind::WarpFailed { source, .. } => engine_error_kind(source),
        EngineErrorKind::WarpPanicked { .. } => "warp_panicked",
        EngineErrorKind::WorkerFailed { .. } => "worker_failed",
        EngineErrorKind::Memory(_) => "memory_error",
        EngineErrorKind::AddressSpace(_) => "address_space_error",
        EngineErrorKind::Synchronization(source) => synchronization_error_kind(source),
        EngineErrorKind::CompletionFailed { source, .. } => synchronization_error_kind(source),
        EngineErrorKind::TmemAccess(source) => source.kind().name(),
        EngineErrorKind::WarpCollectiveDivergence { .. } => "warp_collective_divergence",
        EngineErrorKind::AsyncGroupIssuerViolation { .. } => "async_group_issuer_violation",
        EngineErrorKind::TmaSharedAddressMisaligned { .. } => "tma_shared_address_misaligned",
        EngineErrorKind::AnalysisIncomplete { .. } => "analysis_incomplete",
        EngineErrorKind::OutOfBounds(_) => "oob",
        EngineErrorKind::Message(_) => "engine_error",
    }
}

/// Stable public diagnostic kind for every synchronization-runtime failure.
///
/// Specialized protocols retain their established public names. Generic
/// coordination failures use a `synchronization_` prefix so they remain exact
/// without implying a more specific barrier protocol than the typed variant
/// carries.
pub(crate) fn synchronization_error_kind(error: &SynchronizationError) -> &'static str {
    match error {
        SynchronizationError::EmptyParticipantSet => "synchronization_empty_participant_set",
        SynchronizationError::InvalidWarpGroupWidth => "synchronization_invalid_warpgroup_width",
        SynchronizationError::KeyScopeMismatch { .. } => "synchronization_key_scope_mismatch",
        SynchronizationError::NotAParticipant { .. } => "synchronization_not_a_participant",
        SynchronizationError::ContractMismatch { .. } => "synchronization_contract_mismatch",
        SynchronizationError::PhaseOutOfOrder { .. } => "synchronization_phase_out_of_order",
        SynchronizationError::PhaseAlreadyArmed { .. } => "synchronization_phase_already_armed",
        SynchronizationError::PhaseNotArmed { .. } => "synchronization_phase_not_armed",
        SynchronizationError::PhaseAlreadyComplete { .. } => {
            "synchronization_phase_already_complete"
        }
        SynchronizationError::DuplicateArrival { key, .. }
            if matches!(key.scope(), crate::ScopeInstance::Cluster { .. }) =>
        {
            "cluster_barrier_early_arrival"
        }
        SynchronizationError::DuplicateArrival { .. } => "synchronization_duplicate_arrival",
        SynchronizationError::DuplicateContribution { .. } => {
            "synchronization_duplicate_contribution"
        }
        SynchronizationError::DuplicateWaiter { key, .. }
            if matches!(key.scope(), crate::ScopeInstance::Cluster { .. }) =>
        {
            "cluster_barrier_duplicate_wait"
        }
        SynchronizationError::DuplicateWaiter { .. } => "synchronization_duplicate_waiter",
        SynchronizationError::UndefinedOccurrence { .. } => "synchronization_undefined_occurrence",
        SynchronizationError::TransactionOverflow { .. } => "synchronization_transaction_overflow",
        SynchronizationError::InvalidBarrierArrivalCount { .. } => {
            "mbarrier_invalid_expected_arrivals"
        }
        SynchronizationError::DuplicateMbarrierArrivalTarget { .. } => {
            "mbarrier_duplicate_arrival_target"
        }
        SynchronizationError::MbarrierLocalArriveRemoteAddress { .. } => {
            "mbarrier_local_arrive_remote_address"
        }
        SynchronizationError::InvalidBarrierPhase { .. } => "mbarrier_invalid_phase",
        SynchronizationError::InvalidMbarrierStateToken { .. } => "mbarrier_invalid_state_token",
        SynchronizationError::BarrierUninitialized { .. } => "mbarrier_use_before_init",
        SynchronizationError::BarrierReinitializedWhileWaiting { .. } => {
            "mbarrier_reinit_before_consumption"
        }
        SynchronizationError::BarrierReinitializedWhileActive { .. } => {
            "mbarrier_reinit_while_active"
        }
        SynchronizationError::BarrierReinitializedWithoutInvalidation { .. } => {
            "mbarrier_reinit_without_inval"
        }
        SynchronizationError::BarrierArrivalOverflow { .. } => "mbarrier_arrival_overflow",
        SynchronizationError::PartialWarpSynchronization { .. } => "cluster_barrier_partial_warp",
        SynchronizationError::ClusterBarrierContextMismatch { .. } => {
            "cluster_barrier_context_mismatch"
        }
        SynchronizationError::ClusterBarrierWaitBeforeArrival { .. } => {
            "cluster_barrier_wait_before_arrival"
        }
        SynchronizationError::ClusterBarrierRearrivalWithoutWait { .. } => {
            "cluster_barrier_rearrival_without_wait"
        }
        SynchronizationError::ClusterBarrierPhaseOverflow { .. } => {
            "cluster_barrier_generation_overflow"
        }
        SynchronizationError::TcgenLifecycle(source) => source.kind().name(),
        SynchronizationError::Setmaxnreg(source) => source.kind().name(),
        SynchronizationError::CollectivePublication { .. } => {
            "synchronization_collective_publication"
        }
        SynchronizationError::CompletionSourceOperationFailed { .. } => {
            "completion_source_operation_failed"
        }
        SynchronizationError::CompletionSourceNotQuiescent { .. } => {
            "completion_source_not_quiescent"
        }
    }
}

impl From<MemoryError> for EngineError {
    fn from(value: MemoryError) -> Self {
        Self::from_kind(EngineErrorKind::Memory(value))
    }
}

impl From<AddressSpaceError> for EngineError {
    fn from(value: AddressSpaceError) -> Self {
        Self::from_kind(EngineErrorKind::AddressSpace(value))
    }
}

impl From<SynchronizationError> for EngineError {
    fn from(value: SynchronizationError) -> Self {
        Self::from_kind(EngineErrorKind::Synchronization(Box::new(value)))
    }
}

impl From<crate::TmemAccessError> for EngineError {
    fn from(value: crate::TmemAccessError) -> Self {
        Self::from_kind(EngineErrorKind::TmemAccess(value))
    }
}

type ClusterReadyQueue = PriorityReadyQueue;
type RunnableClusterQueue = PriorityReadyQueue;

struct WarpWaker {
    warp_id: usize,
    worker_id: usize,
    cluster_slot: usize,
    ready: Arc<ClusterReadyQueue>,
    runnable_clusters: Arc<RunnableClusterQueue>,
    control: Arc<RunControl>,
}

impl WarpWaker {
    fn schedule(&self) {
        let priority = current_wake_priority();
        if let Some(priority) = self.ready.schedule(self.warp_id, priority) {
            self.runnable_clusters.schedule(self.cluster_slot, priority);
            self.control.notify_worker(self.worker_id);
        }
    }
}

impl Wake for WarpWaker {
    fn wake(self: Arc<Self>) {
        self.schedule();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.schedule();
    }
}

#[derive(Clone)]
enum RunStop {
    Deadlock {
        blocked_operations: Vec<BlockedOperation>,
    },
    PollLimit {
        limit: usize,
    },
    Error(EngineError),
}

impl RunStop {
    /// Whether this stop carries a concrete worker failure rather than a
    /// quiescence observation (`Deadlock`, `PollLimit`).
    const fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WorkerStatus {
    Active,
    Idle,
    Complete,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompletionPumpAttempt {
    Pumped(bool),
    Contended { generation: usize },
}

struct RunControlState {
    workers: Vec<WorkerStatus>,
    stop: Option<RunStop>,
}

/// Launch-wide quiescence and failure state.
///
/// Every legal wake is produced by an active worker while polling a warp or
/// synchronously pumping a completion source. If background completion threads
/// are added later, deadlock detection must also account for those producers.
struct RunControl {
    stopped: AtomicBool,
    parked: Vec<AtomicBool>,
    poll_limit: Option<usize>,
    poll_count: AtomicUsize,
    completion_pump_interval: usize,
    completion_pump_active: AtomicBool,
    completion_pump_generation: AtomicUsize,
    completion_pump_waiters: AtomicUsize,
    completion_sources: Vec<Arc<dyn CompletionSource>>,
    state: Mutex<RunControlState>,
    changed: Vec<Condvar>,
}

impl RunControl {
    fn new(
        worker_count: usize,
        completion_sources: Vec<Arc<dyn CompletionSource>>,
        poll_limit: Option<usize>,
        scheduling_domain_count: usize,
    ) -> Self {
        Self {
            stopped: AtomicBool::new(false),
            parked: (0..worker_count).map(|_| AtomicBool::new(false)).collect(),
            poll_limit,
            poll_count: AtomicUsize::new(0),
            completion_pump_interval: scheduling_domain_count.max(1),
            completion_pump_active: AtomicBool::new(false),
            completion_pump_generation: AtomicUsize::new(0),
            completion_pump_waiters: AtomicUsize::new(0),
            completion_sources,
            state: Mutex::new(RunControlState {
                workers: vec![WorkerStatus::Active; worker_count],
                stop: None,
            }),
            changed: (0..worker_count).map(|_| Condvar::new()).collect(),
        }
    }

    fn is_stopped(&self) -> bool {
        self.stopped.load(AtomicOrdering::Acquire)
    }

    fn stop_reason(&self) -> Option<RunStop> {
        self.state
            .lock()
            .expect("run control mutex poisoned")
            .stop
            .clone()
    }

    fn reserve_poll(&self) -> Option<usize> {
        match self.poll_limit {
            Some(limit) => self
                .poll_count
                .fetch_update(
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Acquire,
                    |poll_count| (poll_count < limit).then_some(poll_count + 1),
                )
                .ok()
                .map(|poll_count| poll_count + 1),
            None => Some(self.poll_count.fetch_add(1, AtomicOrdering::Relaxed) + 1),
        }
    }

    fn should_pump_after(&self, poll_sequence: usize) -> bool {
        poll_sequence.is_multiple_of(self.completion_pump_interval)
    }

    fn stop_at_poll_limit(&self, worker_id: usize) {
        let mut state = self.state.lock().expect("run control mutex poisoned");
        state.workers[worker_id] = WorkerStatus::Failed;
        self.parked[worker_id].store(false, AtomicOrdering::Release);
        if state.stop.is_none() {
            state.stop = Some(RunStop::PollLimit {
                limit: self.poll_limit.expect("reserved poll limit must exist"),
            });
        }
        self.stopped.store(true, AtomicOrdering::Release);
        self.notify_all_workers();
    }

    fn pump_completions(
        &self,
        completion_pump_count: &mut usize,
        completion_operation_count: &mut usize,
    ) -> Result<CompletionPumpAttempt, EngineError> {
        if self.completion_sources.is_empty() {
            return Ok(CompletionPumpAttempt::Pumped(false));
        }
        if self
            .completion_pump_active
            .compare_exchange(
                false,
                true,
                AtomicOrdering::Acquire,
                AtomicOrdering::Relaxed,
            )
            .is_err()
        {
            return Ok(CompletionPumpAttempt::Contended {
                generation: self
                    .completion_pump_generation
                    .load(AtomicOrdering::Acquire),
            });
        }
        let _lease = CompletionPumpLease(self);
        pump_completion_round(
            &self.completion_sources,
            completion_pump_count,
            completion_operation_count,
        )
        .map(CompletionPumpAttempt::Pumped)
    }

    fn wait_for_completion_pump(&self, worker_id: usize, generation: usize) -> bool {
        self.completion_pump_waiters
            .fetch_add(1, AtomicOrdering::AcqRel);
        let mut state = self.state.lock().expect("run control mutex poisoned");
        while state.stop.is_none()
            && self.completion_pump_active.load(AtomicOrdering::Acquire)
            && self
                .completion_pump_generation
                .load(AtomicOrdering::Acquire)
                == generation
        {
            state = self.changed[worker_id]
                .wait(state)
                .expect("run control mutex poisoned while waiting for completion pump");
        }
        let running = state.stop.is_none();
        drop(state);
        self.completion_pump_waiters
            .fetch_sub(1, AtomicOrdering::AcqRel);
        running
    }

    fn finish_completion_pump(&self) {
        // Publish the generation before releasing ownership so a successor
        // cannot acquire the CAS while waiters still identify the old pump.
        self.completion_pump_generation
            .fetch_add(1, AtomicOrdering::AcqRel);
        self.completion_pump_active
            .store(false, AtomicOrdering::Release);
        if self.completion_pump_waiters.load(AtomicOrdering::Acquire) != 0 {
            let _state = self.state.lock().expect("run control mutex poisoned");
            self.notify_all_workers();
        }
    }

    fn notify_worker(&self, worker_id: usize) {
        if self.is_stopped() || !self.parked[worker_id].load(AtomicOrdering::Acquire) {
            return;
        }
        let mut state = self.state.lock().expect("run control mutex poisoned");
        if state.stop.is_some() || state.workers[worker_id] != WorkerStatus::Idle {
            return;
        }
        state.workers[worker_id] = WorkerStatus::Active;
        self.parked[worker_id].store(false, AtomicOrdering::Release);
        self.changed[worker_id].notify_one();
    }

    fn wait_for_work(&self, worker_id: usize, has_ready: impl Fn() -> bool) -> bool {
        let mut state = self.state.lock().expect("run control mutex poisoned");
        self.parked[worker_id].store(true, AtomicOrdering::Release);
        if state.stop.is_some() {
            self.parked[worker_id].store(false, AtomicOrdering::Release);
            return false;
        }
        if has_ready() {
            self.parked[worker_id].store(false, AtomicOrdering::Release);
            return true;
        }

        state.workers[worker_id] = WorkerStatus::Idle;
        if self.detect_deadlock(&mut state) {
            self.stopped.store(true, AtomicOrdering::Release);
        }
        if state.stop.is_some() {
            self.parked[worker_id].store(false, AtomicOrdering::Release);
            self.notify_all_workers();
            return false;
        }

        while state.stop.is_none() && state.workers[worker_id] == WorkerStatus::Idle {
            state = self.changed[worker_id]
                .wait(state)
                .expect("run control mutex poisoned while parked");
        }
        self.parked[worker_id].store(false, AtomicOrdering::Release);
        state.stop.is_none()
    }

    fn try_complete_worker(&self, worker_id: usize, completion_checked: bool) -> bool {
        let mut state = self.state.lock().expect("run control mutex poisoned");
        if !completion_checked
            && (self.completion_pump_active.load(AtomicOrdering::Acquire)
                || state
                    .workers
                    .iter()
                    .enumerate()
                    .any(|(peer_id, status)| peer_id != worker_id && *status == WorkerStatus::Idle))
        {
            return false;
        }
        state.workers[worker_id] = WorkerStatus::Complete;
        self.parked[worker_id].store(false, AtomicOrdering::Release);
        if self.detect_deadlock(&mut state) {
            self.stopped.store(true, AtomicOrdering::Release);
            self.notify_all_workers();
        }
        true
    }

    fn fail_worker(&self, worker_id: usize, error: EngineError) {
        let mut state = self.state.lock().expect("run control mutex poisoned");
        state.workers[worker_id] = WorkerStatus::Failed;
        self.parked[worker_id].store(false, AtomicOrdering::Release);
        // A failure outranks a quiescence observation, which is only its
        // consequence.  Among errors the first still wins.
        if !state.stop.as_ref().is_some_and(RunStop::is_error) {
            state.stop = Some(RunStop::Error(error));
        }
        self.stopped.store(true, AtomicOrdering::Release);
        self.notify_all_workers();
    }

    fn detect_deadlock(&self, state: &mut RunControlState) -> bool {
        if state.stop.is_some() {
            return false;
        }
        let has_pending_worker = state.workers.contains(&WorkerStatus::Idle);
        let no_active_worker = state
            .workers
            .iter()
            .all(|status| matches!(status, WorkerStatus::Idle | WorkerStatus::Complete));
        if has_pending_worker && no_active_worker {
            state.stop = Some(RunStop::Deadlock {
                blocked_operations: blocked_operations(&self.completion_sources),
            });
            return true;
        }
        false
    }

    fn notify_all_workers(&self) {
        for changed in &self.changed {
            changed.notify_all();
        }
    }
}

struct ClusterRuntime {
    cluster_id: usize,
    core: WarpRuntimeCore,
    poll_order: Vec<usize>,
}

impl ClusterRuntime {
    fn new(cluster_id: usize, tasks: Vec<WarpTask>) -> Self {
        Self {
            cluster_id,
            core: WarpRuntimeCore::new(tasks, ReadyOrder::Fifo)
                .expect("executor validated globally unique warp IDs"),
            poll_order: Vec::new(),
        }
    }

    fn poll_next(
        &mut self,
        worker_id: usize,
        cluster_slot: usize,
        runnable_clusters: &Arc<RunnableClusterQueue>,
        control: &Arc<RunControl>,
    ) -> Option<WarpPollOutcome> {
        loop {
            let (warp_id, priority) = self.core.pop_ready_with_priority()?;
            let waker = Waker::from(Arc::new(WarpWaker {
                warp_id,
                worker_id,
                cluster_slot,
                ready: self.core.ready_handle(),
                runnable_clusters: Arc::clone(runnable_clusters),
                control: Arc::clone(control),
            }));
            let Some(poll_result) = self.core.poll(warp_id, &waker) else {
                continue;
            };
            self.poll_order.push(warp_id);
            let result = match poll_result {
                CorePollResult::Completed => WarpPollResult::Ready,
                CorePollResult::Pending { .. } => WarpPollResult::Pending,
                CorePollResult::Failed(error) => WarpPollResult::Failed(error),
                CorePollResult::Panicked(message) => WarpPollResult::Panicked(message),
            };
            return Some(WarpPollOutcome {
                warp_id,
                priority,
                result,
            });
        }
    }

    fn is_complete(&self) -> bool {
        self.core.is_complete()
    }

    fn has_ready(&self) -> bool {
        self.core.has_ready()
    }

    fn highest_ready_priority(&self) -> Option<WakePriority> {
        self.core.highest_ready_priority()
    }

    fn completed_task_count(&self) -> usize {
        self.core.completed_task_count()
    }

    fn pending_warps(&self) -> impl Iterator<Item = usize> + '_ {
        self.core.pending_warps().into_iter()
    }
}

struct PendingClusterQueue {
    clusters: Mutex<VecDeque<ClusterRuntime>>,
}

impl PendingClusterQueue {
    fn new(clusters: impl IntoIterator<Item = ClusterRuntime>) -> Self {
        Self {
            clusters: Mutex::new(clusters.into_iter().collect()),
        }
    }

    fn claim(&self) -> Option<ClusterRuntime> {
        self.clusters
            .lock()
            .expect("pending cluster queue poisoned")
            .pop_front()
    }

    fn pending_warps(&self) -> Vec<usize> {
        let mut pending = self
            .clusters
            .lock()
            .expect("pending cluster queue poisoned")
            .iter()
            .flat_map(ClusterRuntime::pending_warps)
            .collect::<Vec<_>>();
        pending.sort_unstable();
        pending
    }
}

struct WorkerHandle {
    worker_id: usize,
    thread: JoinHandle<WorkerReport>,
}

struct WorkerPool {
    workers: Vec<WorkerHandle>,
    control: Arc<RunControl>,
}

fn configure_worker_float_environment(worker_id: usize) -> Result<(), EngineError> {
    set_round_to_nearest().map_err(|error| {
        EngineError::worker_failed(
            worker_id,
            format!("failed to select round-to-nearest floating-point mode: {error}"),
        )
    })
}

impl WorkerPool {
    fn new(
        initial_clusters: Vec<ClusterRuntime>,
        pending_clusters: Arc<PendingClusterQueue>,
        control: Arc<RunControl>,
        max_clusters_per_worker: usize,
        affinity_release: Option<Arc<WorkerAffinityRelease>>,
    ) -> Result<Self, EngineError> {
        let pending_claim_poll_interval =
            pending_cluster_claim_poll_interval(initial_clusters.len());
        let mut workers: Vec<WorkerHandle> = Vec::with_capacity(initial_clusters.len());
        for (worker_id, cluster) in initial_clusters.into_iter().enumerate() {
            let worker_control = Arc::clone(&control);
            let thread_control = Arc::clone(&worker_control);
            let thread_pending_clusters = Arc::clone(&pending_clusters);
            let thread_affinity_release = affinity_release.clone();
            let thread =
                match native_worker_builder(format!("{WORKER_THREAD_PREFIX}{worker_id}")).spawn(move || {
                    match panic::catch_unwind(AssertUnwindSafe(|| {
                        configure_worker_float_environment(worker_id)?;
                        Ok::<_, EngineError>(worker_main(
                            worker_id,
                            vec![cluster],
                            thread_pending_clusters,
                            Arc::clone(&thread_control),
                            max_clusters_per_worker,
                            pending_claim_poll_interval,
                            thread_affinity_release,
                        ))
                    })) {
                        Ok(Ok(report)) => report,
                        Ok(Err(error)) => {
                            thread_control.fail_worker(worker_id, error);
                            WorkerReport::default()
                        }
                        Err(payload) => {
                            thread_control.fail_worker(
                                worker_id,
                                EngineError::worker_failed(
                                    worker_id,
                                    panic_payload_message(payload.as_ref()),
                                ),
                            );
                            WorkerReport::default()
                        }
                    }
                }) {
                    Ok(thread) => thread,
                    Err(error) => {
                        let error = EngineError::worker_failed(
                            worker_id,
                            format!("failed to spawn worker thread: {error}"),
                        );
                        control.fail_worker(worker_id, error.clone());
                        for worker in workers {
                            let _ = worker.thread.join();
                        }
                        return Err(error);
                    }
                };
            workers.push(WorkerHandle { worker_id, thread });
        }
        Ok(Self { workers, control })
    }

    fn join(self) -> Vec<WorkerReport> {
        let mut reports = Vec::with_capacity(self.workers.len());
        for worker in self.workers {
            match worker.thread.join() {
                Ok(report) => reports.push(report),
                Err(payload) => {
                    self.control.fail_worker(
                        worker.worker_id,
                        EngineError::worker_failed(
                            worker.worker_id,
                            panic_payload_message(payload.as_ref()),
                        ),
                    );
                    reports.push(WorkerReport::default());
                }
            }
        }
        reports
    }
}

#[derive(Default)]
struct WorkerReport {
    completed_task_count: usize,
    poll_count: usize,
    normal_poll_count: usize,
    poll_recheck_poll_count: usize,
    cluster_poll_orders: Vec<(usize, Vec<usize>)>,
    completion_pump_count: usize,
    completion_operation_count: usize,
    pending_warps: Vec<usize>,
}

impl WorkerReport {
    fn from_clusters(
        clusters: &[ClusterRuntime],
        poll_count: usize,
        normal_poll_count: usize,
        poll_recheck_poll_count: usize,
        completion_pump_count: usize,
        completion_operation_count: usize,
    ) -> Self {
        let mut pending_warps = clusters
            .iter()
            .flat_map(ClusterRuntime::pending_warps)
            .collect::<Vec<_>>();
        pending_warps.sort_unstable();
        Self {
            completed_task_count: clusters
                .iter()
                .map(ClusterRuntime::completed_task_count)
                .sum(),
            poll_count,
            normal_poll_count,
            poll_recheck_poll_count,
            cluster_poll_orders: clusters
                .iter()
                .map(|cluster| (cluster.cluster_id, cluster.poll_order.clone()))
                .collect(),
            completion_pump_count,
            completion_operation_count,
            pending_warps,
        }
    }
}

struct WarpPollOutcome {
    warp_id: usize,
    priority: WakePriority,
    result: WarpPollResult,
}

enum WarpPollResult {
    Ready,
    Pending,
    Failed(EngineError),
    Panicked(String),
}

fn worker_main(
    worker_id: usize,
    mut clusters: Vec<ClusterRuntime>,
    pending_clusters: Arc<PendingClusterQueue>,
    control: Arc<RunControl>,
    max_clusters_per_worker: usize,
    pending_claim_poll_interval: usize,
    affinity_release: Option<Arc<WorkerAffinityRelease>>,
) -> WorkerReport {
    let _profile_timer = ProfileTimer::new(ProfileKind::WorkerTotal);
    let mut affinity_widened = false;
    let runnable_clusters = Arc::new(RunnableClusterQueue::new(
        0..clusters.len(),
        ReadyOrder::Fifo,
    ));
    let mut remaining_clusters = clusters.len();
    let mut poll_count = 0_usize;
    let mut normal_poll_count = 0;
    let mut poll_recheck_poll_count = 0;
    let mut completion_pump_count = 0;
    let mut completion_operation_count = 0;
    let mut completion_checked_before_exit = false;

    loop {
        if control.is_stopped() {
            break;
        }
        if let Some(release) = &affinity_release {
            release.widen_current_thread(&mut affinity_widened);
        }
        if remaining_clusters == 0 {
            if claim_pending_cluster(
                &mut clusters,
                &runnable_clusters,
                &pending_clusters,
                &mut remaining_clusters,
                max_clusters_per_worker,
            ) {
                completion_checked_before_exit = false;
                continue;
            }
            if control.try_complete_worker(worker_id, completion_checked_before_exit) {
                break;
            }
            // An idle peer may have lost the pump CAS just before this worker
            // finished. Cover that slow path before terminal deadlock detection.
            match control
                .pump_completions(&mut completion_pump_count, &mut completion_operation_count)
            {
                Ok(CompletionPumpAttempt::Pumped(_)) => {
                    completion_checked_before_exit = true;
                }
                Ok(CompletionPumpAttempt::Contended { generation }) => {
                    if !control.wait_for_completion_pump(worker_id, generation) {
                        break;
                    }
                }
                Err(error) => {
                    control.fail_worker(worker_id, error);
                    break;
                }
            }
            continue;
        }

        if let Some(cluster_slot) = runnable_clusters.pop() {
            if !clusters[cluster_slot].has_ready() {
                continue;
            }
            let Some(poll_sequence) = control.reserve_poll() else {
                control.stop_at_poll_limit(worker_id);
                break;
            };
            let outcome = clusters[cluster_slot]
                .poll_next(worker_id, cluster_slot, &runnable_clusters, &control)
                .expect("runnable cluster must contain a ready warp");
            poll_count += 1;
            match outcome.priority {
                WakePriority::Normal => normal_poll_count += 1,
                WakePriority::PollRecheck => poll_recheck_poll_count += 1,
            }
            let error = match outcome.result {
                WarpPollResult::Ready | WarpPollResult::Pending => None,
                WarpPollResult::Failed(source) => {
                    Some(EngineError::warp_failed(outcome.warp_id, source))
                }
                WarpPollResult::Panicked(message) => {
                    Some(EngineError::warp_panicked(outcome.warp_id, message))
                }
            };
            if let Some(error) = error {
                control.fail_worker(worker_id, error);
                break;
            }

            if clusters[cluster_slot].is_complete() {
                runnable_clusters.retire(cluster_slot);
                remaining_clusters -= 1;
            } else if let Some(priority) = clusters[cluster_slot].highest_ready_priority() {
                runnable_clusters.schedule(cluster_slot, priority);
            }

            if control.should_pump_after(poll_sequence) {
                if let Err(error) = control
                    .pump_completions(&mut completion_pump_count, &mut completion_operation_count)
                {
                    control.fail_worker(worker_id, error);
                    break;
                }
            }
            if outcome.priority == WakePriority::PollRecheck
                || poll_count.is_multiple_of(pending_claim_poll_interval)
            {
                claim_pending_cluster(
                    &mut clusters,
                    &runnable_clusters,
                    &pending_clusters,
                    &mut remaining_clusters,
                    max_clusters_per_worker,
                );
            }
            continue;
        }

        if claim_pending_cluster(
            &mut clusters,
            &runnable_clusters,
            &pending_clusters,
            &mut remaining_clusters,
            max_clusters_per_worker,
        ) {
            continue;
        }

        match control.pump_completions(&mut completion_pump_count, &mut completion_operation_count)
        {
            Ok(CompletionPumpAttempt::Pumped(true)) => continue,
            Ok(CompletionPumpAttempt::Pumped(false)) => {}
            Ok(CompletionPumpAttempt::Contended { generation }) => {
                if !control.wait_for_completion_pump(worker_id, generation) {
                    break;
                }
                continue;
            }
            Err(error) => {
                control.fail_worker(worker_id, error);
                break;
            }
        }

        if !control.wait_for_work(worker_id, || runnable_clusters.has_ready()) {
            break;
        }
    }

    WorkerReport::from_clusters(
        &clusters,
        poll_count,
        normal_poll_count,
        poll_recheck_poll_count,
        completion_pump_count,
        completion_operation_count,
    )
}

fn claim_pending_cluster(
    clusters: &mut Vec<ClusterRuntime>,
    runnable_clusters: &Arc<RunnableClusterQueue>,
    pending_clusters: &PendingClusterQueue,
    remaining_clusters: &mut usize,
    max_clusters_per_worker: usize,
) -> bool {
    if clusters.len() >= max_clusters_per_worker {
        return false;
    }
    let Some(cluster) = pending_clusters.claim() else {
        return false;
    };
    let cluster_slot = clusters.len();
    clusters.push(cluster);
    runnable_clusters.activate_normal_front(cluster_slot);
    *remaining_clusters += 1;
    true
}

pub(crate) fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

struct CompletionPumpLease<'a>(&'a RunControl);

impl Drop for CompletionPumpLease<'_> {
    fn drop(&mut self) {
        self.0.finish_completion_pump();
    }
}

fn pump_completion_round(
    completions: &[Arc<dyn CompletionSource>],
    completion_pump_count: &mut usize,
    completion_operation_count: &mut usize,
) -> Result<bool, EngineError> {
    let _profile_timer = ProfileTimer::new(ProfileKind::CompletionPump);
    let mut made_progress = false;
    for source in completions {
        *completion_pump_count += 1;
        let progress = source
            .pump()
            .map_err(|error| EngineError::completion_failed(source.source_name(), error))?;
        *completion_operation_count += progress.completed_operations;
        made_progress |= progress.made_progress();
    }
    Ok(made_progress)
}

fn blocked_operations(completions: &[Arc<dyn CompletionSource>]) -> Vec<BlockedOperation> {
    let mut blocked = completions
        .iter()
        .flat_map(|source| source.blocked_operations())
        .collect::<Vec<_>>();
    blocked.sort_by(BlockedOperation::diagnostic_cmp);
    blocked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::{poll_recheck_waker, reschedule};
    use crate::{CompletionProgress, GlobalMemory, LaunchTopology, WarpMask};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::{Duration, Instant};

    #[inline(never)]
    fn exercise_large_native_frame() {
        let mut frame = [0_u8; 3 * 1024 * 1024];
        frame[frame.len() - 1] = 1;
        std::hint::black_box(&mut frame);
    }

    #[test]
    fn native_worker_stack_has_a_floor_and_preserves_larger_configuration() {
        let larger = MIN_NATIVE_WORKER_STACK_BYTES * 2;
        let larger_text = larger.to_string();

        assert_eq!(
            native_worker_stack_bytes(None),
            MIN_NATIVE_WORKER_STACK_BYTES
        );
        assert_eq!(
            native_worker_stack_bytes(Some("invalid")),
            MIN_NATIVE_WORKER_STACK_BYTES
        );
        assert_eq!(
            native_worker_stack_bytes(Some("1048576")),
            MIN_NATIVE_WORKER_STACK_BYTES
        );
        assert_eq!(native_worker_stack_bytes(Some(&larger_text)), larger);
    }

    #[test]
    fn pending_cluster_claim_cadence_prioritizes_locality_but_bounds_starvation() {
        assert_eq!(pending_cluster_claim_poll_interval(0), 1);
        assert_eq!(pending_cluster_claim_poll_interval(1), 1);
        assert_eq!(
            pending_cluster_claim_poll_interval(2),
            MAX_LOCAL_POLLS_BEFORE_PENDING_CLUSTER_CLAIM
        );
        assert_eq!(
            pending_cluster_claim_poll_interval(8),
            MAX_LOCAL_POLLS_BEFORE_PENDING_CLUSTER_CLAIM
        );
        assert_eq!(
            pending_cluster_claim_poll_interval(64),
            MAX_LOCAL_POLLS_BEFORE_PENDING_CLUSTER_CLAIM
        );
    }

    #[test]
    fn native_worker_threads_have_stack_margin_for_unoptimized_generated_code() {
        let task = WarpTask::new(0, async {
            exercise_large_native_frame();
            Ok(())
        });
        let stats = Executor::default().run([task]).unwrap();
        assert_eq!(stats.completed_task_count, 1);
    }

    #[test]
    fn out_of_bounds_kind_survives_warp_context_and_rendering() {
        let error = EngineError::warp_failed(
            3,
            EngineError::out_of_bounds("negative buffer index -1 on lane 7"),
        );

        assert!(error.is_out_of_bounds());
        let message = error.to_string();
        assert!(message.contains("warp 3 failed"));
        assert!(message.contains("negative buffer index -1 on lane 7"));
        assert!(message.contains(OUT_OF_BOUNDS_ERROR_PREFIX));

        let memory = GlobalMemory::new();
        let allocation = memory.allocate_zeroed(4).unwrap();
        let view = memory.full_view(allocation).unwrap();
        let memory_error = memory
            .subview(&view, 2, 4)
            .err()
            .expect("view overrun must fail");
        let memory_error = EngineError::from(memory_error);
        assert!(memory_error.is_out_of_bounds());
        assert!(memory_error.to_string().starts_with(OUT_OF_BOUNDS_ERROR_PREFIX));
        assert_eq!(engine_error_kind(&error), "oob");
        assert_eq!(engine_error_kind(&memory_error), "oob");
    }

    #[test]
    fn typed_error_kinds_preserve_nested_synchronization_causes() {
        let key = crate::OccurrenceKey::new(
            17,
            "mbarrier.wait",
            [3],
            crate::ScopeInstance::Cta { global_cta_id: 0 },
        );
        let uninitialized = SynchronizationError::BarrierUninitialized { key: key.clone() };
        assert_eq!(
            synchronization_error_kind(&uninitialized),
            "mbarrier_use_before_init"
        );
        assert_eq!(
            engine_error_kind(&EngineError::warp_failed(
                2,
                EngineError::synchronization(uninitialized.clone()),
            )),
            "mbarrier_use_before_init"
        );
        assert_eq!(
            engine_error_kind(&EngineError::completion_failed(
                "physical-mbarrier",
                uninitialized,
            )),
            "mbarrier_use_before_init"
        );
        let contextual = EngineError::synchronization(SynchronizationError::BarrierUninitialized {
            key: key.clone(),
        })
        .with_context("mbarrier.wait: ");
        assert_eq!(engine_error_kind(&contextual), "mbarrier_use_before_init");
        assert!(contextual.to_string().starts_with("mbarrier.wait: "));

        let cluster_key = crate::OccurrenceKey::new(
            18,
            "barrier.cluster.arrive",
            [4],
            crate::ScopeInstance::Cluster { cluster_id: 1 },
        );
        assert_eq!(
            synchronization_error_kind(&SynchronizationError::DuplicateArrival {
                key: cluster_key,
                phase: 0,
                warp_id: 4,
            }),
            "cluster_barrier_early_arrival"
        );
        assert_eq!(
            engine_error_kind(&EngineError::analysis_incomplete(
                crate::AnalysisGapKind::ClusterBarrierUnaligned.name(),
            )),
            "analysis_incomplete"
        );
        assert_eq!(
            engine_error_kind(&EngineError::message("opaque runtime failure")),
            "engine_error"
        );
    }

    struct YieldOnce {
        yielded: bool,
    }

    impl Future for YieldOnce {
        type Output = Result<(), EngineError>;

        fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
            if self.yielded {
                Poll::Ready(Ok(()))
            } else {
                self.yielded = true;
                context.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    struct NeverReady;

    impl Future for NeverReady {
        type Output = Result<(), EngineError>;

        fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }

    #[derive(Default)]
    struct BlockingCompletionState {
        pump_count: usize,
        first_pump_started: bool,
        release_first_pump: bool,
    }

    #[derive(Default)]
    struct BlockingCompletionSource {
        state: Mutex<BlockingCompletionState>,
        changed: Condvar,
    }

    impl BlockingCompletionSource {
        fn wait_for_first_pump(&self) {
            let state = self.state.lock().unwrap();
            let (state, timeout) = self
                .changed
                .wait_timeout_while(state, Duration::from_secs(2), |state| {
                    !state.first_pump_started
                })
                .unwrap();
            assert!(!timeout.timed_out() || state.first_pump_started);
        }

        fn release_first_pump(&self) {
            let mut state = self.state.lock().unwrap();
            state.release_first_pump = true;
            self.changed.notify_all();
        }

        fn pump_count(&self) -> usize {
            self.state.lock().unwrap().pump_count
        }
    }

    impl CompletionSource for BlockingCompletionSource {
        fn source_name(&self) -> &'static str {
            "blocking-test-completion"
        }

        fn pump(&self) -> Result<CompletionProgress, SynchronizationError> {
            let mut state = self.state.lock().unwrap();
            state.pump_count += 1;
            if state.pump_count == 1 {
                state.first_pump_started = true;
                self.changed.notify_all();
                state = self
                    .changed
                    .wait_while(state, |state| !state.release_first_pump)
                    .unwrap();
            }
            drop(state);
            Ok(CompletionProgress::default())
        }

        fn blocked_operations(&self) -> Vec<BlockedOperation> {
            Vec::new()
        }

        fn validate_quiescent(&self) -> Result<(), SynchronizationError> {
            Ok(())
        }
    }

    struct SpinUntilPublished {
        polls: usize,
        published: Arc<AtomicBool>,
    }

    impl Future for SpinUntilPublished {
        type Output = Result<(), EngineError>;

        fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
            if self.published.load(Ordering::SeqCst) {
                return Poll::Ready(Ok(()));
            }
            if self.polls >= 4 {
                return Poll::Ready(Err(EngineError::message(
                    "runnable cluster starved an unstarted peer",
                )));
            }
            self.polls += 1;
            context.waker().wake_by_ref();
            Poll::Pending
        }
    }

    struct AdvanceOnSecondPoll {
        first_poll: bool,
        state: Arc<(Mutex<bool>, Condvar)>,
    }

    impl Future for AdvanceOnSecondPoll {
        type Output = Result<(), EngineError>;

        fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
            if self.first_poll {
                self.first_poll = false;
                context.waker().wake_by_ref();
                return Poll::Pending;
            }
            let (advanced, changed) = &*self.state;
            *advanced.lock().unwrap() = true;
            changed.notify_all();
            Poll::Ready(Ok(()))
        }
    }

    struct WaitForOtherCluster {
        state: Arc<(Mutex<bool>, Condvar)>,
    }

    impl Future for WaitForOtherCluster {
        type Output = Result<(), EngineError>;

        fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Self::Output> {
            let (advanced, changed) = &*self.state;
            let (advanced, timeout) = changed
                .wait_timeout_while(
                    advanced.lock().unwrap(),
                    Duration::from_secs(2),
                    |advanced| !*advanced,
                )
                .unwrap();
            if timeout.timed_out() && !*advanced {
                return Poll::Ready(Err(EngineError::message(
                    "another cluster was stalled behind a global polling wave",
                )));
            }
            Poll::Ready(Ok(()))
        }
    }

    #[derive(Default)]
    struct WakeDuringPollState {
        target_waker: Option<Waker>,
        wake_sent: bool,
    }

    struct WakeDuringPollTarget {
        first_poll: bool,
        state: Arc<(Mutex<WakeDuringPollState>, Condvar)>,
    }

    impl Future for WakeDuringPollTarget {
        type Output = Result<(), EngineError>;

        fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
            if !self.first_poll {
                return Poll::Ready(Ok(()));
            }
            self.first_poll = false;
            let (state, changed) = &*self.state;
            let mut state = state.lock().unwrap();
            state.target_waker = Some(context.waker().clone());
            changed.notify_all();
            let (state, timeout) = changed
                .wait_timeout_while(state, Duration::from_secs(2), |state| !state.wake_sent)
                .unwrap();
            if timeout.timed_out() && !state.wake_sent {
                return Poll::Ready(Err(EngineError::message(
                    "cross-cluster wake was not delivered during poll",
                )));
            }
            Poll::Pending
        }
    }

    struct WakeDuringPollPeer {
        state: Arc<(Mutex<WakeDuringPollState>, Condvar)>,
    }

    impl Future for WakeDuringPollPeer {
        type Output = Result<(), EngineError>;

        fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Self::Output> {
            let (state_mutex, changed) = &*self.state;
            let state = state_mutex.lock().unwrap();
            let (mut state, timeout) = changed
                .wait_timeout_while(state, Duration::from_secs(2), |state| {
                    state.target_waker.is_none()
                })
                .unwrap();
            if timeout.timed_out() && state.target_waker.is_none() {
                return Poll::Ready(Err(EngineError::message(
                    "target cluster never registered its waker",
                )));
            }
            let target_waker = state
                .target_waker
                .take()
                .expect("target waker must be registered");
            drop(state);
            target_waker.wake();
            let mut state = state_mutex.lock().unwrap();
            state.wake_sent = true;
            changed.notify_all();
            Poll::Ready(Ok(()))
        }
    }

    #[derive(Default)]
    struct BarrierState {
        arrivals: usize,
        target: usize,
        waiters: BTreeMap<usize, Waker>,
    }

    struct BarrierFuture {
        warp_id: usize,
        arrived: bool,
        state: Arc<Mutex<BarrierState>>,
    }

    impl Future for BarrierFuture {
        type Output = Result<(), EngineError>;

        fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
            let warp_id = self.warp_id;
            let first_arrival = !self.arrived;
            self.arrived = true;
            let mut state = self.state.lock().unwrap();
            if state.arrivals == state.target {
                return Poll::Ready(Ok(()));
            }
            if first_arrival {
                state.arrivals += 1;
            }
            if state.arrivals == state.target {
                let waiters = std::mem::take(&mut state.waiters);
                drop(state);
                for (_, waker) in waiters {
                    waker.wake();
                }
                Poll::Ready(Ok(()))
            } else {
                state.waiters.insert(warp_id, context.waker().clone());
                Poll::Pending
            }
        }
    }

    #[test]
    fn topology_creates_exactly_one_task_per_warp() {
        let topology = LaunchTopology::new(2, 3, 4).unwrap();
        let ran = Arc::new(AtomicUsize::new(0));
        let tasks = topology
            .warp_contexts()
            .map(|context| {
                let ran = Arc::clone(&ran);
                WarpTask::new(context.global_warp_id(), async move {
                    ran.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
            })
            .collect::<Vec<_>>();

        let stats = Executor::default().run(tasks).unwrap();
        assert_eq!(stats.task_count, topology.warp_count());
        assert_eq!(stats.completed_task_count, topology.warp_count());
        assert_eq!(stats.poll_count, topology.warp_count());
        assert_eq!(ran.load(Ordering::SeqCst), topology.warp_count());
        assert_eq!(stats.worker_count, 1);
        assert_eq!(stats.scheduling_domain_count, 1);
    }

    #[test]
    fn polling_and_self_wakeup_are_deterministic() {
        let tasks = [2, 0, 1]
            .into_iter()
            .map(|warp_id| WarpTask::new(warp_id, YieldOnce { yielded: false }));
        let stats = Executor::default().run(tasks).unwrap();
        assert_eq!(stats.task_count, 3);
        assert_eq!(stats.poll_count, 6);
        assert_eq!(stats.poll_order, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn contended_completion_pump_waits_for_generation_and_retries() {
        let source = Arc::new(BlockingCompletionSource::default());
        let completion_sources: Vec<Arc<dyn CompletionSource>> = vec![source.clone()];
        let control = Arc::new(RunControl::new(1, completion_sources, None, 2));
        let first_control = Arc::clone(&control);
        let first = thread::spawn(move || {
            let mut pump_count = 0;
            let mut operation_count = 0;
            first_control.pump_completions(&mut pump_count, &mut operation_count)
        });

        source.wait_for_first_pump();
        let mut pump_count = 0;
        let mut operation_count = 0;
        let generation = match control
            .pump_completions(&mut pump_count, &mut operation_count)
            .unwrap()
        {
            CompletionPumpAttempt::Contended { generation } => generation,
            attempt => panic!("expected completion-pump contention, got {attempt:?}"),
        };
        let waiter_control = Arc::clone(&control);
        let waiter = thread::spawn(move || waiter_control.wait_for_completion_pump(0, generation));
        let waiter_deadline = Instant::now() + Duration::from_secs(2);
        while control
            .completion_pump_waiters
            .load(AtomicOrdering::Acquire)
            == 0
        {
            assert!(Instant::now() < waiter_deadline);
            thread::yield_now();
        }
        source.release_first_pump();

        assert!(waiter.join().unwrap());
        assert_eq!(
            first.join().unwrap().unwrap(),
            CompletionPumpAttempt::Pumped(false)
        );
        assert_eq!(
            control
                .pump_completions(&mut pump_count, &mut operation_count)
                .unwrap(),
            CompletionPumpAttempt::Pumped(false)
        );
        assert_eq!(source.pump_count(), 2);
    }

    #[test]
    fn finishing_worker_checks_completions_before_deadlocking_an_idle_peer() {
        let control = RunControl::new(2, Vec::new(), None, 2);
        control.state.lock().unwrap().workers[1] = WorkerStatus::Idle;

        assert!(!control.try_complete_worker(0, false));
        assert!(matches!(
            control.state.lock().unwrap().workers[0],
            WorkerStatus::Active
        ));
        assert!(control.try_complete_worker(0, true));
        assert!(matches!(
            control.stop_reason(),
            Some(RunStop::Deadlock { .. })
        ));
    }

    fn stop_label(stop: Option<&RunStop>) -> &'static str {
        match stop {
            None => "no stop",
            Some(RunStop::Deadlock { .. }) => "a deadlock",
            Some(RunStop::PollLimit { .. }) => "a poll limit",
            Some(RunStop::Error(_)) => "an error",
        }
    }

    #[test]
    fn worker_failure_replaces_an_already_recorded_deadlock() {
        let control = RunControl::new(2, Vec::new(), None, 2);
        control.state.lock().unwrap().workers[1] = WorkerStatus::Idle;
        assert!(control.try_complete_worker(0, true));
        assert!(matches!(
            control.stop_reason(),
            Some(RunStop::Deadlock { .. })
        ));

        control.fail_worker(1, EngineError::message("genuine worker failure"));

        match control.stop_reason() {
            Some(RunStop::Error(error)) => {
                assert_eq!(error.to_string(), "genuine worker failure");
            }
            stop => panic!(
                "a worker failure must outrank a recorded deadlock, got {}",
                stop_label(stop.as_ref())
            ),
        }
    }

    #[test]
    fn a_second_worker_failure_does_not_replace_the_first() {
        let control = RunControl::new(2, Vec::new(), None, 2);
        control.fail_worker(0, EngineError::message("first failure"));
        control.fail_worker(1, EngineError::message("second failure"));

        match control.stop_reason() {
            Some(RunStop::Error(error)) => assert_eq!(error.to_string(), "first failure"),
            stop => panic!(
                "the first recorded error must survive, got {}",
                stop_label(stop.as_ref())
            ),
        }
    }

    #[test]
    fn a_deadlock_does_not_replace_a_recorded_worker_failure() {
        let control = RunControl::new(2, Vec::new(), None, 2);
        control.fail_worker(0, EngineError::message("first failure"));
        control.state.lock().unwrap().workers[1] = WorkerStatus::Idle;
        // A quiescence observation reached after a failure must not downgrade it.
        assert!(control.try_complete_worker(0, true));

        match control.stop_reason() {
            Some(RunStop::Error(error)) => assert_eq!(error.to_string(), "first failure"),
            stop => panic!(
                "a deadlock must not outrank a recorded error, got {}",
                stop_label(stop.as_ref())
            ),
        }
    }

    #[test]
    fn cooperative_reschedule_rotates_same_cluster_fifo_deterministically() {
        let topology = LaunchTopology::new(1, 1, 3).unwrap();
        let tasks = [2, 0, 1].into_iter().map(|warp_id| {
            WarpTask::new(warp_id, async move {
                reschedule().await;
                reschedule().await;
                Ok(())
            })
        });

        let stats = Executor::default()
            .run_with_topology(tasks, topology)
            .unwrap();

        assert_eq!(stats.poll_count, 9);
        assert_eq!(stats.poll_order, vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn cooperative_reschedule_preserves_locals_across_awaits() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let first_observed = Arc::clone(&observed);
        let tasks = [
            WarpTask::new(0, async move {
                let mut local = String::from("native");
                reschedule().await;
                local.push_str("-rust");
                reschedule().await;
                first_observed.lock().unwrap().push(local);
                Ok(())
            }),
            WarpTask::new(1, async move {
                reschedule().await;
                Ok(())
            }),
        ];

        let stats = Executor::default().run(tasks).unwrap();

        assert_eq!(stats.poll_order, vec![0, 1, 0, 1, 0]);
        assert_eq!(*observed.lock().unwrap(), [String::from("native-rust")]);
    }

    /// A warp's structured-control active mask lives in its `WarpContext`,
    /// which the generated kernel body carries across cooperative awaits.
    /// Suspending must restore it verbatim, exactly like the native locals
    /// covered by `cooperative_reschedule_preserves_locals_across_awaits`.
    #[test]
    fn cooperative_reschedule_preserves_active_masks_across_awaits() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let tasks = topology.warp_contexts().map(|mut context| {
            let observed = Arc::clone(&observed);
            WarpTask::new(context.global_warp_id(), async move {
                let mask = WarpMask::from_bits(if context.global_warp_id() == 0 {
                    0x5555_5555
                } else {
                    0xaaaa_aaaa
                });
                context.set_active_mask(mask);
                reschedule().await;
                observed
                    .lock()
                    .unwrap()
                    .push((context.global_warp_id(), context.active_mask()));
                Ok(())
            })
        });

        let stats = Executor::default()
            .run_with_topology(tasks, topology)
            .unwrap();

        assert_eq!(stats.completed_task_count, 2);
        // Both warps really suspended between the write and the read back.
        assert_eq!(stats.poll_order, vec![0, 1, 0, 1]);
        assert_eq!(
            *observed.lock().unwrap(),
            [
                (0, WarpMask::from_bits(0x5555_5555)),
                (1, WarpMask::from_bits(0xaaaa_aaaa)),
            ]
        );
    }

    #[test]
    fn cluster_ready_queue_is_fifo_deduplicated_and_isolated() {
        let first = ClusterReadyQueue::new([0, 1], ReadyOrder::Fifo);
        let second = ClusterReadyQueue::new([2], ReadyOrder::Fifo);

        assert_eq!(first.pop(), Some(0));
        assert_eq!(
            first.schedule(0, WakePriority::Normal),
            Some(WakePriority::Normal)
        );
        assert_eq!(first.schedule(0, WakePriority::Normal), None);
        assert_eq!(first.pop(), Some(1));
        assert_eq!(first.pop(), Some(0));
        assert_eq!(second.pop(), Some(2));
        assert_eq!(second.pop(), None);

        first.retire(0);
        assert_eq!(first.schedule(0, WakePriority::Normal), None);
    }

    #[test]
    fn cluster_ready_queue_defers_poll_rechecks_and_promotes_normal_wakes() {
        let ready = ClusterReadyQueue::new([0, 1, 2], ReadyOrder::Fifo);
        assert_eq!(ready.pop(), Some(0));
        assert_eq!(ready.pop(), Some(1));
        assert_eq!(ready.pop(), Some(2));

        assert_eq!(
            ready.schedule(0, WakePriority::PollRecheck),
            Some(WakePriority::PollRecheck)
        );
        assert_eq!(
            ready.schedule(1, WakePriority::Normal),
            Some(WakePriority::Normal)
        );
        assert_eq!(ready.pop(), Some(1));

        assert_eq!(
            ready.schedule(0, WakePriority::Normal),
            Some(WakePriority::Normal)
        );
        assert_eq!(ready.pop(), Some(0));
        assert_eq!(ready.pop(), None);
    }

    #[test]
    fn runnable_cluster_queue_is_fifo_and_deduplicated() {
        let ready = RunnableClusterQueue::new(0..3, ReadyOrder::Fifo);

        assert_eq!(ready.pop(), Some(0));
        assert!(ready.schedule(0, WakePriority::Normal).is_some());
        assert!(ready.schedule(0, WakePriority::Normal).is_none());
        assert_eq!(ready.pop(), Some(1));
        assert_eq!(ready.pop(), Some(2));
        assert_eq!(ready.pop(), Some(0));

        ready.retire(0);
        assert!(ready.schedule(0, WakePriority::Normal).is_none());
    }

    #[test]
    fn runnable_cluster_queue_defers_poll_rechecks_and_promotes_normal_wakes() {
        let ready = RunnableClusterQueue::new(0..2, ReadyOrder::Fifo);
        assert_eq!(ready.pop(), Some(0));
        assert_eq!(ready.pop(), Some(1));

        assert!(ready.schedule(0, WakePriority::PollRecheck).is_some());
        assert!(ready.schedule(1, WakePriority::Normal).is_some());
        assert_eq!(ready.pop(), Some(1));

        assert!(ready.schedule(0, WakePriority::Normal).is_some());
        assert_eq!(ready.pop(), Some(0));
        assert_eq!(ready.pop(), None);
    }

    #[test]
    fn poll_recheck_wake_runs_after_an_ordinary_ready_warp() {
        let parked_waker = Arc::new(Mutex::new(None::<Waker>));
        let poller_waker = Arc::clone(&parked_waker);
        let producer_waker = Arc::clone(&parked_waker);
        let mut first_poll = true;
        let tasks = [
            WarpTask::new(
                0,
                std::future::poll_fn(move |context| {
                    if first_poll {
                        first_poll = false;
                        *poller_waker.lock().unwrap() = Some(poll_recheck_waker(context.waker()));
                        Poll::Pending
                    } else {
                        Poll::Ready(Ok(()))
                    }
                }),
            ),
            WarpTask::new(1, async move {
                producer_waker
                    .lock()
                    .unwrap()
                    .take()
                    .expect("poller must park before the producer runs")
                    .wake();
                reschedule().await;
                Ok(())
            }),
        ];

        let stats = Executor::default().run(tasks).unwrap();

        assert_eq!(stats.poll_order, vec![0, 1, 1, 0]);
    }

    #[test]
    fn blocked_cluster_does_not_stall_another_clusters_ready_queue() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let state = Arc::new((Mutex::new(false), Condvar::new()));
        let tasks = [
            WarpTask::new(
                0,
                AdvanceOnSecondPoll {
                    first_poll: true,
                    state: Arc::clone(&state),
                },
            ),
            WarpTask::new(1, WaitForOtherCluster { state }),
        ];

        let stats = Executor::with_max_workers(2)
            .run_with_topology(tasks, topology)
            .unwrap();

        assert_eq!(stats.completed_task_count, 2);
        assert_eq!(stats.poll_count, 3);
        assert_eq!(stats.worker_count, 2);
        assert_eq!(stats.scheduling_domain_count, 2);
        assert_eq!(stats.poll_order, vec![0, 0, 1]);
    }

    #[test]
    fn cross_cluster_wake_during_poll_is_not_lost() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let state = Arc::new((Mutex::new(WakeDuringPollState::default()), Condvar::new()));
        let tasks = [
            WarpTask::new(
                0,
                WakeDuringPollTarget {
                    first_poll: true,
                    state: Arc::clone(&state),
                },
            ),
            WarpTask::new(1, WakeDuringPollPeer { state }),
        ];

        let stats = Executor::with_max_workers(2)
            .run_with_topology(tasks, topology)
            .unwrap();

        assert_eq!(stats.completed_task_count, 2);
        assert_eq!(stats.poll_count, 3);
        assert_eq!(stats.poll_order, vec![0, 0, 1]);
    }

    #[test]
    fn different_clusters_execute_concurrently() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let tasks = topology
            .warp_contexts()
            .map(|context| {
                let active = Arc::clone(&active);
                let max_active = Arc::clone(&max_active);
                WarpTask::new(context.global_warp_id(), async move {
                    let concurrent = active.fetch_add(1, Ordering::SeqCst) + 1;
                    max_active.fetch_max(concurrent, Ordering::SeqCst);
                    thread::sleep(Duration::from_millis(25));
                    active.fetch_sub(1, Ordering::SeqCst);
                    Ok(())
                })
            })
            .collect::<Vec<_>>();

        let stats = Executor::with_max_workers(2)
            .run_with_topology(tasks, topology)
            .unwrap();
        assert_eq!(max_active.load(Ordering::SeqCst), 2);
        assert_eq!(stats.worker_count, 2);
        assert_eq!(stats.scheduling_domain_count, 2);
        assert_eq!(stats.poll_order, vec![0, 1]);
    }

    #[test]
    fn idle_worker_claims_an_unstarted_cluster() {
        let topology = LaunchTopology::new(3, 1, 1).unwrap();
        let state = Arc::new((Mutex::new(false), Condvar::new()));
        let waiting_state = Arc::clone(&state);
        let publishing_state = Arc::clone(&state);
        let tasks = [
            WarpTask::new(0, async move {
                let (published, changed) = &*waiting_state;
                let (published, timeout) = changed
                    .wait_timeout_while(
                        published.lock().unwrap(),
                        Duration::from_secs(2),
                        |published| !*published,
                    )
                    .unwrap();
                if timeout.timed_out() && !*published {
                    return Err(EngineError::message(
                        "unstarted cluster stayed pinned behind a blocked cluster",
                    ));
                }
                Ok(())
            }),
            WarpTask::new(1, async { Ok(()) }),
            WarpTask::new(2, async move {
                let (published, changed) = &*publishing_state;
                *published.lock().unwrap() = true;
                changed.notify_all();
                Ok(())
            }),
        ];

        let stats = Executor::with_max_workers(2)
            .run_with_topology(tasks, topology)
            .unwrap();

        assert_eq!(stats.completed_task_count, 3);
        assert_eq!(stats.worker_count, 2);
        assert_eq!(stats.scheduling_domain_count, 3);
    }

    #[test]
    fn first_worker_cannot_pin_every_unstarted_cluster() {
        let topology = LaunchTopology::new(8, 1, 1).unwrap();
        let second_worker_started = Arc::new((Mutex::new(false), Condvar::new()));
        let worker_threads = Arc::new(Mutex::new(BTreeMap::new()));
        let tasks = topology
            .warp_contexts()
            .map(|context| {
                let second_worker_started = Arc::clone(&second_worker_started);
                let worker_threads = Arc::clone(&worker_threads);
                WarpTask::new(context.global_warp_id(), async move {
                    if context.cluster_id() == 0 {
                        let (started, changed) = &*second_worker_started;
                        let _guard = changed
                            .wait_while(started.lock().unwrap(), |started| !*started)
                            .unwrap();
                    } else if context.cluster_id() == 1 {
                        let (started, changed) = &*second_worker_started;
                        *started.lock().unwrap() = true;
                        changed.notify_all();
                        thread::sleep(Duration::from_millis(100));
                    }
                    worker_threads
                        .lock()
                        .unwrap()
                        .insert(context.global_warp_id(), thread::current().id());
                    Ok(())
                })
            })
            .collect::<Vec<_>>();

        let stats = Executor::with_max_workers(2)
            .run_with_topology(tasks, topology)
            .unwrap();
        let worker_threads = worker_threads.lock().unwrap();
        let first = worker_threads[&0];
        let second = worker_threads[&1];

        assert_ne!(first, second);
        assert_eq!(
            worker_threads
                .values()
                .filter(|thread| **thread == first)
                .count(),
            4
        );
        assert_eq!(
            worker_threads
                .values()
                .filter(|thread| **thread == second)
                .count(),
            4
        );
        assert_eq!(stats.completed_task_count, 8);
    }

    #[test]
    fn runnable_cluster_does_not_starve_an_unstarted_peer_on_one_worker() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let published = Arc::new(AtomicBool::new(false));
        let publisher_state = Arc::clone(&published);
        let tasks = [
            WarpTask::new(
                0,
                SpinUntilPublished {
                    polls: 0,
                    published,
                },
            ),
            WarpTask::new(1, async move {
                publisher_state.store(true, Ordering::SeqCst);
                Ok(())
            }),
        ];

        let stats = Executor::with_max_workers(1)
            .run_with_topology(tasks, topology)
            .unwrap();

        assert_eq!(stats.completed_task_count, 2);
        assert_eq!(stats.poll_count, 3);
    }

    #[test]
    fn every_warp_in_a_cluster_stays_on_one_worker() {
        let topology = LaunchTopology::new(2, 2, 2).unwrap();
        let worker_threads = Arc::new(Mutex::new(BTreeMap::new()));
        let tasks = topology
            .warp_contexts()
            .map(|context| {
                let worker_threads = Arc::clone(&worker_threads);
                WarpTask::new(context.global_warp_id(), async move {
                    worker_threads
                        .lock()
                        .unwrap()
                        .insert(context.global_warp_id(), thread::current().id());
                    Ok(())
                })
            })
            .collect::<Vec<_>>();

        Executor::with_max_workers(2)
            .run_with_topology(tasks, topology)
            .unwrap();
        let worker_threads = worker_threads.lock().unwrap();
        let first_cluster = topology.cluster_warp_range(0).unwrap();
        let second_cluster = topology.cluster_warp_range(1).unwrap();
        let first_thread = worker_threads[&first_cluster.start];
        let second_thread = worker_threads[&second_cluster.start];
        assert!(first_cluster
            .map(|warp_id| worker_threads[&warp_id])
            .all(|thread_id| thread_id == first_thread));
        assert!(second_cluster
            .map(|warp_id| worker_threads[&warp_id])
            .all(|thread_id| thread_id == second_thread));
        assert_ne!(first_thread, second_thread);
    }

    #[test]
    fn blocked_tasks_are_woken_by_an_engine_style_rendezvous() {
        let state = Arc::new(Mutex::new(BarrierState {
            target: 3,
            ..BarrierState::default()
        }));
        let tasks = (0..3).map(|warp_id| {
            WarpTask::new(
                warp_id,
                BarrierFuture {
                    warp_id,
                    arrived: false,
                    state: Arc::clone(&state),
                },
            )
        });

        let stats = Executor::default().run(tasks).unwrap();
        assert_eq!(stats.task_count, 3);
        assert_eq!(stats.completed_task_count, 3);
        assert_eq!(stats.poll_order, vec![0, 1, 2, 0, 1]);
    }

    #[test]
    fn pending_without_wakeup_reports_deadlock() {
        let error = Executor::default()
            .run([WarpTask::new(4, NeverReady), WarpTask::new(1, NeverReady)])
            .unwrap_err();
        assert_eq!(error, EngineError::deadlock(vec![1, 4], vec![], 2));
        assert!(error.to_string().contains("blocked warps: [1, 4]"));
    }

    #[test]
    fn duplicate_ids_and_poll_limit_fail_closed() {
        let duplicate = Executor::default()
            .run([
                WarpTask::new(0, async { Ok(()) }),
                WarpTask::new(0, async { Ok(()) }),
            ])
            .unwrap_err();
        assert_eq!(duplicate, EngineError::duplicate_warp_id(0));

        let limited = Executor::with_poll_limit(1)
            .run([WarpTask::new(3, YieldOnce { yielded: false })])
            .unwrap_err();
        assert_eq!(limited, EngineError::poll_limit_exceeded(1, vec![3]));
    }

    #[test]
    fn poll_limit_is_launch_wide_across_workers() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let mut executor = Executor {
            poll_limit: Some(1),
            max_workers: 2,
        };
        let error = executor
            .run_with_topology(
                [
                    WarpTask::new(0, YieldOnce { yielded: false }),
                    WarpTask::new(1, YieldOnce { yielded: false }),
                ],
                topology,
            )
            .unwrap_err();

        assert_eq!(error, EngineError::poll_limit_exceeded(1, vec![0, 1]));
    }

    #[test]
    fn parallel_errors_panics_and_deadlocks_terminate() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();

        let failed = Executor::with_max_workers(2)
            .run_with_topology(
                [
                    WarpTask::new(0, async { Err(EngineError::message("boom")) }),
                    WarpTask::new(1, async { Ok(()) }),
                ],
                topology,
            )
            .unwrap_err();
        assert_eq!(
            failed,
            EngineError::warp_failed(0, EngineError::message("boom"))
        );

        let panicked = Executor::with_max_workers(2)
            .run_with_topology(
                [
                    WarpTask::new(0, async {
                        panic!("future panic");
                        #[allow(unreachable_code)]
                        Ok(())
                    }),
                    WarpTask::new(1, async { Ok(()) }),
                ],
                topology,
            )
            .unwrap_err();
        assert_eq!(panicked, EngineError::warp_panicked(0, "future panic"));

        let deadlocked = Executor::with_max_workers(2)
            .run_with_topology(
                [WarpTask::new(0, NeverReady), WarpTask::new(1, NeverReady)],
                topology,
            )
            .unwrap_err();
        assert_eq!(deadlocked, EngineError::deadlock(vec![0, 1], vec![], 2));
    }

    #[test]
    fn topology_rejects_out_of_range_warp_ids() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let error = Executor::default()
            .run_with_topology([WarpTask::new(1, async { Ok(()) })], topology)
            .unwrap_err();
        assert_eq!(error, EngineError::warp_outside_topology(1, 1));
    }
}
