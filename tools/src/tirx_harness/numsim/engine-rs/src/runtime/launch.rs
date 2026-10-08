use std::collections::BTreeSet;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;

use crate::completion::CompletionPublicationRegistry;
use crate::mode_completions::ModeCompletionSource;
use crate::{
    AsyncGroupHub, ClusterBarrierHub, CompletionRegistry, CtaId, CtaReduceHub, DeferredPayloadHub,
    EngineError, EngineMode, ExecutionReport, ExecutionStats, Executor, KernelEngine,
    LaunchTopology, MbarrierInitFenceTracker, NamedBarrierHub, OrderingHub, PhysicalBarrierHub,
    PhysicalMemory, RendezvousHub, SetmaxnregHub, SharedAllocation, TcgenLifecycleHub,
    TmemAllocation, WarpContext, WarpEngine, WarpId, WarpPrivateAllocation, WarpPrivateMemory,
    WarpTask,
};

pub const DEFAULT_NATIVE_LOOP_ITERATION_BUDGET: usize = 1_000_000;
pub const DEFAULT_NATIVE_LOOP_RESCHEDULE_QUANTUM: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionPolicy {
    native_loop_iteration_budget: usize,
    native_loop_reschedule_quantum: usize,
    setmaxnreg_calling_initial_count: Option<u32>,
    tmem_columns: usize,
}

impl ExecutionPolicy {
    pub fn new(
        native_loop_iteration_budget: usize,
        native_loop_reschedule_quantum: usize,
    ) -> Result<Self, EngineError> {
        if native_loop_iteration_budget == 0 {
            return Err(EngineError::message(
                "NumSim native_loop_iteration_budget must be positive",
            ));
        }
        if native_loop_reschedule_quantum == 0 {
            return Err(EngineError::message(
                "NumSim native_loop_reschedule_quantum must be positive",
            ));
        }
        Ok(Self {
            native_loop_iteration_budget,
            native_loop_reschedule_quantum,
            setmaxnreg_calling_initial_count: None,
            tmem_columns: crate::TMEM_COLUMN_CAPACITY,
        })
    }

    pub const fn native_loop_iteration_budget(&self) -> usize {
        self.native_loop_iteration_budget
    }

    pub const fn native_loop_reschedule_quantum(&self) -> usize {
        self.native_loop_reschedule_quantum
    }

    /// Attach compiler-derived caller register metadata once at launch. This
    /// is deliberately not an operand of `setmaxnreg` instructions.
    pub const fn with_setmaxnreg_calling_initial_count(mut self, count: Option<u32>) -> Self {
        self.setmaxnreg_calling_initial_count = count;
        self
    }

    pub fn with_tmem_column_capacity(mut self, columns: usize) -> Self {
        self.tmem_columns = columns;
        self
    }
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            native_loop_iteration_budget: DEFAULT_NATIVE_LOOP_ITERATION_BUDGET,
            native_loop_reschedule_quantum: DEFAULT_NATIVE_LOOP_RESCHEDULE_QUANTUM,
            setmaxnreg_calling_initial_count: None,
            tmem_columns: crate::TMEM_COLUMN_CAPACITY,
        }
    }
}

/// Launch-scoped model of the pending logical clusters visible to CLC.
///
/// A NumSim execution subset represents the clusters that are already
/// resident.  `try_cancel` walks the launch's logical cluster IDs exactly
/// once and returns each non-resident task to one caller.
pub(crate) struct ClcTaskCounter {
    next_task: AtomicUsize,
    task_count: usize,
    ctas_per_cluster: usize,
    resident_clusters: BTreeSet<usize>,
}

impl ClcTaskCounter {
    fn for_launch(topology: LaunchTopology, selected_warps: &BTreeSet<usize>) -> Self {
        let resident_clusters = selected_warps
            .iter()
            .filter_map(|&warp_id| topology.cluster_id_for_warp(warp_id))
            .collect();
        Self {
            next_task: AtomicUsize::new(0),
            task_count: topology.clusters(),
            ctas_per_cluster: topology.ctas_per_cluster(),
            resident_clusters,
        }
    }

    pub(crate) fn try_cancel(&self) -> Result<u32, EngineError> {
        loop {
            let task = self.next_task.fetch_add(1, AtomicOrdering::Relaxed);
            if task >= self.task_count {
                return Ok(u32::MAX);
            }
            if self.resident_clusters.contains(&task) {
                continue;
            }
            let base_cta = task
                .checked_mul(self.ctas_per_cluster)
                .ok_or_else(|| EngineError::message("CLC base CTA ID overflow"))?;
            return u32::try_from(base_cta)
                .map_err(|_| EngineError::message("CLC base CTA ID exceeds uint32"));
        }
    }
}

#[derive(Clone)]
pub struct LaunchRuntimeServices {
    mbarriers: Arc<PhysicalBarrierHub>,
    mbarrier_init_fences: Arc<MbarrierInitFenceTracker>,
    deferred_payloads: Arc<DeferredPayloadHub>,
    named_barriers: Arc<NamedBarrierHub>,
    rendezvous: Arc<RendezvousHub>,
    cluster_barriers: Arc<ClusterBarrierHub>,
    cta_reduce: Arc<CtaReduceHub>,
    tcgen: Arc<TcgenLifecycleHub>,
    setmaxnreg: Arc<SetmaxnregHub>,
    async_groups: Arc<AsyncGroupHub>,
    clc_tasks: Arc<ClcTaskCounter>,
    completion_publications: Arc<CompletionPublicationRegistry>,
    ordering: Arc<OrderingHub>,
    execution_policy: ExecutionPolicy,
}

pub type KernelRuntimeServices = LaunchRuntimeServices;

impl LaunchRuntimeServices {
    fn new(
        topology: LaunchTopology,
        selected_warps: Arc<BTreeSet<usize>>,
        execution_policy: ExecutionPolicy,
        ordering: Arc<OrderingHub>,
    ) -> Result<Self, EngineError> {
        let mbarriers = Arc::new(PhysicalBarrierHub::new());
        let deferred_payloads = Arc::new(DeferredPayloadHub::new(Arc::clone(&mbarriers)));
        let setmaxnreg = Arc::new(SetmaxnregHub::new(topology));
        if let Some(count) = execution_policy.setmaxnreg_calling_initial_count {
            setmaxnreg.configure_calling_initial_count(i64::from(count))?;
        }
        Ok(Self {
            mbarriers,
            mbarrier_init_fences: Arc::new(MbarrierInitFenceTracker::new()),
            deferred_payloads,
            named_barriers: Arc::new(NamedBarrierHub::new()),
            rendezvous: Arc::new(RendezvousHub::for_launch(Arc::clone(&selected_warps))),
            cluster_barriers: Arc::new(ClusterBarrierHub::for_launch(
                topology,
                Arc::clone(&selected_warps),
            )),
            cta_reduce: Arc::new(CtaReduceHub::cta_reduce_hub(topology)),
            tcgen: Arc::new(TcgenLifecycleHub::with_column_capacity(
                topology,
                execution_policy.tmem_columns,
            )),
            setmaxnreg,
            async_groups: Arc::new(AsyncGroupHub::new(topology)),
            clc_tasks: Arc::new(ClcTaskCounter::for_launch(topology, &selected_warps)),
            completion_publications: Arc::new(CompletionPublicationRegistry::default()),
            ordering,
            execution_policy,
        })
    }

    pub fn mbarriers(&self) -> Arc<PhysicalBarrierHub> {
        self.mbarriers.clone()
    }

    pub(crate) fn mbarrier_init_fences(&self) -> Arc<MbarrierInitFenceTracker> {
        Arc::clone(&self.mbarrier_init_fences)
    }

    pub fn deferred_payloads(&self) -> Arc<DeferredPayloadHub> {
        self.deferred_payloads.clone()
    }

    pub fn named_barriers(&self) -> Arc<NamedBarrierHub> {
        self.named_barriers.clone()
    }

    pub fn rendezvous(&self) -> Arc<RendezvousHub> {
        self.rendezvous.clone()
    }

    pub fn cluster_barriers(&self) -> Arc<ClusterBarrierHub> {
        self.cluster_barriers.clone()
    }

    pub fn cta_reduce(&self) -> Arc<CtaReduceHub> {
        self.cta_reduce.clone()
    }

    pub fn tcgen(&self) -> Arc<TcgenLifecycleHub> {
        self.tcgen.clone()
    }

    pub fn setmaxnreg(&self) -> Arc<SetmaxnregHub> {
        self.setmaxnreg.clone()
    }

    pub fn async_groups(&self) -> Arc<AsyncGroupHub> {
        self.async_groups.clone()
    }

    pub(crate) fn clc_tasks(&self) -> Arc<ClcTaskCounter> {
        Arc::clone(&self.clc_tasks)
    }

    pub(crate) fn completion_publications(&self) -> Arc<CompletionPublicationRegistry> {
        Arc::clone(&self.completion_publications)
    }

    pub(crate) fn ordering(&self) -> Arc<OrderingHub> {
        self.ordering.clone()
    }

    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.execution_policy
    }

    fn completion_registry(&self) -> CompletionRegistry {
        let mut completions = CompletionRegistry::new();
        completions.register(self.deferred_payloads());
        completions.register(self.named_barriers());
        completions.register(self.rendezvous());
        completions.register(self.cluster_barriers());
        completions.register(self.cta_reduce());
        completions.register(self.tcgen());
        completions.register(self.setmaxnreg());
        completions.register(self.async_groups());
        completions.register(self.ordering());
        completions
    }

    fn completion_registry_for_mode<M: EngineMode>(
        &self,
        physical: PhysicalMemory,
        mode_state: Arc<M::LaunchState>,
        topology: LaunchTopology,
        max_workers: usize,
    ) -> CompletionRegistry {
        let mode_completions = Arc::new(ModeCompletionSource::<M>::new(
            self.deferred_payloads(),
            self.async_groups(),
            self.setmaxnreg(),
            self.completion_publications(),
            physical,
            mode_state,
            topology,
            max_workers,
        ));
        let mut completions = CompletionRegistry::new();
        completions.register(mode_completions);
        completions.register(self.named_barriers());
        completions.register(self.rendezvous());
        completions.register(self.cluster_barriers());
        completions.register(self.cta_reduce());
        completions.register(self.tcgen());
        completions.register(self.ordering());
        completions
    }
}

#[derive(Clone, Debug, Default)]
pub struct LaunchSelection {
    cluster_ids: Option<Vec<usize>>,
    cta_ids: Option<Vec<usize>>,
}

impl LaunchSelection {
    pub fn new(mut cluster_ids: Option<Vec<usize>>, mut cta_ids: Option<Vec<usize>>) -> Self {
        for ids in [&mut cluster_ids, &mut cta_ids].into_iter().flatten() {
            ids.sort_unstable();
            ids.dedup();
        }
        Self {
            cluster_ids,
            cta_ids,
        }
    }

    pub(crate) fn validate(&self, topology: LaunchTopology) -> Result<(), EngineError> {
        if let Some(cluster_id) = self
            .cluster_ids
            .as_ref()
            .and_then(|ids| ids.iter().find(|&&id| id >= topology.clusters()))
        {
            return Err(EngineError::message(format!(
                "NumSim subset cluster ID {cluster_id} is outside [0, {})",
                topology.clusters()
            )));
        }
        let Some(cta_ids) = self.cta_ids.as_ref() else {
            return Ok(());
        };
        if let Some(cta_id) = cta_ids.iter().find(|&&id| id >= topology.cta_count()) {
            return Err(EngineError::message(format!(
                "NumSim subset CTA ID {cta_id} is outside [0, {})",
                topology.cta_count()
            )));
        }
        if topology.ctas_per_cluster() == 1 {
            return Ok(());
        }

        for selected in cta_ids.chunks(topology.ctas_per_cluster()) {
            let first = selected[0];
            let cluster_id = first / topology.ctas_per_cluster();
            let start = cluster_id * topology.ctas_per_cluster();
            let end = start + topology.ctas_per_cluster();
            if selected.len() != topology.ctas_per_cluster()
                || !selected.iter().copied().eq(start..end)
            {
                return Err(EngineError::message(format!(
                    "NumSim CTA subset must be a union of complete clusters; cluster \
                     {cluster_id} contains global CTA IDs {start}..{end}, but selected {selected:?}"
                )));
            }
        }
        Ok(())
    }

    pub fn includes(&self, context: WarpContext) -> bool {
        let cluster_selected = self
            .cluster_ids
            .as_ref()
            .map(|ids| ids.binary_search(&context.cluster_id()).is_ok())
            .unwrap_or(true);
        let cta_selected = self
            .cta_ids
            .as_ref()
            .map(|ids| ids.binary_search(&context.global_cta_id()).is_ok())
            .unwrap_or(true);
        cluster_selected && cta_selected
    }

    pub(crate) fn selected_warp_count(&self, topology: LaunchTopology) -> usize {
        topology
            .warp_contexts()
            .filter(|context| self.includes(*context))
            .count()
    }

    pub(crate) fn selected_cluster_ids(&self, topology: LaunchTopology) -> Vec<usize> {
        topology
            .warp_contexts()
            .filter(|context| self.includes(*context))
            .map(WarpContext::cluster_id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

}

/// Launch-scoped runtime state prepared independently from any executor.
///
/// Keeping selection, contexts, services, and completion sources together lets
/// alternative executors reuse the exact same launch assembly without changing
/// numerical runtime semantics.
pub(crate) struct PreparedLaunch {
    topology: LaunchTopology,
    contexts: Vec<WarpContext>,
    services: KernelRuntimeServices,
    completions: CompletionRegistry,
}

impl PreparedLaunch {
    pub(crate) fn new(
        topology: LaunchTopology,
        selection: LaunchSelection,
        execution_policy: ExecutionPolicy,
    ) -> Result<Self, EngineError> {
        Self::new_ordered(
            topology,
            selection,
            execution_policy,
            Arc::new(OrderingHub::new(topology)),
        )
    }

    pub(crate) fn new_ordered(
        topology: LaunchTopology,
        selection: LaunchSelection,
        execution_policy: ExecutionPolicy,
        ordering: Arc<OrderingHub>,
    ) -> Result<Self, EngineError> {
        selection.validate(topology)?;
        let contexts = topology
            .warp_contexts()
            .filter(|context| selection.includes(*context))
            .collect::<Vec<_>>();
        if contexts.is_empty() {
            return Err(EngineError::message(
                "NumSim subset resolves to no executable warps",
            ));
        }
        let selected_warps = Arc::new(
            contexts
                .iter()
                .map(|context| context.global_warp_id())
                .collect::<BTreeSet<_>>(),
        );
        let services =
            KernelRuntimeServices::new(topology, selected_warps, execution_policy, ordering)?;
        let completions = services.completion_registry();
        Ok(Self {
            topology,
            contexts,
            services,
            completions,
        })
    }

    pub(crate) const fn topology(&self) -> LaunchTopology {
        self.topology
    }

    pub(crate) fn contexts(&self) -> &[WarpContext] {
        &self.contexts
    }

    pub(crate) const fn services(&self) -> &KernelRuntimeServices {
        &self.services
    }

    pub(crate) const fn completions(&self) -> &CompletionRegistry {
        &self.completions
    }

    fn run<MakeWarp, WarpFuture>(
        self,
        max_workers: usize,
        make_warp: MakeWarp,
    ) -> Result<ExecutionStats, EngineError>
    where
        MakeWarp: FnMut(WarpContext, KernelRuntimeServices) -> WarpFuture,
        WarpFuture: Future<Output = Result<(), EngineError>> + Send + 'static,
    {
        self.run_report(max_workers, None, None, make_warp)
            .into_result()
    }

    fn run_report<MakeWarp, WarpFuture>(
        self,
        max_workers: usize,
        poll_limit: Option<usize>,
        mode_completions: Option<CompletionRegistry>,
        mut make_warp: MakeWarp,
    ) -> ExecutionReport
    where
        MakeWarp: FnMut(WarpContext, KernelRuntimeServices) -> WarpFuture,
        WarpFuture: Future<Output = Result<(), EngineError>> + Send + 'static,
    {
        let tasks = self.tasks(|context, services| make_warp(context, services));
        let completions = mode_completions.as_ref().unwrap_or(self.completions());
        let mut report = Executor::with_max_workers_and_poll_limit(max_workers, poll_limit)
            .run_with_topology_and_completions_report(tasks, self.topology(), completions);
        if report.is_success() {
            if let Some((blocked_warps, blocked_operations)) =
                self.services.rendezvous().participation_deadlock()
            {
                report.terminal = Err(EngineError::deadlock(
                    blocked_warps,
                    blocked_operations,
                    report.stats.poll_count,
                ));
            }
        }
        if report.is_success() {
            match self
                .services
                .async_groups()
                .commit_open_bulk_groups_for_exit()
            {
                Ok(groups) => {
                    self.services
                        .completion_publications()
                        .publish_async_groups(groups.iter().flat_map(|group| {
                            [group.source_read_action_id(), group.full_action_id()]
                        }))
                }
                Err(error) => report.terminal = Err(error),
            }
        }
        if report.is_success() {
            match completions.drain_to_stable_and_validate() {
                Ok(drained) => {
                    report.stats.completion_pump_count += drained.pump_count;
                    report.stats.completion_operation_count += drained.completed_operation_count;
                }
                Err(error) => report.terminal = Err(error.into()),
            }
        }
        if report.is_success() {
            if let Err(error) = self.services.ordering.validate_quiescent() {
                report.terminal = Err(error);
            }
        }
        report
    }

    fn tasks<MakeWarp, WarpFuture>(&self, mut make_warp: MakeWarp) -> Vec<WarpTask>
    where
        MakeWarp: FnMut(WarpContext, KernelRuntimeServices) -> WarpFuture,
        WarpFuture: Future<Output = Result<(), EngineError>> + Send + 'static,
    {
        self.contexts()
            .iter()
            .copied()
            .map(|context| {
                WarpTask::new(
                    context.global_warp_id(),
                    make_warp(context, self.services().clone()),
                )
            })
            .collect()
    }
}

pub(crate) fn execute_kernel_launch<MakeWarp, WarpFuture>(
    topology: LaunchTopology,
    selection: LaunchSelection,
    max_workers: usize,
    execution_policy: ExecutionPolicy,
    make_warp: MakeWarp,
) -> Result<ExecutionStats, EngineError>
where
    MakeWarp: FnMut(WarpContext, LaunchRuntimeServices) -> WarpFuture,
    WarpFuture: Future<Output = Result<(), EngineError>> + Send + 'static,
{
    execute_kernel_launch_ordered(
        topology,
        selection,
        max_workers,
        execution_policy,
        Arc::new(OrderingHub::new(topology)),
        make_warp,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_kernel_launch_ordered<MakeWarp, WarpFuture>(
    topology: LaunchTopology,
    selection: LaunchSelection,
    max_workers: usize,
    execution_policy: ExecutionPolicy,
    ordering: Arc<OrderingHub>,
    make_warp: MakeWarp,
) -> Result<ExecutionStats, EngineError>
where
    MakeWarp: FnMut(WarpContext, LaunchRuntimeServices) -> WarpFuture,
    WarpFuture: Future<Output = Result<(), EngineError>> + Send + 'static,
{
    PreparedLaunch::new_ordered(topology, selection, execution_policy, ordering)?
        .run(max_workers, make_warp)
}

/// Run one mode-generic kernel body over the existing numerical runtime.
pub fn run_kernel_engine_launch<M, MakeWarp, WarpFuture>(
    physical: PhysicalMemory,
    kernel_index: usize,
    mode_state: Arc<M::LaunchState>,
    selection: LaunchSelection,
    max_workers: usize,
    execution_policy: ExecutionPolicy,
    make_warp: MakeWarp,
) -> Result<ExecutionStats, EngineError>
where
    M: EngineMode,
    MakeWarp: FnMut(WarpEngine<M>) -> WarpFuture,
    WarpFuture: Future<Output = Result<(), EngineError>> + Send + 'static,
{
    run_kernel_engine_launch_report(
        physical,
        kernel_index,
        mode_state,
        selection,
        max_workers,
        execution_policy,
        make_warp,
    )
    .into_result()
}

/// Ordinary launch variant that retains partial statistics and its typed
/// terminal error for analysis payloads.
pub fn run_kernel_engine_launch_report<M, MakeWarp, WarpFuture>(
    physical: PhysicalMemory,
    kernel_index: usize,
    mode_state: Arc<M::LaunchState>,
    selection: LaunchSelection,
    max_workers: usize,
    execution_policy: ExecutionPolicy,
    make_warp: MakeWarp,
) -> ExecutionReport
where
    M: EngineMode,
    MakeWarp: FnMut(WarpEngine<M>) -> WarpFuture,
    WarpFuture: Future<Output = Result<(), EngineError>> + Send + 'static,
{
    run_kernel_engine_launch_report_with_poll_limit(
        physical,
        kernel_index,
        mode_state,
        selection,
        max_workers,
        None,
        execution_policy,
        make_warp,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_kernel_engine_launch_report_with_poll_limit<M, MakeWarp, WarpFuture>(
    physical: PhysicalMemory,
    kernel_index: usize,
    mode_state: Arc<M::LaunchState>,
    selection: LaunchSelection,
    max_workers: usize,
    poll_limit: Option<usize>,
    execution_policy: ExecutionPolicy,
    make_warp: MakeWarp,
) -> ExecutionReport
where
    M: EngineMode,
    MakeWarp: FnMut(WarpEngine<M>) -> WarpFuture,
    WarpFuture: Future<Output = Result<(), EngineError>> + Send + 'static,
{
    run_kernel_engine_launch_ordered_report(
        physical,
        kernel_index,
        mode_state,
        selection,
        max_workers,
        poll_limit,
        execution_policy,
        make_warp,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_kernel_engine_launch_ordered_report<M, MakeWarp, WarpFuture>(
    physical: PhysicalMemory,
    kernel_index: usize,
    mode_state: Arc<M::LaunchState>,
    selection: LaunchSelection,
    max_workers: usize,
    poll_limit: Option<usize>,
    execution_policy: ExecutionPolicy,
    mut make_warp: MakeWarp,
) -> ExecutionReport
where
    M: EngineMode,
    MakeWarp: FnMut(WarpEngine<M>) -> WarpFuture,
    WarpFuture: Future<Output = Result<(), EngineError>> + Send + 'static,
{
    let topology = physical.topology();
    let prepared = match PreparedLaunch::new_ordered(
        topology,
        selection,
        execution_policy,
        physical.ordering(),
    ) {
        Ok(prepared) => prepared,
        Err(error) => return ExecutionReport::failure(ExecutionStats::default(), error),
    };
    let mode_completions = prepared.services().completion_registry_for_mode::<M>(
        physical.clone(),
        Arc::clone(&mode_state),
        prepared.topology(),
        max_workers,
    );
    prepared.run_report(
        max_workers,
        poll_limit,
        Some(mode_completions),
        move |context, services| {
            let kernel = KernelEngine::<M>::new(
                kernel_index,
                physical.clone(),
                services,
                Arc::clone(&mode_state),
            );
            make_warp(WarpEngine::new(context, kernel))
        },
    )
}

pub fn allocate_cta_shared(
    physical: &PhysicalMemory,
    topology: LaunchTopology,
    byte_len: usize,
) -> Result<Vec<SharedAllocation>, EngineError> {
    let mut allocations = Vec::with_capacity(topology.cta_count());
    for cluster_id in 0..topology.clusters() {
        for cta_id_in_cluster in 0..topology.ctas_per_cluster() {
            let owner = CtaId::new(topology, cluster_id, cta_id_in_cluster)?;
            allocations.push(
                physical
                    .shared()
                    .allocate_cta_uninitialized(owner, byte_len)?,
            );
        }
    }
    Ok(allocations)
}

pub fn allocate_warp_private(
    memory: &WarpPrivateMemory,
    topology: LaunchTopology,
    bytes_per_lane: usize,
) -> Result<Vec<WarpPrivateAllocation>, EngineError> {
    let mut allocations = Vec::with_capacity(topology.warp_count());
    for context in topology.warp_contexts() {
        allocations
            .push(memory.allocate_uninitialized(WarpId::from_context(context), bytes_per_lane)?);
    }
    Ok(allocations)
}

// Backings and allocation lifetimes are CTA-local. Physical SM placement and
// exclusive-allocation resource contention are intentionally not modeled.
pub fn allocate_cta_tmem(
    physical: &PhysicalMemory,
    topology: LaunchTopology,
    lanes: usize,
    columns: usize,
) -> Result<Vec<TmemAllocation>, EngineError> {
    let mut allocations = Vec::with_capacity(topology.cta_count());
    for cluster_id in 0..topology.clusters() {
        for cta_id_in_cluster in 0..topology.ctas_per_cluster() {
            let owner = CtaId::new(topology, cluster_id, cta_id_in_cluster)?;
            allocations.push(
                physical
                    .tmem()
                    .allocate_uninitialized(owner, lanes, columns)?,
            );
        }
    }
    Ok(allocations)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use crate::runtime::{load_scalar_warp, PhysicalPtr, RuntimeBuffer};
    use crate::{
        CtaId, LaunchTopology, OperationEffect, OperationKind, OrderingHub, PhysicalAccessKind,
        PhysicalBarrierId, PhysicalMemory, StaticOpId, WarpMask, WarpValue,
    };

    use super::{
        execute_kernel_launch, run_kernel_engine_launch, ExecutionPolicy, LaunchRuntimeServices,
        LaunchSelection, PreparedLaunch,
    };

    struct CountingMode;

    impl crate::engine_mode::EngineModeImpl for CountingMode {
        type LaunchState = AtomicUsize;
        type GlobalMemoryTransactionGuard<'a> = ();

        const NAME: &'static str = "counting";
        const OBSERVES_OPERATIONS: bool = true;

        fn before_operation(
            state: &Self::LaunchState,
            _operation: &crate::OperationContext,
        ) -> Result<(), crate::EngineError> {
            state.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn after_operation(
            state: &Self::LaunchState,
            _operation: &crate::OperationContext,
        ) -> Result<(), crate::EngineError> {
            state.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    struct EffectRecordingMode;

    #[derive(Default)]
    struct EffectRecordingState {
        events: Mutex<Vec<String>>,
        operation_events: Mutex<Vec<String>>,
        physical_accesses: Mutex<Vec<(String, PhysicalAccessKind, Vec<usize>)>>,
        reject_before_effect: bool,
    }

    impl crate::engine_mode::EngineModeImpl for EffectRecordingMode {
        type LaunchState = EffectRecordingState;
        type GlobalMemoryTransactionGuard<'a> = ();

        const NAME: &'static str = "effect-recording";
        const OBSERVES_OPERATIONS: bool = true;

        fn before_operation(
            state: &Self::LaunchState,
            operation: &crate::OperationContext,
        ) -> Result<(), crate::EngineError> {
            state
                .operation_events
                .lock()
                .unwrap()
                .push(format!("before:{}", operation.id().source_op_id()));
            Ok(())
        }

        fn after_operation(
            state: &Self::LaunchState,
            operation: &crate::OperationContext,
        ) -> Result<(), crate::EngineError> {
            state
                .operation_events
                .lock()
                .unwrap()
                .push(format!("after:{}", operation.id().source_op_id()));
            Ok(())
        }

        fn before_effect(
            state: &Self::LaunchState,
            operation: &crate::OperationContext,
            effect: OperationEffect<'_>,
        ) -> Result<(), crate::EngineError> {
            if let OperationEffect::PhysicalAccess(batch) = effect {
                state.physical_accesses.lock().unwrap().push((
                    batch.logical_buffer().unwrap_or_default().to_string(),
                    batch.descriptor().kind(),
                    batch
                        .lanes()
                        .iter()
                        .map(|access| access.provenance().lane())
                        .collect(),
                ));
            }
            state.events.lock().unwrap().push(format!(
                "before:{}:{}",
                operation.id().source_op_id(),
                effect.name()
            ));
            if state.reject_before_effect {
                return Err(crate::EngineError::message("effect rejected before apply"));
            }
            Ok(())
        }

        fn after_effect(
            state: &Self::LaunchState,
            operation: &crate::OperationContext,
            effect: OperationEffect<'_>,
        ) -> Result<(), crate::EngineError> {
            state.events.lock().unwrap().push(format!(
                "after:{}:{}",
                operation.id().source_op_id(),
                effect.name()
            ));
            Ok(())
        }
    }

    fn one_lane_shared_pointer(physical: &PhysicalMemory) -> PhysicalPtr {
        let topology = physical.topology();
        let owner = CtaId::new(topology, 0, 0).unwrap();
        let allocation = physical.shared().allocate_cta_zeroed(owner, 8).unwrap();
        PhysicalPtr::new(
            RuntimeBuffer::Shared {
                allocations: Arc::new(vec![allocation]),
                byte_offset: 0,
                byte_len: 8,
                backing_byte_len: 8,
                virtual_base: 0,
            },
            WarpValue::splat(0_i64),
            8,
        )
    }

    #[test]
    fn execution_policy_defaults_and_constructor_are_validated() {
        let defaults = ExecutionPolicy::default();
        assert_eq!(defaults.native_loop_iteration_budget(), 1_000_000);
        assert_eq!(defaults.native_loop_reschedule_quantum(), 64);

        let configured = ExecutionPolicy::new(2_000_000, 17).unwrap();
        assert_eq!(configured.native_loop_iteration_budget(), 2_000_000);
        assert_eq!(configured.native_loop_reschedule_quantum(), 17);
        assert!(ExecutionPolicy::new(0, 17)
            .unwrap_err()
            .to_string()
            .contains("native_loop_iteration_budget must be positive"));
        assert!(ExecutionPolicy::new(17, 0)
            .unwrap_err()
            .to_string()
            .contains("native_loop_reschedule_quantum must be positive"));
    }

    #[test]
    fn kernel_runtime_services_clone_one_launch_scoped_service_set() {
        let topology = LaunchTopology::new(2, 2, 4).unwrap();
        let services = LaunchRuntimeServices::new(
            topology,
            Arc::new((0..topology.warp_count()).collect()),
            ExecutionPolicy::new(123, 7).unwrap(),
            Arc::new(OrderingHub::new(topology)),
        )
        .unwrap();
        let clone = services.clone();

        assert!(Arc::ptr_eq(&services.mbarriers(), &clone.mbarriers()));
        assert!(Arc::ptr_eq(
            &services.named_barriers(),
            &clone.named_barriers()
        ));
        assert!(Arc::ptr_eq(&services.rendezvous(), &clone.rendezvous()));
        assert!(Arc::ptr_eq(
            &services.cluster_barriers(),
            &clone.cluster_barriers()
        ));
        assert!(Arc::ptr_eq(&services.cta_reduce(), &clone.cta_reduce()));
        assert!(Arc::ptr_eq(&services.tcgen(), &clone.tcgen()));
        assert_eq!(
            services.execution_policy().native_loop_iteration_budget(),
            123
        );
        assert_eq!(
            services.execution_policy().native_loop_reschedule_quantum(),
            7
        );

        let _completions = services.completion_registry();
    }

    #[test]
    fn prepared_launch_exposes_selected_contexts_services_and_completions() {
        let topology = LaunchTopology::new(2, 2, 2).unwrap();
        let prepared = PreparedLaunch::new(
            topology,
            LaunchSelection::new(Some(vec![1]), Some(vec![2, 3])),
            ExecutionPolicy::new(123, 7).unwrap(),
        )
        .unwrap();

        assert_eq!(prepared.topology(), topology);
        assert_eq!(
            prepared
                .contexts()
                .iter()
                .map(|context| context.global_warp_id())
                .collect::<Vec<_>>(),
            vec![4, 5, 6, 7]
        );
        assert_eq!(
            prepared
                .services()
                .execution_policy()
                .native_loop_iteration_budget(),
            123
        );
        assert_eq!(prepared.completions().len(), 9);
    }

    #[test]
    fn run_kernel_launch_owns_task_assembly_and_subset_filtering() {
        let topology = LaunchTopology::new(2, 2, 2).unwrap();
        let visited = Arc::new(Mutex::new(Vec::new()));
        let visited_by_warps = visited.clone();

        let stats = execute_kernel_launch(
            topology,
            LaunchSelection::new(Some(vec![1]), Some(vec![2, 3])),
            8,
            ExecutionPolicy::default(),
            move |context, _services| {
                let visited = visited_by_warps.clone();
                async move {
                    visited.lock().unwrap().push(context.global_warp_id());
                    Ok(())
                }
            },
        )
        .unwrap();

        let mut visited = visited.lock().unwrap().clone();
        visited.sort_unstable();
        assert_eq!(visited, vec![4, 5, 6, 7]);
        assert_eq!(stats.task_count, 4);
        assert_eq!(stats.completed_task_count, 4);
        assert_eq!(stats.scheduling_domain_count, 1);
        assert_eq!(stats.worker_count, 1);
    }

    #[test]
    fn mode_generic_launch_shares_one_engine_state_across_warps() {
        let topology = LaunchTopology::new(1, 1, 3).unwrap();
        let physical = PhysicalMemory::new(topology);
        let state = Arc::new(AtomicUsize::new(0));

        let stats = run_kernel_engine_launch::<CountingMode, _, _>(
            physical,
            5,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            |mut warp| async move {
                assert_eq!(warp.kernel().mode_name(), "counting");
                assert_eq!(warp.kernel().kernel_index(), 5);
                assert_eq!(
                    warp.kernel().physical().topology(),
                    warp.context().topology()
                );
                let masked = warp
                    .context()
                    .with_active_mask(WarpMask::from_lanes([0, 3]).unwrap());
                warp.push_loop_frame(9, 2).unwrap();
                let first = warp
                    .begin_current_operation(masked, StaticOpId::new(17), OperationKind::Barrier)
                    .unwrap();
                warp.finish_operation(&first).unwrap();
                warp.pop_loop_frame(9).unwrap();
                let second = warp
                    .begin_current_operation(masked, StaticOpId::new(18), OperationKind::Store)
                    .unwrap();
                warp.finish_operation(&second).unwrap();
                assert_eq!(first.id().kernel_index(), 5);
                assert_eq!(first.id().global_warp_id(), warp.context().global_warp_id());
                assert_eq!(first.id().per_warp_sequence(), 0);
                assert_eq!(first.active_mask(), masked.active_mask());
                assert_eq!(first.id().loop_frames().len(), 1);
                assert_eq!(second.id().per_warp_sequence(), 1);
                assert!(second.id().loop_frames().is_empty());
                warp.push_loop_frame(99, 0).unwrap();
                assert!(warp
                    .pop_loop_frame(100)
                    .unwrap_err()
                    .to_string()
                    .contains("loop frame stack mismatch"));
                assert!(warp
                    .pop_loop_frame(99)
                    .unwrap_err()
                    .to_string()
                    .contains("loop frame stack underflow"));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(state.load(Ordering::SeqCst), 12);
        assert_eq!(stats.completed_task_count, 3);
    }

    #[test]
    fn typed_effect_hook_runs_before_and_after_the_numeric_mbarrier_effect() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let pointer = one_lane_shared_pointer(&physical);
        let state = Arc::new(EffectRecordingState::default());

        let stats = run_kernel_engine_launch::<EffectRecordingMode, _, _>(
            physical,
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let pointer = pointer.clone();
                async move {
                    let operation = warp.begin_operation(
                        warp.context(),
                        StaticOpId::new(41),
                        OperationKind::Barrier,
                        [],
                    )?;
                    warp.mbarrier_init(
                        Some(&operation),
                        &pointer,
                        WarpMask::from_lanes([0]).unwrap(),
                        &WarpValue::splat(1),
                    )?;
                    warp.finish_operation(&operation)?;
                    Ok(())
                }
            },
        )
        .unwrap();

        assert_eq!(stats.completed_task_count, 1);
        assert_eq!(
            state.events.lock().unwrap().as_slice(),
            ["before:op:41:mbarrier.init", "after:op:41:mbarrier.init"]
        );
    }

    #[test]
    fn typed_effect_rejection_prevents_numeric_mbarrier_mutation() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let pointer = one_lane_shared_pointer(&physical);
        let state = Arc::new(EffectRecordingState {
            events: Mutex::new(Vec::new()),
            reject_before_effect: true,
            ..EffectRecordingState::default()
        });

        let error = run_kernel_engine_launch::<EffectRecordingMode, _, _>(
            physical,
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let pointer = pointer.clone();
                async move {
                    let operation = warp.begin_operation(
                        warp.context(),
                        StaticOpId::new(42),
                        OperationKind::Barrier,
                        [],
                    )?;
                    warp.mbarrier_init(
                        Some(&operation),
                        &pointer,
                        WarpMask::from_lanes([0]).unwrap(),
                        &WarpValue::splat(1),
                    )?;
                    Ok(())
                }
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("effect rejected before apply"));
        assert_eq!(
            state.events.lock().unwrap().as_slice(),
            ["before:op:42:mbarrier.init"]
        );
    }

    #[test]
    fn physical_access_gateway_resolves_every_active_lane_before_numeric_execution() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocation = physical.global().allocate_zeroed(8).unwrap();
        let buffer = RuntimeBuffer::Global(physical.global().full_view(allocation).unwrap());
        let state = Arc::new(EffectRecordingState::default());
        let numeric_ran = Arc::new(AtomicBool::new(false));
        let numeric_ran_by_warp = Arc::clone(&numeric_ran);

        let error = run_kernel_engine_launch::<EffectRecordingMode, _, _>(
            physical,
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let buffer = buffer.clone();
                let numeric_ran = Arc::clone(&numeric_ran_by_warp);
                async move {
                    let mask = WarpMask::from_lanes([0, 1]).unwrap();
                    let context = warp.context().with_active_mask(mask);
                    let indices = WarpValue::from_fn(|lane| if lane == 0 { 0 } else { 99 });
                    let operation = warp.begin_operation(
                        context,
                        StaticOpId::new(50),
                        OperationKind::Load,
                        [],
                    )?;
                    let physical = warp.kernel().physical().clone();
                    warp.runtime_named_physical_access(
                        Some(&operation),
                        OperationKind::Load,
                        4,
                        4,
                        "input",
                        &buffer,
                        &indices,
                        mask,
                        false,
                        &mut || {
                            numeric_ran.store(true, Ordering::SeqCst);
                            load_scalar_warp::<u32>(&physical, &context, &buffer, &indices, mask)
                        },
                    )?;
                    Ok(())
                }
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("buffer element 99"));
        assert!(!numeric_ran.load(Ordering::SeqCst));
        assert!(state.events.lock().unwrap().is_empty());
    }

    #[test]
    fn named_scalar_load_facade_preserves_mask_and_logical_buffer() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocation = physical
            .global()
            .allocate_from_bytes([17_u32.to_le_bytes(), 29_u32.to_le_bytes()].concat())
            .unwrap();
        let buffer = RuntimeBuffer::Global(physical.global().full_view(allocation).unwrap());
        let state = Arc::new(EffectRecordingState::default());

        let stats = run_kernel_engine_launch::<EffectRecordingMode, _, _>(
            physical,
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let buffer = buffer.clone();
                async move {
                    let mask = WarpMask::from_lanes([0]).unwrap();
                    let indices = WarpValue::from_fn(|lane| if lane == 0 { 0 } else { 99 });
                    let operation = warp.begin_optional_physical_operation(
                        warp.context().with_active_mask(mask),
                        53,
                        OperationKind::Load,
                        &buffer,
                    )?;
                    let values = warp.load_named_scalar::<u32>(
                        operation.as_ref(),
                        warp.context(),
                        4,
                        "input",
                        &buffer,
                        &indices,
                        mask,
                        None,
                    )?;
                    warp.finish_optional_operation(&operation)?;
                    assert_eq!(values[0], 17);
                    Ok(())
                }
            },
        )
        .unwrap();

        assert_eq!(stats.completed_task_count, 1);
        assert_eq!(
            state.events.lock().unwrap().as_slice(),
            [
                "before:op:53:physical_access",
                "after:op:53:physical_access"
            ]
        );
        assert_eq!(
            state.operation_events.lock().unwrap().as_slice(),
            ["before:op:53", "after:op:53"]
        );
        assert_eq!(
            state.physical_accesses.lock().unwrap().as_slice(),
            [("input".to_string(), PhysicalAccessKind::Read, vec![0])]
        );
    }

    #[test]
    fn named_scalar_store_facade_rejection_prevents_numeric_write_and_finish() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocation = physical.global().allocate_zeroed(4).unwrap();
        let view = physical.global().full_view(allocation).unwrap();
        let buffer = RuntimeBuffer::Global(view.clone());
        let inspected_physical = physical.clone();
        let state = Arc::new(EffectRecordingState {
            reject_before_effect: true,
            ..EffectRecordingState::default()
        });

        let error = run_kernel_engine_launch::<EffectRecordingMode, _, _>(
            physical,
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let buffer = buffer.clone();
                async move {
                    let mask = WarpMask::from_lanes([0]).unwrap();
                    let context = warp.context();
                    let operation = warp.begin_optional_physical_operation(
                        context.with_active_mask(mask),
                        54,
                        OperationKind::Store,
                        &buffer,
                    )?;
                    warp.store_named_scalar::<u32>(
                        operation.as_ref(),
                        context,
                        4,
                        "output",
                        &buffer,
                        &WarpValue::splat(0_i64),
                        &WarpValue::splat(37_u32),
                        mask,
                    )?;
                    warp.finish_optional_operation(&operation)?;
                    Ok(())
                }
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("effect rejected before apply"));
        assert_eq!(
            inspected_physical.global().read_bytes(&view, 0, 4).unwrap(),
            [0, 0, 0, 0]
        );
        assert_eq!(
            state.events.lock().unwrap().as_slice(),
            ["before:op:54:physical_access"]
        );
        assert_eq!(
            state.operation_events.lock().unwrap().as_slice(),
            ["before:op:54"]
        );
    }

    #[test]
    fn physical_access_gateway_ignores_concrete_oob_lanes_excluded_by_mask() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocation = physical
            .global()
            .allocate_from_bytes([17_u32.to_le_bytes(), 29_u32.to_le_bytes()].concat())
            .unwrap();
        let buffer = RuntimeBuffer::Global(physical.global().full_view(allocation).unwrap());
        let state = Arc::new(EffectRecordingState::default());
        let state_by_warp = Arc::clone(&state);

        let stats = run_kernel_engine_launch::<EffectRecordingMode, _, _>(
            physical,
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let buffer = buffer.clone();
                let state = Arc::clone(&state_by_warp);
                async move {
                    let mask = WarpMask::from_lanes([0]).unwrap();
                    let context = warp.context().with_active_mask(mask);
                    let indices = WarpValue::from_fn(|lane| if lane == 0 { 0 } else { 99 });
                    let operation = warp.begin_operation(
                        context,
                        StaticOpId::new(51),
                        OperationKind::Load,
                        [],
                    )?;
                    let physical = warp.kernel().physical().clone();
                    let values = warp.runtime_named_physical_access(
                        Some(&operation),
                        OperationKind::Load,
                        4,
                        4,
                        "input",
                        &buffer,
                        &indices,
                        mask,
                        false,
                        &mut || {
                            state.events.lock().unwrap().push("numeric".to_string());
                            load_scalar_warp::<u32>(&physical, &context, &buffer, &indices, mask)
                        },
                    )?;
                    assert_eq!(values[0], 17);
                    warp.finish_operation(&operation)?;
                    Ok(())
                }
            },
        )
        .unwrap();

        assert_eq!(stats.completed_task_count, 1);
        assert_eq!(
            state.events.lock().unwrap().as_slice(),
            [
                "before:op:51:physical_access",
                "numeric",
                "after:op:51:physical_access"
            ]
        );
    }

    #[test]
    fn named_physical_access_propagates_numeric_error_before_after_effect() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocation = physical.global().allocate_zeroed(4).unwrap();
        let buffer = RuntimeBuffer::Global(physical.global().full_view(allocation).unwrap());
        let state = Arc::new(EffectRecordingState::default());
        let state_by_warp = Arc::clone(&state);

        let error = run_kernel_engine_launch::<EffectRecordingMode, _, _>(
            physical,
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let buffer = buffer.clone();
                let state = Arc::clone(&state_by_warp);
                async move {
                    let mask = WarpMask::from_lanes([0]).unwrap();
                    let context = warp.context().with_active_mask(mask);
                    let operation = warp.begin_operation(
                        context,
                        StaticOpId::new(52),
                        OperationKind::Load,
                        [],
                    )?;
                    warp.runtime_named_physical_access(
                        Some(&operation),
                        OperationKind::Load,
                        4,
                        4,
                        "input",
                        &buffer,
                        &WarpValue::splat(0_i64),
                        mask,
                        false,
                        &mut || {
                            state.events.lock().unwrap().push("numeric".to_string());
                            Err::<(), _>(crate::EngineError::message("numeric failed"))
                        },
                    )?;
                    Ok(())
                }
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("numeric failed"));
        assert_eq!(
            state.events.lock().unwrap().as_slice(),
            ["before:op:52:physical_access", "numeric"]
        );
    }

    #[test]
    fn run_kernel_launch_rejects_unfinished_physical_completion() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let id = PhysicalBarrierId::new(7, 0, 0);

        let error = execute_kernel_launch(
            topology,
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |_context, services| async move {
                services.mbarriers().init(id, 1)?;
                services
                    .mbarriers()
                    .enqueue_transaction_completion(id, 64)?;
                Ok(())
            },
        )
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("completion source mbarrier-payload failed"));
        assert!(error.to_string().contains("not quiescent at kernel exit"));
        assert!(error.to_string().contains("transactions=64/0"));
    }

    #[test]
    fn launch_selection_canonicalizes_unordered_duplicate_ids() {
        let topology = LaunchTopology::new(2, 2, 1).unwrap();
        let contexts = topology.warp_contexts().collect::<Vec<_>>();
        let selection = LaunchSelection::new(Some(vec![1, 0, 1]), Some(vec![3, 2, 3]));

        assert!(!selection.includes(contexts[0]));
        assert!(!selection.includes(contexts[1]));
        assert!(selection.includes(contexts[2]));
        assert!(selection.includes(contexts[3]));
        assert_eq!(selection.selected_cluster_ids(topology), vec![1]);
        selection.validate(topology).unwrap();
    }

    #[test]
    fn launch_selection_rejects_partial_clusters() {
        let topology = LaunchTopology::new(2, 2, 1).unwrap();
        let selection = LaunchSelection::new(None, Some(vec![0]));

        let error = selection.validate(topology).unwrap_err();
        assert!(error
            .to_string()
            .contains("CTA subset must be a union of complete clusters"));
    }

    #[test]
    fn launch_selection_allows_single_cta_clusters() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let selection = LaunchSelection::new(None, Some(vec![1]));

        selection.validate(topology).unwrap();
    }

    #[test]
    fn run_kernel_launch_rejects_an_empty_resolved_subset() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let error = execute_kernel_launch(
            topology,
            LaunchSelection::new(Some(vec![0]), Some(vec![1])),
            1,
            ExecutionPolicy::default(),
            |_context, _services| async { Ok(()) },
        )
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("resolves to no executable warps"));
    }
}
