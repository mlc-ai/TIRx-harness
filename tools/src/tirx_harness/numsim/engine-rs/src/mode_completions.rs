use std::sync::Arc;

use crate::completion::CompletionPublicationRegistry;
use crate::engine_mode::begin_global_memory_transaction;
use crate::{
    AsyncGroupHub, BlockedOperation, CompletionActionEffect, CompletionEffect, CompletionProgress,
    CompletionSource, DeferredPayloadHub, EngineError, EngineMode, MbarrierCompletionAction,
    MbarrierCompletionOutcome, PhysicalMemory, ProfileKind, ProfileTimer, SetmaxnregHub,
    SynchronizationError,
};

/// Ordinary-executor completion source that brackets each analysis-visible
/// completion with the selected mode's callbacks.
///
/// Numeric actions are filtered through `publications` so another executor
/// worker cannot complete an action between its numeric enqueue and the
/// issuer's successful `after_effect` commit.
pub(crate) struct ModeCompletionSource<M: EngineMode> {
    deferred_payloads: Arc<DeferredPayloadHub>,
    async_groups: Arc<AsyncGroupHub>,
    setmaxnreg: Arc<SetmaxnregHub>,
    publications: Arc<CompletionPublicationRegistry>,
    physical: PhysicalMemory,
    mode_state: Arc<M::LaunchState>,
    topology: crate::LaunchTopology,
    /// Upper bound on clusters whose deferred completions are applied
    /// concurrently by one pump round (1 = strictly sequential, the
    /// single-worker order).
    parallelism: usize,
}

impl<M: EngineMode> ModeCompletionSource<M> {
    pub(crate) fn new(
        deferred_payloads: Arc<DeferredPayloadHub>,
        async_groups: Arc<AsyncGroupHub>,
        setmaxnreg: Arc<SetmaxnregHub>,
        publications: Arc<CompletionPublicationRegistry>,
        physical: PhysicalMemory,
        mode_state: Arc<M::LaunchState>,
        topology: crate::LaunchTopology,
        parallelism: usize,
    ) -> Self {
        Self {
            deferred_payloads,
            async_groups,
            setmaxnreg,
            publications,
            physical,
            mode_state,
            topology,
            parallelism: parallelism.max(1),
        }
    }

    /// Apply one published deferred completion action, bracketed by the
    /// mode's callbacks and (when it touches global memory) its transaction.
    fn apply_deferred_action(
        &self,
        action: &MbarrierCompletionAction,
    ) -> Result<CompletionProgress, SynchronizationError> {
        let participates =
            M::USES_GLOBAL_MEMORY_TRANSACTION && deferred_action_touches_global(action);
        let spans = if participates {
            deferred_action_global_spans(action)
        } else {
            Vec::new()
        };
        let exclusive = match action {
            MbarrierCompletionAction::Physical(_) => true,
            MbarrierCompletionAction::DeferredPayload(action) => {
                crate::physical_access::global_transaction_exclusive(action.completion_accesses())
            }
        };
        let _profile =
            participates.then(|| ProfileTimer::new(ProfileKind::RaceGlobalTransactionCompletion));
        let _transaction = begin_global_memory_transaction::<M>(
            self.mode_state.as_ref(),
            participates,
            exclusive,
            &spans,
        )
        .map_err(mode_completion_error)?;
        {
            let _profile = ProfileTimer::new(ProfileKind::CompletionDeferredBefore);
            match action {
                MbarrierCompletionAction::Physical(action) => M::before_completion(
                    self.mode_state.as_ref(),
                    CompletionActionEffect::PhysicalMbarrier(action),
                ),
                MbarrierCompletionAction::DeferredPayload(action) => M::before_completion(
                    self.mode_state.as_ref(),
                    CompletionActionEffect::DeferredPayload(action),
                ),
            }
            .map_err(mode_completion_error)?;
        }
        let outcome = {
            let _profile = ProfileTimer::new(ProfileKind::CompletionDeferredApply);
            self.deferred_payloads
                .apply_completion_action_detailed_with_outcome(action, |outcome| {
                    match outcome {
                        MbarrierCompletionOutcome::Physical(outcome) => M::after_completion(
                            self.mode_state.as_ref(),
                            CompletionEffect::PhysicalMbarrier(outcome),
                        ),
                        MbarrierCompletionOutcome::DeferredPayload(outcome) => M::after_completion(
                            self.mode_state.as_ref(),
                            CompletionEffect::DeferredPayload(outcome),
                        ),
                    }?;
                    // Nonblocking mbarrier polling loops subscribe to the
                    // launch semantic-progress channel. A deferred
                    // completion can make their predicate true without
                    // mutating memory, so publish that state transition
                    // after the mode has committed its completion effect.
                    self.physical.record_semantic_progress();
                    Ok(())
                })
                .map_err(mode_completion_error)?
        };
        Ok(outcome.progress())
    }

    /// The scheduling domain (cluster) an action completes in.
    fn deferred_action_cluster(&self, action: &MbarrierCompletionAction) -> usize {
        let warp = match action {
            MbarrierCompletionAction::Physical(action) => {
                return action.barrier_id().target_global_cta_id()
                    / self.topology.ctas_per_cluster().max(1);
            }
            MbarrierCompletionAction::DeferredPayload(action) => {
                action.token().issue_operation().global_warp_id()
            }
        };
        self.topology.cluster_id_for_warp(warp).unwrap_or(0)
    }

    /// Apply the published deferred actions, sequentially per cluster and —
    /// when the launch runs several workers — concurrently across clusters.
    /// Different clusters' completions only meet in the mode state through
    /// its own per-cluster and per-allocation locks, so they are independent.
    fn pump_deferred_actions(
        &self,
        actions: Vec<MbarrierCompletionAction>,
    ) -> Result<CompletionProgress, SynchronizationError> {
        let mut progress = CompletionProgress::default();
        if self.parallelism <= 1 || actions.len() <= 1 {
            for action in &actions {
                if !self.deferred_action_is_published(action) {
                    continue;
                }
                add_progress(&mut progress, self.apply_deferred_action(action)?);
            }
            return Ok(progress);
        }
        let mut partitions: std::collections::BTreeMap<usize, Vec<MbarrierCompletionAction>> =
            std::collections::BTreeMap::new();
        for action in actions {
            if !self.deferred_action_is_published(&action) {
                continue;
            }
            partitions
                .entry(self.deferred_action_cluster(&action))
                .or_default()
                .push(action);
        }
        if partitions.len() <= 1 {
            for actions in partitions.into_values() {
                for action in &actions {
                    add_progress(&mut progress, self.apply_deferred_action(action)?);
                }
            }
            return Ok(progress);
        }
        // Spread the clusters over at most `parallelism` threads; each
        // thread applies its clusters' actions in discovery order. Threads
        // claim whole clusters from a shared counter, largest first, so a
        // heavy cluster no longer strands the threads whose round-robin
        // share finished early.
        let mut partitions = partitions.into_values().collect::<Vec<_>>();
        partitions.sort_by_key(|actions| std::cmp::Reverse(actions.len()));
        let thread_count = self.parallelism.min(partitions.len());
        let next_partition = std::sync::atomic::AtomicUsize::new(0);
        let results: Vec<Result<CompletionProgress, SynchronizationError>> =
            std::thread::scope(|scope| {
                let handles = (0..thread_count)
                    .map(|_| {
                        let partitions = &partitions;
                        let next_partition = &next_partition;
                        scope.spawn(move || {
                            let mut progress = CompletionProgress::default();
                            loop {
                                let index = next_partition
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                let Some(actions) = partitions.get(index) else {
                                    break;
                                };
                                for action in actions {
                                    add_progress(
                                        &mut progress,
                                        self.apply_deferred_action(action)?,
                                    );
                                }
                            }
                            Ok(progress)
                        })
                    })
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|handle| {
                        handle
                            .join()
                            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                    })
                    .collect()
            });
        for result in results {
            add_progress(&mut progress, result?);
        }
        Ok(progress)
    }

    fn deferred_action_is_published(&self, action: &MbarrierCompletionAction) -> bool {
        match action {
            MbarrierCompletionAction::Physical(action) => {
                self.publications.physical_is_published(action.id())
            }
            MbarrierCompletionAction::DeferredPayload(action) => action
                .action_ids()
                .all(|action_id| self.publications.physical_is_published(action_id)),
        }
    }

    fn pump_async_groups(&self) -> Result<CompletionProgress, SynchronizationError> {
        let actions = {
            let _profile = ProfileTimer::new(ProfileKind::CompletionAsyncDiscovery);
            self.async_groups.pending_completion_actions()
        };
        let group_error =
            |error: EngineError| SynchronizationError::CompletionSourceOperationFailed {
                source_name: self.async_groups.source_name(),
                details: error.to_string(),
            };
        let mut progress = CompletionProgress::default();
        for action in actions {
            if !self.publications.async_group_is_published(action.id()) {
                continue;
            }
            let accesses = || {
                action
                    .members()
                    .iter()
                    .flat_map(|member| match action.milestone() {
                        crate::AsyncGroupMilestone::SourceReadComplete => member.source_accesses(),
                        crate::AsyncGroupMilestone::FullComplete => member.destination_accesses(),
                    })
            };
            let participates = M::USES_GLOBAL_MEMORY_TRANSACTION
                && accesses().any(|batch| batch.descriptor().space().has_read_from_versions());
            let _profile = participates
                .then(|| ProfileTimer::new(ProfileKind::RaceGlobalTransactionCompletion));
            let spans = if participates {
                crate::physical_access::global_batch_spans(accesses())
            } else {
                Vec::new()
            };
            let _transaction = begin_global_memory_transaction::<M>(
                self.mode_state.as_ref(),
                participates,
                crate::physical_access::global_transaction_exclusive(accesses()),
                &spans,
            )
            .map_err(group_error)?;
            M::before_completion(
                self.mode_state.as_ref(),
                CompletionActionEffect::AsyncGroup(&action),
            )
            .map_err(group_error)?;
            let outcome = {
                let _profile = ProfileTimer::new(ProfileKind::CompletionAsyncApply);
                self.async_groups
                    .apply_completion_action_detailed_with_outcome(&action, |outcome| {
                        M::after_completion(
                            self.mode_state.as_ref(),
                            CompletionEffect::AsyncGroup(outcome),
                        )
                    })
                    .map_err(group_error)?
            };
            if outcome.action().milestone() == crate::AsyncGroupMilestone::FullComplete {
                self.publications.publish_physical(
                    outcome
                        .action()
                        .physical_actions()
                        .iter()
                        .map(|action| action.id()),
                );
            }
            add_progress(&mut progress, outcome.progress());
        }
        Ok(progress)
    }
}

impl<M: EngineMode> CompletionSource for ModeCompletionSource<M> {
    fn source_name(&self) -> &'static str {
        "mode-aware-completions"
    }

    fn pump(&self) -> Result<CompletionProgress, SynchronizationError> {
        let mut progress = self.setmaxnreg.pump_collective_only()?;

        let deferred_actions = {
            let _profile = ProfileTimer::new(ProfileKind::CompletionDeferredDiscovery);
            self.deferred_payloads.pending_completion_actions()
        };
        add_progress(&mut progress, self.pump_deferred_actions(deferred_actions)?);

        add_progress(&mut progress, self.pump_async_groups()?);

        // Register-pool grants compete for one mutable CTA-local capacity.
        // A snapshot may contain several individually enabled grants, but
        // applying one can disable the rest. Recompute on the next pump rather
        // than executing stale candidates from the old snapshot.
        if let Some(action) = self
            .setmaxnreg
            .pending_completion_actions()
            .into_iter()
            .next()
        {
            M::before_completion(
                self.mode_state.as_ref(),
                CompletionActionEffect::Setmaxnreg(&action),
            )
            .map_err(mode_completion_error)?;
            let outcome =
                self.setmaxnreg
                    .apply_completion_detailed_with_outcome(action.id(), |outcome| {
                        M::after_completion(
                            self.mode_state.as_ref(),
                            CompletionEffect::Setmaxnreg(outcome),
                        )
                        .map_err(mode_completion_error)
                    })?;
            add_progress(&mut progress, outcome.progress());
        }

        Ok(progress)
    }

    fn blocked_operations(&self) -> Vec<BlockedOperation> {
        let mut blocked = self.deferred_payloads.blocked_operations();
        blocked.extend(self.async_groups.blocked_operations());
        blocked.extend(self.setmaxnreg.blocked_operations());
        blocked.sort_by(BlockedOperation::diagnostic_cmp);
        blocked
    }

    fn validate_quiescent(&self) -> Result<(), SynchronizationError> {
        self.deferred_payloads.validate_quiescent()?;
        self.async_groups.validate_quiescent()?;
        self.setmaxnreg.validate_quiescent()
    }
}

fn add_progress(total: &mut CompletionProgress, progress: CompletionProgress) {
    total.completed_operations = total
        .completed_operations
        .saturating_add(progress.completed_operations);
    total.woken_warps = total.woken_warps.saturating_add(progress.woken_warps);
}

fn mode_completion_error(error: EngineError) -> SynchronizationError {
    SynchronizationError::CompletionSourceOperationFailed {
        source_name: "mode-aware-completions",
        details: error.to_string(),
    }
}

fn deferred_action_global_spans(action: &MbarrierCompletionAction) -> Vec<crate::PhysicalByteSpan> {
    match action {
        MbarrierCompletionAction::Physical(_) => Vec::new(),
        MbarrierCompletionAction::DeferredPayload(action) => {
            crate::physical_access::global_batch_spans(action.completion_accesses())
        }
    }
}

fn deferred_action_touches_global(action: &MbarrierCompletionAction) -> bool {
    match action {
        MbarrierCompletionAction::Physical(_) => false,
        MbarrierCompletionAction::DeferredPayload(action) => action
            .completion_accesses()
            .iter()
            .any(|batch| batch.descriptor().space().has_read_from_versions()),
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::sync::Arc;
    use std::task::{Context, Poll};

    use super::*;
    use crate::{
        AsyncGroupHub, LaunchTopology, NumSimMode, PhysicalBarrierHub, PhysicalBarrierId,
        SetmaxnregHub,
    };

    #[test]
    fn deferred_mbarrier_completion_wakes_semantic_progress_pollers() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        physical.enable_semantic_progress();
        let observed = physical.semantic_progress_snapshot();
        let mut watch = Box::pin(physical.watch_semantic_progress(observed, false));
        let waker = std::task::Waker::noop();
        let mut context = Context::from_waker(waker);
        assert!(watch.as_mut().poll(&mut context).is_pending());

        let mbarriers = Arc::new(PhysicalBarrierHub::new());
        let barrier = PhysicalBarrierId::new(1, 0, 0);
        mbarriers.init(barrier, 1).unwrap();
        mbarriers.arrive_expect_tx(barrier, 0, 1, 1).unwrap();
        let action_id = mbarriers
            .enqueue_transaction_completion(barrier, 1)
            .unwrap();
        let deferred_payloads = Arc::new(DeferredPayloadHub::new(mbarriers));
        let publications = Arc::new(CompletionPublicationRegistry::default());
        publications.publish_physical([action_id]);
        let source = ModeCompletionSource::<NumSimMode>::new(
            deferred_payloads,
            Arc::new(AsyncGroupHub::new(topology)),
            Arc::new(SetmaxnregHub::new(topology)),
            publications,
            physical,
            Arc::new(()),
            topology,
            1,
        );

        assert!(source.pump().unwrap().made_progress());
        assert_eq!(watch.as_mut().poll(&mut context), Poll::Ready(()));
    }
}
