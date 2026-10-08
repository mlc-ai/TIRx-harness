//! TCGEN pipeline descriptors and portable thread-fence frontiers.
//!
//! The authoritative TCGEN clocks live in one cluster-local `RaceCheckState`.
//! Local execution-ordering operations carry immutable handles to those clocks
//! inside that same shard. Only a real global-memory boundary converts such a
//! handle into the token-keyed representation below, which can safely cross
//! shards through the launch-wide global-memory model.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::effect::{TcgenMmaPipelineClass, TcgenPipelineOperation};
use crate::AsyncTokenId;

/// Static facts needed to decide whether two asynchronous TCGEN operations
/// form one of PTX's five architected pipelines.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct TcgenPipelineDescriptor {
    operation: TcgenPipelineOperation,
    cta_group: u32,
    mma_pipeline_class: Option<TcgenMmaPipelineClass>,
}

impl TcgenPipelineDescriptor {
    pub(crate) fn new(
        operation: TcgenPipelineOperation,
        cta_group: u32,
        mma_pipeline_class: Option<TcgenMmaPipelineClass>,
    ) -> Self {
        Self {
            operation,
            cta_group,
            mma_pipeline_class,
        }
    }

    pub(crate) const fn operation(&self) -> TcgenPipelineOperation {
        self.operation
    }

    /// Whether PTX defines an execution pipeline from `self` into `next`.
    ///
    /// PTX ISA section 9.7.17.6.2 lists exactly five pairings. Thread identity
    /// is checked by the caller; this descriptor owns the remaining CTA-group,
    /// accumulator-dtype and instruction-shape requirements.
    pub(crate) fn pipelines_into(&self, next: &Self) -> bool {
        if self.cta_group != next.cta_group {
            return false;
        }
        match (self.operation, next.operation) {
            (TcgenPipelineOperation::Mma, TcgenPipelineOperation::Mma) => {
                self.mma_pipeline_class.is_some()
                    && self.mma_pipeline_class == next.mma_pipeline_class
            }
            (
                TcgenPipelineOperation::Copy | TcgenPipelineOperation::Copy4x256b,
                TcgenPipelineOperation::Mma,
            )
            | (TcgenPipelineOperation::Shift, TcgenPipelineOperation::Mma)
            | (TcgenPipelineOperation::Shift, TcgenPipelineOperation::Copy4x256b)
            | (TcgenPipelineOperation::Mma, TcgenPipelineOperation::Shift) => true,
            _ => false,
        }
    }
}

pub(crate) type TcgenComponentFrontier = Arc<BTreeMap<AsyncTokenId, u64>>;

/// Portable payloads indexed by lane only while crossing a launch-wide
/// global-memory release/read-from boundary.
pub(crate) type TcgenLaneFrontiers = BTreeMap<usize, TcgenFenceFrontier>;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct TcgenFenceSnapshot {
    pipeline: BTreeMap<TcgenPipelineDescriptor, TcgenComponentFrontier>,
    completed: TcgenComponentFrontier,
}

/// Immutable TCGEN frontier captured by `fence::before_thread_sync`.
///
/// Cloning or transporting an unchanged frontier clones only the outer `Arc`.
/// Contributor joins materialize a new snapshot only when they add a component.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TcgenFenceFrontier {
    snapshot: Arc<TcgenFenceSnapshot>,
}

impl TcgenFenceFrontier {
    pub(crate) fn pipeline(&self) -> &BTreeMap<TcgenPipelineDescriptor, TcgenComponentFrontier> {
        &self.snapshot.pipeline
    }

    pub(crate) fn completed(&self) -> &BTreeMap<AsyncTokenId, u64> {
        &self.snapshot.completed
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.snapshot.pipeline.is_empty() && self.snapshot.completed.is_empty()
    }

    pub(crate) fn covers(&self, other: &Self) -> bool {
        token_epochs_cover(&self.snapshot.completed, &other.snapshot.completed)
            && other
                .snapshot
                .pipeline
                .iter()
                .all(|(descriptor, incoming)| {
                    self.snapshot
                        .pipeline
                        .get(descriptor)
                        .is_some_and(|current| token_epochs_cover(current, incoming))
                })
    }

    pub(crate) fn merge_pipeline_frontier(
        &mut self,
        descriptor: TcgenPipelineDescriptor,
        components: &TcgenComponentFrontier,
    ) {
        let destination = Arc::make_mut(&mut self.snapshot)
            .pipeline
            .entry(descriptor)
            .or_default();
        merge_token_epochs(destination, components);
    }

    pub(crate) fn merge_completed_frontier(&mut self, components: &TcgenComponentFrontier) {
        merge_token_epochs(&mut Arc::make_mut(&mut self.snapshot).completed, components);
    }

    pub(crate) fn merge(&mut self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.snapshot, &other.snapshot) || other.is_empty() {
            return false;
        }
        if self.is_empty() {
            self.snapshot = Arc::clone(&other.snapshot);
            return true;
        }
        // Independently assembled lane frontiers can be equal without sharing
        // an allocation. Intern the equal snapshot at a rendezvous so subsequent
        // lane joins use the constant-time identity path.
        if self.snapshot == other.snapshot {
            self.snapshot = Arc::clone(&other.snapshot);
            return false;
        }
        if token_epochs_cover(&self.snapshot.completed, &other.snapshot.completed)
            && other
                .snapshot
                .pipeline
                .iter()
                .all(|(descriptor, incoming)| {
                    self.snapshot.pipeline.get(descriptor).map_or_else(
                        || incoming.values().all(|epoch| *epoch == 0),
                        |current| token_epochs_cover(current, incoming),
                    )
                })
        {
            return false;
        }
        let snapshot = Arc::make_mut(&mut self.snapshot);
        for (descriptor, incoming) in &other.snapshot.pipeline {
            merge_token_epochs(
                snapshot.pipeline.entry(descriptor.clone()).or_default(),
                incoming,
            );
        }
        merge_token_epochs(&mut snapshot.completed, &other.snapshot.completed);
        true
    }

    pub(crate) fn shares_snapshot_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.snapshot, &other.snapshot)
    }

    pub(crate) fn upgraded_to_completed(&self) -> Self {
        if self.snapshot.pipeline.is_empty() {
            return self.clone();
        }
        let mut upgraded = self.clone();
        let pipeline = upgraded
            .snapshot
            .pipeline
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let completed = &mut Arc::make_mut(&mut upgraded.snapshot).completed;
        for components in pipeline {
            merge_token_epochs(completed, &components);
        }
        upgraded
    }
}

fn merge_token_epochs(target: &mut TcgenComponentFrontier, incoming: &TcgenComponentFrontier) {
    if Arc::ptr_eq(target, incoming) {
        return;
    }
    if target.is_empty() {
        *target = Arc::clone(incoming);
        return;
    }
    if token_epochs_cover(target, incoming) {
        return;
    }
    let target = Arc::make_mut(target);
    for (token, epoch) in incoming.iter() {
        let current = target.entry(token.clone()).or_default();
        *current = (*current).max(*epoch);
    }
}

fn token_epochs_cover(current: &TcgenComponentFrontier, incoming: &TcgenComponentFrontier) -> bool {
    // Different outer snapshots often carry the same immutable component map.
    // Repeated global-memory acquisitions must not rescan that entire history.
    Arc::ptr_eq(current, incoming)
        || incoming
            .iter()
            .all(|(token, epoch)| *epoch <= current.get(token).copied().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DynamicOpId, StaticOpId};

    #[test]
    fn shared_components_preserve_copy_on_write_and_epoch_order() {
        let token = AsyncTokenId::new(DynamicOpId::new(0, 0, 1, StaticOpId::new(7), []), 0);
        let components = Arc::new(BTreeMap::from([(token.clone(), 3)]));
        let descriptor = TcgenPipelineDescriptor::new(TcgenPipelineOperation::Shift, 1, None);
        let mut first = TcgenFenceFrontier::default();
        first.merge_completed_frontier(&components);
        first.merge_pipeline_frontier(descriptor.clone(), &components);
        let independently_built = Arc::new((*components).clone());
        assert!(!Arc::ptr_eq(&components, &independently_built));
        let mut second = TcgenFenceFrontier::default();
        second.merge_completed_frontier(&independently_built);
        second.merge_pipeline_frontier(descriptor.clone(), &independently_built);
        assert!(!first.shares_snapshot_with(&second));
        assert!(first.covers(&second));
        assert!(!first.merge(&second));
        assert!(first.shares_snapshot_with(&second));

        let before = first.clone();
        let advanced = Arc::new(BTreeMap::from([(token.clone(), 4)]));
        second.merge_pipeline_frontier(descriptor.clone(), &advanced);
        assert!(!first.covers(&second));
        assert!(first.merge(&second));
        assert_eq!(before.pipeline()[&descriptor][&token], 3);
        assert_eq!(first.pipeline()[&descriptor][&token], 4);
        assert_eq!(first.completed()[&token], 3);
        assert!(first.covers(&second));
        assert!(!first.merge(&before));
    }
}
