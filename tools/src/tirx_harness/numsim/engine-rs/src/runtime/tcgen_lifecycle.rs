use std::sync::Arc;

use crate::{
    EngineError, OccurrenceKey, TcgenAllocation, TcgenLifecycleAction, TcgenLifecycleError,
    TcgenLifecycleErrorKind, TcgenLifecycleHub, TcgenLifecycleResult, TcgenLifecycleWait,
    WarpContext, WarpMask,
};

/// Fully resolved `tcgen05` allocation-permit lifecycle operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcgenLifecyclePlan {
    kernel_index: usize,
    static_op_id: u64,
    loop_iteration_path: Box<[i64]>,
    context: WarpContext,
    action: TcgenLifecycleAction,
    address: u32,
    columns: usize,
    cta_group: usize,
    participant_ctas: Box<[usize]>,
    participant_warps: Box<[usize]>,
    exclusive: bool,
    capacity: usize,
}

impl TcgenLifecyclePlan {
    #[allow(clippy::too_many_arguments)]
    fn new(
        kernel_index: usize,
        static_op_id: u64,
        loop_iteration_path: impl IntoIterator<Item = i64>,
        context: WarpContext,
        action: TcgenLifecycleAction,
        address: u32,
        columns: usize,
        cta_group: usize,
    ) -> Result<Self, EngineError> {
        let loop_iteration_path = loop_iteration_path
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let key = lifecycle_key(
            static_op_id,
            action,
            &loop_iteration_path,
            context,
            cta_group,
        );
        if context.active_mask() != WarpMask::FULL {
            return Err(lifecycle_error(
                TcgenLifecycleErrorKind::PartialWarpParticipation,
                key,
                format!(
                    "{} requires all {} lanes for warp {}, got mask 0x{:08x}",
                    action.label(),
                    crate::WARP_SIZE,
                    context.global_warp_id(),
                    context.active_mask().bits(),
                ),
            ));
        }
        let topology = context.topology();
        let (participant_ctas, participant_warps) = match cta_group {
            1 => (
                vec![context.global_cta_id()].into_boxed_slice(),
                vec![context.global_warp_id()].into_boxed_slice(),
            ),
            2 => {
                let local_cta = context.cta_id_in_cluster();
                let pair_base = local_cta & !1;
                let peer = pair_base + 1;
                if peer >= topology.ctas_per_cluster() {
                    return Err(lifecycle_error(
                        TcgenLifecycleErrorKind::MissingPeerCta,
                        key,
                        format!(
                            "tcgen05 cta_group=2 has no peer for CTA {local_cta} in cluster size {}",
                            topology.ctas_per_cluster(),
                        ),
                    ));
                }
                let cluster_base = context.cluster_id() * topology.ctas_per_cluster();
                let global_ctas = [cluster_base + pair_base, cluster_base + peer];
                let warp_in_cta = context.warp_id_in_cta();
                let global_warps =
                    global_ctas.map(|cta| cta * topology.warps_per_cta() + warp_in_cta);
                (
                    global_ctas.to_vec().into_boxed_slice(),
                    global_warps.to_vec().into_boxed_slice(),
                )
            }
            _ => {
                return Err(lifecycle_error(
                    TcgenLifecycleErrorKind::InvalidCtaGroup,
                    key,
                    format!("tcgen05 cta_group must be 1 or 2, got {cta_group}"),
                ))
            }
        };

        Ok(Self {
            kernel_index,
            static_op_id,
            loop_iteration_path,
            context,
            action,
            address,
            columns,
            cta_group,
            exclusive: false,
            capacity: crate::TMEM_COLUMN_CAPACITY,
            participant_ctas,
            participant_warps,
        })
    }

    pub const fn kernel_index(&self) -> usize {
        self.kernel_index
    }

    pub const fn static_op_id(&self) -> u64 {
        self.static_op_id
    }

    pub fn loop_iteration_path(&self) -> &[i64] {
        &self.loop_iteration_path
    }

    pub const fn context(&self) -> WarpContext {
        self.context
    }

    pub const fn action(&self) -> TcgenLifecycleAction {
        self.action
    }

    pub const fn address(&self) -> u32 {
        self.address
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub const fn cta_group(&self) -> usize {
        self.cta_group
    }

    pub fn participant_ctas(&self) -> &[usize] {
        &self.participant_ctas
    }

    pub fn participant_warps(&self) -> &[usize] {
        &self.participant_warps
    }

    pub fn with_column_capacity(
        mut self,
        capacity: usize,
        exclusive: bool,
    ) -> Result<Self, EngineError> {
        self.exclusive = exclusive;
        self.capacity = capacity;
        if self.action != TcgenLifecycleAction::Relinquish {
            crate::tcgen::validate_columns(self.columns, exclusive, self.capacity, || {
                self.occurrence_key()
            })?;
        }
        Ok(self)
    }

    pub const fn exclusive(&self) -> bool {
        self.exclusive
    }
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn occurrence_key(&self) -> OccurrenceKey {
        lifecycle_key(
            self.static_op_id,
            self.action,
            &self.loop_iteration_path,
            self.context,
            self.cta_group,
        )
    }

    pub fn register(
        &self,
        lifecycle: &Arc<TcgenLifecycleHub>,
    ) -> Result<TcgenLifecycleRegistration, EngineError> {
        let wait = match self.action {
            TcgenLifecycleAction::Allocate => lifecycle.allocate(
                self.static_op_id,
                self.loop_iteration_path.iter().copied(),
                self.context,
                self.columns,
                self.cta_group,
            )?,
            TcgenLifecycleAction::Deallocate => lifecycle.deallocate(
                self.static_op_id,
                self.loop_iteration_path.iter().copied(),
                self.context,
                self.address,
                self.columns,
                self.cta_group,
            )?,
            TcgenLifecycleAction::Relinquish => lifecycle.relinquish(
                self.static_op_id,
                self.loop_iteration_path.iter().copied(),
                self.context,
                self.cta_group,
            )?,
        };
        Ok(TcgenLifecycleRegistration {
            plan: self.clone(),
            wait,
        })
    }
}

pub fn plan_tcgen_allocate(
    kernel_index: usize,
    static_op_id: u64,
    loop_iteration_path: impl IntoIterator<Item = i64>,
    context: WarpContext,
    columns: usize,
    cta_group: usize,
) -> Result<TcgenLifecyclePlan, EngineError> {
    TcgenLifecyclePlan::new(
        kernel_index,
        static_op_id,
        loop_iteration_path,
        context,
        TcgenLifecycleAction::Allocate,
        0,
        columns,
        cta_group,
    )
}

pub fn plan_tcgen_deallocate(
    kernel_index: usize,
    static_op_id: u64,
    loop_iteration_path: impl IntoIterator<Item = i64>,
    context: WarpContext,
    address: u32,
    columns: usize,
    cta_group: usize,
) -> Result<TcgenLifecyclePlan, EngineError> {
    TcgenLifecyclePlan::new(
        kernel_index,
        static_op_id,
        loop_iteration_path,
        context,
        TcgenLifecycleAction::Deallocate,
        address,
        columns,
        cta_group,
    )
}

pub fn plan_tcgen_relinquish(
    kernel_index: usize,
    static_op_id: u64,
    loop_iteration_path: impl IntoIterator<Item = i64>,
    context: WarpContext,
    cta_group: usize,
) -> Result<TcgenLifecyclePlan, EngineError> {
    TcgenLifecyclePlan::new(
        kernel_index,
        static_op_id,
        loop_iteration_path,
        context,
        TcgenLifecycleAction::Relinquish,
        0,
        0,
        cta_group,
    )
}

pub struct TcgenLifecycleRegistration {
    plan: TcgenLifecyclePlan,
    wait: TcgenLifecycleWait,
}

impl TcgenLifecycleRegistration {
    pub const fn plan(&self) -> &TcgenLifecyclePlan {
        &self.plan
    }

    pub async fn resume(self) -> Result<TcgenLifecycleResumePlan, EngineError> {
        let result = self.wait.await?;
        Ok(TcgenLifecycleResumePlan {
            plan: self.plan,
            result,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcgenLifecycleResumePlan {
    plan: TcgenLifecyclePlan,
    result: Arc<TcgenLifecycleResult>,
}

impl TcgenLifecycleResumePlan {
    pub const fn plan(&self) -> &TcgenLifecyclePlan {
        &self.plan
    }

    pub fn result(&self) -> &TcgenLifecycleResult {
        self.result.as_ref()
    }

    pub fn allocation(&self) -> Option<TcgenAllocation> {
        self.result.allocation
    }
}

fn lifecycle_key(
    static_op_id: u64,
    action: TcgenLifecycleAction,
    loop_iteration_path: &[i64],
    context: WarpContext,
    cta_group: usize,
) -> OccurrenceKey {
    if cta_group == 2 {
        OccurrenceKey::for_cluster(
            static_op_id,
            action.label(),
            loop_iteration_path.iter().copied(),
            context,
        )
    } else {
        OccurrenceKey::for_cta(
            static_op_id,
            action.label(),
            loop_iteration_path.iter().copied(),
            context,
        )
    }
}

fn lifecycle_error(
    kind: TcgenLifecycleErrorKind,
    key: OccurrenceKey,
    message: impl Into<String>,
) -> EngineError {
    crate::SynchronizationError::TcgenLifecycle(TcgenLifecycleError::new(kind, key, message)).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LaunchTopology;

    fn context(topology: LaunchTopology, cta: usize) -> WarpContext {
        topology
            .warp_contexts()
            .find(|context| context.global_cta_id() == cta)
            .unwrap()
    }

    #[test]
    fn paired_plan_has_both_cta_resources() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let plan = plan_tcgen_allocate(4, 7, [3], context(topology, 1), 64, 2).unwrap();

        assert_eq!(plan.participant_ctas(), &[0, 1]);
        assert_eq!(plan.participant_warps(), &[0, 1]);
        assert_eq!(plan.loop_iteration_path(), &[3]);
    }

    #[test]
    fn partial_warp_is_a_typed_lifecycle_error() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = context(topology, 0).with_active_mask(WarpMask::from_bits(0xffff));
        let error = plan_tcgen_relinquish(4, 9, [], context, 1).unwrap_err();
        let crate::EngineErrorKind::Synchronization(error) = error.kind() else {
            panic!("expected synchronization error")
        };
        let crate::SynchronizationError::TcgenLifecycle(error) = error.as_ref() else {
            panic!("expected typed lifecycle error")
        };
        assert_eq!(
            error.kind(),
            TcgenLifecycleErrorKind::PartialWarpParticipation
        );
    }
}
