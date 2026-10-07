//! Engine-private mapped-element execution shared by typed async instructions.

use crate::engine_mode::EngineMode;
use crate::kernel_engine::WarpEngine;
use crate::runtime::{async_copy_element_at_lane, AsyncSourceFill, RuntimeBuffer};
use crate::{
    AsyncGroupDomain, DeferredGlobalReduction, DeferredGlobalWrite, EngineError,
    PhysicalAccessKind, PhysicalAccessSpace, WarpContext, WarpMask,
};

impl<M: EngineMode> WarpEngine<M> {
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    pub(crate) fn typed_async_group_issue_elements(
        &self,
        operation: Option<&crate::OperationContext>,
        context: &WarpContext,
        domain: AsyncGroupDomain,
        defer_global_writes: bool,
        mask: WarpMask,
        itemsize: usize,
        source: &RuntimeBuffer,
        destination: &RuntimeBuffer,
        multicast_cta_mask: Option<u64>,
        remote_cta_id: Option<usize>,
        source_fill: AsyncSourceFill,
        round_to_tf32: bool,
        reduction: Option<DeferredGlobalReduction>,
        elements: &mut dyn FnMut(
            &mut dyn FnMut(i64, bool, i64, bool, usize) -> Result<(), EngineError>,
        ) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        if source.uniform_physical_space() == Some(PhysicalAccessSpace::Shared)
            && destination.uniform_physical_space() == Some(PhysicalAccessSpace::Global)
        {
            self.validate_typed_tma_shared_alignment(
                operation,
                context,
                itemsize,
                source,
                PhysicalAccessKind::Read,
                |element| {
                    elements(&mut |source_index,
                                   source_in_bounds,
                                   _destination_index,
                                   _destination_in_bounds,
                                   lane| {
                        element(source_index, source_in_bounds, lane)
                    })
                },
            )?;
        }
        let elements = std::cell::RefCell::new(elements);
        self.async_group_issue_with_destination_kind(
            operation,
            domain,
            defer_global_writes,
            mask,
            if reduction.is_some() {
                PhysicalAccessKind::AtomicReadModifyWrite
            } else {
                PhysicalAccessKind::Write
            },
            |plan_element| {
                let mut replay = elements.borrow_mut();
                (**replay)(&mut |source_index,
                                 source_in_bounds,
                                 destination_index,
                                 destination_in_bounds,
                                 lane| {
                    plan_element(
                        itemsize,
                        source,
                        source_index,
                        source_in_bounds,
                        destination,
                        destination_index,
                        destination_in_bounds,
                        multicast_cta_mask,
                        remote_cta_id,
                        source_fill != AsyncSourceFill::None,
                        lane,
                    )
                })
            },
            || {
                let mut writes = Vec::new();
                let mut replay = elements.borrow_mut();
                match itemsize {
                    1 => replay_numeric::<1, M>(
                        self,
                        context,
                        source,
                        destination,
                        multicast_cta_mask,
                        remote_cta_id,
                        source_fill,
                        round_to_tf32,
                        reduction,
                        &mut writes,
                        &mut **replay,
                    )?,
                    2 => replay_numeric::<2, M>(
                        self,
                        context,
                        source,
                        destination,
                        multicast_cta_mask,
                        remote_cta_id,
                        source_fill,
                        round_to_tf32,
                        reduction,
                        &mut writes,
                        &mut **replay,
                    )?,
                    4 => replay_numeric::<4, M>(
                        self,
                        context,
                        source,
                        destination,
                        multicast_cta_mask,
                        remote_cta_id,
                        source_fill,
                        round_to_tf32,
                        reduction,
                        &mut writes,
                        &mut **replay,
                    )?,
                    8 => replay_numeric::<8, M>(
                        self,
                        context,
                        source,
                        destination,
                        multicast_cta_mask,
                        remote_cta_id,
                        source_fill,
                        round_to_tf32,
                        reduction,
                        &mut writes,
                        &mut **replay,
                    )?,
                    _ => {
                        return Err(EngineError::message(format!(
                            "typed async element width {itemsize} is unsupported"
                        )));
                    }
                }
                Ok(((), writes))
            },
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn replay_numeric<const ITEMSIZE: usize, M: EngineMode>(
    warp: &WarpEngine<M>,
    context: &WarpContext,
    source: &RuntimeBuffer,
    destination: &RuntimeBuffer,
    multicast_cta_mask: Option<u64>,
    remote_cta_id: Option<usize>,
    source_fill: AsyncSourceFill,
    round_to_tf32: bool,
    reduction: Option<DeferredGlobalReduction>,
    writes: &mut Vec<DeferredGlobalWrite>,
    elements: &mut dyn FnMut(
        &mut dyn FnMut(i64, bool, i64, bool, usize) -> Result<(), EngineError>,
    ) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    elements(&mut |source_index,
                   source_in_bounds,
                   destination_index,
                   destination_in_bounds,
                   lane| {
        async_copy_element_at_lane::<ITEMSIZE>(
            warp.kernel().physical(),
            context,
            source,
            source_index,
            source_in_bounds,
            destination,
            destination_index,
            destination_in_bounds,
            multicast_cta_mask,
            remote_cta_id,
            source_fill,
            round_to_tf32,
            reduction,
            writes,
            lane,
        )?;
        Ok(())
    })
}
