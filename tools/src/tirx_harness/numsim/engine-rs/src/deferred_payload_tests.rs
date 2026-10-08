use std::sync::Arc;

use crate::runtime::{
    execute_async_copy_element_at_lane, read_runtime_bytes, write_runtime_bytes, AsyncSourceFill,
    PhysicalMbarrierCompletionIssuePlan, RuntimeBuffer,
};
use crate::{
    f32_to_tf32, CtaId, DeferredGlobalReduction, DynamicOpId, LaunchTopology, OperationContext,
    OperationKind, PhysicalAccessBatch, PhysicalAccessDescriptor, PhysicalAccessKind,
    PhysicalAccessSpace, PhysicalAllocationId, PhysicalBarrierHub, PhysicalBarrierId,
    PhysicalByteSpan, PhysicalMemory, StaticOpId,
};

use super::{DeferredPayloadHub, MbarrierCompletionAction};

fn buffers() -> (
    PhysicalMemory,
    crate::WarpContext,
    RuntimeBuffer,
    RuntimeBuffer,
) {
    let topology = LaunchTopology::new(1, 1, 1).unwrap();
    let context = topology.warp_contexts().next().unwrap();
    let physical = PhysicalMemory::new(topology);
    let source_allocation = physical
        .global()
        .allocate_from_bytes(41_u32.to_le_bytes())
        .unwrap();
    let source = RuntimeBuffer::Global(physical.global().full_view(source_allocation).unwrap());
    let owner = CtaId::new(topology, 0, 0).unwrap();
    let destination_allocation = physical.shared().allocate_cta_zeroed(owner, 4).unwrap();
    let destination = RuntimeBuffer::Shared {
        allocations: Arc::new(vec![destination_allocation]),
        byte_offset: 0,
        byte_len: 4,
        backing_byte_len: 4,
        virtual_base: 0,
    };
    (physical, context, source, destination)
}

#[test]
fn payload_values_execute_at_issue() {
    let (physical, context, source, destination) = buffers();
    let delivered = execute_async_copy_element_at_lane(
        &physical,
        &context,
        4,
        &source,
        0,
        true,
        &destination,
        0,
        true,
        None,
        None,
        AsyncSourceFill::None,
        false,
        None,
        0,
    )
    .unwrap();
    assert_eq!(delivered, 4);
    assert_eq!(
        read_runtime_bytes(&physical, &context, &destination, 0, 0, 4).unwrap(),
        41_u32.to_le_bytes()
    );

    write_runtime_bytes(&physical, &context, &source, 0, 0, &99_u32.to_le_bytes()).unwrap();
    assert_eq!(
        read_runtime_bytes(&physical, &context, &destination, 0, 0, 4).unwrap(),
        41_u32.to_le_bytes()
    );
}

#[test]
fn enqueue_defers_completion_after_values_execute() {
    let (physical, context, source, destination) = buffers();
    let delivered = execute_async_copy_element_at_lane(
        &physical,
        &context,
        4,
        &source,
        0,
        true,
        &destination,
        0,
        true,
        None,
        None,
        AsyncSourceFill::None,
        false,
        None,
        0,
    )
    .unwrap();

    let operation = OperationContext::new(
        DynamicOpId::new(0, context.global_warp_id(), 0, StaticOpId::new(1), []),
        OperationKind::AsyncIssue,
        context.active_mask(),
    );
    let descriptor =
        PhysicalAccessDescriptor::new(PhysicalAccessKind::Write, PhysicalAccessSpace::Shared, 4)
            .unwrap();
    let access = PhysicalAccessBatch::resolve(operation.clone(), descriptor, |_| {
        Ok::<_, std::convert::Infallible>(vec![PhysicalByteSpan::new(
            PhysicalAllocationId::new(77),
            0,
            4,
        )
        .unwrap()])
    })
    .unwrap();
    let barrier_id = PhysicalBarrierId::new(1, 0, 0);
    let effect = crate::AsyncPayloadEffect::new(
        operation,
        [access.clone()],
        PhysicalMbarrierCompletionIssuePlan::single(barrier_id, 4),
    )
    .unwrap();
    let mbarriers = Arc::new(PhysicalBarrierHub::new());
    mbarriers.init(barrier_id, 1).unwrap();
    let hub = DeferredPayloadHub::new(Arc::clone(&mbarriers));

    let action_ids = hub.enqueue_payload(&effect, delivered).unwrap();

    assert_eq!(
        read_runtime_bytes(&physical, &context, &destination, 0, 0, 4).unwrap(),
        41_u32.to_le_bytes()
    );
    assert_eq!(mbarriers.pending_completion_count(), 1);
    assert!(hub.completed_tokens(barrier_id, 0).is_empty());
    let pending = hub.pending_completion_actions();
    let [MbarrierCompletionAction::DeferredPayload(action)] = pending.as_slice() else {
        panic!("payload completion must remain independently pending")
    };
    assert_eq!(action.completion_accesses(), effect.completion_accesses());
    assert_eq!(
        action.completion_accesses()[0]
            .descriptor()
            .memory_semantics()
            .proxy(),
        crate::MemoryProxy::Async,
    );

    hub.apply_completion_detailed(action_ids[0]).unwrap();
    assert_eq!(
        hub.completed_tokens(barrier_id, 0),
        [effect.token().clone()]
    );
}

#[test]
fn eager_payload_registration_defers_only_completion() {
    let topology = LaunchTopology::new(1, 1, 1).unwrap();
    let context = topology.warp_contexts().next().unwrap();
    let operation = OperationContext::new(
        DynamicOpId::new(0, context.global_warp_id(), 0, StaticOpId::new(2), []),
        OperationKind::AsyncIssue,
        context.active_mask(),
    );
    let barrier_id = PhysicalBarrierId::new(2, 0, 0);
    let completion_plan = PhysicalMbarrierCompletionIssuePlan::single(barrier_id, 4);
    let mbarriers = Arc::new(PhysicalBarrierHub::new());
    mbarriers.init(barrier_id, 1).unwrap();
    let hub = DeferredPayloadHub::new(Arc::clone(&mbarriers));

    hub.enqueue_eager_payload_completion(&operation, &completion_plan, 4)
        .unwrap();

    let pending = hub.pending_completion_actions();
    let [MbarrierCompletionAction::DeferredPayload(action)] = pending.as_slice() else {
        panic!("eager payload completion must remain independently pending")
    };
    assert!(action.completion_accesses().is_empty());
    let action_id = action.scheduler_action_id();
    let token = action.token().clone();

    hub.apply_completion_detailed(action_id).unwrap();
    assert_eq!(hub.completed_tokens(barrier_id, 0), [token]);
}

#[test]
fn payload_applies_oob_nan_fill_and_tf32_at_issue() {
    let (physical, context, source, destination) = buffers();
    execute_async_copy_element_at_lane(
        &physical,
        &context,
        4,
        &source,
        99,
        false,
        &destination,
        0,
        true,
        None,
        None,
        AsyncSourceFill::OobNan,
        true,
        None,
        0,
    )
    .unwrap();

    let filled = f32::from_le_bytes([0xf7, 0x7f, 0xf7, 0x7f]);
    assert_eq!(
        read_runtime_bytes(&physical, &context, &destination, 0, 0, 4).unwrap(),
        f32_to_tf32(filled).to_le_bytes()
    );
}

#[test]
fn multicast_payload_executes_for_each_target_cta() {
    let topology = LaunchTopology::new(1, 2, 1).unwrap();
    let context = topology.warp_contexts().next().unwrap();
    let physical = PhysicalMemory::new(topology);
    let source_allocation = physical
        .global()
        .allocate_from_bytes(73_u32.to_le_bytes())
        .unwrap();
    let source = RuntimeBuffer::Global(physical.global().full_view(source_allocation).unwrap());
    let allocations = (0..2)
        .map(|cta| {
            physical
                .shared()
                .allocate_cta_zeroed(CtaId::new(topology, 0, cta).unwrap(), 4)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let destination = RuntimeBuffer::Shared {
        allocations: Arc::new(allocations.clone()),
        byte_offset: 0,
        byte_len: 4,
        backing_byte_len: 4,
        virtual_base: 0,
    };

    execute_async_copy_element_at_lane(
        &physical,
        &context,
        4,
        &source,
        0,
        true,
        &destination,
        0,
        true,
        Some(0b11),
        None,
        AsyncSourceFill::None,
        false,
        None,
        0,
    )
    .unwrap();

    for (cta, allocation) in allocations.iter().enumerate() {
        let mut bytes = [0_u8; 4];
        physical
            .shared()
            .read_cta_bytes_into(
                CtaId::new(topology, 0, cta).unwrap(),
                allocation,
                0,
                4,
                0,
                &mut bytes,
            )
            .unwrap();
        assert_eq!(bytes, 73_u32.to_le_bytes());
    }
}

#[test]
fn g2s_payload_rejects_reduction_before_mutating_destination() {
    let (physical, context, source, destination) = buffers();
    let error = execute_async_copy_element_at_lane(
        &physical,
        &context,
        4,
        &source,
        0,
        true,
        &destination,
        0,
        true,
        None,
        None,
        AsyncSourceFill::None,
        false,
        Some(DeferredGlobalReduction::AddU32),
        0,
    )
    .unwrap_err();

    assert!(error
        .to_string()
        .contains("mbarrier-backed G2S payload does not support TMA reductions"));
    assert_eq!(
        read_runtime_bytes(&physical, &context, &destination, 0, 0, 4).unwrap(),
        [0; 4]
    );
}
