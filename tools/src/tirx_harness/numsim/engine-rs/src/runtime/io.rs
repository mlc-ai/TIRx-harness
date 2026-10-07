use std::mem::size_of;

use crate::memory::{OwnerPrivateWriteSession, ReadSource, SharedReadSession};
use crate::physical_access::ProxyMemoryDomain;
use crate::{
    CtaId, DeferredGlobalReduction, DeferredGlobalWrite, EngineError, F32RoundingMode, F32x4,
    OperationContext, PhysicalAccessBatch, PhysicalAccessDescriptor, PhysicalAccessKind,
    PhysicalAccessSpace, PhysicalAddress, PhysicalAllocationId, PhysicalByteSpan, PhysicalMemory,
    RuntimeScalar, WARP_SIZE, WarpContext, WarpId, WarpMask, WarpValue, add_f32_ftz,
    bf16_bits_to_f32, f32_to_bf16_bits, f32_to_fp16_bits, fp16_bits_to_f32,
};

use super::{
    PhysicalPtr, PointerSpace, PtxStateSpace, RuntimeBuffer, ViewAccess,
    peel_runtime_buffer_wrappers, runtime_buffer_base_at, runtime_buffer_byte_len,
};

pub fn store_physical_ptr_u32(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    mask: WarpMask,
    value: u32,
) -> Result<(), EngineError> {
    let first_lane = mask.first_active().ok_or_else(|| {
        EngineError::message("cannot store through a physical pointer with an empty mask")
    })?;
    let byte_offset = pointer.lane_write_byte_offset(first_lane, 4)?;
    for lane in mask {
        let lane_offset = pointer.lane_write_byte_offset(lane, 4)?;
        if lane_offset != byte_offset {
            return Err(EngineError::message(format!(
                "physical pointer store is lane-varying: lane {first_lane} has byte offset {byte_offset}, lane {lane} has {lane_offset}"
            )));
        }
    }
    if pointer.pointee_itemsize() != 4 {
        return Err(EngineError::message(format!(
            "tcgen05.alloc destination itemsize must be 4 bytes, got {}",
            pointer.pointee_itemsize()
        )));
    }
    if !matches!(pointer.buffer(), RuntimeBuffer::Shared { .. }) {
        return Err(EngineError::message(
            "tcgen05.alloc destination must be local shared memory",
        ));
    }
    write_runtime_bytes(
        physical,
        context,
        pointer.buffer(),
        first_lane,
        byte_offset,
        &value.to_le_bytes(),
    )
}

#[inline]
fn fill_element_byte_offsets(
    buffer: &RuntimeBuffer,
    indices: &WarpValue<i64>,
    itemsize: usize,
    mask: WarpMask,
    offsets: &mut WarpValue<usize>,
) -> Result<(), EngineError> {
    let byte_len = runtime_buffer_byte_len(buffer);
    if let Some(first_lane) = mask.first_active() {
        let first_index = indices[first_lane];
        if mask.into_iter().all(|lane| indices[lane] == first_index) {
            let element_index = usize::try_from(first_index).map_err(|_| {
                EngineError::out_of_bounds(format!(
                    "negative buffer index {first_index} on lane {first_lane}"
                ))
            })?;
            let byte_offset =
                element_byte_offset_with_len(element_index, itemsize, first_lane, byte_len)?;
            offsets.masked_fill(mask, byte_offset);
            return Ok(());
        }
    }
    for lane in mask {
        let element_index = usize::try_from(indices[lane]).map_err(|_| {
            EngineError::out_of_bounds(format!(
                "negative buffer index {} on lane {lane}",
                indices[lane]
            ))
        })?;
        offsets[lane] = element_byte_offset_with_len(element_index, itemsize, lane, byte_len)?;
    }
    Ok(())
}

#[inline]
pub(crate) fn runtime_scalar_byte_offsets(
    buffer: &RuntimeBuffer,
    indices: &WarpValue<i64>,
    itemsize: usize,
    mask: WarpMask,
) -> Result<WarpValue<usize>, EngineError> {
    let mut byte_offsets = WarpValue::splat(0_usize);
    fill_element_byte_offsets(buffer, indices, itemsize, mask, &mut byte_offsets)?;
    Ok(byte_offsets)
}

pub const MAX_RUNTIME_SCALAR_BYTES: usize = 16;

pub fn element_byte_offset(
    buffer: &RuntimeBuffer,
    element_index: usize,
    itemsize: usize,
    lane: usize,
) -> Result<usize, EngineError> {
    element_byte_offset_with_len(
        element_index,
        itemsize,
        lane,
        runtime_buffer_byte_len(buffer),
    )
}

pub(crate) fn element_byte_offset_with_len(
    element_index: usize,
    itemsize: usize,
    lane: usize,
    byte_len: usize,
) -> Result<usize, EngineError> {
    let byte_offset = element_index.checked_mul(itemsize).ok_or_else(|| {
        EngineError::out_of_bounds(format!(
            "buffer element offset overflow at index {element_index} on lane {lane}"
        ))
    })?;
    let byte_end = byte_offset.checked_add(itemsize).ok_or_else(|| {
        EngineError::out_of_bounds(format!(
            "buffer element end overflow at index {element_index} on lane {lane}"
        ))
    })?;
    if byte_end > byte_len {
        return Err(EngineError::out_of_bounds(format!(
            "buffer element {element_index} ({itemsize} bytes) is outside {byte_len} bytes on lane {lane}"
        )));
    }
    Ok(byte_offset)
}

/// Exact physical byte identity resolved from one runtime buffer access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedRuntimePhysicalAccess {
    space: PhysicalAccessSpace,
    proxy_domain: ProxyMemoryDomain,
    span: PhysicalByteSpan,
}

impl ResolvedRuntimePhysicalAccess {
    pub const fn space(self) -> PhysicalAccessSpace {
        self.space
    }

    pub const fn span(self) -> PhysicalByteSpan {
        self.span
    }

    pub(crate) const fn proxy_memory_domain(self) -> ProxyMemoryDomain {
        self.proxy_domain
    }

    /// Construct a resolved access directly, for unit tests that exercise
    /// classification rules without standing up a launch.
    #[cfg(test)]
    pub(crate) const fn for_test(
        space: PhysicalAccessSpace,
        proxy_domain: ProxyMemoryDomain,
        span: PhysicalByteSpan,
    ) -> Self {
        Self {
            space,
            proxy_domain,
            span,
        }
    }
}

/// Resolve a checked runtime access without reading or mutating memory.
///
/// Allocation identity and absolute byte offsets match the underlying native
/// memory objects, so aliases through distinct logical buffers collapse to the
/// same racecheck key. Access-view permissions and per-space ownership/bounds
/// are validated before a span is returned.
pub fn resolve_runtime_physical_access(
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    lane: usize,
    byte_offset: usize,
    byte_len: usize,
    kind: PhysicalAccessKind,
) -> Result<ResolvedRuntimePhysicalAccess, EngineError> {
    if lane >= WARP_SIZE {
        return Err(EngineError::message(format!(
            "runtime physical access lane {lane} is outside warp size {WARP_SIZE}"
        )));
    }
    if byte_len == 0 {
        return Err(EngineError::message(
            "runtime physical access byte width must be nonzero",
        ));
    }
    let buffer = peel_runtime_buffer_wrappers(
        buffer,
        lane,
        ViewAccess::Kind {
            reads: kind.reads(),
            writes: kind.writes(),
        },
    )?;
    match buffer {
        RuntimeBuffer::AccessView { .. } | RuntimeBuffer::LaneSelected { .. } => {
            unreachable!("runtime buffer wrappers are peeled before the leaf walk")
        }
        RuntimeBuffer::Global(view) => {
            checked_runtime_view_end(view.byte_len(), byte_offset, byte_len, lane)?;
            let absolute = view.byte_offset().checked_add(byte_offset).ok_or_else(|| {
                EngineError::out_of_bounds("global physical byte offset overflow")
            })?;
            resolved_runtime_span(
                PhysicalAccessSpace::Global,
                ProxyMemoryDomain::Global,
                view.allocation(),
                absolute,
                byte_len,
            )
        }
        RuntimeBuffer::Shared {
            allocations,
            byte_offset: view_offset,
            byte_len: view_len,
            ..
        } => {
            let view_end = checked_runtime_view_end(*view_len, byte_offset, byte_len, lane)?;
            let allocation = allocations
                .get(context.global_cta_id())
                .ok_or_else(|| EngineError::message("shared-memory CTA allocation is missing"))?;
            let absolute = view_offset.checked_add(byte_offset).ok_or_else(|| {
                EngineError::out_of_bounds("shared physical byte offset overflow")
            })?;
            let absolute_end = view_offset
                .checked_add(view_end)
                .ok_or_else(|| EngineError::out_of_bounds("shared physical byte end overflow"))?;
            if absolute_end > allocation.byte_len() {
                return Err(EngineError::out_of_bounds(format!(
                    "shared physical byte range [{absolute}, {absolute_end}) exceeds allocation {} bytes on lane {lane}",
                    allocation.byte_len()
                )));
            }
            resolved_runtime_span(
                PhysicalAccessSpace::Shared,
                ProxyMemoryDomain::SharedCta,
                allocation.allocation(),
                absolute,
                byte_len,
            )
        }
        RuntimeBuffer::RemoteShared {
            allocations,
            byte_offsets,
            byte_len: view_len,
            target_cta_ids,
            ..
        } => {
            let view_end = checked_runtime_view_end(*view_len, byte_offset, byte_len, lane)?;
            let target_local = usize::try_from(target_cta_ids[lane]).map_err(|_| {
                EngineError::message(format!("negative remote shared CTA id on lane {lane}"))
            })?;
            let topology = context.topology();
            let target = CtaId::new(topology, context.cluster_id(), target_local)?;
            let target_global = target.global_cta_id(topology)?;
            let allocation = allocations.get(target_global).ok_or_else(|| {
                EngineError::message("remote shared-memory CTA allocation is missing")
            })?;
            let view_offset = usize::try_from(byte_offsets[lane]).map_err(|_| {
                EngineError::out_of_bounds(format!(
                    "negative remote shared byte offset on lane {lane}"
                ))
            })?;
            let absolute = view_offset.checked_add(byte_offset).ok_or_else(|| {
                EngineError::out_of_bounds("remote shared physical byte offset overflow")
            })?;
            let absolute_end = view_offset.checked_add(view_end).ok_or_else(|| {
                EngineError::out_of_bounds("remote shared physical byte end overflow")
            })?;
            if absolute_end > allocation.byte_len() {
                return Err(EngineError::out_of_bounds(format!(
                    "remote shared physical byte range [{absolute}, {absolute_end}) exceeds allocation {} bytes on lane {lane}",
                    allocation.byte_len()
                )));
            }
            resolved_runtime_span(
                PhysicalAccessSpace::Shared,
                ProxyMemoryDomain::SharedCluster,
                allocation.allocation(),
                absolute,
                byte_len,
            )
        }
        RuntimeBuffer::Local {
            allocations,
            byte_offset: view_offset,
            byte_len: view_len,
        }
        | RuntimeBuffer::Register {
            allocations,
            byte_offset: view_offset,
            byte_len: view_len,
        } => {
            let view_end = checked_runtime_view_end(*view_len, byte_offset, byte_len, lane)?;
            let allocation = allocations
                .get(context.global_warp_id())
                .ok_or_else(|| EngineError::message("warp-private allocation is missing"))?;
            let lane_base = lane
                .checked_mul(allocation.bytes_per_lane())
                .ok_or_else(|| {
                    EngineError::out_of_bounds("warp-private lane byte offset overflow")
                })?;
            let relative = view_offset.checked_add(byte_offset).ok_or_else(|| {
                EngineError::out_of_bounds("warp-private view byte offset overflow")
            })?;
            let relative_end = view_offset
                .checked_add(view_end)
                .ok_or_else(|| EngineError::out_of_bounds("warp-private view byte end overflow"))?;
            if relative_end > allocation.bytes_per_lane() {
                return Err(EngineError::out_of_bounds(format!(
                    "warp-private physical byte range [{relative}, {relative_end}) exceeds {} bytes per lane on lane {lane}",
                    allocation.bytes_per_lane()
                )));
            }
            let absolute = lane_base.checked_add(relative).ok_or_else(|| {
                EngineError::out_of_bounds("warp-private physical byte offset overflow")
            })?;
            let space = if matches!(buffer, RuntimeBuffer::Local { .. }) {
                PhysicalAccessSpace::Local
            } else {
                PhysicalAccessSpace::Register
            };
            resolved_runtime_span(
                space,
                ProxyMemoryDomain::Other,
                allocation.allocation(),
                absolute,
                byte_len,
            )
        }
        RuntimeBuffer::Tmem { .. } => Err(EngineError::message(
            "TMEM physical access resolution requires explicit TLane/TCol coordinates",
        )),
    }
}

/// Resolve one access against a specific CTA's shared-memory allocation in
/// the current cluster. High-level cta_group=2 TCGEN operations use the same
/// byte index for numerical reads and analysis footprints.
pub fn resolve_runtime_physical_access_at_cta(
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    target_cta_id_in_cluster: usize,
    lane: usize,
    byte_offset: usize,
    byte_len: usize,
    kind: PhysicalAccessKind,
) -> Result<ResolvedRuntimePhysicalAccess, EngineError> {
    match buffer {
        RuntimeBuffer::AccessView {
            buffer,
            readable_lanes,
            writable_lanes,
        } => {
            if kind.reads() && !readable_lanes.contains(lane) {
                return Err(EngineError::message(format!(
                    "read through a non-readable DeclBuffer view on lane {lane}"
                )));
            }
            if kind.writes() && !writable_lanes.contains(lane) {
                return Err(EngineError::message(format!(
                    "write through a non-writable DeclBuffer view on lane {lane}"
                )));
            }
            resolve_runtime_physical_access_at_cta(
                context,
                buffer,
                target_cta_id_in_cluster,
                lane,
                byte_offset,
                byte_len,
                kind,
            )
        }
        RuntimeBuffer::Shared {
            allocations,
            byte_offset: view_offset,
            byte_len: view_len,
            ..
        } => {
            let view_end = checked_runtime_view_end(*view_len, byte_offset, byte_len, lane)?;
            let target = CtaId::new(
                context.topology(),
                context.cluster_id(),
                target_cta_id_in_cluster,
            )?;
            let target_global = target.global_cta_id(context.topology())?;
            let allocation = allocations.get(target_global).ok_or_else(|| {
                EngineError::message("shared-memory target CTA allocation is missing")
            })?;
            let absolute = view_offset.checked_add(byte_offset).ok_or_else(|| {
                EngineError::out_of_bounds("shared physical byte offset overflow")
            })?;
            let absolute_end = view_offset
                .checked_add(view_end)
                .ok_or_else(|| EngineError::out_of_bounds("shared physical byte end overflow"))?;
            if absolute_end > allocation.byte_len() {
                return Err(EngineError::out_of_bounds(format!(
                    "shared physical byte range [{absolute}, {absolute_end}) exceeds allocation {} bytes on target CTA {target_cta_id_in_cluster} lane {lane}",
                    allocation.byte_len()
                )));
            }
            resolved_runtime_span(
                PhysicalAccessSpace::Shared,
                ProxyMemoryDomain::SharedCta,
                allocation.allocation(),
                absolute,
                byte_len,
            )
        }
        _ if target_cta_id_in_cluster == context.cta_id_in_cluster() => {
            resolve_runtime_physical_access(context, buffer, lane, byte_offset, byte_len, kind)
        }
        _ => Err(EngineError::message(
            "cross-CTA physical access resolution requires a shared-memory buffer",
        )),
    }
}

pub(crate) fn single_lane_physical_access_batch(
    operation: &OperationContext,
    lane: usize,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    spans: Vec<PhysicalByteSpan>,
) -> Result<PhysicalAccessBatch, EngineError> {
    single_lane_physical_access_batch_with(operation, lane, kind, space, spans, false)
}

/// One single-lane batch whose spans stay separate access units: the lane's
/// footprint keeps adjacent spans unfused, so every span is reported exactly
/// as if it had been resolved as its own batch.
pub(super) fn single_lane_physical_access_batch_unmerged(
    operation: &OperationContext,
    lane: usize,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    spans: Vec<PhysicalByteSpan>,
) -> Result<PhysicalAccessBatch, EngineError> {
    single_lane_physical_access_batch_with(operation, lane, kind, space, spans, true)
}

/// A single-lane transfer whose spans are runs of `unit_bytes` units.
pub(super) fn single_lane_transfer_run_batch(
    operation: &OperationContext,
    lane: usize,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    runs: Vec<PhysicalByteSpan>,
    unit_bytes: u32,
) -> Result<PhysicalAccessBatch, EngineError> {
    validate_single_lane_async_operation(operation, lane)?;
    let byte_width = runs.iter().try_fold(0_usize, |total, span| {
        total
            .checked_add(span.byte_len())
            .ok_or_else(|| EngineError::message("async payload byte width overflow"))
    })?;
    let descriptor = PhysicalAccessDescriptor::new(kind, space, byte_width)
        .map_err(|error| EngineError::message(error.to_string()))?;
    let resolve_lane = |provenance: &crate::LaneProvenance| {
        if provenance.lane() != lane {
            return Err(EngineError::message(format!(
                "async payload resolved unexpected lane {} instead of {lane}",
                provenance.lane()
            )));
        }
        Ok(runs.clone())
    };
    PhysicalAccessBatch::resolve_transfer_runs(
        operation.clone(),
        descriptor,
        resolve_lane,
        unit_bytes,
    )
    .map_err(|error| EngineError::message(error.to_string()))
}

fn single_lane_physical_access_batch_with(
    operation: &OperationContext,
    lane: usize,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    spans: Vec<PhysicalByteSpan>,
    unmerged: bool,
) -> Result<PhysicalAccessBatch, EngineError> {
    validate_single_lane_async_operation(operation, lane)?;
    let byte_width = spans.iter().try_fold(0_usize, |total, span| {
        total
            .checked_add(span.byte_len())
            .ok_or_else(|| EngineError::message("async payload byte width overflow"))
    })?;
    let descriptor = PhysicalAccessDescriptor::new(kind, space, byte_width)
        .map_err(|error| EngineError::message(error.to_string()))?;
    let resolve_lane = |provenance: &crate::LaneProvenance| {
        if provenance.lane() != lane {
            return Err(EngineError::message(format!(
                "async payload resolved unexpected lane {} instead of {lane}",
                provenance.lane()
            )));
        }
        Ok(spans.clone())
    };
    if unmerged {
        PhysicalAccessBatch::resolve_unmerged(operation.clone(), descriptor, resolve_lane)
    } else {
        PhysicalAccessBatch::resolve(operation.clone(), descriptor, resolve_lane)
    }
    .map_err(|error| EngineError::message(error.to_string()))
}

fn validate_single_lane_async_operation(
    operation: &OperationContext,
    lane: usize,
) -> Result<(), EngineError> {
    let expected_mask =
        WarpMask::from_lanes([lane]).map_err(|error| EngineError::message(error.to_string()))?;
    if operation.active_mask() != expected_mask {
        return Err(EngineError::message(format!(
            "async payload operation mask {:#010x} does not match issuing lane {lane}",
            operation.active_mask().bits()
        )));
    }
    Ok(())
}

pub fn resolve_shared_runtime_physical_access_to_cta(
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    lane: usize,
    target_cta_id_in_cluster: usize,
    byte_offset: usize,
    byte_len: usize,
    kind: PhysicalAccessKind,
) -> Result<ResolvedRuntimePhysicalAccess, EngineError> {
    if let RuntimeBuffer::AccessView {
        buffer,
        readable_lanes,
        writable_lanes,
    } = buffer
    {
        if kind.reads() && !readable_lanes.contains(lane) {
            return Err(EngineError::message(format!(
                "read through a non-readable DeclBuffer view on lane {lane}"
            )));
        }
        if kind.writes() && !writable_lanes.contains(lane) {
            return Err(EngineError::message(format!(
                "write through a non-writable DeclBuffer view on lane {lane}"
            )));
        }
        return resolve_shared_runtime_physical_access_to_cta(
            context,
            buffer,
            lane,
            target_cta_id_in_cluster,
            byte_offset,
            byte_len,
            kind,
        );
    }
    let RuntimeBuffer::Shared {
        allocations,
        byte_offset: view_offset,
        byte_len: view_len,
        ..
    } = buffer
    else {
        return Err(EngineError::message(
            "remote/multicast access target must be shared memory",
        ));
    };
    let view_end = checked_runtime_view_end(*view_len, byte_offset, byte_len, lane)?;
    let topology = context.topology();
    let target = CtaId::new(topology, context.cluster_id(), target_cta_id_in_cluster)?;
    let target_global = target.global_cta_id(topology)?;
    let allocation = allocations
        .get(target_global)
        .ok_or_else(|| EngineError::message("remote shared-memory CTA allocation is missing"))?;
    let absolute = view_offset
        .checked_add(byte_offset)
        .ok_or_else(|| EngineError::out_of_bounds("remote shared physical byte offset overflow"))?;
    let absolute_end = view_offset
        .checked_add(view_end)
        .ok_or_else(|| EngineError::out_of_bounds("remote shared physical byte end overflow"))?;
    if absolute_end > allocation.byte_len() {
        return Err(EngineError::out_of_bounds(format!(
            "remote shared physical byte range [{absolute}, {absolute_end}) exceeds allocation {} bytes on lane {lane}",
            allocation.byte_len()
        )));
    }
    resolved_runtime_span(
        PhysicalAccessSpace::Shared,
        ProxyMemoryDomain::SharedCluster,
        allocation.allocation(),
        absolute,
        byte_len,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn visit_async_copy_element_at_lane_accesses(
    operation: &OperationContext,
    context: &WarpContext,
    itemsize: usize,
    source: &RuntimeBuffer,
    source_index: i64,
    source_in_bounds: bool,
    destination: &RuntimeBuffer,
    destination_index: i64,
    destination_in_bounds: bool,
    multicast_cta_mask: Option<u64>,
    remote_cta_id: Option<usize>,
    allow_source_zero_fill: bool,
    lane: usize,
    mut visit: impl FnMut(PhysicalAccessKind, ResolvedRuntimePhysicalAccess) -> Result<(), EngineError>,
) -> Result<u64, EngineError> {
    if multicast_cta_mask.is_some() && remote_cta_id.is_some() {
        return Err(EngineError::message(
            "copy_async cannot combine multicast and one remote CTA target",
        ));
    }
    if let Some(mask) = multicast_cta_mask {
        validate_tma_multicast_mask(mask, context.topology().ctas_per_cluster())?;
    }
    if !context.active_mask().contains(lane) {
        return Err(EngineError::message(format!(
            "copy_async issuing lane {lane} is not active"
        )));
    }

    validate_single_lane_async_operation(operation, lane)?;
    if source_in_bounds {
        let source_index = usize::try_from(source_index).map_err(|_| {
            EngineError::out_of_bounds(format!(
                "copy_async source index {source_index} is negative on lane {lane}"
            ))
        })?;
        let source_offset = element_byte_offset(source, source_index, itemsize, lane)?;
        let source_access = resolve_runtime_physical_access(
            context,
            source,
            lane,
            source_offset,
            itemsize,
            PhysicalAccessKind::Read,
        )?;
        visit(PhysicalAccessKind::Read, source_access)?;
    } else if !allow_source_zero_fill {
        return Err(EngineError::out_of_bounds(format!(
            "copy_async source coordinate is outside its buffer on lane {lane}"
        )));
    }

    if !destination_in_bounds {
        if matches!(
            runtime_buffer_base_at(destination, lane),
            RuntimeBuffer::Global(_)
        ) {
            return Ok(0);
        }
        return Err(EngineError::out_of_bounds(format!(
            "copy_async destination coordinate is outside its buffer on lane {lane}"
        )));
    }
    let destination_index = usize::try_from(destination_index).map_err(|_| {
        EngineError::out_of_bounds(format!(
            "copy_async destination index {destination_index} is negative on lane {lane}"
        ))
    })?;
    let destination_offset = element_byte_offset(destination, destination_index, itemsize, lane)?;

    if let Some(mask) = multicast_cta_mask {
        for target in 0..context.topology().ctas_per_cluster() {
            if mask & (1_u64 << target) == 0 {
                continue;
            }
            let destination_access = resolve_shared_runtime_physical_access_to_cta(
                context,
                destination,
                lane,
                target,
                destination_offset,
                itemsize,
                PhysicalAccessKind::Write,
            )?;
            visit(PhysicalAccessKind::Write, destination_access)?;
        }
    } else if let Some(target) = remote_cta_id {
        let destination_access = resolve_shared_runtime_physical_access_to_cta(
            context,
            destination,
            lane,
            target,
            destination_offset,
            itemsize,
            PhysicalAccessKind::Write,
        )?;
        visit(PhysicalAccessKind::Write, destination_access)?;
    } else {
        let destination_access = resolve_runtime_physical_access(
            context,
            destination,
            lane,
            destination_offset,
            itemsize,
            PhysicalAccessKind::Write,
        )?;
        visit(PhysicalAccessKind::Write, destination_access)?;
    }
    Ok(itemsize as u64)
}

pub fn plan_async_copy_element_at_lane_accesses(
    operation: &OperationContext,
    context: &WarpContext,
    itemsize: usize,
    source: &RuntimeBuffer,
    source_index: i64,
    source_in_bounds: bool,
    destination: &RuntimeBuffer,
    destination_index: i64,
    destination_in_bounds: bool,
    multicast_cta_mask: Option<u64>,
    remote_cta_id: Option<usize>,
    allow_source_zero_fill: bool,
    lane: usize,
) -> Result<(Vec<PhysicalAccessBatch>, Vec<PhysicalAccessBatch>, u64), EngineError> {
    let mut source_accesses = Vec::new();
    let mut destination_accesses = Vec::new();
    let copied = visit_async_copy_element_at_lane_accesses(
        operation,
        context,
        itemsize,
        source,
        source_index,
        source_in_bounds,
        destination,
        destination_index,
        destination_in_bounds,
        multicast_cta_mask,
        remote_cta_id,
        allow_source_zero_fill,
        lane,
        |kind, access| {
            let batch = single_lane_physical_access_batch(
                operation,
                lane,
                kind,
                access.space(),
                vec![access.span()],
            )?
            .with_proxy_memory_domain(async_copy_proxy_domain(kind, access));
            match kind {
                PhysicalAccessKind::Read => source_accesses.push(batch),
                PhysicalAccessKind::Write => destination_accesses.push(batch),
                PhysicalAccessKind::AtomicReadModifyWrite => {
                    unreachable!("typed async copy planning never emits an atomic access")
                }
            }
            Ok(())
        },
    )?;
    Ok((source_accesses, destination_accesses, copied))
}

struct CompactAsyncCopyAccessGroup {
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    proxy_domain: ProxyMemoryDomain,
    spans: Vec<PhysicalByteSpan>,
    semantic_access_count: usize,
}

/// Engine-private accumulator for the direct checker path.
///
/// Generated code still supplies every exact element address through the
/// existing callback ABI. The engine stores raw spans and constructs one
/// union batch per access kind/space only after the complete region resolves.
#[derive(Default)]
pub(crate) struct CompactAsyncCopyAccessPlan {
    groups: Vec<CompactAsyncCopyAccessGroup>,
    delivered_bytes: u64,
}

impl CompactAsyncCopyAccessPlan {
    pub(crate) fn plan_full_local_destination(
        operation: &OperationContext,
        context: &WarpContext,
        itemsize: usize,
        source: &RuntimeBuffer,
        source_element_base: Option<i64>,
        source_semantic_access_count: usize,
        destination: &RuntimeBuffer,
        destination_semantic_access_count: usize,
        lane: usize,
    ) -> Result<(Vec<PhysicalAccessBatch>, u64), EngineError> {
        validate_single_lane_async_operation(operation, lane)?;
        if itemsize == 0 {
            return Err(EngineError::message(
                "copy_async full-buffer plan requires a nonzero item size",
            ));
        }
        if !matches!(source, RuntimeBuffer::Global(_)) {
            return Err(EngineError::message(
                "copy_async full-buffer plan requires a direct global source",
            ));
        }
        if !matches!(destination, RuntimeBuffer::Shared { .. }) {
            return Err(EngineError::message(
                "copy_async full-buffer plan requires a direct local shared destination",
            ));
        }

        let source_buffer_byte_len = runtime_buffer_byte_len(source);
        let source_capacity = source_buffer_byte_len / itemsize;
        if source_buffer_byte_len % itemsize != 0 || source_semantic_access_count > source_capacity
        {
            return Err(EngineError::message(format!(
                "copy_async full-buffer source count {source_semantic_access_count} exceeds \
                 {source_buffer_byte_len} bytes at item size {itemsize}",
            )));
        }

        let destination_byte_len = runtime_buffer_byte_len(destination);
        let delivered_bytes = destination_semantic_access_count
            .checked_mul(itemsize)
            .ok_or_else(|| EngineError::message("copy_async delivered-byte count overflow"))?;
        if delivered_bytes != destination_byte_len {
            return Err(EngineError::message(format!(
                "copy_async full-buffer destination count {destination_semantic_access_count} \
                 covers {delivered_bytes} of {destination_byte_len} bytes",
            )));
        }

        let mut batches = Vec::with_capacity(2);
        if source_semantic_access_count != 0 {
            let (source_byte_offset, source_byte_len) = match source_element_base {
                Some(element_base) => {
                    let element_base = usize::try_from(element_base).map_err(|_| {
                        EngineError::out_of_bounds(format!(
                            "copy_async source index {element_base} is negative on lane {lane}"
                        ))
                    })?;
                    (
                        element_byte_offset(source, element_base, itemsize, lane)?,
                        source_semantic_access_count
                            .checked_mul(itemsize)
                            .ok_or_else(|| {
                                EngineError::message("copy_async source byte count overflow")
                            })?,
                    )
                }
                None => (0, source_buffer_byte_len),
            };
            let source_access = resolve_runtime_physical_access(
                context,
                source,
                lane,
                source_byte_offset,
                source_byte_len,
                PhysicalAccessKind::Read,
            )?;
            batches.push(
                single_lane_physical_access_batch(
                    operation,
                    lane,
                    PhysicalAccessKind::Read,
                    source_access.space(),
                    vec![source_access.span()],
                )?
                .with_proxy_memory_domain(source_access.proxy_memory_domain())
                .with_semantic_access_count(source_semantic_access_count),
            );
        }
        let destination_access = resolve_runtime_physical_access(
            context,
            destination,
            lane,
            0,
            destination_byte_len,
            PhysicalAccessKind::Write,
        )?;
        batches.push(
            single_lane_physical_access_batch(
                operation,
                lane,
                PhysicalAccessKind::Write,
                destination_access.space(),
                vec![destination_access.span()],
            )?
            .with_proxy_memory_domain(ProxyMemoryDomain::SharedCluster)
            .with_semantic_access_count(destination_semantic_access_count),
        );
        Ok((batches, delivered_bytes as u64))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan_element(
        &mut self,
        operation: &OperationContext,
        context: &WarpContext,
        itemsize: usize,
        source: &RuntimeBuffer,
        source_index: i64,
        source_in_bounds: bool,
        destination: &RuntimeBuffer,
        destination_index: i64,
        destination_in_bounds: bool,
        multicast_cta_mask: Option<u64>,
        remote_cta_id: Option<usize>,
        allow_source_zero_fill: bool,
        lane: usize,
    ) -> Result<(), EngineError> {
        let copied = visit_async_copy_element_at_lane_accesses(
            operation,
            context,
            itemsize,
            source,
            source_index,
            source_in_bounds,
            destination,
            destination_index,
            destination_in_bounds,
            multicast_cta_mask,
            remote_cta_id,
            allow_source_zero_fill,
            lane,
            |kind, access| {
                let group = self.groups.iter_mut().find(|group| {
                    group.kind == kind
                        && group.space == access.space()
                        && group.proxy_domain == async_copy_proxy_domain(kind, access)
                });
                let group = match group {
                    Some(group) => group,
                    None => {
                        self.groups.push(CompactAsyncCopyAccessGroup {
                            kind,
                            space: access.space(),
                            proxy_domain: async_copy_proxy_domain(kind, access),
                            spans: Vec::new(),
                            semantic_access_count: 0,
                        });
                        self.groups
                            .last_mut()
                            .expect("a newly inserted async-copy group exists")
                    }
                };
                group.spans.push(access.span());
                group.semantic_access_count = group.semantic_access_count.saturating_add(1);
                Ok(())
            },
        )?;
        self.delivered_bytes = self
            .delivered_bytes
            .checked_add(copied)
            .ok_or_else(|| EngineError::message("copy_async delivered-byte count overflow"))?;
        Ok(())
    }

    pub(crate) fn finish(
        self,
        operation: &OperationContext,
        lane: usize,
    ) -> Result<(Vec<PhysicalAccessBatch>, u64), EngineError> {
        validate_single_lane_async_operation(operation, lane)?;
        let mut batches = Vec::with_capacity(self.groups.len());
        for group in self.groups {
            let mut spans = Some(group.spans);
            let batch = PhysicalAccessBatch::resolve_lane_unions(
                operation.clone(),
                group.kind,
                group.space,
                |provenance| {
                    if provenance.lane() != lane {
                        return Err(EngineError::message(format!(
                            "compact async payload resolved unexpected lane {} instead of {lane}",
                            provenance.lane()
                        )));
                    }
                    Ok(spans
                        .take()
                        .expect("one-lane compact footprint resolves once"))
                },
            )
            .map_err(|error| EngineError::message(error.to_string()))?
            .with_proxy_memory_domain(group.proxy_domain)
            .with_semantic_access_count(group.semantic_access_count);
            batches.push(batch);
        }
        Ok((batches, self.delivered_bytes))
    }
}

fn async_copy_proxy_domain(
    kind: PhysicalAccessKind,
    access: ResolvedRuntimePhysicalAccess,
) -> ProxyMemoryDomain {
    match (access.space(), kind) {
        (PhysicalAccessSpace::Shared, PhysicalAccessKind::Write) => {
            // Every shared destination is reported as `shared::cluster`. That
            // is deliberately coarser than the instruction's own state space --
            // `cp.async.bulk.tensor` has `.shared::cta` forms too, including
            // `.tile::gather4` -- and it is verdict-neutral rather than exact.
            //
            // A domain is compared exactly only as the *earlier* access of a
            // proxy bridge. On the later coordinate `proxy_alias_domains`
            // unions `SharedCta` with `SharedCluster`, and the only bridge that
            // can order an async access before a later generic one is the
            // implicit generic-async fence PTX guarantees at completion, which
            // is filled from this same batch's domain -- so fill and lookup
            // cannot disagree. `raw_bulk_copy_proxy_domain` keys on the PTX
            // state space instead, for representational exactness.
            ProxyMemoryDomain::SharedCluster
        }
        _ => access.proxy_memory_domain(),
    }
}

fn checked_runtime_view_end(
    view_len: usize,
    byte_offset: usize,
    byte_len: usize,
    lane: usize,
) -> Result<usize, EngineError> {
    let byte_end = byte_offset
        .checked_add(byte_len)
        .ok_or_else(|| EngineError::out_of_bounds("runtime physical byte range overflow"))?;
    if byte_end > view_len {
        return Err(EngineError::out_of_bounds(format!(
            "runtime physical byte range [{byte_offset}, {byte_end}) is outside {view_len} bytes on lane {lane}"
        )));
    }
    Ok(byte_end)
}

fn resolved_runtime_span(
    space: PhysicalAccessSpace,
    proxy_domain: ProxyMemoryDomain,
    allocation: crate::AllocationId,
    byte_offset: usize,
    byte_len: usize,
) -> Result<ResolvedRuntimePhysicalAccess, EngineError> {
    let span = PhysicalByteSpan::new(
        PhysicalAllocationId::from(allocation),
        byte_offset,
        byte_len,
    )
    .map_err(|error| EngineError::out_of_bounds(error.to_string()))?;
    Ok(ResolvedRuntimePhysicalAccess {
        space,
        proxy_domain,
        span,
    })
}

pub fn read_runtime_bytes(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    lane: usize,
    byte_offset: usize,
    byte_len: usize,
) -> Result<Vec<u8>, EngineError> {
    let mut bytes = vec![0; byte_len];
    read_runtime_bytes_into(physical, context, buffer, lane, byte_offset, &mut bytes)?;
    Ok(bytes)
}

pub fn read_runtime_bytes_into(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    lane: usize,
    byte_offset: usize,
    output: &mut [u8],
) -> Result<(), EngineError> {
    read_runtime_bytes_into_impl(physical, context, buffer, lane, byte_offset, output)
}

fn read_runtime_bytes_unordered(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    lane: usize,
    byte_offset: usize,
    byte_len: usize,
) -> Result<Vec<u8>, EngineError> {
    let mut bytes = vec![0; byte_len];
    read_runtime_bytes_into_impl(physical, context, buffer, lane, byte_offset, &mut bytes)?;
    Ok(bytes)
}

fn read_runtime_bytes_into_impl(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    lane: usize,
    byte_offset: usize,
    output: &mut [u8],
) -> Result<(), EngineError> {
    let buffer = peel_runtime_buffer_wrappers(buffer, lane, ViewAccess::Read)?;
    match buffer {
        RuntimeBuffer::AccessView { .. } | RuntimeBuffer::LaneSelected { .. } => {
            unreachable!("runtime buffer wrappers are peeled before the leaf walk")
        }
        RuntimeBuffer::Global(view) => {
            physical
                .global()
                .read_bytes_into(view, byte_offset, output)?;
            Ok(())
        }
        RuntimeBuffer::Shared {
            allocations,
            byte_offset: view_offset,
            byte_len: view_len,
            ..
        } => {
            let allocation = allocations
                .get(context.global_cta_id())
                .ok_or_else(|| EngineError::message("shared-memory CTA allocation is missing"))?;
            let owner = CtaId::from_context(*context);
            physical.shared().read_cta_bytes_into(
                owner,
                allocation,
                *view_offset,
                *view_len,
                byte_offset,
                output,
            )?;
            Ok(())
        }
        RuntimeBuffer::RemoteShared {
            allocations,
            byte_offsets,
            byte_len: view_len,
            target_cta_ids,
            ..
        } => {
            let target_local = usize::try_from(target_cta_ids[lane]).map_err(|_| {
                EngineError::message(format!("negative remote shared CTA id on lane {lane}"))
            })?;
            let topology = context.topology();
            let requester = CtaId::from_context(*context);
            let target = CtaId::new(topology, context.cluster_id(), target_local)?;
            let target_global = target.global_cta_id(topology)?;
            let allocation = allocations.get(target_global).ok_or_else(|| {
                EngineError::message("remote shared-memory CTA allocation is missing")
            })?;
            let view_offset = usize::try_from(byte_offsets[lane]).map_err(|_| {
                EngineError::out_of_bounds(format!(
                    "negative remote shared byte offset on lane {lane}"
                ))
            })?;
            physical.shared().read_remote_cta_bytes_into(
                requester,
                target,
                allocation,
                view_offset,
                *view_len,
                byte_offset,
                output,
            )?;
            Ok(())
        }
        RuntimeBuffer::Local {
            allocations,
            byte_offset: view_offset,
            byte_len: view_len,
        }
        | RuntimeBuffer::Register {
            allocations,
            byte_offset: view_offset,
            byte_len: view_len,
        } => {
            let allocation = allocations
                .get(context.global_warp_id())
                .ok_or_else(|| EngineError::message("warp-private allocation is missing"))?;
            let owner = WarpId::from_context(*context);
            let memory = match buffer {
                RuntimeBuffer::Local { .. } => physical.local(),
                RuntimeBuffer::Register { .. } => physical.registers(),
                _ => unreachable!(),
            };
            memory.read_lane_bytes_into(
                owner,
                allocation,
                lane,
                *view_offset,
                *view_len,
                byte_offset,
                output,
            )?;
            Ok(())
        }
        RuntimeBuffer::Tmem { .. } => Err(EngineError::message(
            "TMEM requires explicit TLane/TCol coordinates",
        )),
    }
}

pub fn read_runtime_bytes_zero_filled_into(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    lane: usize,
    byte_offset: usize,
    target: &mut [u8],
) -> Result<(), EngineError> {
    read_runtime_bytes_zero_filled_into_impl(physical, context, buffer, lane, byte_offset, target)
}

fn read_runtime_bytes_zero_filled_into_impl(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    lane: usize,
    byte_offset: usize,
    target: &mut [u8],
) -> Result<(), EngineError> {
    if let RuntimeBuffer::AccessView {
        buffer,
        readable_lanes,
        ..
    } = buffer
    {
        if !readable_lanes.contains(lane) {
            return Err(EngineError::message(format!(
                "read through a non-readable DeclBuffer view on lane {lane}"
            )));
        }
        return read_runtime_bytes_zero_filled_into_impl(
            physical,
            context,
            buffer,
            lane,
            byte_offset,
            target,
        );
    }
    let RuntimeBuffer::Shared {
        allocations,
        byte_offset: view_offset,
        byte_len: view_len,
        ..
    } = buffer
    else {
        return Err(EngineError::message(
            "zero-filled padding reads require local shared memory",
        ));
    };
    let allocation = allocations
        .get(context.global_cta_id())
        .ok_or_else(|| EngineError::message("shared-memory CTA allocation is missing"))?;
    let owner = CtaId::from_context(*context);
    physical.shared().read_cta_bytes_zero_filled_into(
        owner,
        allocation,
        *view_offset,
        *view_len,
        byte_offset,
        target,
    )?;
    Ok(())
}

pub fn write_runtime_bytes(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    lane: usize,
    byte_offset: usize,
    bytes: &[u8],
) -> Result<(), EngineError> {
    write_runtime_bytes_impl(physical, context, buffer, lane, byte_offset, bytes)
}

fn write_runtime_bytes_impl(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    lane: usize,
    byte_offset: usize,
    bytes: &[u8],
) -> Result<(), EngineError> {
    let buffer = peel_runtime_buffer_wrappers(buffer, lane, ViewAccess::Write)?;
    match buffer {
        RuntimeBuffer::AccessView { .. } | RuntimeBuffer::LaneSelected { .. } => {
            unreachable!("runtime buffer wrappers are peeled before the leaf walk")
        }
        RuntimeBuffer::Global(view) => {
            physical.global().write_bytes(view, byte_offset, bytes)?;
        }
        RuntimeBuffer::Shared {
            allocations,
            byte_offset: view_offset,
            byte_len: view_len,
            ..
        } => {
            let allocation = allocations
                .get(context.global_cta_id())
                .ok_or_else(|| EngineError::message("shared-memory CTA allocation is missing"))?;
            let owner = CtaId::from_context(*context);
            physical.shared().write_cta_bytes(
                owner,
                allocation,
                *view_offset,
                *view_len,
                byte_offset,
                bytes,
            )?;
        }
        RuntimeBuffer::RemoteShared {
            allocations,
            byte_offsets,
            byte_len: view_len,
            target_cta_ids,
            ..
        } => {
            let target_local = usize::try_from(target_cta_ids[lane]).map_err(|_| {
                EngineError::message(format!("negative remote shared CTA id on lane {lane}"))
            })?;
            let topology = context.topology();
            let requester = CtaId::from_context(*context);
            let target = CtaId::new(topology, context.cluster_id(), target_local)?;
            let target_global = target.global_cta_id(topology)?;
            let allocation = allocations.get(target_global).ok_or_else(|| {
                EngineError::message("remote shared-memory CTA allocation is missing")
            })?;
            let view_offset = usize::try_from(byte_offsets[lane]).map_err(|_| {
                EngineError::out_of_bounds(format!(
                    "negative remote shared byte offset on lane {lane}"
                ))
            })?;
            physical.shared().write_remote_cta_bytes(
                requester,
                target,
                allocation,
                view_offset,
                *view_len,
                byte_offset,
                bytes,
            )?;
        }
        RuntimeBuffer::Local {
            allocations,
            byte_offset: view_offset,
            byte_len: view_len,
        }
        | RuntimeBuffer::Register {
            allocations,
            byte_offset: view_offset,
            byte_len: view_len,
        } => {
            let allocation = allocations
                .get(context.global_warp_id())
                .ok_or_else(|| EngineError::message("warp-private allocation is missing"))?;
            let owner = WarpId::from_context(*context);
            let memory = match buffer {
                RuntimeBuffer::Local { .. } => physical.local(),
                RuntimeBuffer::Register { .. } => physical.registers(),
                _ => unreachable!(),
            };
            memory.write_lane_bytes(
                owner,
                allocation,
                lane,
                *view_offset,
                *view_len,
                byte_offset,
                bytes,
            )?;
        }
        RuntimeBuffer::Tmem { .. } => {
            return Err(EngineError::message(
                "TMEM requires explicit TLane/TCol coordinates",
            ));
        }
    }
    Ok(())
}

pub fn defer_global_runtime_bytes(
    physical: &PhysicalMemory,
    buffer: &RuntimeBuffer,
    lane: usize,
    byte_offset: usize,
    bytes: Vec<u8>,
) -> Result<DeferredGlobalWrite, EngineError> {
    let buffer = peel_runtime_buffer_wrappers(buffer, lane, ViewAccess::Write)?;
    match buffer {
        RuntimeBuffer::Global(view) => {
            Ok(physical
                .global()
                .defer_write_bytes(view, byte_offset, bytes)?)
        }
        _ => Err(EngineError::message(
            "deferred async-group writes require a global-memory destination",
        )),
    }
}

fn defer_or_extend_global_runtime_bytes(
    physical: &PhysicalMemory,
    buffer: &RuntimeBuffer,
    lane: usize,
    byte_offset: usize,
    bytes: &[u8],
    writes: &mut Vec<DeferredGlobalWrite>,
) -> Result<(), EngineError> {
    let buffer = peel_runtime_buffer_wrappers(buffer, lane, ViewAccess::Write)?;
    match buffer {
        RuntimeBuffer::Global(view) => {
            Ok(physical
                .global()
                .defer_or_extend_write_bytes(writes, view, byte_offset, bytes)?)
        }
        _ => Err(EngineError::message(
            "deferred async-group writes require a global-memory destination",
        )),
    }
}

pub fn defer_global_runtime_masked_bytes(
    physical: &PhysicalMemory,
    buffer: &RuntimeBuffer,
    lane: usize,
    byte_offset: usize,
    bytes: Vec<u8>,
    masks: Vec<u8>,
) -> Result<DeferredGlobalWrite, EngineError> {
    let buffer = peel_runtime_buffer_wrappers(buffer, lane, ViewAccess::Write)?;
    match buffer {
        RuntimeBuffer::Global(view) => {
            Ok(physical
                .global()
                .defer_masked_write_bytes(view, byte_offset, bytes, masks)?)
        }
        _ => Err(EngineError::message(
            "deferred async-group writes require a global-memory destination",
        )),
    }
}

pub fn defer_global_runtime_reduction_bytes(
    physical: &PhysicalMemory,
    buffer: &RuntimeBuffer,
    lane: usize,
    byte_offset: usize,
    bytes: Vec<u8>,
    reduction: DeferredGlobalReduction,
) -> Result<DeferredGlobalWrite, EngineError> {
    let buffer = peel_runtime_buffer_wrappers(buffer, lane, ViewAccess::Write)?;
    match buffer {
        RuntimeBuffer::Global(view) => Ok(physical.global().defer_reduction_write_bytes(
            view,
            byte_offset,
            bytes,
            reduction,
        )?),
        _ => Err(EngineError::message(
            "deferred TMA reductions require a global-memory destination",
        )),
    }
}

fn write_or_defer_async_destination(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &RuntimeBuffer,
    lane: usize,
    byte_offset: usize,
    bytes: &[u8],
    reduction: Option<DeferredGlobalReduction>,
    deferred_global_writes: &mut Vec<DeferredGlobalWrite>,
) -> Result<(), EngineError> {
    if matches!(
        runtime_buffer_base_at(destination, lane),
        RuntimeBuffer::Global(_)
    ) {
        let write = match reduction {
            Some(reduction) => defer_global_runtime_reduction_bytes(
                physical,
                destination,
                lane,
                byte_offset,
                bytes.to_vec(),
                reduction,
            )?,
            None => {
                return defer_or_extend_global_runtime_bytes(
                    physical,
                    destination,
                    lane,
                    byte_offset,
                    bytes,
                    deferred_global_writes,
                );
            }
        };
        deferred_global_writes.push(write);
        return Ok(());
    }
    if reduction.is_some() {
        return Err(EngineError::message(
            "TMA reductions require a global-memory destination",
        ));
    }
    write_runtime_bytes(physical, context, destination, lane, byte_offset, bytes)
}

pub fn invalidate_runtime_bytes(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    lane: usize,
    byte_offset: usize,
    byte_len: usize,
) -> Result<(), EngineError> {
    let buffer = peel_runtime_buffer_wrappers(buffer, lane, ViewAccess::Invalidate)?;
    match buffer {
        RuntimeBuffer::AccessView { .. } | RuntimeBuffer::LaneSelected { .. } => {
            unreachable!("runtime buffer wrappers are peeled before the leaf walk")
        }
        RuntimeBuffer::Global(view) => {
            physical.global().invalidate(view, byte_offset, byte_len)?;
        }
        RuntimeBuffer::Shared {
            allocations,
            byte_offset: view_offset,
            byte_len: view_len,
            ..
        } => {
            let allocation = allocations
                .get(context.global_cta_id())
                .ok_or_else(|| EngineError::message("shared-memory CTA allocation is missing"))?;
            let owner = CtaId::from_context(*context);
            let view = physical
                .shared()
                .cta_view(owner, allocation, *view_offset, *view_len)?;
            physical.shared().invalidate(&view, byte_offset, byte_len)?;
        }
        RuntimeBuffer::RemoteShared {
            allocations,
            byte_offsets,
            byte_len: view_len,
            target_cta_ids,
            ..
        } => {
            let target_local = usize::try_from(target_cta_ids[lane]).map_err(|_| {
                EngineError::message(format!("negative remote shared CTA id on lane {lane}"))
            })?;
            let topology = context.topology();
            let requester = CtaId::from_context(*context);
            let target = CtaId::new(topology, context.cluster_id(), target_local)?;
            let target_global = target.global_cta_id(topology)?;
            let allocation = allocations.get(target_global).ok_or_else(|| {
                EngineError::message("remote shared-memory CTA allocation is missing")
            })?;
            let view_offset = usize::try_from(byte_offsets[lane]).map_err(|_| {
                EngineError::out_of_bounds(format!(
                    "negative remote shared byte offset on lane {lane}"
                ))
            })?;
            let view = physical.shared().remote_cta_view(
                requester,
                target,
                allocation,
                view_offset,
                *view_len,
            )?;
            physical.shared().invalidate(&view, byte_offset, byte_len)?;
        }
        RuntimeBuffer::Local {
            allocations,
            byte_offset: view_offset,
            byte_len: view_len,
        }
        | RuntimeBuffer::Register {
            allocations,
            byte_offset: view_offset,
            byte_len: view_len,
        } => {
            let allocation = allocations
                .get(context.global_warp_id())
                .ok_or_else(|| EngineError::message("warp-private allocation is missing"))?;
            let owner = WarpId::from_context(*context);
            let memory = match buffer {
                RuntimeBuffer::Local { .. } => physical.local(),
                RuntimeBuffer::Register { .. } => physical.registers(),
                _ => unreachable!(),
            };
            let view = memory.lane_view(owner, allocation, lane, *view_offset, *view_len)?;
            memory.invalidate(&view, byte_offset, byte_len)?;
        }
        RuntimeBuffer::Tmem { .. } => {
            return Err(EngineError::message(
                "TMEM requires explicit TLane/TCol coordinates",
            ));
        }
    }
    Ok(())
}

pub fn write_runtime_bytes_with_validity(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    lane: usize,
    byte_offset: usize,
    bytes: &[u8],
    validity: &[bool],
) -> Result<(), EngineError> {
    if bytes.len() != validity.len() {
        return Err(EngineError::message(format!(
            "runtime write has {} data bytes but {} validity bits",
            bytes.len(),
            validity.len()
        )));
    }
    let mut run_start = 0;
    while run_start < bytes.len() {
        let valid = validity[run_start];
        let mut run_end = run_start + 1;
        while run_end < bytes.len() && validity[run_end] == valid {
            run_end += 1;
        }
        let run_offset = byte_offset
            .checked_add(run_start)
            .ok_or_else(|| EngineError::out_of_bounds("runtime write byte offset overflow"))?;
        if valid {
            write_runtime_bytes(
                physical,
                context,
                buffer,
                lane,
                run_offset,
                &bytes[run_start..run_end],
            )?;
        } else {
            invalidate_runtime_bytes(
                physical,
                context,
                buffer,
                lane,
                run_offset,
                run_end - run_start,
            )?;
        }
        run_start = run_end;
    }
    Ok(())
}

pub fn write_shared_runtime_bytes_to_cta(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    lane: usize,
    target_cta_id_in_cluster: usize,
    byte_offset: usize,
    bytes: &[u8],
) -> Result<(), EngineError> {
    if let RuntimeBuffer::AccessView {
        buffer,
        writable_lanes,
        ..
    } = buffer
    {
        if !writable_lanes.contains(lane) {
            return Err(EngineError::message(format!(
                "write through a non-writable DeclBuffer view on lane {lane}"
            )));
        }
        return write_shared_runtime_bytes_to_cta(
            physical,
            context,
            buffer,
            lane,
            target_cta_id_in_cluster,
            byte_offset,
            bytes,
        );
    }
    let RuntimeBuffer::Shared {
        allocations,
        byte_offset: view_offset,
        byte_len: view_len,
        ..
    } = buffer
    else {
        return Err(EngineError::message(
            "remote/multicast copy destination must be shared memory",
        ));
    };
    let topology = context.topology();
    let requester = CtaId::from_context(*context);
    let target = CtaId::new(topology, context.cluster_id(), target_cta_id_in_cluster)?;
    let target_global = target.global_cta_id(topology)?;
    let allocation = allocations
        .get(target_global)
        .ok_or_else(|| EngineError::message("remote shared-memory CTA allocation is missing"))?;
    let view = physical.shared().remote_cta_view(
        requester,
        target,
        allocation,
        *view_offset,
        *view_len,
    )?;
    physical.shared().write_bytes(&view, byte_offset, bytes)?;
    Ok(())
}

#[inline]
pub(crate) fn with_shared_runtime_write_session<R>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    lane: usize,
    target_cta_id_in_cluster: Option<usize>,
    operation: impl FnOnce(&mut OwnerPrivateWriteSession<'_>) -> Result<R, EngineError>,
) -> Result<R, EngineError> {
    let buffer = peel_runtime_buffer_wrappers(buffer, lane, ViewAccess::Write)?;

    let requester = CtaId::from_context(*context);
    let shared = physical.shared();
    let view = match (buffer, target_cta_id_in_cluster) {
        (
            RuntimeBuffer::Shared {
                allocations,
                byte_offset,
                byte_len,
                ..
            },
            None,
        ) => {
            let allocation = allocations
                .get(context.global_cta_id())
                .ok_or_else(|| EngineError::message("shared-memory CTA allocation is missing"))?;
            shared.cta_view(requester, allocation, *byte_offset, *byte_len)?
        }
        (
            RuntimeBuffer::Shared {
                allocations,
                byte_offset,
                byte_len,
                ..
            },
            Some(target_cta_id_in_cluster),
        ) => {
            let topology = context.topology();
            let target = CtaId::new(topology, context.cluster_id(), target_cta_id_in_cluster)?;
            let target_global = target.global_cta_id(topology)?;
            let allocation = allocations.get(target_global).ok_or_else(|| {
                EngineError::message("remote shared-memory CTA allocation is missing")
            })?;
            shared.remote_cta_view(requester, target, allocation, *byte_offset, *byte_len)?
        }
        (
            RuntimeBuffer::RemoteShared {
                allocations,
                byte_offsets,
                byte_len,
                target_cta_ids,
                ..
            },
            None,
        ) => {
            let target_cta_id_in_cluster = usize::try_from(target_cta_ids[lane]).map_err(|_| {
                EngineError::message(format!("negative remote shared CTA id on lane {lane}"))
            })?;
            let view_offset = usize::try_from(byte_offsets[lane]).map_err(|_| {
                EngineError::out_of_bounds(format!(
                    "negative remote shared byte offset on lane {lane}"
                ))
            })?;
            let topology = context.topology();
            let target = CtaId::new(topology, context.cluster_id(), target_cta_id_in_cluster)?;
            let target_global = target.global_cta_id(topology)?;
            let allocation = allocations.get(target_global).ok_or_else(|| {
                EngineError::message("remote shared-memory CTA allocation is missing")
            })?;
            shared.remote_cta_view(requester, target, allocation, view_offset, *byte_len)?
        }
        _ => {
            return Err(EngineError::message(
                "issue-local payload write session requires shared memory",
            ));
        }
    };
    shared.with_write_session(&view, operation)?
}

#[inline(always)]
pub(crate) fn supports_shared_runtime_write_session(
    buffer: &RuntimeBuffer,
    lane: usize,
    target_cta_id_in_cluster: Option<usize>,
) -> bool {
    match buffer {
        RuntimeBuffer::LaneSelected { buffers } => {
            supports_shared_runtime_write_session(&buffers[lane], lane, target_cta_id_in_cluster)
        }
        RuntimeBuffer::AccessView { buffer, .. } => {
            supports_shared_runtime_write_session(buffer, lane, target_cta_id_in_cluster)
        }
        RuntimeBuffer::Shared { .. } => true,
        RuntimeBuffer::RemoteShared { .. } => target_cta_id_in_cluster.is_none(),
        _ => false,
    }
}

pub(crate) fn validate_tma_multicast_mask(
    mask: u64,
    ctas_per_cluster: usize,
) -> Result<(), EngineError> {
    if mask == 0 {
        return Err(EngineError::message(
            "copy_async multicast mask must select at least one CTA",
        ));
    }
    // Operand widths belong to the canonical PTX schema. The shared physical
    // routing path accepts the widest supported cluster mask.
    if mask > u32::MAX as u64 {
        return Err(EngineError::message(format!(
            "copy_async multicast mask 0x{mask:x} exceeds the 32-bit PTX operand"
        )));
    }
    if ctas_per_cluster < u64::BITS as usize && (mask >> ctas_per_cluster) != 0 {
        return Err(EngineError::message(format!(
            "copy_async multicast mask 0x{mask:x} names a CTA outside cluster size {ctas_per_cluster}"
        )));
    }
    Ok(())
}

pub fn read_shared_scalar_at_cta<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    target_cta_id_in_cluster: usize,
    element_index: i64,
    execution_lane: usize,
) -> Result<T, EngineError> {
    if let RuntimeBuffer::AccessView {
        buffer,
        readable_lanes,
        ..
    } = buffer
    {
        if !readable_lanes.contains(execution_lane) {
            return Err(EngineError::message(format!(
                "read through a non-readable DeclBuffer view on lane {execution_lane}"
            )));
        }
        return read_shared_scalar_at_cta(
            physical,
            context,
            buffer,
            target_cta_id_in_cluster,
            element_index,
            execution_lane,
        );
    }
    let RuntimeBuffer::Shared {
        allocations,
        byte_offset: view_offset,
        byte_len: view_len,
        ..
    } = buffer
    else {
        return Err(EngineError::message(
            "cta_group GEMM operands must use physical shared memory",
        ));
    };
    let element_index = usize::try_from(element_index).map_err(|_| {
        EngineError::out_of_bounds(format!(
            "negative remote shared element index {element_index} on lane {execution_lane}"
        ))
    })?;
    let byte_offset = element_byte_offset(buffer, element_index, T::BYTE_LEN, execution_lane)?;
    let topology = context.topology();
    let requester = CtaId::from_context(*context);
    let target = CtaId::new(topology, context.cluster_id(), target_cta_id_in_cluster)?;
    let target_global = target.global_cta_id(topology)?;
    let allocation = allocations.get(target_global).ok_or_else(|| {
        EngineError::message("remote GEMM shared-memory CTA allocation is missing")
    })?;
    let view = physical.shared().remote_cta_view(
        requester,
        target,
        allocation,
        *view_offset,
        *view_len,
    )?;
    T::decode_le(
        &physical
            .shared()
            .read_bytes(&view, byte_offset, T::BYTE_LEN)?,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsyncSourceFill {
    None,
    Zero,
    OobNan,
}

fn fill_async_source_bytes(
    bytes: &mut [u8],
    fill: AsyncSourceFill,
    lane: usize,
) -> Result<(), EngineError> {
    match fill {
        AsyncSourceFill::None => Err(EngineError::out_of_bounds(format!(
            "copy_async source coordinate is outside its buffer on lane {lane}"
        ))),
        AsyncSourceFill::Zero => {
            bytes.fill(0);
            Ok(())
        }
        AsyncSourceFill::OobNan => {
            if bytes.len() < 2 || !bytes.len().is_multiple_of(2) {
                return Err(EngineError::message(format!(
                    "copy_async OOB-NaN requires an even floating-point element width, got {} bytes",
                    bytes.len()
                )));
            }
            for chunk in bytes.chunks_exact_mut(2) {
                chunk.copy_from_slice(&crate::scalar::PTX_OOB_NAN.to_le_bytes());
            }
            Ok(())
        }
    }
}

fn apply_async_copy_tf32_rounding(bytes: &mut [u8], enabled: bool) -> Result<(), EngineError> {
    if !enabled {
        return Ok(());
    }
    let raw: [u8; 4] = bytes
        .try_into()
        .map_err(|_| EngineError::message("TF32 TMA copy requires a four-byte float32 element"))?;
    bytes.copy_from_slice(
        &super::tensor_map::tma_f32_to_tf32(f32::from_le_bytes(raw)).to_le_bytes(),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn async_copy_element_at_lane<const ITEMSIZE: usize>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    source: &RuntimeBuffer,
    source_index: i64,
    source_in_bounds: bool,
    destination: &RuntimeBuffer,
    destination_index: i64,
    destination_in_bounds: bool,
    multicast_cta_mask: Option<u64>,
    remote_cta_id: Option<usize>,
    source_fill: AsyncSourceFill,
    round_to_tf32: bool,
    reduction: Option<DeferredGlobalReduction>,
    deferred_global_writes: &mut Vec<DeferredGlobalWrite>,
    lane: usize,
) -> Result<u64, EngineError> {
    if multicast_cta_mask.is_some() && remote_cta_id.is_some() {
        return Err(EngineError::message(
            "copy_async cannot combine multicast and one remote CTA target",
        ));
    }
    if reduction.is_some()
        && (!matches!(
            runtime_buffer_base_at(source, lane),
            RuntimeBuffer::Shared { .. }
        ) || !matches!(
            runtime_buffer_base_at(destination, lane),
            RuntimeBuffer::Global(_)
        ))
    {
        return Err(EngineError::message(
            "TMA reductions require a shared-memory source and global-memory destination",
        ));
    }
    if reduction.is_some() && (multicast_cta_mask.is_some() || remote_cta_id.is_some()) {
        return Err(EngineError::message(
            "TMA reductions cannot use multicast or a remote CTA target",
        ));
    }
    if let Some(mask) = multicast_cta_mask {
        validate_tma_multicast_mask(mask, context.topology().ctas_per_cluster())?;
    }
    if !context.active_mask().contains(lane) {
        return Err(EngineError::message(format!(
            "copy_async issuing lane {lane} is not active"
        )));
    }

    let mut bytes = [0_u8; ITEMSIZE];
    if source_in_bounds {
        let source_index = usize::try_from(source_index).map_err(|_| {
            EngineError::out_of_bounds(format!(
                "copy_async source index {source_index} is negative on lane {lane}"
            ))
        })?;
        let source_offset = element_byte_offset(source, source_index, ITEMSIZE, lane)?;
        read_runtime_bytes_into(physical, context, source, lane, source_offset, &mut bytes)?;
    } else {
        fill_async_source_bytes(&mut bytes, source_fill, lane)?;
    }
    apply_async_copy_tf32_rounding(&mut bytes, round_to_tf32 && source_in_bounds)?;

    if !destination_in_bounds {
        if matches!(
            runtime_buffer_base_at(destination, lane),
            RuntimeBuffer::Global(_)
        ) {
            // TMA stores suppress out-of-range destination elements.
            return Ok(0);
        }
        return Err(EngineError::out_of_bounds(format!(
            "copy_async destination coordinate is outside its buffer on lane {lane}"
        )));
    }
    let destination_index = usize::try_from(destination_index).map_err(|_| {
        EngineError::out_of_bounds(format!(
            "copy_async destination index {destination_index} is negative on lane {lane}"
        ))
    })?;
    let destination_offset = element_byte_offset(destination, destination_index, ITEMSIZE, lane)?;

    let mut delivered = 0_u64;
    if let Some(mask) = multicast_cta_mask {
        let ctas = context.topology().ctas_per_cluster();
        for target in 0..ctas {
            if mask & (1_u64 << target) == 0 {
                continue;
            }
            write_shared_runtime_bytes_to_cta(
                physical,
                context,
                destination,
                lane,
                target,
                destination_offset,
                &bytes,
            )?;
        }
        delivered = delivered
            .checked_add(ITEMSIZE as u64)
            .ok_or_else(|| EngineError::message("copy_async delivered-byte count overflow"))?;
    } else if let Some(target) = remote_cta_id {
        write_shared_runtime_bytes_to_cta(
            physical,
            context,
            destination,
            lane,
            target,
            destination_offset,
            &bytes,
        )?;
        delivered = delivered
            .checked_add(ITEMSIZE as u64)
            .ok_or_else(|| EngineError::message("copy_async delivered-byte count overflow"))?;
    } else {
        write_or_defer_async_destination(
            physical,
            context,
            destination,
            lane,
            destination_offset,
            &bytes,
            reduction,
            deferred_global_writes,
        )?;
        delivered = delivered
            .checked_add(ITEMSIZE as u64)
            .ok_or_else(|| EngineError::message("copy_async delivered-byte count overflow"))?;
    }
    Ok(delivered)
}

/// Execute one singleton-issuer G2S/TMA element at payload issue time.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_async_copy_element_at_lane(
    physical: &PhysicalMemory,
    context: &WarpContext,
    itemsize: usize,
    source: &RuntimeBuffer,
    source_index: i64,
    source_in_bounds: bool,
    destination: &RuntimeBuffer,
    destination_index: i64,
    destination_in_bounds: bool,
    multicast_cta_mask: Option<u64>,
    remote_cta_id: Option<usize>,
    source_fill: AsyncSourceFill,
    round_to_tf32: bool,
    reduction: Option<DeferredGlobalReduction>,
    lane: usize,
) -> Result<u64, EngineError> {
    if multicast_cta_mask.is_some() && remote_cta_id.is_some() {
        return Err(EngineError::message(
            "copy_async cannot combine multicast and one remote CTA target",
        ));
    }
    if reduction.is_some() {
        return Err(EngineError::message(
            "mbarrier-backed G2S payload does not support TMA reductions",
        ));
    }
    if let Some(mask) = multicast_cta_mask {
        validate_tma_multicast_mask(mask, context.topology().ctas_per_cluster())?;
    }
    if !context.active_mask().contains(lane) {
        return Err(EngineError::message(format!(
            "copy_async issuing lane {lane} is not active"
        )));
    }

    execute_async_copy_element_with_writer(
        physical,
        context,
        itemsize,
        source,
        source_index,
        source_in_bounds,
        destination,
        destination_index,
        destination_in_bounds,
        source_fill,
        round_to_tf32,
        lane,
        |destination_offset, bytes| {
            if let Some(mask) = multicast_cta_mask {
                let ctas = context.topology().ctas_per_cluster();
                for target in 0..ctas {
                    if mask & (1_u64 << target) == 0 {
                        continue;
                    }
                    write_shared_runtime_bytes_to_cta(
                        physical,
                        context,
                        destination,
                        lane,
                        target,
                        destination_offset,
                        bytes,
                    )?;
                }
            } else if let Some(target) = remote_cta_id {
                write_shared_runtime_bytes_to_cta(
                    physical,
                    context,
                    destination,
                    lane,
                    target,
                    destination_offset,
                    bytes,
                )?;
            } else {
                write_runtime_bytes(
                    physical,
                    context,
                    destination,
                    lane,
                    destination_offset,
                    bytes,
                )?;
            }
            Ok(())
        },
    )
}

/// Execute up to one warp-width of fixed-target G2S elements with one global
/// gather and one destination write session.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn execute_async_copy_element_batch_in_write_session(
    itemsize: usize,
    source: &SharedReadSession<'_>,
    source_element_capacity: usize,
    elements: &[(i64, i64, bool, bool); WARP_SIZE],
    destination_element_capacity: usize,
    source_fill: AsyncSourceFill,
    round_to_tf32: bool,
    lane: usize,
    writer: &mut OwnerPrivateWriteSession<'_>,
    count: usize,
) -> Result<u64, EngineError> {
    const BATCH_STRIDE: usize = MAX_RUNTIME_SCALAR_BYTES;

    debug_assert!(count <= WARP_SIZE);
    debug_assert!(itemsize != 0);
    debug_assert!(itemsize <= BATCH_STRIDE);
    let mut source_offsets = WarpValue::splat(0_usize);
    let mut destination_offsets = [0_usize; WARP_SIZE];
    let mut read_mask_bits = 0_u32;
    for element in 0..count {
        let (source_index, destination_index, source_in_bounds, destination_in_bounds) =
            elements[element];
        if source_in_bounds {
            let source_index = usize::try_from(source_index).map_err(|_| {
                EngineError::out_of_bounds(format!(
                    "copy_async source index {} is negative on lane {lane}",
                    source_index
                ))
            })?;
            if source_index >= source_element_capacity {
                let source_byte_len = source.byte_len();
                return Err(EngineError::out_of_bounds(format!(
                    "buffer element {source_index} ({itemsize} bytes) is outside {source_byte_len} bytes on lane {lane}"
                )));
            }
            source_offsets[element] = source_index * itemsize;
            read_mask_bits |= 1_u32 << element;
        }

        if !destination_in_bounds {
            return Err(EngineError::out_of_bounds(format!(
                "copy_async destination coordinate is outside its buffer on lane {lane}"
            )));
        }
        let destination_index = usize::try_from(destination_index).map_err(|_| {
            EngineError::out_of_bounds(format!(
                "copy_async destination index {} is negative on lane {lane}",
                destination_index
            ))
        })?;
        if destination_index >= destination_element_capacity {
            let destination_byte_len = writer.byte_len();
            return Err(EngineError::out_of_bounds(format!(
                "buffer element {destination_index} ({itemsize} bytes) is outside {destination_byte_len} bytes on lane {lane}"
            )));
        }
        destination_offsets[element] = destination_index * itemsize;
    }

    let mut batch_bytes = [[0_u8; BATCH_STRIDE]; WARP_SIZE];
    let read_mask = WarpMask::from_bits(read_mask_bits);
    if read_mask.len() != count {
        for element in 0..count {
            if !elements[element].2 {
                fill_async_source_bytes(&mut batch_bytes[element][..itemsize], source_fill, lane)?;
            }
        }
    }
    if !read_mask.is_empty() {
        source.read_bytes_batch_into_prevalidated(
            &source_offsets,
            read_mask,
            itemsize,
            BATCH_STRIDE,
            batch_bytes.as_flattened_mut(),
        )?;
    }
    if round_to_tf32 {
        for element in 0..count {
            let bytes = &mut batch_bytes[element][..itemsize];
            apply_async_copy_tf32_rounding(bytes, elements[element].2)?;
        }
    }
    writer.write_strided_batch_prevalidated(
        &destination_offsets,
        batch_bytes.as_flattened(),
        BATCH_STRIDE,
        itemsize,
        count,
    );
    u64::try_from(
        itemsize
            .checked_mul(count)
            .ok_or_else(|| EngineError::message("copy_async delivered-byte count overflow"))?,
    )
    .map_err(|_| EngineError::message("copy_async delivered-byte count overflow"))
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn execute_async_copy_element_with_writer(
    physical: &PhysicalMemory,
    context: &WarpContext,
    itemsize: usize,
    source: &RuntimeBuffer,
    source_index: i64,
    source_in_bounds: bool,
    destination: &RuntimeBuffer,
    destination_index: i64,
    destination_in_bounds: bool,
    source_fill: AsyncSourceFill,
    round_to_tf32: bool,
    lane: usize,
    write: impl FnOnce(usize, &[u8]) -> Result<(), EngineError>,
) -> Result<u64, EngineError> {
    const INLINE_ASYNC_COPY_BYTES: usize = 16;
    let mut inline_bytes = [0_u8; INLINE_ASYNC_COPY_BYTES];
    let mut heap_bytes = Vec::new();
    let bytes = if itemsize <= INLINE_ASYNC_COPY_BYTES {
        &mut inline_bytes[..itemsize]
    } else {
        heap_bytes.resize(itemsize, 0);
        heap_bytes.as_mut_slice()
    };
    if source_in_bounds {
        let source_index = usize::try_from(source_index).map_err(|_| {
            EngineError::out_of_bounds(format!(
                "copy_async source index {source_index} is negative on lane {lane}"
            ))
        })?;
        let source_offset = element_byte_offset(source, source_index, itemsize, lane)?;
        read_runtime_bytes_into(physical, context, source, lane, source_offset, &mut *bytes)?;
    } else {
        fill_async_source_bytes(&mut *bytes, source_fill, lane)?;
    }
    apply_async_copy_tf32_rounding(&mut *bytes, round_to_tf32 && source_in_bounds)?;

    if !destination_in_bounds {
        if matches!(
            runtime_buffer_base_at(destination, lane),
            RuntimeBuffer::Global(_)
        ) {
            return Ok(0);
        }
        return Err(EngineError::out_of_bounds(format!(
            "copy_async destination coordinate is outside its buffer on lane {lane}"
        )));
    }
    let destination_index = usize::try_from(destination_index).map_err(|_| {
        EngineError::out_of_bounds(format!(
            "copy_async destination index {destination_index} is negative on lane {lane}"
        ))
    })?;
    let destination_offset = element_byte_offset(destination, destination_index, itemsize, lane)?;
    write(destination_offset, bytes)?;
    Ok(itemsize as u64)
}

#[inline]
pub fn load_scalar_warp<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    indices: &WarpValue<i64>,
    mask: WarpMask,
) -> Result<WarpValue<T>, EngineError> {
    let byte_offsets = runtime_scalar_byte_offsets(buffer, indices, T::BYTE_LEN, mask)?;
    load_scalar_warp_at_byte_offsets(physical, context, buffer, &byte_offsets, mask, None)
}

#[inline]
pub(crate) fn load_scalar_warp_with_source<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    indices: &WarpValue<i64>,
    mask: WarpMask,
    source: ReadSource,
) -> Result<WarpValue<T>, EngineError> {
    let byte_offsets = runtime_scalar_byte_offsets(buffer, indices, T::BYTE_LEN, mask)?;
    load_scalar_warp_at_byte_offsets(
        physical,
        context,
        buffer,
        &byte_offsets,
        mask,
        Some(source),
    )
}

#[inline]
pub(crate) fn load_scalar_warp_at_byte_offsets<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    byte_offsets: &WarpValue<usize>,
    mask: WarpMask,
    source: Option<ReadSource>,
) -> Result<WarpValue<T>, EngineError> {
    let owner_private = match buffer {
        RuntimeBuffer::Local {
            allocations,
            byte_offset,
            byte_len,
        } => Some((physical.local(), allocations, *byte_offset, *byte_len)),
        RuntimeBuffer::Register {
            allocations,
            byte_offset,
            byte_len,
        } => Some((physical.registers(), allocations, *byte_offset, *byte_len)),
        _ => None,
    };
    if let Some((memory, allocations, view_offset, view_len)) = owner_private {
        let allocation = allocations
            .get(context.global_warp_id())
            .ok_or_else(|| EngineError::message("warp-private allocation is missing"))?;
        return memory.read_lane_scalars_batch::<T>(
            WarpId::from_context(*context),
            allocation,
            view_offset,
            view_len,
            byte_offsets,
            mask,
            source,
        );
    }
    if matches!(
        buffer,
        RuntimeBuffer::Global(_) | RuntimeBuffer::Shared { .. }
    ) {
        let mut lane_bytes = [0_u8; MAX_RUNTIME_SCALAR_BYTES * WARP_SIZE];
        let packed_len = WARP_SIZE * T::BYTE_LEN;
        let uniform_lane = mask.first_active().filter(|first_lane| {
            let byte_offset = byte_offsets[*first_lane];
            mask.into_iter()
                .all(|lane| byte_offsets[lane] == byte_offset)
        });
        // Global and CTA-shared buffers name one physical allocation for the
        // whole warp. When every active lane addresses the same scalar, one
        // numeric read and decode is the exact warp result. Racecheck still
        // receives the original lane mask and footprint through the enclosing
        // physical-access operation.
        let numeric_mask = uniform_lane
            .map(|lane| WarpMask::from_bits(1_u32 << lane))
            .unwrap_or(mask);
        match buffer {
            RuntimeBuffer::Global(view) => physical.global().read_shared_bytes_batch_into(
                view,
                byte_offsets,
                numeric_mask,
                T::BYTE_LEN,
                T::BYTE_LEN,
                &mut lane_bytes[..packed_len],
            )?,
            RuntimeBuffer::Shared {
                allocations,
                byte_offset: view_offset,
                byte_len: view_len,
                ..
            } => {
                let allocation = allocations.get(context.global_cta_id()).ok_or_else(|| {
                    EngineError::message("shared-memory CTA allocation is missing")
                })?;
                physical.shared().read_cta_bytes_batch_into(
                    CtaId::from_context(*context),
                    allocation,
                    *view_offset,
                    *view_len,
                    byte_offsets,
                    numeric_mask,
                    T::BYTE_LEN,
                    T::BYTE_LEN,
                    &mut lane_bytes[..packed_len],
                )?;
            }
            _ => unreachable!(),
        }
        let mut values = WarpValue::splat(T::zero());
        if let Some(lane) = uniform_lane {
            let start = lane * T::BYTE_LEN;
            values.masked_fill(mask, T::decode_le(&lane_bytes[start..start + T::BYTE_LEN])?);
            return Ok(values);
        }
        for lane in mask {
            let start = lane * T::BYTE_LEN;
            values[lane] = T::decode_le(&lane_bytes[start..start + T::BYTE_LEN])?;
        }
        return Ok(values);
    }
    let mut values = WarpValue::splat(T::zero());
    let mut bytes = [0_u8; MAX_RUNTIME_SCALAR_BYTES];
    for lane in mask {
        read_runtime_bytes_into(
            physical,
            context,
            buffer,
            lane,
            byte_offsets[lane],
            &mut bytes[..T::BYTE_LEN],
        )?;
        values[lane] = T::decode_le(&bytes[..T::BYTE_LEN])?;
    }
    Ok(values)
}

/// Validate one projected warp-scalar access without reading or writing data.
pub fn validate_runtime_scalar_warp_access(
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    indices: &WarpValue<i64>,
    mask: WarpMask,
    byte_len: usize,
    kind: PhysicalAccessKind,
) -> Result<(), EngineError> {
    let byte_offsets = runtime_scalar_byte_offsets(buffer, indices, byte_len, mask)?;
    validate_runtime_scalar_warp_access_at_byte_offsets(
        context,
        buffer,
        &byte_offsets,
        mask,
        byte_len,
        kind,
    )
}

pub(crate) fn validate_runtime_scalar_warp_access_at_byte_offsets(
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    byte_offsets: &WarpValue<usize>,
    mask: WarpMask,
    byte_len: usize,
    kind: PhysicalAccessKind,
) -> Result<(), EngineError> {
    for lane in mask {
        resolve_runtime_physical_access(context, buffer, lane, byte_offsets[lane], byte_len, kind)?;
    }
    Ok(())
}

fn load_scalar_lane_impl<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    index: i64,
    lane: usize,
    zero_filled: bool,
) -> Result<T, EngineError> {
    let index = usize::try_from(index).map_err(|_| {
        EngineError::out_of_bounds(format!("negative buffer index {index} on lane {lane}"))
    })?;
    let byte_offset = element_byte_offset(buffer, index, T::BYTE_LEN, lane)?;
    let mut bytes = [0_u8; MAX_RUNTIME_SCALAR_BYTES];
    if zero_filled {
        read_runtime_bytes_zero_filled_into(
            physical,
            context,
            buffer,
            lane,
            byte_offset,
            &mut bytes[..T::BYTE_LEN],
        )?;
    } else {
        read_runtime_bytes_into(
            physical,
            context,
            buffer,
            lane,
            byte_offset,
            &mut bytes[..T::BYTE_LEN],
        )?;
    }
    T::decode_le(&bytes[..T::BYTE_LEN])
}

pub fn load_scalar_lane<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    index: i64,
    lane: usize,
) -> Result<T, EngineError> {
    load_scalar_lane_impl(physical, context, buffer, index, lane, false)
}

pub fn load_warp_private_scalar_at_thread<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    target_warp_id_in_cta: usize,
    index: i64,
    lane: usize,
) -> Result<T, EngineError> {
    if let RuntimeBuffer::AccessView {
        buffer,
        readable_lanes,
        ..
    } = buffer
    {
        if !readable_lanes.contains(lane) {
            return Err(EngineError::message(format!(
                "read through a non-readable DeclBuffer view on lane {lane}"
            )));
        }
        return load_warp_private_scalar_at_thread(
            physical,
            context,
            buffer,
            target_warp_id_in_cta,
            index,
            lane,
        );
    }
    let (allocations, view_offset, view_len, memory) = match buffer {
        RuntimeBuffer::Local {
            allocations,
            byte_offset,
            byte_len,
        } => (allocations, *byte_offset, *byte_len, physical.local()),
        RuntimeBuffer::Register {
            allocations,
            byte_offset,
            byte_len,
        } => (allocations, *byte_offset, *byte_len, physical.registers()),
        _ => {
            return Err(EngineError::message(
                "warp-private owner transport requires local or register memory",
            ));
        }
    };
    let topology = context.topology();
    let target = WarpId::new(
        topology,
        CtaId::from_context(*context),
        target_warp_id_in_cta,
    )?;
    let target_global = target.global_warp_id(topology)?;
    let allocation = allocations.get(target_global).ok_or_else(|| {
        EngineError::message("warp-private owner transport allocation is missing")
    })?;
    let index = usize::try_from(index).map_err(|_| {
        EngineError::out_of_bounds(format!(
            "negative warp-private owner transport index {index} on lane {lane}"
        ))
    })?;
    let byte_offset = element_byte_offset(buffer, index, T::BYTE_LEN, lane)?;
    let mut bytes = [0_u8; MAX_RUNTIME_SCALAR_BYTES];
    memory.read_lane_bytes_into(
        target,
        allocation,
        lane,
        view_offset,
        view_len,
        byte_offset,
        &mut bytes[..T::BYTE_LEN],
    )
    .map_err(|error| {
        EngineError::message(format!(
            "warp-private owner transport read from target warp {target_warp_id_in_cta}, lane {lane}, index {index}: {error}"
        ))
    })?;
    T::decode_le(&bytes[..T::BYTE_LEN])
}

#[inline]
pub fn store_scalar_warp<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    indices: &WarpValue<i64>,
    values: &WarpValue<T>,
    mask: WarpMask,
) -> Result<(), EngineError> {
    let byte_offsets = runtime_scalar_byte_offsets(buffer, indices, T::BYTE_LEN, mask)?;
    store_scalar_warp_at_byte_offsets(physical, context, buffer, &byte_offsets, values, mask)
}

#[inline]
pub(crate) fn store_scalar_warp_at_byte_offsets<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    byte_offsets: &WarpValue<usize>,
    values: &WarpValue<T>,
    mask: WarpMask,
) -> Result<(), EngineError> {
    let owner_private = match buffer {
        RuntimeBuffer::Local {
            allocations,
            byte_offset,
            byte_len,
        } => Some((physical.local(), allocations, *byte_offset, *byte_len)),
        RuntimeBuffer::Register {
            allocations,
            byte_offset,
            byte_len,
        } => Some((physical.registers(), allocations, *byte_offset, *byte_len)),
        _ => None,
    };
    if let Some((memory, allocations, view_offset, view_len)) = owner_private {
        let allocation = allocations
            .get(context.global_warp_id())
            .ok_or_else(|| EngineError::message("warp-private allocation is missing"))?;
        memory.write_lane_scalars_batch(
            WarpId::from_context(*context),
            allocation,
            view_offset,
            view_len,
            byte_offsets,
            values,
            mask,
        )?;
        return Ok(());
    }
    if matches!(
        buffer,
        RuntimeBuffer::Global(_) | RuntimeBuffer::Shared { .. }
    ) {
        let mut lane_bytes = [0_u8; MAX_RUNTIME_SCALAR_BYTES * WARP_SIZE];
        let packed_len = WARP_SIZE * T::BYTE_LEN;
        for lane in mask {
            let start = lane * T::BYTE_LEN;
            values[lane].encode_le_into(&mut lane_bytes[start..start + T::BYTE_LEN]);
        }
        match buffer {
            RuntimeBuffer::Global(view) => physical.global().write_shared_bytes_batch(
                view,
                byte_offsets,
                mask,
                T::BYTE_LEN,
                T::BYTE_LEN,
                &lane_bytes[..packed_len],
            )?,
            RuntimeBuffer::Shared {
                allocations,
                byte_offset: view_offset,
                byte_len: view_len,
                ..
            } => {
                let allocation = allocations.get(context.global_cta_id()).ok_or_else(|| {
                    EngineError::message("shared-memory CTA allocation is missing")
                })?;
                physical.shared().write_cta_bytes_batch(
                    CtaId::from_context(*context),
                    allocation,
                    *view_offset,
                    *view_len,
                    byte_offsets,
                    mask,
                    T::BYTE_LEN,
                    T::BYTE_LEN,
                    &lane_bytes[..packed_len],
                )?;
            }
            _ => unreachable!(),
        }
        return Ok(());
    }
    let mut bytes = [0_u8; MAX_RUNTIME_SCALAR_BYTES];
    for lane in mask {
        values[lane].encode_le_into(&mut bytes[..T::BYTE_LEN]);
        write_runtime_bytes(
            physical,
            context,
            buffer,
            lane,
            byte_offsets[lane],
            &bytes[..T::BYTE_LEN],
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarpMmaFragmentRole {
    A,
    B,
    C,
}

pub fn load_warp_mma_m16n8_fragment<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    role: WarpMmaFragmentRole,
    mma_k: usize,
) -> Result<Vec<T>, EngineError> {
    super::require_full_warp_sync(context.active_mask(), "warp MMA fragment load")?;
    if !matches!(mma_k, 8 | 16) {
        return Err(EngineError::message(format!(
            "warp MMA fragment has unsupported K={mma_k}"
        )));
    }
    let slots_per_lane = match role {
        WarpMmaFragmentRole::A => mma_k / 2,
        WarpMmaFragmentRole::B => mma_k / 4,
        WarpMmaFragmentRole::C => 4,
    };
    let (memory, allocations, view_offset, view_len) = match buffer {
        RuntimeBuffer::Local {
            allocations,
            byte_offset,
            byte_len,
        } => (physical.local(), allocations, *byte_offset, *byte_len),
        RuntimeBuffer::Register {
            allocations,
            byte_offset,
            byte_len,
        } => (physical.registers(), allocations, *byte_offset, *byte_len),
        _ => {
            return Err(EngineError::message(
                "warp MMA fragment load requires an owner-private buffer",
            ));
        }
    };
    let allocation = allocations
        .get(context.global_warp_id())
        .ok_or_else(|| EngineError::message("warp-private allocation is missing"))?;
    let row_bytes = slots_per_lane
        .checked_mul(T::BYTE_LEN)
        .ok_or_else(|| EngineError::message("warp MMA fragment row size overflow"))?;
    let mut encoded = vec![0_u8; WARP_SIZE * row_bytes];
    memory.read_lane_bytes_batch_into(
        WarpId::from_context(*context),
        allocation,
        view_offset,
        view_len,
        &WarpValue::splat(0),
        context.active_mask(),
        row_bytes,
        row_bytes,
        &mut encoded,
    )?;
    let physical_values = encoded
        .chunks_exact(T::BYTE_LEN)
        .map(T::decode_le)
        .collect::<Result<Vec<_>, _>>()?;
    let mut output = vec![
        T::zero();
        match role {
            WarpMmaFragmentRole::A => 16 * mma_k,
            WarpMmaFragmentRole::B => 8 * mma_k,
            WarpMmaFragmentRole::C => 16 * 8,
        }
    ];
    match role {
        WarpMmaFragmentRole::A => {
            for row in 0..16 {
                for inner in 0..mma_k {
                    let lane = 4 * (row % 8) + (inner % 8) / 2;
                    let slot = 4 * ((inner % mma_k) / 8) + 2 * ((row % 16) / 8) + inner % 2;
                    output[row * mma_k + inner] = physical_values[lane * slots_per_lane + slot];
                }
            }
        }
        WarpMmaFragmentRole::B => {
            for col in 0..8 {
                for inner in 0..mma_k {
                    let lane = 4 * (col % 8) + (inner % 8) / 2;
                    let slot = 2 * ((inner % mma_k) / 8) + inner % 2;
                    output[col * mma_k + inner] = physical_values[lane * slots_per_lane + slot];
                }
            }
        }
        WarpMmaFragmentRole::C => {
            for row in 0..16 {
                for col in 0..8 {
                    let lane = 4 * (row % 8) + (col % 8) / 2;
                    let slot = 2 * ((row % 16) / 8) + col % 2;
                    output[row * 8 + col] = physical_values[lane * slots_per_lane + slot];
                }
            }
        }
    }
    Ok(output)
}

pub fn store_warp_mma_m16n8_f32_fragment(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    values: &[f32],
) -> Result<(), EngineError> {
    super::require_full_warp_sync(context.active_mask(), "warp MMA fragment store")?;
    if values.len() != 16 * 8 {
        return Err(EngineError::message(format!(
            "warp MMA output fragment has {} values, expected 128",
            values.len()
        )));
    }
    let (memory, allocations, view_offset, view_len) = match buffer {
        RuntimeBuffer::Local {
            allocations,
            byte_offset,
            byte_len,
        } => (physical.local(), allocations, *byte_offset, *byte_len),
        RuntimeBuffer::Register {
            allocations,
            byte_offset,
            byte_len,
        } => (physical.registers(), allocations, *byte_offset, *byte_len),
        _ => {
            return Err(EngineError::message(
                "warp MMA fragment store requires an owner-private buffer",
            ));
        }
    };
    let allocation = allocations
        .get(context.global_warp_id())
        .ok_or_else(|| EngineError::message("warp-private allocation is missing"))?;
    let mut physical_values = [0.0_f32; WARP_SIZE * 4];
    for row in 0..16 {
        for col in 0..8 {
            let lane = 4 * (row % 8) + (col % 8) / 2;
            let slot = 2 * ((row % 16) / 8) + col % 2;
            physical_values[lane * 4 + slot] = values[row * 8 + col];
        }
    }
    let mut encoded = Vec::with_capacity(physical_values.len() * size_of::<f32>());
    for value in physical_values {
        encoded.extend_from_slice(&value.to_le_bytes());
    }
    memory.write_lane_bytes_batch(
        WarpId::from_context(*context),
        allocation,
        view_offset,
        view_len,
        &WarpValue::splat(0),
        context.active_mask(),
        4 * size_of::<f32>(),
        4 * size_of::<f32>(),
        &encoded,
    )?;
    Ok(())
}

pub fn load_physical_ptr_warp<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    mask: WarpMask,
) -> Result<WarpValue<T>, EngineError> {
    validate_physical_access(pointer, mask, T::BYTE_LEN, "load")?;
    let mut byte_offsets = WarpValue::splat(0_usize);
    for lane in mask {
        byte_offsets[lane] = pointer.lane_read_byte_offset(lane, T::BYTE_LEN)?;
    }
    load_scalar_warp_at_byte_offsets(
        physical,
        context,
        pointer.buffer(),
        &byte_offsets,
        mask,
        None,
    )
}

pub fn store_physical_ptr_warp<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    values: &WarpValue<T>,
    mask: WarpMask,
) -> Result<(), EngineError> {
    validate_physical_access(pointer, mask, T::BYTE_LEN, "store")?;
    let mut byte_offsets = WarpValue::splat(0_usize);
    for lane in mask {
        byte_offsets[lane] = pointer.lane_write_byte_offset(lane, T::BYTE_LEN)?;
    }
    store_scalar_warp_at_byte_offsets(
        physical,
        context,
        pointer.buffer(),
        &byte_offsets,
        values,
        mask,
    )
}

/// Validate one physical access of `access_bytes` bytes per lane.
///
/// The instruction's width determines alignment, independently of the backing
/// dtype: raw PTX can read or overwrite part of an element. A register has no
/// address to align. Use each lane's actual space, including generic pointers.
fn validate_physical_access(
    pointer: &PhysicalPtr,
    mask: WarpMask,
    access_bytes: usize,
    operation: &str,
) -> Result<(), EngineError> {
    if access_bytes == 0 {
        return Err(EngineError::message(format!(
            "{operation} access width must be nonzero"
        )));
    }
    for lane in mask {
        if matches!(pointer.pointer_space_at_lane(lane)?, PointerSpace::Register) {
            continue;
        }
        if pointer.lane_physical_byte_offset(lane, access_bytes)? % access_bytes != 0 {
            return Err(EngineError::message(format!(
                "{operation} requires {access_bytes}-byte alignment on lane {lane}"
            )));
        }
    }
    Ok(())
}

pub fn raw_load_physical_ptr_warp<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    mask: WarpMask,
    ptx_space: PtxStateSpace,
) -> Result<WarpValue<T>, EngineError> {
    let byte_offsets = pointer.resolve_load_byte_offsets(ptx_space, mask, T::BYTE_LEN)?;
    load_scalar_warp_at_byte_offsets(
        physical,
        context,
        pointer.buffer(),
        &byte_offsets,
        mask,
        None,
    )
}

pub fn raw_load_physical_ptr_warp_atomic<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    mask: WarpMask,
    ptx_space: PtxStateSpace,
) -> Result<WarpValue<T>, EngineError> {
    pointer.require_ptx_space_for_mask(ptx_space, mask)?;
    let ordering = physical.ordering();
    let access_ranges = atomic_access_ranges(context, pointer, mask, T::BYTE_LEN)?;
    let atomic_access = ordering.begin_atomic_access(*context, access_ranges)?;
    let mut values = WarpValue::splat(T::zero());
    for lane in mask {
        if pointer.lane_physical_byte_offset(lane, T::BYTE_LEN)? % T::BYTE_LEN != 0 {
            return Err(EngineError::message(format!(
                "atomic-coherent load requires {}-byte alignment on lane {lane}",
                T::BYTE_LEN
            )));
        }
        let byte_offset = pointer.lane_read_byte_offset(lane, T::BYTE_LEN)?;
        let bytes = read_runtime_bytes_unordered(
            physical,
            context,
            pointer.buffer(),
            lane,
            byte_offset,
            T::BYTE_LEN,
        )?;
        values[lane] = T::decode_le(&bytes)?;
    }
    if let Some(atomic_access) = atomic_access {
        atomic_access.finish_load()?;
    }
    Ok(values)
}

pub fn raw_store_physical_ptr_warp<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    values: &WarpValue<T>,
    mask: WarpMask,
    ptx_space: PtxStateSpace,
) -> Result<(), EngineError> {
    pointer.require_ptx_space_for_mask(ptx_space, mask)?;
    store_physical_ptr_warp::<T>(physical, context, pointer, values, mask)
}

/// Classic cp.async reads a prefix but writes (and aligns to) the full width.
/// Resolve this contract identically for numerical execution and observations.
fn cp_async_lane_offsets(
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    lane: usize,
    source_bytes: usize,
    byte_count: usize,
) -> Result<(usize, Option<usize>), EngineError> {
    if !matches!(byte_count, 4 | 8 | 16) || source_bytes > byte_count {
        return Err(EngineError::message(format!(
            "cp.async requires a 4/8/16-byte width and source size in [0, {byte_count}], got {source_bytes} on lane {lane}"
        )));
    }
    let lane_mask =
        WarpMask::from_lanes([lane]).map_err(|error| EngineError::message(error.to_string()))?;
    destination.require_ptx_space_for_mask(PtxStateSpace::Shared, lane_mask)?;
    validate_physical_access(destination, lane_mask, byte_count, "cp.async destination")?;
    let destination_offset = destination.lane_write_byte_offset(lane, byte_count)?;
    let source_offset = if source_bytes != 0 {
        source.require_ptx_space_for_mask(PtxStateSpace::Global, lane_mask)?;
        if source.lane_physical_byte_offset(lane, source_bytes)? % byte_count != 0 {
            return Err(EngineError::message(format!(
                "cp.async source requires {byte_count}-byte alignment on lane {lane}"
            )));
        }
        Some(source.lane_read_byte_offset(lane, source_bytes)?)
    } else {
        None
    };
    Ok((destination_offset, source_offset))
}

pub fn copy_physical_ptr_bytes(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    source_sizes: &WarpValue<u32>,
    active_mask: WarpMask,
    byte_count: usize,
) -> Result<(), EngineError> {
    for lane in active_mask {
        let source_bytes = source_sizes[lane] as usize;
        let (destination_offset, source_offset) =
            cp_async_lane_offsets(destination, source, lane, source_bytes, byte_count)?;
        let mut bytes = if let Some(source_offset) = source_offset {
            read_runtime_bytes(
                physical,
                context,
                source.buffer(),
                lane,
                source_offset,
                source_bytes,
            )?
        } else {
            Vec::new()
        };
        bytes.resize(byte_count, 0);
        write_runtime_bytes(
            physical,
            context,
            destination.buffer(),
            lane,
            destination_offset,
            &bytes,
        )?;
    }
    Ok(())
}

pub fn plan_cp_async_physical_ptr_lane_accesses(
    operation: &OperationContext,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    lane: usize,
    source_bytes: usize,
    byte_count: usize,
) -> Result<(Vec<PhysicalAccessBatch>, Vec<PhysicalAccessBatch>), EngineError> {
    validate_single_lane_async_operation(operation, lane)?;
    let (destination_offset, source_offset) =
        cp_async_lane_offsets(destination, source, lane, source_bytes, byte_count)?;

    let mut source_accesses = Vec::new();
    if let Some(source_offset) = source_offset {
        let access = resolve_runtime_physical_access(
            context,
            source.buffer(),
            lane,
            source_offset,
            source_bytes,
            PhysicalAccessKind::Read,
        )?;
        source_accesses.push(
            single_lane_physical_access_batch(
                operation,
                lane,
                PhysicalAccessKind::Read,
                access.space(),
                vec![access.span()],
            )?
            .with_proxy_memory_domain(access.proxy_memory_domain()),
        );
    }

    let access = resolve_runtime_physical_access(
        context,
        destination.buffer(),
        lane,
        destination_offset,
        byte_count,
        PhysicalAccessKind::Write,
    )?;
    let destination_accesses = vec![
        single_lane_physical_access_batch(
            operation,
            lane,
            PhysicalAccessKind::Write,
            access.space(),
            vec![access.span()],
        )?
        .with_proxy_memory_domain(access.proxy_memory_domain()),
    ];
    Ok((source_accesses, destination_accesses))
}

pub fn raw_store_vector_physical_ptr_warp<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    values: &[&WarpValue<T>],
    mask: WarpMask,
    ptx_space: PtxStateSpace,
    sinks: u8,
) -> Result<(), EngineError> {
    if values.is_empty() {
        return Err(EngineError::message(
            "vector store requires at least one value",
        ));
    }
    pointer.require_ptx_space_for_mask(ptx_space, mask)?;
    let byte_len = T::BYTE_LEN
        .checked_mul(values.len())
        .ok_or_else(|| EngineError::message("vector store byte width overflow"))?;
    validate_physical_access(pointer, mask, byte_len, "vector store")?;
    let mut byte_offsets = WarpValue::splat(0_usize);
    for lane in mask {
        byte_offsets[lane] = pointer.lane_write_byte_offset(lane, byte_len)?;
    }
    for lane in mask {
        let mut bytes = vec![0_u8; byte_len];
        for (index, value) in values.iter().enumerate() {
            let start = index * T::BYTE_LEN;
            value[lane].encode_le_into(&mut bytes[start..start + T::BYTE_LEN]);
        }
        if sinks == 0 {
            write_runtime_bytes(
                physical,
                context,
                pointer.buffer(),
                lane,
                byte_offsets[lane],
                &bytes,
            )?;
        } else {
            for index in 0..values.len() {
                if sinks & (1 << index) == 0 {
                    let start = index * T::BYTE_LEN;
                    write_runtime_bytes(
                        physical,
                        context,
                        pointer.buffer(),
                        lane,
                        byte_offsets[lane] + start,
                        &bytes[start..start + T::BYTE_LEN],
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawAtomicOperation {
    Add,
    AddNoFtz,
    BitAnd,
    BitOr,
    BitXor,
    Exchange,
    Increment,
    Decrement,
    Minimum,
    Maximum,
}

pub trait RawAtomicScalar: RuntimeScalar {
    fn apply_atomic(
        operation: RawAtomicOperation,
        old: Self,
        operand: Self,
        ptx_space: PtxStateSpace,
    ) -> Result<Self, EngineError>;
}

fn atomic_access_ranges(
    context: &WarpContext,
    pointer: &PhysicalPtr,
    mask: WarpMask,
    byte_len: usize,
) -> Result<Vec<(PhysicalAddress, usize)>, EngineError> {
    let mut ranges = Vec::with_capacity(mask.len());
    for lane in mask {
        let lane_mask = WarpMask::from_bits(1_u32 << lane);
        ranges.push((pointer.resolve_uniform(context, lane_mask)?, byte_len));
    }
    Ok(ranges)
}

fn unsupported_atomic_operation<T>(operation: RawAtomicOperation) -> Result<T, EngineError> {
    Err(EngineError::message(format!(
        "atomic operation {operation:?} is invalid for this scalar type"
    )))
}

impl RawAtomicScalar for i32 {
    fn apply_atomic(
        operation: RawAtomicOperation,
        old: Self,
        operand: Self,
        _ptx_space: PtxStateSpace,
    ) -> Result<Self, EngineError> {
        match operation {
            RawAtomicOperation::Add => Ok(old.wrapping_add(operand)),
            RawAtomicOperation::Minimum => Ok(old.min(operand)),
            RawAtomicOperation::Maximum => Ok(old.max(operand)),
            _ => unsupported_atomic_operation(operation),
        }
    }
}

impl RawAtomicScalar for i64 {
    fn apply_atomic(
        operation: RawAtomicOperation,
        old: Self,
        operand: Self,
        _ptx_space: PtxStateSpace,
    ) -> Result<Self, EngineError> {
        match operation {
            RawAtomicOperation::Minimum => Ok(old.min(operand)),
            RawAtomicOperation::Maximum => Ok(old.max(operand)),
            _ => unsupported_atomic_operation(operation),
        }
    }
}

impl RawAtomicScalar for u32 {
    fn apply_atomic(
        operation: RawAtomicOperation,
        old: Self,
        operand: Self,
        _ptx_space: PtxStateSpace,
    ) -> Result<Self, EngineError> {
        match operation {
            RawAtomicOperation::Add => Ok(old.wrapping_add(operand)),
            RawAtomicOperation::BitAnd => Ok(old & operand),
            RawAtomicOperation::BitOr => Ok(old | operand),
            RawAtomicOperation::BitXor => Ok(old ^ operand),
            RawAtomicOperation::Exchange => Ok(operand),
            RawAtomicOperation::Increment => Ok(if old >= operand { 0 } else { old + 1 }),
            RawAtomicOperation::Decrement => Ok(if old == 0 || old > operand {
                operand
            } else {
                old - 1
            }),
            RawAtomicOperation::Minimum => Ok(old.min(operand)),
            RawAtomicOperation::Maximum => Ok(old.max(operand)),
            RawAtomicOperation::AddNoFtz => unsupported_atomic_operation(operation),
        }
    }
}

impl RawAtomicScalar for u64 {
    fn apply_atomic(
        operation: RawAtomicOperation,
        old: Self,
        operand: Self,
        _ptx_space: PtxStateSpace,
    ) -> Result<Self, EngineError> {
        match operation {
            RawAtomicOperation::Add => Ok(old.wrapping_add(operand)),
            RawAtomicOperation::BitAnd => Ok(old & operand),
            RawAtomicOperation::BitOr => Ok(old | operand),
            RawAtomicOperation::BitXor => Ok(old ^ operand),
            RawAtomicOperation::Exchange => Ok(operand),
            RawAtomicOperation::Minimum => Ok(old.min(operand)),
            RawAtomicOperation::Maximum => Ok(old.max(operand)),
            _ => unsupported_atomic_operation(operation),
        }
    }
}

impl RawAtomicScalar for crate::scalar::U64x2 {
    fn apply_atomic(
        operation: RawAtomicOperation,
        _old: Self,
        operand: Self,
        _ptx_space: PtxStateSpace,
    ) -> Result<Self, EngineError> {
        match operation {
            RawAtomicOperation::Exchange => Ok(operand),
            _ => unsupported_atomic_operation(operation),
        }
    }
}

impl RawAtomicScalar for f32 {
    fn apply_atomic(
        operation: RawAtomicOperation,
        old: Self,
        operand: Self,
        ptx_space: PtxStateSpace,
    ) -> Result<Self, EngineError> {
        if !matches!(
            operation,
            RawAtomicOperation::Add | RawAtomicOperation::AddNoFtz
        ) {
            return unsupported_atomic_operation(operation);
        }
        Ok(
            if ptx_space == PtxStateSpace::Global && operation == RawAtomicOperation::Add {
                add_f32_ftz(old, operand, F32RoundingMode::Nearest)
            } else if operation == RawAtomicOperation::AddNoFtz {
                crate::scalar::add_f32(old, operand, F32RoundingMode::Nearest)
            } else {
                old + operand
            },
        )
    }
}

impl RawAtomicScalar for f64 {
    fn apply_atomic(
        operation: RawAtomicOperation,
        old: Self,
        operand: Self,
        _ptx_space: PtxStateSpace,
    ) -> Result<Self, EngineError> {
        match operation {
            RawAtomicOperation::Add => Ok(old + operand),
            _ => unsupported_atomic_operation(operation),
        }
    }
}

pub fn raw_atomic_update_physical_ptr_warp<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    operands: &WarpValue<T>,
    mask: WarpMask,
    ptx_space: PtxStateSpace,
    update: impl Fn(usize, T, T) -> Result<T, EngineError>,
) -> Result<WarpValue<T>, EngineError> {
    pointer.require_ptx_space_for_mask(ptx_space, mask)?;
    for lane in mask {
        if pointer.lane_physical_byte_offset(lane, T::BYTE_LEN)? % T::BYTE_LEN != 0 {
            return Err(EngineError::message(format!(
                "atomic operation requires {}-byte alignment on lane {lane}",
                T::BYTE_LEN
            )));
        }
    }
    let ordering = physical.ordering();
    let access_ranges = atomic_access_ranges(context, pointer, mask, T::BYTE_LEN)?;
    let atomic_access = ordering.begin_atomic_access(*context, access_ranges.clone())?;
    let mut previous = WarpValue::splat(T::zero());
    for lane in mask {
        let byte_offset = pointer.lane_read_write_byte_offset(lane, T::BYTE_LEN)?;
        if let RuntimeBuffer::Global(view) = pointer.buffer() {
            previous[lane] = physical.global().atomic_update_bytes(
                view,
                byte_offset,
                T::BYTE_LEN,
                |bytes| -> Result<(T, Vec<u8>), EngineError> {
                    let old = T::decode_le(bytes)?;
                    Ok((old, update(lane, old, operands[lane])?.encode_le()))
                },
            )?;
            continue;
        }
        let bytes = read_runtime_bytes_unordered(
            physical,
            context,
            pointer.buffer(),
            lane,
            byte_offset,
            T::BYTE_LEN,
        )?;
        let old = T::decode_le(&bytes)?;
        let new_value = update(lane, old, operands[lane])?;
        write_runtime_bytes(
            physical,
            context,
            pointer.buffer(),
            lane,
            byte_offset,
            &new_value.encode_le(),
        )?;
        previous[lane] = old;
    }
    if let Some(atomic_access) = atomic_access {
        atomic_access.mark_value_complete()?;
    }
    Ok(previous)
}

pub fn raw_atomic_scalar_physical_ptr_warp<T: RawAtomicScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    operands: &WarpValue<T>,
    mask: WarpMask,
    ptx_space: PtxStateSpace,
    operation: RawAtomicOperation,
) -> Result<WarpValue<T>, EngineError> {
    raw_atomic_update_physical_ptr_warp(
        physical,
        context,
        pointer,
        operands,
        mask,
        ptx_space,
        |_, old, operand| T::apply_atomic(operation, old, operand, ptx_space),
    )
}

pub fn raw_atomic_add_fp16_physical_ptr_warp(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    operands: &WarpValue<f32>,
    mask: WarpMask,
    ptx_space: PtxStateSpace,
) -> Result<WarpValue<f32>, EngineError> {
    let operand_bits = WarpValue::from_fn(|lane| f32_to_fp16_bits(operands[lane]));
    let previous = raw_atomic_update_physical_ptr_warp(
        physical,
        context,
        pointer,
        &operand_bits,
        mask,
        ptx_space,
        |_, old, operand| {
            Ok(f32_to_fp16_bits(
                fp16_bits_to_f32(old) + fp16_bits_to_f32(operand),
            ))
        },
    )?;
    Ok(WarpValue::from_fn(|lane| fp16_bits_to_f32(previous[lane])))
}

pub fn raw_atomic_add_bf16_physical_ptr_warp(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    operands: &WarpValue<f32>,
    mask: WarpMask,
    ptx_space: PtxStateSpace,
) -> Result<WarpValue<f32>, EngineError> {
    let operand_bits = WarpValue::from_fn(|lane| f32_to_bf16_bits(operands[lane]));
    let previous = raw_atomic_update_physical_ptr_warp(
        physical,
        context,
        pointer,
        &operand_bits,
        mask,
        ptx_space,
        |_, old, operand| {
            Ok(f32_to_bf16_bits(
                bf16_bits_to_f32(old) + bf16_bits_to_f32(operand),
            ))
        },
    )?;
    Ok(WarpValue::from_fn(|lane| bf16_bits_to_f32(previous[lane])))
}

fn raw_atomic_update_float_vector_physical_ptr_warp<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    operands: &WarpValue<T>,
    mask: WarpMask,
    ptx_space: PtxStateSpace,
    component_byte_len: usize,
    update_component: impl Fn(T, usize, &[u8]) -> Result<Vec<u8>, EngineError>,
) -> Result<WarpValue<T>, EngineError> {
    pointer.require_ptx_space_for_mask(ptx_space, mask)?;
    if component_byte_len == 0 || T::BYTE_LEN % component_byte_len != 0 {
        return Err(EngineError::message(format!(
            "packed atomic operand width {} is not divisible by component width {component_byte_len}",
            T::BYTE_LEN,
        )));
    }
    for lane in mask {
        if pointer.lane_physical_byte_offset(lane, T::BYTE_LEN)? % T::BYTE_LEN != 0 {
            return Err(EngineError::message(format!(
                "packed float atomic requires {}-byte alignment on lane {lane}",
                T::BYTE_LEN,
            )));
        }
    }

    let ordering = physical.ordering();
    let access_ranges = atomic_access_ranges(context, pointer, mask, T::BYTE_LEN)?;
    let atomic_access = ordering.begin_atomic_access(*context, access_ranges.clone())?;
    let mut previous = WarpValue::splat(T::zero());
    for lane in mask {
        let byte_offset = pointer.lane_read_write_byte_offset(lane, T::BYTE_LEN)?;
        let mut previous_bytes = vec![0_u8; T::BYTE_LEN];
        for component in 0..T::BYTE_LEN / component_byte_len {
            let component_offset = byte_offset + component * component_byte_len;
            let old_bytes = if let RuntimeBuffer::Global(view) = pointer.buffer() {
                physical.global().atomic_update_bytes(
                    view,
                    component_offset,
                    component_byte_len,
                    |bytes| -> Result<(Vec<u8>, Vec<u8>), EngineError> {
                        Ok((
                            bytes.to_vec(),
                            update_component(operands[lane], component, bytes)?,
                        ))
                    },
                )?
            } else {
                let bytes = read_runtime_bytes(
                    physical,
                    context,
                    pointer.buffer(),
                    lane,
                    component_offset,
                    component_byte_len,
                )?;
                let updated = update_component(operands[lane], component, &bytes)?;
                write_runtime_bytes(
                    physical,
                    context,
                    pointer.buffer(),
                    lane,
                    component_offset,
                    &updated,
                )?;
                bytes
            };
            let start = component * component_byte_len;
            previous_bytes[start..start + component_byte_len].copy_from_slice(&old_bytes);
        }
        previous[lane] = T::decode_le(&previous_bytes)?;
    }
    if let Some(atomic_access) = atomic_access {
        atomic_access.mark_value_complete()?;
    }
    Ok(previous)
}

pub fn raw_atomic_add_fp16x2_physical_ptr_warp(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    operands: &WarpValue<u32>,
    mask: WarpMask,
    ptx_space: PtxStateSpace,
) -> Result<WarpValue<u32>, EngineError> {
    raw_atomic_update_float_vector_physical_ptr_warp(
        physical,
        context,
        pointer,
        operands,
        mask,
        ptx_space,
        2,
        |operand, component, bytes| {
            let old =
                u16::from_le_bytes(bytes.try_into().map_err(|_| {
                    EngineError::message("float16x2 atomic component must be 2 bytes")
                })?);
            let addend = ((operand >> (component * 16)) & 0xffff_u32) as u16;
            Ok(
                f32_to_fp16_bits(fp16_bits_to_f32(old) + fp16_bits_to_f32(addend))
                    .to_le_bytes()
                    .to_vec(),
            )
        },
    )
}

/// Half-vector PTX atomics update each 16-bit element independently, including
/// when the instruction presents two elements in a packed 32-bit register.
pub fn raw_atomic_half_vector_physical_ptr_warp<T: RuntimeScalar, const BF16: bool>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    operands: &WarpValue<T>,
    mask: WarpMask,
    ptx_space: PtxStateSpace,
    operation: RawAtomicOperation,
) -> Result<WarpValue<T>, EngineError> {
    if ptx_space != PtxStateSpace::Global || !matches!(T::BYTE_LEN, 4 | 8 | 16) {
        return Err(EngineError::message(
            "half-vector atomics require global memory and 2/4/8 half elements",
        ));
    }
    raw_atomic_update_float_vector_physical_ptr_warp(
        physical,
        context,
        pointer,
        operands,
        mask,
        ptx_space,
        2,
        |operand, component, bytes| {
            let mut encoded = [0_u8; 16];
            operand.encode_le_into(&mut encoded[..T::BYTE_LEN]);
            let offset = component * 2;
            let operand = u16::from_le_bytes([encoded[offset], encoded[offset + 1]]);
            let old = u16::from_le_bytes(
                bytes
                    .try_into()
                    .map_err(|_| EngineError::message("half atomic component must be 2 bytes"))?,
            );
            let decode = if BF16 {
                bf16_bits_to_f32
            } else {
                fp16_bits_to_f32
            };
            let encode = if BF16 {
                f32_to_bf16_bits
            } else {
                f32_to_fp16_bits
            };
            let (old, operand) = (decode(old), decode(operand));
            let result = match operation {
                RawAtomicOperation::Add => old + operand,
                RawAtomicOperation::Minimum => {
                    crate::scalar::ptx_min_f32(old, operand, false, false)
                }
                RawAtomicOperation::Maximum => {
                    crate::scalar::ptx_max_f32(old, operand, false, false)
                }
                _ => return unsupported_atomic_operation(operation),
            };
            Ok(encode(result).to_le_bytes().to_vec())
        },
    )
}

pub fn raw_atomic_add_bf16x2_physical_ptr_warp(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    operands: &WarpValue<u32>,
    mask: WarpMask,
    ptx_space: PtxStateSpace,
) -> Result<WarpValue<u32>, EngineError> {
    raw_atomic_update_float_vector_physical_ptr_warp(
        physical,
        context,
        pointer,
        operands,
        mask,
        ptx_space,
        2,
        |operand, component, bytes| {
            let old = u16::from_le_bytes(bytes.try_into().map_err(|_| {
                EngineError::message("bfloat16x2 atomic component must be 2 bytes")
            })?);
            let addend = ((operand >> (component * 16)) & 0xffff_u32) as u16;
            Ok(
                f32_to_bf16_bits(bf16_bits_to_f32(old) + bf16_bits_to_f32(addend))
                    .to_le_bytes()
                    .to_vec(),
            )
        },
    )
}

pub fn raw_atomic_add_f32x2_physical_ptr_warp<const NOFTZ: bool>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    operands: &WarpValue<u64>,
    mask: WarpMask,
) -> Result<WarpValue<u64>, EngineError> {
    raw_atomic_update_float_vector_physical_ptr_warp(
        physical,
        context,
        pointer,
        operands,
        mask,
        PtxStateSpace::Global,
        4,
        |operand, component, bytes| {
            let old =
                f32::from_le_bytes(bytes.try_into().map_err(|_| {
                    EngineError::message("float32x2 atomic component must be 4 bytes")
                })?);
            let addend = f32::from_bits((operand >> (component * 32)) as u32);
            Ok(f32::apply_atomic(
                if NOFTZ {
                    RawAtomicOperation::AddNoFtz
                } else {
                    RawAtomicOperation::Add
                },
                old,
                addend,
                PtxStateSpace::Global,
            )?
            .to_le_bytes()
            .to_vec())
        },
    )
}

pub fn raw_atomic_add_f32x4_physical_ptr_warp<const NOFTZ: bool>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    operands: &WarpValue<F32x4>,
    mask: WarpMask,
) -> Result<WarpValue<F32x4>, EngineError> {
    raw_atomic_update_float_vector_physical_ptr_warp(
        physical,
        context,
        pointer,
        operands,
        mask,
        PtxStateSpace::Global,
        4,
        |operand, component, bytes| {
            let old =
                f32::from_le_bytes(bytes.try_into().map_err(|_| {
                    EngineError::message("float32x4 atomic component must be 4 bytes")
                })?);
            Ok(f32::apply_atomic(
                if NOFTZ {
                    RawAtomicOperation::AddNoFtz
                } else {
                    RawAtomicOperation::Add
                },
                old,
                operand[component],
                PtxStateSpace::Global,
            )?
            .to_le_bytes()
            .to_vec())
        },
    )
}

pub fn raw_atomic_cas_physical_ptr_warp<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointer: &PhysicalPtr,
    compares: &WarpValue<T>,
    values: &WarpValue<T>,
    mask: WarpMask,
    ptx_space: PtxStateSpace,
) -> Result<WarpValue<T>, EngineError> {
    pointer.require_ptx_space_for_mask(ptx_space, mask)?;
    for lane in mask {
        if pointer.lane_physical_byte_offset(lane, T::BYTE_LEN)? % T::BYTE_LEN != 0 {
            return Err(EngineError::message(format!(
                "atomic CAS requires {}-byte alignment on lane {lane}",
                T::BYTE_LEN
            )));
        }
    }
    let ordering = physical.ordering();
    let access_ranges = atomic_access_ranges(context, pointer, mask, T::BYTE_LEN)?;
    let atomic_access = ordering.begin_atomic_access(*context, access_ranges.clone())?;
    let mut previous = WarpValue::splat(T::zero());
    for lane in mask {
        let byte_offset = pointer.lane_read_write_byte_offset(lane, T::BYTE_LEN)?;
        if let RuntimeBuffer::Global(view) = pointer.buffer() {
            let mut compare_mismatch = None;
            let attempt =
                physical
                    .global()
                    .atomic_update_bytes(view, byte_offset, T::BYTE_LEN, |bytes| {
                        let old = T::decode_le(bytes)?;
                        if old.encode_le() != compares[lane].encode_le() {
                            compare_mismatch = Some(old);
                            return Err(EngineError::message("atomic CAS compare mismatch"));
                        }
                        Ok((old, values[lane].encode_le()))
                    });
            previous[lane] = match attempt {
                Ok(old) => old,
                Err(error) => match compare_mismatch {
                    Some(old) => old,
                    None => return Err(error),
                },
            };
            continue;
        }
        let bytes = read_runtime_bytes(
            physical,
            context,
            pointer.buffer(),
            lane,
            byte_offset,
            T::BYTE_LEN,
        )?;
        let old = T::decode_le(&bytes)?;
        if old.encode_le() == compares[lane].encode_le() {
            write_runtime_bytes(
                physical,
                context,
                pointer.buffer(),
                lane,
                byte_offset,
                &values[lane].encode_le(),
            )?;
        }
        previous[lane] = old;
    }
    if let Some(atomic_access) = atomic_access {
        atomic_access.mark_value_complete()?;
    }
    Ok(previous)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::physical_access::ProxyMemoryDomain;
    use crate::runtime::launch::{allocate_cta_shared, allocate_warp_private};
    use crate::{
        DeferredGlobalReduction, DynamicOpId, LaunchTopology, OperationContext, OperationKind,
        PhysicalAccessKind, PhysicalAccessSpace, PhysicalMemory, StaticOpId, WarpMask, WarpValue,
    };

    use super::{
        CompactAsyncCopyAccessPlan, RuntimeBuffer, defer_global_runtime_bytes,
        defer_global_runtime_masked_bytes, defer_global_runtime_reduction_bytes,
        element_byte_offset, fill_element_byte_offsets, load_scalar_warp,
        load_warp_private_scalar_at_thread, read_runtime_bytes, resolve_runtime_physical_access,
        validate_tma_multicast_mask, write_runtime_bytes, write_runtime_bytes_with_validity,
    };

    fn global_buffer(physical: &PhysicalMemory, value: u32) -> RuntimeBuffer {
        let allocation = physical
            .global()
            .allocate_from_bytes(value.to_le_bytes())
            .unwrap();
        RuntimeBuffer::Global(physical.global().full_view(allocation).unwrap())
    }

    fn global_buffer_from_values(physical: &PhysicalMemory, values: &[u32]) -> RuntimeBuffer {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let allocation = physical
            .global()
            .allocate_from_bytes_with_validity(bytes.clone(), vec![1_u8; bytes.len()])
            .unwrap();
        RuntimeBuffer::Global(physical.global().full_view(allocation).unwrap())
    }

    #[test]
    fn compact_async_copy_plan_unions_spans_and_retains_fragment_counts() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let physical = PhysicalMemory::new(topology);
        let source = global_buffer_from_values(&physical, &[10, 20]);
        let destination = RuntimeBuffer::Shared {
            allocations: Arc::new(allocate_cta_shared(&physical, topology, 8).unwrap()),
            byte_offset: 0,
            byte_len: 8,
            backing_byte_len: 8,
            virtual_base: 0,
        };
        let operation = OperationContext::new(
            DynamicOpId::new(0, 0, 0, StaticOpId::new(7), []),
            OperationKind::AsyncIssue,
            WarpMask::from_lanes([0]).unwrap(),
        );
        let mut plan = CompactAsyncCopyAccessPlan::default();
        for element in 0..2 {
            plan.plan_element(
                &operation,
                &context,
                4,
                &source,
                element,
                true,
                &destination,
                element,
                true,
                None,
                None,
                false,
                0,
            )
            .unwrap();
        }

        let (batches, delivered) = plan.finish(&operation, 0).unwrap();

        assert_eq!(delivered, 8);
        assert_eq!(batches.len(), 2);
        for batch in &batches {
            assert_eq!(batch.semantic_access_count(), 2);
            let footprint = batch.lane(0).unwrap().footprint();
            assert_eq!(footprint.byte_len(), 8);
            assert_eq!(footprint.spans().len(), 1);
        }
        assert_eq!(batches[0].descriptor().kind(), PhysicalAccessKind::Read);
        assert_eq!(batches[0].descriptor().space(), PhysicalAccessSpace::Global);
        assert_eq!(
            batches[0].descriptor().proxy_memory_domain(),
            ProxyMemoryDomain::Global,
        );
        assert_eq!(batches[1].descriptor().kind(), PhysicalAccessKind::Write);
        assert_eq!(batches[1].descriptor().space(), PhysicalAccessSpace::Shared);
        assert_eq!(
            batches[1].descriptor().proxy_memory_domain(),
            ProxyMemoryDomain::SharedCluster,
        );
    }

    #[test]
    fn lane_selected_global_buffers_preserve_deferred_and_validity_operations() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let physical = PhysicalMemory::new(topology);
        let left = global_buffer(&physical, 10);
        let right = global_buffer(&physical, 20);
        let selected = RuntimeBuffer::LaneSelected {
            buffers: WarpValue::from_fn(|lane| {
                Arc::new(if lane % 2 == 0 {
                    left.clone()
                } else {
                    right.clone()
                })
            }),
        };

        defer_global_runtime_bytes(&physical, &selected, 0, 0, 11_u32.to_le_bytes().to_vec())
            .unwrap()
            .publish()
            .unwrap();
        defer_global_runtime_masked_bytes(
            &physical,
            &selected,
            1,
            0,
            5_u32.to_le_bytes().to_vec(),
            vec![0xff, 0, 0, 0],
        )
        .unwrap()
        .publish()
        .unwrap();
        defer_global_runtime_reduction_bytes(
            &physical,
            &selected,
            1,
            0,
            7_u32.to_le_bytes().to_vec(),
            DeferredGlobalReduction::AddU32,
        )
        .unwrap()
        .publish()
        .unwrap();

        assert_eq!(
            u32::from_le_bytes(
                read_runtime_bytes(&physical, &context, &selected, 0, 0, 4)
                    .unwrap()
                    .try_into()
                    .unwrap()
            ),
            11
        );
        assert_eq!(
            u32::from_le_bytes(
                read_runtime_bytes(&physical, &context, &selected, 1, 0, 4)
                    .unwrap()
                    .try_into()
                    .unwrap()
            ),
            12
        );

        write_runtime_bytes_with_validity(
            &physical,
            &context,
            &selected,
            1,
            0,
            &13_u32.to_le_bytes(),
            &[true, false, true, true],
        )
        .unwrap();
        assert!(read_runtime_bytes(&physical, &context, &selected, 1, 0, 4)
            .unwrap_err()
            .to_string()
            .contains("invalid byte"));
    }

    #[test]
    fn tma_member_mask_is_nonempty_in_cluster_and_32_bit() {
        assert!(validate_tma_multicast_mask(0b11, 2).is_ok());
        assert!(validate_tma_multicast_mask(0, 2).is_err());
        assert!(validate_tma_multicast_mask(0b100, 2).is_err());
        assert!(validate_tma_multicast_mask(1_u64 << 16, 64).is_ok());
        assert!(validate_tma_multicast_mask(1_u64 << 32, 64).is_err());
    }

    #[test]
    fn global_warp_load_preserves_uniform_and_per_lane_results() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let physical = PhysicalMemory::new(topology);
        let buffer = global_buffer_from_values(&physical, &[11, 29]);
        let mask = WarpMask::from_lanes([0, 1]).unwrap();

        let uniform =
            load_scalar_warp::<u32>(&physical, &context, &buffer, &WarpValue::splat(1_i64), mask)
                .unwrap();
        assert_eq!((uniform[0], uniform[1]), (29, 29));

        let indices = WarpValue::from_fn(|lane| (lane % 2) as i64);
        let per_lane =
            load_scalar_warp::<u32>(&physical, &context, &buffer, &indices, mask).unwrap();
        assert_eq!((per_lane[0], per_lane[1]), (11, 29));

        let negative = load_scalar_warp::<u32>(
            &physical,
            &context,
            &buffer,
            &WarpValue::splat(-1_i64),
            WarpMask::from_lanes([0]).unwrap(),
        )
        .unwrap_err();
        assert!(negative.is_out_of_bounds());
        assert!(negative.to_string().contains("negative buffer index -1"));

        let beyond_end = load_scalar_warp::<u32>(
            &physical,
            &context,
            &buffer,
            &WarpValue::splat(2_i64),
            WarpMask::from_lanes([0]).unwrap(),
        )
        .unwrap_err();
        assert!(beyond_end.is_out_of_bounds());
        assert!(beyond_end.to_string().contains("outside 8 bytes"));

        let overflow = element_byte_offset(&buffer, usize::MAX, 4, 0).unwrap_err();
        assert!(overflow.is_out_of_bounds());
        assert!(overflow.to_string().contains("element offset overflow"));

        let restricted = RuntimeBuffer::AccessView {
            buffer: Arc::new(buffer),
            readable_lanes: WarpMask::from_lanes([0]).unwrap(),
            writable_lanes: WarpMask::EMPTY,
        };
        assert!(load_scalar_warp::<u32>(
            &physical,
            &context,
            &restricted,
            &WarpValue::splat(0_i64),
            mask,
        )
        .unwrap_err()
        .to_string()
        .contains("non-readable DeclBuffer view"));
    }

    #[test]
    fn fused_element_offsets_validate_only_active_lanes() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let buffer = global_buffer_from_values(&physical, &[11, 29]);
        let indices = WarpValue::from_fn(|lane| match lane {
            0 => 1,
            1 => -1,
            _ => i64::MAX,
        });
        let mut offsets = WarpValue::splat(usize::MAX);

        fill_element_byte_offsets(
            &buffer,
            &indices,
            4,
            WarpMask::from_lanes([0]).unwrap(),
            &mut offsets,
        )
        .unwrap();
        assert_eq!(offsets[0], 4);
        assert_eq!(offsets[1], usize::MAX);
        assert_eq!(offsets[31], usize::MAX);

        let negative = fill_element_byte_offsets(
            &buffer,
            &indices,
            4,
            WarpMask::from_lanes([0, 1]).unwrap(),
            &mut offsets,
        )
        .unwrap_err();
        assert!(negative.is_out_of_bounds());
        assert!(negative.to_string().contains("negative buffer index -1"));

        let beyond_end = fill_element_byte_offsets(
            &buffer,
            &WarpValue::splat(2_i64),
            4,
            WarpMask::from_lanes([0]).unwrap(),
            &mut offsets,
        )
        .unwrap_err();
        assert!(beyond_end.is_out_of_bounds());
        assert!(beyond_end.to_string().contains("outside 8 bytes"));
    }

    #[test]
    fn warp_private_owner_transport_preserves_views_permissions_and_initialization() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let contexts = topology.warp_contexts().collect::<Vec<_>>();
        let physical = PhysicalMemory::new(topology);
        let allocations = Arc::new(allocate_warp_private(physical.local(), topology, 8).unwrap());
        let local = RuntimeBuffer::Local {
            allocations,
            byte_offset: 4,
            byte_len: 4,
        };
        write_runtime_bytes(&physical, &contexts[1], &local, 7, 0, &37_u32.to_le_bytes()).unwrap();

        assert_eq!(
            load_warp_private_scalar_at_thread::<u32>(&physical, &contexts[0], &local, 1, 0, 7,)
                .unwrap(),
            37
        );
        assert!(load_warp_private_scalar_at_thread::<u32>(
            &physical,
            &contexts[0],
            &local,
            1,
            1,
            7,
        )
        .unwrap_err()
        .to_string()
        .contains("outside 4 bytes"));
        assert!(load_warp_private_scalar_at_thread::<u32>(
            &physical,
            &contexts[0],
            &local,
            1,
            0,
            6,
        )
        .unwrap_err()
        .to_string()
        .contains("includes invalid byte"));

        let denied = RuntimeBuffer::AccessView {
            buffer: Arc::new(local),
            readable_lanes: WarpMask::EMPTY,
            writable_lanes: WarpMask::EMPTY,
        };
        assert!(load_warp_private_scalar_at_thread::<u32>(
            &physical,
            &contexts[0],
            &denied,
            1,
            0,
            7,
        )
        .unwrap_err()
        .to_string()
        .contains("non-readable DeclBuffer view"));
    }

    #[test]
    fn physical_resolver_collapses_distinct_global_views_to_one_byte_identity() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocation = physical.global().allocate_zeroed(16).unwrap();
        let full = physical.global().full_view(allocation).unwrap();
        let alias = physical.global().subview(&full, 4, 8).unwrap();

        let through_full = resolve_runtime_physical_access(
            &context,
            &RuntimeBuffer::Global(full),
            0,
            4,
            4,
            PhysicalAccessKind::Read,
        )
        .unwrap();
        let through_alias = resolve_runtime_physical_access(
            &context,
            &RuntimeBuffer::Global(alias),
            0,
            0,
            4,
            PhysicalAccessKind::Read,
        )
        .unwrap();

        assert_eq!(through_full, through_alias);
        assert_eq!(through_full.space(), PhysicalAccessSpace::Global);
        assert_eq!(through_full.span().byte_offset(), 4);
        assert_eq!(through_full.span().byte_len(), 4);
    }

    #[test]
    fn physical_resolver_uses_target_allocation_for_remote_shared_aliases() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let contexts = topology.warp_contexts().collect::<Vec<_>>();
        let physical = PhysicalMemory::new(topology);
        let allocations =
            Arc::new(crate::runtime::allocate_cta_shared(&physical, topology, 16).unwrap());
        let local = RuntimeBuffer::Shared {
            allocations: Arc::clone(&allocations),
            byte_offset: 0,
            byte_len: 16,
            backing_byte_len: 16,
            virtual_base: 0,
        };
        let remote = RuntimeBuffer::RemoteShared {
            allocations,
            byte_offsets: WarpValue::splat(0_i64),
            byte_len: 16,
            target_cta_ids: WarpValue::splat(1_i64),
            virtual_base: 0,
        };

        let remote_access = resolve_runtime_physical_access(
            &contexts[0],
            &remote,
            0,
            4,
            4,
            PhysicalAccessKind::Write,
        )
        .unwrap();
        let owner_access = resolve_runtime_physical_access(
            &contexts[1],
            &local,
            0,
            4,
            4,
            PhysicalAccessKind::Read,
        )
        .unwrap();

        assert_eq!(remote_access.span(), owner_access.span());
        assert_eq!(remote_access.space(), owner_access.space());
        assert_eq!(
            remote_access.proxy_memory_domain(),
            ProxyMemoryDomain::SharedCluster,
        );
        assert_eq!(
            owner_access.proxy_memory_domain(),
            ProxyMemoryDomain::SharedCta,
        );
        assert_eq!(remote_access.space(), PhysicalAccessSpace::Shared);
    }

    #[test]
    fn physical_resolver_keeps_cta_and_lane_private_allocations_distinct() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let contexts = topology.warp_contexts().collect::<Vec<_>>();
        let physical = PhysicalMemory::new(topology);
        let shared_allocations =
            Arc::new(crate::runtime::allocate_cta_shared(&physical, topology, 8).unwrap());
        let shared = RuntimeBuffer::Shared {
            allocations: shared_allocations,
            byte_offset: 0,
            byte_len: 8,
            backing_byte_len: 8,
            virtual_base: 0,
        };
        let first = resolve_runtime_physical_access(
            &contexts[0],
            &shared,
            0,
            0,
            4,
            PhysicalAccessKind::Read,
        )
        .unwrap();
        let second = resolve_runtime_physical_access(
            &contexts[1],
            &shared,
            0,
            0,
            4,
            PhysicalAccessKind::Read,
        )
        .unwrap();
        assert_ne!(first.span().allocation(), second.span().allocation());

        let private = RuntimeBuffer::Local {
            allocations: Arc::new(
                crate::runtime::allocate_warp_private(physical.local(), topology, 8).unwrap(),
            ),
            byte_offset: 0,
            byte_len: 8,
        };
        let lane0 = resolve_runtime_physical_access(
            &contexts[0],
            &private,
            0,
            0,
            4,
            PhysicalAccessKind::Read,
        )
        .unwrap();
        let lane1 = resolve_runtime_physical_access(
            &contexts[0],
            &private,
            1,
            0,
            4,
            PhysicalAccessKind::Read,
        )
        .unwrap();
        assert_eq!(lane0.span().allocation(), lane1.span().allocation());
        assert_eq!(lane0.span().byte_offset(), 0);
        assert_eq!(lane1.span().byte_offset(), 8);
    }

    #[test]
    fn physical_resolver_rejects_permissions_and_bounds_before_batch_commit() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let physical = PhysicalMemory::new(topology);
        let base = Arc::new(global_buffer(&physical, 17));
        let read_only = RuntimeBuffer::AccessView {
            buffer: base,
            readable_lanes: WarpMask::from_lanes([0]).unwrap(),
            writable_lanes: WarpMask::EMPTY,
        };

        let permission = resolve_runtime_physical_access(
            &context,
            &read_only,
            0,
            0,
            4,
            PhysicalAccessKind::Write,
        )
        .unwrap_err();
        assert!(permission.to_string().contains("non-writable"));

        let bounds = resolve_runtime_physical_access(
            &context,
            &read_only,
            0,
            2,
            4,
            PhysicalAccessKind::Read,
        )
        .unwrap_err();
        assert!(bounds.is_out_of_bounds());
        assert!(bounds.to_string().contains("outside 4 bytes"));
    }
}
