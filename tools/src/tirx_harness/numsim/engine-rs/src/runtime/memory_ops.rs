use crate::physical_access::ProxyMemoryDomain;
use crate::{
    DeferredGlobalReduction, DeferredGlobalWrite, DiagnosticLabel, EngineError,
    MemoryAccessSemantics, MemoryScope, OperationContext, PhysicalAccessBatch,
    PhysicalAccessBatchError, PhysicalAccessDescriptor, PhysicalAccessKind, PhysicalAccessSpace,
    PhysicalMemory, WarpContext, WarpMask, WarpValue, WARP_SIZE,
};

use super::io::{resolve_shared_runtime_physical_access_to_cta, ResolvedRuntimePhysicalAccess};
use super::{
    defer_global_runtime_bytes, defer_global_runtime_masked_bytes,
    defer_global_runtime_reduction_bytes, read_runtime_bytes, resolve_runtime_physical_access,
    single_lane_physical_access_batch, write_runtime_bytes, write_runtime_bytes_with_validity,
    write_shared_runtime_bytes_to_cta, PhysicalPtr, PointerSpace, PtxStateSpace, RuntimeBuffer,
};

pub(crate) fn raw_ldmatrix_b16_fragments(
    physical: &PhysicalMemory,
    context: &WarpContext,
    source: &PhysicalPtr,
    source_element_offsets: &WarpValue<i64>,
    source_itemsize: usize,
    matrix_count: usize,
    transpose: bool,
    mask: WarpMask,
) -> Result<Vec<WarpValue<u32>>, EngineError> {
    if !matches!(matrix_count, 1 | 2 | 4) {
        return Err(EngineError::message(format!(
            "ldmatrix b16 matrix count must be 1, 2, or 4, got {matrix_count}"
        )));
    }
    super::require_full_warp_sync(mask, "ldmatrix.sync.aligned.m8n8.shared.b16")?;
    source.require_ptx_space_for_mask(PtxStateSpace::Shared, mask)?;
    if !matches!(source.buffer(), RuntimeBuffer::Shared { .. }) {
        return Err(EngineError::message(
            "ldmatrix source must address local CTA shared memory",
        ));
    }

    let source_base = |lane: usize| -> Result<usize, EngineError> {
        usize::try_from(source_element_offsets[lane])
            .map_err(|_| EngineError::message("negative ldmatrix source element offset"))?
            .checked_mul(source_itemsize)
            .ok_or_else(|| EngineError::message("ldmatrix source byte offset overflow"))
    };
    let mut fragments = vec![WarpValue::splat(0_u32); matrix_count];
    for (matrix, matrix_fragments) in fragments.iter_mut().enumerate() {
        for (lane, fragment_value) in matrix_fragments.lanes_mut().iter_mut().enumerate() {
            let row = lane / 4;
            let fragment = lane % 4;
            if transpose {
                let source_lane0 = matrix * 8 + fragment * 2;
                let source_lane1 = source_lane0 + 1;
                let column_offset = row * 2;
                let relative0 = source_base(source_lane0)?
                    .checked_add(column_offset)
                    .ok_or_else(|| EngineError::message("ldmatrix source offset overflow"))?;
                let relative1 = source_base(source_lane1)?
                    .checked_add(column_offset)
                    .ok_or_else(|| EngineError::message("ldmatrix source offset overflow"))?;
                let offset0 = source.lane_read_byte_offset_at(source_lane0, relative0, 2)?;
                let offset1 = source.lane_read_byte_offset_at(source_lane1, relative1, 2)?;
                let low = read_runtime_bytes(
                    physical,
                    context,
                    source.buffer(),
                    source_lane0,
                    offset0,
                    2,
                )?;
                let high = read_runtime_bytes(
                    physical,
                    context,
                    source.buffer(),
                    source_lane1,
                    offset1,
                    2,
                )?;
                *fragment_value = u32::from_le_bytes([low[0], low[1], high[0], high[1]]);
            } else {
                let address_lane = matrix * 8 + row;
                let relative = source_base(address_lane)?
                    .checked_add(fragment * 4)
                    .ok_or_else(|| EngineError::message("ldmatrix source offset overflow"))?;
                let byte_offset = source.lane_read_byte_offset_at(address_lane, relative, 4)?;
                let bytes = read_runtime_bytes(
                    physical,
                    context,
                    source.buffer(),
                    address_lane,
                    byte_offset,
                    4,
                )?;
                *fragment_value = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            }
        }
    }
    Ok(fragments)
}

/// One output byte's source span. Both execution and Racecheck use this
/// mapping; padding after each packed row is never read.
fn ldmatrix_b8_element(
    source: &PhysicalPtr,
    register: usize,
    lane: usize,
    element: usize,
    transpose: bool,
    source_bits: usize,
) -> Result<(usize, usize, usize, usize), EngineError> {
    let (provider, column) = if transpose {
        (
            register / 2 * 16 + lane % 4 * 4 + element,
            register % 2 * 8 + lane / 4,
        )
    } else {
        (register * 8 + lane / 4, lane % 4 * 4 + element)
    };
    let bit = column * source_bits;
    let shift = bit % 8;
    let width = (shift + source_bits).div_ceil(8);
    let row_base = source.lane_read_byte_offset_at(provider, 0, 0)?;
    if row_base % 16 != 0 {
        return Err(EngineError::message(
            "ldmatrix row address must be 16-byte aligned",
        ));
    }
    let offset = source.lane_read_byte_offset_at(provider, bit / 8, width)?;
    Ok((provider, offset, width, shift))
}

pub(crate) fn raw_ldmatrix_b8_fragments(
    physical: &PhysicalMemory,
    context: &WarpContext,
    source: &PhysicalPtr,
    register_count: usize,
    transpose: bool,
    source_bits: usize,
) -> Result<Vec<WarpValue<u32>>, EngineError> {
    source.require_ptx_space_for_mask(PtxStateSpace::Shared, context.active_mask())?;
    if !matches!(source.buffer(), RuntimeBuffer::Shared { .. }) {
        return Err(EngineError::message(
            "ldmatrix source must address local CTA shared memory",
        ));
    }
    let mut fragments = vec![WarpValue::splat(0_u32); register_count];
    for (register, fragment) in fragments.iter_mut().enumerate() {
        for (lane, value) in fragment.lanes_mut().iter_mut().enumerate() {
            for element in 0..4 {
                let (provider, offset, width, shift) =
                    ldmatrix_b8_element(source, register, lane, element, transpose, source_bits)?;
                let bytes = read_runtime_bytes(
                    physical,
                    context,
                    source.buffer(),
                    provider,
                    offset,
                    width,
                )?;
                let packed = u16::from(bytes[0]) | (u16::from(*bytes.get(1).unwrap_or(&0)) << 8);
                *value |= u32::from((packed >> shift) & ((1 << source_bits) - 1)) << (element * 8);
            }
        }
    }
    Ok(fragments)
}

pub(crate) fn plan_raw_ldmatrix_access(
    operation: &OperationContext,
    context: &WarpContext,
    source: &PhysicalPtr,
    register_count: usize,
    transpose: bool,
    source_bits: usize,
    mask: WarpMask,
) -> Result<PhysicalAccessBatch, EngineError> {
    if !matches!(register_count, 1 | 2 | 4) {
        return Err(EngineError::message(format!(
            "ldmatrix fragment count must be 1, 2, or 4, got {register_count}"
        )));
    }
    super::require_full_warp_sync(mask, "ldmatrix.sync.aligned")?;
    source.require_ptx_space_for_mask(PtxStateSpace::Shared, mask)?;
    if !matches!(source.buffer(), RuntimeBuffer::Shared { .. }) {
        return Err(EngineError::message(
            "ldmatrix source must address local CTA shared memory",
        ));
    }
    let descriptor = PhysicalAccessDescriptor::new(
        PhysicalAccessKind::Read,
        PhysicalAccessSpace::Shared,
        register_count * 4,
    )
    .map_err(|error| EngineError::message(error.to_string()))?;
    let resolve_lane = |provenance: &crate::LaneProvenance| {
        let consumer_lane = provenance.lane();
        let row = consumer_lane / 4;
        let fragment = consumer_lane % 4;
        let mut spans = Vec::with_capacity(if source_bits != 16 {
            register_count * 4
        } else if transpose {
            register_count * 2
        } else {
            register_count
        });
        if source_bits != 16 {
            for register in 0..register_count {
                for element in 0..4 {
                    let (provider, offset, width, _) = ldmatrix_b8_element(
                        source,
                        register,
                        consumer_lane,
                        element,
                        transpose,
                        source_bits,
                    )?;
                    spans.push(
                        resolve_runtime_physical_access(
                            context,
                            source.buffer(),
                            provider,
                            offset,
                            width,
                            PhysicalAccessKind::Read,
                        )?
                        .span(),
                    );
                }
            }
            return Ok(spans);
        }
        for matrix in 0..register_count {
            if transpose {
                let source_lane0 = matrix * 8 + fragment * 2;
                let source_lane1 = source_lane0 + 1;
                let column_offset = row * 2;
                for source_lane in [source_lane0, source_lane1] {
                    let byte_offset =
                        source.lane_read_byte_offset_at(source_lane, column_offset, 2)?;
                    spans.push(
                        resolve_runtime_physical_access(
                            context,
                            source.buffer(),
                            source_lane,
                            byte_offset,
                            2,
                            PhysicalAccessKind::Read,
                        )?
                        .span(),
                    );
                }
            } else {
                let address_lane = matrix * 8 + row;
                let byte_offset = source.lane_read_byte_offset_at(address_lane, fragment * 4, 4)?;
                spans.push(
                    resolve_runtime_physical_access(
                        context,
                        source.buffer(),
                        address_lane,
                        byte_offset,
                        4,
                        PhysicalAccessKind::Read,
                    )?
                    .span(),
                );
            }
        }
        Ok(spans)
    };
    // Several packed elements can share one physical byte; 6-bit transposed
    // elements can straddle bytes. Retain their union as one lane event.
    let batch = if source_bits == 16 {
        PhysicalAccessBatch::resolve(operation.clone(), descriptor, resolve_lane)
    } else {
        PhysicalAccessBatch::resolve_lane_unions(
            operation.clone(),
            PhysicalAccessKind::Read,
            PhysicalAccessSpace::Shared,
            resolve_lane,
        )
    };
    batch.map_err(|error| match error {
        PhysicalAccessBatchError::LaneResolution { source, .. } => source,
        other => EngineError::message(other.to_string()),
    })
}

fn raw_stmatrix_write(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &PhysicalPtr,
    provider_lane: usize,
    byte_delta: usize,
    bytes: &[u8],
) -> Result<(), EngineError> {
    destination.require_ptx_space_for_mask(
        PtxStateSpace::Shared,
        WarpMask::from_bits(1_u32 << provider_lane),
    )?;
    let offset = destination.lane_write_byte_offset_at(provider_lane, byte_delta, bytes.len())?;
    write_runtime_bytes(
        physical,
        context,
        destination.buffer(),
        provider_lane,
        offset,
        bytes,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StmatrixDescriptor {
    M8n8B16 {
        transpose: bool,
        space: PtxStateSpace,
    },
    M16n8B8Transposed {
        space: PtxStateSpace,
    },
}

impl StmatrixDescriptor {
    pub(crate) const fn state_space(self) -> PtxStateSpace {
        match self {
            Self::M8n8B16 { space, .. } | Self::M16n8B8Transposed { space } => space,
        }
    }
}

pub(crate) fn plan_raw_stmatrix_access(
    operation: &OperationContext,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source_count: usize,
    stmatrix: StmatrixDescriptor,
) -> Result<PhysicalAccessBatch, EngineError> {
    if !matches!(source_count, 1 | 2 | 4) {
        return Err(EngineError::message(format!(
            "stmatrix source count must be 1, 2, or 4, got {source_count}"
        )));
    }
    super::require_full_warp_sync(context.active_mask(), "stmatrix.sync.aligned")?;
    destination.require_ptx_space_for_mask(stmatrix.state_space(), context.active_mask())?;
    let descriptor = PhysicalAccessDescriptor::new(
        PhysicalAccessKind::Write,
        PhysicalAccessSpace::Shared,
        source_count * 4,
    )
    .map_err(|error| EngineError::message(error.to_string()))?;
    PhysicalAccessBatch::resolve(operation.clone(), descriptor, |provenance| {
        let source_lane = provenance.lane();
        let mut spans = Vec::with_capacity(source_count * 4);
        match stmatrix {
            StmatrixDescriptor::M8n8B16 { transpose, .. } => {
                for matrix in 0..source_count {
                    for half_index in 0..2_usize {
                        let (row, column) = if transpose {
                            (2 * (source_lane % 4) + half_index, source_lane / 4)
                        } else {
                            (source_lane / 4, 2 * (source_lane % 4) + half_index)
                        };
                        let provider_lane = matrix * 8 + row;
                        let byte_offset =
                            destination.lane_write_byte_offset_at(provider_lane, column * 2, 2)?;
                        spans.push(
                            resolve_runtime_physical_access(
                                context,
                                destination.buffer(),
                                provider_lane,
                                byte_offset,
                                2,
                                PhysicalAccessKind::Write,
                            )?
                            .span(),
                        );
                    }
                }
            }
            StmatrixDescriptor::M16n8B8Transposed { .. } => {
                for matrix in 0..source_count {
                    for byte_index in 0..4_usize {
                        let row = 2 * (source_lane % 4) + (byte_index % 2);
                        let column = source_lane / 4 + 8 * (byte_index / 2);
                        let provider_lane = matrix * 8 + row;
                        let byte_offset =
                            destination.lane_write_byte_offset_at(provider_lane, column, 1)?;
                        spans.push(
                            resolve_runtime_physical_access(
                                context,
                                destination.buffer(),
                                provider_lane,
                                byte_offset,
                                1,
                                PhysicalAccessKind::Write,
                            )?
                            .span(),
                        );
                    }
                }
            }
        }
        Ok(spans)
    })
    .map_err(|error| match error {
        PhysicalAccessBatchError::LaneResolution { source, .. } => source,
        other => EngineError::message(other.to_string()),
    })
}

pub fn raw_stmatrix(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &PhysicalPtr,
    sources: &[&WarpValue<u32>],
    descriptor: StmatrixDescriptor,
) -> Result<(), EngineError> {
    if !matches!(sources.len(), 1 | 2 | 4) {
        return Err(EngineError::message(format!(
            "stmatrix source count must be 1, 2, or 4, got {}",
            sources.len()
        )));
    }
    let ptx_space = descriptor.state_space();
    if !matches!(ptx_space, PtxStateSpace::Shared | PtxStateSpace::SharedCta) {
        return Err(EngineError::message(format!(
            "stmatrix state space must be shared or shared::cta, got {ptx_space}"
        )));
    }
    destination.require_ptx_space_for_mask(ptx_space, context.active_mask())?;
    if !matches!(destination.buffer(), RuntimeBuffer::Shared { .. }) {
        return Err(EngineError::message(
            "stmatrix destination must address local CTA shared memory",
        ));
    }
    super::require_full_warp_sync(context.active_mask(), "stmatrix.sync.aligned")?;
    for provider_lane in 0..sources.len() * 8 {
        let address = destination.lane_physical_byte_offset(provider_lane, 16)?;
        if address % 16 != 0 {
            return Err(EngineError::message(format!(
                "stmatrix row address requires 16-byte alignment on lane {provider_lane}"
            )));
        }
    }

    match descriptor {
        StmatrixDescriptor::M8n8B16 { transpose, .. } => {
            for (matrix, source) in sources.iter().enumerate() {
                for source_lane in 0..WARP_SIZE {
                    let register = source[source_lane];
                    for half_index in 0..2_usize {
                        let (row, column) = if transpose {
                            (2 * (source_lane % 4) + half_index, source_lane / 4)
                        } else {
                            (source_lane / 4, 2 * (source_lane % 4) + half_index)
                        };
                        let value = (register >> (half_index * 16)) as u16;
                        raw_stmatrix_write(
                            physical,
                            context,
                            destination,
                            matrix * 8 + row,
                            column * 2,
                            &value.to_le_bytes(),
                        )?;
                    }
                }
            }
        }
        StmatrixDescriptor::M16n8B8Transposed { .. } => {
            for (matrix, source) in sources.iter().enumerate() {
                for source_lane in 0..WARP_SIZE {
                    let register = source[source_lane];
                    for byte_index in 0..4_usize {
                        let row = 2 * (source_lane % 4) + (byte_index % 2);
                        let column = source_lane / 4 + 8 * (byte_index / 2);
                        let byte = [(register >> (byte_index * 8)) as u8];
                        raw_stmatrix_write(
                            physical,
                            context,
                            destination,
                            matrix * 8 + row,
                            column,
                            &byte,
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn single_issuing_lane(mask: WarpMask, operation: &str) -> Result<usize, EngineError> {
    if mask.len() != 1 {
        return Err(EngineError::message(format!(
            "{operation} requires exactly one active issuing lane, got {}",
            mask.len(),
        )));
    }
    mask.first_active()
        .ok_or_else(|| EngineError::message(format!("{operation} has no issuing lane")))
}
fn st_bulk_byte_count(value: i64, lane: usize) -> Result<usize, EngineError> {
    let byte_len = usize::try_from(value).map_err(|_| {
        EngineError::message(format!("st.bulk byte count is negative on lane {lane}"))
    })?;
    if byte_len % 8 != 0 || byte_len > 16_777_216 {
        return Err(EngineError::message(format!(
            "st.bulk byte count {byte_len} must be a multiple of 8 with maximum 16777216 on lane {lane}"
        )));
    }
    Ok(byte_len)
}


pub fn raw_st_bulk_zero(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &PhysicalPtr,
    num_bytes: &WarpValue<i64>,
    mask: WarpMask,
    ptx_space: PtxStateSpace,
) -> Result<(), EngineError> {
    if ptx_space == PtxStateSpace::Generic
        && !matches!(
            destination.pointer_space_for_mask(mask)?,
            PointerSpace::Shared
        )
    {
        return Err(EngineError::message(
            "generic st.bulk address must resolve to shared memory",
        ));
    }
    destination.require_ptx_space_for_mask(ptx_space, mask)?;
    for lane in mask {
        let byte_len = st_bulk_byte_count(num_bytes[lane], lane)?;
        if byte_len == 0 {
            continue;
        }
        let byte_offset = destination.lane_write_byte_offset(lane, byte_len)?;
        let bytes = vec![0_u8; byte_len];
        write_runtime_bytes(
            physical,
            context,
            destination.buffer(),
            lane,
            byte_offset,
            &bytes,
        )?;
    }
    Ok(())
}

pub(crate) fn plan_raw_st_bulk_zero_access(
    operation: &OperationContext,
    context: &WarpContext,
    destination: &PhysicalPtr,
    num_bytes: &WarpValue<i64>,
    mask: WarpMask,
    ptx_space: PtxStateSpace,
) -> Result<PhysicalAccessBatch, EngineError> {
    if operation.active_mask() != mask {
        return Err(EngineError::message(format!(
            "st.bulk operation mask {:#010x} does not match execution mask {:#010x}",
            operation.active_mask().bits(),
            mask.bits(),
        )));
    }
    if ptx_space == PtxStateSpace::Generic
        && !matches!(
            destination.pointer_space_for_mask(mask)?,
            PointerSpace::Shared
        )
    {
        return Err(EngineError::message(
            "generic st.bulk address must resolve to shared memory",
        ));
    }
    destination.require_ptx_space_for_mask(ptx_space, mask)?;
    PhysicalAccessBatch::resolve_lane_widths(
        operation.clone(),
        PhysicalAccessKind::Write,
        PhysicalAccessSpace::Shared,
        |provenance| {
            let lane = provenance.lane();
            let byte_len = st_bulk_byte_count(num_bytes[lane], lane)?;
            let byte_offset = destination.lane_write_byte_offset(lane, byte_len)?;
            Ok(vec![resolve_runtime_physical_access(
                context,
                destination.buffer(),
                lane,
                byte_offset,
                byte_len,
                PhysicalAccessKind::Write,
            )?
            .span()])
        },
    )
    .map(|batch| {
        batch.with_proxy_memory_domain(match ptx_space {
            PtxStateSpace::SharedCluster => ProxyMemoryDomain::SharedCluster,
            _ => ProxyMemoryDomain::SharedCta,
        })
    })
    .map_err(|error| match error {
        PhysicalAccessBatchError::LaneResolution { source, .. } => source,
        other => EngineError::message(other.to_string()),
    })
}

fn bulk_byte_len(
    num_bytes: &WarpValue<i64>,
    lane: usize,
    operation: &DiagnosticLabel,
) -> Result<usize, EngineError> {
    let byte_len = usize::try_from(num_bytes[lane]).map_err(|_| {
        operation.engine_error(format_args!(" byte count is negative on lane {lane}"))
    })?;
    if byte_len == 0 || byte_len % 16 != 0 {
        return Err(operation.engine_error(format_args!(
            " byte count {byte_len} must be a positive multiple of 16 on lane {lane}"
        )));
    }
    Ok(byte_len)
}

#[allow(clippy::too_many_arguments)]
fn raw_bulk_copy_payloads(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: &WarpValue<i64>,
    mask: WarpMask,
    destination_space: PtxStateSpace,
    source_space: PtxStateSpace,
    operation: &DiagnosticLabel,
) -> Result<Vec<(usize, usize, Vec<u8>)>, EngineError> {
    destination.require_ptx_space_for_mask(destination_space, mask)?;
    source.require_ptx_space_for_mask(source_space, mask)?;
    let mut payloads = Vec::with_capacity(mask.len());
    for lane in mask {
        let byte_len = bulk_byte_len(num_bytes, lane, operation)?;
        let source_offset = source.lane_read_byte_offset(lane, byte_len)?;
        let destination_offset = destination.lane_write_byte_offset(lane, byte_len)?;
        let source_physical = source.lane_physical_byte_offset(lane, byte_len)?;
        let destination_physical = destination.lane_physical_byte_offset(lane, byte_len)?;
        if source_physical % 16 != 0 || destination_physical % 16 != 0 {
            return Err(operation.engine_error(format_args!(
                " requires 16-byte aligned source and destination addresses on lane {lane}"
            )));
        }
        let bytes = read_runtime_bytes(
            physical,
            context,
            source.buffer(),
            lane,
            source_offset,
            byte_len,
        )?;
        payloads.push((lane, destination_offset, bytes));
    }
    Ok(payloads)
}

#[allow(clippy::too_many_arguments)]
fn raw_bulk_copy_layout(
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: i64,
    mask: WarpMask,
    destination_space: PtxStateSpace,
    source_space: PtxStateSpace,
    operation: &str,
) -> Result<(usize, usize, usize, usize), EngineError> {
    destination.require_ptx_space_for_mask(destination_space, mask)?;
    source.require_ptx_space_for_mask(source_space, mask)?;
    let lane = single_issuing_lane(mask, operation)?;
    let byte_len = usize::try_from(num_bytes)
        .map_err(|_| EngineError::message(format!("{operation} byte count is negative")))?;
    if byte_len == 0 || byte_len % 16 != 0 {
        return Err(EngineError::message(format!(
            "{operation} byte count {byte_len} must be a positive multiple of 16"
        )));
    }
    let source_offset = source.lane_read_byte_offset(lane, byte_len)?;
    let destination_offset = destination.lane_write_byte_offset(lane, byte_len)?;
    Ok((lane, source_offset, destination_offset, byte_len))
}

#[allow(clippy::too_many_arguments)]
fn raw_bulk_copy_payload(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: i64,
    mask: WarpMask,
    destination_space: PtxStateSpace,
    source_space: PtxStateSpace,
    operation: &str,
) -> Result<(usize, usize, Vec<u8>), EngineError> {
    let (lane, source_offset, destination_offset, byte_len) = raw_bulk_copy_layout(
        destination,
        source,
        num_bytes,
        mask,
        destination_space,
        source_space,
        operation,
    )?;
    let bytes = read_runtime_bytes(
        physical,
        context,
        source.buffer(),
        lane,
        source_offset,
        byte_len,
    )?;
    Ok((lane, destination_offset, bytes))
}

pub(crate) fn plan_raw_bulk_copy_g2s_accesses(
    operation: &OperationContext,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: i64,
    mask: WarpMask,
    destination_space: PtxStateSpace,
    scope: Option<MemoryScope>,
) -> Result<(Vec<crate::PhysicalAccessBatch>, u64), EngineError> {
    let (lane, source_offset, destination_offset, byte_len) = raw_bulk_copy_layout(
        destination,
        source,
        num_bytes,
        mask,
        destination_space,
        PtxStateSpace::Global,
        "cp.async.bulk.g2s",
    )?;
    let source_access = resolve_runtime_physical_access(
        context,
        source.buffer(),
        lane,
        source_offset,
        byte_len,
        PhysicalAccessKind::Read,
    )?;
    let destination_access = resolve_runtime_physical_access(
        context,
        destination.buffer(),
        lane,
        destination_offset,
        byte_len,
        PhysicalAccessKind::Write,
    )?;
    let mut accesses = vec![
        single_lane_physical_access_batch(
            operation,
            lane,
            PhysicalAccessKind::Read,
            source_access.space(),
            vec![source_access.span()],
        )?
        .with_proxy_memory_domain(source_access.proxy_memory_domain()),
        single_lane_physical_access_batch(
            operation,
            lane,
            PhysicalAccessKind::Write,
            destination_access.space(),
            vec![destination_access.span()],
        )?
        .with_proxy_memory_domain(destination_access.proxy_memory_domain()),
    ];
    scope_bulk_copy_accesses(accesses.iter_mut(), scope)?;
    Ok((
        accesses,
        u64::try_from(byte_len).map_err(|_| EngineError::message("bulk byte count exceeds u64"))?,
    ))
}

pub(crate) fn plan_raw_bulk_copy_s2s_accesses(
    operation: &OperationContext,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: i64,
    mask: WarpMask,
    reduction_width: Option<usize>,
) -> Result<(Vec<crate::PhysicalAccessBatch>, u64), EngineError> {
    let (lane, source_offset, destination_offset, byte_len) = raw_bulk_copy_layout(
        destination,
        source,
        num_bytes,
        mask,
        PtxStateSpace::SharedCluster,
        PtxStateSpace::SharedCta,
        "cp.async.bulk.s2s.cluster",
    )?;
    let source_access = resolve_runtime_physical_access(
        context,
        source.buffer(),
        lane,
        source_offset,
        byte_len,
        PhysicalAccessKind::Read,
    )?;
    let mut accesses = vec![single_lane_physical_access_batch(
        operation,
        lane,
        PhysicalAccessKind::Read,
        source_access.space(),
        vec![source_access.span()],
    )?
    .with_proxy_memory_domain(source_access.proxy_memory_domain())];
    let width = reduction_width.unwrap_or(byte_len);
    if width == 0 || byte_len % width != 0 {
        return Err(EngineError::message(
            "shared bulk destination has invalid element width",
        ));
    }
    let kind = if reduction_width.is_some() {
        PhysicalAccessKind::AtomicReadModifyWrite
    } else {
        PhysicalAccessKind::Write
    };
    for offset in (0..byte_len).step_by(width) {
        let access = resolve_runtime_physical_access(
            context,
            destination.buffer(),
            lane,
            destination_offset + offset,
            width,
            kind,
        )?;
        accesses.push(
            single_lane_physical_access_batch(
                operation,
                lane,
                kind,
                access.space(),
                vec![access.span()],
            )?
            .with_proxy_memory_domain(access.proxy_memory_domain()),
        );
    }
    Ok((
        accesses,
        u64::try_from(byte_len).map_err(|_| EngineError::message("bulk byte count exceeds u64"))?,
    ))
}

fn plan_raw_bulk_s2g_accesses(
    operation: &OperationContext,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: i64,
    mask: WarpMask,
    reduction: Option<(MemoryScope, DeferredGlobalReduction)>,
    label: &str,
) -> Result<
    (
        Vec<crate::PhysicalAccessBatch>,
        Vec<crate::PhysicalAccessBatch>,
    ),
    EngineError,
> {
    let (lane, source_offset, destination_offset, byte_len) = raw_bulk_copy_layout(
        destination,
        source,
        num_bytes,
        mask,
        PtxStateSpace::Global,
        PtxStateSpace::SharedCta,
        label,
    )?;
    let source_access = resolve_runtime_physical_access(
        context,
        source.buffer(),
        lane,
        source_offset,
        byte_len,
        PhysicalAccessKind::Read,
    )?;
    let (destination_kind, destination_unit_bytes, destination_semantics) = match reduction {
        Some((scope, reduction)) => (
            PhysicalAccessKind::AtomicReadModifyWrite,
            reduction.byte_len(),
            MemoryAccessSemantics::async_reduction_at(scope),
        ),
        None => (
            PhysicalAccessKind::Write,
            byte_len,
            MemoryAccessSemantics::plain(),
        ),
    };
    if destination_unit_bytes == 0 || !byte_len.is_multiple_of(destination_unit_bytes) {
        return Err(EngineError::message(format!(
            "{label} byte count {byte_len} is not divisible by destination unit width \
             {destination_unit_bytes}"
        )));
    }
    let mut destination_accesses = Vec::with_capacity(byte_len / destination_unit_bytes);
    // Keep reduction elements in separate batches. `PhysicalFootprint`
    // canonicalizes adjacent spans, but PTX guarantees atomicity per element,
    // not for the bulk destination window as one wide RMW.
    for unit_offset in (0..byte_len).step_by(destination_unit_bytes) {
        let byte_offset = destination_offset
            .checked_add(unit_offset)
            .ok_or_else(|| EngineError::message(format!("{label} destination offset overflow")))?;
        let destination_access = resolve_runtime_physical_access(
            context,
            destination.buffer(),
            lane,
            byte_offset,
            destination_unit_bytes,
            destination_kind,
        )?;
        destination_accesses.push(
            single_lane_physical_access_batch(
                operation,
                lane,
                destination_kind,
                destination_access.space(),
                vec![destination_access.span()],
            )?
            .with_proxy_memory_domain(destination_access.proxy_memory_domain())
            .with_memory_semantics(destination_semantics),
        );
    }
    Ok((
        vec![single_lane_physical_access_batch(
            operation,
            lane,
            PhysicalAccessKind::Read,
            source_access.space(),
            vec![source_access.span()],
        )?
        .with_proxy_memory_domain(ProxyMemoryDomain::SharedCta)],
        destination_accesses,
    ))
}

pub fn plan_raw_bulk_copy_s2g_accesses(
    operation: &OperationContext,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: i64,
    mask: WarpMask,
    scope: Option<MemoryScope>,
) -> Result<
    (
        Vec<crate::PhysicalAccessBatch>,
        Vec<crate::PhysicalAccessBatch>,
    ),
    EngineError,
> {
    let (mut reads, mut writes) = plan_raw_bulk_s2g_accesses(
        operation,
        context,
        destination,
        source,
        num_bytes,
        mask,
        None,
        "cp.async.bulk.s2g",
    )?;
    scope_bulk_copy_accesses(reads.iter_mut().chain(writes.iter_mut()), scope)?;
    Ok((reads, writes))
}

pub(crate) fn bulk_copy_memory_semantics(scope: Option<MemoryScope>) -> MemoryAccessSemantics {
    match scope {
        Some(scope) => MemoryAccessSemantics::scoped(
            crate::MemoryOrder::Relaxed,
            scope,
            crate::MemoryProxy::Async,
            crate::MemoryAccessClass::Atomic,
        ),
        None => MemoryAccessSemantics::async_proxy(),
    }
}

pub(crate) fn scope_bulk_copy_accesses<'a>(
    accesses: impl IntoIterator<Item = &'a mut PhysicalAccessBatch>,
    scope: Option<MemoryScope>,
) -> Result<(), EngineError> {
    if scope.is_none() {
        return Ok(());
    }
    for batch in accesses {
        let descriptor = batch.descriptor();
        let lane = &batch.lanes()[0];
        let spans = lane.footprint().spans();
        if descriptor.space() == PhysicalAccessSpace::Shared
            && spans.iter().any(|span| !span.byte_len().is_multiple_of(16))
        {
            return Err(EngineError::message(
                "strong bulk copies require complete b128 elements",
            ));
        }
        let resolved = if descriptor.space() == PhysicalAccessSpace::Shared {
            // The shared shadow compares individual elements. The global shadow
            // can keep the same identities in a compressed run.
            let mut elements = Vec::new();
            for span in spans {
                for offset in (0..span.byte_len()).step_by(16) {
                    elements.push(
                        crate::PhysicalByteSpan::new(
                            span.allocation(),
                            span.byte_offset() + offset,
                            16,
                        )
                        .map_err(|error| EngineError::message(error.to_string()))?,
                    );
                }
            }
            super::io::single_lane_physical_access_batch_unmerged(
                batch.operation(),
                lane.provenance().lane(),
                descriptor.kind(),
                descriptor.space(),
                elements,
            )?
        } else {
            super::io::single_lane_physical_access_batch_unmerged(
                batch.operation(),
                lane.provenance().lane(),
                descriptor.kind(),
                descriptor.space(),
                spans.to_vec(),
            )?
        };
        let resolved = resolved
            .with_proxy_memory_domain(descriptor.proxy_memory_domain())
            .with_memory_semantics(bulk_copy_memory_semantics(scope));
        *batch = if descriptor.space() == PhysicalAccessSpace::Global {
            resolved.with_aligned_transfer_units(16)
        } else {
            resolved
        };
    }
    Ok(())
}

pub fn plan_raw_bulk_reduce_s2g_accesses(
    operation: &OperationContext,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: i64,
    mask: WarpMask,
    scope: MemoryScope,
    reduction: DeferredGlobalReduction,
) -> Result<
    (
        Vec<crate::PhysicalAccessBatch>,
        Vec<crate::PhysicalAccessBatch>,
    ),
    EngineError,
> {
    plan_raw_bulk_s2g_accesses(
        operation,
        context,
        destination,
        source,
        num_bytes,
        mask,
        Some((scope, reduction)),
        "cp.reduce.async.bulk.s2g",
    )
}

/// Proxy domain of one raw bulk-copy access, keyed on the instruction's PTX
/// state space rather than on the runtime buffer that happens to back it.
///
/// PTX scopes `fence.proxy.async` by *state space*: "the memory ordering is
/// limited only to operations performed on objects in the state space
/// specified". A `.shared::cluster`-spelled copy whose destination happens to
/// resolve to the local CTA is still a `.shared::cluster` operation, so
/// deriving the domain from the runtime buffer would report `shared::cta` for
/// it. Non-shared spaces keep the resolved access's own domain, which is
/// already exact. This is the rule `plan_raw_st_bulk_zero_access` already
/// applies inline.
///
/// The choice is currently *verdict-neutral* for every form here, and that is
/// load-bearing to know: a domain is only compared exactly when it is the
/// earlier access of a proxy bridge, and for an async access the only bridge
/// that can order it to a later generic access is the implicit generic-async
/// fence PTX guarantees at completion --- which is filled from the very same
/// batch's domain, so fill and lookup can never disagree. On the later-access
/// side `proxy_alias_domains` unions `SharedCta` with `SharedCluster`. No
/// kernel can therefore distinguish the two tags for these copies; the rule is
/// here for representational correctness and for any future async form whose
/// completion stops publishing that implicit bridge.
fn raw_bulk_copy_proxy_domain(
    space: PtxStateSpace,
    access: ResolvedRuntimePhysicalAccess,
) -> ProxyMemoryDomain {
    match space {
        PtxStateSpace::SharedCta => ProxyMemoryDomain::SharedCta,
        PtxStateSpace::SharedCluster => ProxyMemoryDomain::SharedCluster,
        _ => access.proxy_memory_domain(),
    }
}

/// How one raw bulk-copy form narrows its footprint inside each lane's window.
///
/// Both fields are `None` for the plain forms; each `Some` corresponds to one
/// PTX qualifier that makes the touched bytes differ from the whole window.
#[derive(Clone, Copy, Default)]
pub(crate) struct RawBulkCopyFootprintShape<'a> {
    /// `.cp_mask` per-lane 16-bit destination byte masks; only the selected
    /// bytes of each 16-byte group are written.
    pub byte_masks: Option<&'a WarpValue<i64>>,
    /// `.multicast` per-lane CTA masks; the destination write is published once
    /// per selected CTA in the cluster.
    pub multicast_cta_masks: Option<&'a WarpValue<i64>>,
}

/// Split one lane's destination window into the byte runs a `.cp_mask` writes.
///
/// The mask repeats every 16 bytes, matching the numeric path in
/// [`raw_bulk_copy_s2g_masked`]. Adjacent selected bytes are merged so a full
/// mask yields exactly one span.
fn masked_destination_runs(
    byte_masks: &WarpValue<i64>,
    lane: usize,
    destination_offset: usize,
    byte_len: usize,
) -> Result<Vec<(usize, usize)>, EngineError> {
    let byte_mask = u16::try_from(byte_masks[lane]).map_err(|_| {
        EngineError::message(format!(
            "cp.async.bulk.s2g byte mask is outside 16 bits on lane {lane}"
        ))
    })?;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for index in 0..byte_len {
        if byte_mask & (1_u16 << (index % 16)) == 0 {
            continue;
        }
        let start = destination_offset + index;
        match runs.last_mut() {
            Some((run_start, run_len)) if *run_start + *run_len == start => *run_len += 1,
            _ => runs.push((start, 1)),
        }
    }
    Ok(runs)
}

/// Pin one batch's space and proxy domain to the first lane that resolves, and
/// reject any later lane that classifies differently.
fn record_access_class(
    class: &mut Option<(PhysicalAccessSpace, ProxyMemoryDomain)>,
    resolved: (PhysicalAccessSpace, ProxyMemoryDomain),
    kind: PhysicalAccessKind,
    lane: usize,
    label: &DiagnosticLabel,
) -> Result<(), EngineError> {
    match class {
        Some(existing) if *existing != resolved => Err(label.engine_error(format_args!(
            " lane {lane} resolves {kind:?} to {resolved:?}, but an earlier lane resolved {existing:?}"
        ))),
        Some(_) => Ok(()),
        None => {
            *class = Some(resolved);
            Ok(())
        }
    }
}

/// Build one batch from per-lane span lists.
///
/// Unlike `single_lane_physical_access_batch` this admits several issuing
/// lanes, which `.multicast` and `.cp_mask` allow and which the single-lane
/// planners cannot express.
fn lane_spans_physical_access_batch(
    operation: &OperationContext,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    proxy_domain: ProxyMemoryDomain,
    lane_spans: &[Vec<crate::PhysicalByteSpan>; WARP_SIZE],
) -> Result<crate::PhysicalAccessBatch, EngineError> {
    crate::PhysicalAccessBatch::resolve_lane_widths(operation.clone(), kind, space, |provenance| {
        Ok(lane_spans[provenance.lane()].clone())
    })
    .map(|batch| batch.with_proxy_memory_domain(proxy_domain))
    .map_err(|error| match error {
        PhysicalAccessBatchError::LaneResolution { source, .. } => source,
        other => EngineError::message(other.to_string()),
    })
}

/// Resolve the exact footprint of one raw bulk copy that may issue from several
/// lanes at once.
///
/// `.multicast` and `.cp_mask` are per-thread instructions whose operands
/// genuinely vary by lane (`lane_varying_bulk_multicast_cta_mask` and
/// `lane_varying_bulk_s2g_size_and_mask` in the NumSim runtime suite both issue
/// from two lanes), so they cannot use the single-issuing-lane layout the plain
/// forms share. Every other raw form stays on that simpler planner.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_raw_bulk_copy_lane_varying_accesses(
    operation: &OperationContext,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: &WarpValue<i64>,
    mask: WarpMask,
    destination_space: PtxStateSpace,
    source_space: PtxStateSpace,
    label: &DiagnosticLabel,
    shape: RawBulkCopyFootprintShape<'_>,
) -> Result<
    (
        Vec<crate::PhysicalAccessBatch>,
        Vec<crate::PhysicalAccessBatch>,
        WarpValue<u64>,
    ),
    EngineError,
> {
    if operation.active_mask() != mask {
        return Err(label.engine_error(format_args!(
            " operation mask {:#010x} does not match execution mask {:#010x}",
            operation.active_mask().bits(),
            mask.bits(),
        )));
    }
    destination.require_ptx_space_for_mask(destination_space, mask)?;
    source.require_ptx_space_for_mask(source_space, mask)?;
    let mut delivered = WarpValue::splat(0_u64);
    // One batch carries one space and one proxy domain, and the PTX state space
    // is pinned across the mask above, so every lane must agree. Recording the
    // first lane's classification and rejecting a disagreement is what keeps a
    // later lane from being silently labelled with the first lane's domain.
    let mut source_class: Option<(PhysicalAccessSpace, ProxyMemoryDomain)> = None;
    let mut destination_class: Option<(PhysicalAccessSpace, ProxyMemoryDomain)> = None;
    let mut source_spans = [const { Vec::new() }; WARP_SIZE];
    let mut destination_spans = [const { Vec::new() }; WARP_SIZE];

    for lane in mask {
        let byte_len = bulk_byte_len(num_bytes, lane, label)?;
        let source_offset = source.lane_read_byte_offset(lane, byte_len)?;
        let destination_offset = destination.lane_write_byte_offset(lane, byte_len)?;
        let source_physical = source.lane_physical_byte_offset(lane, byte_len)?;
        let destination_physical = destination.lane_physical_byte_offset(lane, byte_len)?;
        if source_physical % 16 != 0 || destination_physical % 16 != 0 {
            return Err(label.engine_error(format_args!(
                " requires 16-byte aligned source and destination addresses on lane {lane}"
            )));
        }
        delivered[lane] = u64::try_from(byte_len)
            .map_err(|_| label.engine_error(format_args!(" byte count exceeds u64")))?;

        let source_access = resolve_runtime_physical_access(
            context,
            source.buffer(),
            lane,
            source_offset,
            byte_len,
            PhysicalAccessKind::Read,
        )?;
        record_access_class(
            &mut source_class,
            (
                source_access.space(),
                raw_bulk_copy_proxy_domain(source_space, source_access),
            ),
            PhysicalAccessKind::Read,
            lane,
            label,
        )?;
        source_spans[lane].push(source_access.span());

        let destination_runs = match shape.byte_masks {
            Some(byte_masks) => {
                masked_destination_runs(byte_masks, lane, destination_offset, byte_len)?
            }
            None => vec![(destination_offset, byte_len)],
        };
        for (run_offset, run_len) in destination_runs {
            let targets = match shape.multicast_cta_masks {
                Some(cta_masks) => {
                    let cta_mask = u64::try_from(cta_masks[lane]).map_err(|_| {
                        label.engine_error(format_args!(" CTA mask is negative on lane {lane}"))
                    })?;
                    super::io::validate_tma_multicast_mask(
                        cta_mask,
                        context.topology().ctas_per_cluster(),
                    )?;
                    (0..context.topology().ctas_per_cluster())
                        .filter(|target| cta_mask & (1_u64 << target) != 0)
                        .map(Some)
                        .collect::<Vec<_>>()
                }
                None => vec![None],
            };
            for target in targets {
                let access = match target {
                    Some(target_cta) => resolve_shared_runtime_physical_access_to_cta(
                        context,
                        destination.buffer(),
                        lane,
                        target_cta,
                        run_offset,
                        run_len,
                        PhysicalAccessKind::Write,
                    )?,
                    None => resolve_runtime_physical_access(
                        context,
                        destination.buffer(),
                        lane,
                        run_offset,
                        run_len,
                        PhysicalAccessKind::Write,
                    )?,
                };
                record_access_class(
                    &mut destination_class,
                    (
                        access.space(),
                        raw_bulk_copy_proxy_domain(destination_space, access),
                    ),
                    PhysicalAccessKind::Write,
                    lane,
                    label,
                )?;
                destination_spans[lane].push(access.span());
            }
        }
    }

    let mut source_reads = Vec::new();
    let mut destination_writes = Vec::new();
    if let Some((space, domain)) = source_class {
        source_reads.push(lane_spans_physical_access_batch(
            operation,
            PhysicalAccessKind::Read,
            space,
            domain,
            &source_spans,
        )?);
    }
    if let Some((space, domain)) = destination_class {
        destination_writes.push(lane_spans_physical_access_batch(
            operation,
            PhysicalAccessKind::Write,
            space,
            domain,
            &destination_spans,
        )?);
    }
    Ok((source_reads, destination_writes, delivered))
}

const IGNORE_OOB_LABEL: &str = "cp.async.bulk.g2s.cta.ignore_oob";

/// One lane's `.ignore_oob` window.
///
/// The destination window is written whole; only the in-bounds middle slice is
/// read, so the source pointer is never bounds-checked over the ignored bytes.
/// The numeric transfer and the footprint planner both take their geometry from
/// here, which is what keeps the planner from rejecting a source binding the
/// copy legitimately reads past the end of.
#[derive(Clone, Copy)]
struct RawBulkCopyIgnoreOobWindow {
    destination_offset: usize,
    byte_len: usize,
    valid_start: usize,
    valid_len: usize,
    /// Resolved source byte offset of the valid slice, absent when the window
    /// ignores everything.
    source_offset: Option<usize>,
}

fn raw_bulk_copy_ignore_oob_lane_window(
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: &WarpValue<i64>,
    ignore_bytes_left: &WarpValue<i64>,
    ignore_bytes_right: &WarpValue<i64>,
    lane: usize,
) -> Result<RawBulkCopyIgnoreOobWindow, EngineError> {
    let byte_len = bulk_byte_len(num_bytes, lane, &DiagnosticLabel::new(IGNORE_OOB_LABEL))?;
    let ignore_left = usize::try_from(ignore_bytes_left[lane]).map_err(|_| {
        EngineError::message(format!(
            "{IGNORE_OOB_LABEL} left byte count is negative on lane {lane}"
        ))
    })?;
    let ignore_right = usize::try_from(ignore_bytes_right[lane]).map_err(|_| {
        EngineError::message(format!(
            "{IGNORE_OOB_LABEL} right byte count is negative on lane {lane}"
        ))
    })?;
    if ignore_left > 15 || ignore_right > 15 {
        return Err(EngineError::message(format!(
            "{IGNORE_OOB_LABEL} ignored byte counts must be in 0..=15 on lane {lane}"
        )));
    }
    let destination_offset = destination.lane_write_byte_offset(lane, byte_len)?;
    let source_physical = source.lane_physical_byte_offset(lane, 0)?;
    let destination_physical = destination.lane_physical_byte_offset(lane, byte_len)?;
    if source_physical % 16 != 0 || destination_physical % 16 != 0 {
        return Err(EngineError::message(format!(
            "{IGNORE_OOB_LABEL} requires 16-byte aligned source and destination addresses on lane {lane}"
        )));
    }
    let valid_start = ignore_left.min(byte_len);
    let valid_end = byte_len.saturating_sub(ignore_right).max(valid_start);
    let valid_len = valid_end - valid_start;
    let source_offset = if valid_len == 0 {
        None
    } else {
        Some(source.lane_read_byte_offset_at(lane, valid_start, valid_len)?)
    };
    Ok(RawBulkCopyIgnoreOobWindow {
        destination_offset,
        byte_len,
        valid_start,
        valid_len,
        source_offset,
    })
}

/// Resolve the exact footprint of one `.ignore_oob` global-to-shared copy.
///
/// Shaped like [`plan_raw_bulk_copy_g2s_accesses`]: one issuing lane, one read
/// batch and one write batch. The difference is that the read covers only the
/// in-bounds slice while the write covers the whole destination window, which
/// is what `.ignore_oob` means.
pub(crate) fn plan_raw_bulk_copy_g2s_ignore_oob_accesses(
    operation: &OperationContext,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: &WarpValue<i64>,
    ignore_bytes_left: &WarpValue<i64>,
    ignore_bytes_right: &WarpValue<i64>,
    mask: WarpMask,
) -> Result<(Vec<crate::PhysicalAccessBatch>, u64), EngineError> {
    destination.require_ptx_space_for_mask(PtxStateSpace::SharedCta, mask)?;
    source.require_ptx_space_for_mask(PtxStateSpace::Global, mask)?;
    let lane = single_issuing_lane(mask, IGNORE_OOB_LABEL)?;
    let window = raw_bulk_copy_ignore_oob_lane_window(
        destination,
        source,
        num_bytes,
        ignore_bytes_left,
        ignore_bytes_right,
        lane,
    )?;
    let mut accesses = Vec::new();
    if let Some(source_offset) = window.source_offset {
        let source_access = resolve_runtime_physical_access(
            context,
            source.buffer(),
            lane,
            source_offset,
            window.valid_len,
            PhysicalAccessKind::Read,
        )?;
        accesses.push(
            single_lane_physical_access_batch(
                operation,
                lane,
                PhysicalAccessKind::Read,
                source_access.space(),
                vec![source_access.span()],
            )?
            .with_proxy_memory_domain(source_access.proxy_memory_domain()),
        );
    }
    let destination_access = resolve_runtime_physical_access(
        context,
        destination.buffer(),
        lane,
        window.destination_offset,
        window.byte_len,
        PhysicalAccessKind::Write,
    )?;
    accesses.push(
        single_lane_physical_access_batch(
            operation,
            lane,
            PhysicalAccessKind::Write,
            destination_access.space(),
            vec![destination_access.span()],
        )?
        .with_proxy_memory_domain(raw_bulk_copy_proxy_domain(
            PtxStateSpace::SharedCta,
            destination_access,
        )),
    );
    Ok((
        accesses,
        u64::try_from(window.byte_len)
            .map_err(|_| EngineError::message("bulk byte count exceeds u64"))?,
    ))
}

pub fn raw_bulk_copy_g2s_cta_ignore_oob(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: &WarpValue<i64>,
    ignore_bytes_left: &WarpValue<i64>,
    ignore_bytes_right: &WarpValue<i64>,
    mask: WarpMask,
) -> Result<WarpValue<u64>, EngineError> {
    destination.require_ptx_space_for_mask(PtxStateSpace::SharedCta, mask)?;
    source.require_ptx_space_for_mask(PtxStateSpace::Global, mask)?;
    // The single-issuing-lane rule belongs here, not only in the planner: the
    // completion credits one lane's byte count, so a multi-lane issue that got
    // this far would under-credit the barrier and surface as a phantom
    // executor deadlock in NumSim mode instead of this exact rejection.
    let lane = single_issuing_lane(mask, IGNORE_OOB_LABEL)?;
    let window = raw_bulk_copy_ignore_oob_lane_window(
        destination,
        source,
        num_bytes,
        ignore_bytes_left,
        ignore_bytes_right,
        lane,
    )?;
    let mut bytes = vec![0_u8; window.byte_len];
    let mut validity = vec![false; window.byte_len];
    if let Some(source_offset) = window.source_offset {
        let valid_end = window.valid_start + window.valid_len;
        let valid_bytes = read_runtime_bytes(
            physical,
            context,
            source.buffer(),
            lane,
            source_offset,
            window.valid_len,
        )?;
        bytes[window.valid_start..valid_end].copy_from_slice(&valid_bytes);
        validity[window.valid_start..valid_end].fill(true);
    }
    write_runtime_bytes_with_validity(
        physical,
        context,
        destination.buffer(),
        lane,
        window.destination_offset,
        &bytes,
        &validity,
    )?;
    let mut delivered = WarpValue::splat(0_u64);
    delivered[lane] = u64::try_from(window.byte_len)
        .map_err(|_| EngineError::message("bulk byte count exceeds u64"))?;
    Ok(delivered)
}

pub fn raw_bulk_copy_g2s_multicast(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: &WarpValue<i64>,
    mask: WarpMask,
    cta_masks: &WarpValue<i64>,
    report_pattern: u32,
) -> Result<(WarpValue<u64>, WarpValue<bool>), EngineError> {
    let payloads = raw_bulk_copy_payloads(
        physical,
        context,
        destination,
        source,
        num_bytes,
        mask,
        PtxStateSpace::SharedCluster,
        PtxStateSpace::Global,
        &DiagnosticLabel::new("cp.async.bulk.g2s.cluster.multicast"),
    )?;
    let mut delivered = WarpValue::splat(0_u64);
    let mut reported = WarpValue::splat(false);
    for (lane, destination_offset, bytes) in payloads {
        reported[lane] = copy_report_matches(report_pattern, &bytes)?;
        let cta_mask = u64::try_from(cta_masks[lane]).map_err(|_| {
            EngineError::message(format!("negative bulk multicast CTA mask on lane {lane}"))
        })?;
        super::io::validate_tma_multicast_mask(cta_mask, context.topology().ctas_per_cluster())?;
        for target in 0..context.topology().ctas_per_cluster() {
            if cta_mask & (1_u64 << target) == 0 {
                continue;
            }
            write_shared_runtime_bytes_to_cta(
                physical,
                context,
                destination.buffer(),
                lane,
                target,
                destination_offset,
                &bytes,
            )?;
        }
        delivered[lane] = u64::try_from(bytes.len())
            .map_err(|_| EngineError::message("bulk byte count exceeds u64"))?;
    }
    Ok((delivered, reported))
}

/// PTX 9.4 permits one implementation-chosen sample per aligned 16-byte
/// chunk. NumSim chooses its first element (low nibble for FP4). `0xff`
/// instead inspects every byte; zero is the explicitly disabled mechanism.
pub(crate) fn copy_report_matches(pattern: u32, bytes: &[u8]) -> Result<bool, EngineError> {
    copy_report_matches_runs(pattern, std::iter::once((0, bytes)))
}

/// Tensor copies can touch discontiguous parts of a source chunk. Choose the
/// lowest-addressed copied element of each chunk, independently of the tensor
/// walk, shared swizzle, padding, and OOB fill. Never inspect extra source bytes.
pub(crate) fn copy_report_matches_runs<'a>(
    pattern: u32,
    runs: impl IntoIterator<Item = (usize, &'a [u8])>,
) -> Result<bool, EngineError> {
    let width = match pattern {
        0 => return Ok(false),
        0xff => return Ok(runs.into_iter().any(|(_, bytes)| bytes.contains(&0xff))),
        0x8 | 0x80 => 1,
        0x8000 => 2,
        0x80000000 => 4,
        _ => return Err(EngineError::message("invalid bulk copy report pattern")),
    };
    let mut samples = std::collections::BTreeMap::new();
    for (offset, bytes) in runs {
        if !offset.is_multiple_of(width) || !bytes.len().is_multiple_of(width) {
            return Err(EngineError::message(
                "bulk report sample is not a whole aligned element",
            ));
        }
        for (index, element) in bytes.chunks_exact(width).enumerate() {
            let address = offset
                .checked_add(index * width)
                .ok_or_else(|| EngineError::message("bulk report source address overflow"))?;
            let mut bits = [0u8; 4];
            bits[..width].copy_from_slice(element);
            if pattern == 0x8 {
                bits[0] &= 0xf;
            }
            let sample = (address, u32::from_le_bytes(bits));
            let previous = samples.entry(address / 16).or_insert(sample);
            if address < previous.0 {
                *previous = sample;
            }
        }
    }
    Ok(samples.values().any(|&(_, value)| value == pattern))
}


pub(crate) fn execute_raw_bulk_copy_g2s(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: i64,
    mask: WarpMask,
    destination_space: PtxStateSpace,
    report_pattern: u32,
) -> Result<(u64, bool), EngineError> {
    let (lane, destination_offset, bytes) = raw_bulk_copy_payload(
        physical,
        context,
        destination,
        source,
        num_bytes,
        mask,
        destination_space,
        PtxStateSpace::Global,
        "cp.async.bulk.g2s",
    )?;
    let delivered = u64::try_from(bytes.len())
        .map_err(|_| EngineError::message("bulk byte count exceeds u64"))?;
    let reported = copy_report_matches(report_pattern, &bytes)?;
    write_runtime_bytes(
        physical,
        context,
        destination.buffer(),
        lane,
        destination_offset,
        &bytes,
    )?;
    Ok((delivered, reported))
}

pub(crate) fn execute_raw_bulk_copy_s2s(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: i64,
    mask: WarpMask,
) -> Result<u64, EngineError> {
    let (lane, destination_offset, bytes) = raw_bulk_copy_payload(
        physical,
        context,
        destination,
        source,
        num_bytes,
        mask,
        PtxStateSpace::SharedCluster,
        PtxStateSpace::SharedCta,
        "cp.async.bulk.s2s.cluster",
    )?;
    let delivered = u64::try_from(bytes.len())
        .map_err(|_| EngineError::message("bulk byte count exceeds u64"))?;
    write_runtime_bytes(
        physical,
        context,
        destination.buffer(),
        lane,
        destination_offset,
        &bytes,
    )?;
    Ok(delivered)
}

pub(crate) fn execute_raw_bulk_reduce_s2c<T: super::RawAtomicScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: i64,
    mask: WarpMask,
    operation: super::RawAtomicOperation,
) -> Result<u64, EngineError> {
    let (lane, _, bytes) = raw_bulk_copy_payload(
        physical,
        context,
        destination,
        source,
        num_bytes,
        mask,
        PtxStateSpace::SharedCluster,
        PtxStateSpace::SharedCta,
        "cp.reduce.async.bulk.s2c",
    )?;
    for (index, element) in bytes.chunks_exact(T::BYTE_LEN).enumerate() {
        let pointer = destination.with_byte_offset(
            &WarpValue::splat((index * T::BYTE_LEN) as i64),
            T::BYTE_LEN,
            mask,
        )?;
        let mut operand = WarpValue::splat(T::zero());
        operand[lane] = T::decode_le(element)?;
        super::raw_atomic_scalar_physical_ptr_warp(
            physical,
            context,
            &pointer,
            &operand,
            mask,
            PtxStateSpace::SharedCluster,
            operation,
        )?;
    }
    Ok(bytes.len() as u64)
}

pub fn raw_bulk_copy_s2g(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: &WarpValue<i64>,
    mask: WarpMask,
) -> Result<Vec<DeferredGlobalWrite>, EngineError> {
    let payloads = raw_bulk_copy_payloads(
        physical,
        context,
        destination,
        source,
        num_bytes,
        mask,
        PtxStateSpace::Global,
        PtxStateSpace::SharedCta,
        &DiagnosticLabel::new("cp.async.bulk.s2g"),
    )?;
    payloads
        .into_iter()
        .map(|(lane, destination_offset, bytes)| {
            defer_global_runtime_bytes(
                physical,
                destination.buffer(),
                lane,
                destination_offset,
                bytes,
            )
        })
        .collect()
}

pub fn raw_bulk_reduce_s2g(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: i64,
    mask: WarpMask,
    reduction: DeferredGlobalReduction,
) -> Result<Vec<DeferredGlobalWrite>, EngineError> {
    let payloads = raw_bulk_copy_payloads(
        physical,
        context,
        destination,
        source,
        &WarpValue::splat(num_bytes),
        mask,
        PtxStateSpace::Global,
        PtxStateSpace::SharedCta,
        &DiagnosticLabel::new("cp.reduce.async.bulk.s2g"),
    )?;
    let mut writes = Vec::new();
    for (lane, destination_offset, bytes) in payloads {
        for (element, value) in bytes.chunks_exact(reduction.byte_len()).enumerate() {
            let element_byte_offset =
                element.checked_mul(reduction.byte_len()).ok_or_else(|| {
                    EngineError::message("bulk reduction destination offset overflow")
                })?;
            let element_offset = destination_offset
                .checked_add(element_byte_offset)
                .ok_or_else(|| {
                    EngineError::message("bulk reduction destination offset overflow")
                })?;
            writes.push(defer_global_runtime_reduction_bytes(
                physical,
                destination.buffer(),
                lane,
                element_offset,
                value.to_vec(),
                reduction,
            )?);
        }
    }
    Ok(writes)
}

pub fn raw_bulk_copy_s2g_masked(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &PhysicalPtr,
    source: &PhysicalPtr,
    num_bytes: &WarpValue<i64>,
    mask: WarpMask,
    byte_masks: &WarpValue<i64>,
) -> Result<Vec<DeferredGlobalWrite>, EngineError> {
    let payloads = raw_bulk_copy_payloads(
        physical,
        context,
        destination,
        source,
        num_bytes,
        mask,
        PtxStateSpace::Global,
        PtxStateSpace::SharedCta,
        &DiagnosticLabel::new("cp.async.bulk.s2g.cp_mask"),
    )?;
    payloads
        .into_iter()
        .map(|(lane, destination_offset, bytes)| {
            let byte_mask = u16::try_from(byte_masks[lane]).map_err(|_| {
                EngineError::message(format!(
                    "cp.async.bulk.s2g byte mask is outside 16 bits on lane {lane}"
                ))
            })?;
            let masks = (0..bytes.len())
                .map(|index| {
                    if byte_mask & (1_u16 << (index % 16)) != 0 {
                        0xff
                    } else {
                        0
                    }
                })
                .collect();
            defer_global_runtime_masked_bytes(
                physical,
                destination.buffer(),
                lane,
                destination_offset,
                bytes,
                masks,
            )
        })
        .collect()
}

#[cfg(test)]
mod raw_bulk_copy_proxy_domain_tests {
    use super::*;
    use crate::PhysicalAllocationId;

    fn access(
        space: PhysicalAccessSpace,
        domain: ProxyMemoryDomain,
    ) -> ResolvedRuntimePhysicalAccess {
        let span = crate::PhysicalByteSpan::new(PhysicalAllocationId::new(0), 0, 16)
            .expect("test span is well formed");
        ResolvedRuntimePhysicalAccess::for_test(space, domain, span)
    }

    /// The domain follows the instruction's PTX state space, not its buffer.
    ///
    /// The middle row is the one a buffer-derived rule gets wrong: a
    /// `.shared::cluster`-spelled copy whose destination resolves to the
    /// executing CTA is backed by a local `Shared` buffer, which resolves as
    /// `SharedCta`.
    ///
    /// This is asserted at the rule rather than through a kernel because no
    /// kernel can distinguish the two tags for these copies --- see the
    /// `raw_bulk_copy_proxy_domain` doc comment for why.
    #[test]
    fn proxy_domain_follows_the_instruction_state_space() {
        assert_eq!(
            raw_bulk_copy_proxy_domain(
                PtxStateSpace::SharedCta,
                access(PhysicalAccessSpace::Shared, ProxyMemoryDomain::SharedCta),
            ),
            ProxyMemoryDomain::SharedCta,
        );
        assert_eq!(
            raw_bulk_copy_proxy_domain(
                PtxStateSpace::SharedCluster,
                access(PhysicalAccessSpace::Shared, ProxyMemoryDomain::SharedCta),
            ),
            ProxyMemoryDomain::SharedCluster,
        );
        assert_eq!(
            raw_bulk_copy_proxy_domain(
                PtxStateSpace::SharedCluster,
                access(
                    PhysicalAccessSpace::Shared,
                    ProxyMemoryDomain::SharedCluster
                ),
            ),
            ProxyMemoryDomain::SharedCluster,
        );
    }

    /// A space PTX does not qualify keeps the resolved access's own domain.
    #[test]
    fn proxy_domain_falls_back_to_the_access_for_unqualified_spaces() {
        assert_eq!(
            raw_bulk_copy_proxy_domain(
                PtxStateSpace::Global,
                access(PhysicalAccessSpace::Global, ProxyMemoryDomain::Global),
            ),
            ProxyMemoryDomain::Global,
        );
    }

    #[test]
    fn planner_marks_the_global_destination_as_an_async_atomic_reduction() {
        use std::sync::Arc;
        use crate::runtime::allocate_cta_shared;
        use crate::{
            AsyncGroupDomain, AsyncGroupIssueEffect, DynamicOpId, LaunchTopology,
            MemoryAccessClass, MemoryOrder, MemoryProxy, OperationKind, StaticOpId,
        };

        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let physical = PhysicalMemory::new(topology);
        let global_allocation = physical.global().allocate_zeroed(16).unwrap();
        let destination = PhysicalPtr::new(
            RuntimeBuffer::Global(physical.global().full_view(global_allocation).unwrap()),
            WarpValue::splat(0),
            4,
        );
        let source = PhysicalPtr::new(
            RuntimeBuffer::Shared {
                allocations: Arc::new(allocate_cta_shared(&physical, topology, 16).unwrap()),
                byte_offset: 0,
                byte_len: 16,
                backing_byte_len: 16,
                virtual_base: 0,
            },
            WarpValue::splat(0),
            4,
        );
        let mask = WarpMask::from_lanes([0]).unwrap();
        let operation = OperationContext::new(
            DynamicOpId::new(0, 0, 0, StaticOpId::new(7), []),
            OperationKind::AsyncIssue,
            mask,
        );

        let (sources, destinations) = plan_raw_bulk_reduce_s2g_accesses(
            &operation,
            &context,
            &destination,
            &source,
            16,
            mask,
            MemoryScope::Sys,
            DeferredGlobalReduction::AddF32Ftz,
        )
        .unwrap();
        assert_eq!(sources[0].descriptor().kind(), PhysicalAccessKind::Read);
        assert_eq!(
            destinations[0].descriptor().kind(),
            PhysicalAccessKind::AtomicReadModifyWrite
        );

        let effect =
            AsyncGroupIssueEffect::new(operation, AsyncGroupDomain::Bulk, sources, destinations)
                .unwrap();
        let semantics = effect.destination_accesses()[0]
            .descriptor()
            .memory_semantics();
        assert_eq!(semantics.class(), MemoryAccessClass::Reduction);
        assert_eq!(semantics.order(), MemoryOrder::Relaxed);
        assert_eq!(semantics.scope(), Some(MemoryScope::Sys));
        assert_eq!(semantics.proxy(), MemoryProxy::Async);
    }
}
