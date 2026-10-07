use std::mem::size_of;

use crate::{
    AddressSpaceError, CtaId, EngineError, PhysicalAllocationId, PhysicalByteSpan, PhysicalMemory,
    RuntimeScalar, TcgenLifecycleHub, TmemAccessMode, TmemAllocation, TmemView, WarpContext,
    WarpId, WarpMask, WarpValue, TMEM_CELL_BYTES, WARP_SIZE,
};

use super::{warp_ops::require_uniform_i64, RuntimeBuffer, MAX_RUNTIME_SCALAR_BYTES};

const FAST_TMEM_F32_COLUMNS: usize = 64;

#[allow(clippy::too_many_arguments)]
fn fast_tmem_f32_view(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    buffer: &RuntimeBuffer,
    base_lane: i64,
    maximum_lane_offset: usize,
    base_tcol: i64,
    allocated_addr: i64,
    column_count: usize,
) -> Result<(TmemView, usize, usize), EngineError> {
    let RuntimeBuffer::Tmem {
        allocations,
        lane_span,
        tcol_span_elements,
        elem_offset,
        itemsize,
    } = buffer
    else {
        return Err(EngineError::message(
            "fast TMEM-to-local transfer requires a TMEM source",
        ));
    };
    if *itemsize != size_of::<f32>() {
        return Err(EngineError::message(format!(
            "fast TMEM-to-local f32 transfer received {itemsize}-byte elements"
        )));
    }
    let base_lane = usize::try_from(base_lane)
        .map_err(|_| EngineError::message("fast TMEM-to-local transfer has a negative TLane"))?;
    let last_lane = base_lane
        .checked_add(maximum_lane_offset)
        .ok_or_else(|| EngineError::message("fast TMEM-to-local TLane range overflow"))?;
    if last_lane >= *lane_span {
        return Err(EngineError::message(format!(
            "fast TMEM-to-local TLane {last_lane} is outside layout span {lane_span}"
        )));
    }
    let base_tcol = usize::try_from(base_tcol).map_err(|_| {
        EngineError::message("fast TMEM-to-local transfer has a negative TCol element")
    })?;
    if !(1..=FAST_TMEM_F32_COLUMNS).contains(&column_count) {
        return Err(EngineError::message(format!(
            "fast TMEM-to-local transfer has invalid f32 column count {column_count}"
        )));
    }
    let tcol_end = base_tcol
        .checked_add(column_count)
        .ok_or_else(|| EngineError::message("fast TMEM-to-local TCol range overflow"))?;
    if tcol_end > *tcol_span_elements {
        return Err(EngineError::message(format!(
            "fast TMEM-to-local TCol range [{base_tcol}, {tcol_end}) exceeds layout span {tcol_span_elements}"
        )));
    }
    let allocated_addr = usize::try_from(allocated_addr).map_err(|_| {
        EngineError::message("fast TMEM-to-local transfer has a negative allocated_addr")
    })?;
    let first_column = allocated_addr
        .checked_add(*elem_offset)
        .and_then(|column| column.checked_add(base_tcol))
        .ok_or_else(|| EngineError::message("fast TMEM-to-local column offset overflow"))?;
    lifecycle.validate_access_range_at_cta(
        *context,
        context.cta_id_in_cluster(),
        first_column,
        column_count,
        access_mode,
    )?;
    let allocation = allocations
        .get(context.global_cta_id())
        .ok_or_else(|| EngineError::message("TMEM CTA allocation is missing"))?;
    let owner = CtaId::from_context(*context);
    let view = physical.tmem().full_view(owner, allocation)?;
    Ok((view, base_lane, first_column))
}

fn write_fast_tmem_f32_rows(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &RuntimeBuffer,
    values: &[f32],
    columns_per_lane: usize,
) -> Result<(), EngineError> {
    let expected = WARP_SIZE
        .checked_mul(columns_per_lane)
        .ok_or_else(|| EngineError::message("fast TMEM-to-local output shape overflow"))?;
    if values.len() != expected {
        return Err(EngineError::message(format!(
            "fast TMEM-to-local output has {} values, expected {expected}",
            values.len()
        )));
    }
    let row_bytes = columns_per_lane
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| EngineError::message("fast TMEM-to-local row byte size overflow"))?;
    let mut encoded = Vec::with_capacity(expected * size_of::<f32>());
    for value in values {
        encoded.extend_from_slice(&value.to_le_bytes());
    }
    let owner_private = match destination {
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
        memory.write_lane_bytes_batch(
            WarpId::from_context(*context),
            allocation,
            view_offset,
            view_len,
            &WarpValue::splat(0),
            context.active_mask(),
            row_bytes,
            row_bytes,
            &encoded,
        )?;
        return Ok(());
    }
    for lane in 0..WARP_SIZE {
        let start = lane * row_bytes;
        super::write_runtime_bytes(
            physical,
            context,
            destination,
            lane,
            0,
            &encoded[start..start + row_bytes],
        )?;
    }
    Ok(())
}

/// Execute the canonical register-to-TMEM data movement of one
/// `tcgen05.st.32x32b` warp slice without scalarizing its register tuple.
///
/// The caller has already selected the PTX variant and proved the register
/// and TMEM layouts.  Runtime values are therefore limited to the TMEM
/// origin; this helper still validates allocation bounds and lifecycle state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_register_32x32b_to_tmem_warp<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    source: &RuntimeBuffer,
    destination: &RuntimeBuffer,
    base_lanes: &WarpValue<i64>,
    base_tcols: &WarpValue<i64>,
    allocated_addrs: &WarpValue<i64>,
    elements_per_lane: usize,
    mask: WarpMask,
) -> Result<(), EngineError> {
    super::require_full_warp_sync(mask, "canonical tcgen05.st.32x32b")?;
    let base_lane = require_uniform_i64(base_lanes, mask, "canonical TCGEN ST TLane")?;
    let base_tcol = require_uniform_i64(base_tcols, mask, "canonical TCGEN ST TCol")?;
    let allocated_addr =
        require_uniform_i64(allocated_addrs, mask, "canonical TCGEN ST allocated_addr")?;
    let row_bytes = elements_per_lane
        .checked_mul(T::BYTE_LEN)
        .ok_or_else(|| EngineError::message("canonical TCGEN ST register row overflows usize"))?;
    if row_bytes == 0 || row_bytes % TMEM_CELL_BYTES != 0 {
        return Err(EngineError::message(
            "canonical tcgen05.st.32x32b requires a nonempty whole-register row",
        ));
    }

    let (source_memory, source_allocations, source_offset, source_len) = match source {
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
                "canonical tcgen05.st.32x32b requires a lane-private register source",
            ));
        }
    };
    let source_allocation = source_allocations
        .get(context.global_warp_id())
        .ok_or_else(|| EngineError::message("warp-private allocation is missing"))?;
    let payload_len = WARP_SIZE
        .checked_mul(row_bytes)
        .ok_or_else(|| EngineError::message("canonical TCGEN ST payload overflows usize"))?;
    let mut payload = vec![0_u8; payload_len];
    source_memory.read_lane_bytes_batch_into(
        WarpId::from_context(*context),
        source_allocation,
        source_offset,
        source_len,
        &WarpValue::splat(0),
        mask,
        row_bytes,
        row_bytes,
        &mut payload,
    )?;

    let warp_lane_offset = (context.warp_id_in_cta() % 4)
        .checked_mul(WARP_SIZE)
        .ok_or_else(|| EngineError::message("canonical TCGEN ST warp lane offset overflow"))?;
    for execution_lane in mask {
        let lane_offset = warp_lane_offset
            .checked_add(execution_lane)
            .ok_or_else(|| EngineError::message("canonical TCGEN ST lane offset overflow"))?;
        let mapped_lane =
            base_lane
                .checked_add(i64::try_from(lane_offset).map_err(|_| {
                    EngineError::message("canonical TCGEN ST lane offset exceeds i64")
                })?)
                .ok_or_else(|| EngineError::message("canonical TCGEN ST TLane overflow"))?;
        let location = resolve_tmem_span_at_cta(
            context,
            Some((lifecycle, access_mode)),
            destination,
            context.cta_id_in_cluster(),
            TmemElementAddress {
                mapped_lane,
                tcol_element: base_tcol,
                allocated_addr,
                access_bytes: row_bytes,
                execution_lane,
            },
        )?;
        if location.byte_in_cell != 0 {
            return Err(EngineError::message(
                "canonical tcgen05.st.32x32b TMEM origin is not register aligned",
            ));
        }
        let view = physical
            .tmem()
            .full_view(location.owner, location.allocation)?;
        let start = execution_lane
            .checked_mul(row_bytes)
            .ok_or_else(|| EngineError::message("canonical TCGEN ST row offset overflow"))?;
        let row = &payload[start..start + row_bytes];
        let cells = row
            .chunks_exact(TMEM_CELL_BYTES)
            .map(|bytes| {
                bytes
                    .try_into()
                    .expect("whole-register TCGEN row has four-byte cells")
            })
            .collect::<Vec<[u8; TMEM_CELL_BYTES]>>();
        physical
            .tmem()
            .write_lane_cells(&view, location.lane, location.column, &cells)?;
    }
    Ok(())
}

/// Execute the canonical TMEM-to-register data movement of one
/// `tcgen05.ld.32x32b` warp slice without scalarizing its register tuple.
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_tmem_32x32b_to_register_warp<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    source: &RuntimeBuffer,
    destination: &RuntimeBuffer,
    base_lanes: &WarpValue<i64>,
    base_tcols: &WarpValue<i64>,
    allocated_addrs: &WarpValue<i64>,
    elements_per_lane: usize,
    mask: WarpMask,
) -> Result<(), EngineError> {
    super::require_full_warp_sync(mask, "canonical tcgen05.ld.32x32b")?;
    let base_lane = require_uniform_i64(base_lanes, mask, "canonical TCGEN LD TLane")?;
    let base_tcol = require_uniform_i64(base_tcols, mask, "canonical TCGEN LD TCol")?;
    let allocated_addr =
        require_uniform_i64(allocated_addrs, mask, "canonical TCGEN LD allocated_addr")?;
    let row_bytes = elements_per_lane
        .checked_mul(T::BYTE_LEN)
        .ok_or_else(|| EngineError::message("canonical TCGEN LD register row overflows usize"))?;
    if row_bytes == 0 || row_bytes % TMEM_CELL_BYTES != 0 {
        return Err(EngineError::message(
            "canonical tcgen05.ld.32x32b requires a nonempty whole-register row",
        ));
    }
    let payload_len = WARP_SIZE
        .checked_mul(row_bytes)
        .ok_or_else(|| EngineError::message("canonical TCGEN LD payload overflows usize"))?;
    let mut payload = vec![0_u8; payload_len];
    let warp_lane_offset = (context.warp_id_in_cta() % 4)
        .checked_mul(WARP_SIZE)
        .ok_or_else(|| EngineError::message("canonical TCGEN LD warp lane offset overflow"))?;
    for execution_lane in mask {
        let lane_offset = warp_lane_offset
            .checked_add(execution_lane)
            .ok_or_else(|| EngineError::message("canonical TCGEN LD lane offset overflow"))?;
        let mapped_lane =
            base_lane
                .checked_add(i64::try_from(lane_offset).map_err(|_| {
                    EngineError::message("canonical TCGEN LD lane offset exceeds i64")
                })?)
                .ok_or_else(|| EngineError::message("canonical TCGEN LD TLane overflow"))?;
        let location = resolve_tmem_span_at_cta(
            context,
            Some((lifecycle, access_mode)),
            source,
            context.cta_id_in_cluster(),
            TmemElementAddress {
                mapped_lane,
                tcol_element: base_tcol,
                allocated_addr,
                access_bytes: row_bytes,
                execution_lane,
            },
        )?;
        if location.byte_in_cell != 0 {
            return Err(EngineError::message(
                "canonical tcgen05.ld.32x32b TMEM origin is not register aligned",
            ));
        }
        let view = physical
            .tmem()
            .full_view(location.owner, location.allocation)?;
        let mut cells = vec![[0_u8; TMEM_CELL_BYTES]; row_bytes / TMEM_CELL_BYTES];
        physical
            .tmem()
            .read_lane_cells_into(&view, location.lane, location.column, &mut cells)?;
        let start = execution_lane
            .checked_mul(row_bytes)
            .ok_or_else(|| EngineError::message("canonical TCGEN LD row offset overflow"))?;
        payload[start..start + row_bytes].copy_from_slice(cells.as_flattened());
    }

    let (destination_memory, destination_allocations, destination_offset, destination_len) =
        match destination {
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
                    "canonical tcgen05.ld.32x32b requires a lane-private register destination",
                ));
            }
        };
    let destination_allocation = destination_allocations
        .get(context.global_warp_id())
        .ok_or_else(|| EngineError::message("warp-private allocation is missing"))?;
    destination_memory.write_lane_bytes_batch(
        WarpId::from_context(*context),
        destination_allocation,
        destination_offset,
        destination_len,
        &WarpValue::splat(0),
        mask,
        row_bytes,
        row_bytes,
        &payload,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn copy_tmem_f32_m64_to_local_warp(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    source: &RuntimeBuffer,
    destination: &RuntimeBuffer,
    base_lanes: &WarpValue<i64>,
    base_tcols: &WarpValue<i64>,
    allocated_addrs: &WarpValue<i64>,
    num: usize,
    mask: WarpMask,
) -> Result<(), EngineError> {
    super::require_full_warp_sync(mask, "fast m64 tcgen05.ld")?;
    let base_lane = require_uniform_i64(base_lanes, mask, "fast m64 TCGEN TLane")?;
    let base_tcol = require_uniform_i64(base_tcols, mask, "fast m64 TCGEN TCol")?;
    let allocated_addr =
        require_uniform_i64(allocated_addrs, mask, "fast m64 TCGEN allocated_addr")?;
    if !matches!(num, 1 | 8) {
        return Err(EngineError::message(format!(
            "canonical m64 tcgen05.ld requires .x1 or .x8, got .x{num}"
        )));
    }
    let columns = num
        .checked_mul(8)
        .ok_or_else(|| EngineError::message("canonical m64 TCGEN column count overflow"))?;
    let registers_per_lane = num
        .checked_mul(4)
        .ok_or_else(|| EngineError::message("canonical m64 TCGEN register count overflow"))?;
    let warp_lane_offset = (context.warp_id_in_cta() % 4)
        .checked_mul(32)
        .ok_or_else(|| EngineError::message("fast m64 TCGEN warp lane offset overflow"))?;
    let (view, base_lane, first_column) = fast_tmem_f32_view(
        physical,
        context,
        lifecycle,
        access_mode,
        source,
        base_lane,
        warp_lane_offset + 15,
        base_tcol,
        allocated_addr,
        columns,
    )?;
    let mut output = vec![0.0_f32; WARP_SIZE * registers_per_lane];
    let mut cells = vec![[0_u8; size_of::<f32>()]; columns];
    for lane_group in 0..8 {
        for lane_bank in 0..2 {
            let mapped_lane = base_lane + warp_lane_offset + lane_bank * 8 + lane_group;
            physical
                .tmem()
                .read_lane_cells_into(&view, mapped_lane, first_column, &mut cells)?;
            for lane_in_group in 0..4 {
                let execution_lane = lane_group * 4 + lane_in_group;
                for block in 0..num {
                    let first_slot = block * 4 + lane_bank * 2;
                    let first_cell = block * 8 + lane_in_group * 2;
                    output[execution_lane * registers_per_lane + first_slot] =
                        f32::from_le_bytes(cells[first_cell]);
                    output[execution_lane * registers_per_lane + first_slot + 1] =
                        f32::from_le_bytes(cells[first_cell + 1]);
                }
            }
        }
    }
    write_fast_tmem_f32_rows(physical, context, destination, &output, registers_per_lane)
}

pub fn get_tmem_addr(encoded: u32, row_offset: i32, column_offset: u32) -> u32 {
    let row = ((encoded >> 16) & 0xffff).wrapping_add(row_offset as u32) & 0xffff;
    let column = (encoded & 0xffff).wrapping_add(column_offset) & 0xffff;
    (row << 16) | column
}

#[derive(Clone)]
struct TmemElementLocation<'a> {
    allocation: &'a TmemAllocation,
    owner: CtaId,
    lane: usize,
    column: usize,
    byte_in_cell: usize,
}

#[derive(Clone, Copy)]
struct TmemElementAddress {
    mapped_lane: i64,
    tcol_element: i64,
    allocated_addr: i64,
    access_bytes: usize,
    execution_lane: usize,
}

type TmemAccessValidation<'a> = Option<(&'a TcgenLifecycleHub, TmemAccessMode)>;

#[derive(Clone, Copy)]
enum TmemAccessWidth {
    Scalar,
    ContiguousSpan,
}

fn resolve_tmem_element_at_cta<'a>(
    context: &WarpContext,
    validation: TmemAccessValidation<'_>,
    buffer: &'a RuntimeBuffer,
    target_cta_id_in_cluster: usize,
    address: TmemElementAddress,
) -> Result<TmemElementLocation<'a>, EngineError> {
    resolve_tmem_element_at_cta_with_width(
        context,
        validation,
        buffer,
        target_cta_id_in_cluster,
        address,
        TmemAccessWidth::Scalar,
    )
}

fn resolve_tmem_span_at_cta<'a>(
    context: &WarpContext,
    validation: TmemAccessValidation<'_>,
    buffer: &'a RuntimeBuffer,
    target_cta_id_in_cluster: usize,
    address: TmemElementAddress,
) -> Result<TmemElementLocation<'a>, EngineError> {
    resolve_tmem_element_at_cta_with_width(
        context,
        validation,
        buffer,
        target_cta_id_in_cluster,
        address,
        TmemAccessWidth::ContiguousSpan,
    )
}

fn resolve_tmem_element_at_cta_with_width<'a>(
    context: &WarpContext,
    validation: TmemAccessValidation<'_>,
    buffer: &'a RuntimeBuffer,
    target_cta_id_in_cluster: usize,
    address: TmemElementAddress,
    width: TmemAccessWidth,
) -> Result<TmemElementLocation<'a>, EngineError> {
    resolve_tmem_element_at_cta_validated(
        context,
        buffer,
        target_cta_id_in_cluster,
        address,
        width,
        |column, byte_in_cell, access_bytes| match validation {
            Some((lifecycle, access_mode)) => {
                let column_count = tmem_access_column_count(byte_in_cell, access_bytes)?;
                lifecycle
                    .validate_access_range_at_cta_exact(
                        *context,
                        target_cta_id_in_cluster,
                        column,
                        column_count,
                        access_mode,
                    )
                    .map_err(EngineError::from)
            }
            None => Ok(()),
        },
    )
}

/// Lifecycle validation for the many TMEM spans of one instruction: each
/// span is validated immediately and in order, exactly like
/// [`resolve_tmem_physical_span_at_cta`], but the lifecycle lock is taken
/// once per target CTA instead of once per span.
pub(crate) struct TmemLifecycleValidator<'a> {
    lifecycle: &'a crate::tcgen::TcgenLifecycleHub,
    mode: TmemAccessMode,
    snapshots: Vec<(usize, usize, crate::tcgen::TcgenCtaSnapshot)>,
}

impl<'a> TmemLifecycleValidator<'a> {
    pub(crate) fn new(
        lifecycle: &'a crate::tcgen::TcgenLifecycleHub,
        mode: TmemAccessMode,
    ) -> Self {
        Self {
            lifecycle,
            mode,
            snapshots: Vec::new(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_tmem_physical_span_at_cta_cached(
    context: &WarpContext,
    validator: &mut TmemLifecycleValidator<'_>,
    buffer: &RuntimeBuffer,
    target_cta_id_in_cluster: usize,
    mapped_lane: i64,
    tcol_element: i64,
    allocated_addr: i64,
    access_bytes: usize,
    execution_lane: usize,
) -> Result<PhysicalByteSpan, EngineError> {
    let TmemLifecycleValidator {
        lifecycle,
        mode,
        snapshots,
    } = validator;
    let location = resolve_tmem_element_at_cta_validated(
        context,
        buffer,
        target_cta_id_in_cluster,
        TmemElementAddress {
            mapped_lane,
            tcol_element,
            allocated_addr,
            access_bytes,
            execution_lane,
        },
        TmemAccessWidth::ContiguousSpan,
        |column, byte_in_cell, access_bytes| {
            let column_count = tmem_access_column_count(byte_in_cell, access_bytes)?;
            lifecycle
                .validate_access_range_at_cta_exact_cached(
                    *context,
                    target_cta_id_in_cluster,
                    column,
                    column_count,
                    *mode,
                    snapshots,
                )
                .map_err(EngineError::from)
        },
    )?;
    tmem_location_physical_span(&location, access_bytes)
}

fn tmem_access_column_count(
    byte_in_cell: usize,
    access_bytes: usize,
) -> Result<usize, EngineError> {
    byte_in_cell
        .checked_add(access_bytes)
        .and_then(|bytes| bytes.checked_add(3))
        .map(|bytes| bytes / 4)
        .ok_or_else(|| EngineError::message("TMEM access column span overflow"))
}

fn resolve_tmem_element_at_cta_validated<'a>(
    context: &WarpContext,
    buffer: &'a RuntimeBuffer,
    target_cta_id_in_cluster: usize,
    address: TmemElementAddress,
    width: TmemAccessWidth,
    validate: impl FnOnce(usize, usize, usize) -> Result<(), EngineError>,
) -> Result<TmemElementLocation<'a>, EngineError> {
    let TmemElementAddress {
        mapped_lane,
        tcol_element,
        allocated_addr,
        access_bytes,
        execution_lane,
    } = address;
    let RuntimeBuffer::Tmem {
        allocations,
        lane_span,
        tcol_span_elements,
        elem_offset,
        itemsize,
    } = buffer
    else {
        return Err(EngineError::message("TMEM access used a non-TMEM buffer"));
    };
    match width {
        TmemAccessWidth::Scalar if access_bytes != *itemsize => {
            return Err(EngineError::message(format!(
                "TMEM scalar width {access_bytes} does not match buffer itemsize {itemsize}"
            )));
        }
        TmemAccessWidth::ContiguousSpan if access_bytes == 0 || access_bytes % *itemsize != 0 => {
            return Err(EngineError::message(format!(
                "TMEM contiguous span width {access_bytes} is not a positive multiple of buffer itemsize {itemsize}"
            )));
        }
        _ => {}
    }
    let mapped_lane = usize::try_from(mapped_lane).map_err(|_| {
        EngineError::message(format!(
            "negative TMEM TLane {mapped_lane} on execution lane {execution_lane}"
        ))
    })?;
    if mapped_lane >= *lane_span {
        return Err(EngineError::message(format!(
            "TMEM TLane {mapped_lane} is outside layout span {lane_span} on execution lane {execution_lane}"
        )));
    }
    let tcol_element = usize::try_from(tcol_element).map_err(|_| {
        EngineError::message(format!(
            "negative TMEM TCol element {tcol_element} on execution lane {execution_lane}"
        ))
    })?;
    if tcol_element >= *tcol_span_elements {
        return Err(EngineError::message(format!(
            "TMEM TCol element {tcol_element} is outside layout span {tcol_span_elements} on execution lane {execution_lane}"
        )));
    }
    if matches!(width, TmemAccessWidth::ContiguousSpan) {
        let element_count = access_bytes / *itemsize;
        let tcol_end = tcol_element
            .checked_add(element_count)
            .ok_or_else(|| EngineError::message("TMEM contiguous TCol span overflow"))?;
        if tcol_end > *tcol_span_elements {
            return Err(EngineError::message(format!(
                "TMEM TCol span [{tcol_element}, {tcol_end}) exceeds layout span {tcol_span_elements} on execution lane {execution_lane}"
            )));
        }
    }
    let allocated_addr = usize::try_from(allocated_addr).map_err(|_| {
        EngineError::message(format!(
            "negative TMEM allocated_addr {allocated_addr} on execution lane {execution_lane}"
        ))
    })?;
    let element = elem_offset
        .checked_add(tcol_element)
        .ok_or_else(|| EngineError::message("TMEM element offset overflow"))?;
    let bit_offset = element
        .checked_mul(*itemsize)
        .and_then(|value| value.checked_mul(8))
        .ok_or_else(|| EngineError::message("TMEM bit offset overflow"))?;
    let column = allocated_addr
        .checked_add(bit_offset / 32)
        .ok_or_else(|| EngineError::message("TMEM column offset overflow"))?;
    let byte_in_cell = (bit_offset % 32) / 8;
    let topology = context.topology();
    let target = CtaId::new(topology, context.cluster_id(), target_cta_id_in_cluster)?;
    let target_global = target.global_cta_id(topology)?;
    validate(column, byte_in_cell, access_bytes)?;
    let allocation = allocations
        .get(target_global)
        .ok_or_else(|| EngineError::message("TMEM CTA allocation is missing"))?;
    Ok(TmemElementLocation {
        allocation,
        owner: target,
        lane: mapped_lane,
        column,
        byte_in_cell,
    })
}

fn resolve_tmem_element<'a>(
    context: &WarpContext,
    validation: TmemAccessValidation<'_>,
    buffer: &'a RuntimeBuffer,
    address: TmemElementAddress,
) -> Result<TmemElementLocation<'a>, EngineError> {
    resolve_tmem_element_at_cta(
        context,
        validation,
        buffer,
        context.cta_id_in_cluster(),
        address,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn resolve_tmem_physical_access(
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    buffer: &RuntimeBuffer,
    mapped_lane: i64,
    tcol_element: i64,
    allocated_addr: i64,
    access_bytes: usize,
    execution_lane: usize,
) -> Result<PhysicalByteSpan, EngineError> {
    resolve_tmem_physical_access_at_cta(
        context,
        lifecycle,
        access_mode,
        buffer,
        context.cta_id_in_cluster(),
        mapped_lane,
        tcol_element,
        allocated_addr,
        access_bytes,
        execution_lane,
    )
}

/// Validate one projected warp-scalar TMEM access without reading or writing data.
#[allow(clippy::too_many_arguments)]
pub fn validate_tmem_scalar_warp_access(
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    buffer: &RuntimeBuffer,
    mapped_lanes: &WarpValue<i64>,
    tcol_elements: &WarpValue<i64>,
    allocated_addrs: &WarpValue<i64>,
    access_bytes: usize,
    mask: WarpMask,
) -> Result<(), EngineError> {
    for execution_lane in mask {
        resolve_tmem_physical_access(
            context,
            lifecycle,
            access_mode,
            buffer,
            mapped_lanes[execution_lane],
            tcol_elements[execution_lane],
            allocated_addrs[execution_lane],
            access_bytes,
            execution_lane,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn resolve_tmem_physical_access_at_cta(
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    buffer: &RuntimeBuffer,
    target_cta_id_in_cluster: usize,
    mapped_lane: i64,
    tcol_element: i64,
    allocated_addr: i64,
    access_bytes: usize,
    execution_lane: usize,
) -> Result<PhysicalByteSpan, EngineError> {
    let location = resolve_tmem_element_at_cta(
        context,
        Some((lifecycle, access_mode)),
        buffer,
        target_cta_id_in_cluster,
        TmemElementAddress {
            mapped_lane,
            tcol_element,
            allocated_addr,
            access_bytes,
            execution_lane,
        },
    )?;
    tmem_location_physical_span(&location, access_bytes)
}

#[allow(clippy::too_many_arguments)]
pub fn resolve_tmem_physical_span(
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    buffer: &RuntimeBuffer,
    mapped_lane: i64,
    tcol_element: i64,
    allocated_addr: i64,
    access_bytes: usize,
    execution_lane: usize,
) -> Result<PhysicalByteSpan, EngineError> {
    resolve_tmem_physical_span_at_cta(
        context,
        lifecycle,
        access_mode,
        buffer,
        context.cta_id_in_cluster(),
        mapped_lane,
        tcol_element,
        allocated_addr,
        access_bytes,
        execution_lane,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn resolve_tmem_physical_span_at_cta(
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    buffer: &RuntimeBuffer,
    target_cta_id_in_cluster: usize,
    mapped_lane: i64,
    tcol_element: i64,
    allocated_addr: i64,
    access_bytes: usize,
    execution_lane: usize,
) -> Result<PhysicalByteSpan, EngineError> {
    let location = resolve_tmem_span_at_cta(
        context,
        Some((lifecycle, access_mode)),
        buffer,
        target_cta_id_in_cluster,
        TmemElementAddress {
            mapped_lane,
            tcol_element,
            allocated_addr,
            access_bytes,
            execution_lane,
        },
    )?;
    tmem_location_physical_span(&location, access_bytes)
}

fn tmem_location_physical_span(
    location: &TmemElementLocation<'_>,
    access_bytes: usize,
) -> Result<PhysicalByteSpan, EngineError> {
    let bytes_per_lane = location
        .allocation
        .columns()
        .checked_mul(TMEM_CELL_BYTES)
        .ok_or(AddressSpaceError::SizeOverflow)?;
    let lane_relative = location
        .column
        .checked_mul(TMEM_CELL_BYTES)
        .and_then(|base| base.checked_add(location.byte_in_cell))
        .ok_or(AddressSpaceError::SizeOverflow)?;
    let lane_end = lane_relative
        .checked_add(access_bytes)
        .ok_or(AddressSpaceError::SizeOverflow)?;
    if location.lane >= location.allocation.lanes() || lane_end > bytes_per_lane {
        let column_count = location
            .byte_in_cell
            .checked_add(access_bytes)
            .and_then(|bytes| bytes.checked_add(TMEM_CELL_BYTES - 1))
            .map(|bytes| bytes / TMEM_CELL_BYTES)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        return Err(AddressSpaceError::TmemRegionOutOfBounds {
            allocation: location.allocation.allocation(),
            lanes: location.allocation.lanes(),
            columns: location.allocation.columns(),
            lane_offset: location.lane,
            lane_count: 1,
            column_offset: location.column,
            column_count,
        }
        .into());
    }
    let absolute = location
        .lane
        .checked_mul(bytes_per_lane)
        .and_then(|base| base.checked_add(lane_relative))
        .ok_or(AddressSpaceError::SizeOverflow)?;
    PhysicalByteSpan::new(
        PhysicalAllocationId::from(location.allocation.allocation()),
        absolute,
        access_bytes,
    )
    .map_err(|error| EngineError::message(error.to_string()))
}

fn read_tmem_element_bytes_into(
    physical: &PhysicalMemory,
    location: &TmemElementLocation<'_>,
    output: &mut [u8],
) -> Result<(), EngineError> {
    let mut consumed = 0;
    let mut column = location.column;
    let mut byte_in_cell = location.byte_in_cell;
    while consumed < output.len() {
        let chunk = (4 - byte_in_cell).min(output.len() - consumed);
        physical.tmem().read_allocation_cell_bytes_into(
            location.owner,
            location.allocation,
            location.lane,
            column,
            byte_in_cell,
            &mut output[consumed..consumed + chunk],
        )?;
        consumed += chunk;
        column = column
            .checked_add(1)
            .ok_or_else(|| EngineError::message("TMEM column increment overflow"))?;
        byte_in_cell = 0;
    }
    Ok(())
}

fn write_tmem_element_bytes(
    physical: &PhysicalMemory,
    location: &TmemElementLocation<'_>,
    bytes: &[u8],
) -> Result<(), EngineError> {
    let mut consumed = 0;
    let mut column = location.column;
    let mut byte_in_cell = location.byte_in_cell;
    while consumed < bytes.len() {
        let chunk = (4 - byte_in_cell).min(bytes.len() - consumed);
        physical.tmem().write_allocation_cell_bytes(
            location.owner,
            location.allocation,
            location.lane,
            column,
            byte_in_cell,
            &bytes[consumed..consumed + chunk],
        )?;
        consumed += chunk;
        column = column
            .checked_add(1)
            .ok_or_else(|| EngineError::message("TMEM column increment overflow"))?;
        byte_in_cell = 0;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn load_tmem_scalar_at_cta<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    buffer: &RuntimeBuffer,
    target_cta_id_in_cluster: usize,
    mapped_lane: i64,
    tcol_element: i64,
    allocated_addr: i64,
    execution_lane: usize,
) -> Result<T, EngineError> {
    let location = resolve_tmem_element_at_cta(
        context,
        Some((lifecycle, access_mode)),
        buffer,
        target_cta_id_in_cluster,
        TmemElementAddress {
            mapped_lane,
            tcol_element,
            allocated_addr,
            access_bytes: T::BYTE_LEN,
            execution_lane,
        },
    )?;
    let mut bytes = [0_u8; MAX_RUNTIME_SCALAR_BYTES];
    read_tmem_element_bytes_into(physical, &location, &mut bytes[..T::BYTE_LEN])?;
    T::decode_le(&bytes[..T::BYTE_LEN])
}

/// Store one typed TMEM element at an explicitly selected CTA. Typed tile
/// façades use this after their frontend-supplied pure map has resolved
/// `(TLane, TCol, allocated_addr)`; lifecycle validation remains here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn store_tmem_scalar_at_cta<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    buffer: &RuntimeBuffer,
    target_cta_id_in_cluster: usize,
    mapped_lane: i64,
    tcol_element: i64,
    allocated_addr: i64,
    value: T,
    execution_lane: usize,
) -> Result<(), EngineError> {
    let location = resolve_tmem_element_at_cta(
        context,
        Some((lifecycle, access_mode)),
        buffer,
        target_cta_id_in_cluster,
        TmemElementAddress {
            mapped_lane,
            tcol_element,
            allocated_addr,
            access_bytes: T::BYTE_LEN,
            execution_lane,
        },
    )?;
    let mut bytes = [0_u8; MAX_RUNTIME_SCALAR_BYTES];
    value.encode_le_into(&mut bytes[..T::BYTE_LEN]);
    write_tmem_element_bytes(physical, &location, &bytes[..T::BYTE_LEN])
}

#[allow(clippy::too_many_arguments)]
pub fn store_tcgen_mma_f32_matrix_at_cta(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    buffer: &RuntimeBuffer,
    target_cta_id_in_cluster: usize,
    first_tcol_element: i64,
    allocated_addr: i64,
    rows: usize,
    columns: usize,
    cta_group: usize,
    values: &[f32],
) -> Result<(), EngineError> {
    if !matches!(rows, 64 | 128)
        || !matches!(cta_group, 1 | 2)
        || values.len() != rows.saturating_mul(columns)
    {
        return Err(EngineError::message(
            "TCGEN f32 matrix store requires 64 or 128 complete rows and CTA group 1 or 2",
        ));
    }
    let layout_b = rows == 64 && cta_group == 2;
    if layout_b && columns % 2 != 0 {
        return Err(EngineError::message(
            "TCGEN Layout B f32 matrix store requires an even column count",
        ));
    }
    let physical_columns = if layout_b { columns / 2 } else { columns };
    let RuntimeBuffer::Tmem {
        allocations,
        lane_span,
        tcol_span_elements,
        elem_offset,
        itemsize,
    } = buffer
    else {
        return Err(EngineError::message(
            "dense TCGEN f32 matrix store used a non-TMEM buffer",
        ));
    };
    if *itemsize != 4 {
        return Err(EngineError::message(
            "TCGEN f32 matrix store requires four-byte TMEM elements",
        ));
    }
    let first_tcol = usize::try_from(first_tcol_element)
        .map_err(|_| EngineError::message("TCGEN matrix TCol is negative"))?;
    let end_tcol = first_tcol
        .checked_add(physical_columns)
        .ok_or_else(|| EngineError::message("TCGEN matrix TCol overflow"))?;
    if end_tcol > *tcol_span_elements {
        return Err(EngineError::message(
            "TCGEN matrix exceeds the buffer TCol span",
        ));
    }
    let allocated_addr = usize::try_from(allocated_addr)
        .map_err(|_| EngineError::message("TCGEN allocated_addr is negative"))?;
    let first_column = allocated_addr
        .checked_add(*elem_offset)
        .and_then(|value| value.checked_add(first_tcol))
        .ok_or_else(|| EngineError::message("TCGEN matrix column overflow"))?;
    lifecycle.validate_access_range_at_cta(
        *context,
        target_cta_id_in_cluster,
        first_column,
        physical_columns,
        access_mode,
    )?;
    let topology = context.topology();
    let target = CtaId::new(topology, context.cluster_id(), target_cta_id_in_cluster)?;
    let target_global = target.global_cta_id(topology)?;
    let allocation = allocations
        .get(target_global)
        .ok_or_else(|| EngineError::message("TMEM CTA allocation is missing"))?;
    let lanes = (0..if layout_b { 128 } else { rows })
        .map(|row| {
            if rows == 64 && !layout_b {
                (row / 16) * 32 + row % 16
            } else {
                row
            }
        })
        .collect::<Vec<_>>();
    if lanes.iter().any(|&lane| lane >= *lane_span) {
        return Err(EngineError::message(
            "TCGEN matrix row exceeds the buffer TLane span",
        ));
    }
    let physical_values;
    let values = if layout_b {
        physical_values = {
            let mut reordered = Vec::with_capacity(values.len());
            for half in 0..2 {
                for row in 0..rows {
                    let start = row * columns + half * physical_columns;
                    reordered.extend_from_slice(
                        values.get(start..start + physical_columns).ok_or_else(|| {
                            EngineError::message("TCGEN Layout B logical row is incomplete")
                        })?,
                    );
                }
            }
            reordered
        };
        physical_values.as_slice()
    } else {
        values
    };
    physical.tmem().write_allocation_f32_rows(
        target,
        allocation,
        &lanes,
        first_column,
        values,
        physical_columns,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn store_tmem_f32_matrix_at_cta(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    buffer: &RuntimeBuffer,
    target_cta_id_in_cluster: usize,
    first_tcol_element: i64,
    allocated_addr: i64,
    rows: usize,
    columns: usize,
    values: &[f32],
) -> Result<(), EngineError> {
    store_tcgen_mma_f32_matrix_at_cta(
        physical,
        context,
        lifecycle,
        access_mode,
        buffer,
        target_cta_id_in_cluster,
        first_tcol_element,
        allocated_addr,
        rows,
        columns,
        1,
        values,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn load_tmem_scalar_warp<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    buffer: &RuntimeBuffer,
    mapped_lanes: &WarpValue<i64>,
    tcol_elements: &WarpValue<i64>,
    allocated_addrs: &WarpValue<i64>,
    mask: WarpMask,
) -> Result<WarpValue<T>, EngineError> {
    let mut values = WarpValue::splat(T::zero());
    let mut bytes = [0_u8; MAX_RUNTIME_SCALAR_BYTES];
    for lane in mask {
        let location = resolve_tmem_element(
            context,
            Some((lifecycle, access_mode)),
            buffer,
            TmemElementAddress {
                mapped_lane: mapped_lanes[lane],
                tcol_element: tcol_elements[lane],
                allocated_addr: allocated_addrs[lane],
                access_bytes: T::BYTE_LEN,
                execution_lane: lane,
            },
        )?;
        read_tmem_element_bytes_into(physical, &location, &mut bytes[..T::BYTE_LEN])?;
        values[lane] = T::decode_le(&bytes[..T::BYTE_LEN])?;
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
pub fn store_tmem_scalar_warp<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    buffer: &RuntimeBuffer,
    mapped_lanes: &WarpValue<i64>,
    tcol_elements: &WarpValue<i64>,
    allocated_addrs: &WarpValue<i64>,
    values: &WarpValue<T>,
    mask: WarpMask,
) -> Result<(), EngineError> {
    let mut locations = WarpValue::splat(None);
    for lane in mask {
        locations[lane] = Some(resolve_tmem_element(
            context,
            Some((lifecycle, access_mode)),
            buffer,
            TmemElementAddress {
                mapped_lane: mapped_lanes[lane],
                tcol_element: tcol_elements[lane],
                allocated_addr: allocated_addrs[lane],
                access_bytes: T::BYTE_LEN,
                execution_lane: lane,
            },
        )?);
    }
    let mut bytes = [0_u8; MAX_RUNTIME_SCALAR_BYTES];
    for lane in mask {
        values[lane].encode_le_into(&mut bytes[..T::BYTE_LEN]);
        write_tmem_element_bytes(
            physical,
            locations[lane]
                .as_ref()
                .expect("active TMEM lane has a resolved location"),
            &bytes[..T::BYTE_LEN],
        )?;
    }
    Ok(())
}
