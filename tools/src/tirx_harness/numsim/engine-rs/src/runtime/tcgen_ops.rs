mod integer;
pub(crate) use integer::{
    raw_tcgen05_integer_shape, raw_tcgen05_integer_shared_footprints, raw_tcgen05_mma_integer,
    RawTcgenIntegerKind,
};
#[derive(Clone, Copy)]
pub(crate) enum RawTcgenMmaA {
    Shared(u64),
    Tmem(u32),
}
/// PTX zero-column descriptor. One immutable mapping owns both B addresses and
/// the numeric zero mask; this is not collector state.
#[derive(Clone, Copy)]
pub(crate) struct RawTcgenColumnMask {
    bits: u64,
    bank_columns: usize,
}

impl RawTcgenColumnMask {
    pub(crate) fn new(
        bits: u64,
        m: usize,
        n: usize,
        instruction: u32,
    ) -> Result<Self, EngineError> {
        let maximum_shift = match instruction >> 30 {
            0 => 0,
            code => 4 << code,
        };
        if !matches!(m, 32 | 64 | 128)
            || n == 0
            || !n.is_multiple_of(128 / m)
            || bits & 0xc000_0070_0000_0000 != 0
            || ((bits >> 56) & 63) > if m == 32 { 16 } else { 32 }
            || ((bits >> 56) & 63) > maximum_shift
        {
            return Err(EngineError::message(
                "invalid TCGEN zero-column mask descriptor",
            ));
        }
        Ok(Self {
            bits,
            bank_columns: n / (128 / m),
        })
    }

    pub(crate) fn source_column(self, column: usize) -> Option<usize> {
        let bank = column / self.bank_columns;
        if self.bits & (1 << 39) != 0 {
            let start = ((self.bits >> (8 * bank)) & 255) as usize;
            let first_zero = self.bits & (1 << (32 + bank)) != 0;
            // PTX examples 2--4 and the SM100 oracle: use-span gives the
            // number of B columns, skip-span gives the number of zero columns.
            let used = ((self.bits >> 48) & 255) as usize + 1;
            let zero = ((self.bits >> 40) & 255) as usize + 1;
            // The initial-span counter saturates at one remaining column;
            // SM100's runtime-descriptor oracle covers start >= first span.
            let start = start.min(if first_zero { zero } else { used } - 1);
            let position = (column % self.bank_columns + start) % (used + zero);
            if if first_zero {
                position < zero
            } else {
                position >= used
            } {
                return None;
            }
        }
        Some(column + ((self.bits >> 56) & 63) as usize)
    }
}

#[test]
fn zero_column_mask_shift_contract() {
    let shifted = 2_u64 << 56;
    assert!(RawTcgenColumnMask::new(shifted, 128, 64, 0).is_err());
    assert_eq!(
        RawTcgenColumnMask::new(shifted, 128, 64, 1 << 30)
            .unwrap()
            .source_column(5),
        Some(7)
    );
    assert!(RawTcgenColumnMask::new(17 << 56, 32, 64, 3 << 30).is_err());
    assert!(RawTcgenColumnMask::new(1 << 36, 128, 64, 0).is_err());
    assert!(RawTcgenColumnMask::new(0, 32, 0, 0).is_err());
    let saturated = (1 << 39) | (2 << 40) | (3 << 48) | (1 << 32) | 255;
    let mask = RawTcgenColumnMask::new(saturated, 128, 64, 0).unwrap();
    assert_eq!(
        (0..8)
            .map(|i| mask.source_column(i).is_none())
            .collect::<Vec<_>>(),
        [true, false, false, false, false, true, true, true]
    );
}

#[cfg(feature = "python")]
use crate::numpy_backend::{matmul_f32_abt, F32Matrix};
use crate::profile::{ProfileKind, ProfileTimer};
use crate::{
    bf16_bits_to_f32, f32_to_fp16_bits, float4_e2m1fn_bits_to_f32, float8_e4m3fn_bits_to_f32,
    float8_e8m0fnu_bits_to_f32, fp16_bits_to_f32, narrow_float_bits_to_f32_checked, CtaId,
    EngineError, NarrowFloatFormat, PhysicalMemory, SharedView, TcgenLifecycleHub, TmemAccessMode,
    TmemView, WarpContext, WarpMask, FLOAT4_E2M1, FLOAT6_E2M3, FLOAT6_E3M2, FLOAT8_E4M3,
    FLOAT8_E5M2,
};

fn raw_tcgen05_tf32_payload_to_f32(bits: u32) -> f32 {
    f32::from_bits(bits & 0xffff_e000)
}

use super::{
    read_runtime_bytes, runtime_buffer_base, runtime_buffer_byte_len,
    runtime_buffer_readable_lanes, store_tmem_f32_matrix_at_cta, write_runtime_bytes_with_validity,
    PhysicalPtr, RuntimeBuffer,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileGemmOperandLayout {
    atom_columns: usize,
    per_element_shift: u32,
    outer_mask: usize,
    atom_shift: u32,
}

impl TileGemmOperandLayout {
    pub const fn new(
        atom_columns: usize,
        per_element_shift: u32,
        outer_mask: usize,
        atom_shift: u32,
    ) -> Self {
        Self {
            atom_columns,
            per_element_shift,
            outer_mask,
            atom_shift,
        }
    }
}

pub type TileGemmBf16OperandLayout = TileGemmOperandLayout;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileGemmBf16Descriptor {
    m: usize,
    n: usize,
    k: usize,
    a_layout: TileGemmOperandLayout,
    b_layout: TileGemmOperandLayout,
    reuse_a_as_b: bool,
}

impl TileGemmBf16Descriptor {
    pub const fn new(
        m: usize,
        n: usize,
        k: usize,
        a_layout: TileGemmOperandLayout,
        b_layout: TileGemmOperandLayout,
        reuse_a_as_b: bool,
    ) -> Self {
        Self {
            m,
            n,
            k,
            a_layout,
            b_layout,
            reuse_a_as_b,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawTcgenLdstShape {
    Shape16x32bx2(usize),
    Shape16x64b,
    Shape16x128b,
    Shape32x32b,
    Shape16x256b,
}

fn tile_gemm_operand_physical_elements(
    rows: usize,
    columns: usize,
    layout: TileGemmOperandLayout,
) -> Result<Vec<usize>, EngineError> {
    let element_count = rows
        .checked_mul(columns)
        .ok_or_else(|| EngineError::message("tile GEMM matrix shape overflows usize"))?;
    if rows == 0
        || columns == 0
        || layout.atom_columns == 0
        || columns % layout.atom_columns != 0
        || layout.per_element_shift >= usize::BITS
        || layout.atom_shift >= usize::BITS
    {
        return Err(EngineError::message("invalid tile GEMM operand layout"));
    }
    let element_group = 1_usize << layout.per_element_shift;
    if element_count % element_group != 0 {
        return Err(EngineError::message(
            "tile GEMM operand layout does not divide its element domain",
        ));
    }
    let quotient_domain = element_count / element_group;
    if layout.outer_mask >= quotient_domain {
        return Err(EngineError::message(
            "tile GEMM operand swizzle exceeds its element domain",
        ));
    }

    let element_mask = element_group - 1;
    let column_tiles = columns / layout.atom_columns;
    let mut physical_elements = Vec::with_capacity(element_count);
    for row in 0..rows {
        for column_tile in 0..column_tiles {
            let physical_tile_base = column_tile
                .checked_mul(rows)
                .and_then(|value| value.checked_mul(layout.atom_columns))
                .and_then(|value| value.checked_add(row * layout.atom_columns))
                .ok_or_else(|| EngineError::message("tile GEMM operand tile offset overflow"))?;
            for inner_column in 0..layout.atom_columns {
                let unswizzled = physical_tile_base
                    .checked_add(inner_column)
                    .ok_or_else(|| EngineError::message("tile GEMM operand offset overflow"))?;
                let quotient = unswizzled >> layout.per_element_shift;
                let swizzled_quotient =
                    quotient ^ ((quotient & layout.outer_mask) >> layout.atom_shift);
                let physical =
                    (swizzled_quotient << layout.per_element_shift) | (unswizzled & element_mask);
                if physical >= element_count {
                    return Err(EngineError::message(
                        "tile GEMM operand swizzle produced an out-of-range element",
                    ));
                }
                physical_elements.push(physical);
            }
        }
    }
    Ok(physical_elements)
}

fn decode_tile_gemm_bf16_snapshot(
    snapshot: &[u8],
    rows: usize,
    columns: usize,
    layout: TileGemmOperandLayout,
) -> Result<Vec<f32>, EngineError> {
    let physical_elements = tile_gemm_operand_physical_elements(rows, columns, layout)?;
    let expected_bytes = physical_elements
        .len()
        .checked_mul(2)
        .ok_or_else(|| EngineError::message("tile BF16 GEMM snapshot size overflows usize"))?;
    if snapshot.len() != expected_bytes {
        return Err(EngineError::message(format!(
            "tile BF16 GEMM snapshot has {} bytes, expected {expected_bytes}",
            snapshot.len()
        )));
    }
    physical_elements
        .into_iter()
        .map(|physical| {
            let byte = physical * 2;
            Ok(bf16_bits_to_f32(u16::from_le_bytes([
                snapshot[byte],
                snapshot[byte + 1],
            ])))
        })
        .collect()
}

fn read_tile_gemm_bf16_operand(
    physical: &PhysicalMemory,
    context: &WarpContext,
    source: &RuntimeBuffer,
    rows: usize,
    columns: usize,
    layout: TileGemmOperandLayout,
    issuing_lane: usize,
) -> Result<Vec<f32>, EngineError> {
    let physical_elements = tile_gemm_operand_physical_elements(rows, columns, layout)?;
    let expected_bytes = physical_elements
        .len()
        .checked_mul(2)
        .ok_or_else(|| EngineError::message("tile BF16 GEMM operand size overflows usize"))?;
    let actual_bytes = runtime_buffer_byte_len(source);
    if actual_bytes != expected_bytes {
        return Err(EngineError::message(format!(
            "tile BF16 GEMM operand has {actual_bytes} bytes, expected {expected_bytes}"
        )));
    }
    let snapshot = read_runtime_bytes(physical, context, source, issuing_lane, 0, expected_bytes)?;
    decode_tile_gemm_bf16_snapshot(&snapshot, rows, columns, layout)
}

#[allow(clippy::too_many_arguments)]
pub fn tile_gemm_bf16_f32_ss_cta1(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    destination: &RuntimeBuffer,
    a_source: &RuntimeBuffer,
    b_source: &RuntimeBuffer,
    destination_first_tcol: i64,
    destination_allocated_addr: i64,
    descriptor: TileGemmBf16Descriptor,
    issuing_lane: usize,
) -> Result<(), EngineError> {
    if !matches!(descriptor.m, 64 | 128)
        || descriptor.n < 64
        || descriptor.k < 64
        || (descriptor.reuse_a_as_b
            && (descriptor.m != descriptor.n || descriptor.a_layout != descriptor.b_layout))
    {
        return Err(EngineError::message(
            "invalid tile BF16 GEMM descriptor for the dense CTA1 path",
        ));
    }

    let a = read_tile_gemm_bf16_operand(
        physical,
        context,
        a_source,
        descriptor.m,
        descriptor.k,
        descriptor.a_layout,
        issuing_lane,
    )?;
    let b = if descriptor.reuse_a_as_b {
        None
    } else {
        Some(read_tile_gemm_bf16_operand(
            physical,
            context,
            b_source,
            descriptor.n,
            descriptor.k,
            descriptor.b_layout,
            issuing_lane,
        )?)
    };
    #[cfg(feature = "python")]
    let output_values = {
        let a_matrix = F32Matrix::new(descriptor.m, descriptor.k, a)
            .map_err(|error| EngineError::message(error.to_string()))?;
        let b_matrix = if descriptor.reuse_a_as_b {
            a_matrix.clone()
        } else {
            let b = b.ok_or_else(|| EngineError::message("tile BF16 GEMM B operand is missing"))?;
            F32Matrix::new(descriptor.n, descriptor.k, b)
                .map_err(|error| EngineError::message(error.to_string()))?
        };
        matmul_f32_abt(a_matrix, b_matrix)
            .map(F32Matrix::into_values)
            .map_err(|error| EngineError::message(error.to_string()))?
    };
    #[cfg(not(feature = "python"))]
    let output_values = {
        let b_values = b.as_ref().map_or(a.as_slice(), Vec::as_slice);
        mma_f32_abt_increasing_k(descriptor.m, descriptor.n, descriptor.k, &a, b_values, None)?
    };

    store_tmem_f32_matrix_at_cta(
        physical,
        context,
        lifecycle,
        access_mode,
        destination,
        context.cta_id_in_cluster(),
        destination_first_tcol,
        destination_allocated_addr,
        descriptor.m,
        descriptor.n,
        &output_values,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn tile_gemm_bf16_f32_ss_cta1_increasing_k(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    destination: &RuntimeBuffer,
    a_source: &RuntimeBuffer,
    b_source: &RuntimeBuffer,
    destination_first_tcol: i64,
    destination_allocated_addr: i64,
    descriptor: TileGemmBf16Descriptor,
    issuing_lane: usize,
) -> Result<(), EngineError> {
    if !matches!(descriptor.m, 64 | 128)
        || descriptor.n < 64
        || descriptor.k < 64
        || (descriptor.reuse_a_as_b
            && (descriptor.m != descriptor.n || descriptor.a_layout != descriptor.b_layout))
    {
        return Err(EngineError::message(
            "invalid tile BF16 GEMM descriptor for the dense CTA1 path",
        ));
    }

    let a = read_tile_gemm_bf16_operand(
        physical,
        context,
        a_source,
        descriptor.m,
        descriptor.k,
        descriptor.a_layout,
        issuing_lane,
    )?;
    let b = if descriptor.reuse_a_as_b {
        None
    } else {
        Some(read_tile_gemm_bf16_operand(
            physical,
            context,
            b_source,
            descriptor.n,
            descriptor.k,
            descriptor.b_layout,
            issuing_lane,
        )?)
    };
    let b_values = b.as_ref().map_or(a.as_slice(), Vec::as_slice);
    let output_values =
        mma_f32_abt_increasing_k(descriptor.m, descriptor.n, descriptor.k, &a, b_values, None)?;
    store_tmem_f32_matrix_at_cta(
        physical,
        context,
        lifecycle,
        access_mode,
        destination,
        context.cta_id_in_cluster(),
        destination_first_tcol,
        destination_allocated_addr,
        descriptor.m,
        descriptor.n,
        &output_values,
    )?;
    Ok(())
}

fn raw_tcgen05_tmem_view(
    physical: &PhysicalMemory,
    context: &WarpContext,
    anchor: &RuntimeBuffer,
    target_cta_id_in_cluster: usize,
) -> Result<TmemView, EngineError> {
    let RuntimeBuffer::Tmem { allocations, .. } = anchor else {
        return Err(EngineError::message(
            "raw TCGEN transfer requires a TMEM anchor",
        ));
    };
    let target = CtaId::new(
        context.topology(),
        context.cluster_id(),
        target_cta_id_in_cluster,
    )?;
    let target_global = target.global_cta_id(context.topology())?;
    let allocation = allocations
        .get(target_global)
        .ok_or_else(|| EngineError::message("raw TCGEN target CTA has no TMEM allocation"))?;
    physical
        .tmem()
        .full_view(target, allocation)
        .map_err(EngineError::from)
}

fn validate_raw_tcgen05_tmem_columns(
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    context: &WarpContext,
    target_cta_id_in_cluster: usize,
    column: usize,
    columns: usize,
) -> Result<(), EngineError> {
    lifecycle
        .validate_access_range_at_cta(
            *context,
            target_cta_id_in_cluster,
            column,
            columns,
            access_mode,
        )
        .map_err(EngineError::from)
}

/// Decode a TMEM address and validate the column window it opens on the
/// issuing CTA.
///
/// Every raw `tcgen05.mma` operand check repeats this pair; the mnemonic rows
/// differ only in which address they hand over and how many columns it spans.
fn validate_raw_tcgen05_address_columns(
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    context: &WarpContext,
    address: u32,
    columns: usize,
) -> Result<(), EngineError> {
    let (_, column) = raw_tcgen05_address(address, 0, 0)?;
    validate_raw_tcgen05_tmem_columns(
        lifecycle,
        access_mode,
        context,
        context.cta_id_in_cluster(),
        column,
        columns,
    )
}

fn raw_tcgen05_address(
    address: u32,
    row_offset: i64,
    col_offset: i64,
) -> Result<(usize, usize), EngineError> {
    let row = (i64::from((address >> 16) & 0xffff) + row_offset).rem_euclid(1_i64 << 16);
    let col = (i64::from(address & 0xffff) + col_offset).rem_euclid(1_i64 << 16);
    Ok((
        usize::try_from(row).map_err(|_| EngineError::message("negative TCGEN row"))?,
        usize::try_from(col).map_err(|_| EngineError::message("negative TCGEN column"))?,
    ))
}

/// Decode the physical location carried by a block-scale TMEM operand.
///
/// Canonical kernels also carry the runtime SFA/SFB sub-column selector in
/// bits 30-31 of this scalar and copy that selector into the instruction
/// descriptor. Those two bits are metadata, not part of the physical lane.
fn raw_tcgen05_block_scale_address(address: u32) -> Result<(usize, usize), EngineError> {
    raw_tcgen05_address(address & !0xc000_0000_u32, 0, 0)
}

pub(crate) fn raw_tcgen05_ldst_location(
    context: &WarpContext,
    address: u32,
    row_offset: i64,
    col_offset: i64,
    shape: RawTcgenLdstShape,
    packed: bool,
    register_index: usize,
    execution_lane: usize,
) -> Result<(usize, usize), EngineError> {
    let (base_row, base_col) = raw_tcgen05_address(address, row_offset, col_offset)?;
    let warp_in_group = context.warp_id_in_cta() % 4;
    let (row_delta, col_delta) = match shape {
        RawTcgenLdstShape::Shape16x32bx2(_) => (execution_lane % 16, register_index),
        RawTcgenLdstShape::Shape16x64b => (
            (execution_lane >> 2)
                .checked_add(8 * (execution_lane & 1))
                .ok_or_else(|| EngineError::message("tcgen05.16x64b row overflow"))?,
            ((execution_lane >> 1) & 1)
                .checked_add(2 * register_index)
                .ok_or_else(|| EngineError::message("tcgen05.16x64b column overflow"))?,
        ),
        RawTcgenLdstShape::Shape16x128b => (
            (execution_lane >> 2)
                .checked_add(8 * (register_index & 1))
                .ok_or_else(|| EngineError::message("tcgen05.16x128b row overflow"))?,
            (execution_lane & 3)
                .checked_add(4 * (register_index >> 1))
                .ok_or_else(|| EngineError::message("tcgen05.16x128b column overflow"))?,
        ),
        RawTcgenLdstShape::Shape32x32b => (execution_lane, register_index),
        RawTcgenLdstShape::Shape16x256b => (
            (execution_lane >> 2)
                .checked_add(8 * ((register_index >> 1) & 1))
                .ok_or_else(|| EngineError::message("tcgen05.16x256b row overflow"))?,
            (register_index & 1)
                .checked_add(2 * (execution_lane & 3))
                .and_then(|value| value.checked_add(8 * (register_index >> 2)))
                .ok_or_else(|| EngineError::message("tcgen05.16x256b column overflow"))?,
        ),
    };
    let warp_lane_base = warp_in_group * 32;
    // TIRx has two established raw-call conventions.  Tile lowering and the
    // GPU differential corpus pass a lane within the issuing warp's 32-lane
    // TMEM partition, while hand-written PTX translations may pass the
    // absolute 0..127 TMEM lane encoded in taddr.  Normalize both conventions
    // before applying the instruction-fragment mapping.
    let normalized_base_row = if base_row < 32 {
        warp_lane_base
            .checked_add(base_row)
            .ok_or_else(|| EngineError::message("raw TCGEN warp-relative row overflow"))?
    } else if (warp_lane_base..warp_lane_base + 32).contains(&base_row) {
        base_row
    } else {
        return Err(EngineError::message(format!(
            "raw TCGEN base row {base_row} is outside warp {warp_in_group}'s accessible TMEM lanes {warp_lane_base}..{}",
            warp_lane_base + 32
        )));
    };
    let row = normalized_base_row
        .checked_add(row_delta)
        .ok_or_else(|| EngineError::message("raw TCGEN row overflow"))?;
    let physical_col_delta = if packed {
        col_delta
            .checked_mul(2)
            .ok_or_else(|| EngineError::message("raw TCGEN packed column overflow"))?
    } else {
        col_delta
    };
    // The second half starts at taddr + immHalfSplitoff, independently of
    // the repeat count and pack/unpack width of each half.
    let half_offset = match shape {
        RawTcgenLdstShape::Shape16x32bx2(offset) => (execution_lane / 16)
            .checked_mul(offset)
            .ok_or_else(|| EngineError::message("raw TCGEN half-split overflow"))?,
        _ => 0,
    };
    let column = base_col
        .checked_add(physical_col_delta)
        .and_then(|column| column.checked_add(half_offset))
        .ok_or_else(|| EngineError::message("raw TCGEN column overflow"))?;
    if row >= 128 {
        return Err(EngineError::message(format!(
            "raw TCGEN row {row} is outside 128 TMEM lanes"
        )));
    }
    let accessible_start = warp_lane_base;
    let accessible_end = accessible_start + 32;
    if !(accessible_start..accessible_end).contains(&row) {
        return Err(EngineError::message(format!(
            "raw TCGEN row {row} is outside warp {warp_in_group}'s accessible TMEM lanes {accessible_start}..{accessible_end}"
        )));
    }
    Ok((row, column))
}

#[allow(clippy::too_many_arguments)]
pub fn raw_tcgen05_ld_register(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    anchor: &RuntimeBuffer,
    destination: &PhysicalPtr,
    address: u32,
    row_offset: i64,
    col_offset: i64,
    shape: RawTcgenLdstShape,
    packed: bool,
    register_index: usize,
    mask: WarpMask,
) -> Result<(), EngineError> {
    if destination.pointee_itemsize() != 4 {
        return Err(EngineError::message(
            "raw tcgen05.ld destination register must be 32 bits",
        ));
    }
    raw_tcgen05_read_register(
        physical,
        context,
        lifecycle,
        access_mode,
        anchor,
        address,
        row_offset,
        col_offset,
        shape,
        packed,
        register_index,
        mask,
        |lane, bytes, validity| {
            let byte_offset = destination.lane_write_byte_offset(lane, 4)?;
            write_runtime_bytes_with_validity(
                physical,
                context,
                destination.buffer(),
                lane,
                byte_offset,
                &bytes,
                &validity,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn raw_tcgen05_read_register(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    anchor: &RuntimeBuffer,
    address: u32,
    row_offset: i64,
    col_offset: i64,
    shape: RawTcgenLdstShape,
    packed: bool,
    register_index: usize,
    mask: WarpMask,
    mut consume: impl FnMut(usize, [u8; 4], [bool; 4]) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    if register_index == 0 {
        physical
            .ordering()
            .tcgen_issue(*context, crate::TcgenTransferKind::Load, mask)?;
    }
    let view = raw_tcgen05_tmem_view(physical, context, anchor, context.cta_id_in_cluster())?;
    for lane in mask {
        let (tmem_lane, column) = raw_tcgen05_ldst_location(
            context,
            address,
            row_offset,
            col_offset,
            shape,
            packed,
            register_index,
            lane,
        )?;
        let column_count = if packed { 2 } else { 1 };
        validate_raw_tcgen05_tmem_columns(
            lifecycle,
            access_mode,
            context,
            context.cta_id_in_cluster(),
            column,
            column_count,
        )?;
        let mut bytes = [0_u8; 4];
        let mut validity = [false; 4];
        if packed {
            for half in 0..2 {
                let packed_column = column
                    .checked_add(half)
                    .ok_or_else(|| EngineError::message("raw TCGEN packed column overflow"))?;
                let cell_validity =
                    physical
                        .tmem()
                        .cell_byte_validity(&view, tmem_lane, packed_column)?;
                if cell_validity.len() != 4 {
                    return Err(EngineError::message(format!(
                        "raw TCGEN packed cell validity has {} bytes, expected 4",
                        cell_validity.len()
                    )));
                }
                validity[2 * half..2 * half + 2].copy_from_slice(&cell_validity[..2]);
                if cell_validity[..2].iter().all(|valid| *valid) {
                    physical.tmem().read_cell_bytes_into(
                        &view,
                        tmem_lane,
                        packed_column,
                        0,
                        &mut bytes[2 * half..2 * half + 2],
                    )?;
                }
            }
        } else {
            let cell_validity = physical
                .tmem()
                .cell_byte_validity(&view, tmem_lane, column)?;
            if cell_validity.len() != validity.len() {
                return Err(EngineError::message(format!(
                    "raw TCGEN cell validity has {} bytes, expected {}",
                    cell_validity.len(),
                    validity.len()
                )));
            }
            validity.copy_from_slice(&cell_validity);
            let mut run_start = 0;
            while run_start < bytes.len() {
                let valid = validity[run_start];
                let mut run_end = run_start + 1;
                while run_end < bytes.len() && validity[run_end] == valid {
                    run_end += 1;
                }
                if valid {
                    physical.tmem().read_cell_bytes_into(
                        &view,
                        tmem_lane,
                        column,
                        run_start,
                        &mut bytes[run_start..run_end],
                    )?;
                }
                run_start = run_end;
            }
        }
        consume(lane, bytes, validity)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn raw_tcgen05_st_register(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    anchor: &RuntimeBuffer,
    source: &crate::WarpValue<u32>,
    address: u32,
    row_offset: i64,
    col_offset: i64,
    shape: RawTcgenLdstShape,
    unpacked: bool,
    register_index: usize,
    mask: WarpMask,
) -> Result<(), EngineError> {
    if register_index == 0 {
        physical
            .ordering()
            .tcgen_issue(*context, crate::TcgenTransferKind::Store, mask)?;
    }
    let view = raw_tcgen05_tmem_view(physical, context, anchor, context.cta_id_in_cluster())?;
    for lane in mask {
        let bytes = source[lane].to_le_bytes();
        let (tmem_lane, column) = raw_tcgen05_ldst_location(
            context,
            address,
            row_offset,
            col_offset,
            shape,
            unpacked,
            register_index,
            lane,
        )?;
        let column_count = if unpacked { 2 } else { 1 };
        validate_raw_tcgen05_tmem_columns(
            lifecycle,
            access_mode,
            context,
            context.cta_id_in_cluster(),
            column,
            column_count,
        )?;
        if unpacked {
            for half in 0..2 {
                let unpacked_column = column
                    .checked_add(half)
                    .ok_or_else(|| EngineError::message("raw TCGEN unpacked column overflow"))?;
                physical.tmem().write_cell_bytes(
                    &view,
                    tmem_lane,
                    unpacked_column,
                    0,
                    &bytes[2 * half..2 * half + 2],
                )?;
            }
        } else {
            physical
                .tmem()
                .write_cell_bytes(&view, tmem_lane, column, 0, &bytes)?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct RawTcgenMatrixDescriptor {
    start_address: usize,
    leading_byte_offset: usize,
    absolute_leading_address: bool,
    stride_byte_offset: usize,
    swizzle_bits: usize,
    swizzle_atom_bytes: usize,
    swizzle_xor_shift: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RawTcgenMatrixDescriptorLayout {
    Sm100,
    Sm103,
    Sm107,
}

impl RawTcgenMatrixDescriptorLayout {
    fn supports_f8f6f4_k64(self) -> bool {
        matches!(self, Self::Sm107)
    }
}

fn decode_raw_tcgen_matrix_descriptor(
    descriptor: u64,
) -> Result<RawTcgenMatrixDescriptor, EngineError> {
    decode_raw_tcgen_matrix_descriptor_for_layout(descriptor, RawTcgenMatrixDescriptorLayout::Sm100)
}

fn decode_raw_tcgen_matrix_descriptor_for_layout(
    descriptor: u64,
    layout: RawTcgenMatrixDescriptorLayout,
) -> Result<RawTcgenMatrixDescriptor, EngineError> {
    let field_bits = match layout {
        RawTcgenMatrixDescriptorLayout::Sm100 | RawTcgenMatrixDescriptorLayout::Sm103 => 14,
        RawTcgenMatrixDescriptorLayout::Sm107 => 15,
    };
    if ((descriptor >> 46) & 0x3) != 1 {
        return Err(EngineError::message(
            "raw tcgen05.cp descriptor has an invalid version field",
        ));
    }
    let field_mask = (1_u64 << field_bits) - 1;
    let start_reserved_mask = 0xffff_u64 & !field_mask;
    let ldo_reserved_mask = start_reserved_mask << 16;
    if descriptor & start_reserved_mask != 0
        || descriptor & ldo_reserved_mask != 0
        || descriptor & (((1_u64 << 13) - 1) << 48) != 0
    {
        return Err(EngineError::message(
            "raw tcgen05.cp descriptor uses unsupported reserved/base/LBO-mode bits",
        ));
    }
    let layout_type = usize::try_from((descriptor >> 61) & 0x7)
        .map_err(|_| EngineError::message("matrix descriptor layout conversion failed"))?;
    let (swizzle_bits, swizzle_atom_bytes, swizzle_xor_shift) = match layout_type {
        0 => (0, 16, 3),
        6 => (1, 16, 3),
        4 => (2, 16, 3),
        2 => (3, 16, 3),
        1 => (2, 32, 2),
        _ => {
            return Err(EngineError::message(format!(
                "raw tcgen05.cp descriptor has invalid layout type {layout_type}"
            )));
        }
    };
    let start_address = usize::try_from(descriptor & field_mask)
        .map_err(|_| EngineError::message("matrix descriptor start conversion failed"))?
        << 4;
    if swizzle_atom_bytes == 32 && start_address % 32 != 0 {
        return Err(EngineError::message(
            "raw tcgen05.cp 128B/32B-atomic descriptor is not 32-byte aligned",
        ));
    }
    Ok(RawTcgenMatrixDescriptor {
        start_address,
        absolute_leading_address: false,
        leading_byte_offset: usize::try_from((descriptor >> 16) & field_mask)
            .map_err(|_| EngineError::message("matrix descriptor LDO conversion failed"))?
            << 4,
        stride_byte_offset: usize::try_from((descriptor >> 32) & 0x3fff)
            .map_err(|_| EngineError::message("matrix descriptor SDO conversion failed"))?
            << 4,
        swizzle_bits,
        swizzle_atom_bytes,
        swizzle_xor_shift,
    })
}

/// PTX 9.7.18.3.1.2: 48B K-major packed rows may straddle two 128B
/// swizzle chunks. Other consumers still reject bit 52 in the ordinary decoder.
fn decode_raw_tcgen_packed_matrix_descriptor(
    bits: u64,
    layout: RawTcgenMatrixDescriptorLayout,
    row_bytes: usize,
    transpose: bool,
) -> Result<RawTcgenMatrixDescriptor, EngineError> {
    let absolute = bits & (1_u64 << 52) != 0;
    let mut descriptor =
        decode_raw_tcgen_matrix_descriptor_for_layout(bits & !(1_u64 << 52), layout)?;
    if absolute
        && (layout == RawTcgenMatrixDescriptorLayout::Sm100
            || row_bytes != 48
            || transpose
            || descriptor.swizzle_bits != 3
            || descriptor.swizzle_atom_bytes != 16
            || descriptor.leading_byte_offset % 128 != 0)
    {
        return Err(EngineError::message("absolute LDO requires an SM103/SM107 48B K-major row, 128B/16B swizzle and aligned second chunk"));
    }
    descriptor.absolute_leading_address = absolute;
    Ok(descriptor)
}

fn raw_tcgen05_shared_source(
    _context: &WarpContext,
    candidates: &[&RuntimeBuffer],
    start_address: usize,
    issuing_lane: usize,
) -> Result<RuntimeBuffer, EngineError> {
    let mut root: Option<(std::sync::Arc<Vec<crate::SharedAllocation>>, usize, usize)> = None;
    let mut readable = false;
    for candidate in candidates {
        let candidate_base = runtime_buffer_base(candidate);
        let RuntimeBuffer::Shared {
            allocations,
            byte_offset,
            virtual_base,
            byte_len,
            backing_byte_len,
            ..
        } = candidate_base
        else {
            continue;
        };
        let end = virtual_base
            .checked_add(*byte_len)
            .ok_or_else(|| EngineError::message("shared virtual range overflow"))?;
        if start_address >= *virtual_base && start_address < end {
            let backing_virtual_base = virtual_base.checked_sub(*byte_offset).ok_or_else(|| {
                EngineError::message("shared view precedes its backing virtual base")
            })?;
            if let Some((selected_allocations, selected_base, selected_len)) = &root {
                if !std::sync::Arc::ptr_eq(selected_allocations, allocations)
                    || *selected_base != backing_virtual_base
                    || *selected_len != *backing_byte_len
                {
                    return Err(EngineError::message(format!(
                        "raw TCGEN descriptor address {start_address} ambiguously names multiple physical shared-memory backings"
                    )));
                }
            } else {
                root = Some((
                    std::sync::Arc::clone(allocations),
                    backing_virtual_base,
                    *backing_byte_len,
                ));
            }
            readable |= runtime_buffer_readable_lanes(candidate).contains(issuing_lane);
        }
    }
    let Some((allocations, virtual_base, byte_len)) = root else {
        return Err(EngineError::message(format!(
            "raw TCGEN descriptor address {start_address} does not name physical shared memory"
        )));
    };
    if !readable {
        return Err(EngineError::message(format!(
            "raw TCGEN descriptor reads a non-readable DeclBuffer view on lane {issuing_lane}"
        )));
    }
    Ok(RuntimeBuffer::AccessView {
        buffer: std::sync::Arc::new(RuntimeBuffer::Shared {
            allocations,
            byte_offset: 0,
            byte_len,
            backing_byte_len: byte_len,
            virtual_base,
        }),
        readable_lanes: WarpMask::from_bits(1_u32 << issuing_lane),
        writable_lanes: WarpMask::EMPTY,
    })
}

fn validate_raw_tcgen05_issuing_lane(
    context: &WarpContext,
    issuing_lane: usize,
    operation: &crate::DiagnosticLabel,
) -> Result<(), EngineError> {
    if !context.active_mask().contains(issuing_lane) {
        return Err(operation.engine_error(format_args!(
            " issuing lane {issuing_lane} is not active in mask 0x{:08x}",
            context.active_mask().bits()
        )));
    }
    Ok(())
}

/// The operand prologue every shared-memory-sourced raw `tcgen05.mma` shares:
/// decode both matrix descriptors, check the issuing lane is active under the
/// mnemonic's own diagnostic label, then resolve both descriptor addresses to
/// physical shared memory.
fn raw_tcgen05_mma_shared_operands(
    context: &WarpContext,
    shared_candidates: &[&RuntimeBuffer],
    a_descriptor_bits: u64,
    b_descriptor_bits: u64,
    issuing_lane: usize,
    label: &str,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
    mxf4_k: Option<usize>,
) -> Result<
    (
        RawTcgenMatrixDescriptor,
        RawTcgenMatrixDescriptor,
        RuntimeBuffer,
        RuntimeBuffer,
    ),
    EngineError,
> {
    let decode = |bits| match mxf4_k {
        Some(k) => decode_raw_tcgen_packed_matrix_descriptor(bits, descriptor_layout, k / 2, false),
        None => decode_raw_tcgen_matrix_descriptor_for_layout(bits, descriptor_layout),
    };
    let a_descriptor = decode(a_descriptor_bits)?;
    let b_descriptor = decode(b_descriptor_bits)?;
    validate_raw_tcgen05_issuing_lane(context, issuing_lane, &crate::DiagnosticLabel::new(label))?;
    let a_source = raw_tcgen05_shared_source(
        context,
        shared_candidates,
        a_descriptor.start_address,
        issuing_lane,
    )?;
    let b_source = raw_tcgen05_shared_source(
        context,
        shared_candidates,
        b_descriptor.start_address,
        issuing_lane,
    )?;
    Ok((a_descriptor, b_descriptor, a_source, b_source))
}

/// `2^-scale_input_d`, carrying the mnemonic's verbatim conversion diagnostic.
///
/// The `0..=15` range check stays at each mnemonic's own call site: the three
/// scaled mnemonics validate the range at different points in their operand
/// sequence, and that ordering is observable through which error surfaces
/// first.
fn raw_tcgen05_input_scale(
    scale_input_d: usize,
    conversion_message: &'static str,
) -> Result<f32, EngineError> {
    Ok(2.0_f32
        .powi(-i32::try_from(scale_input_d).map_err(|_| EngineError::message(conversion_message))?))
}

fn raw_tcgen05_shared_view_at_cta(
    physical: &PhysicalMemory,
    context: &WarpContext,
    source: &RuntimeBuffer,
    target_cta_id_in_cluster: usize,
) -> Result<SharedView, EngineError> {
    let RuntimeBuffer::Shared {
        allocations,
        byte_offset,
        byte_len,
        ..
    } = runtime_buffer_base(source)
    else {
        return Err(EngineError::message(
            "raw TCGEN descriptor source is not shared memory",
        ));
    };
    let topology = context.topology();
    let requester = CtaId::from_context(*context);
    let target = CtaId::new(topology, context.cluster_id(), target_cta_id_in_cluster)?;
    let target_global = target.global_cta_id(topology)?;
    let allocation = allocations
        .get(target_global)
        .ok_or_else(|| EngineError::message("raw TCGEN target CTA shared allocation is missing"))?;
    let view = if requester == target {
        physical
            .shared()
            .cta_view(requester, allocation, *byte_offset, *byte_len)?
    } else {
        physical
            .shared()
            .remote_cta_view(requester, target, allocation, *byte_offset, *byte_len)?
    };
    Ok(view)
}

fn raw_tcgen05_read_shared_bytes_at_cta_into(
    physical: &PhysicalMemory,
    context: &WarpContext,
    source: &RuntimeBuffer,
    target_cta_id_in_cluster: usize,
    byte_offset: usize,
    target_bytes: &mut [u8],
) -> Result<(), EngineError> {
    let view = raw_tcgen05_shared_view_at_cta(physical, context, source, target_cta_id_in_cluster)?;
    physical
        .shared()
        .read_bytes_into(&view, byte_offset, target_bytes)?;
    Ok(())
}

fn raw_tcgen05_read_shared_bytes_into(
    physical: &PhysicalMemory,
    context: &WarpContext,
    source: &RuntimeBuffer,
    byte_offset: usize,
    target_bytes: &mut [u8],
) -> Result<(), EngineError> {
    raw_tcgen05_read_shared_bytes_at_cta_into(
        physical,
        context,
        source,
        context.cta_id_in_cluster(),
        byte_offset,
        target_bytes,
    )
}

fn raw_tcgen05_shared_byte_offset(
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    row: usize,
    byte_in_row: usize,
    access_bytes: usize,
) -> Result<usize, EngineError> {
    let row_stride = descriptor
        .swizzle_atom_bytes
        .checked_shl(u32::try_from(descriptor.swizzle_bits).map_err(|_| {
            EngineError::message("matrix descriptor swizzle length conversion failed")
        })?)
        .ok_or_else(|| EngineError::message("matrix descriptor row stride overflow"))?;
    let atom = byte_in_row / descriptor.swizzle_atom_bytes;
    let byte_in_atom = byte_in_row % descriptor.swizzle_atom_bytes;
    if byte_in_atom
        .checked_add(access_bytes)
        .ok_or_else(|| EngineError::message("raw TCGEN access size overflow"))?
        > descriptor.swizzle_atom_bytes
    {
        return Err(EngineError::message(
            "raw TCGEN scalar access crosses a descriptor swizzle atom",
        ));
    }
    let column_stride = if descriptor.swizzle_bits == 0 {
        if atom != 0 && descriptor.leading_byte_offset == 0 {
            return Err(EngineError::message(
                "non-swizzled raw TCGEN descriptor needs nonzero LDO for multiple 128-bit columns",
            ));
        }
        descriptor
            .leading_byte_offset
            .max(descriptor.swizzle_atom_bytes)
    } else {
        descriptor.swizzle_atom_bytes
    };
    let (start_address, column_offset) = if descriptor.absolute_leading_address
        && descriptor.start_address % 128 + byte_in_row >= 128
    {
        (
            descriptor.leading_byte_offset,
            (descriptor.start_address % 128 + byte_in_row) % 128,
        )
    } else {
        (
            descriptor.start_address,
            atom * column_stride + byte_in_atom,
        )
    };
    let unswizzled = start_address
        .checked_add((row % 8) * row_stride)
        .and_then(|value| value.checked_add((row / 8) * descriptor.stride_byte_offset))
        .and_then(|value| value.checked_add(column_offset))
        .ok_or_else(|| EngineError::message("raw TCGEN source address overflow"))?;
    raw_tcgen05_finish_shared_byte_offset(source, descriptor, unswizzled, access_bytes)
}

fn raw_tcgen05_finish_shared_byte_offset(
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    unswizzled: usize,
    access_bytes: usize,
) -> Result<usize, EngineError> {
    let access_view = matches!(source, RuntimeBuffer::AccessView { .. });
    let RuntimeBuffer::Shared {
        virtual_base,
        byte_offset: view_offset,
        byte_len: view_len,
        backing_byte_len,
        ..
    } = runtime_buffer_base(source)
    else {
        return Err(EngineError::message(
            "raw TCGEN source is not shared memory",
        ));
    };
    let swizzled_virtual = if descriptor.swizzle_bits == 0 {
        unswizzled
    } else {
        let atom_shift = descriptor.swizzle_atom_bytes.trailing_zeros();
        let byte_in_atom = unswizzled & (descriptor.swizzle_atom_bytes - 1);
        let atom_index = unswizzled >> atom_shift;
        let swizzle_mask = (1_usize << descriptor.swizzle_bits) - 1;
        let swizzled_atom = atom_index
            ^ ((atom_index & (swizzle_mask << descriptor.swizzle_xor_shift))
                >> descriptor.swizzle_xor_shift);
        (swizzled_atom << atom_shift) | byte_in_atom
    };
    let swizzled_relative =
        usize::try_from(swizzled_virtual as i128 - *virtual_base as i128 + *view_offset as i128)
            .map_err(|_| EngineError::message("raw TCGEN source precedes its shared backing"))?;
    let access_end = swizzled_relative
        .checked_add(access_bytes)
        .ok_or_else(|| EngineError::message("raw TCGEN source end overflow"))?;
    if access_end > *backing_byte_len {
        return Err(EngineError::message(format!(
            "raw TCGEN source range [{swizzled_relative}, {access_end}) exceeds shared backing {backing_byte_len}"
        )));
    }
    let view_end = view_offset
        .checked_add(*view_len)
        .ok_or_else(|| EngineError::message("raw TCGEN source view end overflow"))?;
    if access_view && (swizzled_relative < *view_offset || access_end > view_end) {
        return Err(EngineError::message(format!(
            "raw TCGEN source range [{swizzled_relative}, {access_end}) exceeds selected shared view [{view_offset}, {view_end})"
        )));
    }
    // Both numerical reads and checker accesses are relative to this same
    // selected source view. Its owner offset is applied once by memory IO.
    swizzled_relative
        .checked_sub(*view_offset)
        .ok_or_else(|| EngineError::message("raw TCGEN source precedes its shared view"))
}

/// PTX Table 67: MN-major atoms are 128B x 4 K elements for TF32
/// (32B atomicity), and swizzle-width x 8 K elements for 8/16-bit operands.
/// Numeric gathers and checker footprints must use this same address owner.
fn raw_tcgen05_matrix_byte_offset(
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    row: usize,
    column: usize,
    transpose: bool,
    element_bytes: usize,
) -> Result<usize, EngineError> {
    if !transpose {
        let byte_in_row = column
            .checked_mul(element_bytes)
            .ok_or_else(|| EngineError::message("raw MMA matrix byte offset overflow"))?;
        return raw_tcgen05_shared_byte_offset(source, descriptor, row, byte_in_row, element_bytes);
    }
    let tf32 = element_bytes == 4;
    if tf32 != (descriptor.swizzle_atom_bytes == 32) {
        return Err(EngineError::message(
            "MN-major TF32 requires 128B/32B-atomic swizzling; other operand widths forbid it",
        ));
    }
    let swizzle_bytes = descriptor.swizzle_atom_bytes << descriptor.swizzle_bits;
    let elements_per_swizzle = swizzle_bytes / element_bytes;
    let k_per_atom = if tf32 { 4 } else { 8 };
    // Without swizzling, the LDO/SDO roles are exchanged.
    let (mn_stride, k_stride) = if descriptor.swizzle_bits == 0 {
        (
            descriptor.stride_byte_offset,
            descriptor.leading_byte_offset,
        )
    } else {
        (
            descriptor.leading_byte_offset,
            descriptor.stride_byte_offset,
        )
    };
    let unswizzled = descriptor
        .start_address
        .checked_add((row % elements_per_swizzle) * element_bytes)
        .and_then(|v| v.checked_add((row / elements_per_swizzle).checked_mul(mn_stride)?))
        .and_then(|v| v.checked_add((column % k_per_atom) * swizzle_bytes))
        .and_then(|v| v.checked_add((column / k_per_atom).checked_mul(k_stride)?))
        .ok_or_else(|| EngineError::message("raw MMA MN-major source address overflow"))?;
    raw_tcgen05_finish_shared_byte_offset(source, descriptor, unswizzled, element_bytes)
}

fn raw_tcgen05_b16_matrix_byte_offset(
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    row: usize,
    column: usize,
    transpose: bool,
) -> Result<usize, EngineError> {
    raw_tcgen05_matrix_byte_offset(source, descriptor, row, column, transpose, 2)
}

fn raw_tcgen05_8bit_matrix_byte_offset(
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    row: usize,
    column: usize,
    transpose: bool,
) -> Result<usize, EngineError> {
    raw_tcgen05_matrix_byte_offset(source, descriptor, row, column, transpose, 1)
}

fn raw_tcgen05_shared_word_offset(
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    row: usize,
    word: usize,
) -> Result<usize, EngineError> {
    let byte_in_row = word
        .checked_mul(4)
        .ok_or_else(|| EngineError::message("raw TCGEN word offset overflow"))?;
    raw_tcgen05_shared_byte_offset(source, descriptor, row, byte_in_row, 4)
}

fn raw_tcgen05_cp_source_span(
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    row: usize,
    word: usize,
    decompress: u8,
) -> Result<(usize, usize), EngineError> {
    match decompress {
        0 => Ok((
            raw_tcgen05_shared_word_offset(source, descriptor, row, word)?,
            4,
        )),
        1 => {
            let atom = word / 4;
            let word_in_atom = word % 4;
            let byte_in_row = atom
                .checked_mul(16)
                .and_then(|value| value.checked_add(word_in_atom * 2))
                .ok_or_else(|| EngineError::message("raw tcgen05.cp b4 source overflow"))?;
            Ok((
                raw_tcgen05_shared_byte_offset(source, descriptor, row, byte_in_row, 2)?,
                2,
            ))
        }
        2 => {
            let atom = word / 4;
            let word_in_atom = word % 4;
            let byte_in_row = atom
                .checked_mul(16)
                .and_then(|value| value.checked_add(word_in_atom * 3))
                .ok_or_else(|| EngineError::message("raw tcgen05.cp b6 source overflow"))?;
            Ok((
                raw_tcgen05_shared_byte_offset(source, descriptor, row, byte_in_row, 3)?,
                3,
            ))
        }
        _ => Err(EngineError::message(format!(
            "raw tcgen05.cp decompression code {decompress} is invalid"
        ))),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RawTcgenCpDestinationLanes {
    values: [usize; 4],
    len: usize,
}

impl RawTcgenCpDestinationLanes {
    const fn one(first: usize) -> Self {
        Self {
            values: [first, 0, 0, 0],
            len: 1,
        }
    }

    const fn two(first: usize, second: usize) -> Self {
        Self {
            values: [first, second, 0, 0],
            len: 2,
        }
    }

    const fn four(values: [usize; 4]) -> Self {
        Self { values, len: 4 }
    }

    const fn as_slice(&self) -> &[usize] {
        self.values.split_at(self.len).0
    }
}

fn raw_tcgen05_cp_destination_lanes(
    shape: u8,
    source_row: usize,
) -> Result<RawTcgenCpDestinationLanes, EngineError> {
    match shape {
        0 => Ok(RawTcgenCpDestinationLanes::four([
            source_row,
            32 + source_row,
            64 + source_row,
            96 + source_row,
        ])),
        1 => {
            let half = source_row / 32;
            let row = source_row % 32;
            Ok(RawTcgenCpDestinationLanes::two(
                half * 32 + row,
                (half + 2) * 32 + row,
            ))
        }
        2 | 3 => Ok(RawTcgenCpDestinationLanes::one(source_row)),
        4 => Ok(RawTcgenCpDestinationLanes::one(
            source_row
                .checked_mul(32)
                .ok_or_else(|| EngineError::message("raw tcgen05.cp 4x256b lane overflow"))?,
        )),
        5 => {
            let half = source_row / 32;
            let row = source_row % 32;
            let base = half
                .checked_mul(64)
                .ok_or_else(|| EngineError::message("raw tcgen05.cp multicast overflow"))?;
            Ok(RawTcgenCpDestinationLanes::two(base + row, base + 32 + row))
        }
        _ => Err(EngineError::message(format!(
            "raw tcgen05.cp shape code {shape} is invalid"
        ))),
    }
}

fn raw_tcgen05_cp_decode_word(
    packed: &[u8; 4],
    source_byte_len: usize,
    decompress: u8,
) -> Result<[u8; 4], EngineError> {
    match decompress {
        0 if source_byte_len == 4 => Ok(*packed),
        1 if source_byte_len == 2 => Ok([
            (packed[0] & 0x0f) << 2,
            (packed[0] >> 4) << 2,
            (packed[1] & 0x0f) << 2,
            (packed[1] >> 4) << 2,
        ]),
        2 if source_byte_len == 3 => {
            Ok(raw_tcgen05_unpack_b6([packed[0], packed[1], packed[2]]))
        }
        _ => Err(EngineError::message(format!(
            "raw tcgen05.cp decompression code {decompress} has invalid source width {source_byte_len}"
        ))),
    }
}

pub(crate) type RawTcgenRuntimeAccess = (usize, Option<usize>, usize, usize);
pub(crate) type RawTcgenTmemAccess = (usize, usize, Option<usize>, i64, i64, i64, usize);

pub(crate) struct RawTcgenCpFootprints {
    pub(crate) source: RuntimeBuffer,
    pub(crate) source_accesses: Vec<RawTcgenRuntimeAccess>,
    pub(crate) destination_accesses: Vec<RawTcgenTmemAccess>,
    pub(crate) words: usize,
}

/// Exact shared-memory and TMEM operands of a modeled raw tcgen05.cp.
pub(crate) fn raw_tcgen05_cp_footprints(
    context: &WarpContext,
    shared_candidates: &[&RuntimeBuffer],
    address: u32,
    descriptor_bits: u64,
    row_offset: i64,
    col_offset: i64,
    shape: u8,
    decompress: u8,
    cta_group: usize,
    issuing_lane: usize,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
) -> Result<RawTcgenCpFootprints, EngineError> {
    let descriptor =
        decode_raw_tcgen_matrix_descriptor_for_layout(descriptor_bits, descriptor_layout)?;
    let source = raw_tcgen05_shared_source(
        context,
        shared_candidates,
        descriptor.start_address,
        issuing_lane,
    )?;
    let (base_row, base_col) = raw_tcgen05_address(address, row_offset, col_offset)?;
    let (rows, words) = match shape {
        0 => (32_usize, 4_usize),
        1 => (64_usize, 4_usize),
        2 => (128_usize, 4_usize),
        3 => (128_usize, 8_usize),
        4 => (4_usize, 8_usize),
        5 => (64_usize, 4_usize),
        _ => {
            return Err(EngineError::message(format!(
                "raw tcgen05.cp shape code {shape} is invalid"
            )));
        }
    };
    let mut targets = vec![context.cta_id_in_cluster()];
    if cta_group == 2 {
        let peer = context.cta_id_in_cluster() ^ 1;
        if peer >= context.topology().ctas_per_cluster() {
            return Err(EngineError::message(
                "raw tcgen05.cp cta_group=2 has no paired CTA",
            ));
        }
        targets.push(peer);
    } else if cta_group != 1 {
        return Err(EngineError::message(format!(
            "raw tcgen05.cp cta_group must be 1 or 2, got {cta_group}"
        )));
    }
    let mut source_accesses = Vec::with_capacity(targets.len() * rows * words);
    let mut destination_accesses = Vec::with_capacity(targets.len() * rows * words);
    for target_cta in targets {
        for source_row in 0..rows {
            let destination_lanes = raw_tcgen05_cp_destination_lanes(shape, source_row)?;
            for word in 0..words {
                let (byte_offset, byte_len) =
                    raw_tcgen05_cp_source_span(&source, descriptor, source_row, word, decompress)?;
                source_accesses.push((issuing_lane, Some(target_cta), byte_offset, byte_len));
                let destination_column = base_col
                    .checked_add(word)
                    .ok_or_else(|| EngineError::message("raw tcgen05.cp TMEM column overflow"))?;
                for destination_lane in destination_lanes.as_slice() {
                    let destination_lane = base_row
                        .checked_add(*destination_lane)
                        .ok_or_else(|| EngineError::message("raw tcgen05.cp TMEM lane overflow"))?;
                    if destination_lane >= 128 {
                        return Err(EngineError::message(format!(
                            "raw tcgen05.cp TMEM lane {destination_lane} is outside 128 lanes"
                        )));
                    }
                    destination_accesses.push((
                        issuing_lane,
                        issuing_lane,
                        Some(target_cta),
                        i64::try_from(destination_lane)
                            .map_err(|_| EngineError::message("raw TCGEN lane exceeds i64"))?,
                        0,
                        i64::try_from(destination_column)
                            .map_err(|_| EngineError::message("raw TCGEN column exceeds i64"))?,
                        4,
                    ));
                }
            }
        }
    }
    Ok(RawTcgenCpFootprints {
        source,
        source_accesses,
        destination_accesses,
        words,
    })
}

fn raw_tcgen05_unpack_b6(packed: [u8; 3]) -> [u8; 4] {
    let bits = u32::from(packed[0]) | (u32::from(packed[1]) << 8) | (u32::from(packed[2]) << 16);
    [
        ((bits >> 0) & 0x3f) as u8,
        ((bits >> 6) & 0x3f) as u8,
        ((bits >> 12) & 0x3f) as u8,
        ((bits >> 18) & 0x3f) as u8,
    ]
}

#[allow(clippy::too_many_arguments)]
pub fn raw_tcgen05_cp(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    anchor: &RuntimeBuffer,
    shared_candidates: &[&RuntimeBuffer],
    address: u32,
    descriptor_bits: u64,
    row_offset: i64,
    col_offset: i64,
    shape: u8,
    decompress: u8,
    cta_group: usize,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
) -> Result<(), EngineError> {
    let descriptor =
        decode_raw_tcgen_matrix_descriptor_for_layout(descriptor_bits, descriptor_layout)?;
    let issuing_lane = context
        .active_mask()
        .first_active()
        .ok_or_else(|| EngineError::message("raw tcgen05.cp has no issuing lane"))?;
    let source = raw_tcgen05_shared_source(
        context,
        shared_candidates,
        descriptor.start_address,
        issuing_lane,
    )?;
    let (base_row, base_col) = raw_tcgen05_address(address, row_offset, col_offset)?;
    let (rows, words) = match shape {
        0 => (32_usize, 4_usize),
        1 => (64_usize, 4_usize),
        2 => (128_usize, 4_usize),
        3 => (128_usize, 8_usize),
        4 => (4_usize, 8_usize),
        5 => (64_usize, 4_usize),
        _ => {
            return Err(EngineError::message(format!(
                "raw tcgen05.cp shape code {shape} is invalid"
            )));
        }
    };
    let mut target_ctas = vec![context.cta_id_in_cluster()];
    if cta_group == 2 {
        let peer = context.cta_id_in_cluster() ^ 1;
        if peer >= context.topology().ctas_per_cluster() {
            return Err(EngineError::message(
                "raw tcgen05.cp cta_group=2 has no paired CTA",
            ));
        }
        target_ctas.push(peer);
    } else if cta_group != 1 {
        return Err(EngineError::message(format!(
            "raw tcgen05.cp cta_group must be 1 or 2, got {cta_group}"
        )));
    }
    for target in target_ctas.iter().copied() {
        validate_raw_tcgen05_tmem_columns(
            lifecycle,
            access_mode,
            context,
            target,
            base_col,
            words,
        )?;
    }
    let target_views = target_ctas
        .into_iter()
        .map(|target| -> Result<_, EngineError> {
            Ok((
                target,
                raw_tcgen05_tmem_view(physical, context, anchor, target)?,
                raw_tcgen05_shared_view_at_cta(physical, context, &source, target)?,
            ))
        })
        .collect::<Result<Vec<_>, EngineError>>()?;
    let column_end = base_col
        .checked_add(words)
        .ok_or_else(|| EngineError::message("raw tcgen05.cp TMEM column overflow"))?;
    let mut lane_end = 0_usize;
    for source_row in 0..rows {
        let destination_lanes = raw_tcgen05_cp_destination_lanes(shape, source_row)?;
        for &destination_lane in destination_lanes.as_slice() {
            let lane = base_row
                .checked_add(destination_lane)
                .ok_or_else(|| EngineError::message("raw tcgen05.cp TMEM lane overflow"))?;
            if lane >= 128 {
                return Err(EngineError::message(format!(
                    "raw tcgen05.cp TMEM lane {lane} is outside 128 lanes"
                )));
            }
            lane_end = lane_end.max(lane + 1);
        }
        for word in 0..words {
            raw_tcgen05_cp_source_span(&source, descriptor, source_row, word, decompress)?;
        }
    }
    for (_target_cta, view, _source_view) in &target_views {
        physical
            .tmem()
            .validate_cell_rectangle(view, lane_end, column_end)?;
    }

    for (_target_cta, view, source_view) in &target_views {
        physical
            .shared()
            .with_read_session(source_view, |source_session| {
                if !source_session.records_uninitialized_read_reviews()
                    && !source_session.all_bytes_initialized()
                {
                    for source_row in 0..rows {
                        for word in 0..words {
                            let (source_offset, source_byte_len) = raw_tcgen05_cp_source_span(
                                &source, descriptor, source_row, word, decompress,
                            )?;
                            source_session.validate_initialized_bytes_prevalidated(
                                source_offset,
                                source_byte_len,
                            )?;
                        }
                    }
                }

                physical
                    .tmem()
                    .with_write_session(view, |destination_session| {
                        for source_row in 0..rows {
                            let destination_lanes =
                                raw_tcgen05_cp_destination_lanes(shape, source_row)?;
                            for word in 0..words {
                                let (source_offset, source_byte_len) = raw_tcgen05_cp_source_span(
                                    &source, descriptor, source_row, word, decompress,
                                )?;
                                let mut packed = [0_u8; 4];
                                let packed_source = &mut packed[..source_byte_len];
                                if source_session.records_uninitialized_read_reviews() {
                                    source_session.read_reviewed_bytes_into_prevalidated(
                                        source_offset,
                                        packed_source,
                                    );
                                } else {
                                    source_session.read_initialized_bytes_into_prevalidated(
                                        source_offset,
                                        packed_source,
                                    );
                                }
                                let bytes = raw_tcgen05_cp_decode_word(
                                    &packed,
                                    source_byte_len,
                                    decompress,
                                )?;
                                let column = base_col + word;
                                for &destination_lane in destination_lanes.as_slice() {
                                    destination_session.write_cell_bytes_prevalidated(
                                        base_row + destination_lane,
                                        column,
                                        0,
                                        &bytes,
                                    );
                                }
                            }
                        }
                        Ok::<(), EngineError>(())
                    })??;
                Ok::<(), EngineError>(())
            })??;
    }
    Ok(())
}

/// Which `.kind::mxf4`-family block-scale spelling a raw dense `tcgen05.mma`
/// descriptor carries.
///
/// The PTX mnemonic fixes the scale-vector width while descriptor bit 23 fixes
/// the scale format. Each legal combination fixes three facts at once, so it
/// rides on one row rather than three parallel flags:
///
/// - the scale matrix type in instruction-descriptor bit 23, from the PTX ISA
///   instruction-descriptor format for `.kind::mxf4` and `.kind::mxf4nvf4`
///   (`#tcgen05-instuction-desc-kind-mxf4-mxf4nvf4`): `UE8M0 = 1`, `UE4M3 = 0`;
/// - the legal scale-factor IDs, from the PTX ISA scale-factor A/B ID layout
///   sections (`#tcgen05-mma-scale-factor-a`): `.scale_vec::2X` selects a
///   half-word, so 0 or 2, and `.scale_vec::4X` uses all four bytes of the
///   word, so the ID must be 0;
/// - the fixed vector count (2 or 4), or its block-size spelling at K=64.
///   The descriptor decoder resolves the effective block width for the actual K.
///
/// The ID rule above is the K64 rule; the instruction decoder also validates
/// the K96/K128 continuation layouts against the authored architecture.
///
/// Anchors rather than table or section numbers, which move between ISA
/// editions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RawTcgenMxf4ScaleSpelling {
    /// UE8M0, two fixed vectors or block32.
    Ue8m0Vec2x,
    Ue4m3Vec2x,
    Ue5m3Vec2x,
    /// UE8M0, four fixed vectors or block16.
    Ue8m0Vec4x,
    /// UE4M3, four fixed vectors or block16.
    Ue4m3Vec4x,
    Ue5m3Vec4x,
}

impl RawTcgenMxf4ScaleSpelling {
    /// Diagnostic prefix; also the name the PTX `.kind` qualifier spells.
    fn label(self) -> &'static str {
        match self {
            Self::Ue8m0Vec2x => "raw mxf4",
            _ => "raw mxf4nvf4",
        }
    }

    /// Required value of instruction-descriptor bits 23-24 (scale matrix type).
    fn scale_format_bit(self) -> u32 {
        match self {
            Self::Ue8m0Vec2x | Self::Ue8m0Vec4x => 1,
            Self::Ue4m3Vec2x | Self::Ue4m3Vec4x => 0,
            Self::Ue5m3Vec2x | Self::Ue5m3Vec4x => 2,
        }
    }

    fn scale_format_name(self) -> &'static str {
        match self {
            Self::Ue8m0Vec2x | Self::Ue8m0Vec4x => "UE8M0",
            Self::Ue4m3Vec2x | Self::Ue4m3Vec4x => "UE4M3",
            Self::Ue5m3Vec2x | Self::Ue5m3Vec4x => "UE5M3",
        }
    }

    fn vector_count(self) -> usize {
        match self {
            Self::Ue8m0Vec2x | Self::Ue4m3Vec2x | Self::Ue5m3Vec2x => 2,
            Self::Ue8m0Vec4x | Self::Ue4m3Vec4x | Self::Ue5m3Vec4x => 4,
        }
    }

    fn scale_id_is_legal(self, scale_id: usize) -> bool {
        match self.vector_count() {
            2 => matches!(scale_id, 0 | 2),
            _ => scale_id == 0,
        }
    }

    fn legal_scale_ids(self) -> &'static str {
        match self.vector_count() {
            2 => "scale_vec::2X requires SFA/SFB IDs 0 or 2",
            _ => "scale_vec::4X requires SFA/SFB IDs 0",
        }
    }

    /// How one stored scale byte decodes.
    fn decoder(self) -> RawTcgenScaleDecoder {
        match self {
            Self::Ue8m0Vec2x | Self::Ue8m0Vec4x => raw_tcgen05_decode_ue8m0_scale,
            Self::Ue4m3Vec2x | Self::Ue4m3Vec4x => raw_tcgen05_decode_ue4m3_scale,
            Self::Ue5m3Vec2x | Self::Ue5m3Vec4x => raw_tcgen05_decode_ue5m3_scale,
        }
    }
}

pub(crate) fn raw_tcgen05_mxf4nvf4_vec4x_scale(descriptor: u32) -> RawTcgenMxf4ScaleSpelling {
    match (descriptor >> 23) & 3 {
        0 => RawTcgenMxf4ScaleSpelling::Ue4m3Vec4x,
        2 => RawTcgenMxf4ScaleSpelling::Ue5m3Vec4x,
        _ => RawTcgenMxf4ScaleSpelling::Ue8m0Vec4x,
    }
}

pub(crate) fn raw_tcgen05_mxf4nvf4_vec2x_scale(descriptor: u32) -> RawTcgenMxf4ScaleSpelling {
    match (descriptor >> 23) & 3 {
        0 => RawTcgenMxf4ScaleSpelling::Ue4m3Vec2x,
        2 => RawTcgenMxf4ScaleSpelling::Ue5m3Vec2x,
        _ => RawTcgenMxf4ScaleSpelling::Ue8m0Vec2x,
    }
}

/// How one block-scale byte in TMEM becomes the factor the MMA multiplies by.
type RawTcgenScaleDecoder = fn(u8) -> Result<f32, EngineError>;

fn raw_tcgen05_decode_ue8m0_scale(bits: u8) -> Result<f32, EngineError> {
    Ok(float8_e8m0fnu_bits_to_f32(bits))
}

fn raw_tcgen05_decode_ue5m3_scale(bits: u8) -> Result<f32, EngineError> {
    Ok(crate::numpy_backend::narrow_float_bits_to_f32_checked(
        bits,
        crate::numpy_backend::FLOAT8_UE5M3,
    )
    .unwrap_or(f32::NAN))
}

/// `ue4m3` is a *7-bit* unsigned format whose MSB is padding PTX ISA 5.2.3
/// requires to be zero, so the remaining bits decode exactly as `e4m3` with a
/// clear sign bit and `0x7f` is its only NaN. A set padding bit is undefined on
/// hardware, so it fails closed rather than reading as a negative scale.
fn raw_tcgen05_decode_ue4m3_scale(bits: u8) -> Result<f32, EngineError> {
    if bits & 0x80 != 0 {
        return Err(EngineError::message(format!(
            "raw mxf4nvf4 ue4m3 scale byte {bits:#04x} sets the padding MSB"
        )));
    }
    Ok(float8_e4m3fn_bits_to_f32(bits))
}

/// PTX Tables 52/53, bit 26 selects the SM107 SFA layout (Figures 238--256).
fn raw_tcgen05_sfa_lanes(
    descriptor: u32,
    layout: RawTcgenMatrixDescriptorLayout,
) -> Result<usize, EngineError> {
    if descriptor & (1 << 26) == 0 {
        return Ok(32);
    }
    if layout != RawTcgenMatrixDescriptorLayout::Sm107 {
        return Err(EngineError::message("128-lane SFA layout requires SM107"));
    }
    Ok(128)
}

#[derive(Clone, Copy, Debug)]
struct RawTcgenMxf4InstructionDescriptor {
    sfa_lanes: usize,
    m: usize,
    n: usize,
    k: usize,
    sfa_id: usize,
    sfb_id: usize,
    negate_a: bool,
    negate_b: bool,
    block_elements: usize,
}

#[cfg(test)]
fn decode_raw_tcgen_mxf4_instruction_descriptor(
    descriptor: u32,
    scale: RawTcgenMxf4ScaleSpelling,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
) -> Result<RawTcgenMxf4InstructionDescriptor, EngineError> {
    decode_raw_tcgen_mxf4_instruction_descriptor_for_cta_group(
        descriptor,
        scale,
        1,
        descriptor_layout,
        false,
    )
}

fn decode_raw_tcgen_mxf4_instruction_descriptor_for_cta_group(
    descriptor: u32,
    scale: RawTcgenMxf4ScaleSpelling,
    cta_group: usize,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
    fixed_vectors: bool,
) -> Result<RawTcgenMxf4InstructionDescriptor, EngineError> {
    let label = scale.label();
    // PTX 9.4 Table 53: dense MXF4 still requires the architecture's
    // sparsity-version encoding, even though no sparse metadata is read.
    let version = u32::from(descriptor_layout == RawTcgenMatrixDescriptorLayout::Sm107);
    if (descriptor >> 12) & 1 != version {
        return Err(EngineError::message(format!(
            "{label} descriptor sparsity version must be v{version} for {descriptor_layout:?}"
        )));
    }
    let k = match ((descriptor >> 3) & 1, descriptor >> 31) {
        (0, 0) => 64,
        (0, 1) if descriptor_layout != RawTcgenMatrixDescriptorLayout::Sm100 => 96,
        (1, 0) if descriptor_layout == RawTcgenMatrixDescriptorLayout::Sm107 => 128,
        _ => {
            return Err(EngineError::message(format!(
                "{label} descriptor has unsupported K encoding for {descriptor_layout:?}"
            )))
        }
    };
    let block_elements = if fixed_vectors {
        k / scale.vector_count()
    } else {
        64 / scale.vector_count()
    };
    if !matches!(cta_group, 1 | 2) {
        return Err(EngineError::message(format!(
            "{label} tcgen05.mma cta_group must be 1 or 2, got {cta_group}"
        )));
    }
    if descriptor & 0x3 != 0
        || descriptor & (1_u32 << 2) != 0
        || descriptor & (1_u32 << 6) != 0
        || descriptor & (1_u32 << 25) != 0
    {
        return Err(EngineError::message(format!(
            "{label} tcgen05.mma descriptor uses sparse or reserved bits ({descriptor:#010x})"
        )));
    }
    if ((descriptor >> 7) & 0x7) != 1 || ((descriptor >> 10) & 0x3) != 1 {
        return Err(EngineError::message(format!(
            "{label} tcgen05.mma descriptor must encode E2M1 A and B"
        )));
    }
    if ((descriptor >> 23) & 0x3) != scale.scale_format_bit() {
        return Err(EngineError::message(format!(
            "{label} tcgen05.mma descriptor must encode {} scales",
            scale.scale_format_name()
        )));
    }
    if descriptor_layout != RawTcgenMatrixDescriptorLayout::Sm107
        && matches!(
            scale,
            RawTcgenMxf4ScaleSpelling::Ue4m3Vec2x
                | RawTcgenMxf4ScaleSpelling::Ue5m3Vec2x
                | RawTcgenMxf4ScaleSpelling::Ue5m3Vec4x
        )
    {
        return Err(EngineError::message(
            "UE5M3 scales and block32 UE4M3 require SM107",
        ));
    }
    if descriptor & ((1_u32 << 15) | (1_u32 << 16)) != 0 {
        return Err(EngineError::message(format!(
            "{label} tcgen05.mma transpose descriptors are not implemented"
        )));
    }
    let m = usize::try_from((descriptor >> 27) & 0x3)
        .map_err(|_| EngineError::message(format!("{label} M conversion failed")))?
        .checked_mul(128)
        .ok_or_else(|| EngineError::message(format!("{label} M overflow")))?;
    let n = usize::try_from((descriptor >> 17) & 0x3f)
        .map_err(|_| EngineError::message(format!("{label} N conversion failed")))?
        .checked_mul(8)
        .ok_or_else(|| EngineError::message(format!("{label} N overflow")))?;
    let expected_m = 128 * cta_group;
    let n_granularity = 8 * cta_group;
    if m != expected_m || !(n_granularity..=256).contains(&n) || n % n_granularity != 0 {
        return Err(EngineError::message(format!(
            "{label} tcgen05.mma cta_group={cta_group} requires M={expected_m} and N in {n_granularity}..=256 by {n_granularity}, got M={m}, N={n}"
        )));
    }
    let sfa_id = usize::try_from((descriptor >> 29) & 0x3)
        .map_err(|_| EngineError::message(format!("{label} SFA ID conversion failed")))?;
    let sfb_id = usize::try_from((descriptor >> 4) & 0x3)
        .map_err(|_| EngineError::message(format!("{label} SFB ID conversion failed")))?;
    let valid_id = |id| {
        if fixed_vectors {
            scale.scale_id_is_legal(id)
        } else {
            match (k, scale.vector_count()) {
                (96, 2) => id < 4,
                (96, _) => matches!(id, 0 | 2),
                (128, _) => id == 0,
                _ => scale.scale_id_is_legal(id),
            }
        }
    };
    if !valid_id(sfa_id) || !valid_id(sfb_id) {
        if k != 64 {
            return Err(EngineError::message(format!(
                "{label} descriptor has unsupported scale IDs {sfa_id}/{sfb_id} for K={k}"
            )));
        }
        return Err(EngineError::message(format!(
            "{label} {}, got {sfa_id}/{sfb_id}",
            scale.legal_scale_ids()
        )));
    }
    Ok(RawTcgenMxf4InstructionDescriptor {
        sfa_lanes: raw_tcgen05_sfa_lanes(descriptor, descriptor_layout)?,
        m,
        n,
        k,
        sfa_id,
        sfb_id,
        negate_a: descriptor & (1_u32 << 13) != 0,
        negate_b: descriptor & (1_u32 << 14) != 0,
        block_elements,
    })
}

pub(crate) fn raw_tcgen05_mma_block_mxf4_shape(
    descriptor: u32,
    scale: RawTcgenMxf4ScaleSpelling,
    cta_group: usize,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
    fixed_vectors: bool,
) -> Result<(usize, usize, usize), EngineError> {
    let instruction = decode_raw_tcgen_mxf4_instruction_descriptor_for_cta_group(
        descriptor,
        scale,
        cta_group,
        descriptor_layout,
        fixed_vectors,
    )?;
    Ok((instruction.m, instruction.n, instruction.k))
}

fn decode_raw_tcgen_sparse_mxf4_instruction_descriptor(
    descriptor: u32,
) -> Result<RawTcgenMxf4InstructionDescriptor, EngineError> {
    if descriptor & 0x3 != 0
        || descriptor & (1_u32 << 2) == 0
        || descriptor & (1_u32 << 3) != 0
        || descriptor & (1_u32 << 6) != 0
        || descriptor & (1_u32 << 12) != 0
        || descriptor & (0x7_u32 << 24) != 0
        || descriptor & (1_u32 << 31) != 0
    {
        return Err(EngineError::message(
            "raw sparse mxf4 descriptor must encode sparse K=128 without reserved bits",
        ));
    }
    if ((descriptor >> 7) & 0x7) != 1 || ((descriptor >> 10) & 0x3) != 1 {
        return Err(EngineError::message(
            "raw sparse mxf4 descriptor must encode E2M1 A and B",
        ));
    }
    if ((descriptor >> 23) & 0x1) != 1 {
        return Err(EngineError::message(
            "raw sparse mxf4 descriptor must encode UE8M0 scales",
        ));
    }
    if descriptor & ((1_u32 << 15) | (1_u32 << 16)) != 0 {
        return Err(EngineError::message(
            "raw sparse mxf4 transpose descriptors are not implemented",
        ));
    }
    let m = usize::try_from((descriptor >> 27) & 0x3)
        .map_err(|_| EngineError::message("raw sparse mxf4 M conversion failed"))?
        .checked_mul(128)
        .ok_or_else(|| EngineError::message("raw sparse mxf4 M overflow"))?;
    let n = usize::try_from((descriptor >> 17) & 0x3f)
        .map_err(|_| EngineError::message("raw sparse mxf4 N conversion failed"))?
        .checked_mul(8)
        .ok_or_else(|| EngineError::message("raw sparse mxf4 N overflow"))?;
    if m != 128 || !(8..=256).contains(&n) || n % 8 != 0 {
        return Err(EngineError::message(format!(
            "raw sparse mxf4 cta_group=1 requires M=128 and N in 8..=256 by 8, got M={m}, N={n}"
        )));
    }
    let sfa_id = usize::try_from((descriptor >> 29) & 0x3)
        .map_err(|_| EngineError::message("raw sparse mxf4 SFA ID conversion failed"))?;
    let sfb_id = usize::try_from((descriptor >> 4) & 0x3)
        .map_err(|_| EngineError::message("raw sparse mxf4 SFB ID conversion failed"))?;
    if !matches!(sfa_id, 0 | 2) || !matches!(sfb_id, 0 | 2) {
        return Err(EngineError::message(format!(
            "raw sparse mxf4 scale_vec::2X requires SFA/SFB IDs 0 or 2, got {sfa_id}/{sfb_id}"
        )));
    }
    Ok(RawTcgenMxf4InstructionDescriptor {
        sfa_lanes: 32,
        m,
        n,
        k: 128,
        sfa_id,
        sfb_id,
        negate_a: descriptor & (1_u32 << 13) != 0,
        negate_b: descriptor & (1_u32 << 14) != 0,
        block_elements: 64,
    })
}

/// Four-byte scale words occupy four columns per 128 rows in the 32-lane
/// layout, or one column in SM107's 128-lane SFA layout. Numerical reads,
/// footprints and ownership bounds share this calculation; CTA2 B supplies
/// the joint N, not the local N / 2.
fn raw_tcgen05_scale_chunk(
    address: u32,
    scale_id: usize,
    vector_index: usize,
    matrix_rows: usize,
    lanes_per_column: usize,
) -> Result<(u32, usize), EngineError> {
    if !matches!(lanes_per_column, 32 | 128) {
        return Err(EngineError::message(
            "scale layout requires 32 or 128 lanes",
        ));
    }
    let column_stride = matrix_rows.div_ceil(128) * (128 / lanes_per_column);
    let byte = scale_id
        .checked_add(vector_index)
        .ok_or_else(|| EngineError::message("raw TCGEN scale byte overflow"))?;
    let word = byte / 4;
    if word != 0 && column_stride == 0 {
        return Err(EngineError::message(
            "raw TCGEN scale byte exceeds its TMEM word",
        ));
    }
    let offset = word
        .checked_mul(column_stride)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| EngineError::message("raw TCGEN scale column overflow"))?;
    let (lane, column) = raw_tcgen05_block_scale_address(address)?;
    let next_column = column
        .checked_add(offset as usize)
        .filter(|v| *v <= 0xffff)
        .ok_or_else(|| EngineError::message("raw TCGEN scale column overflow"))?;
    Ok((((lane as u32) << 16) | next_column as u32, byte % 4))
}

fn raw_tcgen05_scale_columns(
    rows: usize,
    scale_id: usize,
    count: usize,
    lanes_per_column: usize,
) -> Result<usize, EngineError> {
    let (last_address, _) =
        raw_tcgen05_scale_chunk(0, scale_id, count - 1, rows, lanes_per_column)?;
    Ok(last_address as usize + rows.div_ceil(lanes_per_column))
}

/// SM100 block scales repeat across 32-lane TMEM partitions. M128 CTA2
/// splits SFB's N dimension between the lower and upper pair of partitions.
/// See CUTLASS tmem_sf_frg (ScaleFactorDuplicated4by1 / Duplicated2by2).
#[derive(Clone, Copy)]
enum RawTcgenScaleLayout {
    Replicated,
    SplitB { rows_per_half: usize },
}

impl RawTcgenScaleLayout {
    fn replicas(self) -> usize {
        match self {
            Self::Replicated => 4,
            Self::SplitB { .. } => 2,
        }
    }

    fn location(
        self,
        address: u32,
        row: usize,
        replica: usize,
    ) -> Result<(usize, usize), EngineError> {
        let (base_lane, base_column) = raw_tcgen05_block_scale_address(address)?;
        let (partition, row) = match self {
            Self::Replicated => (replica, row),
            Self::SplitB { rows_per_half } => {
                (2 * (row / rows_per_half) + replica, row % rows_per_half)
            }
        };
        let lane = base_lane + partition * 32 + row % 32;
        let column = base_column + row / 32;
        if lane >= 128 || column > 0xffff {
            return Err(EngineError::message(
                "raw TCGEN scale replica is outside TMEM",
            ));
        }
        Ok((lane, column))
    }
}

fn raw_tcgen05_mxf8_scale_layout(
    m: usize,
    n: usize,
    cta_group: usize,
    is_b: bool,
) -> RawTcgenScaleLayout {
    if is_b && cta_group == 2 && m == 128 {
        RawTcgenScaleLayout::SplitB {
            rows_per_half: n / 2,
        }
    } else {
        RawTcgenScaleLayout::Replicated
    }
}

/// Read every required copy before selecting the common scale value. Missing
/// or contradictory copies have no valid block-scaled matrix semantics.
fn raw_tcgen05_mxf8_scale_values(
    physical: &PhysicalMemory,
    view: &TmemView,
    address: u32,
    scale_id: usize,
    rows: usize,
    layout: RawTcgenScaleLayout,
) -> Result<Vec<f32>, EngineError> {
    if scale_id >= 4 {
        return Err(EngineError::message("raw TCGEN scale byte is outside TMEM"));
    }
    let mut locations = Vec::with_capacity(rows * layout.replicas());
    let mut lane_end = 0;
    let mut column_end = 0;
    for row in 0..rows {
        for replica in 0..layout.replicas() {
            let location = layout.location(address, row, replica)?;
            lane_end = lane_end.max(location.0 + 1);
            column_end = column_end.max(location.1 + 1);
            locations.push(location);
        }
    }
    physical
        .tmem()
        .validate_cell_rectangle(view, lane_end, column_end)?;
    physical.tmem().with_read_session(view, |session| {
        let reviewed = session.records_uninitialized_read_reviews();
        if !reviewed && !session.all_bytes_initialized() {
            for &(lane, column) in &locations {
                session.validate_initialized_cell_bytes_prevalidated(lane, column, scale_id, 1)?;
            }
        }
        let mut values = Vec::with_capacity(rows);
        for copies in locations.chunks_exact(layout.replicas()) {
            let mut common = None;
            for &(lane, column) in copies {
                let mut bits = [0_u8; 1];
                if reviewed {
                    session.read_reviewed_cell_bytes_into_prevalidated(
                        lane, column, scale_id, &mut bits,
                    );
                } else {
                    session.read_initialized_cell_bytes_into_prevalidated(
                        lane, column, scale_id, &mut bits,
                    );
                }
                if common.is_some_and(|value| value != bits[0]) {
                    return Err(EngineError::message(
                        "raw TCGEN block-scale replicas disagree",
                    ));
                }
                common = Some(bits[0]);
            }
            values.push(raw_tcgen05_decode_ue8m0_scale(
                common.expect("scale has replicas"),
            )?);
        }
        Ok(values)
    })?
}

/// One block-scale factor: its TMEM byte and the format-exact decode.
///
/// `.scale_vec_size` changes only how many bytes of the 32-bit word one
/// instruction consumes, so `vector_index` walks the same `scale_id`-relative
/// byte run for every spelling -- see the PTX ISA scale-factor A/B ID layout
/// sections (`#tcgen05-mma-scale-factor-a`).
#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_read_block_scale(
    physical: &PhysicalMemory,
    view: &TmemView,
    address: u32,
    scale_id: usize,
    matrix_row: usize,
    vector_index: usize,
    decode: RawTcgenScaleDecoder,
    matrix_rows: usize,
    lanes_per_column: usize,
) -> Result<f32, EngineError> {
    let (address, byte_in_cell) = raw_tcgen05_scale_chunk(
        address,
        scale_id,
        vector_index,
        matrix_rows,
        lanes_per_column,
    )?;
    let (base_lane, base_col) = raw_tcgen05_block_scale_address(address)?;
    let lane = base_lane
        .checked_add(matrix_row % lanes_per_column)
        .ok_or_else(|| EngineError::message("raw TCGEN scale lane overflow"))?;
    let column = base_col
        .checked_add(matrix_row / lanes_per_column)
        .ok_or_else(|| EngineError::message("raw TCGEN scale column overflow"))?;
    if lane >= 128 || byte_in_cell >= 4 {
        return Err(EngineError::message(format!(
            "raw TCGEN scale location lane={lane}, byte={byte_in_cell} is outside TMEM"
        )));
    }
    let mut bytes = [0_u8; 1];
    physical
        .tmem()
        .read_cell_bytes_into(view, lane, column, byte_in_cell, &mut bytes)?;
    decode(bytes[0])
}

#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_gather_mxf4_matrix(
    physical: &PhysicalMemory,
    context: &WarpContext,
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    rows: usize,
    scale_view: &TmemView,
    scale_address: u32,
    scale_id: usize,
    scale: RawTcgenMxf4ScaleSpelling,
    negate: bool,
    _issuing_lane: usize,
    k: usize,
    block_elements: usize,
    lanes_per_column: usize,
) -> Result<Vec<f32>, EngineError> {
    // The decoded spelling determines how many nibbles each scale covers.
    let atom_bytes = block_elements / 2;
    let mut values = Vec::with_capacity(
        rows.checked_mul(k)
            .ok_or_else(|| EngineError::message("raw mxf4 gather shape overflow"))?,
    );
    let mut source_view = None;
    for row in 0..rows {
        let scales = (0..(k / (2 * atom_bytes)))
            .map(|vector_index| {
                raw_tcgen05_read_block_scale(
                    physical,
                    scale_view,
                    scale_address,
                    scale_id,
                    row,
                    vector_index,
                    scale.decoder(),
                    rows,
                    lanes_per_column,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if source_view.is_none() {
            source_view = Some(raw_tcgen05_shared_view_at_cta(
                physical,
                context,
                source,
                context.cta_id_in_cluster(),
            )?);
        }
        let source_view = source_view
            .as_ref()
            .expect("raw mxf4 rows are non-empty after scale reads");
        for (atom_index, scale_value) in scales.into_iter().enumerate() {
            let mut sub = 0;
            while sub < atom_bytes {
                let column = atom_index * atom_bytes + sub;
                let chunk_bytes = (atom_bytes - sub)
                    .min(16)
                    .min(descriptor.swizzle_atom_bytes - column % descriptor.swizzle_atom_bytes);

                let source_offset = raw_tcgen05_shared_byte_offset(
                    source,
                    descriptor,
                    row,
                    atom_index * atom_bytes + sub,
                    chunk_bytes,
                )?;
                let mut packed_storage = [0_u8; 16];
                let packed_bytes = &mut packed_storage[..chunk_bytes];
                physical
                    .shared()
                    .read_bytes_into(source_view, source_offset, packed_bytes)?;
                for &packed in packed_bytes.iter() {
                    for nibble_index in 0..2 {
                        let bits = if nibble_index == 0 {
                            packed & 0xf
                        } else {
                            packed >> 4
                        };
                        let mut value = float4_e2m1fn_bits_to_f32(bits) * scale_value;
                        if negate {
                            value = -value;
                        }
                        values.push(value);
                    }
                }
                sub += chunk_bytes;
            }
        }
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_gather_mxf4_cta2_matrix(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    anchor: &RuntimeBuffer,
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    rows_per_cta: usize,
    scale_rows_are_joint: bool,
    scale_address: u32,
    scale_id: usize,
    scale: RawTcgenMxf4ScaleSpelling,
    negate: bool,
    k: usize,
    block_elements: usize,
    lanes_per_column: usize,
) -> Result<Vec<f32>, EngineError> {
    let pair_base = context.cta_id_in_cluster() & !1_usize;
    if pair_base + 1 >= context.topology().ctas_per_cluster() {
        return Err(EngineError::message(
            "raw mxf4 cta_group=2 tcgen05.mma has no paired CTA",
        ));
    }
    let scale_rows = rows_per_cta * if scale_rows_are_joint { 2 } else { 1 };
    let atom_bytes = block_elements / 2;
    let mut values = Vec::with_capacity(
        rows_per_cta
            .checked_mul(2)
            .and_then(|rows| rows.checked_mul(k))
            .ok_or_else(|| EngineError::message("raw mxf4 cta2 gather shape overflow"))?,
    );
    for target_offset in 0..2 {
        let target_cta = pair_base + target_offset;
        let first_scale_row = if scale_rows_are_joint {
            target_offset
                .checked_mul(rows_per_cta)
                .ok_or_else(|| EngineError::message("raw mxf4 cta2 scale row overflow"))?
        } else {
            0
        };
        let last_scale_row = first_scale_row
            .checked_add(rows_per_cta - 1)
            .ok_or_else(|| EngineError::message("raw mxf4 cta2 scale row overflow"))?;
        let (_, scale_base_column) = raw_tcgen05_block_scale_address(scale_address)?;
        let first_scale_column = scale_base_column
            .checked_add(first_scale_row / lanes_per_column)
            .ok_or_else(|| EngineError::message("raw mxf4 cta2 scale column overflow"))?;
        let (last_address, _) = raw_tcgen05_scale_chunk(
            scale_address,
            scale_id,
            k / (2 * atom_bytes) - 1,
            scale_rows,
            lanes_per_column,
        )?;
        let (_, last_base_column) = raw_tcgen05_block_scale_address(last_address)?;
        let last_scale_column = last_base_column
            .checked_add(last_scale_row / lanes_per_column)
            .ok_or_else(|| EngineError::message("raw mxf4 cta2 scale column overflow"))?;
        validate_raw_tcgen05_tmem_columns(
            lifecycle,
            access_mode,
            context,
            target_cta,
            first_scale_column,
            last_scale_column - first_scale_column + 1,
        )?;
        let scale_view = raw_tcgen05_tmem_view(physical, context, anchor, target_cta)?;
        let source_view = raw_tcgen05_shared_view_at_cta(physical, context, source, target_cta)?;
        for row in 0..rows_per_cta {
            let matrix_row = if scale_rows_are_joint {
                target_offset
                    .checked_mul(rows_per_cta)
                    .and_then(|value| value.checked_add(row))
                    .ok_or_else(|| EngineError::message("raw mxf4 cta2 scale row overflow"))?
            } else {
                row
            };
            let scales = (0..(k / (2 * atom_bytes)))
                .map(|vector_index| {
                    raw_tcgen05_read_block_scale(
                        physical,
                        &scale_view,
                        scale_address,
                        scale_id,
                        matrix_row,
                        vector_index,
                        scale.decoder(),
                        scale_rows,
                        lanes_per_column,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            for (atom_index, scale_value) in scales.into_iter().enumerate() {
                let mut sub = 0;
                while sub < atom_bytes {
                    let column = atom_index * atom_bytes + sub;
                    let chunk_bytes = (atom_bytes - sub).min(16).min(
                        descriptor.swizzle_atom_bytes - column % descriptor.swizzle_atom_bytes,
                    );

                    let source_offset = raw_tcgen05_shared_byte_offset(
                        source,
                        descriptor,
                        row,
                        atom_index * atom_bytes + sub,
                        chunk_bytes,
                    )?;
                    let mut packed_storage = [0_u8; 16];
                    let packed_bytes = &mut packed_storage[..chunk_bytes];
                    physical
                        .shared()
                        .read_bytes_into(&source_view, source_offset, packed_bytes)?;
                    for &packed in packed_bytes.iter() {
                        for nibble_index in 0..2 {
                            let bits = if nibble_index == 0 {
                                packed & 0xf
                            } else {
                                packed >> 4
                            };
                            let mut value = float4_e2m1fn_bits_to_f32(bits) * scale_value;
                            if negate {
                                value = -value;
                            }
                            values.push(value);
                        }
                    }
                    sub += chunk_bytes;
                }
            }
        }
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_gather_sparse_mxf4_e8m0_matrix(
    physical: &PhysicalMemory,
    context: &WarpContext,
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    rows: usize,
    columns: usize,
    scale_view: &TmemView,
    scale_address: u32,
    scale_id: usize,
    negate: bool,
) -> Result<Vec<f32>, EngineError> {
    if !matches!(columns, 64 | 128) {
        return Err(EngineError::message(
            "sparse mxf4 gather columns must be packed-A 64 or dense-B 128",
        ));
    }
    let mut values = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or_else(|| EngineError::message("sparse mxf4 gather shape overflow"))?,
    );
    for row in 0..rows {
        let scales = [
            raw_tcgen05_read_block_scale(
                physical,
                scale_view,
                scale_address,
                scale_id,
                row,
                0,
                raw_tcgen05_decode_ue8m0_scale,
                0,
                32,
            )?,
            raw_tcgen05_read_block_scale(
                physical,
                scale_view,
                scale_address,
                scale_id,
                row,
                1,
                raw_tcgen05_decode_ue8m0_scale,
                0,
                32,
            )?,
        ];
        for element in 0..columns {
            let byte_in_row = element / 2;
            let source_offset =
                raw_tcgen05_shared_byte_offset(source, descriptor, row, byte_in_row, 1)?;
            let mut packed = [0_u8; 1];
            raw_tcgen05_read_shared_bytes_into(
                physical,
                context,
                source,
                source_offset,
                &mut packed,
            )?;
            let bits = if element % 2 == 0 {
                packed[0] & 0xf
            } else {
                packed[0] >> 4
            };
            let mut value =
                float4_e2m1fn_bits_to_f32(bits) * scales[usize::from(element >= columns / 2)];
            if negate {
                value = -value;
            }
            values.push(value);
        }
    }
    Ok(values)
}

fn raw_tcgen05_sparse_mxf4_metadata_code(
    physical: &PhysicalMemory,
    view: &TmemView,
    address: u32,
    row: usize,
    chunk: usize,
) -> Result<u8, EngineError> {
    let (base_lane, base_column) = raw_tcgen05_address(address, 0, 0)?;
    let lane = base_lane
        .checked_add(row)
        .ok_or_else(|| EngineError::message("sparse mxf4 metadata lane overflow"))?;
    let column = base_column
        .checked_add(chunk / 8)
        .ok_or_else(|| EngineError::message("sparse mxf4 metadata column overflow"))?;
    if lane >= 128 {
        return Err(EngineError::message(
            "sparse mxf4 metadata exceeds TMEM lanes",
        ));
    }
    let mut bytes = [0_u8; 4];
    physical
        .tmem()
        .read_cell_bytes_into(view, lane, column, 0, &mut bytes)?;
    Ok(((u32::from_le_bytes(bytes) >> (4 * (chunk % 8))) & 0xf) as u8)
}

fn raw_tcgen05_expand_sparse_mxf4_a(
    physical: &PhysicalMemory,
    view: &TmemView,
    metadata_address: u32,
    packed: &[f32],
) -> Result<Vec<f32>, EngineError> {
    const M: usize = 128;
    const K: usize = 128;
    if packed.len() != M * (K / 2) {
        return Err(EngineError::message(
            "sparse mxf4 packed A has the wrong shape",
        ));
    }
    let mut dense = vec![0.0_f32; M * K];
    for row in 0..M {
        for chunk in 0..(K / 8) {
            let code = raw_tcgen05_sparse_mxf4_metadata_code(
                physical,
                view,
                metadata_address,
                row,
                chunk,
            )?;
            let first_pair = usize::from(code & 0x3);
            let second_pair = usize::from((code >> 2) & 0x3);
            if first_pair == second_pair {
                return Err(EngineError::message(format!(
                    "sparse mxf4 metadata code 0x{code:x} repeats one pair"
                )));
            }
            let packed_base = row * (K / 2) + chunk * 4;
            let dense_base = row * K + chunk * 8;
            dense[dense_base + first_pair * 2] = packed[packed_base];
            dense[dense_base + first_pair * 2 + 1] = packed[packed_base + 1];
            dense[dense_base + second_pair * 2] = packed[packed_base + 2];
            dense[dense_base + second_pair * 2 + 1] = packed[packed_base + 3];
        }
    }
    Ok(dense)
}

/// Dense FP4 block-scale MMA; shared and TMEM A share scale and accumulator rules.
#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_tcgen05_mma_block_scale_mxf4(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    anchor: &RuntimeBuffer,
    shared_candidates: &[&RuntimeBuffer],
    destination_address: u32,
    a_operand: RawTcgenMmaA,
    b_descriptor_bits: u64,
    sfa_address: u32,
    sfb_address: u32,
    instruction_descriptor: u32,
    enable_input_d: bool,
    issuing_lane: usize,
    scale: RawTcgenMxf4ScaleSpelling,
    cta_group: usize,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
    fixed_vectors: bool,
) -> Result<(), EngineError> {
    let instruction = decode_raw_tcgen_mxf4_instruction_descriptor_for_cta_group(
        instruction_descriptor,
        scale,
        cta_group,
        descriptor_layout,
        fixed_vectors,
    )?;
    validate_raw_tcgen05_issuing_lane(
        context,
        issuing_lane,
        &crate::DiagnosticLabel::new("raw FP4 block-scale MMA"),
    )?;
    let b_descriptor = decode_raw_tcgen_packed_matrix_descriptor(
        b_descriptor_bits,
        descriptor_layout,
        instruction.k / 2,
        false,
    )?;
    let b_source = raw_tcgen05_shared_source(
        context,
        shared_candidates,
        b_descriptor.start_address,
        issuing_lane,
    )?;
    let first_cta = context.cta_id_in_cluster() & !(cta_group - 1);
    if first_cta + cta_group > context.topology().ctas_per_cluster() {
        return Err(EngineError::message(
            "FP4 block-scale MMA has no paired CTA",
        ));
    }
    let (_, destination_column) = raw_tcgen05_address(destination_address, 0, 0)?;
    for cta in first_cta..first_cta + cta_group {
        validate_raw_tcgen05_tmem_columns(
            lifecycle,
            access_mode,
            context,
            cta,
            destination_column,
            instruction.n,
        )?;
    }
    let cta1_view = if cta_group == 1 {
        for (address, rows, id, lanes) in [
            (
                sfa_address,
                instruction.m,
                instruction.sfa_id,
                instruction.sfa_lanes,
            ),
            (sfb_address, instruction.n, instruction.sfb_id, 32),
        ] {
            validate_raw_tcgen05_address_columns(
                lifecycle,
                access_mode,
                context,
                address,
                raw_tcgen05_scale_columns(
                    rows,
                    id,
                    instruction.k / instruction.block_elements,
                    lanes,
                )?,
            )?;
        }
        Some(raw_tcgen05_tmem_view(
            physical,
            context,
            anchor,
            context.cta_id_in_cluster(),
        )?)
    } else {
        None
    };
    let gather_shared =
        |source: &RuntimeBuffer, descriptor, rows, scale_address, scale_id, negate, b| {
            if let Some(view) = cta1_view.as_ref() {
                raw_tcgen05_gather_mxf4_matrix(
                    physical,
                    context,
                    source,
                    descriptor,
                    rows,
                    view,
                    scale_address,
                    scale_id,
                    scale,
                    negate,
                    issuing_lane,
                    instruction.k,
                    instruction.block_elements,
                    if b { 32 } else { instruction.sfa_lanes },
                )
            } else {
                raw_tcgen05_gather_mxf4_cta2_matrix(
                    physical,
                    context,
                    lifecycle,
                    access_mode,
                    anchor,
                    source,
                    descriptor,
                    rows / 2,
                    b,
                    scale_address,
                    scale_id,
                    scale,
                    negate,
                    instruction.k,
                    instruction.block_elements,
                    if b { 32 } else { instruction.sfa_lanes },
                )
            }
        };
    let a_values = match a_operand {
        RawTcgenMmaA::Shared(bits) => {
            let descriptor = decode_raw_tcgen_packed_matrix_descriptor(
                bits,
                descriptor_layout,
                instruction.k / 2,
                false,
            )?;
            let source = raw_tcgen05_shared_source(
                context,
                shared_candidates,
                descriptor.start_address,
                issuing_lane,
            )?;
            gather_shared(
                &source,
                descriptor,
                instruction.m,
                sfa_address,
                instruction.sfa_id,
                instruction.negate_a,
                false,
            )?
        }
        RawTcgenMmaA::Tmem(address) => raw_tcgen05_gather_scaled_tmem_a(
            physical,
            context,
            lifecycle,
            access_mode,
            anchor,
            address,
            instruction.m,
            instruction.k,
            instruction.k / 8,
            cta_group,
            sfa_address,
            instruction.sfa_id,
            instruction.block_elements,
            scale.decoder(),
            instruction.negate_a,
            |words| {
                Ok(words
                    .iter()
                    .flat_map(|word| {
                        (0..8)
                            .map(move |i| float4_e2m1fn_bits_to_f32(((word >> (i * 4)) & 15) as u8))
                    })
                    .collect())
            },
            instruction.sfa_lanes,
        )?,
    };
    let b_values = gather_shared(
        &b_source,
        b_descriptor,
        instruction.n,
        sfb_address,
        instruction.sfb_id,
        instruction.negate_b,
        true,
    )?;
    let destination = match cta1_view.as_ref() {
        Some(tmem_view) => RawMmaDestination::LaneCells(RawMmaWindow {
            physical,
            view: tmem_view,
            destination_address,
            layout: RawTcgenDenseTmemLayout::D,
            cell_dtype: RawMmaCellDtype::F32,
            disable_output_lane: [0_u32; 4],
        }),
        None => RawMmaDestination::Cta2 {
            physical,
            context,
            anchor,
            destination_address,
            layout: raw_tcgen05_cta1_dense_tmem_layout(instruction.m / 2, true)?,
            disable_output_lane: [0_u32; 8],
            cell_dtype: RawMmaCellDtype::F32,
        },
    };
    RawMmaTail {
        m: instruction.m,
        n: instruction.n,
        k: instruction.k,
        enable_input_d,
        destination,
        scale: || Ok(1.0_f32),
    }
    .run(&a_values, &b_values)
}

#[allow(clippy::too_many_arguments)]
pub fn raw_tcgen05_mma_sp_block_scale_mxf4_e8m0_ss_cta1(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    anchor: &RuntimeBuffer,
    shared_candidates: &[&RuntimeBuffer],
    destination_address: u32,
    a_descriptor_bits: u64,
    b_descriptor_bits: u64,
    sfa_address: u32,
    sfb_address: u32,
    metadata_address: u32,
    instruction_descriptor: u32,
    enable_input_d: bool,
    issuing_lane: usize,
) -> Result<(), EngineError> {
    let instruction = decode_raw_tcgen_sparse_mxf4_instruction_descriptor(instruction_descriptor)?;
    let (a_descriptor, b_descriptor, a_source, b_source) = raw_tcgen05_mma_shared_operands(
        context,
        shared_candidates,
        a_descriptor_bits,
        b_descriptor_bits,
        issuing_lane,
        "raw sparse mxf4 tcgen05.mma",
        RawTcgenMatrixDescriptorLayout::Sm100,
        None,
    )?;
    validate_raw_tcgen05_address_columns(
        lifecycle,
        access_mode,
        context,
        destination_address,
        instruction.n,
    )?;
    validate_raw_tcgen05_address_columns(lifecycle, access_mode, context, metadata_address, 2)?;
    validate_raw_tcgen05_address_columns(lifecycle, access_mode, context, sfa_address, 4)?;
    validate_raw_tcgen05_address_columns(
        lifecycle,
        access_mode,
        context,
        sfb_address,
        instruction.n.div_ceil(32),
    )?;
    let view = raw_tcgen05_tmem_view(physical, context, anchor, context.cta_id_in_cluster())?;
    let packed_a = raw_tcgen05_gather_sparse_mxf4_e8m0_matrix(
        physical,
        context,
        &a_source,
        a_descriptor,
        128,
        64,
        &view,
        sfa_address,
        instruction.sfa_id,
        instruction.negate_a,
    )?;
    let a_values = raw_tcgen05_expand_sparse_mxf4_a(physical, &view, metadata_address, &packed_a)?;
    let b_values = raw_tcgen05_gather_sparse_mxf4_e8m0_matrix(
        physical,
        context,
        &b_source,
        b_descriptor,
        instruction.n,
        128,
        &view,
        sfb_address,
        instruction.sfb_id,
        instruction.negate_b,
    )?;
    RawMmaTail {
        m: 128,
        n: instruction.n,
        k: 128,
        enable_input_d,
        destination: RawMmaDestination::Dense(RawMmaWindow {
            physical,
            view: &view,
            destination_address,
            layout: RawTcgenDenseTmemLayout::D,
            cell_dtype: RawMmaCellDtype::F32,
            disable_output_lane: [0_u32; 4],
        }),
        scale: || Ok(1.0_f32),
    }
    .run(&a_values, &b_values)?;
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct RawTcgenDenseInstructionDescriptor {
    cell_dtype: RawMmaCellDtype,
    m: usize,
    n: usize,
    negate_a: bool,
    negate_b: bool,
    transpose_a: bool,
    transpose_b: bool,
}

#[derive(Clone, Copy, Debug)]
struct RawTcgenF8f6f4InstructionDescriptor {
    m: usize,
    n: usize,
    k: usize,
    negate_a: bool,
    negate_b: bool,
    transpose_a: bool,
    transpose_b: bool,
}

/// Decode one dense or sparse `kind::f8f6f4` instruction descriptor.
///
/// The three static operand types are carried by the caller's `V`
/// specialization, so the descriptor bits are *checked* against them rather
/// than read out of them: bits 4-5 hold the D format (0 = f16, 1 = f32) and
/// bits 7-9 / 10-12 hold the independently selected A / B narrow formats.
#[allow(clippy::too_many_arguments)]
fn decode_raw_tcgen_f8f6f4_instruction_descriptor(
    descriptor: u32,
    a_format: RawTcgenNarrowFormat,
    b_format: RawTcgenNarrowFormat,
    d_f16: bool,
    cta_group: u32,
    supports_k64: bool,
    weight_stationary: bool,
    sparse: bool,
) -> Result<RawTcgenF8f6f4InstructionDescriptor, EngineError> {
    if !matches!(cta_group, 1 | 2) {
        return Err(EngineError::message(format!(
            "raw f8f6f4 tcgen05.mma cta_group must be 1 or 2, got {cta_group}"
        )));
    }
    let d_format = u32::from(!d_f16);
    // Bit 29 selects SM107 K=64; WS independently owns bits 30..31.
    let reserved_high_mask = (if weight_stationary { 0 } else { 0x3_u32 << 30 })
        | (if supports_k64 { 0 } else { 1_u32 << 29 });
    let k = if supports_k64 && descriptor & (1_u32 << 29) != 0 {
        64
    } else {
        32
    };
    // Table 48 restricts dense K64 to FP8, not sparse K128. Ordinary
    // sparse operands retain the existing padded-atom / padded-TMEM codecs.
    if k == 64
        && !sparse
        && (a_format.format().width_bits != 8 || b_format.format().width_bits != 8)
    {
        return Err(EngineError::message(
            "dense f8f6f4 K=64 requires E4M3/E5M2 operands",
        ));
    }
    if descriptor & 0xf != if sparse { 4 } else { 0 }
        || descriptor & (1_u32 << 6) != 0
        || descriptor & (1_u32 << 23) != 0
        || descriptor & reserved_high_mask != 0
        || ((descriptor >> 4) & 0x3) != d_format
        || RawTcgenNarrowFormat::decode((descriptor >> 7) & 7, "A")? != a_format
        || RawTcgenNarrowFormat::decode((descriptor >> 10) & 7, "B")? != b_format
    {
        let d_name = if d_f16 { "F16" } else { "F32" };
        let density = if sparse {
            "sparse selector-zero"
        } else {
            "dense"
        };
        return Err(EngineError::message(format!(
            "raw f8f6f4 tcgen05.mma descriptor must encode {density} {d_name}/{a_format:?}/{b_format:?}"
        )));
    }
    if (descriptor & (1 << 15) != 0 && a_format.format().width_bits != 8)
        || (descriptor & (1 << 16) != 0 && b_format.format().width_bits != 8)
    {
        return Err(EngineError::message(
            "raw f8f6f4 MN-major operands must have 8-bit elements",
        ));
    }
    let m = usize::try_from((descriptor >> 24) & 0x1f)
        .map_err(|_| EngineError::message("raw f8f6f4 M conversion failed"))?
        .checked_mul(16)
        .ok_or_else(|| EngineError::message("raw f8f6f4 M overflow"))?;
    let n = usize::try_from((descriptor >> 17) & 0x3f)
        .map_err(|_| EngineError::message("raw f8f6f4 N conversion failed"))?
        .checked_mul(8)
        .ok_or_else(|| EngineError::message("raw f8f6f4 N overflow"))?;
    let b_mn_major = descriptor & (1_u32 << 16) != 0;
    // PTX 9.7.17.10.1: 8-bit MN-major B requires N by 16 for CTA1,
    // by 32 for CTA2. Reuse the existing byte walk for either CTA group.
    let cta1_n_granularity = if b_mn_major { 16 } else { 8 };
    let cta2_n_granularity = if b_mn_major { 32 } else { 16 };
    let valid_geometry = match (cta_group, weight_stationary) {
        (1, true) => matches!(m, 32 | 64 | 128) && (matches!(n, 64 | 128) || (!sparse && n == 256)),
        (1, false) => {
            matches!(m, 64 | 128)
                && (cta1_n_granularity..=256).contains(&n)
                && n % cta1_n_granularity == 0
        }
        (2, false) => {
            matches!(m, 128 | 256)
                && (cta2_n_granularity..=256).contains(&n)
                && n % cta2_n_granularity == 0
        }
        _ => false,
    };
    // Table 48's extended-K M restriction is for dense K=64, not sparse K=128.
    if k == 64 && !sparse && !weight_stationary && m != 128 * cta_group as usize {
        return Err(EngineError::message(
            "dense f8f6f4 K=64 requires M=128 per CTA",
        ));
    }
    if !valid_geometry {
        // PTX Table 48: .ws has its own discrete N shapes, for both dense
        // and sparse forms. Do not report the non-.ws range on this path.
        let requirement = if weight_stationary {
            let n_shapes = if sparse { "{64, 128}" } else { "{64, 128, 256}" };
            format!(".ws cta_group=1, M in {{32, 64, 128}} and N in {n_shapes}")
        } else if cta_group == 1 {
            format!("M in {{64, 128}} and N in {cta1_n_granularity}..=256 by {cta1_n_granularity}")
        } else {
            let b_major = if b_mn_major { "MN-major" } else { "K-major" };
            format!(
                "M in {{128, 256}} and N in {cta2_n_granularity}..=256 by {cta2_n_granularity} for {b_major} B"
            )
        };
        return Err(EngineError::message(format!(
            "raw f8f6f4 tcgen05.mma cta_group={cta_group} requires {requirement}, got M={m}, N={n}"
        )));
    }
    Ok(RawTcgenF8f6f4InstructionDescriptor {
        m,
        n,
        k,
        negate_a: descriptor & (1_u32 << 13) != 0,
        negate_b: descriptor & (1_u32 << 14) != 0,
        transpose_a: descriptor & (1_u32 << 15) != 0,
        transpose_b: descriptor & (1_u32 << 16) != 0,
    })
}

pub(crate) fn raw_tcgen05_mma_f8f6f4_cta1_shape(
    descriptor: u32,
    a_format: RawTcgenNarrowFormat,
    b_format: RawTcgenNarrowFormat,
    d_f16: bool,
    weight_stationary: bool,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
) -> Result<(usize, usize, usize, bool, bool), EngineError> {
    let instruction = decode_raw_tcgen_f8f6f4_instruction_descriptor(
        descriptor,
        a_format,
        b_format,
        d_f16,
        1,
        descriptor_layout.supports_f8f6f4_k64(),
        weight_stationary,
        false,
    )?;
    Ok((
        instruction.m,
        instruction.n,
        instruction.k,
        instruction.transpose_a,
        instruction.transpose_b,
    ))
}

pub(crate) fn raw_tcgen05_mma_f8f6f4_cta2_shape(
    descriptor: u32,
    a_format: RawTcgenNarrowFormat,
    b_format: RawTcgenNarrowFormat,
    d_f16: bool,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
) -> Result<(usize, usize, usize, bool, bool), EngineError> {
    let instruction = decode_raw_tcgen_f8f6f4_instruction_descriptor(
        descriptor,
        a_format,
        b_format,
        d_f16,
        2,
        descriptor_layout.supports_f8f6f4_k64(),
        false,
        false,
    )?;
    Ok((
        instruction.m,
        instruction.n,
        instruction.k,
        instruction.transpose_a,
        instruction.transpose_b,
    ))
}

// PTX Table 48 gives F16/BF16 and TF32 the same M/N contract. Their
// implicit K sizes, operand formats and transpose checks remain family-owned.
fn valid_b16_tf32_shape(
    cta_group: usize,
    m: usize,
    n: usize,
    weight_stationary: bool,
    sparse: bool,
) -> bool {
    match (cta_group, m, weight_stationary) {
        (1, 32, true) => matches!(n, 64 | 128) || (!sparse && n == 256),
        (1, 64, true) if sparse => matches!(n, 64 | 128),
        (1, 64, _) => (8..=256).contains(&n) && n.is_multiple_of(8),
        (1, 128, false) => (8..=256).contains(&n) && n.is_multiple_of(8),
        (1, 128, true) => matches!(n, 64 | 128) || (!sparse && n == 256),
        (2, 128 | 256, false) => (16..=256).contains(&n) && n.is_multiple_of(16),
        _ => false,
    }
}

fn decode_raw_tcgen_tf32_instruction_descriptor(
    descriptor: u32,
    cta_group: usize,
    weight_stationary: bool,
    sparse: bool,
) -> Result<RawTcgenDenseInstructionDescriptor, EngineError> {
    if (descriptor & 4 != 0) != sparse
        || descriptor & (if sparse { 0xa } else { 0xf }) != 0
        || descriptor & (1_u32 << 6) != 0
        || descriptor & (1_u32 << 23) != 0
        || descriptor
            & (if weight_stationary {
                1_u32 << 29
            } else {
                0x7_u32 << 29
            })
            != 0
        || ((descriptor >> 4) & 0x3) != 1
        || ((descriptor >> 7) & 0x7) != 2
        || ((descriptor >> 10) & 0x7) != 2
    {
        return Err(EngineError::message(
            "raw TF32 tcgen05.mma descriptor must encode F32/TF32/TF32 with matching sparsity and a valid selector",
        ));
    }
    let m = usize::try_from((descriptor >> 24) & 0x1f)
        .map_err(|_| EngineError::message("raw TF32 M conversion failed"))?
        .checked_mul(16)
        .ok_or_else(|| EngineError::message("raw TF32 M overflow"))?;
    let n = usize::try_from((descriptor >> 17) & 0x3f)
        .map_err(|_| EngineError::message("raw TF32 N conversion failed"))?
        .checked_mul(8)
        .ok_or_else(|| EngineError::message("raw TF32 N overflow"))?;
    if !valid_b16_tf32_shape(cta_group, m, n, weight_stationary, sparse) {
        return Err(EngineError::message(format!(
            "raw TF32 tcgen05.mma invalid cta_group={cta_group}, WS={weight_stationary}, M={m}, N={n} geometry"
        )));
    }
    Ok(RawTcgenDenseInstructionDescriptor {
        cell_dtype: RawMmaCellDtype::F32,
        m,
        n,
        negate_a: descriptor & (1_u32 << 13) != 0,
        negate_b: descriptor & (1_u32 << 14) != 0,
        transpose_a: descriptor & (1 << 15) != 0,
        transpose_b: descriptor & (1 << 16) != 0,
    })
}
fn decode_raw_tcgen_b16_instruction_descriptor(
    descriptor: u32,
    a_bf16: bool,
    b_bf16: bool,
    cta_group: u32,
    weight_stationary: bool,
    sparse: bool,
) -> Result<RawTcgenDenseInstructionDescriptor, EngineError> {
    let a_format = (descriptor >> 7) & 7;
    let b_format = (descriptor >> 10) & 7;
    if (descriptor & 4 != 0) != sparse
        || descriptor & (if sparse { 0xa } else { 0xf }) != 0
        || descriptor & (1_u32 << 6) != 0
        || descriptor & (1_u32 << 23) != 0
        || descriptor
            & (if weight_stationary {
                1_u32 << 29
            } else {
                0x7_u32 << 29
            })
            != 0
        || ((descriptor >> 4) & 0x3) > 1
        || a_format > 1
        || b_format > 1
    {
        return Err(EngineError::message(
            "raw f16 tcgen05.mma descriptor must encode F16-or-F32/F16-or-BF16 with matching sparsity and a valid selector",
        ));
    }
    if descriptor & (1 << 4) == 0 && (a_format != 0 || b_format != 0) {
        return Err(EngineError::message(
            "F16 accumulation requires F16 operands; BF16 requires F32",
        ));
    }
    let m = usize::try_from((descriptor >> 24) & 0x1f)
        .map_err(|_| EngineError::message("raw f16 M conversion failed"))?
        .checked_mul(16)
        .ok_or_else(|| EngineError::message("raw f16 M overflow"))?;
    let n = usize::try_from((descriptor >> 17) & 0x3f)
        .map_err(|_| EngineError::message("raw f16 N conversion failed"))?
        .checked_mul(8)
        .ok_or_else(|| EngineError::message("raw f16 N overflow"))?;
    if !valid_b16_tf32_shape(cta_group as usize, m, n, weight_stationary, sparse) {
        return Err(EngineError::message(format!(
            "raw f16 tcgen05.mma has invalid cta_group={cta_group}, M={m}, N={n} geometry"
        )));
    }
    // NVIDIA's Blackwell functionality table specifies F16 x F16 or BF16 x
    // BF16 for kind::f16; its SM100 builder enforces identical input types.
    // Validate actual descriptor bits, independently of the selected codec.
    if a_format != b_format {
        return Err(EngineError::message(
            "tcgen05.mma.kind::f16 requires matching F16/BF16 operand types; mixed F16/BF16 is invalid",
        ));
    }
    if a_format != u32::from(a_bf16) || b_format != u32::from(b_bf16) {
        return Err(EngineError::message(
            "raw f16 descriptor operand formats do not match its codec specialization",
        ));
    }
    Ok(RawTcgenDenseInstructionDescriptor {
        cell_dtype: RawMmaCellDtype::from_half(descriptor & (1 << 4) == 0),
        m,
        n,
        negate_a: descriptor & (1_u32 << 13) != 0,
        negate_b: descriptor & (1_u32 << 14) != 0,
        transpose_a: descriptor & (1_u32 << 15) != 0,
        transpose_b: descriptor & (1_u32 << 16) != 0,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_tcgen05_b16_shared_masked_footprints(
    context: &WarpContext,
    shared_candidates: &[&RuntimeBuffer],
    descriptor_bits: u64,
    rows: usize,
    columns: usize,
    transpose: bool,
    issuing_lane: usize,
    mask: Option<RawTcgenColumnMask>,
) -> Result<(RuntimeBuffer, Vec<(usize, Option<usize>, usize, usize)>), EngineError> {
    let descriptor = decode_raw_tcgen_matrix_descriptor(descriptor_bits)?;
    let source = raw_tcgen05_shared_source(
        context,
        shared_candidates,
        descriptor.start_address,
        issuing_lane,
    )?;
    let mut footprints = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or_else(|| EngineError::message("raw f16 MMA footprint shape overflow"))?,
    );
    for row in 0..rows {
        let Some(row) = mask.map_or(Some(row), |mask| mask.source_column(row)) else {
            continue;
        };
        for column in 0..columns {
            let byte_offset =
                raw_tcgen05_b16_matrix_byte_offset(&source, descriptor, row, column, transpose)?;
            footprints.push((issuing_lane, None, byte_offset, 2));
        }
    }
    Ok((source, footprints))
}

pub(crate) fn raw_tcgen05_tf32_shared_footprints(
    context: &WarpContext,
    shared_candidates: &[&RuntimeBuffer],
    descriptor_bits: u64,
    rows: usize,
    columns: usize,
    transpose: bool,
    issuing_lane: usize,
    cta_group: usize,
    mask: Option<RawTcgenColumnMask>,
) -> Result<(RuntimeBuffer, Vec<(usize, Option<usize>, usize, usize)>), EngineError> {
    let descriptor = decode_raw_tcgen_matrix_descriptor(descriptor_bits)?;
    let source = raw_tcgen05_shared_source(
        context,
        shared_candidates,
        descriptor.start_address,
        issuing_lane,
    )?;
    let mut footprints = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or_else(|| EngineError::message("raw TF32 MMA footprint shape overflow"))?,
    );
    let first = context.cta_id_in_cluster() & !(cta_group - 1);
    if first + cta_group > context.topology().ctas_per_cluster() {
        return Err(EngineError::message("TF32 footprint has no paired CTA"));
    }
    for cta in first..first + cta_group {
        for row in 0..rows {
            let Some(row) = mask.map_or(Some(row), |mask| mask.source_column(row)) else {
                continue;
            };
            for k in 0..columns {
                let byte_offset =
                    raw_tcgen05_matrix_byte_offset(&source, descriptor, row, k, transpose, 4)?;
                footprints.push((issuing_lane, Some(cta), byte_offset, 4));
            }
        }
    }
    Ok((source, footprints))
}

pub(crate) fn raw_tcgen05_b16_shared_footprints_cta2(
    context: &WarpContext,
    shared_candidates: &[&RuntimeBuffer],
    descriptor_bits: u64,
    rows_per_cta: usize,
    columns: usize,
    transpose: bool,
    issuing_lane: usize,
) -> Result<(RuntimeBuffer, Vec<(usize, Option<usize>, usize, usize)>), EngineError> {
    let descriptor = decode_raw_tcgen_matrix_descriptor(descriptor_bits)?;
    let source = raw_tcgen05_shared_source(
        context,
        shared_candidates,
        descriptor.start_address,
        issuing_lane,
    )?;
    let pair_base = context.cta_id_in_cluster() & !1_usize;
    if pair_base + 1 >= context.topology().ctas_per_cluster() {
        return Err(EngineError::message(
            "raw f16 cta_group=2 shared footprint has no paired CTA",
        ));
    }
    let mut footprints = Vec::with_capacity(
        rows_per_cta
            .checked_mul(columns)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| EngineError::message("raw f16 cta2 footprint shape overflow"))?,
    );
    for target_cta in [pair_base, pair_base + 1] {
        for row in 0..rows_per_cta {
            for column in 0..columns {
                let byte_offset = raw_tcgen05_b16_matrix_byte_offset(
                    &source, descriptor, row, column, transpose,
                )?;
                footprints.push((issuing_lane, Some(target_cta), byte_offset, 2));
            }
        }
    }
    Ok((source, footprints))
}

pub(crate) fn raw_tcgen05_dense_f32_tmem_footprints(
    destination_address: u32,
    m: usize,
    n: usize,
    layout: RawTcgenDenseTmemLayout,
    disable_output_lane: [u32; 4],
    provenance_lane: usize,
    execution_lane: usize,
) -> Result<Vec<(usize, usize, Option<usize>, i64, i64, i64, usize)>, EngineError> {
    let (base_lane, base_column) = raw_tcgen05_address(destination_address, 0, 0)?;
    let mut footprints = Vec::with_capacity(
        m.checked_mul(n)
            .ok_or_else(|| EngineError::message("raw f16 MMA TMEM footprint shape overflow"))?,
    );
    for row in 0..m {
        for column in 0..n {
            let (lane_delta, column_delta) = layout.location(row, column, m, n)?;
            let mapped_lane = base_lane
                .checked_add(lane_delta)
                .ok_or_else(|| EngineError::message("raw f16 MMA TMEM lane overflow"))?;
            if mapped_lane >= 128 {
                return Err(EngineError::message(format!(
                    "raw f16 MMA TMEM lane {mapped_lane} is outside 128 lanes"
                )));
            }
            if ((disable_output_lane[mapped_lane / 32] >> (mapped_lane % 32)) & 1) != 0 {
                continue;
            }
            let allocated_column = base_column
                .checked_add(column_delta)
                .ok_or_else(|| EngineError::message("raw f16 MMA TMEM column overflow"))?;
            footprints.push((
                provenance_lane,
                execution_lane,
                None,
                i64::try_from(mapped_lane)
                    .map_err(|_| EngineError::message("raw f16 MMA lane exceeds i64"))?,
                0,
                i64::try_from(allocated_column)
                    .map_err(|_| EngineError::message("raw f16 MMA column exceeds i64"))?,
                4,
            ));
        }
    }
    Ok(footprints)
}

// F16 K=16 and FP8 K=32 occupy the same eight 32-bit words per CTA1 row.
const RAW_TCGEN_CTA1_PACKED_A_COLUMNS: usize = 8;

pub(crate) fn raw_tcgen05_packed_tmem_a_column_footprints(
    address: u32,
    m: usize,
    layout: RawTcgenDenseTmemLayout,
    columns: usize,
    provenance_lane: usize,
    execution_lane: usize,
) -> Result<Vec<RawTcgenTmemAccess>, EngineError> {
    let (base_lane, base_column) = raw_tcgen05_address(address, 0, 0)?;
    let mut footprints = Vec::with_capacity(
        layout
            .packed_a_banks()
            .checked_mul(m)
            .and_then(|value| value.checked_mul(columns))
            .ok_or_else(|| EngineError::message("raw packed TMEM A footprint shape overflow"))?,
    );
    for bank in 0..layout.packed_a_banks() {
        for row in 0..m {
            for packed_column in 0..columns {
                let (lane_delta, column_delta) =
                    layout.packed_a_location(bank, row, packed_column, m, columns)?;
                let lane = base_lane
                    .checked_add(lane_delta)
                    .ok_or_else(|| EngineError::message("raw packed TMEM A lane overflow"))?;
                if lane >= 128 {
                    return Err(EngineError::message(format!(
                        "raw packed TMEM A lane {lane} is outside 128 lanes"
                    )));
                }
                let column = base_column
                    .checked_add(column_delta)
                    .ok_or_else(|| EngineError::message("raw packed TMEM A column overflow"))?;
                footprints.push((
                    provenance_lane,
                    execution_lane,
                    None,
                    i64::try_from(lane)
                        .map_err(|_| EngineError::message("raw packed TMEM A lane exceeds i64"))?,
                    0,
                    i64::try_from(column).map_err(|_| {
                        EngineError::message("raw packed TMEM A column exceeds i64")
                    })?,
                    4,
                ));
            }
        }
    }
    Ok(footprints)
}

/// F32 CTA-pair accumulators use two 64-row banks for M=128, one for M=256.
fn raw_tcgen05_cta2_columns_per_bank(m: usize, n: usize) -> usize {
    if m == 128 {
        n / 2
    } else {
        n
    }
}

pub(crate) fn raw_tcgen05_cta2_f32_tmem_footprints(
    context: &WarpContext,
    destination_address: u32,
    m: usize,
    n: usize,
    disable_output_lane: [u32; 8],
    provenance_lane: usize,
    execution_lane: usize,
) -> Result<Vec<(usize, usize, Option<usize>, i64, i64, i64, usize)>, EngineError> {
    raw_tcgen05_cta2_layout_tmem_footprints(
        context,
        destination_address,
        m,
        n,
        raw_tcgen05_cta1_dense_tmem_layout(m / 2, true)?,
        disable_output_lane,
        provenance_lane,
        execution_lane,
    )
}

pub(crate) fn raw_tcgen05_cta2_layout_tmem_footprints(
    context: &WarpContext,
    destination_address: u32,
    m: usize,
    n: usize,
    layout: RawTcgenDenseTmemLayout,
    disable_output_lane: [u32; 8],
    provenance_lane: usize,
    execution_lane: usize,
) -> Result<Vec<RawTcgenTmemAccess>, EngineError> {
    let pair_base = context.cta_id_in_cluster() & !1;
    if !matches!(m, 128 | 256) || pair_base + 1 >= context.topology().ctas_per_cluster() {
        return Err(EngineError::message(
            "invalid CTA2 TMEM shape or missing paired CTA",
        ));
    }
    let mut footprints = Vec::with_capacity(m * n);
    for cta in 0..2 {
        let local = raw_tcgen05_dense_f32_tmem_footprints(
            destination_address,
            m / 2,
            n,
            layout,
            std::array::from_fn(|i| disable_output_lane[cta * 4 + i]),
            provenance_lane,
            execution_lane,
        )?;
        footprints.extend(local.into_iter().map(|mut access| {
            access.2 = Some(pair_base + cta);
            access
        }));
    }
    Ok(footprints)
}

pub(crate) fn raw_tcgen05_cta2_packed_tmem_a_footprints(
    context: &WarpContext,
    address: u32,
    m: usize,
    layout: RawTcgenDenseTmemLayout,
    columns: usize,
    provenance_lane: usize,
    execution_lane: usize,
) -> Result<Vec<RawTcgenTmemAccess>, EngineError> {
    let pair_base = context.cta_id_in_cluster() & !1;
    if !matches!(m, 128 | 256) || pair_base + 1 >= context.topology().ctas_per_cluster() {
        return Err(EngineError::message(
            "invalid CTA2 packed A shape or missing paired CTA",
        ));
    }
    let local = raw_tcgen05_packed_tmem_a_column_footprints(
        address,
        m / 2,
        layout,
        columns,
        provenance_lane,
        execution_lane,
    )?;
    Ok((0..2)
        .flat_map(|cta| {
            local.iter().copied().map(move |mut access| {
                access.2 = Some(pair_base + cta);
                access
            })
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_narrow_shared_atom_accesses(
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    row: usize,
    k: usize,
    format: RawTcgenNarrowFormat,
    atom: usize,
    mut visit: impl FnMut(usize, usize) -> Result<(), EngineError>,
    padded_atoms: bool,
) -> Result<(), EngineError> {
    let bytes = format.payload_bytes_per_k16();
    // PTX 9.7.18.10.4.4: K32 pads each group of 16 values to 16 bytes;
    // K64 packs them contiguously. A packed FP6 group can cross a swizzle atom.
    // Ordinary sparse F8/F6/F4 retains 16-byte atoms even though B has K=64.
    // Block-scaled K64 uses contiguous payloads instead; logical K alone is insufficient.
    let start = atom
        * if padded_atoms {
            16
        } else {
            format.shared_atom_stride(k)
        };
    let mut done = 0;
    while done < bytes {
        let column = start + done;
        let count = (bytes - done)
            .min(descriptor.swizzle_atom_bytes - column % descriptor.swizzle_atom_bytes);
        let offset = raw_tcgen05_shared_byte_offset(source, descriptor, row, column, count)?;
        visit(offset, count)?;
        done += count;
    }
    Ok(())
}

/// PTX 9.4 Tables 49 and Figures 279--282: LUT-B reads a full 48-byte
/// compressed K=128 row; descriptor bit 53 selects one 24-byte K=64 segment.
/// Keep this walk shared by numerical reads and asynchronous read footprints.
fn raw_tcgen05_lut_b_row_accesses(
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    row: usize,
    mut visit: impl FnMut(usize, usize) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    let mut byte = 0;
    while byte < 48 {
        let count =
            (48 - byte).min(descriptor.swizzle_atom_bytes - byte % descriptor.swizzle_atom_bytes);
        visit(
            raw_tcgen05_shared_byte_offset(source, descriptor, row, byte, count)?,
            count,
        )?;
        byte += count;
    }
    Ok(())
}

fn raw_tcgen05_lut_b_location(
    address: u32,
    group: usize,
    word: usize,
) -> Result<(usize, usize), EngineError> {
    let (base_row, base_column) = raw_tcgen05_address(address, 0, 0)?;
    if address & 1 != 0 {
        return Err(EngineError::message(
            "tcgen05.mma LUT-B address must be two-column aligned",
        ));
    }
    let row = base_row
        .checked_add(group)
        .filter(|row| *row < 128)
        .ok_or_else(|| EngineError::message("tcgen05.mma LUT-B row is outside TMEM"))?;
    let column = base_column
        .checked_add(word)
        .filter(|column| *column < 65536)
        .ok_or_else(|| EngineError::message("tcgen05.mma LUT-B column overflow"))?;
    Ok((row, column))
}

fn raw_tcgen05_f8_b_descriptor(
    bits: u64,
    layout: RawTcgenMatrixDescriptorLayout,
    lut_b: Option<u32>,
) -> Result<RawTcgenMatrixDescriptor, EngineError> {
    if lut_b.is_some() {
        if layout != RawTcgenMatrixDescriptorLayout::Sm107 {
            return Err(EngineError::message("tcgen05.mma LUT-B requires SM107"));
        }
        decode_raw_tcgen_packed_matrix_descriptor(bits & !(1_u64 << 53), layout, 48, false)
    } else {
        decode_raw_tcgen_matrix_descriptor_for_layout(bits, layout)
    }
}

fn raw_tcgen05_validate_lut_b(
    lut_b: Option<u32>,
    k: usize,
    b_format: RawTcgenNarrowFormat,
    transpose_b: bool,
) -> Result<(), EngineError> {
    if let Some(address) = lut_b {
        if k != 64 || b_format != RawTcgenNarrowFormat::E4M3 || transpose_b {
            return Err(EngineError::message(
                "tcgen05.mma LUT-B requires K=64 and non-transposed E4M3 B",
            ));
        }
        raw_tcgen05_lut_b_location(address, 0, 0)?;
    }
    Ok(())
}

pub(crate) fn raw_tcgen05_lut_b_tmem_footprints(
    context: &WarpContext,
    address: u32,
    rows: usize,
    cta_group: usize,
    issuing_lane: usize,
) -> Result<Vec<RawTcgenTmemAccess>, EngineError> {
    if !matches!(cta_group, 1 | 2) || rows == 0 || rows % 8 != 0 {
        return Err(EngineError::message("invalid LUT-B geometry"));
    }
    let first = context.cta_id_in_cluster() & !(cta_group - 1);
    if first + cta_group > context.topology().ctas_per_cluster() {
        return Err(EngineError::message("LUT-B has no paired CTA"));
    }
    let mut accesses = Vec::with_capacity(rows / 8 * 2 * cta_group);
    for cta in first..first + cta_group {
        for group in 0..rows / 8 {
            for word in 0..2 {
                let (row, col) = raw_tcgen05_lut_b_location(address, group, word)?;
                accesses.push((
                    issuing_lane,
                    issuing_lane,
                    Some(cta),
                    row as i64,
                    0,
                    col as i64,
                    4,
                ));
            }
        }
    }
    Ok(accesses)
}

#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_gather_lut_b(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    anchor: &RuntimeBuffer,
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    segment: usize,
    lookup_address: u32,
    rows: usize,
    cta_group: usize,
    negate: bool,
) -> Result<Vec<f32>, EngineError> {
    let first_cta = context.cta_id_in_cluster() & !(cta_group - 1);
    let (_, column) = raw_tcgen05_lut_b_location(lookup_address, 0, 0)?;
    let mut values = Vec::with_capacity(rows * 64 * cta_group);
    for cta in first_cta..first_cta + cta_group {
        validate_raw_tcgen05_tmem_columns(lifecycle, access_mode, context, cta, column, 2)?;
        let view = raw_tcgen05_tmem_view(physical, context, anchor, cta)?;
        for group in 0..rows / 8 {
            let mut lookup = [0_u8; 8];
            for word in 0..2 {
                let (row, column) = raw_tcgen05_lut_b_location(lookup_address, group, word)?;
                physical.tmem().read_cell_bytes_into(
                    &view,
                    row,
                    column,
                    0,
                    &mut lookup[word * 4..word * 4 + 4],
                )?;
            }
            for row in group * 8..group * 8 + 8 {
                let mut compressed = [0_u8; 48];
                let mut done = 0;
                raw_tcgen05_lut_b_row_accesses(source, descriptor, row, |offset, count| {
                    raw_tcgen05_read_shared_bytes_at_cta_into(
                        physical,
                        context,
                        source,
                        cta,
                        offset,
                        &mut compressed[done..done + count],
                    )?;
                    done += count;
                    Ok(())
                })?;
                for k in 0..64 {
                    let bit = segment * 24 * 8 + k * 3;
                    let byte = bit / 8;
                    let mut index = u16::from(compressed[byte]) >> (bit % 8);
                    if bit % 8 > 5 {
                        index |= u16::from(compressed[byte + 1]) << (8 - bit % 8);
                    }
                    let value =
                        RawTcgenNarrowFormat::E4M3.decode_value(lookup[usize::from(index & 7)]);
                    values.push(if negate { -value } else { value });
                }
            }
        }
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_gather_f8_shared_matrix(
    physical: &PhysicalMemory,
    context: &WarpContext,
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    rows: usize,
    k_extent: usize,
    format: RawTcgenNarrowFormat,
    negate: bool,
    transpose: bool,
    cta_group: usize,
    mask: Option<RawTcgenColumnMask>,
    padded_atoms: bool,
) -> Result<Vec<f32>, EngineError> {
    if !matches!(cta_group, 1 | 2)
        || k_extent % 16 != 0
        || (transpose && format.format().width_bits != 8)
    {
        return Err(EngineError::message(
            "invalid narrow MMA gather geometry/transpose",
        ));
    }
    let first_cta = context.cta_id_in_cluster() & !(cta_group - 1);
    if first_cta + cta_group > context.topology().ctas_per_cluster() {
        return Err(EngineError::message(
            "raw f8f6f4 shared gather has no paired CTA",
        ));
    }
    let count = rows
        .checked_mul(k_extent)
        .and_then(|n| n.checked_mul(cta_group))
        .ok_or_else(|| EngineError::message("raw f8f6f4 gather shape overflow"))?;
    let mut values = Vec::with_capacity(count);
    for target_cta in first_cta..first_cta + cta_group {
        for row in 0..rows {
            let Some(row) = mask.map_or(Some(row), |mask| mask.source_column(row)) else {
                values.resize(values.len() + k_extent, 0.0);
                continue;
            };
            for atom in 0..k_extent / 16 {
                let decoded = if transpose {
                    let mut decoded = [0.0; 16];
                    for (i, value) in decoded.iter_mut().enumerate() {
                        let offset = raw_tcgen05_8bit_matrix_byte_offset(
                            source,
                            descriptor,
                            row,
                            atom * 16 + i,
                            true,
                        )?;
                        let mut bits = [0_u8];
                        raw_tcgen05_read_shared_bytes_at_cta_into(
                            physical, context, source, target_cta, offset, &mut bits,
                        )?;
                        *value = format.decode_value(bits[0]);
                    }
                    decoded
                } else {
                    let mut bits = [0_u8; 16];
                    let mut done = 0;
                    raw_tcgen05_narrow_shared_atom_accesses(
                        source,
                        descriptor,
                        row,
                        k_extent,
                        format,
                        atom,
                        |offset, count| {
                            raw_tcgen05_read_shared_bytes_at_cta_into(
                                physical,
                                context,
                                source,
                                target_cta,
                                offset,
                                &mut bits[done..done + count],
                            )?;
                            done += count;
                            Ok(())
                        },
                        padded_atoms,
                    )?;
                    format.decode_shared_atom(bits)
                };
                values.extend(decoded.into_iter().map(|v| if negate { -v } else { v }));
            }
        }
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_gather_tf32_shared_matrix(
    physical: &PhysicalMemory,
    context: &WarpContext,
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    rows: usize,
    columns: usize,
    transpose: bool,
    negate: bool,
    cta_group: usize,
    mask: Option<RawTcgenColumnMask>,
) -> Result<Vec<f32>, EngineError> {
    let mut values = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or_else(|| EngineError::message("raw TF32 gather shape overflow"))?,
    );
    let first = context.cta_id_in_cluster() & !(cta_group - 1);
    for cta in first..first + cta_group {
        let view = raw_tcgen05_shared_view_at_cta(physical, context, source, cta)?;
        for row in 0..rows {
            let Some(row) = mask.map_or(Some(row), |mask| mask.source_column(row)) else {
                values.resize(values.len() + columns, 0.0);
                continue;
            };
            for k in 0..columns {
                let source_offset =
                    raw_tcgen05_matrix_byte_offset(source, descriptor, row, k, transpose, 4)?;
                let mut bytes = [0_u8; 4];
                physical
                    .shared()
                    .read_bytes_into(&view, source_offset, &mut bytes)?;
                let mut value = raw_tcgen05_tf32_payload_to_f32(u32::from_le_bytes(bytes));
                if negate {
                    value = -value;
                }
                values.push(value);
            }
        }
    }
    Ok(values)
}

fn raw_tcgen05_gather_b16_shared_matrix(
    physical: &PhysicalMemory,
    context: &WarpContext,
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    rows: usize,
    columns: usize,
    bf16: bool,
    negate: bool,
    transpose: bool,
    mask: Option<RawTcgenColumnMask>,
) -> Result<Vec<f32>, EngineError> {
    raw_tcgen05_gather_b16_shared_masked_with(
        physical,
        context,
        source,
        descriptor,
        rows,
        columns,
        transpose,
        mask,
        |bits| {
            let value = if bf16 {
                bf16_bits_to_f32(bits)
            } else {
                fp16_bits_to_f32(bits)
            };
            Ok(if negate { -value } else { value })
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_gather_b16_shared_masked_with<Scalar: Default>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    rows: usize,
    columns: usize,
    transpose: bool,
    mask: Option<RawTcgenColumnMask>,
    decode: impl Fn(u16) -> Result<Scalar, EngineError>,
) -> Result<Vec<Scalar>, EngineError> {
    let mut values = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or_else(|| EngineError::message("raw sparse b16 gather shape overflow"))?,
    );
    for row in 0..rows {
        let Some(row) = mask.map_or(Some(row), |mask| mask.source_column(row)) else {
            values.extend(std::iter::repeat_with(Scalar::default).take(columns));
            continue;
        };
        for column in 0..columns {
            let offset =
                raw_tcgen05_b16_matrix_byte_offset(source, descriptor, row, column, transpose)?;
            let mut bytes = [0_u8; 2];
            raw_tcgen05_read_shared_bytes_into(physical, context, source, offset, &mut bytes)?;
            values.push(decode(u16::from_le_bytes(bytes))?);
        }
    }
    Ok(values)
}

#[derive(Clone, Copy)]
pub(crate) enum RawTcgenSparseMetadataLayout {
    B16 { selector: usize },
    Narrow { k: usize },
}
impl RawTcgenSparseMetadataLayout {
    fn chunks(self) -> usize {
        match self {
            Self::B16 { .. } => 8,
            Self::Narrow { k } => k / 4,
        }
    }
}

fn raw_tcgen05_sparse_metadata_code(
    physical: &PhysicalMemory,
    view: &TmemView,
    address: u32,
    metadata_layout: RawTcgenSparseMetadataLayout,
    row: usize,
    chunk: usize,
) -> Result<u8, EngineError> {
    let (lane, column, nibble) =
        raw_tcgen05_sparse_metadata_location(address, metadata_layout, row, chunk)?;
    let mut bytes = [0_u8; 4];
    physical
        .tmem()
        .read_cell_bytes_into(view, lane, column, 0, &mut bytes)?;
    Ok(((u32::from_le_bytes(bytes) >> (4 * nibble)) & 0xf) as u8)
}

fn raw_tcgen05_sparse_metadata_location(
    address: u32,
    metadata_layout: RawTcgenSparseMetadataLayout,
    row: usize,
    chunk: usize,
) -> Result<(usize, usize, usize), EngineError> {
    let (base_lane, base_column) = raw_tcgen05_address(address, 0, 0)?;
    if let RawTcgenSparseMetadataLayout::Narrow { k } = metadata_layout {
        if !matches!(k, 64 | 128) || chunk >= k / 4 {
            return Err(EngineError::message(
                "sparse narrow metadata chunk exceeds its K extent",
            ));
        }
        let lane = base_lane
            .checked_add(row)
            .ok_or_else(|| EngineError::message("sparse metadata lane overflow"))?;
        let column = base_column
            .checked_add(chunk / 8)
            .ok_or_else(|| EngineError::message("sparse metadata column overflow"))?;
        if lane >= 128 {
            return Err(EngineError::message("sparse metadata lane exceeds TMEM"));
        }
        return Ok((lane, column, chunk % 8));
    }
    let RawTcgenSparseMetadataLayout::B16 { selector } = metadata_layout else {
        unreachable!()
    };
    let row_in_partition = row % 32;
    let lane = base_lane
        .checked_add((row / 32) * 32)
        .and_then(|value| value.checked_add(row_in_partition % 8))
        .and_then(|value| value.checked_add(16 * (row_in_partition / 16)))
        .and_then(|value| value.checked_add(8 * (chunk / 4)))
        .ok_or_else(|| EngineError::message("sparse f16 metadata lane overflow"))?;
    let column = base_column
        .checked_add(selector)
        .ok_or_else(|| EngineError::message("sparse f16 metadata column overflow"))?;
    if lane >= 128 {
        return Err(EngineError::message(format!(
            "sparse f16 metadata lane {lane} is outside TMEM"
        )));
    }
    let nibble = 4 * ((row_in_partition % 16) / 8) + chunk % 4;
    Ok((lane, column, nibble))
}
pub(crate) fn raw_tcgen05_sparse_metadata_footprints(
    context: &WarpContext,
    address: u32,
    metadata_layout: RawTcgenSparseMetadataLayout,
    rows: usize,
    layout: RawTcgenDenseTmemLayout,
    issuing_lane: usize,
    cta_group: usize,
) -> Result<Vec<RawTcgenTmemAccess>, EngineError> {
    let mut locations = std::collections::BTreeSet::new();
    for bank in 0..layout.packed_a_banks() {
        for row in 0..rows {
            let physical_row = layout
                .packed_a_location(bank, row, 0, rows, RAW_TCGEN_CTA1_PACKED_A_COLUMNS)?
                .0;
            for chunk in 0..metadata_layout.chunks() {
                let (lane, column, _) = raw_tcgen05_sparse_metadata_location(
                    address,
                    metadata_layout,
                    physical_row,
                    chunk,
                )?;
                locations.insert((lane, column));
            }
        }
    }
    let first_cta = context.cta_id_in_cluster() & !(cta_group - 1);
    Ok((0..cta_group)
        .flat_map(|cta| {
            locations.iter().copied().map(move |(lane, col)| {
                (
                    issuing_lane,
                    issuing_lane,
                    if cta_group == 2 {
                        Some(first_cta + cta)
                    } else {
                        None
                    },
                    lane as i64,
                    0,
                    col as i64,
                    4,
                )
            })
        })
        .collect())
}

fn raw_tcgen05_sparse_2of4_indices(code: u8) -> Result<[usize; 2], EngineError> {
    if !matches!(code, 0x4 | 0x8 | 0xc | 0x9 | 0xd | 0x6 | 0xe) {
        return Err(EngineError::message(format!(
            "sparse 2:4 metadata code 0x{code:x} is not a defined index pair"
        )));
    }
    Ok([usize::from(code & 3), usize::from((code >> 2) & 3)])
}

fn raw_tcgen05_expand_sparse_2of4<Scalar: Copy + Default>(
    packed: &[Scalar],
    rows: usize,
    layout: RawTcgenDenseTmemLayout,
    k: usize,
    metadata: impl Fn(usize, usize) -> Result<u8, EngineError>,
) -> Result<Vec<Scalar>, EngineError> {
    let banks = layout.packed_a_banks();
    let packed_banks = packed.len() / (rows * (k / 2));
    if !matches!(packed_banks, 1) && packed_banks != banks || packed.len() % (rows * (k / 2)) != 0 {
        return Err(EngineError::message(
            "sparse 2:4 packed A has the wrong shape",
        ));
    }
    let mut dense = vec![Scalar::default(); banks * rows * k];
    for bank in 0..banks {
        for row in 0..rows {
            let physical_row = layout
                .packed_a_location(bank, row, 0, rows, RAW_TCGEN_CTA1_PACKED_A_COLUMNS)?
                .0;
            for chunk in 0..(k / 4) {
                let code = metadata(physical_row, chunk)?;
                let [first, second] = raw_tcgen05_sparse_2of4_indices(code)?;
                let source = ((bank % packed_banks) * rows + row) * (k / 2) + chunk * 2;
                let destination = (bank * rows + row) * k + chunk * 4;
                dense[destination + first] = packed[source];
                dense[destination + second] = packed[source + 1];
            }
        }
    }
    Ok(dense)
}
fn raw_tcgen05_layout_f_lane(row: usize) -> Result<usize, EngineError> {
    if row >= 64 {
        return Err(EngineError::message(format!(
            "raw TCGEN Layout F row {row} is outside 64 rows"
        )));
    }
    (row / 16)
        .checked_mul(32)
        .and_then(|value| value.checked_add(row % 16))
        .ok_or_else(|| EngineError::message("raw TCGEN Layout F lane overflow"))
}

/// Physical TMEM organization selected by a dense CTA1 MMA.
///
/// PTX ISA 9.3 section 9.7.17.10.5 assigns M=128 to Layout D, M=64 `.ws`
/// to Layout E, and M=64 non-`.ws` to Layout F.  Keeping the letter in the
/// runtime contract prevents the two M=64 forms from being conflated merely
/// because they have the same logical matrix shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RawTcgenDenseTmemLayout {
    D,
    E,
    F,
    G,
}

pub(crate) fn raw_tcgen05_cta1_dense_tmem_layout(
    m: usize,
    weight_stationary: bool,
) -> Result<RawTcgenDenseTmemLayout, EngineError> {
    match (m, weight_stationary) {
        (128, _) => Ok(RawTcgenDenseTmemLayout::D),
        (64, true) => Ok(RawTcgenDenseTmemLayout::E),
        (64, false) => Ok(RawTcgenDenseTmemLayout::F),
        (32, true) => Ok(RawTcgenDenseTmemLayout::G),
        _ => Err(EngineError::message(format!(
            "raw CTA1 dense TMEM layout has unsupported M={m}"
        ))),
    }
}

impl RawTcgenDenseTmemLayout {
    fn physical_columns(self, logical_columns: usize) -> Result<usize, EngineError> {
        match self {
            Self::D | Self::F => Ok(logical_columns),
            Self::E | Self::G if logical_columns.is_multiple_of(self.packed_a_banks()) => {
                Ok(logical_columns / self.packed_a_banks())
            }
            Self::E | Self::G => Err(EngineError::message(format!(
                "raw TCGEN banked layout requires a divisible column count, got {logical_columns}"
            ))),
        }
    }

    fn location(
        self,
        row: usize,
        column: usize,
        rows: usize,
        columns: usize,
    ) -> Result<(usize, usize), EngineError> {
        if row >= rows || column >= columns {
            return Err(EngineError::message(
                "raw dense TMEM logical coordinate is outside its matrix",
            ));
        }
        match self {
            Self::D => Ok((row, column)),
            Self::F => Ok((raw_tcgen05_layout_f_lane(row)?, column)),
            Self::E | Self::G => {
                let bank_rows = 128 / self.packed_a_banks();
                if rows != bank_rows {
                    return Err(EngineError::message(format!(
                        "raw TCGEN banked layout requires {bank_rows} rows, got {rows}"
                    )));
                }
                let bank_columns = self.physical_columns(columns)?;
                let lane = row
                    .checked_add(bank_rows * (column / bank_columns))
                    .ok_or_else(|| EngineError::message("raw TCGEN Layout E lane overflow"))?;
                Ok((lane, column % bank_columns))
            }
        }
    }

    fn packed_a_banks(self) -> usize {
        match self {
            Self::E => 2,
            Self::G => 4,
            _ => 1,
        }
    }

    fn packed_a_location(
        self,
        bank: usize,
        row: usize,
        packed_column: usize,
        rows: usize,
        columns: usize,
    ) -> Result<(usize, usize), EngineError> {
        if bank >= self.packed_a_banks() {
            return Err(EngineError::message(
                "raw packed TMEM A bank is outside its datapath layout",
            ));
        }
        if matches!(self, Self::E | Self::G) {
            let bank_rows = 128 / self.packed_a_banks();
            if rows != bank_rows {
                return Err(EngineError::message(format!(
                    "raw TCGEN banked A layout requires {bank_rows} rows, got {rows}"
                )));
            }
            let lane = row
                .checked_add(bank_rows * bank)
                .ok_or_else(|| EngineError::message("raw TCGEN Layout E A lane overflow"))?;
            return Ok((lane, packed_column));
        }
        self.location(row, packed_column, rows, columns)
    }
}

fn raw_tcgen05_gather_b16_tmem_a(
    physical: &PhysicalMemory,
    view: &TmemView,
    address: u32,
    rows: usize,
    layout: RawTcgenDenseTmemLayout,
    bf16: bool,
    negate: bool,
) -> Result<Vec<f32>, EngineError> {
    raw_tcgen05_gather_packed_tmem_a(physical, view, address, rows, layout, negate, |word| {
        [word as u16, (word >> 16) as u16].map(|bits| {
            if bf16 {
                bf16_bits_to_f32(bits)
            } else {
                fp16_bits_to_f32(bits)
            }
        })
    })
}

fn raw_tcgen05_gather_packed_tmem_a<const ELEMENTS_PER_WORD: usize>(
    physical: &PhysicalMemory,
    view: &TmemView,
    address: u32,
    rows: usize,
    layout: RawTcgenDenseTmemLayout,
    negate: bool,
    decode: impl Fn(u32) -> [f32; ELEMENTS_PER_WORD],
) -> Result<Vec<f32>, EngineError> {
    raw_tcgen05_gather_packed_tmem_a_with(physical, view, address, rows, layout, |word| {
        Ok(decode(word).map(|value| if negate { -value } else { value }))
    })
}

fn raw_tcgen05_gather_packed_tmem_a_with<Scalar, const ELEMENTS_PER_WORD: usize>(
    physical: &PhysicalMemory,
    view: &TmemView,
    address: u32,
    rows: usize,
    layout: RawTcgenDenseTmemLayout,
    decode: impl Fn(u32) -> Result<[Scalar; ELEMENTS_PER_WORD], EngineError>,
) -> Result<Vec<Scalar>, EngineError> {
    raw_tcgen05_gather_packed_tmem_a_columns(
        physical,
        view,
        address,
        rows,
        layout,
        RAW_TCGEN_CTA1_PACKED_A_COLUMNS,
        decode,
    )
}

#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_gather_packed_tmem_a_columns<Scalar, const ELEMENTS_PER_WORD: usize>(
    physical: &PhysicalMemory,
    view: &TmemView,
    address: u32,
    rows: usize,
    layout: RawTcgenDenseTmemLayout,
    columns: usize,
    decode: impl Fn(u32) -> Result<[Scalar; ELEMENTS_PER_WORD], EngineError>,
) -> Result<Vec<Scalar>, EngineError> {
    let (base_lane, base_col) = raw_tcgen05_address(address, 0, 0)?;
    let capacity = rows
        .checked_mul(columns * ELEMENTS_PER_WORD)
        .and_then(|value| value.checked_mul(layout.packed_a_banks()))
        .ok_or_else(|| EngineError::message("raw packed TMEM A shape overflow"))?;
    let mut values = Vec::with_capacity(capacity);
    for bank in 0..layout.packed_a_banks() {
        for row in 0..rows {
            for packed_column in 0..columns {
                let (lane_delta, column_delta) =
                    layout.packed_a_location(bank, row, packed_column, rows, columns)?;
                let lane = base_lane
                    .checked_add(lane_delta)
                    .ok_or_else(|| EngineError::message("raw packed TMEM A lane overflow"))?;
                if lane >= 128 {
                    return Err(EngineError::message(format!(
                        "raw packed TMEM A lane {lane} is outside 128 lanes"
                    )));
                }
                let column = base_col
                    .checked_add(column_delta)
                    .ok_or_else(|| EngineError::message("raw packed TMEM A column overflow"))?;
                let mut bytes = [0_u8; 4];
                physical
                    .tmem()
                    .read_cell_bytes_into(view, lane, column, 0, &mut bytes)?;
                let word = u32::from_le_bytes(bytes);
                for value in decode(word)? {
                    values.push(value);
                }
            }
        }
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_gather_scaled_tmem_a(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access: TmemAccessMode,
    anchor: &RuntimeBuffer,
    address: u32,
    m: usize,
    k: usize,
    columns: usize,
    cta_group: usize,
    scale_address: u32,
    scale_id: usize,
    block_elements: usize,
    scale_decode: RawTcgenScaleDecoder,
    negate: bool,
    decode: impl Fn(&[u32]) -> Result<Vec<f32>, EngineError>,
    lanes_per_column: usize,
) -> Result<Vec<f32>, EngineError> {
    let rows = m / cta_group;
    let first_cta = context.cta_id_in_cluster() & !(cta_group - 1);
    if !(rows == 128 || (cta_group == 2 && rows == 64))
        || columns == 0
        || !k.is_multiple_of(block_elements)
        || first_cta + cta_group > context.topology().ctas_per_cluster()
    {
        return Err(EngineError::message("invalid block-scale TMEM A geometry"));
    }
    let (_, a_column) = raw_tcgen05_address(address, 0, 0)?;
    let (_, scale_column) = raw_tcgen05_block_scale_address(scale_address)?;
    let scale_columns =
        raw_tcgen05_scale_columns(rows, scale_id, k / block_elements, lanes_per_column)?;
    let layout = raw_tcgen05_cta1_dense_tmem_layout(rows, true)?;
    let mut result = vec![0.0; layout.packed_a_banks() * m * k];
    for cta in first_cta..first_cta + cta_group {
        validate_raw_tcgen05_tmem_columns(lifecycle, access, context, cta, a_column, columns)?;
        validate_raw_tcgen05_tmem_columns(
            lifecycle,
            access,
            context,
            cta,
            scale_column,
            scale_columns,
        )?;
        let view = raw_tcgen05_tmem_view(physical, context, anchor, cta)?;
        let words = raw_tcgen05_gather_packed_tmem_a_columns(
            physical,
            &view,
            address,
            rows,
            layout,
            columns,
            |word| Ok([word]),
        )?;
        for (row, words) in words.chunks_exact(columns).enumerate() {
            let bank = row / rows;
            let row = row % rows;
            let mut values = decode(words)?;
            if values.len() != k {
                return Err(EngineError::message(
                    "scaled TMEM decoder returned the wrong K extent",
                ));
            }
            for (block, values) in values.chunks_exact_mut(block_elements).enumerate() {
                let scale = raw_tcgen05_read_block_scale(
                    physical,
                    &view,
                    scale_address,
                    scale_id,
                    row,
                    block,
                    scale_decode,
                    rows,
                    lanes_per_column,
                )?;
                for value in values {
                    *value *= scale;
                    if negate {
                        *value = -*value;
                    }
                }
            }
            let start = (bank * m + (cta - first_cta) * rows + row) * k;
            result[start..start + k].copy_from_slice(&values);
        }
    }
    Ok(result)
}

#[inline]
pub fn mma_f32_dot_increasing_k(
    a_values: &[f32],
    b_values: &[f32],
    mut accumulator: f32,
) -> Result<f32, EngineError> {
    if a_values.len() != b_values.len() {
        return Err(EngineError::message(format!(
            "MMA dot operands have different K extents: {} and {}",
            a_values.len(),
            b_values.len(),
        )));
    }
    for (&a, &b) in a_values.iter().zip(b_values) {
        accumulator = a.mul_add(b, accumulator);
    }
    Ok(accumulator)
}

pub fn mma_f32_abt_increasing_k(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f32],
    b_values: &[f32],
    input_d: Option<(&[f32], f32)>,
) -> Result<Vec<f32>, EngineError> {
    let _profile = ProfileTimer::new(ProfileKind::MmaIncreasingK);
    let a_len = m
        .checked_mul(k)
        .ok_or_else(|| EngineError::message("raw TCGEN A matrix shape overflow"))?;
    let b_len = n
        .checked_mul(k)
        .ok_or_else(|| EngineError::message("raw TCGEN B matrix shape overflow"))?;
    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| EngineError::message("raw TCGEN output matrix shape overflow"))?;
    if a_values.len() != a_len {
        return Err(EngineError::message(format!(
            "raw TCGEN A matrix has {} values, expected {a_len}",
            a_values.len()
        )));
    }
    if b_values.len() != b_len {
        return Err(EngineError::message(format!(
            "raw TCGEN B matrix has {} values, expected {b_len}",
            b_values.len()
        )));
    }
    if let Some((values, _)) = input_d {
        if values.len() != output_len {
            return Err(EngineError::message(format!(
                "raw TCGEN input D matrix has {} values, expected {output_len}",
                values.len()
            )));
        }
    }

    // NumSim uses one stable logical oracle: every output element accumulates K
    // in increasing order by binary32 FMA. D is the initial accumulator, not a
    // second rounded addition after a matrix product.
    //
    // Store B transposed so the independent columns of one output row are
    // contiguous. Advancing all columns together for each increasing K keeps
    // every element's exact FMA chain unchanged while allowing LLVM to
    // vectorize those independent chains.
    let mut b_transposed = vec![0.0; b_len];
    for col in 0..n {
        for inner in 0..k {
            b_transposed[inner * n + col] = b_values[col * k + inner];
        }
    }
    let mut output = input_d
        .map(|(values, scale)| values.iter().map(|&value| value * scale).collect())
        .unwrap_or_else(|| vec![0.0_f32; output_len]);
    numsim_fp_env::fma_f32_abt_increasing_k(m, n, k, a_values, &b_transposed, &mut output).map_err(
        |error| EngineError::message(format!("raw TCGEN MMA SIMD shape error: {error}")),
    )?;
    Ok(output)
}

fn mma_f32_abt_banked_a_increasing_k(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f32],
    b_values: &[f32],
    input_d: Option<(&[f32], f32)>,
    banks: usize,
) -> Result<Vec<f32>, EngineError> {
    if !matches!(banks, 2 | 4) || n % banks != 0 {
        return Err(EngineError::message(format!(
            "raw banked-A MMA requires N divisible by {banks} banks, got {n}"
        )));
    }
    let bank_columns = n / banks;
    let a_bank_len = m
        .checked_mul(k)
        .ok_or_else(|| EngineError::message("raw banked-A shape overflow"))?;
    let expected_a_len = a_bank_len
        .checked_mul(banks)
        .ok_or_else(|| EngineError::message("raw banked-A shape overflow"))?;
    if a_values.len() != expected_a_len {
        return Err(EngineError::message(format!(
            "raw banked A has {} values, expected {expected_a_len}",
            a_values.len()
        )));
    }
    let expected_b_len = n
        .checked_mul(k)
        .ok_or_else(|| EngineError::message("raw banked-B shape overflow"))?;
    if b_values.len() != expected_b_len {
        return Err(EngineError::message(format!(
            "raw B matrix has {} values, expected {expected_b_len}",
            b_values.len()
        )));
    }
    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| EngineError::message("raw banked output shape overflow"))?;
    if let Some((values, _)) = input_d {
        if values.len() != output_len {
            return Err(EngineError::message(format!(
                "raw input D matrix has {} values, expected {output_len}",
                values.len()
            )));
        }
    }

    let mut output = vec![0.0_f32; output_len];
    for bank in 0..banks {
        let a_start = bank * a_bank_len;
        let b_bank_len = bank_columns * k;
        let b_start = bank * b_bank_len;
        let input_bank = input_d.map(|(values, scale)| {
            let mut selected = Vec::with_capacity(m * bank_columns);
            for row in 0..m {
                let start = row * n + bank * bank_columns;
                selected.extend_from_slice(&values[start..start + bank_columns]);
            }
            (selected, scale)
        });
        let bank_output = mma_f32_abt_increasing_k(
            m,
            bank_columns,
            k,
            &a_values[a_start..a_start + a_bank_len],
            &b_values[b_start..b_start + b_bank_len],
            input_bank
                .as_ref()
                .map(|(values, scale)| (values.as_slice(), *scale)),
        )?;
        for row in 0..m {
            let source = row * bank_columns;
            let destination = row * n + bank * bank_columns;
            output[destination..destination + bank_columns]
                .copy_from_slice(&bank_output[source..source + bank_columns]);
        }
    }
    Ok(output)
}

/// How one dense TMEM destination cell carries an accumulator element.
///
/// Both occupy a whole 32-bit TMEM cell, so the physical footprint is the same
/// either way; they differ only in the payload codec. `F16` keeps the value in
/// the low half and writes the high half as zero.
///
/// The cell layout is measured, not assumed: on B200 a `kind::f8f6f4` MMA with
/// a `.f16` destination over a cell pre-poisoned with `0xBEEF3C00` leaves
/// `0x00004c40`, and reading the addend back for `enable_input_d` sees `1.0`
/// (the low half) rather than the whole word reinterpreted.
///
/// The *codec placement* -- one conversion on store rather than a narrower
/// accumulator -- is measured too, but note what that measurement does and does
/// not settle. It rules out rounding the accumulator to binary16 between K
/// steps, and it rules out truncation on the store (round-to-nearest is
/// observed). It does not distinguish an f32 reduction from any wider or
/// differently-associated reduction whose result agrees with f32 on the probe
/// inputs; the PTX ISA is silent on tcgen05 accumulation precision, so no
/// stronger claim is available from the specification either.
#[derive(Clone, Copy, Debug)]
enum RawMmaCellDtype {
    F32,
    F16,
}

impl RawMmaCellDtype {
    fn from_half(half: bool) -> Self {
        if half {
            Self::F16
        } else {
            Self::F32
        }
    }

    fn decode(self, bytes: [u8; 4]) -> f32 {
        match self {
            Self::F32 => f32::from_le_bytes(bytes),
            Self::F16 => fp16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])),
        }
    }

    fn encode(self, value: f32) -> [u8; 4] {
        match self {
            Self::F32 => value.to_le_bytes(),
            Self::F16 => {
                let [low, high] = f32_to_fp16_bits(value).to_le_bytes();
                [low, high, 0, 0]
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_read_dense_tmem<Scalar: Copy>(
    physical: &PhysicalMemory,
    view: &TmemView,
    destination_address: u32,
    m: usize,
    n: usize,
    layout: RawTcgenDenseTmemLayout,
    decode: impl Fn([u8; 4]) -> Scalar,
    disabled_value: Scalar,
    disable_output_lane: [u32; 4],
) -> Result<Vec<Scalar>, EngineError> {
    let (base_lane, base_col) = raw_tcgen05_address(destination_address, 0, 0)?;
    let capacity = m
        .checked_mul(n)
        .ok_or_else(|| EngineError::message("raw TCGEN destination shape overflow"))?;
    let mut values = Vec::with_capacity(capacity);
    for row in 0..m {
        for col in 0..n {
            let (lane_delta, column_delta) = layout.location(row, col, m, n)?;
            let lane = base_lane
                .checked_add(lane_delta)
                .ok_or_else(|| EngineError::message("raw TCGEN destination lane overflow"))?;
            if lane >= 128 {
                return Err(EngineError::message(format!(
                    "raw TCGEN destination lane {lane} is outside 128 lanes"
                )));
            }
            let disabled = ((disable_output_lane[lane / 32] >> (lane % 32)) & 1) != 0;
            let column = base_col
                .checked_add(column_delta)
                .ok_or_else(|| EngineError::message("raw TCGEN destination column overflow"))?;
            let validity = physical.tmem().cell_byte_validity(view, lane, column)?;
            if disabled && validity.iter().any(|valid| !valid) {
                values.push(disabled_value);
                continue;
            }
            let mut bytes = [0_u8; 4];
            physical
                .tmem()
                .read_cell_bytes_into(view, lane, column, 0, &mut bytes)?;
            values.push(decode(bytes));
        }
    }
    Ok(values)
}

fn raw_tcgen05_read_cta2_tmem(
    physical: &PhysicalMemory,
    context: &WarpContext,
    anchor: &RuntimeBuffer,
    destination_address: u32,
    m: usize,
    n: usize,
    layout: RawTcgenDenseTmemLayout,
    disable_output_lane: [u32; 8],
    cell_dtype: RawMmaCellDtype,
) -> Result<Vec<f32>, EngineError> {
    if m != 128 && m != 256 {
        return Err(EngineError::message(format!(
            "raw cta_group=2 destination requires M=128 or 256, got {m}"
        )));
    }
    let pair_base = context.cta_id_in_cluster() & !1_usize;
    let (base_lane, base_col) = raw_tcgen05_address(destination_address, 0, 0)?;
    let rows_per_cta = m / 2;
    let columns_per_bank = layout.physical_columns(n)?;
    let mut values = Vec::with_capacity(m.saturating_mul(n));
    if matches!(
        layout,
        RawTcgenDenseTmemLayout::D | RawTcgenDenseTmemLayout::E
    ) && matches!(cell_dtype, RawMmaCellDtype::F32)
        && disable_output_lane.iter().all(|&word| word == 0)
    {
        let lane_count = if m == 128 {
            rows_per_cta * 2
        } else {
            rows_per_cta
        };
        let lane_end = base_lane
            .checked_add(lane_count)
            .ok_or_else(|| EngineError::message("raw cta_group=2 destination lane overflow"))?;
        if lane_end > 128 {
            return Err(EngineError::message(format!(
                "raw cta_group=2 destination lanes [{base_lane}, {lane_end}) exceed 128 lanes"
            )));
        }
        let column_end = base_col
            .checked_add(columns_per_bank)
            .ok_or_else(|| EngineError::message("raw cta_group=2 destination column overflow"))?;
        values.resize(m.saturating_mul(n), 0.0);
        for target_offset in 0..2 {
            let view = raw_tcgen05_tmem_view(physical, context, anchor, pair_base + target_offset)?;
            physical
                .tmem()
                .validate_cell_rectangle(&view, lane_end, column_end)?;
            physical.tmem().with_read_session(&view, |session| {
                if !session.records_uninitialized_read_reviews() && !session.all_bytes_initialized()
                {
                    for lane in base_lane..lane_end {
                        session.validate_initialized_f32_cells_prevalidated(
                            lane,
                            base_col,
                            columns_per_bank,
                        )?;
                    }
                }
                let target_base = target_offset * rows_per_cta * n;
                for row in 0..rows_per_cta {
                    let target_start = target_base + row * n;
                    let target = &mut values[target_start..target_start + n];
                    if m == 128 {
                        let (first, second) = target.split_at_mut(columns_per_bank);
                        if session.records_uninitialized_read_reviews() {
                            session.read_reviewed_f32_cells_into_prevalidated(
                                base_lane + row,
                                base_col,
                                first,
                            );
                            session.read_reviewed_f32_cells_into_prevalidated(
                                base_lane + rows_per_cta + row,
                                base_col,
                                second,
                            );
                        } else {
                            session.read_initialized_f32_cells_into_prevalidated(
                                base_lane + row,
                                base_col,
                                first,
                            );
                            session.read_initialized_f32_cells_into_prevalidated(
                                base_lane + rows_per_cta + row,
                                base_col,
                                second,
                            );
                        }
                    } else if session.records_uninitialized_read_reviews() {
                        session.read_reviewed_f32_cells_into_prevalidated(
                            base_lane + row,
                            base_col,
                            target,
                        );
                    } else {
                        session.read_initialized_f32_cells_into_prevalidated(
                            base_lane + row,
                            base_col,
                            target,
                        );
                    }
                }
                Ok::<(), EngineError>(())
            })??;
        }
        return Ok(values);
    }
    for target_offset in 0..2 {
        let view = raw_tcgen05_tmem_view(physical, context, anchor, pair_base + target_offset)?;
        for row in 0..rows_per_cta {
            for col in 0..n {
                let (lane_delta, physical_col) = layout.location(row, col, rows_per_cta, n)?;
                let lane = base_lane.checked_add(lane_delta).ok_or_else(|| {
                    EngineError::message("raw cta_group=2 destination lane overflow")
                })?;
                if lane >= 128 {
                    return Err(EngineError::message(format!(
                        "raw cta_group=2 destination lane {lane} is outside 128 lanes"
                    )));
                }
                // Pair-word split is the documented inference; see
                // `raw_tcgen05_cta2_f32_tmem_footprints`.
                let disabled =
                    ((disable_output_lane[target_offset * 4 + lane / 32] >> (lane % 32)) & 1) != 0;
                let column = base_col.checked_add(physical_col).ok_or_else(|| {
                    EngineError::message("raw cta_group=2 destination column overflow")
                })?;
                let validity = physical.tmem().cell_byte_validity(&view, lane, column)?;
                if disabled && validity.iter().any(|valid| !valid) {
                    values.push(f32::NAN);
                    continue;
                }
                let mut bytes = [0_u8; 4];
                physical
                    .tmem()
                    .read_cell_bytes_into(&view, lane, column, 0, &mut bytes)?;
                values.push(cell_dtype.decode(bytes));
            }
        }
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_scatter_dense<Scalar: Copy>(
    physical: &PhysicalMemory,
    view: &TmemView,
    destination_address: u32,
    m: usize,
    n: usize,
    layout: RawTcgenDenseTmemLayout,
    encode: impl Fn(Scalar) -> [u8; 4],
    disable_output_lane: [u32; 4],
    output_values: &[Scalar],
) -> Result<(), EngineError> {
    let (base_lane, base_col) = raw_tcgen05_address(destination_address, 0, 0)?;
    for row in 0..m {
        for col in 0..n {
            let (lane_delta, column_delta) = layout.location(row, col, m, n)?;
            let lane = base_lane
                .checked_add(lane_delta)
                .ok_or_else(|| EngineError::message("raw TCGEN destination lane overflow"))?;
            if lane >= 128 {
                return Err(EngineError::message(format!(
                    "raw TCGEN destination lane {lane} is outside 128 lanes"
                )));
            }
            if ((disable_output_lane[lane / 32] >> (lane % 32)) & 1) != 0 {
                continue;
            }
            let column = base_col
                .checked_add(column_delta)
                .ok_or_else(|| EngineError::message("raw TCGEN destination column overflow"))?;
            let output_index = row
                .checked_mul(n)
                .and_then(|value| value.checked_add(col))
                .ok_or_else(|| EngineError::message("raw TCGEN output index overflow"))?;
            let value = output_values[output_index];
            physical
                .tmem()
                .write_cell_bytes(view, lane, column, 0, &encode(value))?;
        }
    }
    Ok(())
}

/// Where a raw `tcgen05.mma` mnemonic finds its addend and puts its product.
///
/// Read and write are one row because they must name the same destination
/// window, and every mnemonic reads back exactly what it wrote.
/// A contiguous TMEM window reached through one CTA's view, optionally in the
/// `M=64` interleaved lane layout and optionally masking whole output lanes.
///
/// `cell_dtype` is the element codec of the window, not of the numeric core:
/// the core always accumulates in binary32 and the codec applies once, when
/// the addend is read and when the product is stored.
struct RawMmaWindow<'a> {
    physical: &'a PhysicalMemory,
    view: &'a TmemView,
    destination_address: u32,
    layout: RawTcgenDenseTmemLayout,
    cell_dtype: RawMmaCellDtype,
    disable_output_lane: [u32; 4],
}

/// The closed set of destinations the raw `tcgen05.mma` mnemonics write.
///
/// `Dense` and `LaneCells` name the same window and read it identically; they
/// differ only in how the product is stored, so the difference lives in a
/// variant rather than being folded into a shared write path.
enum RawMmaDestination<'a> {
    /// Cell-by-cell stores over a single CTA's window.
    Dense(RawMmaWindow<'a>),
    /// Block-scaled `mxf4`: one whole-lane store per row.
    LaneCells(RawMmaWindow<'a>),
    /// `cta_group=2` spans the CTA pair, so it reaches TMEM through the anchor
    /// rather than one CTA's view. The instruction selects each CTA's layout;
    /// sparse M=128 uses interleaved rows, not the dense N-selected banks.
    Cta2 {
        physical: &'a PhysicalMemory,
        context: &'a WarpContext,
        anchor: &'a RuntimeBuffer,
        destination_address: u32,
        layout: RawTcgenDenseTmemLayout,
        disable_output_lane: [u32; 8],
        cell_dtype: RawMmaCellDtype,
    },
}

impl RawMmaDestination<'_> {
    /// Reads the `m`-by-`n` destination window.
    fn read(&self, m: usize, n: usize) -> Result<Vec<f32>, EngineError> {
        let values = match self {
            Self::Dense(window) | Self::LaneCells(window) => raw_tcgen05_read_dense_tmem(
                window.physical,
                window.view,
                window.destination_address,
                m,
                n,
                window.layout,
                |bytes| window.cell_dtype.decode(bytes),
                f32::NAN,
                window.disable_output_lane,
            ),
            Self::Cta2 {
                physical,
                context,
                anchor,
                destination_address,
                layout,
                disable_output_lane,
                cell_dtype,
            } => raw_tcgen05_read_cta2_tmem(
                physical,
                context,
                anchor,
                *destination_address,
                m,
                n,
                *layout,
                *disable_output_lane,
                *cell_dtype,
            ),
        }?;
        Ok(values)
    }

    /// Writes the `m`-by-`n` product back over that same window.
    fn scatter(&self, m: usize, n: usize, output_values: &[f32]) -> Result<(), EngineError> {
        match self {
            Self::Dense(window) => raw_tcgen05_scatter_dense(
                window.physical,
                window.view,
                window.destination_address,
                m,
                n,
                window.layout,
                |value| window.cell_dtype.encode(value),
                window.disable_output_lane,
                output_values,
            ),
            Self::LaneCells(window) => window.scatter_lane_cells(m, n, output_values),
            Self::Cta2 {
                physical,
                context,
                anchor,
                destination_address,
                layout,
                disable_output_lane,
                cell_dtype,
            } => raw_tcgen05_scatter_cta2(
                physical,
                context,
                anchor,
                *destination_address,
                m,
                n,
                *layout,
                *disable_output_lane,
                *cell_dtype,
                output_values,
            ),
        }
    }
}

impl RawMmaWindow<'_> {
    /// Store a whole TMEM lane per row instead of going cell by cell.
    fn scatter_lane_cells(
        &self,
        m: usize,
        n: usize,
        output_values: &[f32],
    ) -> Result<(), EngineError> {
        // `cell_dtype` is a `RawMmaWindow` field, so `LaneCells` can name F16
        // even though no block-scaled mnemonic has a half-precision
        // destination: `kind::mxf4`/`mxf4nvf4`/`mxf8f6f4` all require D=f32,
        // enforced by `_block_kind` in the transpiler. Assert it here so the
        // combination cannot reach memory if a future row forgets.
        debug_assert!(
            matches!(self.cell_dtype, RawMmaCellDtype::F32),
            "block-scaled destinations are float32 only",
        );
        let (base_lane, base_col) = raw_tcgen05_address(self.destination_address, 0, 0)?;
        let end_lane = base_lane
            .checked_add(m)
            .ok_or_else(|| EngineError::message("raw mxf4 destination lane overflow"))?;
        if end_lane > 128 {
            return Err(EngineError::message(format!(
                "raw mxf4 destination lanes [{base_lane}, {end_lane}) exceed 128 TMEM lanes"
            )));
        }
        let mut row_cells = vec![[0_u8; 4]; n];
        for row in 0..m {
            let lane = base_lane + row;
            for (col, cell) in row_cells.iter_mut().enumerate() {
                let output_index = row
                    .checked_mul(n)
                    .and_then(|value| value.checked_add(col))
                    .ok_or_else(|| EngineError::message("raw mxf4 output index overflow"))?;
                *cell = self.cell_dtype.encode(output_values[output_index]);
            }
            self.physical
                .tmem()
                .write_lane_cells(self.view, lane, base_col, &row_cells)?;
        }
        Ok(())
    }
}

/// One row of the raw `tcgen05.mma` table.
///
/// Every mnemonic ends the same way — read the addend `enable_input_d`
/// selects, resolve the input scale, run the one numeric core, scatter, and
/// write the result. What differs is *how* those
/// touch memory, so `destination` and `scale` are per-mnemonic rows rather
/// than flags: the driver never averages two mnemonics' behaviour, it only
/// sequences whichever rows it was handed.
///
/// `scale` runs where the hand-written entries computed it — immediately
/// after the accumulator snapshot — so mnemonics that range-check
/// `scale-input-d` late keep surfacing their errors in the original order.
struct RawMmaTail<'a, Scale> {
    m: usize,
    n: usize,
    k: usize,
    enable_input_d: bool,
    destination: RawMmaDestination<'a>,
    scale: Scale,
}

impl<Scale> RawMmaTail<'_, Scale>
where
    Scale: FnOnce() -> Result<f32, EngineError>,
{
    fn run(self, a_values: &[f32], b_values: &[f32]) -> Result<(), EngineError> {
        self.run_with(a_values, b_values, mma_f32_abt_increasing_k)
    }

    fn run_with<Numeric>(
        self,
        a_values: &[f32],
        b_values: &[f32],
        numeric: Numeric,
    ) -> Result<(), EngineError>
    where
        Numeric: FnOnce(
            usize,
            usize,
            usize,
            &[f32],
            &[f32],
            Option<(&[f32], f32)>,
        ) -> Result<Vec<f32>, EngineError>,
    {
        let Self {
            m,
            n,
            k,
            enable_input_d,
            destination,
            scale,
        } = self;
        let input_d = if enable_input_d {
            Some(destination.read(m, n)?)
        } else {
            None
        };
        let input_scale = scale()?;
        let output_values = numeric(
            m,
            n,
            k,
            a_values,
            b_values,
            input_d
                .as_ref()
                .map(|values| (values.as_slice(), input_scale)),
        )?;
        destination.scatter(m, n, &output_values)
    }
}
#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_gather_b16_shared_matrix_cta2(
    physical: &PhysicalMemory,
    context: &WarpContext,
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    rows_per_cta: usize,
    columns: usize,
    bf16: bool,
    negate: bool,
    transpose: bool,
) -> Result<Vec<f32>, EngineError> {
    raw_tcgen05_gather_b16_shared_cta2_with(
        physical,
        context,
        source,
        descriptor,
        rows_per_cta,
        columns,
        transpose,
        |bits| {
            let value = if bf16 {
                bf16_bits_to_f32(bits)
            } else {
                fp16_bits_to_f32(bits)
            };
            Ok(if negate { -value } else { value })
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_gather_b16_shared_cta2_with<Scalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    rows_per_cta: usize,
    columns: usize,
    transpose: bool,
    decode: impl Fn(u16) -> Result<Scalar, EngineError>,
) -> Result<Vec<Scalar>, EngineError> {
    let pair_base = context.cta_id_in_cluster() & !1_usize;
    if pair_base + 1 >= context.topology().ctas_per_cluster() {
        return Err(EngineError::message(
            "raw f16 cta_group=2 shared gather has no paired CTA",
        ));
    }
    let capacity = rows_per_cta
        .checked_mul(columns)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| EngineError::message("raw f16 cta2 shared gather shape overflow"))?;
    let mut values = Vec::with_capacity(capacity);
    for target_cta in [pair_base, pair_base + 1] {
        let view = raw_tcgen05_shared_view_at_cta(physical, context, source, target_cta)?;
        for row in 0..rows_per_cta {
            for column in 0..columns {
                let offset =
                    raw_tcgen05_b16_matrix_byte_offset(source, descriptor, row, column, transpose)?;
                let mut bytes = [0_u8; 2];
                physical
                    .shared()
                    .read_bytes_into(&view, offset, &mut bytes)?;
                values.push(decode(u16::from_le_bytes(bytes))?);
            }
        }
    }
    Ok(values)
}

fn raw_tcgen05_gather_packed_tmem_a_cta2_with<Scalar: Copy, const ELEMENTS: usize>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    anchor: &RuntimeBuffer,
    address: u32,
    m: usize,
    layout: RawTcgenDenseTmemLayout,
    columns: usize,
    decode: impl Fn(u32) -> Result<[Scalar; ELEMENTS], EngineError>,
) -> Result<Vec<Scalar>, EngineError> {
    let pair_base = context.cta_id_in_cluster() & !1;
    if !matches!(m, 128 | 256) || pair_base + 1 >= context.topology().ctas_per_cluster() {
        return Err(EngineError::message(
            "invalid CTA2 packed A shape or missing paired CTA",
        ));
    }
    let mut local = Vec::with_capacity(2);
    for cta in 0..2 {
        let view = raw_tcgen05_tmem_view(physical, context, anchor, pair_base + cta)?;
        local.push(raw_tcgen05_gather_packed_tmem_a_columns(
            physical,
            &view,
            address,
            m / 2,
            layout,
            columns,
            &decode,
        )?);
    }
    let bank_size = m / 2 * columns * ELEMENTS;
    let mut values = Vec::with_capacity(m * columns * ELEMENTS * layout.packed_a_banks());
    // Preserve N-selected A banks across the CTA pair (rather than merging them).
    for bank in 0..layout.packed_a_banks() {
        for cta in &local {
            values.extend_from_slice(&cta[bank * bank_size..(bank + 1) * bank_size]);
        }
    }
    Ok(values)
}

fn validate_raw_tcgen05_cta2_destination(
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    context: &WarpContext,
    destination_address: u32,
    m: usize,
    n: usize,
) -> Result<(), EngineError> {
    let pair_base = context.cta_id_in_cluster() & !1_usize;
    if pair_base + 1 >= context.topology().ctas_per_cluster() {
        return Err(EngineError::message(
            "raw f16 cta_group=2 destination has no paired CTA",
        ));
    }
    let (_, destination_column) = raw_tcgen05_address(destination_address, 0, 0)?;
    let physical_columns = raw_tcgen05_cta2_columns_per_bank(m, n);
    for target_cta in [pair_base, pair_base + 1] {
        validate_raw_tcgen05_tmem_columns(
            lifecycle,
            access_mode,
            context,
            target_cta,
            destination_column,
            physical_columns,
        )?;
    }
    Ok(())
}

/// NVIDIA's tcgen05 programming guide requires K-major A when sourced from
/// TMEM. The SM100 UMMA wrappers enforce this for both CTA groups and formats.
/// Shared-A transpose remains governed by its format-specific layout decoder.
pub(crate) fn raw_tcgen05_validate_tmem_a_transpose(transpose: bool) -> Result<(), EngineError> {
    if transpose {
        return Err(EngineError::message(
            "tcgen05.mma TMEM A must be K-major; transpose_a is invalid",
        ));
    }
    Ok(())
}

pub(crate) fn raw_tcgen05_f8_tmem_a_address(
    bits: u64,
    transpose: bool,
) -> Result<u32, EngineError> {
    raw_tcgen05_validate_tmem_a_transpose(transpose)?;
    u32::try_from(bits).map_err(|_| EngineError::message("raw f8f6f4 TMEM A address exceeds u32"))
}

#[allow(clippy::too_many_arguments)]
pub fn raw_tcgen05_mma_f8f6f4_cta1(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    anchor: &RuntimeBuffer,
    shared_candidates: &[&RuntimeBuffer],
    destination_address: u32,
    a_descriptor_bits: u64,
    b_descriptor_bits: u64,
    instruction_descriptor: u32,
    enable_input_d: bool,
    disable_output_lane: [u32; 4],
    issuing_lane: usize,
    predicate: u32,
    a_format: RawTcgenNarrowFormat,
    b_format: RawTcgenNarrowFormat,
    d_f16: bool,
    a_in_tmem: bool,
    ws_mask: Option<u64>,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
    lut_b: Option<u32>,
) -> Result<(), EngineError> {
    if predicate == 0 {
        return Ok(());
    }
    let instruction = decode_raw_tcgen_f8f6f4_instruction_descriptor(
        instruction_descriptor,
        a_format,
        b_format,
        d_f16,
        1,
        descriptor_layout.supports_f8f6f4_k64(),
        ws_mask.is_some(),
        false,
    )?;
    validate_raw_tcgen05_issuing_lane(
        context,
        issuing_lane,
        &crate::DiagnosticLabel::new("raw f8f6f4 tcgen05.mma"),
    )?;
    raw_tcgen05_validate_lut_b(lut_b, instruction.k, b_format, instruction.transpose_b)?;
    let b_descriptor = raw_tcgen05_f8_b_descriptor(b_descriptor_bits, descriptor_layout, lut_b)?;
    let b_source = raw_tcgen05_shared_source(
        context,
        shared_candidates,
        b_descriptor.start_address,
        issuing_lane,
    )?;
    let layout = raw_tcgen05_cta1_dense_tmem_layout(instruction.m, ws_mask.is_some())?;
    validate_raw_tcgen05_address_columns(
        lifecycle,
        access_mode,
        context,
        destination_address,
        layout.physical_columns(instruction.n)?,
    )?;
    let view = raw_tcgen05_tmem_view(physical, context, anchor, context.cta_id_in_cluster())?;
    let a_values = if a_in_tmem {
        let address = raw_tcgen05_f8_tmem_a_address(a_descriptor_bits, instruction.transpose_a)?;
        validate_raw_tcgen05_address_columns(
            lifecycle,
            access_mode,
            context,
            address,
            instruction.k / 4,
        )?;
        raw_tcgen05_gather_packed_tmem_a_columns(
            physical,
            &view,
            address,
            instruction.m,
            layout,
            instruction.k / 4,
            |word| {
                Ok(a_format
                    .decode_tmem_word(word)?
                    .map(|v| if instruction.negate_a { -v } else { v }))
            },
        )?
    } else {
        let a_descriptor =
            decode_raw_tcgen_matrix_descriptor_for_layout(a_descriptor_bits, descriptor_layout)?;
        let a_source = raw_tcgen05_shared_source(
            context,
            shared_candidates,
            a_descriptor.start_address,
            issuing_lane,
        )?;
        raw_tcgen05_gather_f8_shared_matrix(
            physical,
            context,
            &a_source,
            a_descriptor,
            instruction.m,
            instruction.k,
            a_format,
            instruction.negate_a,
            instruction.transpose_a,
            1,
            None,
            false,
        )?
    };
    let column_mask = ws_mask
        .map(|bits| {
            RawTcgenColumnMask::new(bits, instruction.m, instruction.n, instruction_descriptor)
        })
        .transpose()?;
    let b_values = if let Some(lookup_address) = lut_b {
        raw_tcgen05_gather_lut_b(
            physical,
            context,
            lifecycle,
            access_mode,
            anchor,
            &b_source,
            b_descriptor,
            ((b_descriptor_bits >> 53) & 1) as usize,
            lookup_address,
            instruction.n,
            1,
            instruction.negate_b,
        )?
    } else {
        raw_tcgen05_gather_f8_shared_matrix(
            physical,
            context,
            &b_source,
            b_descriptor,
            instruction.n,
            instruction.k,
            b_format,
            instruction.negate_b,
            instruction.transpose_b,
            1,
            column_mask,
            false,
        )?
    };
    // Measured on B200: a `.f16` destination does not round the accumulator
    // between K steps, and its store rounds to nearest rather than truncating.
    // The shared f32 core plus one conversion on store reproduces the hardware
    // bytes on those probes, so only the destination codec changes here. See
    // `RawMmaCellDtype` for the exact scope of that measurement.
    let cell_dtype = if d_f16 {
        RawMmaCellDtype::F16
    } else {
        RawMmaCellDtype::F32
    };
    let tail = RawMmaTail {
        m: instruction.m,
        n: instruction.n,
        k: instruction.k,
        enable_input_d,
        destination: RawMmaDestination::Dense(RawMmaWindow {
            physical,
            view: &view,
            destination_address,
            layout,
            cell_dtype,
            disable_output_lane,
        }),
        scale: || Ok(1.0_f32),
    };
    if a_in_tmem && layout.packed_a_banks() > 1 {
        tail.run_with(&a_values, &b_values, |m, n, k, a, b, d| {
            mma_f32_abt_banked_a_increasing_k(m, n, k, a, b, d, layout.packed_a_banks())
        })
    } else {
        tail.run(&a_values, &b_values)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn raw_tcgen05_mma_f8f6f4_cta2(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    anchor: &RuntimeBuffer,
    shared_candidates: &[&RuntimeBuffer],
    destination_address: u32,
    a_descriptor_bits: u64,
    b_descriptor_bits: u64,
    instruction_descriptor: u32,
    enable_input_d: bool,
    disable_output_lane: [u32; 8],
    issuing_lane: usize,
    predicate: u32,
    a_format: RawTcgenNarrowFormat,
    b_format: RawTcgenNarrowFormat,
    d_f16: bool,
    a_in_tmem: bool,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
    lut_b: Option<u32>,
) -> Result<(), EngineError> {
    if predicate == 0 {
        return Ok(());
    }
    let instruction = decode_raw_tcgen_f8f6f4_instruction_descriptor(
        instruction_descriptor,
        a_format,
        b_format,
        d_f16,
        2,
        descriptor_layout.supports_f8f6f4_k64(),
        false,
        false,
    )?;
    raw_tcgen05_validate_lut_b(lut_b, instruction.k, b_format, instruction.transpose_b)?;
    let b_descriptor = raw_tcgen05_f8_b_descriptor(b_descriptor_bits, descriptor_layout, lut_b)?;
    validate_raw_tcgen05_issuing_lane(
        context,
        issuing_lane,
        &crate::DiagnosticLabel::new("raw f8f6f4 cta2 tcgen05.mma"),
    )?;
    let b_source = raw_tcgen05_shared_source(
        context,
        shared_candidates,
        b_descriptor.start_address,
        issuing_lane,
    )?;
    validate_raw_tcgen05_cta2_destination(
        lifecycle,
        access_mode,
        context,
        destination_address,
        instruction.m,
        instruction.n,
    )?;
    let layout = raw_tcgen05_cta1_dense_tmem_layout(instruction.m / 2, true)?;
    let a_values = if a_in_tmem {
        let address = raw_tcgen05_f8_tmem_a_address(a_descriptor_bits, instruction.transpose_a)?;
        let (_, column) = raw_tcgen05_address(address, 0, 0)?;
        let first = context.cta_id_in_cluster() & !1;
        for cta in first..first + 2 {
            validate_raw_tcgen05_tmem_columns(
                lifecycle,
                access_mode,
                context,
                cta,
                column,
                instruction.k / 4,
            )?;
        }
        raw_tcgen05_gather_packed_tmem_a_cta2_with(
            physical,
            context,
            anchor,
            address,
            instruction.m,
            layout,
            instruction.k / 4,
            |word| {
                Ok(a_format
                    .decode_tmem_word(word)?
                    .map(|v| if instruction.negate_a { -v } else { v }))
            },
        )?
    } else {
        let a_descriptor =
            decode_raw_tcgen_matrix_descriptor_for_layout(a_descriptor_bits, descriptor_layout)?;
        let a_source = raw_tcgen05_shared_source(
            context,
            shared_candidates,
            a_descriptor.start_address,
            issuing_lane,
        )?;
        raw_tcgen05_gather_f8_shared_matrix(
            physical,
            context,
            &a_source,
            a_descriptor,
            instruction.m / 2,
            instruction.k,
            a_format,
            instruction.negate_a,
            instruction.transpose_a,
            2,
            None,
            false,
        )?
    };
    let b_values = if let Some(lookup_address) = lut_b {
        raw_tcgen05_gather_lut_b(
            physical,
            context,
            lifecycle,
            access_mode,
            anchor,
            &b_source,
            b_descriptor,
            ((b_descriptor_bits >> 53) & 1) as usize,
            lookup_address,
            instruction.n / 2,
            2,
            instruction.negate_b,
        )?
    } else {
        raw_tcgen05_gather_f8_shared_matrix(
            physical,
            context,
            &b_source,
            b_descriptor,
            instruction.n / 2,
            instruction.k,
            b_format,
            instruction.negate_b,
            instruction.transpose_b,
            2,
            None,
            false,
        )?
    };
    let tail = RawMmaTail {
        m: instruction.m,
        n: instruction.n,
        k: instruction.k,
        enable_input_d,
        destination: RawMmaDestination::Cta2 {
            physical,
            context,
            anchor,
            destination_address,
            layout: raw_tcgen05_cta1_dense_tmem_layout(instruction.m / 2, true)?,
            disable_output_lane,
            cell_dtype: RawMmaCellDtype::from_half(d_f16),
        },
        scale: || Ok(1.0_f32),
    };
    if a_in_tmem && layout.packed_a_banks() > 1 {
        tail.run_with(&a_values, &b_values, |m, n, k, a, b, d| {
            mma_f32_abt_banked_a_increasing_k(m, n, k, a, b, d, layout.packed_a_banks())
        })
    } else {
        tail.run(&a_values, &b_values)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum RawTcgenFloatKind {
    Tf32,
    B16 {
        a_bf16: bool,
        b_bf16: bool,
    },
    SparseNarrow {
        a_format: RawTcgenNarrowFormat,
        b_format: RawTcgenNarrowFormat,
        descriptor_layout: RawTcgenMatrixDescriptorLayout,
    },
}
impl RawTcgenFloatKind {
    pub(crate) fn metadata_layout(self, descriptor: u32) -> RawTcgenSparseMetadataLayout {
        match self {
            Self::SparseNarrow { .. } => RawTcgenSparseMetadataLayout::Narrow {
                k: self.packed_k(descriptor) * 2,
            },
            _ => RawTcgenSparseMetadataLayout::B16 {
                selector: (descriptor & 1) as usize,
            },
        }
    }

    fn descriptor_layout(self) -> RawTcgenMatrixDescriptorLayout {
        match self {
            Self::SparseNarrow {
                descriptor_layout, ..
            } => descriptor_layout,
            _ => RawTcgenMatrixDescriptorLayout::Sm100,
        }
    }

    fn instruction(
        self,
        instruction_descriptor: u32,
        cta_group: usize,
        weight_stationary: bool,
        sparse: bool,
    ) -> Result<RawTcgenDenseInstructionDescriptor, EngineError> {
        match self {
            Self::Tf32 => decode_raw_tcgen_tf32_instruction_descriptor(
                instruction_descriptor,
                cta_group,
                weight_stationary,
                sparse,
            ),
            Self::B16 { a_bf16, b_bf16 } => decode_raw_tcgen_b16_instruction_descriptor(
                instruction_descriptor,
                a_bf16,
                b_bf16,
                cta_group as u32,
                weight_stationary,
                sparse,
            ),
            Self::SparseNarrow {
                a_format,
                b_format,
                descriptor_layout,
            } => {
                if !sparse || instruction_descriptor & 15 != 4 {
                    return Err(EngineError::message(
                        "sparse F8F6F4 requires sparsity and selector zero",
                    ));
                }
                let half = instruction_descriptor & (1 << 4) == 0;
                let value = decode_raw_tcgen_f8f6f4_instruction_descriptor(
                    instruction_descriptor,
                    a_format,
                    b_format,
                    half,
                    cta_group as u32,
                    descriptor_layout.supports_f8f6f4_k64(),
                    weight_stationary,
                    true,
                )?;
                Ok(RawTcgenDenseInstructionDescriptor {
                    cell_dtype: RawMmaCellDtype::from_half(half),
                    m: value.m,
                    n: value.n,
                    negate_a: value.negate_a,
                    negate_b: value.negate_b,
                    transpose_a: value.transpose_a,
                    transpose_b: value.transpose_b,
                })
            }
        }
    }
    pub(crate) fn packed_k(self, descriptor: u32) -> usize {
        match self {
            Self::Tf32 => 8,
            Self::B16 { .. } => 16,
            Self::SparseNarrow { .. } => {
                if descriptor & (1 << 29) == 0 {
                    32
                } else {
                    64
                }
            }
        }
    }

    pub(crate) fn tmem_a_columns(self, descriptor: u32) -> usize {
        match self {
            Self::SparseNarrow { .. } => self.packed_k(descriptor) / 4,
            _ => 8,
        }
    }

    fn sparse_index(self, code: u8, inner: usize) -> Result<usize, EngineError> {
        match self {
            Self::Tf32 => match code {
                0x4 => Ok(inner * 2),
                0xe => Ok(inner * 2 + 1),
                _ => Err(EngineError::message(format!(
                    "sparse 1:2 TF32 metadata code 0x{code:x} is not a defined index"
                ))),
            },
            Self::B16 { .. } | Self::SparseNarrow { .. } => {
                Ok((inner / 2) * 4 + raw_tcgen05_sparse_2of4_indices(code)?[inner % 2])
            }
        }
    }
}

pub(crate) fn raw_tcgen05_float_shape(
    kind: RawTcgenFloatKind,
    descriptor: u32,
    cta_group: usize,
    ws: bool,
    sparse: bool,
) -> Result<(usize, usize, bool, bool), EngineError> {
    let value = kind.instruction(descriptor, cta_group, ws, sparse)?;
    Ok((value.m, value.n, value.transpose_a, value.transpose_b))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_tcgen05_float_shared_footprints(
    kind: RawTcgenFloatKind,
    context: &WarpContext,
    candidates: &[&RuntimeBuffer],
    bits: u64,
    rows: usize,
    columns: usize,
    transpose: bool,
    lane: usize,
    cta_group: usize,
    mask: Option<RawTcgenColumnMask>,
    is_b: bool,
) -> Result<(RuntimeBuffer, Vec<(usize, Option<usize>, usize, usize)>), EngineError> {
    match kind {
        RawTcgenFloatKind::Tf32 => raw_tcgen05_tf32_shared_footprints(
            context, candidates, bits, rows, columns, transpose, lane, cta_group, mask,
        ),
        RawTcgenFloatKind::B16 { .. } => {
            if cta_group == 2 {
                raw_tcgen05_b16_shared_footprints_cta2(
                    context, candidates, bits, rows, columns, transpose, lane,
                )
            } else {
                raw_tcgen05_b16_shared_masked_footprints(
                    context, candidates, bits, rows, columns, transpose, lane, mask,
                )
            }
        }
        RawTcgenFloatKind::SparseNarrow {
            a_format,
            b_format,
            descriptor_layout,
        } => {
            let desc = decode_raw_tcgen_matrix_descriptor_for_layout(bits, descriptor_layout)?;
            let source = raw_tcgen05_shared_source(context, candidates, desc.start_address, lane)?;
            let first = context.cta_id_in_cluster() & !(cta_group - 1);
            if first + cta_group > context.topology().ctas_per_cluster() {
                return Err(EngineError::message("sparse narrow MMA has no paired CTA"));
            }
            let mut accesses = Vec::new();
            for cta in first..first + cta_group {
                accesses.extend(raw_tcgen05_f8_shared_matrix_accesses(
                    &source,
                    desc,
                    rows,
                    columns,
                    if is_b { b_format } else { a_format },
                    transpose,
                    lane,
                    cta,
                    mask,
                    false,
                    true,
                )?);
            }
            Ok((source, accesses))
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn raw_tcgen05_mma_float(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    anchor: &RuntimeBuffer,
    shared_candidates: &[&RuntimeBuffer],
    destination_address: u32,
    a_operand: RawTcgenMmaA,
    b_descriptor_bits: u64,
    instruction_descriptor: u32,
    enable_input_d: bool,
    scale_input_d: usize,
    disable_output_lane: &[u32],
    issuing_lane: usize,
    cta_group: usize,
    ws_mask: Option<u64>,
    metadata: Option<u32>,
    kind: RawTcgenFloatKind,
) -> Result<(), EngineError> {
    let instruction = kind.instruction(
        instruction_descriptor,
        cta_group,
        ws_mask.is_some(),
        metadata.is_some(),
    )?;
    let packed_k = kind.packed_k(instruction_descriptor);
    let gather_shared =
        |source: &RuntimeBuffer, descriptor, rows, columns, transpose, negate, mask, is_b| {
            match kind {
                RawTcgenFloatKind::SparseNarrow {
                    a_format, b_format, ..
                } => raw_tcgen05_gather_f8_shared_matrix(
                    physical,
                    context,
                    source,
                    descriptor,
                    rows,
                    columns,
                    if is_b { b_format } else { a_format },
                    negate,
                    transpose,
                    cta_group,
                    mask,
                    true,
                ),
                RawTcgenFloatKind::Tf32 => raw_tcgen05_gather_tf32_shared_matrix(
                    physical, context, source, descriptor, rows, columns, transpose, negate,
                    cta_group, mask,
                ),
                RawTcgenFloatKind::B16 { a_bf16, b_bf16 } => {
                    let bf16 = if is_b { b_bf16 } else { a_bf16 };
                    if cta_group == 1 {
                        raw_tcgen05_gather_b16_shared_matrix(
                            physical, context, source, descriptor, rows, columns, bf16, negate,
                            transpose, mask,
                        )
                    } else {
                        raw_tcgen05_gather_b16_shared_matrix_cta2(
                            physical, context, source, descriptor, rows, columns, bf16, negate,
                            transpose,
                        )
                    }
                }
            }
        };
    if disable_output_lane.len() != 4 * cta_group {
        return Err(EngineError::message(
            "floating MMA output mask size does not match CTA group",
        ));
    }
    validate_raw_tcgen05_issuing_lane(
        context,
        issuing_lane,
        &crate::DiagnosticLabel::new("raw floating MMA"),
    )?;
    let b_descriptor =
        decode_raw_tcgen_matrix_descriptor_for_layout(b_descriptor_bits, kind.descriptor_layout())?;
    let b_source = raw_tcgen05_shared_source(
        context,
        shared_candidates,
        b_descriptor.start_address,
        issuing_lane,
    )?;
    let layout = raw_tcgen05_cta1_dense_tmem_layout(
        instruction.m / cta_group,
        ws_mask.is_some() || (cta_group == 2 && metadata.is_none()),
    )?;
    let first_cta = context.cta_id_in_cluster() & !(cta_group - 1);
    let (_, destination_column) = raw_tcgen05_address(destination_address, 0, 0)?;
    for cta in first_cta..first_cta + cta_group {
        validate_raw_tcgen05_tmem_columns(
            lifecycle,
            access_mode,
            context,
            cta,
            destination_column,
            layout.physical_columns(instruction.n)?,
        )?;
    }
    let view = raw_tcgen05_tmem_view(physical, context, anchor, context.cta_id_in_cluster())?;
    let a_values = match a_operand {
        RawTcgenMmaA::Shared(bits) => {
            let descriptor =
                decode_raw_tcgen_matrix_descriptor_for_layout(bits, kind.descriptor_layout())?;
            let source = raw_tcgen05_shared_source(
                context,
                shared_candidates,
                descriptor.start_address,
                issuing_lane,
            )?;
            gather_shared(
                &source,
                descriptor,
                instruction.m / cta_group,
                packed_k,
                instruction.transpose_a,
                instruction.negate_a,
                None,
                false,
            )?
        }
        RawTcgenMmaA::Tmem(address) => {
            raw_tcgen05_validate_tmem_a_transpose(instruction.transpose_a)?;
            let first = context.cta_id_in_cluster() & !(cta_group - 1);
            let (_, column) = raw_tcgen05_address(address, 0, 0)?;
            let columns = kind.tmem_a_columns(instruction_descriptor);
            for cta in first..first + cta_group {
                validate_raw_tcgen05_tmem_columns(
                    lifecycle,
                    access_mode,
                    context,
                    cta,
                    column,
                    columns,
                )?;
            }
            match kind {
                RawTcgenFloatKind::SparseNarrow { a_format, .. } => {
                    let decode = |word| {
                        Ok(a_format.decode_tmem_word(word)?.map(|value| {
                            if instruction.negate_a {
                                -value
                            } else {
                                value
                            }
                        }))
                    };
                    if cta_group == 1 {
                        raw_tcgen05_gather_packed_tmem_a_columns(
                            physical,
                            &view,
                            address,
                            instruction.m,
                            layout,
                            columns,
                            decode,
                        )?
                    } else {
                        raw_tcgen05_gather_packed_tmem_a_cta2_with(
                            physical,
                            context,
                            anchor,
                            address,
                            instruction.m,
                            layout,
                            columns,
                            decode,
                        )?
                    }
                }
                RawTcgenFloatKind::Tf32 => {
                    let decode = |word| {
                        let value = raw_tcgen05_tf32_payload_to_f32(word);
                        Ok([if instruction.negate_a { -value } else { value }])
                    };
                    if cta_group == 1 {
                        raw_tcgen05_gather_packed_tmem_a_with(
                            physical,
                            &view,
                            address,
                            instruction.m,
                            layout,
                            decode,
                        )?
                    } else {
                        raw_tcgen05_gather_packed_tmem_a_cta2_with(
                            physical,
                            context,
                            anchor,
                            address,
                            instruction.m,
                            layout,
                            8,
                            decode,
                        )?
                    }
                }
                RawTcgenFloatKind::B16 { a_bf16, .. } => {
                    if cta_group == 1 {
                        raw_tcgen05_gather_b16_tmem_a(
                            physical,
                            &view,
                            address,
                            instruction.m,
                            layout,
                            a_bf16,
                            instruction.negate_a,
                        )?
                    } else {
                        raw_tcgen05_gather_packed_tmem_a_cta2_with(
                            physical,
                            context,
                            anchor,
                            address,
                            instruction.m,
                            layout,
                            8,
                            |word| {
                                Ok([word as u16, (word >> 16) as u16].map(|bits| {
                                    let value = if a_bf16 {
                                        bf16_bits_to_f32(bits)
                                    } else {
                                        fp16_bits_to_f32(bits)
                                    };
                                    if instruction.negate_a {
                                        -value
                                    } else {
                                        value
                                    }
                                }))
                            },
                        )?
                    }
                }
            }
        }
    };
    let k = packed_k * if metadata.is_some() { 2 } else { 1 };
    if let Some(metadata) = metadata {
        raw_tcgen05_validate_sparse_metadata(
            lifecycle,
            access_mode,
            context,
            metadata,
            destination_address,
            cta_group,
            kind.metadata_layout(instruction_descriptor),
        )?;
    }
    let column_mask = ws_mask
        .map(|bits| {
            RawTcgenColumnMask::new(bits, instruction.m, instruction.n, instruction_descriptor)
        })
        .transpose()?;
    let b_values = gather_shared(
        &b_source,
        b_descriptor,
        instruction.n / cta_group,
        k,
        instruction.transpose_b,
        instruction.negate_b,
        column_mask,
        true,
    )?;
    let destination = if cta_group == 1 {
        RawMmaDestination::Dense(RawMmaWindow {
            physical,
            view: &view,
            destination_address,
            layout,
            cell_dtype: instruction.cell_dtype,
            disable_output_lane: std::array::from_fn(|i| disable_output_lane[i]),
        })
    } else {
        RawMmaDestination::Cta2 {
            physical,
            context,
            anchor,
            destination_address,
            layout,
            cell_dtype: instruction.cell_dtype,
            disable_output_lane: std::array::from_fn(|i| disable_output_lane[i]),
        }
    };
    let tail = RawMmaTail {
        m: instruction.m,
        n: instruction.n,
        k,
        enable_input_d,
        destination,
        scale: || {
            if scale_input_d > 15 {
                return Err(EngineError::message(
                    "raw TF32 scale-input-d is outside 0..=15",
                ));
            }
            raw_tcgen05_input_scale(scale_input_d, "raw TF32 input scale conversion failed")
        },
    };
    if let Some(metadata) = metadata {
        return raw_tcgen05_sparse_float_tail(
            tail,
            &a_values,
            &b_values,
            physical,
            context,
            anchor,
            metadata,
            kind,
            kind.metadata_layout(instruction_descriptor),
            layout,
            cta_group,
        );
    }
    if matches!(a_operand, RawTcgenMmaA::Tmem(_)) && layout.packed_a_banks() > 1 {
        tail.run_with(&a_values, &b_values, |m, n, k, a, b, d| {
            mma_f32_abt_banked_a_increasing_k(m, n, k, a, b, d, layout.packed_a_banks())
        })
    } else {
        tail.run(&a_values, &b_values)
    }
}

fn raw_tcgen05_validate_sparse_metadata(
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    context: &WarpContext,
    metadata: u32,
    destination_address: u32,
    cta_group: usize,
    metadata_layout: RawTcgenSparseMetadataLayout,
) -> Result<(), EngineError> {
    let first_cta = context.cta_id_in_cluster() & !(cta_group - 1);
    let (metadata_lane, metadata_column) = raw_tcgen05_address(metadata, 0, 0)?;
    if metadata_column % 2 != 0
        || metadata_lane != raw_tcgen05_address(destination_address, 0, 0)?.0
    {
        return Err(EngineError::message(
            "sparse floating MMA metadata requires two-column alignment and matching datapath lanes",
        ));
    }
    for cta in first_cta..first_cta + cta_group {
        validate_raw_tcgen05_tmem_columns(
            lifecycle,
            access_mode,
            context,
            cta,
            metadata_column,
            match metadata_layout {
                RawTcgenSparseMetadataLayout::B16 { .. } => 2,
                RawTcgenSparseMetadataLayout::Narrow { k } => k / 32,
            },
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_sparse_float_tail<Scale: FnOnce() -> Result<f32, EngineError>>(
    tail: RawMmaTail<'_, Scale>,
    a_values: &[f32],
    b_values: &[f32],
    physical: &PhysicalMemory,
    context: &WarpContext,
    anchor: &RuntimeBuffer,
    metadata: u32,
    kind: RawTcgenFloatKind,
    metadata_layout: RawTcgenSparseMetadataLayout,
    layout: RawTcgenDenseTmemLayout,
    cta_group: usize,
) -> Result<(), EngineError> {
    let packed_k = tail.k / 2;
    let first_cta = context.cta_id_in_cluster() & !(cta_group - 1);
    // Implicit zeros do not multiply B. In particular they must not turn
    // an unselected Inf/NaN into NaN via 0 * B. Select B's K terms and feed
    // the packed A values (including explicit zeros) to the ordinary FMA core.
    tail.run_with(a_values, b_values, |m, n, k, a, b, d| {
        let rows = m / cta_group;
        let banks = layout.packed_a_banks();
        let bank_columns = n / banks;
        let a_banks = a.len() / (m * packed_k);
        if a.len() % (m * packed_k) != 0 || !(a_banks == 1 || a_banks == banks) {
            return Err(EngineError::message(
                "sparse floating MMA packed A has the wrong shape",
            ));
        }
        let mut output = vec![0.0; m * n];
        let mut selected_b = vec![0.0; bank_columns * packed_k];
        for cta in 0..cta_group {
            let view = raw_tcgen05_tmem_view(physical, context, anchor, first_cta + cta)?;
            for local_row in 0..rows {
                let row = cta * rows + local_row;
                for bank in 0..banks {
                    let physical_row = layout.packed_a_location(bank, local_row, 0, rows, 8)?.0;
                    for inner in 0..packed_k {
                        let code = raw_tcgen05_sparse_metadata_code(
                            physical,
                            &view,
                            metadata,
                            metadata_layout,
                            physical_row,
                            if matches!(kind, RawTcgenFloatKind::Tf32) {
                                inner
                            } else {
                                inner / 2
                            },
                        )?;
                        let selected = kind.sparse_index(code, inner)?;
                        for col in 0..bank_columns {
                            selected_b[col * packed_k + inner] =
                                b[(bank * bank_columns + col) * k + selected];
                        }
                    }
                    let a_start = ((bank % a_banks) * m + row) * packed_k;
                    let start = row * n + bank * bank_columns;
                    let end = start + bank_columns;
                    let product = mma_f32_abt_increasing_k(
                        1,
                        bank_columns,
                        packed_k,
                        &a[a_start..a_start + packed_k],
                        &selected_b,
                        d.map(|(values, scale)| (&values[start..end], scale)),
                    )?;
                    output[start..end].copy_from_slice(&product);
                }
            }
        }
        Ok(output)
    })
}

#[derive(Clone, Copy, Debug)]
struct RawTcgenMxf8f6f4InstructionDescriptor {
    sparse: bool,
    sfa_lanes: usize,
    k: usize,
    m: usize,
    n: usize,
    a_format: RawTcgenNarrowFormat,
    b_format: RawTcgenNarrowFormat,
    sfa_id: usize,
    sfb_id: usize,
    negate_a: bool,
    negate_b: bool,
    transpose_a: bool,
    transpose_b: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RawTcgenNarrowFormat {
    E4M3,
    E5M2,
    E2M3,
    E3M2,
    E2M1,
}

impl RawTcgenNarrowFormat {
    fn decode(bits: u32, operand: &str) -> Result<Self, EngineError> {
        match bits {
            0 => Ok(Self::E4M3),
            1 => Ok(Self::E5M2),
            3 => Ok(Self::E2M3),
            4 => Ok(Self::E3M2),
            5 => Ok(Self::E2M1),
            _ => Err(EngineError::message(format!(
                "raw mxf8f6f4 {operand} format {bits} is reserved"
            ))),
        }
    }

    const fn format(self) -> NarrowFloatFormat {
        match self {
            Self::E4M3 => FLOAT8_E4M3,
            Self::E5M2 => FLOAT8_E5M2,
            Self::E2M3 => FLOAT6_E2M3,
            Self::E3M2 => FLOAT6_E3M2,
            Self::E2M1 => FLOAT4_E2M1,
        }
    }

    /// Dense K64 (and sparse B K128) packs atoms contiguously; K32 pads to 16 bytes.
    fn shared_atom_stride(self, k: usize) -> usize {
        if k >= 64 {
            self.payload_bytes_per_k16()
        } else {
            16
        }
    }

    const fn payload_bytes_per_k16(self) -> usize {
        self.format().width_bits as usize * 2
    }

    fn decode_value(self, bits: u8) -> f32 {
        narrow_float_bits_to_f32_checked(bits, self.format()).unwrap_or(f32::NAN)
    }

    fn decode_shared_atom(self, bytes: [u8; 16]) -> [f32; 16] {
        if matches!(self, Self::E2M1) {
            return std::array::from_fn(|i| {
                float4_e2m1fn_bits_to_f32(bytes[i / 2] >> (4 * (i % 2)))
            });
        }
        let width = self.format().width_bits;
        let packed = u128::from_le_bytes(bytes);
        let mask = (1_u128 << width) - 1;
        std::array::from_fn(|i| self.decode_value(((packed >> (i as u32 * width)) & mask) as u8))
    }

    fn decode_tmem_word(self, word: u32) -> Result<[f32; 4], EngineError> {
        // PTX K=32 TMEM containers: FP4 uses bits 2..5, FP6 uses
        // bits 0..5, and FP8 consumes the full byte. Numeric formats
        // themselves are shared with register conversions.
        let shift = if matches!(self, Self::E2M1) { 2 } else { 0 };
        let mask = (((1_u16 << self.format().width_bits) - 1) << shift) as u8;
        let mut values = [0.0; 4];
        for (value, bits) in values.iter_mut().zip(word.to_le_bytes()) {
            if bits & !mask != 0 {
                return Err(EngineError::message(
                    "block-scale TMEM A has nonzero padding bits",
                ));
            }
            *value = self.decode_value(bits >> shift);
        }
        Ok(values)
    }
    fn block_tmem_columns(self, k: usize) -> Result<usize, EngineError> {
        match (k, self.format().width_bits) {
            (32, _) => Ok(8),
            (64, 6) => Ok(12),
            _ => Err(EngineError::message(
                "MXF8F6F4 K=64 TMEM A requires packed FP6",
            )),
        }
    }

    fn decode_block_tmem_row(self, words: &[u32], k: usize) -> Result<Vec<f32>, EngineError> {
        if words.len() != self.block_tmem_columns(k)? {
            return Err(EngineError::message("invalid narrow TMEM row length"));
        }
        let mut values = Vec::with_capacity(k);
        if k == 32 {
            for &word in words {
                values.extend(self.decode_tmem_word(word)?);
            }
        } else {
            // 16 FP6 values occupy exactly three words; the 6-bit fields
            // crossing a word boundary use the same 96-bit decoder as shared A.
            for group in words.chunks_exact(3) {
                let mut bytes = [0_u8; 16];
                for (dst, word) in bytes.chunks_exact_mut(4).zip(group) {
                    dst.copy_from_slice(&word.to_le_bytes());
                }
                values.extend(self.decode_shared_atom(bytes));
            }
        }
        Ok(values)
    }
}

fn decode_raw_tcgen_mxf8f6f4_instruction_descriptor(
    descriptor: u32,
    cta_group: usize,
    layout: RawTcgenMatrixDescriptorLayout,
) -> Result<RawTcgenMxf8f6f4InstructionDescriptor, EngineError> {
    if !matches!(cta_group, 1 | 2) {
        return Err(EngineError::message(format!(
            "raw mxf8f6f4 cta_group must be 1 or 2, got {cta_group}"
        )));
    }
    if descriptor & 0xb != 0
        || descriptor & (1_u32 << 6) != 0
        || (descriptor & (1_u32 << 31) != 0 && layout != RawTcgenMatrixDescriptorLayout::Sm107)
        || ((descriptor >> 23) & 0x1) != 1
    {
        return Err(EngineError::message(
            "raw mxf8f6f4 descriptor requires UE8M0 scales and valid K/reserved bits",
        ));
    }
    let a_format = RawTcgenNarrowFormat::decode((descriptor >> 7) & 0x7, "A")?;
    let b_format = RawTcgenNarrowFormat::decode((descriptor >> 10) & 0x7, "B")?;
    let transpose_a = descriptor & (1_u32 << 15) != 0;
    let transpose_b = descriptor & (1_u32 << 16) != 0;
    if transpose_a && a_format.format().width_bits != 8 {
        return Err(EngineError::message(
            "raw mxf8f6f4 transpose A requires an 8-bit operand",
        ));
    }
    if transpose_b && b_format.format().width_bits != 8 {
        return Err(EngineError::message(
            "raw mxf8f6f4 transpose B requires an 8-bit operand",
        ));
    }
    let m = usize::try_from(((descriptor & !(1 << 26)) >> 24) & 0x1f)
        .map_err(|_| EngineError::message("raw mxf8f6f4 M conversion failed"))?
        .checked_mul(16)
        .ok_or_else(|| EngineError::message("raw mxf8f6f4 M overflow"))?;
    let n = usize::try_from((descriptor >> 17) & 0x3f)
        .map_err(|_| EngineError::message("raw mxf8f6f4 N conversion failed"))?
        .checked_mul(8)
        .ok_or_else(|| EngineError::message("raw mxf8f6f4 N overflow"))?;
    let expected_m = 128 * cta_group;
    let k = if descriptor & (1 << 31) != 0 { 64 } else { 32 };
    let n_granularity = if transpose_b {
        16 * cta_group
    } else {
        8 * cta_group
    };
    // PTX MMA shape table permits CTA2 M=128 only for dense K=32.
    let valid_m = m == expected_m || (cta_group == 2 && m == 128 && k == 32 && descriptor & 4 == 0);
    if !valid_m || !(n_granularity..=256).contains(&n) || n % n_granularity != 0 {
        return Err(EngineError::message(format!(
            "raw mxf8f6f4 cta_group={cta_group} requires M={expected_m} (also M=128 for dense CTA2 K=32) and N in {n_granularity}..=256 by {n_granularity}, got M={m}, N={n}"
        )));
    }
    let sparse = descriptor & 4 != 0;
    Ok(RawTcgenMxf8f6f4InstructionDescriptor {
        sparse,
        sfa_lanes: raw_tcgen05_sfa_lanes(descriptor, layout)?,
        k: k * if sparse { 2 } else { 1 },
        m,
        n,
        a_format,
        b_format,
        sfa_id: usize::try_from((descriptor >> 29) & 0x3)
            .map_err(|_| EngineError::message("raw mxf8f6f4 SFA ID conversion failed"))?,
        sfb_id: usize::try_from((descriptor >> 4) & 0x3)
            .map_err(|_| EngineError::message("raw mxf8f6f4 SFB ID conversion failed"))?,
        negate_a: descriptor & (1_u32 << 13) != 0,
        negate_b: descriptor & (1_u32 << 14) != 0,
        transpose_a,
        transpose_b,
    })
}

impl RawTcgenMxf8f6f4InstructionDescriptor {
    fn packed_k(self) -> usize {
        self.k / if self.sparse { 2 } else { 1 }
    }
    fn padded_atoms(self) -> bool {
        self.packed_k() == 32
    }
}

pub(crate) fn raw_tcgen05_mma_block_mxf8f6f4_shape(
    descriptor: u32,
    cta_group: usize,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
) -> Result<(usize, usize, usize), EngineError> {
    let instruction =
        decode_raw_tcgen_mxf8f6f4_instruction_descriptor(descriptor, cta_group, descriptor_layout)?;
    Ok((instruction.m, instruction.n, instruction.k))
}

pub(crate) enum RawTcgenMmaAAccess {
    Shared(RuntimeBuffer, Vec<RawTcgenRuntimeAccess>),
    Tmem(Vec<RawTcgenTmemAccess>),
}

pub(crate) struct RawTcgenMmaFootprints {
    pub(crate) a: RawTcgenMmaAAccess,
    pub(crate) b_source: RuntimeBuffer,
    pub(crate) b_accesses: Vec<RawTcgenRuntimeAccess>,
    pub(crate) sfa_accesses: Vec<RawTcgenTmemAccess>,
    pub(crate) sfb_accesses: Vec<RawTcgenTmemAccess>,
    pub(crate) accumulator_accesses: Vec<RawTcgenTmemAccess>,
    pub(crate) n: usize,
}

fn raw_tcgen05_cta1_shared_matrix_accesses(
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    rows: usize,
    issuing_lane: usize,
    target_cta: usize,
    k: usize,
) -> Result<Vec<RawTcgenRuntimeAccess>, EngineError> {
    let mut accesses = Vec::with_capacity(rows * (k / 32));
    for row in 0..rows {
        for atom in 0..(k / 32) {
            let byte_offset =
                raw_tcgen05_shared_byte_offset(source, descriptor, row, atom * 16, 16)?;
            accesses.push((issuing_lane, Some(target_cta), byte_offset, 16));
        }
    }
    Ok(accesses)
}

fn raw_tcgen05_f8_shared_matrix_accesses(
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    rows: usize,
    k_extent: usize,
    format: RawTcgenNarrowFormat,
    transpose: bool,
    issuing_lane: usize,
    target_cta: usize,
    mask: Option<RawTcgenColumnMask>,
    lut_b: bool,
    padded_atoms: bool,
) -> Result<Vec<RawTcgenRuntimeAccess>, EngineError> {
    if lut_b {
        raw_tcgen05_validate_lut_b(Some(0), k_extent, format, transpose)?;
        let mut accesses = Vec::with_capacity(rows * 3);
        for row in 0..rows {
            raw_tcgen05_lut_b_row_accesses(source, descriptor, row, |offset, bytes| {
                accesses.push((issuing_lane, Some(target_cta), offset, bytes));
                Ok(())
            })?;
        }
        return Ok(accesses);
    }
    if transpose {
        if format.format().width_bits != 8 {
            return Err(EngineError::message(
                "MN-major narrow MMA footprints require 8-bit elements",
            ));
        }
        let mut accesses = Vec::with_capacity(rows.saturating_mul(k_extent));
        for row in 0..rows {
            let Some(row) = mask.map_or(Some(row), |mask| mask.source_column(row)) else {
                continue;
            };
            for k in 0..k_extent {
                let byte_offset =
                    raw_tcgen05_8bit_matrix_byte_offset(source, descriptor, row, k, true)?;
                accesses.push((issuing_lane, Some(target_cta), byte_offset, 1));
            }
        }
        return Ok(accesses);
    }
    if k_extent % 16 != 0 {
        return Err(EngineError::message(format!(
            "raw f8f6f4 footprint K={k_extent} is not a multiple of 16"
        )));
    }
    let atoms = k_extent / 16;
    let mut accesses = Vec::with_capacity(rows.saturating_mul(atoms));
    for row in 0..rows {
        let Some(row) = mask.map_or(Some(row), |mask| mask.source_column(row)) else {
            continue;
        };
        for atom in 0..atoms {
            raw_tcgen05_narrow_shared_atom_accesses(
                source,
                descriptor,
                row,
                k_extent,
                format,
                atom,
                |offset, bytes| {
                    accesses.push((issuing_lane, Some(target_cta), offset, bytes));
                    Ok(())
                },
                padded_atoms,
            )?;
        }
    }
    Ok(accesses)
}

fn raw_tcgen05_cta1_accumulator_accesses(
    context: &WarpContext,
    destination_address: u32,
    m: usize,
    n: usize,
    layout_f: bool,
    disable_output_lane: [u32; 4],
    issuing_lane: usize,
) -> Result<Vec<RawTcgenTmemAccess>, EngineError> {
    let (base_lane, base_column) = raw_tcgen05_address(destination_address, 0, 0)?;
    let mut accesses = Vec::with_capacity(m);
    for row in 0..m {
        let lane_delta = if layout_f {
            raw_tcgen05_layout_f_lane(row)?
        } else {
            row
        };
        let lane = base_lane
            .checked_add(lane_delta)
            .ok_or_else(|| EngineError::message("raw TCGEN destination lane overflow"))?;
        if lane >= 128 {
            return Err(EngineError::message(format!(
                "raw TCGEN destination lane {lane} is outside 128 lanes"
            )));
        }
        if ((disable_output_lane[lane / 32] >> (lane % 32)) & 1) != 0 {
            continue;
        }
        accesses.push((
            issuing_lane,
            issuing_lane,
            Some(context.cta_id_in_cluster()),
            i64::try_from(lane)
                .map_err(|_| EngineError::message("raw TCGEN destination lane exceeds i64"))?,
            0,
            i64::try_from(base_column)
                .map_err(|_| EngineError::message("raw TCGEN destination column exceeds i64"))?,
            n.checked_mul(4)
                .ok_or_else(|| EngineError::message("raw TCGEN destination width overflow"))?,
        ));
    }
    Ok(accesses)
}

fn raw_tcgen05_mxf8_scale_accesses(
    address: u32,
    scale_id: usize,
    rows: usize,
    layout: RawTcgenScaleLayout,
    target_cta: usize,
    issuing_lane: usize,
) -> Result<Vec<RawTcgenTmemAccess>, EngineError> {
    let mut accesses = Vec::with_capacity(rows * layout.replicas());
    for row in 0..rows {
        for replica in 0..layout.replicas() {
            let (lane, column) = layout.location(address, row, replica)?;
            accesses.push((
                issuing_lane,
                issuing_lane,
                Some(target_cta),
                lane as i64,
                scale_id as i64,
                column as i64,
                1,
            ));
        }
    }
    Ok(accesses)
}

fn raw_tcgen05_cta1_scale_accesses(
    context: &WarpContext,
    address: u32,
    scale_id: usize,
    scale_bytes: usize,
    rows: usize,
    issuing_lane: usize,
    lanes_per_column: usize,
) -> Result<Vec<RawTcgenTmemAccess>, EngineError> {
    let (base_lane, _) = raw_tcgen05_block_scale_address(address)?;
    let mut accesses = Vec::with_capacity(rows);
    for row in 0..rows {
        let lane = base_lane
            .checked_add(row % lanes_per_column)
            .ok_or_else(|| EngineError::message("raw TCGEN scale lane overflow"))?;
        if lane >= 128 {
            return Err(EngineError::message(format!(
                "raw TCGEN scale location lane={lane}, byte={scale_id} is outside TMEM"
            )));
        }
        let mut vector = 0;
        while vector < scale_bytes {
            let (chunk_address, byte) =
                raw_tcgen05_scale_chunk(address, scale_id, vector, rows, lanes_per_column)?;
            let (_, base_column) = raw_tcgen05_block_scale_address(chunk_address)?;
            let chunk_bytes = (scale_bytes - vector).min(4 - byte);
            let column = base_column
                .checked_add(row / lanes_per_column)
                .ok_or_else(|| EngineError::message("raw TCGEN scale column overflow"))?;
            accesses.push((
                issuing_lane,
                issuing_lane,
                Some(context.cta_id_in_cluster()),
                i64::try_from(lane)
                    .map_err(|_| EngineError::message("raw TCGEN scale lane exceeds i64"))?,
                i64::try_from(byte)
                    .map_err(|_| EngineError::message("raw TCGEN scale byte exceeds i64"))?,
                i64::try_from(column)
                    .map_err(|_| EngineError::message("raw TCGEN scale column exceeds i64"))?,
                chunk_bytes,
            ));
            vector += chunk_bytes;
        }
    }
    Ok(accesses)
}

fn raw_tcgen05_cta2_scale_accesses(
    context: &WarpContext,
    address: u32,
    scale_id: usize,
    scale_bytes: usize,
    rows_per_cta: usize,
    rows_are_joint: bool,
    issuing_lane: usize,
    lanes_per_column: usize,
) -> Result<Vec<RawTcgenTmemAccess>, EngineError> {
    let pair_base = context.cta_id_in_cluster() & !1_usize;
    if pair_base + 1 >= context.topology().ctas_per_cluster() {
        return Err(EngineError::message(
            "raw TCGEN cta_group=2 scale footprint has no paired CTA",
        ));
    }
    let (base_lane, _) = raw_tcgen05_block_scale_address(address)?;
    let mut accesses = Vec::with_capacity(rows_per_cta * 2);
    for target_offset in 0..2 {
        let target_cta = pair_base + target_offset;
        for row in 0..rows_per_cta {
            let matrix_row = if rows_are_joint {
                target_offset
                    .checked_mul(rows_per_cta)
                    .and_then(|value| value.checked_add(row))
                    .ok_or_else(|| EngineError::message("raw TCGEN cta2 scale row overflow"))?
            } else {
                row
            };
            let lane = base_lane
                .checked_add(matrix_row % lanes_per_column)
                .ok_or_else(|| EngineError::message("raw TCGEN cta2 scale lane overflow"))?;
            if lane >= 128 {
                return Err(EngineError::message(format!(
                    "raw TCGEN cta2 scale location lane={lane}, byte={scale_id} is outside TMEM"
                )));
            }
            let mut vector = 0;
            while vector < scale_bytes {
                let (chunk_address, byte) = raw_tcgen05_scale_chunk(
                    address,
                    scale_id,
                    vector,
                    rows_per_cta * if rows_are_joint { 2 } else { 1 },
                    lanes_per_column,
                )?;
                let (_, base_column) = raw_tcgen05_block_scale_address(chunk_address)?;
                let chunk_bytes = (scale_bytes - vector).min(4 - byte);
                let column = base_column
                    .checked_add(matrix_row / lanes_per_column)
                    .ok_or_else(|| EngineError::message("raw TCGEN cta2 scale column overflow"))?;
                accesses.push((
                    issuing_lane,
                    issuing_lane,
                    Some(target_cta),
                    i64::try_from(lane).map_err(|_| {
                        EngineError::message("raw TCGEN cta2 scale lane exceeds i64")
                    })?,
                    i64::try_from(byte).map_err(|_| {
                        EngineError::message("raw TCGEN cta2 scale byte exceeds i64")
                    })?,
                    i64::try_from(column).map_err(|_| {
                        EngineError::message("raw TCGEN cta2 scale column exceeds i64")
                    })?,
                    chunk_bytes,
                ));
                vector += chunk_bytes;
            }
        }
    }
    Ok(accesses)
}

pub(crate) fn raw_tcgen05_f8_shared_footprints(
    context: &WarpContext,
    shared_candidates: &[&RuntimeBuffer],
    descriptor_bits: u64,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
    rows: usize,
    k: usize,
    format: RawTcgenNarrowFormat,
    transpose: bool,
    issuing_lane: usize,
    mask: Option<RawTcgenColumnMask>,
    lut_b: Option<u32>,
) -> Result<(RuntimeBuffer, Vec<(usize, Option<usize>, usize, usize)>), EngineError> {
    let descriptor = raw_tcgen05_f8_b_descriptor(descriptor_bits, descriptor_layout, lut_b)?;
    validate_raw_tcgen05_issuing_lane(
        context,
        issuing_lane,
        &crate::DiagnosticLabel::new("raw f8f6f4 tcgen05.mma footprint"),
    )?;
    let source = raw_tcgen05_shared_source(
        context,
        shared_candidates,
        descriptor.start_address,
        issuing_lane,
    )?;
    let accesses = raw_tcgen05_f8_shared_matrix_accesses(
        &source,
        descriptor,
        rows,
        k,
        format,
        transpose,
        issuing_lane,
        context.cta_id_in_cluster(),
        mask,
        lut_b.is_some(),
        false,
    )?;
    Ok((source, accesses))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_tcgen05_mma_f8f6f4_cta2_footprints(
    context: &WarpContext,
    shared_candidates: &[&RuntimeBuffer],
    destination_address: u32,
    a_descriptor_bits: u64,
    b_descriptor_bits: u64,
    instruction_descriptor: u32,
    disable_output_lane: [u32; 8],
    issuing_lane: usize,
    a_format: RawTcgenNarrowFormat,
    b_format: RawTcgenNarrowFormat,
    d_f16: bool,
    a_in_tmem: bool,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
    lut_b: Option<u32>,
) -> Result<RawTcgenMmaFootprints, EngineError> {
    let instruction = decode_raw_tcgen_f8f6f4_instruction_descriptor(
        instruction_descriptor,
        a_format,
        b_format,
        d_f16,
        2,
        descriptor_layout.supports_f8f6f4_k64(),
        false,
        false,
    )?;
    let b_descriptor = raw_tcgen05_f8_b_descriptor(b_descriptor_bits, descriptor_layout, lut_b)?;
    validate_raw_tcgen05_issuing_lane(
        context,
        issuing_lane,
        &crate::DiagnosticLabel::new("raw f8f6f4 cta2 tcgen05.mma footprint"),
    )?;
    let b_source = raw_tcgen05_shared_source(
        context,
        shared_candidates,
        b_descriptor.start_address,
        issuing_lane,
    )?;
    let pair_base = context.cta_id_in_cluster() & !1_usize;
    if pair_base + 1 >= context.topology().ctas_per_cluster() {
        return Err(EngineError::message(
            "raw f8f6f4 cta_group=2 footprint has no paired CTA",
        ));
    }
    let a = if a_in_tmem {
        let address = raw_tcgen05_f8_tmem_a_address(a_descriptor_bits, instruction.transpose_a)?;
        RawTcgenMmaAAccess::Tmem(raw_tcgen05_cta2_packed_tmem_a_footprints(
            context,
            address,
            instruction.m,
            raw_tcgen05_cta1_dense_tmem_layout(instruction.m / 2, true)?,
            instruction.k / 4,
            issuing_lane,
            issuing_lane,
        )?)
    } else {
        let desc =
            decode_raw_tcgen_matrix_descriptor_for_layout(a_descriptor_bits, descriptor_layout)?;
        let source = raw_tcgen05_shared_source(
            context,
            shared_candidates,
            desc.start_address,
            issuing_lane,
        )?;
        let mut accesses = Vec::new();
        for cta in pair_base..pair_base + 2 {
            accesses.extend(raw_tcgen05_f8_shared_matrix_accesses(
                &source,
                desc,
                instruction.m / 2,
                instruction.k,
                a_format,
                instruction.transpose_a,
                issuing_lane,
                cta,
                None,
                false,
                false,
            )?);
        }
        RawTcgenMmaAAccess::Shared(source, accesses)
    };
    let mut b_accesses = Vec::new();
    for target_cta in [pair_base, pair_base + 1] {
        b_accesses.extend(raw_tcgen05_f8_shared_matrix_accesses(
            &b_source,
            b_descriptor,
            instruction.n / 2,
            instruction.k,
            b_format,
            instruction.transpose_b,
            issuing_lane,
            target_cta,
            None,
            lut_b.is_some(),
            false,
        )?);
    }
    Ok(RawTcgenMmaFootprints {
        a,
        b_source,
        b_accesses,
        sfa_accesses: Vec::new(),
        sfb_accesses: Vec::new(),
        accumulator_accesses: raw_tcgen05_cta2_f32_tmem_footprints(
            context,
            destination_address,
            instruction.m,
            instruction.n,
            disable_output_lane,
            issuing_lane,
            issuing_lane,
        )?,
        n: instruction.n,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_tcgen05_mma_block_mxf4_footprints(
    context: &WarpContext,
    shared_candidates: &[&RuntimeBuffer],
    destination_address: u32,
    a_operand: RawTcgenMmaA,
    b_descriptor_bits: u64,
    sfa_address: u32,
    sfb_address: u32,
    instruction_descriptor: u32,
    issuing_lane: usize,
    scale: RawTcgenMxf4ScaleSpelling,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
    fixed_vectors: bool,
    cta_group: usize,
) -> Result<RawTcgenMmaFootprints, EngineError> {
    let instruction = decode_raw_tcgen_mxf4_instruction_descriptor_for_cta_group(
        instruction_descriptor,
        scale,
        cta_group,
        descriptor_layout,
        fixed_vectors,
    )?;
    validate_raw_tcgen05_issuing_lane(
        context,
        issuing_lane,
        &crate::DiagnosticLabel::new("raw block-scale MMA"),
    )?;
    let first_cta = context.cta_id_in_cluster() & !(cta_group - 1);
    if first_cta + cta_group > context.topology().ctas_per_cluster() {
        return Err(EngineError::message(
            "raw block-scale MMA has no paired CTA",
        ));
    }
    let b_descriptor = decode_raw_tcgen_packed_matrix_descriptor(
        b_descriptor_bits,
        descriptor_layout,
        instruction.k / 2,
        false,
    )?;
    let b_source = raw_tcgen05_shared_source(
        context,
        shared_candidates,
        b_descriptor.start_address,
        issuing_lane,
    )?;
    let rows_a = instruction.m / cta_group;
    let rows_b = instruction.n / cta_group;
    let a = match a_operand {
        RawTcgenMmaA::Shared(bits) => {
            let descriptor = decode_raw_tcgen_packed_matrix_descriptor(
                bits,
                descriptor_layout,
                instruction.k / 2,
                false,
            )?;
            let source = raw_tcgen05_shared_source(
                context,
                shared_candidates,
                descriptor.start_address,
                issuing_lane,
            )?;
            let mut accesses = Vec::new();
            for cta in first_cta..first_cta + cta_group {
                accesses.extend(raw_tcgen05_cta1_shared_matrix_accesses(
                    &source,
                    descriptor,
                    rows_a,
                    issuing_lane,
                    cta,
                    instruction.k,
                )?);
            }
            RawTcgenMmaAAccess::Shared(source, accesses)
        }
        RawTcgenMmaA::Tmem(address) => {
            let mut accesses = Vec::new();
            for cta in first_cta..first_cta + cta_group {
                let mut local = raw_tcgen05_packed_tmem_a_column_footprints(
                    address,
                    rows_a,
                    RawTcgenDenseTmemLayout::D,
                    instruction.k / 8,
                    issuing_lane,
                    issuing_lane,
                )?;
                for access in &mut local {
                    access.2 = Some(cta);
                }
                accesses.extend(local);
            }
            RawTcgenMmaAAccess::Tmem(accesses)
        }
    };
    let mut b_accesses = Vec::new();
    for cta in first_cta..first_cta + cta_group {
        b_accesses.extend(raw_tcgen05_cta1_shared_matrix_accesses(
            &b_source,
            b_descriptor,
            rows_b,
            issuing_lane,
            cta,
            instruction.k,
        )?);
    }
    let (sfa_accesses, sfb_accesses, accumulator_accesses) = if cta_group == 1 {
        (
            raw_tcgen05_cta1_scale_accesses(
                context,
                sfa_address,
                instruction.sfa_id,
                instruction.k / instruction.block_elements,
                rows_a,
                issuing_lane,
                instruction.sfa_lanes,
            )?,
            raw_tcgen05_cta1_scale_accesses(
                context,
                sfb_address,
                instruction.sfb_id,
                instruction.k / instruction.block_elements,
                rows_b,
                issuing_lane,
                32,
            )?,
            raw_tcgen05_cta1_accumulator_accesses(
                context,
                destination_address,
                instruction.m,
                instruction.n,
                false,
                [0; 4],
                issuing_lane,
            )?,
        )
    } else {
        (
            raw_tcgen05_cta2_scale_accesses(
                context,
                sfa_address,
                instruction.sfa_id,
                instruction.k / instruction.block_elements,
                rows_a,
                false,
                issuing_lane,
                instruction.sfa_lanes,
            )?,
            raw_tcgen05_cta2_scale_accesses(
                context,
                sfb_address,
                instruction.sfb_id,
                instruction.k / instruction.block_elements,
                rows_b,
                true,
                issuing_lane,
                32,
            )?,
            raw_tcgen05_cta2_f32_tmem_footprints(
                context,
                destination_address,
                instruction.m,
                instruction.n,
                [0; 8],
                issuing_lane,
                issuing_lane,
            )?,
        )
    };
    Ok(RawTcgenMmaFootprints {
        a,
        b_source,
        b_accesses,
        sfa_accesses,
        sfb_accesses,
        accumulator_accesses,
        n: instruction.n,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_tcgen05_mma_block_mxf8f6f4_footprints(
    context: &WarpContext,
    shared_candidates: &[&RuntimeBuffer],
    destination_address: u32,
    a_operand: RawTcgenMmaA,
    b_descriptor_bits: u64,
    sfa_address: u32,
    sfb_address: u32,
    instruction_descriptor: u32,
    issuing_lane: usize,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
    cta_group: usize,
    lut_b: Option<u32>,
    metadata: Option<u32>,
) -> Result<RawTcgenMmaFootprints, EngineError> {
    let instruction = decode_raw_tcgen_mxf8f6f4_instruction_descriptor(
        instruction_descriptor,
        cta_group,
        descriptor_layout,
    )?;
    if instruction.sparse != metadata.is_some() {
        return Err(EngineError::message(
            "MXF8F6F4 sparsity bit and metadata operand disagree",
        ));
    }
    let packed_k = instruction.packed_k();
    let layout = raw_tcgen05_cta1_dense_tmem_layout(instruction.m / cta_group, true)?;
    validate_raw_tcgen05_issuing_lane(
        context,
        issuing_lane,
        &crate::DiagnosticLabel::new("raw block-scale MMA"),
    )?;
    let first_cta = context.cta_id_in_cluster() & !(cta_group - 1);
    if first_cta + cta_group > context.topology().ctas_per_cluster() {
        return Err(EngineError::message(
            "raw block-scale MMA has no paired CTA",
        ));
    }
    raw_tcgen05_validate_lut_b(
        lut_b,
        instruction.k,
        instruction.b_format,
        instruction.transpose_b,
    )?;
    let b_descriptor = if lut_b.is_some() {
        raw_tcgen05_f8_b_descriptor(b_descriptor_bits, descriptor_layout, lut_b)?
    } else {
        decode_raw_tcgen_packed_matrix_descriptor(
            b_descriptor_bits,
            descriptor_layout,
            instruction.k / 16
                * if instruction.padded_atoms() {
                    16
                } else {
                    instruction.b_format.shared_atom_stride(instruction.k)
                },
            instruction.transpose_a || instruction.transpose_b,
        )?
    };
    let b_source = raw_tcgen05_shared_source(
        context,
        shared_candidates,
        b_descriptor.start_address,
        issuing_lane,
    )?;
    let rows_a = instruction.m / cta_group;
    let rows_b = instruction.n / cta_group;
    let a = match a_operand {
        RawTcgenMmaA::Shared(bits) => {
            let descriptor = decode_raw_tcgen_packed_matrix_descriptor(
                bits,
                descriptor_layout,
                packed_k / 16 * instruction.a_format.shared_atom_stride(packed_k),
                instruction.transpose_a || instruction.transpose_b,
            )?;
            let source = raw_tcgen05_shared_source(
                context,
                shared_candidates,
                descriptor.start_address,
                issuing_lane,
            )?;
            let mut accesses = Vec::new();
            for cta in first_cta..first_cta + cta_group {
                accesses.extend(raw_tcgen05_f8_shared_matrix_accesses(
                    &source,
                    descriptor,
                    rows_a,
                    packed_k,
                    instruction.a_format,
                    instruction.transpose_a,
                    issuing_lane,
                    cta,
                    None,
                    false,
                    false,
                )?);
            }
            RawTcgenMmaAAccess::Shared(source, accesses)
        }
        RawTcgenMmaA::Tmem(address) => {
            raw_tcgen05_validate_tmem_a_transpose(instruction.transpose_a)?;
            let accesses = if cta_group == 1 {
                raw_tcgen05_packed_tmem_a_column_footprints(
                    address,
                    rows_a,
                    RawTcgenDenseTmemLayout::D,
                    instruction.a_format.block_tmem_columns(packed_k)?,
                    issuing_lane,
                    issuing_lane,
                )?
            } else {
                raw_tcgen05_cta2_packed_tmem_a_footprints(
                    context,
                    address,
                    instruction.m,
                    layout,
                    instruction.a_format.block_tmem_columns(packed_k)?,
                    issuing_lane,
                    issuing_lane,
                )?
            };
            RawTcgenMmaAAccess::Tmem(accesses)
        }
    };
    let mut b_accesses = Vec::new();
    for cta in first_cta..first_cta + cta_group {
        b_accesses.extend(raw_tcgen05_f8_shared_matrix_accesses(
            &b_source,
            b_descriptor,
            rows_b,
            instruction.k,
            instruction.b_format,
            instruction.transpose_b,
            issuing_lane,
            cta,
            None,
            lut_b.is_some(),
            instruction.padded_atoms(),
        )?);
    }
    let mut sfa_accesses = Vec::new();
    let mut sfb_accesses = Vec::new();
    for target_cta in first_cta..first_cta + cta_group {
        if instruction.sfa_lanes == 32 {
            sfa_accesses.extend(raw_tcgen05_mxf8_scale_accesses(
                sfa_address, instruction.sfa_id, rows_a,
                raw_tcgen05_mxf8_scale_layout(instruction.m, instruction.n, cta_group, false),
                target_cta, issuing_lane,
            )?);
        }
        sfb_accesses.extend(raw_tcgen05_mxf8_scale_accesses(
            sfb_address, instruction.sfb_id, rows_b * cta_group,
            raw_tcgen05_mxf8_scale_layout(instruction.m, instruction.n, cta_group, true),
            target_cta, issuing_lane,
        )?);
    }
    if instruction.sfa_lanes != 32 {
        sfa_accesses = if cta_group == 1 {
            raw_tcgen05_cta1_scale_accesses(
                context, sfa_address, instruction.sfa_id, 1, rows_a, issuing_lane,
                instruction.sfa_lanes,
            )?
        } else {
            raw_tcgen05_cta2_scale_accesses(
                context, sfa_address, instruction.sfa_id, 1, rows_a, false, issuing_lane,
                instruction.sfa_lanes,
            )?
        };
    }
    let accumulator_accesses = if cta_group == 1 {
        raw_tcgen05_cta1_accumulator_accesses(
            context, destination_address, instruction.m, instruction.n,
            false, [0; 4], issuing_lane,
        )?
    } else {
        raw_tcgen05_cta2_layout_tmem_footprints(
            context, destination_address, instruction.m, instruction.n,
            layout, [0; 8], issuing_lane, issuing_lane,
        )?
    };
    Ok(RawTcgenMmaFootprints {
        a,
        b_source,
        b_accesses,
        sfa_accesses,
        sfb_accesses,
        accumulator_accesses,
        n: instruction.n,
    })
}

#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_gather_mxf8f6f4_matrix(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    anchor: &RuntimeBuffer,
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    rows_per_cta: usize,
    k: usize,
    cta_group: usize,
    format: RawTcgenNarrowFormat,
    transpose: bool,
    scale_rows_are_joint: bool,
    scale_address: u32,
    scale_id: usize,
    scale_layout: RawTcgenScaleLayout,
    negate: bool,
    lanes_per_column: usize,
    lut_b: Option<(u32, usize)>,
    padded_atoms: bool,
) -> Result<Vec<f32>, EngineError> {
    // Dense and block-scaled operands share one packing/address walk. Scaling
    // owns only the extra TMEM reads and one multiply per decoded element.
    let mut values = if let Some((address, segment)) = lut_b {
        raw_tcgen05_gather_lut_b(
            physical,
            context,
            lifecycle,
            access_mode,
            anchor,
            source,
            descriptor,
            segment,
            address,
            rows_per_cta,
            cta_group,
            false,
        )?
    } else {
        raw_tcgen05_gather_f8_shared_matrix(
            physical,
            context,
            source,
            descriptor,
            rows_per_cta,
            k,
            format,
            false,
            transpose,
            cta_group,
            None,
            padded_atoms,
        )?
    };
    let first_cta = context.cta_id_in_cluster() & !(cta_group - 1);
    let mut joint_scales: Option<Vec<f32>> = None;
    for (cta, values) in values.chunks_exact_mut(rows_per_cta * k).enumerate() {
        let target_cta = first_cta + cta;
        let rows = rows_per_cta * if scale_rows_are_joint { cta_group } else { 1 };
        let (_, column) = raw_tcgen05_block_scale_address(scale_address)?;
        let columns = if lanes_per_column == 32 {
            let (_, last) = scale_layout.location(scale_address, rows - 1, 0)?;
            last - column + 1
        } else {
            raw_tcgen05_scale_columns(rows, scale_id, 1, lanes_per_column)?
        };
        validate_raw_tcgen05_tmem_columns(
            lifecycle, access_mode, context, target_cta, column, columns,
        )?;
        let view = raw_tcgen05_tmem_view(physical, context, anchor, target_cta)?;
        let scales = if lanes_per_column == 32 {
            raw_tcgen05_mxf8_scale_values(
                physical, &view, scale_address, scale_id, rows, scale_layout,
            )?
        } else {
            (0..rows).map(|row| raw_tcgen05_read_block_scale(
                physical, &view, scale_address, scale_id, row, 0,
                raw_tcgen05_decode_ue8m0_scale, rows, lanes_per_column,
            )).collect::<Result<Vec<_>, _>>()?
        };
        if scale_rows_are_joint {
            if joint_scales.as_ref().is_some_and(|previous| {
                previous.iter().zip(&scales).any(|(a, b)| a.to_bits() != b.to_bits())
            }) {
                return Err(EngineError::message("raw TCGEN SFB copies disagree across CTAs"));
            }
            joint_scales = Some(scales.clone());
        }
        for (row, values) in values.chunks_exact_mut(k).enumerate() {
            let scale_row = row
                + if scale_rows_are_joint {
                    cta * rows_per_cta
                } else {
                    0
                };
            let scale = scales[scale_row];
            for value in values {
                *value *= scale;
                if negate {
                    *value = -*value;
                }
            }
        }
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn raw_tcgen05_scatter_cta2(
    physical: &PhysicalMemory,
    context: &WarpContext,
    anchor: &RuntimeBuffer,
    destination_address: u32,
    m: usize,
    n: usize,
    layout: RawTcgenDenseTmemLayout,
    disable_output_lane: [u32; 8],
    cell_dtype: RawMmaCellDtype,
    output_values: &[f32],
) -> Result<(), EngineError> {
    if m != 128 && m != 256 {
        return Err(EngineError::message(format!(
            "raw cta_group=2 destination requires M=128 or 256, got {m}"
        )));
    }
    let pair_base = context.cta_id_in_cluster() & !1_usize;
    let (base_lane, base_col) = raw_tcgen05_address(destination_address, 0, 0)?;
    let rows_per_cta = m / 2;
    let columns_per_bank = layout.physical_columns(n)?;
    if matches!(
        layout,
        RawTcgenDenseTmemLayout::D | RawTcgenDenseTmemLayout::E
    ) && matches!(cell_dtype, RawMmaCellDtype::F32)
        && disable_output_lane.iter().all(|&word| word == 0)
    {
        let lane_count = if m == 128 {
            rows_per_cta * 2
        } else {
            rows_per_cta
        };
        let lane_end = base_lane
            .checked_add(lane_count)
            .ok_or_else(|| EngineError::message("raw cta_group=2 lane overflow"))?;
        if lane_end > 128 {
            return Err(EngineError::message(format!(
                "raw cta_group=2 lanes [{base_lane}, {lane_end}) exceed 128 lanes"
            )));
        }
        let column_end = base_col
            .checked_add(columns_per_bank)
            .ok_or_else(|| EngineError::message("raw cta_group=2 column overflow"))?;
        let values_per_cta = rows_per_cta
            .checked_mul(n)
            .ok_or_else(|| EngineError::message("raw cta_group=2 output shape overflow"))?;
        if output_values.len() != values_per_cta.saturating_mul(2) {
            return Err(EngineError::message(format!(
                "raw cta_group=2 output has {} values, expected {}",
                output_values.len(),
                values_per_cta.saturating_mul(2),
            )));
        }
        for target_offset in 0..2 {
            let target_cta = pair_base + target_offset;
            let view = raw_tcgen05_tmem_view(physical, context, anchor, target_cta)?;
            let source = &output_values
                [target_offset * values_per_cta..(target_offset + 1) * values_per_cta];
            physical
                .tmem()
                .validate_cell_rectangle(&view, lane_end, column_end)?;
            physical.tmem().with_write_session(&view, |session| {
                for row in 0..rows_per_cta {
                    let source_start = row * n;
                    if m == 128 {
                        session.write_f32_cells_prevalidated(
                            base_lane + row,
                            base_col,
                            &source[source_start..source_start + columns_per_bank],
                        );
                        session.write_f32_cells_prevalidated(
                            base_lane + rows_per_cta + row,
                            base_col,
                            &source[source_start + columns_per_bank..source_start + n],
                        );
                    } else {
                        session.write_f32_cells_prevalidated(
                            base_lane + row,
                            base_col,
                            &source[source_start..source_start + n],
                        );
                    }
                }
            })?;
        }
        return Ok(());
    }
    for target_offset in 0..2 {
        let target_cta = pair_base + target_offset;
        let view = raw_tcgen05_tmem_view(physical, context, anchor, target_cta)?;
        for row in 0..rows_per_cta {
            for col in 0..n {
                let (lane_delta, physical_col) = layout.location(row, col, rows_per_cta, n)?;
                let lane = base_lane
                    .checked_add(lane_delta)
                    .ok_or_else(|| EngineError::message("raw cta_group=2 lane overflow"))?;
                if lane >= 128 {
                    return Err(EngineError::message(format!(
                        "raw cta_group=2 destination lane {lane} is outside 128 lanes"
                    )));
                }
                // Pair-word split is the documented inference; see
                // `raw_tcgen05_cta2_f32_tmem_footprints`.
                if ((disable_output_lane[target_offset * 4 + lane / 32] >> (lane % 32)) & 1) != 0 {
                    continue;
                }
                let column = base_col
                    .checked_add(physical_col)
                    .ok_or_else(|| EngineError::message("raw cta_group=2 column overflow"))?;
                let output_index = target_offset
                    .checked_mul(rows_per_cta)
                    .and_then(|value| value.checked_add(row))
                    .and_then(|value| value.checked_mul(n))
                    .and_then(|value| value.checked_add(col))
                    .ok_or_else(|| EngineError::message("raw cta_group=2 output index overflow"))?;
                physical.tmem().write_cell_bytes(
                    &view,
                    lane,
                    column,
                    0,
                    &cell_dtype.encode(output_values[output_index]),
                )?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_tcgen05_mma_block_scale_mxf8f6f4(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    anchor: &RuntimeBuffer,
    shared_candidates: &[&RuntimeBuffer],
    destination_address: u32,
    a_operand: RawTcgenMmaA,
    b_descriptor_bits: u64,
    sfa_address: u32,
    sfb_address: u32,
    instruction_descriptor: u32,
    enable_input_d: bool,
    issuing_lane: usize,
    cta_group: usize,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
    lut_b: Option<u32>,
    metadata: Option<u32>,
) -> Result<(), EngineError> {
    let instruction = decode_raw_tcgen_mxf8f6f4_instruction_descriptor(
        instruction_descriptor,
        cta_group,
        descriptor_layout,
    )?;
    if instruction.sparse != metadata.is_some() {
        return Err(EngineError::message(
            "MXF8F6F4 sparsity bit and metadata operand disagree",
        ));
    }
    if let Some(metadata) = metadata {
        raw_tcgen05_validate_sparse_metadata(
            lifecycle,
            access_mode,
            context,
            metadata,
            destination_address,
            cta_group,
            RawTcgenSparseMetadataLayout::Narrow { k: instruction.k },
        )?;
    }
    let packed_k = instruction.packed_k();
    let layout = raw_tcgen05_cta1_dense_tmem_layout(instruction.m / cta_group, true)?;
    validate_raw_tcgen05_issuing_lane(
        context,
        issuing_lane,
        &crate::DiagnosticLabel::new("raw block-scale MMA"),
    )?;
    raw_tcgen05_validate_lut_b(
        lut_b,
        instruction.k,
        instruction.b_format,
        instruction.transpose_b,
    )?;
    let b_descriptor = if lut_b.is_some() {
        raw_tcgen05_f8_b_descriptor(b_descriptor_bits, descriptor_layout, lut_b)?
    } else {
        decode_raw_tcgen_packed_matrix_descriptor(
            b_descriptor_bits,
            descriptor_layout,
            instruction.k / 16
                * if instruction.padded_atoms() {
                    16
                } else {
                    instruction.b_format.shared_atom_stride(instruction.k)
                },
            instruction.transpose_a || instruction.transpose_b,
        )?
    };
    let b_source = raw_tcgen05_shared_source(
        context,
        shared_candidates,
        b_descriptor.start_address,
        issuing_lane,
    )?;
    let first_cta = context.cta_id_in_cluster() & !(cta_group - 1);
    if first_cta + cta_group > context.topology().ctas_per_cluster() {
        return Err(EngineError::message("block-scale MMA has no paired CTA"));
    }
    let (_, destination_column) = raw_tcgen05_address(destination_address, 0, 0)?;
    for cta in first_cta..first_cta + cta_group {
        validate_raw_tcgen05_tmem_columns(
            lifecycle,
            access_mode,
            context,
            cta,
            destination_column,
            layout.physical_columns(instruction.n)?,
        )?;
    }
    let a_values = match a_operand {
        RawTcgenMmaA::Shared(bits) => {
            let descriptor = decode_raw_tcgen_packed_matrix_descriptor(
                bits,
                descriptor_layout,
                packed_k / 16 * instruction.a_format.shared_atom_stride(packed_k),
                instruction.transpose_a || instruction.transpose_b,
            )?;
            let source = raw_tcgen05_shared_source(
                context,
                shared_candidates,
                descriptor.start_address,
                issuing_lane,
            )?;
            raw_tcgen05_gather_mxf8f6f4_matrix(
                physical,
                context,
                lifecycle,
                access_mode,
                anchor,
                &source,
                descriptor,
                instruction.m / cta_group,
                packed_k,
                cta_group,
                instruction.a_format,
                instruction.transpose_a,
                false,
                sfa_address,
                instruction.sfa_id,
                raw_tcgen05_mxf8_scale_layout(instruction.m, instruction.n, cta_group, false),
                instruction.negate_a,
                instruction.sfa_lanes,
                None,
                instruction.padded_atoms(),
            )?
        }
        RawTcgenMmaA::Tmem(address) => {
            raw_tcgen05_validate_tmem_a_transpose(instruction.transpose_a)?;
            raw_tcgen05_gather_scaled_tmem_a(
                physical,
                context,
                lifecycle,
                access_mode,
                anchor,
                address,
                instruction.m,
                packed_k,
                instruction.a_format.block_tmem_columns(packed_k)?,
                cta_group,
                sfa_address,
                instruction.sfa_id,
                packed_k,
                raw_tcgen05_decode_ue8m0_scale,
                instruction.negate_a,
                |words| instruction.a_format.decode_block_tmem_row(words, packed_k),
                instruction.sfa_lanes,
            )?
        }
    };
    let b_values = raw_tcgen05_gather_mxf8f6f4_matrix(
        physical,
        context,
        lifecycle,
        access_mode,
        anchor,
        &b_source,
        b_descriptor,
        instruction.n / cta_group,
        instruction.k,
        cta_group,
        instruction.b_format,
        instruction.transpose_b,
        true,
        sfb_address,
        instruction.sfb_id,
        raw_tcgen05_mxf8_scale_layout(instruction.m, instruction.n, cta_group, true),
        instruction.negate_b,
        32,
        lut_b.map(|address| (address, ((b_descriptor_bits >> 53) & 1) as usize)),
        instruction.padded_atoms(),
    )?;
    let cta1_view = if cta_group == 1 {
        Some(raw_tcgen05_tmem_view(
            physical,
            context,
            anchor,
            context.cta_id_in_cluster(),
        )?)
    } else {
        None
    };
    let destination = match cta1_view.as_ref() {
        Some(view) => RawMmaDestination::LaneCells(RawMmaWindow {
            physical,
            view,
            destination_address,
            layout: RawTcgenDenseTmemLayout::D,
            cell_dtype: RawMmaCellDtype::F32,
            disable_output_lane: [0_u32; 4],
        }),
        None => RawMmaDestination::Cta2 {
            physical,
            context,
            anchor,
            destination_address,
            layout,
            disable_output_lane: [0_u32; 8],
            cell_dtype: RawMmaCellDtype::F32,
        },
    };
    let tail = RawMmaTail {
        m: instruction.m,
        n: instruction.n,
        k: instruction.k,
        enable_input_d,
        destination,
        scale: || Ok(1.0_f32),
    };
    if let Some(metadata) = metadata {
        return raw_tcgen05_sparse_float_tail(
            tail,
            &a_values,
            &b_values,
            physical,
            context,
            anchor,
            metadata,
            RawTcgenFloatKind::SparseNarrow {
                a_format: instruction.a_format,
                b_format: instruction.b_format,
                descriptor_layout,
            },
            RawTcgenSparseMetadataLayout::Narrow { k: instruction.k },
            layout,
            cta_group,
        );
    }
    if matches!(a_operand, RawTcgenMmaA::Tmem(_)) && layout.packed_a_banks() > 1 {
        tail.run_with(&a_values, &b_values, |m, n, k, a, b, d| {
            mma_f32_abt_banked_a_increasing_k(m, n, k, a, b, d, layout.packed_a_banks())
        })
    } else {
        tail.run(&a_values, &b_values)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        raw_tcgen05_cp_destination_lanes, raw_tcgen05_unpack_b6, RawMmaCellDtype,
        RawTcgenNarrowFormat,
    };
    use std::sync::Arc;

    use super::{
        decode_raw_tcgen_f8f6f4_instruction_descriptor, decode_raw_tcgen_matrix_descriptor,
        decode_raw_tcgen_matrix_descriptor_for_layout,
        decode_raw_tcgen_mxf4_instruction_descriptor,
        decode_raw_tcgen_mxf8f6f4_instruction_descriptor, decode_tile_gemm_bf16_snapshot,
        mma_f32_abt_increasing_k, raw_tcgen05_8bit_matrix_byte_offset,
        raw_tcgen05_decode_ue4m3_scale, raw_tcgen05_ldst_location,
        raw_tcgen05_mma_block_mxf4_shape, raw_tcgen05_shared_byte_offset,
        raw_tcgen05_shared_source, raw_tcgen05_tf32_payload_to_f32,
        tile_gemm_operand_physical_elements, validate_raw_tcgen05_issuing_lane, RawTcgenLdstShape,
        RawTcgenMatrixDescriptor, RawTcgenMatrixDescriptorLayout, RawTcgenMxf4ScaleSpelling,
        RuntimeBuffer, TileGemmBf16OperandLayout, TileGemmOperandLayout,
    };
    use crate::{LaunchTopology, WarpMask};

    /// `M=128, N=8, K=64`, E2M1 A and B, no negate or transpose. Bit 23 is the
    /// scale matrix type field, bits 29-30 the matrix A scale-factor data ID.
    fn mxf4_family_descriptor(ue8m0: bool, sfa_id: u32) -> u32 {
        (1 << 7)
            | (1 << 10)
            | ((8 >> 3) << 17)
            | (u32::from(ue8m0) << 23)
            | (8 << 24)
            | (sfa_id << 29)
    }

    fn mxf8f6f4_descriptor(m: u32, n: u32) -> u32 {
        ((n >> 3) << 17) | (1 << 23) | ((m >> 4) << 24)
    }

    #[test]
    fn mxf8f6f4_descriptor_contract_accepts_cta1_eight_column_granularity() {
        for n in [8, 24] {
            let decoded = decode_raw_tcgen_mxf8f6f4_instruction_descriptor(
                mxf8f6f4_descriptor(128, n),
                1,
                RawTcgenMatrixDescriptorLayout::Sm100,
            )
            .unwrap();
            assert_eq!((decoded.m, decoded.n), (128, n as usize));
        }
    }

    #[test]
    fn mxf8f6f4_descriptor_accepts_both_cta2_accumulator_layouts() {
        for m in [128, 256] {
            for n in [16, 256] {
                let decoded =
                    decode_raw_tcgen_mxf8f6f4_instruction_descriptor(mxf8f6f4_descriptor(m, n), 2, RawTcgenMatrixDescriptorLayout::Sm100)
                        .unwrap();
                assert_eq!((decoded.m, decoded.n), (m as usize, n as usize));
            }
        }
        for m in [64, 144, 384] {
            assert!(decode_raw_tcgen_mxf8f6f4_instruction_descriptor(
                mxf8f6f4_descriptor(m, 32),
                2,
                RawTcgenMatrixDescriptorLayout::Sm100,
            )
            .is_err());
        }
    }

    #[test]
    fn mxf8f6f4_descriptor_contract_reports_invalid_m_as_geometry() {
        let error = match decode_raw_tcgen_mxf8f6f4_instruction_descriptor(
            mxf8f6f4_descriptor(144, 16),
            1,
            RawTcgenMatrixDescriptorLayout::Sm100,
        ) {
            Ok(_) => panic!("invalid M descriptor decoded successfully"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("requires M=128"));
    }

    #[test]
    fn mxf8f6f4_descriptor_decodes_e4m3_transpose_and_enforces_b_granularity() {
        let descriptor = mxf8f6f4_descriptor(128, 16) | (1 << 15) | (1 << 16);
        let decoded = decode_raw_tcgen_mxf8f6f4_instruction_descriptor(
            descriptor,
            1,
            RawTcgenMatrixDescriptorLayout::Sm100,
        )
        .unwrap();
        assert!(decoded.transpose_a);
        assert!(decoded.transpose_b);

        let error = decode_raw_tcgen_mxf8f6f4_instruction_descriptor(
            mxf8f6f4_descriptor(128, 8) | (1 << 16),
            1,
            RawTcgenMatrixDescriptorLayout::Sm100,
        )
        .unwrap_err();
        assert!(error.to_string().contains("N in 16..=256 by 16"));
    }

    #[test]
    fn mxf8f6f4_descriptor_rejects_transposed_packed_e2m1() {
        let descriptor = mxf8f6f4_descriptor(128, 16) | (5 << 7) | (1 << 15);
        let error = decode_raw_tcgen_mxf8f6f4_instruction_descriptor(
            descriptor,
            1,
            RawTcgenMatrixDescriptorLayout::Sm100,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("transpose A requires an 8-bit operand"));
    }

    #[test]
    fn mxf8f6f4_mn_major_offset_matches_ptx_canonical_layout() {
        let source = RuntimeBuffer::Shared {
            allocations: Arc::new(Vec::new()),
            byte_offset: 0,
            byte_len: 4096,
            backing_byte_len: 4096,
            virtual_base: 0,
        };
        let descriptor = RawTcgenMatrixDescriptor {
            absolute_leading_address: false,
            start_address: 0,
            leading_byte_offset: 0,
            stride_byte_offset: 1024,
            swizzle_bits: 3,
            swizzle_atom_bytes: 16,
            swizzle_xor_shift: 3,
        };

        assert_eq!(
            raw_tcgen05_8bit_matrix_byte_offset(&source, descriptor, 0, 0, true).unwrap(),
            0
        );
        assert_eq!(
            raw_tcgen05_8bit_matrix_byte_offset(&source, descriptor, 0, 1, true).unwrap(),
            144
        );
        assert_eq!(
            raw_tcgen05_8bit_matrix_byte_offset(&source, descriptor, 127, 31, true).unwrap(),
            3983
        );
    }

    #[test]
    fn mxf4_family_descriptor_requires_the_scale_type_its_kind_spells() {
        let nvf4 = decode_raw_tcgen_mxf4_instruction_descriptor(
            mxf4_family_descriptor(false, 0),
            RawTcgenMxf4ScaleSpelling::Ue4m3Vec4x,
            RawTcgenMatrixDescriptorLayout::Sm100,
        )
        .unwrap();
        assert_eq!((nvf4.m, nvf4.n, nvf4.sfa_id, nvf4.sfb_id), (128, 8, 0, 0));

        // The same bits are an `.kind::mxf4` descriptor with the wrong scale type.
        let error = decode_raw_tcgen_mxf4_instruction_descriptor(
            mxf4_family_descriptor(false, 0),
            RawTcgenMxf4ScaleSpelling::Ue8m0Vec2x,
            RawTcgenMatrixDescriptorLayout::Sm100,
        )
        .unwrap_err();
        assert!(error.to_string().contains("must encode UE8M0 scales"));

        let error = decode_raw_tcgen_mxf4_instruction_descriptor(
            mxf4_family_descriptor(true, 0),
            RawTcgenMxf4ScaleSpelling::Ue4m3Vec4x,
            RawTcgenMatrixDescriptorLayout::Sm100,
        )
        .unwrap_err();
        assert!(error.to_string().contains("must encode UE4M3 scales"));

        let nvf4_ue8m0 = decode_raw_tcgen_mxf4_instruction_descriptor(
            mxf4_family_descriptor(true, 0),
            RawTcgenMxf4ScaleSpelling::Ue8m0Vec4x,
            RawTcgenMatrixDescriptorLayout::Sm100,
        )
        .unwrap();
        assert_eq!(
            (
                nvf4_ue8m0.m,
                nvf4_ue8m0.n,
                nvf4_ue8m0.sfa_id,
                nvf4_ue8m0.sfb_id
            ),
            (128, 8, 0, 0)
        );

        assert_eq!(
            raw_tcgen05_mma_block_mxf4_shape(
                mxf4_family_descriptor(false, 0),
                RawTcgenMxf4ScaleSpelling::Ue4m3Vec4x,
                1,
                RawTcgenMatrixDescriptorLayout::Sm100,
                false,
            )
            .unwrap(),
            (128, 8, 64)
        );
        assert_eq!(
            raw_tcgen05_mma_block_mxf4_shape(
                mxf4_family_descriptor(true, 0),
                RawTcgenMxf4ScaleSpelling::Ue8m0Vec4x,
                1,
                RawTcgenMatrixDescriptorLayout::Sm100,
                false,
            )
            .unwrap(),
            (128, 8, 64)
        );
    }

    #[test]
    fn fp4_new_scale_types_require_sm107_and_reject_reserved_encoding() {
        for (encoding, scale) in [
            (0, RawTcgenMxf4ScaleSpelling::Ue4m3Vec2x),
            (2, RawTcgenMxf4ScaleSpelling::Ue5m3Vec2x),
            (2, RawTcgenMxf4ScaleSpelling::Ue5m3Vec4x),
        ] {
            let descriptor = mxf4_family_descriptor(false, 0) | (encoding << 23);
            let error = decode_raw_tcgen_mxf4_instruction_descriptor(
                descriptor,
                scale,
                RawTcgenMatrixDescriptorLayout::Sm100,
            )
            .unwrap_err();
            assert!(error.to_string().contains("require SM107"));
            decode_raw_tcgen_mxf4_instruction_descriptor(
                descriptor | (1 << 12),
                scale,
                RawTcgenMatrixDescriptorLayout::Sm107,
            )
            .unwrap();
        }
        let reserved = mxf4_family_descriptor(false, 0) | (3 << 23) | (1 << 12);
        assert!(decode_raw_tcgen_mxf4_instruction_descriptor(
            reserved,
            super::raw_tcgen05_mxf4nvf4_vec4x_scale(reserved),
            RawTcgenMatrixDescriptorLayout::Sm107,
        )
        .is_err());
    }

    #[test]
    fn scale_vec_4x_rejects_the_nonzero_scale_factor_ids_2x_allows() {
        // `.scale_vec::2X` selects one byte pair, so SFA ID 2 is legal there.
        decode_raw_tcgen_mxf4_instruction_descriptor(
            mxf4_family_descriptor(true, 2),
            RawTcgenMxf4ScaleSpelling::Ue8m0Vec2x,
            RawTcgenMatrixDescriptorLayout::Sm100,
        )
        .unwrap();

        // `.scale_vec::4X` spends all four bytes, so the ID must be 0.
        let error = decode_raw_tcgen_mxf4_instruction_descriptor(
            mxf4_family_descriptor(false, 2),
            RawTcgenMxf4ScaleSpelling::Ue4m3Vec4x,
            RawTcgenMatrixDescriptorLayout::Sm100,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("scale_vec::4X requires SFA/SFB IDs 0"));

        let error = decode_raw_tcgen_mxf4_instruction_descriptor(
            mxf4_family_descriptor(true, 2),
            RawTcgenMxf4ScaleSpelling::Ue8m0Vec4x,
            RawTcgenMatrixDescriptorLayout::Sm100,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("scale_vec::4X requires SFA/SFB IDs 0"));
    }

    #[test]
    fn ue4m3_scale_decodes_as_unsigned_e4m3_and_rejects_a_set_padding_bit() {
        // Exponent 7 mantissa 0 is 1.0; exponent 7 mantissa 4 is 1.5.
        assert_eq!(raw_tcgen05_decode_ue4m3_scale(0x38).unwrap(), 1.0_f32);
        assert_eq!(raw_tcgen05_decode_ue4m3_scale(0x3c).unwrap(), 1.5_f32);
        assert_eq!(raw_tcgen05_decode_ue4m3_scale(0x00).unwrap(), 0.0_f32);
        assert!(raw_tcgen05_decode_ue4m3_scale(0x7f).unwrap().is_nan());

        let error = raw_tcgen05_decode_ue4m3_scale(0xb8).unwrap_err();
        assert!(error.to_string().contains("sets the padding MSB"));
    }

    #[test]
    fn fp4_k_encoding_and_sparsity_version_follow_the_authored_architecture() {
        use RawTcgenMatrixDescriptorLayout::{Sm100, Sm103, Sm107};
        for (layout, k, k_bits, id) in [
            (Sm100, 64, 0, 0),
            (Sm103, 96, 1 << 31, 0),
            (Sm103, 96, 1 << 31, 2),
            (Sm107, 128, 1 << 3, 0),
        ] {
            let bits =
                mxf4_family_descriptor(false, id) | k_bits | (u32::from(layout == Sm107) << 12);
            let decode = |value, arch| {
                decode_raw_tcgen_mxf4_instruction_descriptor(
                    value,
                    RawTcgenMxf4ScaleSpelling::Ue4m3Vec4x,
                    arch,
                )
            };
            let block = decode(bits, layout).unwrap();
            assert_eq!((block.k, block.block_elements), (k, 16));
            if k == 64 {
                let vector = super::decode_raw_tcgen_mxf4_instruction_descriptor_for_cta_group(
                    bits,
                    RawTcgenMxf4ScaleSpelling::Ue4m3Vec4x,
                    1,
                    layout,
                    true,
                )
                .unwrap();
                let fields = |d: super::RawTcgenMxf4InstructionDescriptor| {
                    (d.sfa_lanes, d.m, d.n, d.k, d.sfa_id, d.sfb_id,
                     d.negate_a, d.negate_b, d.block_elements)
                };
                assert_eq!(fields(block), fields(vector));
            }
            assert!(decode(bits ^ (1 << 12), layout).is_err());
            if k != 64 {
                assert!(decode(bits, Sm100).is_err());
            }
            if k == 128 {
                assert!(decode(bits & !(1 << 12), Sm103).is_err());
            }
        }
        let invalid = mxf4_family_descriptor(false, 0) | (1 << 12) | (1 << 31) | (1 << 3);
        assert!(decode_raw_tcgen_mxf4_instruction_descriptor(
            invalid,
            RawTcgenMxf4ScaleSpelling::Ue4m3Vec4x,
            Sm107,
        )
        .is_err());
    }

    #[test]
    fn fp4_absolute_ldo_maps_a_k96_straddle_and_rejects_other_consumers() {
        use RawTcgenMatrixDescriptorLayout::{Sm100, Sm103, Sm107};
        let source = RuntimeBuffer::Shared {
            allocations: Arc::new(Vec::new()),
            byte_offset: 0,
            byte_len: 65536,
            backing_byte_len: 65536,
            virtual_base: 0,
        };
        // First chunk begins at byte 96; its last 32 bytes precede a separate
        // second chunk at 32 KiB. Check both halves across an eight-row boundary.
        let bits = 0x4010404000000000_u64 | 6 | ((32768_u64 >> 4) << 16);
        let descriptor =
            super::decode_raw_tcgen_packed_matrix_descriptor(bits, Sm103, 48, false).unwrap();
        for row in [0, 1, 7, 8, 127] {
            for column in 0..48 {
                let chunk = if column < 32 {
                    96 + column
                } else {
                    32768 + column - 32
                };
                let linear = chunk + (row % 8) * 128 + (row / 8) * 1024;
                let expected = linear ^ (((linear >> 7) & 7) << 4);
                assert_eq!(
                    raw_tcgen05_shared_byte_offset(&source, descriptor, row, column, 1).unwrap(),
                    expected
                );
            }
        }
        assert!(decode_raw_tcgen_matrix_descriptor_for_layout(bits, Sm103).is_err());
        for (arch, k) in [(Sm100, 96), (Sm103, 64), (Sm107, 128)] {
            assert!(super::decode_raw_tcgen_packed_matrix_descriptor(bits, arch, k / 2, false).is_err());
        }
        for invalid in [
            bits ^ (6_u64 << 61),
            bits | (1 << 49),
            bits | (1 << 53),
            bits | (1 << 16),
        ] {
            assert!(super::decode_raw_tcgen_packed_matrix_descriptor(invalid, Sm103, 48, false).is_err());
        }
    }

    #[test]
    fn fp4_scale_word_continuations_match_ptx_byte_layout_and_footprints() {
        let topology = crate::LaunchTopology::new(1, 2, 1).unwrap();
        let context = crate::WarpContext::from_topology(topology, 0);
        // SF ID 2 at K96 consumes byte 2/3, then the four bytes in the next
        // column block. B's two CTAs use the joint N=256 and stride eight.
        let accesses =
            super::raw_tcgen05_cta2_scale_accesses(&context, 16, 2, 6, 128, true, 0, 32).unwrap();
        assert_eq!(accesses.len(), 512);
        for cta in 0..2 {
            for row in 0..128 {
                let pair = &accesses[(cta * 128 + row) * 2..][..2];
                let lane = (row % 32) as i64;
                let column = 16 + (cta * 128 + row) as i64 / 32;
                assert_eq!(pair[0], (0, 0, Some(cta), lane, 2, column, 2));
                assert_eq!(pair[1], (0, 0, Some(cta), lane, 0, column + 8, 4));
            }
        }
        for (id, count, stride) in [(0, 4, 4), (0, 6, 4), (2, 6, 4), (0, 8, 8)] {
            for vector in 0..count {
                let expected = (16 + ((id + vector) / 4 * stride) as u32, (id + vector) % 4);
                assert_eq!(
                    super::raw_tcgen05_scale_chunk(16, id, vector, stride * 32, 32,).unwrap(),
                    expected
                );
            }
        }
        assert!(super::raw_tcgen05_scale_chunk(16, 0, 4, 0, 32,).is_err());
    }

    /// Encode one dense `kind::f8f6f4` instruction descriptor the way the
    /// frontend does, so the decoder is checked against a real bit layout.
    fn f8f6f4_descriptor(d_format: u32, a_format: u32, b_format: u32, m: u32, n: u32) -> u32 {
        (d_format << 4) | (a_format << 7) | (b_format << 10) | ((n >> 3) << 17) | ((m >> 4) << 24)
    }

    #[test]
    fn b16_half_and_ws_descriptors_validate_the_actual_geometry() {
        let half = f8f6f4_descriptor(0, 0, 0, 128, 64);
        assert!(super::decode_raw_tcgen_b16_instruction_descriptor(
            half, false, false, 1, false, false
        )
        .is_ok());
        assert!(super::decode_raw_tcgen_b16_instruction_descriptor(
            half | (1 << 7),
            true,
            false,
            1,
            false,
            false,
        )
        .is_err());
        for sparse in [false, true] {
            let mixed = f8f6f4_descriptor(1, 1, 0, 128, 16) | if sparse { 4 } else { 0 };
            for a_bf16 in [false, true] {
                let decode = |bits| {
                    super::decode_raw_tcgen_b16_instruction_descriptor(
                        bits, a_bf16, false, 1, false, sparse,
                    )
                    .unwrap_err()
                };
                assert!(decode(mixed)
                    .to_string()
                    .contains("requires matching F16/BF16"));
                assert!(!matches!(
                    decode(mixed).kind(),
                    crate::EngineErrorKind::AnalysisIncomplete { .. }
                ));
                assert!(!matches!(
                    decode(mixed | (1 << 6)).kind(),
                    crate::EngineErrorKind::AnalysisIncomplete { .. }
                ));
            }
        }
        for sparse in [false, true] {
            let flags = if sparse { 4 } else { 0 };
            for n in [8, 16, 24, 256] {
                for (cta_group, m) in [(1, 64), (1, 128), (2, 128), (2, 256)] {
                    let descriptor = f8f6f4_descriptor(1, 0, 0, m, n) | flags;
                    assert_eq!(
                        super::decode_raw_tcgen_b16_instruction_descriptor(
                            descriptor, false, false, cta_group, false, sparse,
                        )
                        .is_ok(),
                        cta_group == 1 || n % 16 == 0,
                    );
                    let tf32 = f8f6f4_descriptor(1, 2, 2, m, n) | flags;
                    assert_eq!(
                        super::decode_raw_tcgen_tf32_instruction_descriptor(
                            tf32,
                            cta_group as usize,
                            false,
                            sparse,
                        )
                        .is_ok(),
                        cta_group == 1 || n % 16 == 0,
                    );
                }
                let ti16 = f8f6f4_descriptor(2, 3, 3, 128, n) | flags;
                assert_eq!(
                    super::raw_tcgen05_integer_shape(
                        super::RawTcgenIntegerKind::Ti16,
                        ti16,
                        1,
                        false,
                        sparse,
                    )
                    .is_ok(),
                    n % 16 == 0,
                );
            }
            for m in [32, 64, 128] {
                for n in [64, 128, 256] {
                    let ws = f8f6f4_descriptor(1, 0, 0, m, n) | (1 << 30) | flags;
                    assert_eq!(
                        super::decode_raw_tcgen_b16_instruction_descriptor(
                            ws, false, false, 1, true, sparse,
                        )
                        .is_ok(),
                        !sparse || n != 256,
                    );
                    let tf32 = f8f6f4_descriptor(1, 2, 2, m, n) | (1 << 30) | flags;
                    assert_eq!(
                        super::decode_raw_tcgen_tf32_instruction_descriptor(tf32, 1, true, sparse)
                            .is_ok(),
                        !sparse || n != 256,
                    );
                }
            }
        }
        let narrow = f8f6f4_descriptor(1, 0, 0, 64, 8);
        assert!(super::decode_raw_tcgen_b16_instruction_descriptor(
            narrow, false, false, 1, false, false
        )
        .is_ok());
        assert!(super::decode_raw_tcgen_b16_instruction_descriptor(
            narrow, false, false, 1, true, false
        )
        .is_ok());
    }

    #[test]
    fn f8f6f4_ws_geometry_and_diagnostics_match_ptx_table_48() {
        for sparse in [false, true] {
            for m in [16, 32, 64, 128, 256] {
                for n in (8..=264).step_by(8) {
                    let descriptor = f8f6f4_descriptor(1, 0, 0, m, n)
                        | if sparse { 4 } else { 0 };
                    let result = decode_raw_tcgen_f8f6f4_instruction_descriptor(
                        descriptor, RawTcgenNarrowFormat::E4M3,
                        RawTcgenNarrowFormat::E4M3, false, 1, false, true, sparse,
                    );
                    let valid = matches!(m, 32 | 64 | 128)
                        && (matches!(n, 64 | 128) || (!sparse && n == 256));
                    assert_eq!(result.is_ok(), valid, "M={m}, N={n}, sparse={sparse}");
                    if let Err(error) = result {
                        let message = error.to_string();
                        let n_shapes = if sparse { "{64, 128}" } else { "{64, 128, 256}" };
                        assert!(message.contains(".ws cta_group=1, M in {32, 64, 128}"));
                        assert!(message.contains(&format!("N in {n_shapes}")));
                        assert!(message.contains(&format!("got M={m}, N={n}")));
                        assert!(!message.contains("by 8"));
                    }
                }
            }
        }
    }

    #[test]
    fn f8f6f4_descriptor_decoding_is_pinned_to_the_specialized_operand_types() {
        // The V specialization names the operand types; the descriptor must
        // agree with it, or the call is a form the engine never modeled.
        let descriptor = f8f6f4_descriptor(1, 1, 0, 128, 16);
        let decoded = decode_raw_tcgen_f8f6f4_instruction_descriptor(
            descriptor,
            RawTcgenNarrowFormat::E5M2,
            RawTcgenNarrowFormat::E4M3,
            false,
            1,
            false,
            false,
            false,
        )
        .unwrap();
        assert_eq!((decoded.m, decoded.n), (128, 16));

        for (a_format, b_format, d_f16) in [
            (
                RawTcgenNarrowFormat::E4M3,
                RawTcgenNarrowFormat::E4M3,
                false,
            ),
            (
                RawTcgenNarrowFormat::E5M2,
                RawTcgenNarrowFormat::E5M2,
                false,
            ),
            (RawTcgenNarrowFormat::E5M2, RawTcgenNarrowFormat::E4M3, true),
        ] {
            let error = decode_raw_tcgen_f8f6f4_instruction_descriptor(
                descriptor, a_format, b_format, d_f16, 1, false, false, false,
            )
            .unwrap_err();
            assert!(
                format!("{error}").contains("must encode dense"),
                "unexpected error {error}"
            );
        }

        // A float16 destination is format 0 in bits 4-5, float32 is format 1.
        let f16_descriptor = f8f6f4_descriptor(0, 0, 1, 64, 8);
        let decoded = decode_raw_tcgen_f8f6f4_instruction_descriptor(
            f16_descriptor,
            RawTcgenNarrowFormat::E4M3,
            RawTcgenNarrowFormat::E5M2,
            true,
            1,
            false,
            false,
            false,
        )
        .unwrap();
        assert_eq!((decoded.m, decoded.n), (64, 8));
        assert!(decode_raw_tcgen_f8f6f4_instruction_descriptor(
            f16_descriptor,
            RawTcgenNarrowFormat::E4M3,
            RawTcgenNarrowFormat::E5M2,
            false,
            1,
            false,
            false,
            false,
        )
        .is_err());
    }

    #[test]
    fn f8f6f4_cta2_descriptor_distinguishes_sm100_and_sm107_k_widths() {
        // Canonical bmm_fp8_rubin tactic 1: M=256, N=128, K=64, F32 D,
        // E5M2 A/B, MN-major B. SM107's K=64 selector is bit 29.
        let descriptor = f8f6f4_descriptor(1, 1, 1, 256, 128) | (1 << 16) | (1 << 29);
        let decoded = decode_raw_tcgen_f8f6f4_instruction_descriptor(
            descriptor,
            RawTcgenNarrowFormat::E5M2,
            RawTcgenNarrowFormat::E5M2,
            false,
            2,
            true,
            false,
            false,
        )
        .unwrap();
        assert_eq!((decoded.m, decoded.n, decoded.k), (256, 128, 64));
        assert!(!decoded.transpose_a);
        assert!(decoded.transpose_b);

        let sm100_error = decode_raw_tcgen_f8f6f4_instruction_descriptor(
            descriptor,
            RawTcgenNarrowFormat::E5M2,
            RawTcgenNarrowFormat::E5M2,
            false,
            2,
            false,
            false,
            false,
        )
        .unwrap_err();
        assert!(sm100_error.to_string().contains("must encode dense"));

        let k32_descriptor = descriptor & !(1 << 29);
        for supports_k64 in [false, true] {
            let decoded = decode_raw_tcgen_f8f6f4_instruction_descriptor(
                k32_descriptor,
                RawTcgenNarrowFormat::E5M2,
                RawTcgenNarrowFormat::E5M2,
                false,
                2,
                supports_k64,
                false,
                false,
            )
            .unwrap();
            assert_eq!(decoded.k, 32);
        }

        let f16 = f8f6f4_descriptor(0, 1, 1, 256, 128) | (1 << 29);
        let decoded = decode_raw_tcgen_f8f6f4_instruction_descriptor(
            f16,
            RawTcgenNarrowFormat::E5M2,
            RawTcgenNarrowFormat::E5M2,
            true,
            2,
            true,
            false,
            false,
        )
        .unwrap();
        assert_eq!((decoded.m, decoded.n, decoded.k), (256, 128, 64));
    }

    #[test]
    fn f8f6f4_cta1_mn_major_b_requires_sixteen_column_granularity() {
        for n in [8, 16, 24, 256] {
            let descriptor = f8f6f4_descriptor(1, 1, 0, 128, n) | (1 << 16);
            let result = decode_raw_tcgen_f8f6f4_instruction_descriptor(
                descriptor,
                RawTcgenNarrowFormat::E5M2,
                RawTcgenNarrowFormat::E4M3,
                false,
                1,
                false,
                false,
                false,
            );
            assert_eq!(result.is_ok(), n % 16 == 0);
        }
    }

    #[test]
    fn f8f6f4_cta2_n_granularity_follows_b_major() {
        let kmajor_n16 = f8f6f4_descriptor(1, 1, 1, 256, 16);
        let decoded = decode_raw_tcgen_f8f6f4_instruction_descriptor(
            kmajor_n16,
            RawTcgenNarrowFormat::E5M2,
            RawTcgenNarrowFormat::E5M2,
            false,
            2,
            false,
            false,
            false,
        )
        .unwrap();
        assert_eq!((decoded.m, decoded.n), (256, 16));
        assert!(!decoded.transpose_b);

        let mnmajor_n16 = kmajor_n16 | (1 << 16);
        let error = decode_raw_tcgen_f8f6f4_instruction_descriptor(
            mnmajor_n16,
            RawTcgenNarrowFormat::E5M2,
            RawTcgenNarrowFormat::E5M2,
            false,
            2,
            false,
            false,
            false,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("N in 32..=256 by 32 for MN-major B"),
            "unexpected error {error}"
        );
    }

    #[test]
    fn sm107_matrix_descriptor_accepts_bit14_but_sm100_and_bit15_fail_closed() {
        // CUTLASS `mma_sm100_desc.hpp` widens start-address and LDO from 14
        // to 15 bits only under CUTLASS_ARCH_MMA_SM107[AF]_ENABLED.
        let base = (1_u64 << 46) | (2_u64 << 61);
        let extended = base | (1_u64 << 14) | (1_u64 << 30);
        let decoded = decode_raw_tcgen_matrix_descriptor_for_layout(
            extended,
            RawTcgenMatrixDescriptorLayout::Sm107,
        )
        .unwrap();

        assert_eq!(decoded.start_address, 1 << 18);
        assert_eq!(decoded.leading_byte_offset, 1 << 18);
        assert!(decode_raw_tcgen_matrix_descriptor_for_layout(
            extended,
            RawTcgenMatrixDescriptorLayout::Sm100,
        )
        .is_err());
        for reserved_bit in [15, 31] {
            assert!(decode_raw_tcgen_matrix_descriptor_for_layout(
                base | (1_u64 << reserved_bit),
                RawTcgenMatrixDescriptorLayout::Sm107,
            )
            .is_err());
        }
    }

    #[test]
    fn the_f16_destination_codec_uses_the_low_half_and_zeroes_the_upper_half() {
        // Measured on B200: the MMA writes the whole 32-bit cell, so a
        // pre-existing upper half does not survive, and the addend it reads
        // back is the low half alone.
        let stored = RawMmaCellDtype::F16.encode(1032.0);
        assert_eq!(stored, [0x08, 0x64, 0x00, 0x00]);
        assert_eq!(
            RawMmaCellDtype::F16.decode([0x00, 0x64, 0xEF, 0xBE]),
            1024.0
        );
        assert_eq!(
            RawMmaCellDtype::F32.decode(1032.0_f32.to_le_bytes()),
            1032.0
        );

        // Rounding happens once, on store, and is round-to-nearest-even.
        assert_eq!(
            RawMmaCellDtype::F16.decode(RawMmaCellDtype::F16.encode(1031.75)),
            1032.0
        );
        assert_eq!(
            RawMmaCellDtype::F16.decode(RawMmaCellDtype::F16.encode(1024.25)),
            1024.0
        );
    }

    #[test]
    fn raw_ldst_accepts_absolute_tmem_lane_addresses_for_each_warp() {
        let topology = LaunchTopology::new(1, 1, 4).unwrap();
        let contexts = topology.warp_contexts().collect::<Vec<_>>();

        for (warp, context) in contexts.iter().enumerate() {
            let address = u32::try_from(warp * 32).unwrap() << 16 | 7;
            let location = raw_tcgen05_ldst_location(
                context,
                address,
                0,
                0,
                RawTcgenLdstShape::Shape32x32b,
                false,
                2,
                3,
            )
            .unwrap();
            assert_eq!(location, (warp * 32 + 3, 9));
        }
    }

    #[test]
    fn raw_ldst_normalizes_a_warp_relative_tmem_lane_address() {
        let topology = LaunchTopology::new(1, 1, 4).unwrap();
        let contexts = topology.warp_contexts().collect::<Vec<_>>();

        for (warp, context) in contexts.iter().enumerate() {
            let location = raw_tcgen05_ldst_location(
                context,
                7_u32 << 16,
                0,
                0,
                RawTcgenLdstShape::Shape32x32b,
                false,
                2,
                3,
            )
            .unwrap();
            assert_eq!(location, (warp * 32 + 10, 2));
        }
    }

    #[test]
    fn raw_ldst_rejects_an_absolute_lane_address_owned_by_another_warp() {
        let topology = LaunchTopology::new(1, 1, 4).unwrap();
        let warp_two = topology.warp_contexts().nth(2).unwrap();

        let error = raw_tcgen05_ldst_location(
            &warp_two,
            32_u32 << 16,
            0,
            0,
            RawTcgenLdstShape::Shape32x32b,
            false,
            0,
            0,
        )
        .unwrap_err();

        assert!(error.to_string().contains("accessible TMEM lanes 64..96"));
    }

    #[test]
    fn explicit_issuing_lane_controls_access_view_validation() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let lane_seven = WarpMask::from_lanes([7]).unwrap();
        let restricted = RuntimeBuffer::AccessView {
            buffer: Arc::new(RuntimeBuffer::Shared {
                allocations: Arc::new(Vec::new()),
                byte_offset: 0,
                byte_len: 16,
                backing_byte_len: 16,
                virtual_base: 0,
            }),
            readable_lanes: lane_seven,
            writable_lanes: lane_seven,
        };

        validate_raw_tcgen05_issuing_lane(&context, 7, &crate::DiagnosticLabel::new("test mma"))
            .unwrap();
        raw_tcgen05_shared_source(&context, &[&restricted], 0, 7).unwrap();
        assert!(raw_tcgen05_shared_source(&context, &[&restricted], 0, 0).is_err());
    }

    #[test]
    fn tile_gemm_bf16_snapshot_decode_reverses_inner_swizzle() {
        let rows = 8;
        let columns = 64;
        let logical = (0..rows * columns)
            .map(|value| (value % 16) as f32)
            .collect::<Vec<_>>();
        let mut snapshot = vec![0_u8; logical.len() * 2];
        for (unswizzled, value) in logical.iter().enumerate() {
            let quotient = unswizzled >> 3;
            let physical = ((quotient ^ ((quotient & 56) >> 3)) << 3) | (unswizzled & 7);
            snapshot[physical * 2..physical * 2 + 2]
                .copy_from_slice(&crate::f32_to_bf16_bits(*value).to_le_bytes());
        }

        assert_eq!(
            decode_tile_gemm_bf16_snapshot(
                &snapshot,
                rows,
                columns,
                TileGemmBf16OperandLayout::new(64, 3, 56, 3),
            )
            .unwrap(),
            logical
        );
    }

    #[test]
    fn tile_gemm_bf16_snapshot_decode_reorders_column_atoms() {
        let rows = 8;
        let columns = 128;
        let mut snapshot = vec![0_u8; rows * columns * 2];
        let mut logical = vec![0.0_f32; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                let unswizzled = (column / 64) * rows * 64 + row * 64 + column % 64;
                let quotient = unswizzled >> 3;
                let physical = ((quotient ^ ((quotient & 56) >> 3)) << 3) | (unswizzled & 7);
                let value = ((row * columns + column) % 16) as f32;
                logical[row * columns + column] = value;
                snapshot[physical * 2..physical * 2 + 2]
                    .copy_from_slice(&crate::f32_to_bf16_bits(value).to_le_bytes());
            }
        }

        let decoded = decode_tile_gemm_bf16_snapshot(
            &snapshot,
            rows,
            columns,
            TileGemmBf16OperandLayout::new(64, 3, 56, 3),
        )
        .unwrap();
        assert_eq!(decoded, logical);
    }

    #[test]
    fn tile_gemm_fp8_swizzle_maps_a_non_power_of_two_row_count_bijectively() {
        let mut physical = tile_gemm_operand_physical_elements(
            120,
            128,
            TileGemmOperandLayout::new(128, 4, 56, 3),
        )
        .unwrap();

        assert_eq!(physical.len(), 120 * 128);
        physical.sort_unstable();
        assert_eq!(physical, (0..120 * 128).collect::<Vec<_>>());
    }

    #[test]
    fn raw_cp_4x256b_places_each_source_row_in_one_warp_lane() {
        assert_eq!(
            raw_tcgen05_cp_destination_lanes(4, 0).unwrap().as_slice(),
            [0]
        );
        assert_eq!(
            raw_tcgen05_cp_destination_lanes(4, 1).unwrap().as_slice(),
            [32]
        );
        assert_eq!(
            raw_tcgen05_cp_destination_lanes(4, 2).unwrap().as_slice(),
            [64]
        );
        assert_eq!(
            raw_tcgen05_cp_destination_lanes(4, 3).unwrap().as_slice(),
            [96]
        );
    }

    #[test]
    fn raw_cp_b6_decompression_preserves_the_six_bit_codes() {
        let values = [1_u32, 2, 31, 63];
        let bits = values[0] | (values[1] << 6) | (values[2] << 12) | (values[3] << 18);
        let packed = [bits as u8, (bits >> 8) as u8, (bits >> 16) as u8];

        assert_eq!(raw_tcgen05_unpack_b6(packed), [1, 2, 31, 63]);
    }

    #[test]
    fn raw_tcgen_tf32_operands_decode_storage_bits_without_rne_conversion() {
        let bits = 1.000_6_f32.to_bits();
        let decoded = raw_tcgen05_tf32_payload_to_f32(bits);

        assert_eq!(decoded.to_bits(), bits & 0xffff_e000);
        assert_ne!(decoded, crate::f32_to_tf32(f32::from_bits(bits)));
    }

    #[test]
    fn shared_descriptor_swizzles_the_absolute_virtual_address() {
        let source = RuntimeBuffer::Shared {
            allocations: Arc::new(Vec::new()),
            byte_offset: 0,
            byte_len: 512,
            backing_byte_len: 512,
            virtual_base: 128,
        };
        let descriptor = RawTcgenMatrixDescriptor {
            absolute_leading_address: false,
            start_address: 128,
            leading_byte_offset: 0,
            stride_byte_offset: 512,
            swizzle_bits: 2,
            swizzle_atom_bytes: 16,
            swizzle_xor_shift: 3,
        };

        assert_eq!(
            raw_tcgen05_shared_byte_offset(&source, descriptor, 0, 0, 2).unwrap(),
            16
        );
        assert_eq!(
            raw_tcgen05_shared_byte_offset(&source, descriptor, 0, 16, 2).unwrap(),
            0
        );
        assert_eq!(
            raw_tcgen05_shared_byte_offset(&source, descriptor, 2, 0, 2).unwrap(),
            160
        );

        let unified_source = RuntimeBuffer::Shared {
            allocations: Arc::new(Vec::new()),
            byte_offset: 0,
            byte_len: 656,
            backing_byte_len: 656,
            virtual_base: 0,
        };
        let crossing_descriptor = RawTcgenMatrixDescriptor {
            start_address: 144,
            ..descriptor
        };
        assert_eq!(
            raw_tcgen05_shared_byte_offset(&unified_source, crossing_descriptor, 0, 0, 2,).unwrap(),
            128
        );
    }

    #[test]
    fn shared_descriptor_resolution_rejects_distinct_overlapping_backings() {
        let first = RuntimeBuffer::Shared {
            allocations: Arc::new(Vec::new()),
            byte_offset: 0,
            byte_len: 16,
            backing_byte_len: 16,
            virtual_base: 0,
        };
        let second = RuntimeBuffer::Shared {
            allocations: Arc::new(Vec::new()),
            byte_offset: 0,
            byte_len: 16,
            backing_byte_len: 16,
            virtual_base: 0,
        };
        let context = crate::WarpContext::from_topology(LaunchTopology::new(1, 1, 1).unwrap(), 0);
        let error = match raw_tcgen05_shared_source(&context, &[&first, &second], 0, 0) {
            Ok(_) => panic!("distinct overlapping shared backings must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("ambiguously names multiple"));
    }

    #[test]
    fn shared_descriptor_resolution_canonicalizes_aliases_of_one_backing() {
        let allocations = Arc::new(Vec::new());
        let whole = RuntimeBuffer::Shared {
            allocations: Arc::clone(&allocations),
            byte_offset: 0,
            byte_len: 16,
            backing_byte_len: 16,
            virtual_base: 0,
        };
        let alias = RuntimeBuffer::Shared {
            allocations,
            byte_offset: 4,
            byte_len: 8,
            backing_byte_len: 16,
            virtual_base: 4,
        };
        let context = crate::WarpContext::from_topology(LaunchTopology::new(1, 1, 1).unwrap(), 0);
        raw_tcgen05_shared_source(&context, &[&whole, &alias], 4, 0).unwrap();
    }

    #[test]
    fn raw_mma_uses_a_fixed_increasing_k_reduction() {
        let large = 16_777_216.0_f32;
        let output =
            mma_f32_abt_increasing_k(1, 1, 3, &[1.0, 1.0, 1.0], &[large, 1.0, -large], None)
                .unwrap();

        assert_eq!(output, vec![0.0]);
        assert_eq!((large + -large) + 1.0, 1.0);
    }

    #[test]
    fn raw_mma_starts_the_fma_chain_from_scaled_input_d() {
        let large = 16_777_216.0_f32;
        let input_d = [2.0_f32];
        let output = mma_f32_abt_increasing_k(
            1,
            1,
            2,
            &[1.0, 1.0],
            &[large, -large],
            Some((&input_d, 0.5)),
        )
        .unwrap();

        assert_eq!(output, vec![0.0]);
        let product_then_d = (large + -large) + input_d[0] * 0.5;
        assert_eq!(product_then_d, 1.0);
    }

    #[test]
    fn raw_mma_vectorized_columns_are_bitwise_equal_to_scalar_dots() {
        let (m, n, k) = (3_usize, 19_usize, 7_usize);
        let a_values = (0..m * k)
            .map(|index| ((index as f32 - 9.0) * 0.3125).sin())
            .collect::<Vec<_>>();
        let b_values = (0..n * k)
            .map(|index| ((index as f32 + 3.0) * -0.21875).cos())
            .collect::<Vec<_>>();
        let input_d = (0..m * n)
            .map(|index| (index as f32 - 6.0) * 0.0625)
            .collect::<Vec<_>>();
        let scale = -0.75_f32;

        let output =
            mma_f32_abt_increasing_k(m, n, k, &a_values, &b_values, Some((&input_d, scale)))
                .unwrap();
        let mut scalar = vec![0.0_f32; m * n];
        for row in 0..m {
            for col in 0..n {
                scalar[row * n + col] = super::mma_f32_dot_increasing_k(
                    &a_values[row * k..(row + 1) * k],
                    &b_values[col * k..(col + 1) * k],
                    input_d[row * n + col] * scale,
                )
                .unwrap();
            }
        }

        assert_eq!(
            output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            scalar
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn raw_mma_rejects_malformed_matrix_payloads() {
        let error = mma_f32_abt_increasing_k(2, 1, 2, &[0.0; 3], &[0.0; 2], None).unwrap_err();
        assert!(error
            .to_string()
            .contains("A matrix has 3 values, expected 4"));

        let error =
            mma_f32_abt_increasing_k(1, 2, 1, &[0.0], &[0.0; 2], Some((&[0.0], 1.0))).unwrap_err();
        assert!(error
            .to_string()
            .contains("input D matrix has 1 values, expected 2"));
    }
}

#[test]
fn packed_fp4_atoms_preserve_every_nibble_and_signed_zero() {
    const EXPECTED: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    for packed in 0_u16..=255 {
        let mut bytes = [0xff; 16];
        for (i, byte) in bytes[..8].iter_mut().enumerate() {
            *byte = (packed as u8).wrapping_add(i as u8);
        }
        let decoded = RawTcgenNarrowFormat::E2M1.decode_shared_atom(bytes);
        for (i, value) in decoded.iter().enumerate() {
            let byte = (packed as u8).wrapping_add((i / 2) as u8);
            let code = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
            assert_eq!(value.to_bits(), EXPECTED[usize::from(code)].to_bits());
        }
    }
}
