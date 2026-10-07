//! Exact integer TCGEN arithmetic; address layouts remain owned by tcgen_ops.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RawTcgenIntegerKind {
    Ti16,
    I8,
}

impl RawTcgenIntegerKind {
    pub(crate) fn packed_k(self) -> usize {
        match self {
            Self::Ti16 => 16,
            Self::I8 => 32,
        }
    }

    fn decode(self, bits: u16, format: u32, negate: bool) -> Result<i32, EngineError> {
        match self {
            Self::Ti16 => decode_ti16(bits, negate),
            Self::I8 => Ok(if format == 1 {
                i32::from(bits as u8 as i8)
            } else {
                i32::from(bits as u8)
            }),
        }
    }

    fn shared_offset(
        self,
        source: &RuntimeBuffer,
        descriptor: RawTcgenMatrixDescriptor,
        row: usize,
        column: usize,
        transpose: bool,
    ) -> Result<usize, EngineError> {
        match self {
            Self::Ti16 => {
                raw_tcgen05_b16_matrix_byte_offset(source, descriptor, row, column, transpose)
            }
            Self::I8 => {
                raw_tcgen05_8bit_matrix_byte_offset(source, descriptor, row, column, transpose)
            }
        }
    }
}

fn decode_ti16(bits: u16, negate: bool) -> Result<i32, EngineError> {
    if bits & 0x7800 != 0 {
        return Err(EngineError::message(
            "s1z4m11 operand has nonzero reserved bits",
        ));
    }
    let magnitude = i32::from(bits & 0x7ff);
    Ok(if (bits & 0x8000 != 0) ^ negate {
        -magnitude
    } else {
        magnitude
    })
}

pub(crate) fn raw_tcgen05_integer_shape(
    kind: RawTcgenIntegerKind,
    descriptor: u32,
    cta_group: usize,
    weight_stationary: bool,
    sparse: bool,
) -> Result<(usize, usize, bool, bool), EngineError> {
    // PTX 9.4 Table 51: both integer kinds accumulate in S32; their
    // operand encodings and saturation permissions differ.
    let mut reserved = if weight_stationary {
        0x2080004f
    } else {
        0xe080004f
    };
    if sparse {
        reserved &= !7;
    }
    if kind == RawTcgenIntegerKind::I8 {
        reserved &= !8;
        reserved |= (1 << 13) | (1 << 14);
    }
    let a_format = (descriptor >> 7) & 7;
    let b_format = (descriptor >> 10) & 7;
    let valid_operands = match kind {
        RawTcgenIntegerKind::Ti16 => {
            a_format == 3 && b_format == 3 && (!sparse || descriptor & 2 == 0)
        }
        RawTcgenIntegerKind::I8 => a_format <= 1 && b_format <= 1 && !sparse,
    };
    if descriptor & reserved != 0
        || (descriptor & 4 != 0) != sparse
        || (descriptor >> 4) & 3 != 2
        || !valid_operands
    {
        return Err(EngineError::message(
            "integer MMA descriptor has invalid reserved bits, sparsity or operand formats",
        ));
    }
    let m = ((descriptor >> 24) & 31) as usize * 16;
    let n = ((descriptor >> 17) & 63) as usize * 8;
    let valid = match (cta_group, m, weight_stationary) {
        (1, 32 | 64 | 128, true) => matches!(n, 64 | 128) || (!sparse && n == 256),
        // PTX Table 48: I8 permits N8/N24 for both M sizes, but above
        // N32 requires multiples of 16. TI16 has different M-dependent rules.
        (1, 64 | 128, false) if kind == RawTcgenIntegerKind::I8 => {
            matches!(n, 8 | 24) || ((16..=256).contains(&n) && n.is_multiple_of(16))
        }
        (1, 64, false) => (8..=256).contains(&n) && n.is_multiple_of(8),
        (1, 128, false) => (16..=256).contains(&n) && n.is_multiple_of(16),
        (2, 128 | 256, false) => (32..=256).contains(&n) && n.is_multiple_of(32),
        _ => false,
    };
    if !valid {
        return Err(EngineError::message(format!(
            "{kind:?} CTA{cta_group} has invalid M={m}, N={n} geometry",
        )));
    }
    // PTX 9.4 Table 51 requires zero transpose bits for TI16, whereas
    // Table 62 says transpose is supported. Neither is hardware-verified.
    if kind == RawTcgenIntegerKind::Ti16 && descriptor & ((1 << 15) | (1 << 16)) != 0 {
        return Err(EngineError::analysis_incomplete("ti16_transpose_unmodeled"));
    }
    Ok((
        m,
        n,
        descriptor & (1 << 15) != 0,
        descriptor & (1 << 16) != 0,
    ))
}

#[allow(clippy::too_many_arguments)]
fn gather_integer_shared(
    kind: RawTcgenIntegerKind,
    physical: &PhysicalMemory,
    context: &WarpContext,
    source: &RuntimeBuffer,
    descriptor: RawTcgenMatrixDescriptor,
    rows: usize,
    columns: usize,
    transpose: bool,
    cta_group: usize,
    mask: Option<RawTcgenColumnMask>,
    decode: impl Fn(u16) -> Result<i32, EngineError>,
) -> Result<Vec<i32>, EngineError> {
    let mut values = Vec::with_capacity(rows * columns * cta_group);
    let first = context.cta_id_in_cluster() & !(cta_group - 1);
    let width = 32 / kind.packed_k();
    for cta in first..first + cta_group {
        let view = raw_tcgen05_shared_view_at_cta(physical, context, source, cta)?;
        for row in 0..rows {
            let Some(row) = mask.map_or(Some(row), |mask| mask.source_column(row)) else {
                values.resize(values.len() + columns, 0);
                continue;
            };
            for column in 0..columns {
                let offset = kind.shared_offset(source, descriptor, row, column, transpose)?;
                let mut bytes = [0_u8; 2];
                physical
                    .shared()
                    .read_bytes_into(&view, offset, &mut bytes[..width])?;
                values.push(decode(u16::from_le_bytes(bytes))?);
            }
        }
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_tcgen05_integer_shared_footprints(
    kind: RawTcgenIntegerKind,
    context: &WarpContext,
    candidates: &[&RuntimeBuffer],
    bits: u64,
    rows: usize,
    columns: usize,
    transpose: bool,
    lane: usize,
    cta_group: usize,
    mask: Option<RawTcgenColumnMask>,
) -> Result<(RuntimeBuffer, Vec<RawTcgenRuntimeAccess>), EngineError> {
    let descriptor = decode_raw_tcgen_matrix_descriptor(bits)?;
    let source = raw_tcgen05_shared_source(context, candidates, descriptor.start_address, lane)?;
    let first = context.cta_id_in_cluster() & !(cta_group - 1);
    let width = 32 / kind.packed_k();
    let atom_elements = if transpose { 1 } else { 16 / width };
    let mut accesses = Vec::new();
    for cta in first..first + cta_group {
        for row in 0..rows {
            let Some(row) = mask.map_or(Some(row), |mask| mask.source_column(row)) else {
                continue;
            };
            for column in (0..columns).step_by(atom_elements) {
                let offset = kind.shared_offset(&source, descriptor, row, column, transpose)?;
                accesses.push((lane, Some(cta), offset, atom_elements * width));
            }
        }
    }
    Ok((source, accesses))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_tcgen05_mma_integer<const MASKS: usize>(
    kind: RawTcgenIntegerKind,
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access: TmemAccessMode,
    anchor: &RuntimeBuffer,
    shared_candidates: &[&RuntimeBuffer],
    destination: u32,
    a_source: RawTcgenMmaA,
    b_bits: u64,
    descriptor: u32,
    enable_d: bool,
    masks: [u32; MASKS],
    lane: usize,
    ws_mask: Option<u64>,
    metadata: Option<u32>,
) -> Result<(), EngineError> {
    let cta_group = match MASKS {
        4 => 1,
        8 => 2,
        _ => {
            return Err(EngineError::message(
                "TI16 requires four lane masks per CTA",
            ))
        }
    };
    let (m, n, transpose_a, transpose_b) = raw_tcgen05_integer_shape(
        kind,
        descriptor,
        cta_group,
        ws_mask.is_some(),
        metadata.is_some(),
    )?;
    let packed_k = kind.packed_k();
    let k = packed_k * if metadata.is_some() { 2 } else { 1 };
    let column_mask = ws_mask
        .map(|bits| RawTcgenColumnMask::new(bits, m, n, descriptor))
        .transpose()?;
    let rows = m / cta_group;
    let layout = if cta_group == 1 {
        raw_tcgen05_cta1_dense_tmem_layout(m, ws_mask.is_some())?
    } else {
        raw_tcgen05_cta1_dense_tmem_layout(rows, metadata.is_none())?
    };
    let first_cta = context.cta_id_in_cluster() & !(cta_group - 1);
    if first_cta + cta_group > context.topology().ctas_per_cluster() {
        return Err(EngineError::message("TI16 has no paired CTA"));
    }
    let b_desc = decode_raw_tcgen_matrix_descriptor(b_bits)?;
    let b_source =
        raw_tcgen05_shared_source(context, shared_candidates, b_desc.start_address, lane)?;
    let (_, destination_column) = raw_tcgen05_address(destination, 0, 0)?;
    for target in first_cta..first_cta + cta_group {
        validate_raw_tcgen05_tmem_columns(
            lifecycle,
            access,
            context,
            target,
            destination_column,
            layout.physical_columns(n)?,
        )?;
    }
    let view = raw_tcgen05_tmem_view(physical, context, anchor, context.cta_id_in_cluster())?;
    let decode_a = |bits| kind.decode(bits, (descriptor >> 7) & 7, descriptor & (1 << 13) != 0);
    let a = match a_source {
        RawTcgenMmaA::Shared(bits) => {
            let desc = decode_raw_tcgen_matrix_descriptor(bits)?;
            let source =
                raw_tcgen05_shared_source(context, shared_candidates, desc.start_address, lane)?;
            gather_integer_shared(
                kind,
                physical,
                context,
                &source,
                desc,
                rows,
                packed_k,
                transpose_a,
                cta_group,
                None,
                decode_a,
            )?
        }
        RawTcgenMmaA::Tmem(address) => {
            raw_tcgen05_validate_tmem_a_transpose(transpose_a)?;
            let (_, a_column) = raw_tcgen05_address(address, 0, 0)?;
            for target in first_cta..first_cta + cta_group {
                validate_raw_tcgen05_tmem_columns(lifecycle, access, context, target, a_column, 8)?;
            }
            let decode2 = |word: u32| Ok([decode_a(word as u16)?, decode_a((word >> 16) as u16)?]);
            let decode4 = |word: u32| {
                Ok([
                    decode_a(word as u8 as u16)?,
                    decode_a((word >> 8) as u8 as u16)?,
                    decode_a((word >> 16) as u8 as u16)?,
                    decode_a((word >> 24) as u16)?,
                ])
            };
            match (kind, cta_group) {
                (RawTcgenIntegerKind::Ti16, 1) => raw_tcgen05_gather_packed_tmem_a_with(
                    physical, &view, address, m, layout, decode2,
                )?,
                (RawTcgenIntegerKind::Ti16, _) => raw_tcgen05_gather_packed_tmem_a_cta2_with(
                    physical,
                    context,
                    anchor,
                    address,
                    m,
                    layout,
                    RAW_TCGEN_CTA1_PACKED_A_COLUMNS,
                    decode2,
                )?,
                (RawTcgenIntegerKind::I8, 1) => raw_tcgen05_gather_packed_tmem_a_with(
                    physical, &view, address, m, layout, decode4,
                )?,
                (RawTcgenIntegerKind::I8, _) => raw_tcgen05_gather_packed_tmem_a_cta2_with(
                    physical,
                    context,
                    anchor,
                    address,
                    m,
                    layout,
                    RAW_TCGEN_CTA1_PACKED_A_COLUMNS,
                    decode4,
                )?,
            }
        }
    };
    let a = if let Some(metadata) = metadata {
        let (metadata_lane, metadata_column) = raw_tcgen05_address(metadata, 0, 0)?;
        if metadata_column % 2 != 0 || metadata_lane != raw_tcgen05_address(destination, 0, 0)?.0 {
            return Err(EngineError::message(
                "sparse TI16 metadata requires two-column alignment and matching datapath lanes",
            ));
        }
        let mut expanded = Vec::new();
        for cta in 0..cta_group {
            validate_raw_tcgen05_tmem_columns(
                lifecycle,
                access,
                context,
                first_cta + cta,
                metadata_column,
                2,
            )?;
            let view = raw_tcgen05_tmem_view(physical, context, anchor, first_cta + cta)?;
            // CTA2 sparse layouts C/A have one A bank per CTA. CTA1 WS can
            // have several N-selected banks and expands them together.
            let packed = if cta_group == 1 {
                &a[..]
            } else {
                &a[cta * rows * packed_k..(cta + 1) * rows * packed_k]
            };
            expanded.extend(raw_tcgen05_expand_sparse_2of4(
                packed,
                rows,
                layout,
                k,
                |row, chunk| {
                    raw_tcgen05_sparse_metadata_code(
                        physical,
                        &view,
                        metadata,
                        RawTcgenSparseMetadataLayout::B16 {
                            selector: (descriptor & 1) as usize,
                        },
                        row,
                        chunk,
                    )
                },
            )?);
        }
        expanded
    } else {
        a
    };
    let decode_b = |bits| kind.decode(bits, (descriptor >> 10) & 7, descriptor & (1 << 14) != 0);
    let b = gather_integer_shared(
        kind,
        physical,
        context,
        &b_source,
        b_desc,
        n / cta_group,
        k,
        transpose_b,
        cta_group,
        column_mask,
        decode_b,
    )?;
    let mut output = Vec::with_capacity(m * n);
    for cta in 0..cta_group {
        let view = raw_tcgen05_tmem_view(physical, context, anchor, first_cta + cta)?;
        let local_masks = std::array::from_fn(|i| masks[cta * 4 + i]);
        if enable_d {
            output.extend(
                raw_tcgen05_read_dense_tmem(
                    physical,
                    &view,
                    destination,
                    rows,
                    n,
                    layout,
                    i32::from_le_bytes,
                    0,
                    local_masks,
                )?
                .into_iter()
                .map(i64::from),
            );
        } else {
            output.resize(output.len() + rows * n, 0_i64);
        }
    }
    let banks = a.len() / (m * k);
    let bank_n = n / banks;
    for bank in 0..banks {
        let mut partial = output
            .chunks_exact(n)
            .flat_map(|row| row[bank * bank_n..(bank + 1) * bank_n].iter().copied())
            .collect::<Vec<_>>();
        numsim_fp_env::multiply_accumulate_i32_abt(
            m,
            bank_n,
            k,
            &a[bank * m * k..(bank + 1) * m * k],
            &b[bank * bank_n * k..(bank + 1) * bank_n * k],
            &mut partial,
        )
        .map_err(|error| EngineError::message(format!("TI16 MMA shape error: {error}")))?;
        for (row, partial_row) in output.chunks_exact_mut(n).zip(partial.chunks_exact(bank_n)) {
            row[bank * bank_n..(bank + 1) * bank_n].copy_from_slice(partial_row);
        }
    }
    // Accumulate exactly, then retain the low 32 bits. Never round through F32.
    for cta in 0..cta_group {
        let view = raw_tcgen05_tmem_view(physical, context, anchor, first_cta + cta)?;
        raw_tcgen05_scatter_dense(
            physical,
            &view,
            destination,
            rows,
            n,
            layout,
            |value: i64| {
                let value = if kind == RawTcgenIntegerKind::I8 && descriptor & 8 != 0 {
                    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX))
                } else {
                    value
                };
                (value as i32).to_le_bytes()
            },
            std::array::from_fn(|i| masks[cta * 4 + i]),
            &output[cta * rows * n..(cta + 1) * rows * n],
        )?;
    }
    Ok(())
}

#[test]
fn integer_formats_geometry_and_reserved_bits() {
    for (bits, expected) in [(0, 0), (0x8000, 0), (0x7ff, 2047), (0x87ff, -2047)] {
        assert_eq!(decode_ti16(bits, false).unwrap(), expected);
        assert_eq!(decode_ti16(bits, true).unwrap(), -expected);
    }
    assert!(decode_ti16(0x800, false).is_err());
    let descriptor = (2 << 4) | (3 << 7) | (3 << 10) | (1 << 17) | (4 << 24);
    assert_eq!(
        raw_tcgen05_integer_shape(RawTcgenIntegerKind::Ti16, descriptor, 1, false, false).unwrap(),
        (64, 8, false, false)
    );
    assert!(
        raw_tcgen05_integer_shape(RawTcgenIntegerKind::Ti16, descriptor | 8, 1, false, false)
            .is_err()
    );
    for bit in [15, 16] {
        let error = raw_tcgen05_integer_shape(
            RawTcgenIntegerKind::Ti16,
            descriptor | (1 << bit),
            1,
            false,
            false,
        )
        .unwrap_err();
        assert_eq!(
            error.kind(),
            EngineError::analysis_incomplete("ti16_transpose_unmodeled").kind()
        );
    }
    // The two kinds share CTA2/WS rules, not the CTA1 non-WS N granularity.
    for (cta, m, n, ws, i8_valid, ti16_valid) in [
        (1, 64, 8, false, true, true),
        (1, 64, 24, false, true, true),
        (1, 128, 8, false, true, false),
        (1, 128, 24, false, true, false),
        (1, 64, 40, false, false, true),
        (1, 128, 40, false, false, false),
        (1, 128, 48, false, true, true),
        (1, 128, 256, false, true, true),
        (1, 128, 264, false, false, false),
        (2, 128, 16, false, false, false),
        (2, 128, 32, false, true, true),
        (2, 256, 24, false, false, false),
        (2, 256, 64, false, true, true),
        (1, 32, 64, true, true, true),
        (1, 128, 24, true, false, false),
        (1, 128, 256, true, true, true),
    ] {
        for (kind, format, valid) in [
            (RawTcgenIntegerKind::I8, 1, i8_valid),
            (RawTcgenIntegerKind::Ti16, 3, ti16_valid),
        ] {
            let descriptor =
                (2 << 4) | (format << 7) | (format << 10) | ((n / 8) << 17) | ((m / 16) << 24);
            assert_eq!(
                raw_tcgen05_integer_shape(kind, descriptor, cta, ws, false).is_ok(),
                valid,
                "{kind:?} CTA{cta} M{m} N{n} WS={ws}"
            );
        }
    }
}
