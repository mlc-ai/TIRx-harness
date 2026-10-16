//! Engine-private execution for one synchronous mapped tile copy.

use crate::engine_mode::EngineMode;
use crate::kernel_engine::WarpEngine;
use crate::runtime::{
    element_byte_offset, read_runtime_bytes, read_runtime_bytes_zero_filled_into,
    resolve_runtime_physical_access, resolve_shared_runtime_physical_access_to_cta,
    write_runtime_bytes, write_shared_runtime_bytes_to_cta, RuntimeBuffer,
};
use crate::{
    EngineError, OperationContext, PhysicalAccessBatch, PhysicalAccessBatchError,
    PhysicalAccessKind, PhysicalAccessSpace, PhysicalByteSpan, WarpContext, WarpMask, WARP_SIZE,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct TypedTileCopyElement {
    pub(crate) source_index: i64,
    pub(crate) source_in_bounds: bool,
    pub(crate) source_lane: usize,
    pub(crate) destination_index: i64,
    pub(crate) destination_in_bounds: bool,
    pub(crate) destination_lane: usize,
}

pub(crate) struct TypedTileCopySnapshot {
    bytes: Vec<Vec<u8>>,
    high_values: Option<Vec<Option<crate::high_precision::ShadowValue>>>,
}

fn checked_index(index: i64, lane: usize, role: &str) -> Result<usize, EngineError> {
    usize::try_from(index).map_err(|_| {
        EngineError::out_of_bounds(format!(
            "tile copy {role} index {index} is negative on lane {lane}"
        ))
    })
}

fn uniform_space(buffer: &RuntimeBuffer, role: &str) -> Result<PhysicalAccessSpace, EngineError> {
    buffer.uniform_physical_space().ok_or_else(|| {
        EngineError::message(format!(
            "tile copy {role} has lane-varying or unsupported memory space"
        ))
    })
}

fn resolve_source_spans(
    context: &WarpContext,
    source: &RuntimeBuffer,
    itemsize: usize,
    elements: &[TypedTileCopyElement],
) -> Result<([Vec<PhysicalByteSpan>; WARP_SIZE], WarpMask), EngineError> {
    let mut spans = std::array::from_fn(|_| Vec::new());
    let mut mask = WarpMask::EMPTY;
    for element in elements.iter().filter(|element| element.source_in_bounds) {
        let lane = element.source_lane;
        let index = checked_index(element.source_index, lane, "source")?;
        let offset = element_byte_offset(source, index, itemsize, lane)?;
        let access = resolve_runtime_physical_access(
            context,
            source,
            lane,
            offset,
            itemsize,
            PhysicalAccessKind::Read,
        )?;
        spans[lane].push(access.span());
        mask |= WarpMask::from_bits(1_u32 << lane);
    }
    Ok((spans, mask))
}

fn resolve_destination_spans(
    context: &WarpContext,
    destination: &RuntimeBuffer,
    destination_rank: Option<usize>,
    itemsize: usize,
    elements: &[TypedTileCopyElement],
) -> Result<([Vec<PhysicalByteSpan>; WARP_SIZE], WarpMask), EngineError> {
    let mut spans = std::array::from_fn(|_| Vec::new());
    let mut mask = WarpMask::EMPTY;
    for element in elements
        .iter()
        .filter(|element| element.destination_in_bounds)
    {
        let lane = element.destination_lane;
        let index = checked_index(element.destination_index, lane, "destination")?;
        let offset = element_byte_offset(destination, index, itemsize, lane)?;
        let access = match destination_rank {
            Some(rank) => resolve_shared_runtime_physical_access_to_cta(
                context,
                destination,
                lane,
                rank,
                offset,
                itemsize,
                PhysicalAccessKind::Write,
            )?,
            None => resolve_runtime_physical_access(
                context,
                destination,
                lane,
                offset,
                itemsize,
                PhysicalAccessKind::Write,
            )?,
        };
        spans[lane].push(access.span());
        mask |= WarpMask::from_bits(1_u32 << lane);
    }
    Ok((spans, mask))
}

fn union_batch(
    operation: &OperationContext,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    spans: &[Vec<PhysicalByteSpan>; WARP_SIZE],
    logical_buffer: Option<&str>,
) -> Result<PhysicalAccessBatch, EngineError> {
    let mut batch =
        PhysicalAccessBatch::resolve_lane_unions(operation.clone(), kind, space, |provenance| {
            Ok::<_, EngineError>(spans[provenance.lane()].clone())
        })
        .map_err(|error| match error {
            PhysicalAccessBatchError::LaneResolution { source, .. } => source,
            error => EngineError::message(error.to_string()),
        })?
        .with_semantic_access_count(1);
    if let Some(logical_buffer) = logical_buffer {
        batch = batch.with_logical_buffer(logical_buffer);
    }
    Ok(batch)
}

impl<M: EngineMode> WarpEngine<M> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn typed_tile_copy_snapshot(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        itemsize: usize,
        source: &RuntimeBuffer,
        logical_buffer: Option<&str>,
        zero_fill: bool,
        elements: &[TypedTileCopyElement],
    ) -> Result<TypedTileCopySnapshot, EngineError> {
        let (spans, mask) = resolve_source_spans(context, source, itemsize, elements)?;
        let numeric = || {
            let physical = self.kernel().physical();
            let bytes = elements
                .iter()
                .map(|element| {
                    if !element.source_in_bounds {
                        return if zero_fill {
                            Ok(vec![0_u8; itemsize])
                        } else {
                            Err(EngineError::out_of_bounds(format!(
                                "tile copy source coordinate is outside its buffer on lane {}",
                                element.source_lane
                            )))
                        };
                    }
                    let lane = element.source_lane;
                    let index = checked_index(element.source_index, lane, "source")?;
                    let offset = element_byte_offset(source, index, itemsize, lane)?;
                    if zero_fill {
                        let mut bytes = vec![0_u8; itemsize];
                        read_runtime_bytes_zero_filled_into(
                            physical, context, source, lane, offset, &mut bytes,
                        )?;
                        Ok(bytes)
                    } else {
                        read_runtime_bytes(physical, context, source, lane, offset, itemsize)
                    }
                })
                .collect::<Result<Vec<_>, EngineError>>()?;
            let high_values = if physical.global().high_precision_enabled() {
                Some(
                    elements
                        .iter()
                        .map(|element| {
                            if !element.source_in_bounds {
                                return Ok(None);
                            }
                            let lane = element.source_lane;
                            let index = checked_index(element.source_index, lane, "source")?;
                            let offset = element_byte_offset(source, index, itemsize, lane)?;
                            let access = resolve_runtime_physical_access(
                                context, source, lane, offset, itemsize, PhysicalAccessKind::Read,
                            )?;
                            crate::high_precision::memory(physical, access.space())
                                .copy_value(access.space(), access.span())
                        })
                        .collect::<Result<Vec<_>, EngineError>>()?,
                )
            } else {
                None
            };
            Ok(TypedTileCopySnapshot { bytes, high_values })
        };
        if mask.is_empty() {
            return numeric();
        }
        let space = uniform_space(source, "source")?;
        if matches!(
            space,
            PhysicalAccessSpace::Local | PhysicalAccessSpace::Register
        ) {
            return numeric();
        }
        self.resolved_physical_access_batch(
            operation,
            PhysicalAccessKind::Read,
            space,
            mask,
            |operation| {
                union_batch(
                    operation,
                    PhysicalAccessKind::Read,
                    space,
                    &spans,
                    logical_buffer,
                )
            },
            numeric,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn typed_tile_copy_restore(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        itemsize: usize,
        destination: &RuntimeBuffer,
        logical_buffer: Option<&str>,
        destination_rank: Option<usize>,
        elements: &[TypedTileCopyElement],
        snapshots: &TypedTileCopySnapshot,
    ) -> Result<(), EngineError> {
        if elements.len() != snapshots.bytes.len() {
            return Err(EngineError::message(
                "tile copy snapshot count does not match its mapped element count",
            ));
        }
        let (spans, mask) =
            resolve_destination_spans(context, destination, destination_rank, itemsize, elements)?;
        let numeric = || {
            for (element_index, (element, bytes)) in elements.iter().zip(&snapshots.bytes).enumerate()
            {
                if !element.destination_in_bounds {
                    if uniform_space(destination, "destination")? == PhysicalAccessSpace::Global {
                        continue;
                    }
                    return Err(EngineError::out_of_bounds(format!(
                        "tile copy destination coordinate is outside its buffer on lane {}",
                        element.destination_lane
                    )));
                }
                let lane = element.destination_lane;
                let index = checked_index(element.destination_index, lane, "destination")?;
                let offset = element_byte_offset(destination, index, itemsize, lane)?;
                if let Some(values) = &snapshots.high_values {
                    let access = match destination_rank {
                        Some(rank) => resolve_shared_runtime_physical_access_to_cta(
                            context, destination, lane, rank, offset, itemsize,
                            PhysicalAccessKind::Write,
                        )?,
                        None => resolve_runtime_physical_access(
                            context, destination, lane, offset, itemsize, PhysicalAccessKind::Write,
                        )?,
                    };
                    crate::high_precision::memory(self.kernel().physical(), access.space())
                        .replace_copy(access.space(), access.span(), values[element_index])?;
                }
                match destination_rank {
                    Some(rank) => write_shared_runtime_bytes_to_cta(
                        self.kernel().physical(),
                        context,
                        destination,
                        lane,
                        rank,
                        offset,
                        bytes,
                    )?,
                    None => write_runtime_bytes(
                        self.kernel().physical(),
                        context,
                        destination,
                        lane,
                        offset,
                        bytes,
                    )?,
                }
            }
            Ok(())
        };
        if mask.is_empty() {
            return numeric();
        }
        let space = uniform_space(destination, "destination")?;
        if matches!(
            space,
            PhysicalAccessSpace::Local | PhysicalAccessSpace::Register
        ) {
            return numeric();
        }
        self.resolved_physical_access_batch(
            operation,
            PhysicalAccessKind::Write,
            space,
            mask,
            |operation| {
                union_batch(
                    operation,
                    PhysicalAccessKind::Write,
                    space,
                    &spans,
                    logical_buffer,
                )
            },
            numeric,
        )
    }
}
