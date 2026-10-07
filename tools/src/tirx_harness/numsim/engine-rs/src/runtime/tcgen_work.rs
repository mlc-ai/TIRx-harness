use std::collections::BTreeSet;

use crate::{
    AsyncTokenId, EngineError, MemoryAccessSemantics, OperationContext, PhysicalAccessBatch,
    PhysicalAccessBatchError, PhysicalAccessKind, PhysicalAccessSpace, PhysicalByteSpan,
    PhysicalFootprintError, WARP_SIZE,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TcgenWorkKind {
    Commit,
    MmaSharedARead,
    Load,
    Store,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum TcgenPipelineOperation {
    Mma,
    // The independently completable shared-A read, not a second MMA.
    MmaSharedARead,
    Copy,
    Copy4x256b,
    Shift,
    Load,
    Store,
}

impl TcgenPipelineOperation {
    pub(crate) const fn work_kind(self) -> TcgenWorkKind {
        match self {
            Self::Mma | Self::Copy | Self::Copy4x256b | Self::Shift => TcgenWorkKind::Commit,
            Self::MmaSharedARead => TcgenWorkKind::MmaSharedARead,
            Self::Load => TcgenWorkKind::Load,
            Self::Store => TcgenWorkKind::Store,
        }
    }
}

/// Element type of a TCGEN MMA accumulator.
///
/// PTX orders one MMA against the next only when both name the same
/// instruction shape *and* the same accumulator type, so this is part of the
/// pipeline class rather than a numeric detail. `F16` is the half-precision
/// destination of `kind::f16` or `kind::f8f6f4`: it occupies one 32-bit TMEM cell per
/// element, with the payload in the low half.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum TcgenAccumulatorDtype {
    F32,
    F16,
    I32,
}

/// Accumulator class and instruction shape required by PTX for the implicit
/// MMA-to-MMA pipeline, apart from the CTA group tracked by Racecheck.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct TcgenMmaPipelineClass {
    instruction_shape: [usize; 3],
    accumulator_dtype: TcgenAccumulatorDtype,
}

impl TcgenMmaPipelineClass {
    pub(crate) const fn new(
        instruction_m: usize,
        instruction_n: usize,
        instruction_k: usize,
        accumulator_dtype: TcgenAccumulatorDtype,
    ) -> Self {
        Self {
            instruction_shape: [instruction_m, instruction_n, instruction_k],
            accumulator_dtype,
        }
    }
}

impl TcgenWorkKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Commit => "tcgen05 commit-bound work",
            Self::MmaSharedARead => "tcgen05.mma shared-A read",
            Self::Load => "tcgen05.ld",
            Self::Store => "tcgen05.st",
        }
    }

    pub const fn uses_commit(self) -> bool {
        matches!(self, Self::Commit | Self::MmaSharedARead)
    }

    pub const fn uses_wait(self) -> bool {
        matches!(self, Self::Load | Self::Store)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcgenWorkIssue {
    operation: OperationContext,
    token: AsyncTokenId,
    cta_group: u32,
    pipeline_operation: TcgenPipelineOperation,
    mma_pipeline_class: Option<TcgenMmaPipelineClass>,
    accesses: Box<[PhysicalAccessBatch]>,
    byte_count: u64,
}

impl TcgenWorkIssue {
    pub fn new(
        operation: OperationContext,
        cta_group: u32,
        pipeline_operation: TcgenPipelineOperation,
        mma_pipeline_class: Option<TcgenMmaPipelineClass>,
        accesses: impl IntoIterator<Item = PhysicalAccessBatch>,
    ) -> Result<Self, EngineError> {
        let kind = pipeline_operation.work_kind();
        if !matches!(cta_group, 1 | 2) {
            return Err(EngineError::message(format!(
                "{} cta_group must be 1 or 2, got {cta_group}",
                kind.name()
            )));
        }
        if pipeline_operation != TcgenPipelineOperation::Mma && mma_pipeline_class.is_some() {
            return Err(EngineError::message(
                "only a TCGEN MMA may carry an MMA pipeline class",
            ));
        }
        let accesses = accesses
            .into_iter()
            .map(|batch| {
                if kind.uses_commit() && batch.descriptor().space() == PhysicalAccessSpace::Shared {
                    batch.with_memory_semantics(MemoryAccessSemantics::async_proxy())
                } else {
                    batch
                }
            })
            .collect::<Vec<_>>();
        let mut byte_count = 0_u64;
        for (index, access) in accesses.iter().enumerate() {
            if kind == TcgenWorkKind::MmaSharedARead
                && (access.descriptor().space() != PhysicalAccessSpace::Shared
                    || access.descriptor().kind() != PhysicalAccessKind::Read)
            {
                return Err(EngineError::message(
                    "MMA shared-A completion can only track shared reads",
                ));
            }
            if access.operation() != &operation {
                return Err(EngineError::message(format!(
                    "{} access batch {index} belongs to {}, expected {}",
                    kind.name(),
                    access.operation().id(),
                    operation.id()
                )));
            }
            if !matches!(
                access.descriptor().space(),
                PhysicalAccessSpace::Shared | PhysicalAccessSpace::Tmem
            ) {
                return Err(EngineError::message(format!(
                    "{} access batch {index} uses unsupported {} space",
                    kind.name(),
                    access.descriptor().space()
                )));
            }
            let batch_bytes = access.lanes().iter().try_fold(0_u64, |total, lane| {
                total.checked_add(lane.footprint().byte_len() as u64)
            });
            byte_count = byte_count
                .checked_add(batch_bytes.ok_or_else(|| {
                    EngineError::message(format!("{} byte count overflow", kind.name()))
                })?)
                .ok_or_else(|| {
                    EngineError::message(format!("{} byte count overflow", kind.name()))
                })?;
        }
        Ok(Self {
            token: AsyncTokenId::new(
                operation.id().clone(),
                u32::from(kind == TcgenWorkKind::MmaSharedARead),
            ),
            operation,
            cta_group,
            pipeline_operation,
            mma_pipeline_class,
            accesses: accesses.into_boxed_slice(),
            byte_count,
        })
    }

    pub const fn operation(&self) -> &OperationContext {
        &self.operation
    }

    pub const fn token(&self) -> &AsyncTokenId {
        &self.token
    }

    pub const fn kind(&self) -> TcgenWorkKind {
        self.pipeline_operation.work_kind()
    }

    pub const fn cta_group(&self) -> u32 {
        self.cta_group
    }

    pub(crate) const fn pipeline_operation(&self) -> TcgenPipelineOperation {
        self.pipeline_operation
    }

    /// Compile-time equivalence class for the PTX same-thread MMA pipeline.
    ///
    /// Equal values prove the same statically known instruction shape and
    /// accumulator dtype. Racecheck combines this class with the CTA group;
    /// operand formats, transpose modes, mappings, and the destination TMEM
    /// address do not affect this pipeline relation. `None` disables implicit
    /// MMA-to-MMA ordering when generated code cannot establish the static
    /// shape (for example, a runtime descriptor).
    pub(crate) const fn mma_pipeline_class(&self) -> Option<TcgenMmaPipelineClass> {
        self.mma_pipeline_class
    }

    pub fn accesses(&self) -> &[PhysicalAccessBatch] {
        &self.accesses
    }

    pub const fn byte_count(&self) -> u64 {
        self.byte_count
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcgenWorkSet {
    kind: TcgenWorkKind,
    cta_group: Option<u32>,
    global_warp_id: usize,
    lane_tokens: Box<[(usize, Box<[AsyncTokenId]>)]>,
    tokens: Box<[AsyncTokenId]>,
}

impl TcgenWorkSet {
    pub(crate) fn new(
        kind: TcgenWorkKind,
        cta_group: Option<u32>,
        global_warp_id: usize,
        lane_tokens: impl IntoIterator<Item = (usize, Box<[AsyncTokenId]>)>,
    ) -> Self {
        let lane_tokens = lane_tokens.into_iter().collect::<Vec<_>>();
        let tokens = lane_tokens
            .iter()
            .flat_map(|(_, tokens)| tokens.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            kind,
            cta_group,
            global_warp_id,
            lane_tokens: lane_tokens.into_boxed_slice(),
            tokens,
        }
    }

    pub const fn kind(&self) -> TcgenWorkKind {
        self.kind
    }

    pub const fn cta_group(&self) -> Option<u32> {
        self.cta_group
    }

    pub const fn global_warp_id(&self) -> usize {
        self.global_warp_id
    }

    pub(crate) fn lane_tokens(&self) -> &[(usize, Box<[AsyncTokenId]>)] {
        &self.lane_tokens
    }

    pub fn tokens(&self) -> &[AsyncTokenId] {
        &self.tokens
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Exact footprint accumulator for one source-level TCGEN operand.
///
/// Overlapping and duplicate element spans are unioned before the checker sees
/// one access batch for the source operation.
pub(crate) struct TcgenAccessFootprintBuilder {
    operation: OperationContext,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    logical_buffer: Option<Box<str>>,
    lane_spans: [Vec<PhysicalByteSpan>; WARP_SIZE],
}

impl TcgenAccessFootprintBuilder {
    pub(crate) fn new(
        operation: OperationContext,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        logical_buffer: Option<&str>,
    ) -> Result<Self, EngineError> {
        if !matches!(
            space,
            PhysicalAccessSpace::Shared | PhysicalAccessSpace::Tmem
        ) {
            return Err(EngineError::message(format!(
                "TCGEN footprint builder does not track {space} space"
            )));
        }
        Ok(Self {
            operation,
            kind,
            space,
            logical_buffer: logical_buffer.map(Into::into),
            lane_spans: std::array::from_fn(|_| Vec::new()),
        })
    }

    pub(crate) fn record_span(
        &mut self,
        execution_lane: usize,
        span: PhysicalByteSpan,
    ) -> Result<(), EngineError> {
        if execution_lane >= WARP_SIZE {
            return Err(EngineError::message(format!(
                "TCGEN footprint execution lane {execution_lane} is outside the warp"
            )));
        }
        if !self.operation.active_mask().contains(execution_lane) {
            return Err(EngineError::message(format!(
                "TCGEN footprint lane {execution_lane} is inactive for {}",
                self.operation.id()
            )));
        }
        self.lane_spans[execution_lane].push(span);
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<PhysicalAccessBatch, EngineError> {
        self.finish_with_lane_spans().map(|(batch, _)| batch)
    }

    /// Finish the batch and also hand back the unioned per-lane spans, so a
    /// caller can memoize the resolved geometry and rebuild later batches
    /// for the same footprint without resolving it again.
    pub(crate) fn finish_with_lane_spans(
        mut self,
    ) -> Result<(PhysicalAccessBatch, TcgenLaneSpans), EngineError> {
        for lane in self.operation.active_mask() {
            self.lane_spans[lane] = union_spans(std::mem::take(&mut self.lane_spans[lane]))?;
            if self.lane_spans[lane].is_empty() {
                return Err(EngineError::message(format!(
                    "TCGEN footprint for {} has no physical bytes for active lane {lane}",
                    self.operation.id()
                )));
            }
        }
        let lane_spans = self.lane_spans;
        let batch = tcgen_batch_from_lane_spans(
            self.operation,
            self.kind,
            self.space,
            self.logical_buffer.as_deref(),
            &lane_spans,
        )?;
        Ok((batch, lane_spans))
    }
}

/// Unioned physical spans of one TCGEN footprint, per execution lane.
pub(crate) type TcgenLaneSpans = [Vec<PhysicalByteSpan>; WARP_SIZE];

/// Build the access batch of a footprint whose per-lane spans are already
/// resolved (freshly, or from a memo of an identical earlier resolution).
pub(crate) fn tcgen_batch_from_lane_spans(
    operation: OperationContext,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    logical_buffer: Option<&str>,
    lane_spans: &TcgenLaneSpans,
) -> Result<PhysicalAccessBatch, EngineError> {
    PhysicalAccessBatch::resolve_lane_widths(operation, kind, space, |provenance| {
        Ok::<_, EngineError>(lane_spans[provenance.lane()].clone())
    })
    .map_err(|error| match error {
        PhysicalAccessBatchError::LaneResolution { source, .. } => source,
        other => EngineError::message(other.to_string()),
    })
    .map(|batch| match logical_buffer {
        Some(logical_buffer) => batch.with_logical_buffer(logical_buffer),
        None => batch,
    })
}

fn union_spans(mut spans: Vec<PhysicalByteSpan>) -> Result<Vec<PhysicalByteSpan>, EngineError> {
    const MAX_DENSE_UNION_BYTES: usize = 1024 * 1024;
    const MAX_DENSE_BYTES_PER_INPUT: usize = 32;

    if let Some(first) = spans.first().copied() {
        let allocation = first.allocation();
        let mut minimum = first.byte_offset();
        let mut maximum = first.byte_end();
        let mut one_allocation = true;
        for span in &spans[1..] {
            one_allocation &= span.allocation() == allocation;
            minimum = minimum.min(span.byte_offset());
            maximum = maximum.max(span.byte_end());
        }
        let dense_bytes = maximum - minimum;
        if one_allocation
            && dense_bytes <= MAX_DENSE_UNION_BYTES
            && dense_bytes <= spans.len().saturating_mul(MAX_DENSE_BYTES_PER_INPUT)
        {
            let mut covered = vec![0_u8; dense_bytes];
            for span in spans {
                covered[span.byte_offset() - minimum..span.byte_end() - minimum].fill(1);
            }
            let mut union = Vec::new();
            let mut cursor = 0_usize;
            while cursor < covered.len() {
                let Some(relative_start) = covered[cursor..].iter().position(|byte| *byte != 0)
                else {
                    break;
                };
                let start = cursor + relative_start;
                let end = covered[start..]
                    .iter()
                    .position(|byte| *byte == 0)
                    .map_or(covered.len(), |relative_end| start + relative_end);
                union.push(
                    PhysicalByteSpan::new(allocation, minimum + start, end - start)
                        .map_err(|error| EngineError::message(error.to_string()))?,
                );
                cursor = end;
            }
            return Ok(union);
        }
    }

    spans.sort_unstable();
    let mut union: Vec<PhysicalByteSpan> = Vec::with_capacity(spans.len());
    for span in spans {
        let Some(previous) = union.last_mut() else {
            union.push(span);
            continue;
        };
        if previous.allocation() != span.allocation() || span.byte_offset() > previous.byte_end() {
            union.push(span);
            continue;
        }
        let end = previous.byte_end().max(span.byte_end());
        *previous = PhysicalByteSpan::new(
            previous.allocation(),
            previous.byte_offset(),
            end - previous.byte_offset(),
        )
        .map_err(|error: PhysicalFootprintError| EngineError::message(error.to_string()))?;
    }
    Ok(union)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DynamicOpId, OperationKind, PhysicalAllocationId, StaticOpId, WarpMask};

    #[test]
    fn the_accumulator_dtype_separates_two_pipeline_classes_of_the_same_shape() {
        // `kind::f8f6f4` reaches the same M/N/K with either a `.f32` or a
        // `.f16` destination, so the dtype must be a real discriminator of the
        // pipeline class rather than a field that always holds one value.
        let shape = (128, 16, 32);
        let f32_class =
            TcgenMmaPipelineClass::new(shape.0, shape.1, shape.2, TcgenAccumulatorDtype::F32);
        let f16_class =
            TcgenMmaPipelineClass::new(shape.0, shape.1, shape.2, TcgenAccumulatorDtype::F16);

        assert_ne!(f32_class, f16_class);
        assert_eq!(
            f32_class,
            TcgenMmaPipelineClass::new(shape.0, shape.1, shape.2, TcgenAccumulatorDtype::F32),
        );
        // A shape difference must still separate classes of the same dtype, so
        // the inequality above is not the only thing keeping them apart.
        assert_ne!(
            f16_class,
            TcgenMmaPipelineClass::new(64, shape.1, shape.2, TcgenAccumulatorDtype::F16),
        );
    }

    #[test]
    fn footprint_builder_unions_duplicate_and_adjacent_element_spans() {
        let mask = WarpMask::from_lanes([3]).unwrap();
        let operation = OperationContext::new(
            DynamicOpId::new(0, 0, 0, StaticOpId::new(7), []),
            OperationKind::AsyncIssue,
            mask,
        );
        let allocation = PhysicalAllocationId::new(9);
        let mut builder = TcgenAccessFootprintBuilder::new(
            operation.clone(),
            PhysicalAccessKind::Read,
            PhysicalAccessSpace::Tmem,
            Some("accumulator"),
        )
        .unwrap();
        for (offset, width) in [(8, 4), (8, 4), (12, 4)] {
            builder
                .record_span(3, PhysicalByteSpan::new(allocation, offset, width).unwrap())
                .unwrap();
        }

        let batch = builder.finish().unwrap();
        assert_eq!(batch.operation(), &operation);
        assert_eq!(batch.logical_buffer(), Some("accumulator"));
        assert_eq!(batch.lane(3).unwrap().footprint().spans().len(), 1);
        assert_eq!(batch.lane(3).unwrap().footprint().byte_len(), 8);
    }
}
