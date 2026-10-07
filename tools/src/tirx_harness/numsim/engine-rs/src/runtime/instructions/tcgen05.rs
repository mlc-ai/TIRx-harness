//! Engine implementation of raw v2 TCGEN05 instruction specializations.

use std::marker::PhantomData;

use super::instruction::{async_instruction, instruction_variant, sync_instruction};
use super::transport::engine;
use super::{
    begin, finish, Address, BufferHandle, DescriptorDomain, EngineError, ExecCtx, Register, Shared,
    SiteId, Tmem, WarpHandle, R,
};
use crate::runtime::matrix_ops::raw_tcgen05_shift;
use crate::runtime::tcgen_ops::{
    raw_tcgen05_cp, raw_tcgen05_cp_footprints, raw_tcgen05_cta1_dense_tmem_layout,
    raw_tcgen05_dense_f32_tmem_footprints, raw_tcgen05_f8_shared_footprints,
    raw_tcgen05_f8_tmem_a_address, raw_tcgen05_ld_register, raw_tcgen05_ldst_location,
    raw_tcgen05_lut_b_tmem_footprints, raw_tcgen05_mma_block_mxf4_footprints,
    raw_tcgen05_mma_block_mxf4_shape, raw_tcgen05_mma_block_mxf8f6f4_footprints,
    raw_tcgen05_mma_block_mxf8f6f4_shape, raw_tcgen05_mma_block_scale_mxf4,
    raw_tcgen05_mma_block_scale_mxf8f6f4, raw_tcgen05_mma_f8f6f4_cta1,
    raw_tcgen05_mma_f8f6f4_cta1_shape, raw_tcgen05_mma_f8f6f4_cta2,
    raw_tcgen05_mma_f8f6f4_cta2_footprints, raw_tcgen05_mma_f8f6f4_cta2_shape,
    raw_tcgen05_mma_float, raw_tcgen05_mma_sp_block_scale_mxf4_e8m0_ss_cta1,
    raw_tcgen05_mxf4nvf4_vec2x_scale, raw_tcgen05_mxf4nvf4_vec4x_scale,
    raw_tcgen05_packed_tmem_a_column_footprints, raw_tcgen05_st_register, RawTcgenColumnMask,
    RawTcgenFloatKind, RawTcgenLdstShape, RawTcgenMatrixDescriptorLayout, RawTcgenMmaA,
    RawTcgenMmaAAccess, RawTcgenMxf4ScaleSpelling, RawTcgenNarrowFormat,
};
use crate::runtime::{TcgenAccumulatorDtype, TcgenMmaPipelineClass, TcgenPipelineOperation};
use crate::{OperationKind, TcgenFenceKind, TcgenTransferKind, TmemAccessMode};

async_instruction!(alloc_spec, AllocVariant, alloc);
async_instruction!(dealloc_spec, DeallocVariant, dealloc);
async_instruction!(
    relinquish_alloc_permit_spec,
    RelinquishAllocPermitVariant,
    relinquish_alloc_permit
);
sync_instruction!(cp_spec, CpVariant, cp);
sync_instruction!(ld_spec, LdVariant, ld);
sync_instruction!(st_spec, StVariant, st);
sync_instruction!(mma_spec, MmaVariant, mma);
sync_instruction!(mma_sp_spec, MmaSpVariant, mma_sp);
sync_instruction!(commit_spec, CommitVariant, commit);
sync_instruction!(fence_spec, FenceVariant, fence);
sync_instruction!(shift_spec, ShiftVariant, shift);

/// Static PTX forms. Counts here are instruction immediates or determine the
/// exact register/descriptor spelling; no generic repetition count is exposed.
pub mod variant {
    /// Shift TMEM A after the product, within the same asynchronous MMA issue.
    pub struct MmaAshift<V>(PhantomData<fn() -> V>);
    pub struct MmaLutB<V>(PhantomData<fn() -> V>);
    pub struct MmaSparseBlock<V>(PhantomData<fn() -> V>);

    use super::PhantomData;

    /// Orthogonal collector transitions around the existing MMA variant.
    /// Bits select A and B0..B3; lastuse is REQUIRE plus DISCARD.
    pub struct MmaCollectors<V, const FILL: u8, const REQUIRE: u8, const DISCARD: u8>(
        PhantomData<fn() -> V>,
    );

    /// Column count and `cta_group` are plain instruction immediates the
    /// lifecycle hub re-checks at runtime, so both ride in `Args` rather than
    /// monomorphizing the marker.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Alloc<const EXCLUSIVE: bool = false>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Dealloc<const EXCLUSIVE: bool = false>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Relinquish;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Commit;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CommitMulticast;
    pub struct CommitSharedA;
    pub struct CommitSharedAMulticast;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct FenceBeforeThreadSync;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct FenceAfterThreadSync;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct StaticTmem;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DynamicTmem;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Shape16x32bx2<const HALF_SPLITOFF: usize>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Shape16x64b;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Shape16x128b;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Shape32x32b;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Shape16x256b;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Num<const N: usize>;

    /// One `tcgen05.ld` with an architected register tuple.
    pub struct Ld<Shape, Num, const PACKED: bool, Access, Reduction = NoReduction>(
        PhantomData<fn() -> (Shape, Num, Access, Reduction)>,
    );
    pub struct NoReduction;
    pub struct ReduceF32<const MAX: bool, const ABS: bool, const NAN: bool>;
    pub struct ReduceU32<const MAX: bool>;
    pub struct ReduceI32<const MAX: bool>;
    pub struct Compress<const MAX: bool, const ABS: bool, Reduction = NoReduction>(
        PhantomData<Reduction>,
    );
    /// One `tcgen05.st` with an architected register tuple.
    pub struct St<Shape, Num, const UNPACKED: bool, Access>(
        PhantomData<fn() -> (Shape, Num, Access)>,
    );

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Cp32x128bWarpx4;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Cp64x128bWarpx2_02_13;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Cp128x128b;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Cp128x256b;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Cp4x256b;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Cp64x128bWarpx2_01_23;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct NoDecompress;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DecompressB4;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DecompressB6;
    pub struct Cp<
        Shape,
        Decompress,
        const CTA_GROUP: usize,
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
    >(PhantomData<fn() -> (Shape, Decompress, Access, DescriptorLayout)>);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Fp16;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Bf16;
    pub struct Scale<const VALUE: usize>;
    pub struct Ti16;
    pub struct I8;

    pub struct MmaF16SsCta1<A, B, Scale, Access>(PhantomData<fn() -> (A, B, Scale, Access)>);
    pub struct MmaF16SsCta1Pred<A, B, Scale, Access>(PhantomData<fn() -> (A, B, Scale, Access)>);
    pub struct MmaF16SsCta1Ws<A, B, Scale, Access>(PhantomData<fn() -> (A, B, Scale, Access)>);
    pub struct MmaIntegerSsCta1<Kind, Access>(PhantomData<fn() -> (Kind, Access)>);
    pub struct MmaIntegerSsCta1Pred<Kind, Access>(PhantomData<fn() -> (Kind, Access)>);
    pub struct MmaIntegerTsCta1<Kind, Access>(PhantomData<fn() -> (Kind, Access)>);
    pub struct MmaIntegerTsCta1Pred<Kind, Access>(PhantomData<fn() -> (Kind, Access)>);
    pub struct MmaIntegerSsCta2<Kind, Access>(PhantomData<fn() -> (Kind, Access)>);
    pub struct MmaIntegerSsCta2Pred<Kind, Access>(PhantomData<fn() -> (Kind, Access)>);
    pub struct MmaIntegerTsCta2<Kind, Access>(PhantomData<fn() -> (Kind, Access)>);
    pub struct MmaIntegerTsCta2Pred<Kind, Access>(PhantomData<fn() -> (Kind, Access)>);
    pub struct MmaIntegerSsCta1Ws<Kind, Access>(PhantomData<fn() -> (Kind, Access)>);
    pub struct MmaIntegerTsCta1Ws<Kind, Access>(PhantomData<fn() -> (Kind, Access)>);
    pub struct MmaSparseSsCta1<Kind, Access>(PhantomData<fn() -> (Kind, Access)>);
    pub struct MmaSparseTsCta1<Kind, Access>(PhantomData<fn() -> (Kind, Access)>);
    pub struct MmaSparseSsCta1Ws<Kind, Access>(PhantomData<fn() -> (Kind, Access)>);
    pub struct MmaSparseTsCta1Ws<Kind, Access>(PhantomData<fn() -> (Kind, Access)>);
    pub struct MmaSparseSsCta2<Kind, Access>(PhantomData<fn() -> (Kind, Access)>);
    pub struct MmaSparseTsCta2<Kind, Access>(PhantomData<fn() -> (Kind, Access)>);
    pub struct MmaF16TsCta1<A, B, Scale, Access>(PhantomData<fn() -> (A, B, Scale, Access)>);
    pub struct MmaF16TsCta1Pred<A, B, Scale, Access>(PhantomData<fn() -> (A, B, Scale, Access)>);
    pub struct MmaF16TsCta1Ws<A, B, Scale, Access>(PhantomData<fn() -> (A, B, Scale, Access)>);
    pub struct MmaF16SsCta2<A, B, Scale, Access>(PhantomData<fn() -> (A, B, Scale, Access)>);
    pub struct MmaF16SsCta2Pred<A, B, Scale, Access>(PhantomData<fn() -> (A, B, Scale, Access)>);
    pub struct MmaF16TsCta2<A, B, Scale, Access>(PhantomData<fn() -> (A, B, Scale, Access)>);
    pub struct MmaF16TsCta2Pred<A, B, Scale, Access>(PhantomData<fn() -> (A, B, Scale, Access)>);
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct E4m3;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct E5m2;
    pub struct E2m3;
    pub struct E3m2;
    pub struct E2m1;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MatrixDescriptorSm100;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MatrixDescriptorSm103;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MatrixDescriptorSm107;
    pub struct MmaF8f6f4F32SsCta1<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F32SsCta1Pred<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F32TsCta1<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F32TsCta1Pred<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F32SsCta2<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F32SsCta2Pred<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F16SsCta2<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F16SsCta2Pred<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F32TsCta2<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F32TsCta2Pred<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F16TsCta2<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F16TsCta2Pred<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F16SsCta1<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F16SsCta1Pred<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F16TsCta1<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F16TsCta1Pred<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaTf32TsCta1<Scale, Access>(PhantomData<fn() -> (Scale, Access)>);
    pub struct MmaTf32TsCta1Pred<Scale, Access>(PhantomData<fn() -> (Scale, Access)>);
    pub struct MmaTf32SsCta1<Scale, Access>(PhantomData<fn() -> (Scale, Access)>);
    pub struct MmaTf32SsCta1Pred<Scale, Access>(PhantomData<fn() -> (Scale, Access)>);
    pub struct MmaTf32SsCta2<Scale, Access>(PhantomData<fn() -> (Scale, Access)>);
    pub struct MmaTf32SsCta2Pred<Scale, Access>(PhantomData<fn() -> (Scale, Access)>);
    pub struct MmaTf32TsCta2<Scale, Access>(PhantomData<fn() -> (Scale, Access)>);
    pub struct MmaTf32TsCta2Pred<Scale, Access>(PhantomData<fn() -> (Scale, Access)>);
    pub struct MmaF8f6f4F32SsCta1Ws<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F16SsCta1Ws<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F32TsCta1Ws<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaF8f6f4F16TsCta1Ws<A, B, DescriptorLayout, Access>(
        PhantomData<fn() -> (A, B, DescriptorLayout, Access)>,
    );
    pub struct MmaTf32SsCta1Ws<Scale, Access>(PhantomData<fn() -> (Scale, Access)>);
    pub struct MmaTf32TsCta1Ws<Scale, Access>(PhantomData<fn() -> (Scale, Access)>);
    pub struct MmaBlockMxf4E8m0SsCta1<
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
        const FIXED_VECTORS: bool = false,
    >(PhantomData<fn() -> (Access, DescriptorLayout)>);
    pub struct MmaBlockMxf4E8m0SsCta2<
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
        const FIXED_VECTORS: bool = false,
    >(PhantomData<fn() -> (Access, DescriptorLayout)>);
    pub struct MmaBlockMxf4E8m0TsCta2<
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
        const FIXED_VECTORS: bool = false,
    >(PhantomData<fn() -> (Access, DescriptorLayout)>);
    pub struct MmaBlockMxf4nvf4Vec2SsCta1<
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
        const FIXED_VECTORS: bool = false,
    >(PhantomData<fn() -> (Access, DescriptorLayout)>);
    pub struct MmaBlockMxf4nvf4Vec2SsCta2<
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
        const FIXED_VECTORS: bool = false,
    >(PhantomData<fn() -> (Access, DescriptorLayout)>);
    pub struct MmaBlockMxf4nvf4Vec2TsCta1<
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
        const FIXED_VECTORS: bool = false,
    >(PhantomData<fn() -> (Access, DescriptorLayout)>);
    pub struct MmaBlockMxf4nvf4Vec2TsCta2<
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
        const FIXED_VECTORS: bool = false,
    >(PhantomData<fn() -> (Access, DescriptorLayout)>);
    pub struct MmaBlockMxf4E8m0TsCta1<
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
        const FIXED_VECTORS: bool = false,
    >(PhantomData<fn() -> (Access, DescriptorLayout)>);
    pub struct MmaBlockMxf4nvf4E2m1TsCta1<
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
        const FIXED_VECTORS: bool = false,
    >(PhantomData<fn() -> (Access, DescriptorLayout)>);
    pub struct MmaBlockMxf4nvf4E2m1TsCta2<
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
        const FIXED_VECTORS: bool = false,
    >(PhantomData<fn() -> (Access, DescriptorLayout)>);
    pub struct MmaBlockMxf4nvf4E2m1SsCta1<
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
        const FIXED_VECTORS: bool = false,
    >(PhantomData<fn() -> (Access, DescriptorLayout)>);
    pub struct MmaBlockMxf4nvf4E2m1SsCta2<
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
        const FIXED_VECTORS: bool = false,
    >(PhantomData<fn() -> (Access, DescriptorLayout)>);
    pub struct MmaBlockMxf8f6f4E8m0SsCta1<
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
        const FIXED_VECTORS: bool = false,
    >(PhantomData<fn() -> (Access, DescriptorLayout)>);
    pub struct MmaBlockMxf8f6f4E8m0SsCta2<
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
        const FIXED_VECTORS: bool = false,
    >(PhantomData<fn() -> (Access, DescriptorLayout)>);
    pub struct MmaBlockMxf8f6f4E8m0TsCta1<
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
        const FIXED_VECTORS: bool = false,
    >(PhantomData<fn() -> (Access, DescriptorLayout)>);
    pub struct MmaBlockMxf8f6f4E8m0TsCta2<
        Access,
        DescriptorLayout = MatrixDescriptorSm100,
        const FIXED_VECTORS: bool = false,
    >(PhantomData<fn() -> (Access, DescriptorLayout)>);

    pub struct MmaSpBlockMxf4E8m0SsCta1<Access>(PhantomData<fn() -> Access>);

    pub struct Shift<const CTA_GROUP: usize, Access>(PhantomData<fn() -> Access>);
}

fn begin_raw_gap(
    warp: &mut impl WarpHandle,
    context: ExecCtx,
    site: SiteId,
    gap: crate::AnalysisGapKind,
    cta_group: u32,
) -> Result<Option<crate::OperationContext>, EngineError> {
    engine(warp)
        .begin_optional_analysis_gap_internal(
            context.into_inner(),
            site.get(),
            OperationKind::TcgenWork,
            gap,
            Some(cta_group),
        )
        .map_err(Into::into)
}

instruction_variant! {
    [impl<const EXCLUSIVE: bool>] alloc_spec, variant::Alloc<EXCLUSIVE>,
    (Address<Shared>, usize, usize) => ();
    async fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        let (destination, columns, cta_group) = args;
        alloc_entry(
            warp,
            context,
            site,
            destination,
            columns,
            cta_group,
            EXCLUSIVE,
        )
        .await
    }
}

#[inline(never)]
async fn alloc_entry(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    destination: Address<Shared>,
    columns: usize,
    cta_group: usize,
    exclusive: bool,
) -> Result<(), EngineError> {
    engine(warp)
        .tcgen_alloc_instruction(
            context.into_inner(),
            site.get(),
            destination.inner(),
            columns,
            cta_group,
            exclusive,
        )
        .await
        .map_err(Into::into)
}

fn uniform_u32(values: &R<u32>, context: ExecCtx, label: &str) -> Result<u32, EngineError> {
    let mask = context.active_mask();
    let lane = mask
        .first_active()
        .ok_or_else(|| EngineError::message(format!("{label} has no issuing lane")))?;
    let value = values[lane];
    if mask.into_iter().any(|other| values[other] != value) {
        return Err(EngineError::message(format!(
            "{label} operand must be uniform across issuing lanes"
        )));
    }
    Ok(value)
}

instruction_variant! {
    [impl<const EXCLUSIVE: bool>] dealloc_spec, variant::Dealloc<EXCLUSIVE>,
    (R<u32>, usize, usize) => ();
    async fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        let (address, columns, cta_group) = args;
        dealloc_entry(warp, context, site, address, columns, cta_group, EXCLUSIVE).await
    }
}

#[inline(never)]
async fn dealloc_entry(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    address: R<u32>,
    columns: usize,
    cta_group: usize,
    exclusive: bool,
) -> Result<(), EngineError> {
    let address = uniform_u32(&address, context, "tcgen05.dealloc")?;
    let operation = begin(warp, context, site, OperationKind::Lifecycle, true)?;
    engine(warp)
        .tcgen_deallocate(operation.as_ref(), address, columns, cta_group, exclusive)
        .await?;
    finish(warp, &operation)
}

instruction_variant! {
    [impl] relinquish_alloc_permit_spec, variant::Relinquish,
    usize => ();
    async fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        cta_group: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        relinquish_entry(warp, context, site, cta_group).await
    }
}

#[inline(never)]
async fn relinquish_entry(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    cta_group: usize,
) -> Result<(), EngineError> {
    let operation = begin(warp, context, site, OperationKind::Lifecycle, true)?;
    engine(warp)
        .tcgen_relinquish(operation.as_ref(), cta_group)
        .await?;
    finish(warp, &operation)
}

#[inline(never)]
fn execute_commit(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    barrier: Address<Shared>,
    cta_masks: Option<R<i64>>,
    cta_group: u32,
    shared_a_only: bool,
) -> Result<(), EngineError> {
    let operation = begin(warp, context, site, OperationKind::AsyncIssue, true)?;
    engine(warp).tcgen_commit_issue_with_work(
        operation.as_ref(),
        barrier.inner(),
        cta_group,
        cta_masks.as_ref().map(R::inner),
        shared_a_only,
    )?;
    finish(warp, &operation)
}

macro_rules! commit_variant {
    ($marker:ident, $shared_a_only:expr, $args:ty, |$value:ident| $split:expr) => {
        instruction_variant! {
            [impl] commit_spec, variant::$marker,
            $args => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                $value: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                let (barrier, cta_masks, cta_group) = $split;
                execute_commit(
                    warp,
                    context,
                    site,
                    barrier,
                    cta_masks,
                    cta_group,
                    $shared_a_only,
                )
            }
        }
    };
}

commit_variant!(Commit, false, (Address<Shared>, u32), |args| (
    args.0, None, args.1
));
commit_variant!(
    CommitMulticast,
    false,
    (Address<Shared>, R<i64>, u32),
    |args| (args.0, Some(args.1), args.2)
);

commit_variant!(CommitSharedA, true, (Address<Shared>, u32), |args| (
    args.0, None, args.1
));
commit_variant!(
    CommitSharedAMulticast,
    true,
    (Address<Shared>, R<i64>, u32),
    |args| (args.0, Some(args.1), args.2)
);

/// Complete prior `tcgen05.ld` work. There is one supported spelling, so no
/// empty variant parameter is exposed.
#[inline(never)]
pub async fn wait_ld(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(context, site, std::any::type_name_of_val(&wait_ld));
    crate::runtime::require_full_warp_sync(
        context.active_mask().into_inner(),
        "tcgen05.wait::ld.sync.aligned",
    )
    .map_err(EngineError::from)?;
    let operation = begin(warp, context, site, OperationKind::Control, false)?;
    engine(warp).tcgen_wait_work_internal(operation.as_ref(), TcgenTransferKind::Load)?;
    finish(warp, &operation)
}

/// Complete prior `tcgen05.st` work.
#[inline(never)]
pub async fn wait_st(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(context, site, std::any::type_name_of_val(&wait_st));
    crate::runtime::require_full_warp_sync(
        context.active_mask().into_inner(),
        "tcgen05.wait::st.sync.aligned",
    )
    .map_err(EngineError::from)?;
    let operation = begin(warp, context, site, OperationKind::Control, false)?;
    engine(warp).tcgen_wait_work_internal(operation.as_ref(), TcgenTransferKind::Store)?;
    finish(warp, &operation)
}

macro_rules! fence_variant {
    ($marker:ty, $kind:expr) => {
        instruction_variant! {
            [impl] fence_spec, $marker,
            () => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                (): Self::Args,
            ) -> Result<Self::Output, EngineError> {
                let operation = begin(warp, context, site, OperationKind::Fence, false)?;
                engine(warp).tcgen_fence(operation.as_ref(), $kind)?;
                finish(warp, &operation)
            }
        }
    };
}

fence_variant!(
    variant::FenceBeforeThreadSync,
    TcgenFenceKind::BeforeThreadSync
);
fence_variant!(
    variant::FenceAfterThreadSync,
    TcgenFenceKind::AfterThreadSync
);

trait AccessMode {
    const VALUE: TmemAccessMode;
}

impl AccessMode for variant::StaticTmem {
    const VALUE: TmemAccessMode = TmemAccessMode::Static;
}

impl AccessMode for variant::DynamicTmem {
    const VALUE: TmemAccessMode = TmemAccessMode::Dynamic;
}

trait LdstShape {
    const VALUE: RawTcgenLdstShape;
    const REGISTERS_PER_NUM: usize;
}

macro_rules! ldst_shape {
    ($marker:ty, $value:ident, $registers:literal) => {
        impl LdstShape for $marker {
            const VALUE: RawTcgenLdstShape = RawTcgenLdstShape::$value;
            const REGISTERS_PER_NUM: usize = $registers;
        }
    };
}

impl<const HALF_SPLITOFF: usize> LdstShape for variant::Shape16x32bx2<HALF_SPLITOFF> {
    const VALUE: RawTcgenLdstShape = RawTcgenLdstShape::Shape16x32bx2(HALF_SPLITOFF);
    const REGISTERS_PER_NUM: usize = 1;
}
ldst_shape!(variant::Shape16x64b, Shape16x64b, 1);
ldst_shape!(variant::Shape16x128b, Shape16x128b, 2);
ldst_shape!(variant::Shape32x32b, Shape32x32b, 1);
ldst_shape!(variant::Shape16x256b, Shape16x256b, 4);

trait NumValue {
    const VALUE: usize;
}

impl<const N: usize> NumValue for variant::Num<N> {
    const VALUE: usize = N;
}

trait ValidNum<Shape>: NumValue {}

macro_rules! valid_nums {
    ($shape:ty; $($num:literal),+ $(,)?) => {
        $(impl ValidNum<$shape> for variant::Num<$num> {})+
    };
}

impl<const N: usize, const HALF_SPLITOFF: usize> ValidNum<variant::Shape16x32bx2<HALF_SPLITOFF>>
    for variant::Num<N>
where
    Self: ValidNum<variant::Shape32x32b>,
{
}
valid_nums!(variant::Shape16x64b; 1, 2, 4, 8, 16, 32, 64, 128);
valid_nums!(variant::Shape16x128b; 1, 2, 4, 8, 16, 32, 64);
valid_nums!(variant::Shape32x32b; 1, 2, 4, 8, 16, 32, 64, 128);
valid_nums!(variant::Shape16x256b; 1, 2, 4, 8, 16, 32);

fn uniform<T: Copy + PartialEq>(
    values: &R<T>,
    context: ExecCtx,
    label: &str,
) -> Result<T, EngineError> {
    let mask = context.active_mask();
    let lane = mask
        .first_active()
        .ok_or_else(|| EngineError::message(format!("{label} has no active lane")))?;
    let value = values[lane];
    if mask.into_iter().any(|other| values[other] != value) {
        return Err(EngineError::message(format!(
            "{label} must be uniform across active lanes"
        )));
    }
    Ok(value)
}

fn raw_ldst_footprint_anchor(
    anchor: &crate::runtime::RuntimeBuffer,
    itemsize: usize,
) -> Result<crate::runtime::RuntimeBuffer, EngineError> {
    raw_tmem_footprint_anchor(anchor, itemsize, 1)
}

fn raw_tmem_footprint_anchor(
    anchor: &crate::runtime::RuntimeBuffer,
    itemsize: usize,
    tcol_span_elements: usize,
) -> Result<crate::runtime::RuntimeBuffer, EngineError> {
    let crate::runtime::RuntimeBuffer::Tmem { allocations, .. } = anchor else {
        return Err(EngineError::message(
            "raw TCGEN footprint requires a TMEM anchor",
        ));
    };
    Ok(crate::runtime::RuntimeBuffer::Tmem {
        allocations: allocations.clone(),
        lane_span: 128,
        tcol_span_elements,
        elem_offset: 0,
        itemsize,
    })
}

#[allow(clippy::too_many_arguments)]
fn raw_ldst_footprints(
    context: &crate::WarpContext,
    provenance_lane: usize,
    address: u32,
    row_offset: i64,
    col_offset: i64,
    shape: RawTcgenLdstShape,
    split_cells: bool,
    register_count: usize,
) -> Result<Vec<(usize, usize, Option<usize>, i64, i64, i64, usize)>, EngineError> {
    let spans_per_register = context
        .active_mask()
        .len()
        .checked_mul(if split_cells { 2 } else { 1 })
        .ok_or_else(|| EngineError::message("raw TCGEN LD/ST footprint count overflow"))?;
    let mut footprints = Vec::with_capacity(
        register_count
            .checked_mul(spans_per_register)
            .ok_or_else(|| EngineError::message("raw TCGEN LD/ST footprint count overflow"))?,
    );
    for register_index in 0..register_count {
        for execution_lane in context.active_mask() {
            let (mapped_lane, column) = raw_tcgen05_ldst_location(
                context,
                address,
                row_offset,
                col_offset,
                shape,
                split_cells,
                register_index,
                execution_lane,
            )?;
            let cell_count = if split_cells { 2 } else { 1 };
            let access_bytes = if split_cells { 2 } else { 4 };
            for cell in 0..cell_count {
                footprints.push((
                    provenance_lane,
                    execution_lane,
                    None,
                    i64::try_from(mapped_lane)
                        .map_err(|_| EngineError::message("raw TCGEN lane exceeds i64"))?,
                    0,
                    i64::try_from(column.checked_add(cell).ok_or_else(|| {
                        EngineError::message("raw TCGEN footprint column overflow")
                    })?)
                    .map_err(|_| EngineError::message("raw TCGEN column exceeds i64"))?,
                    access_bytes,
                ));
            }
        }
    }
    Ok(footprints)
}

fn collector_transition(state: u8, fill: u8, require: u8, discard: u8) -> Result<u8, EngineError> {
    if (fill | require | discard) & !31 != 0 || fill & (require | discard) != 0 {
        return Err(EngineError::message("invalid TCGEN collector transition"));
    }
    if state & require != require {
        return Err(EngineError::message(format!(
            "tcgen05.mma collector use/lastuse requires a valid previous fill (missing slots {:#x})",
            require & !state
        )));
    }
    Ok((state | fill) & !discard)
}

macro_rules! collector_mma_variant {
    ($spec:ident) => {
        instruction_variant! {
            [impl<V: $spec::sealed::Execute, const FILL: u8, const REQUIRE: u8, const DISCARD: u8>]
            $spec, variant::MmaCollectors<V, FILL, REQUIRE, DISCARD>,
            V::Args => V::Output;
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                // The caller has already applied the instruction predicate.
                let (lane, _) = require_single_issue_lane(context, None, "tcgen05.mma collector")?
                    .expect("non-predicated single issue has a lane");
                let next = match collector_transition(
                    engine(warp).tcgen_collectors[lane],
                    FILL,
                    REQUIRE,
                    DISCARD,
                ) {
                    Ok(next) => next,
                    Err(error) => {
                        let operation = begin(warp, context, site, OperationKind::TcgenWork, true)?;
                        let error = crate::EngineError::from(error);
                        return Err(match operation {
                            Some(operation) => error.with_operation_context(&operation),
                            None => error,
                        }
                        .into());
                    }
                };
                // PTX permits opportunistic reloads, so every issue retains
                // its ordinary numeric reads and asynchronous memory effects.
                let result = V::execute(warp, context, site, args)?;
                engine(warp).tcgen_collectors[lane] = next;
                Ok(result)
            }
        }
    };
}

collector_mma_variant!(mma_spec);
collector_mma_variant!(mma_sp_spec);

trait LutBMma: mma_spec::Variant<Output = ()> {
    fn execute_lut_b(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
        lookup: R<u32>,
    ) -> Result<(), EngineError>;
}
instruction_variant! {
    [impl<V: LutBMma>] mma_spec, variant::MmaLutB<V>,
    (V::Args, R<u32>) => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (args, lookup): Self::Args,
    ) -> Result<(), EngineError> {
        V::execute_lut_b(warp, context, site, args, lookup)
    }
}

trait SparseBlockMma: mma_spec::Variant<Output = ()> {
    fn execute_sparse_block(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
        metadata: R<u32>,
    ) -> Result<(), EngineError>;
}
instruction_variant! {
    [impl<V: SparseBlockMma>] mma_spec, variant::MmaSparseBlock<V>,
    (V::Args, R<u32>) => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (args, metadata): Self::Args,
    ) -> Result<(), EngineError> {
        V::execute_sparse_block(warp, context, site, args, metadata)
    }
}

macro_rules! lut_f8_pair {
    ($plain:ident, $predicated:ident, $issue:path, $half:literal, $ts:literal $(, $ws:expr)?) => {
        impl<A: NarrowFloat, B: NarrowFloat, Layout: MatrixDescriptorLayout, Access: AccessMode>
            LutBMma for variant::$plain<A, B, Layout, Access> {
            fn execute_lut_b(warp: &mut super::Engine, context: ExecCtx, site: SiteId, args: Self::Args, lookup: R<u32>) -> Result<(), EngineError> {
                let (anchor, shared, destination, a, b, descriptor, enable_d, masks) = args;
                $issue(warp, context, site, (anchor, shared, destination, a.map(|_, value| u64::from(value)), b, descriptor, enable_d, masks),
                    None, Access::VALUE, A::FORMAT, B::FORMAT, $half, $ts, $($ws,)? Layout::VALUE, Some(lookup))
            }
        }
        impl<A: NarrowFloat, B: NarrowFloat, Layout: MatrixDescriptorLayout, Access: AccessMode>
            LutBMma for variant::$predicated<A, B, Layout, Access> {
            fn execute_lut_b(warp: &mut super::Engine, context: ExecCtx, site: SiteId, args: Self::Args, lookup: R<u32>) -> Result<(), EngineError> {
                let (anchor, shared, destination, a, b, descriptor, enable_d, masks, predicate) = args;
                $issue(warp, context, site, (anchor, shared, destination, a.map(|_, value| u64::from(value)), b, descriptor, enable_d, masks),
                    Some(predicate), Access::VALUE, A::FORMAT, B::FORMAT, $half, $ts, $($ws,)? Layout::VALUE, Some(lookup))
            }
        }
    };
}
lut_f8_pair!(
    MmaF8f6f4F32SsCta1,
    MmaF8f6f4F32SsCta1Pred,
    issue_mma_f8f6f4::<false>,
    false,
    false,
    None
);
lut_f8_pair!(
    MmaF8f6f4F16SsCta1,
    MmaF8f6f4F16SsCta1Pred,
    issue_mma_f8f6f4::<false>,
    true,
    false,
    None
);
lut_f8_pair!(
    MmaF8f6f4F32TsCta1,
    MmaF8f6f4F32TsCta1Pred,
    issue_mma_f8f6f4::<false>,
    false,
    true,
    None
);
lut_f8_pair!(
    MmaF8f6f4F16TsCta1,
    MmaF8f6f4F16TsCta1Pred,
    issue_mma_f8f6f4::<false>,
    true,
    true,
    None
);
lut_f8_pair!(
    MmaF8f6f4F32SsCta2,
    MmaF8f6f4F32SsCta2Pred,
    issue_mma_f8f6f4_cta2::<false>,
    false,
    false
);
lut_f8_pair!(
    MmaF8f6f4F16SsCta2,
    MmaF8f6f4F16SsCta2Pred,
    issue_mma_f8f6f4_cta2::<false>,
    true,
    false
);
lut_f8_pair!(
    MmaF8f6f4F32TsCta2,
    MmaF8f6f4F32TsCta2Pred,
    issue_mma_f8f6f4_cta2::<false>,
    false,
    true
);
lut_f8_pair!(
    MmaF8f6f4F16TsCta2,
    MmaF8f6f4F16TsCta2Pred,
    issue_mma_f8f6f4_cta2::<false>,
    true,
    true
);

trait AshiftMma: mma_spec::Variant<Output = ()> {
    fn execute_ashift(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<(), EngineError>;
}
instruction_variant! {
    [impl<V: AshiftMma>] mma_spec, variant::MmaAshift<V>,
    V::Args => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<(), EngineError> {
        V::execute_ashift(warp, context, site, args)
    }
}

macro_rules! ashift_mma_pair {
    ($plain:ident, $predicated:ident, <$($param:ident : $bound:path),*>, $access:ident, $issue:path, $format:expr) => {
        impl<$($param: $bound,)* $access: AccessMode> AshiftMma for variant::$plain<$($param,)* $access> {
            fn execute_ashift(warp: &mut super::Engine, context: ExecCtx, site: SiteId, args: Self::Args) -> Result<(), EngineError> {
                $issue(warp, context, site, args, None, $format, $access::VALUE)
            }
        }
        impl<$($param: $bound,)* $access: AccessMode> AshiftMma for variant::$predicated<$($param,)* $access> {
            fn execute_ashift(warp: &mut super::Engine, context: ExecCtx, site: SiteId, args: Self::Args) -> Result<(), EngineError> {
                let (anchor, shared, destination, a, b, descriptor, enable_d, masks, predicate) = args;
                $issue(warp, context, site, (anchor, shared, destination, a, b, descriptor, enable_d, masks),
                    Some(predicate), $format, $access::VALUE)
            }
        }
    };
}
ashift_mma_pair!(MmaF16TsCta1, MmaF16TsCta1Pred, <A: B16, B: B16, Scale: ValidScale>, Access, issue_mma_dense_ts::<true, 4>,
    DenseMmaFormat::Float { kind: RawTcgenFloatKind::B16 { a_bf16: A::BF16, b_bf16: B::BF16 }, scale: Scale::VALUE, ws_mask: None, metadata: None, });
ashift_mma_pair!(MmaF16TsCta2, MmaF16TsCta2Pred, <A: B16, B: B16, Scale: ValidScale>, Access, issue_mma_dense_ts::<true, 8>,
    DenseMmaFormat::Float { kind: RawTcgenFloatKind::B16 { a_bf16: A::BF16, b_bf16: B::BF16 }, scale: Scale::VALUE, ws_mask: None, metadata: None, });
ashift_mma_pair!(MmaTf32TsCta1, MmaTf32TsCta1Pred, <Scale: ValidScale>, Access, issue_mma_dense_ts::<true, 4>,
    DenseMmaFormat::Float { kind: RawTcgenFloatKind::Tf32, scale: Scale::VALUE, ws_mask: None, metadata: None });
ashift_mma_pair!(MmaTf32TsCta2, MmaTf32TsCta2Pred, <Scale: ValidScale>, Access, issue_mma_dense_ts::<true, 8>,
    DenseMmaFormat::Float { kind: RawTcgenFloatKind::Tf32, scale: Scale::VALUE, ws_mask: None, metadata: None });
ashift_mma_pair!(MmaIntegerTsCta1, MmaIntegerTsCta1Pred, <Kind: IntegerKind>, Access, issue_mma_dense_ts::<true, 4>,
    DenseMmaFormat::Integer { kind: Kind::VALUE, ws_mask: None, metadata: None });
ashift_mma_pair!(MmaIntegerTsCta2, MmaIntegerTsCta2Pred, <Kind: IntegerKind>, Access, issue_mma_dense_ts::<true, 8>,
    DenseMmaFormat::Integer { kind: Kind::VALUE, ws_mask: None, metadata: None });

macro_rules! ashift_f8_pair {
    ($plain:ident, $predicated:ident, $issue:path, $half:literal $(, $ws:expr)?) => {
        impl<A: NarrowFloat, B: NarrowFloat, Layout: MatrixDescriptorLayout, Access: AccessMode>
            AshiftMma for variant::$plain<A, B, Layout, Access> {
            fn execute_ashift(warp: &mut super::Engine, context: ExecCtx, site: SiteId, args: Self::Args) -> Result<(), EngineError> {
                let (anchor, shared, destination, a, b, descriptor, enable_d, masks) = args;
                $issue(warp, context, site, (anchor, shared, destination, a.map(|_, value| u64::from(value)), b, descriptor, enable_d, masks),
                    None, Access::VALUE, A::FORMAT, B::FORMAT, $half, true, $($ws,)? Layout::VALUE, None)
            }
        }
        impl<A: NarrowFloat, B: NarrowFloat, Layout: MatrixDescriptorLayout, Access: AccessMode>
            AshiftMma for variant::$predicated<A, B, Layout, Access> {
            fn execute_ashift(warp: &mut super::Engine, context: ExecCtx, site: SiteId, args: Self::Args) -> Result<(), EngineError> {
                let (anchor, shared, destination, a, b, descriptor, enable_d, masks, predicate) = args;
                $issue(warp, context, site, (anchor, shared, destination, a.map(|_, value| u64::from(value)), b, descriptor, enable_d, masks),
                    Some(predicate), Access::VALUE, A::FORMAT, B::FORMAT, $half, true, $($ws,)? Layout::VALUE, None)
            }
        }
    };
}
ashift_f8_pair!(
    MmaF8f6f4F32TsCta1,
    MmaF8f6f4F32TsCta1Pred,
    issue_mma_f8f6f4::<true>,
    false,
    None
);
ashift_f8_pair!(
    MmaF8f6f4F16TsCta1,
    MmaF8f6f4F16TsCta1Pred,
    issue_mma_f8f6f4::<true>,
    true,
    None
);
ashift_f8_pair!(
    MmaF8f6f4F32TsCta2,
    MmaF8f6f4F32TsCta2Pred,
    issue_mma_f8f6f4_cta2::<true>,
    false
);
ashift_f8_pair!(
    MmaF8f6f4F16TsCta2,
    MmaF8f6f4F16TsCta2Pred,
    issue_mma_f8f6f4_cta2::<true>,
    true
);

macro_rules! ashift_sparse_mma {
    ($marker:ident, $issue:path) => {
        impl<Kind: SparseKind, Access: AccessMode> AshiftMma for variant::$marker<Kind, Access> {
            fn execute_ashift(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
            ) -> Result<(), EngineError> {
                let (
                    anchor,
                    shared,
                    destination,
                    a,
                    b,
                    descriptor,
                    enable_d,
                    masks,
                    _zero_mask,
                    metadata,
                ) = args;
                let Some((lane, context)) =
                    require_single_issue_lane(context, None, "tcgen05.mma.sp.ashift")?
                else {
                    return Ok(());
                };
                $issue(
                    warp,
                    context,
                    site,
                    (
                        anchor,
                        shared,
                        destination,
                        a,
                        b,
                        descriptor,
                        enable_d,
                        masks,
                    ),
                    None,
                    Kind::format(metadata[lane], None),
                    Access::VALUE,
                )
            }
        }
    };
}
ashift_sparse_mma!(MmaSparseTsCta1, issue_mma_dense_ts::<true, 4>);
ashift_sparse_mma!(MmaSparseTsCta2, issue_mma_dense_ts::<true, 8>);


fn require_single_issue_lane(
    context: ExecCtx,
    predicate: Option<&R<u32>>,
    label: &str,
) -> Result<Option<(usize, ExecCtx)>, EngineError> {
    let active = context.active_mask();
    let lane = if let Some(predicate) = predicate {
        let mut selected = active.into_iter().filter(|&lane| predicate[lane] != 0);
        let first = selected.next();
        if selected.next().is_some() {
            return Err(EngineError::message(format!(
                "{label} predicate selects more than one issuing lane"
            )));
        }
        let Some(first) = first else {
            return Ok(None);
        };
        first
    } else {
        if active.len() != 1 {
            return Err(EngineError::message(format!(
                "{label} requires exactly one issuing lane"
            )));
        }
        active
            .first_active()
            .ok_or_else(|| EngineError::message(format!("{label} has no issuing lane")))?
    };
    let selected = crate::WarpMask::from_bits(1_u32 << lane);
    Ok(Some((
        lane,
        ExecCtx::from_inner(context.into_inner().with_active_mask(selected)),
    )))
}

trait LoadReduction {
    const REDUCE: Option<fn(u32, u32) -> u32>;
    const COMPRESS: Option<(bool, bool)> = None;
}

impl<const MAX: bool, const ABS: bool, Reduction: LoadReduction> LoadReduction
    for variant::Compress<MAX, ABS, Reduction>
{
    const REDUCE: Option<fn(u32, u32) -> u32> = Reduction::REDUCE;
    const COMPRESS: Option<(bool, bool)> = Some((MAX, ABS));
}

impl LoadReduction for variant::NoReduction {
    const REDUCE: Option<fn(u32, u32) -> u32> = None;
}

impl<const MAX: bool, const ABS: bool, const NAN: bool> LoadReduction
    for variant::ReduceF32<MAX, ABS, NAN>
{
    const REDUCE: Option<fn(u32, u32) -> u32> = Some(|lhs, rhs| {
        let value = |bits| {
            let value = f32::from_bits(bits);
            if ABS {
                value.abs()
            } else {
                value
            }
        };
        let (lhs, rhs) = (value(lhs), value(rhs));
        if MAX {
            crate::scalar::ptx_max_f32(lhs, rhs, false, NAN).to_bits()
        } else {
            crate::scalar::ptx_min_f32(lhs, rhs, false, NAN).to_bits()
        }
    });
}

impl<const MAX: bool> LoadReduction for variant::ReduceU32<MAX> {
    const REDUCE: Option<fn(u32, u32) -> u32> =
        Some(|lhs, rhs| if MAX { lhs.max(rhs) } else { lhs.min(rhs) });
}

impl<const MAX: bool> LoadReduction for variant::ReduceI32<MAX> {
    const REDUCE: Option<fn(u32, u32) -> u32> = Some(|lhs, rhs| {
        let (lhs, rhs) = (lhs as i32, rhs as i32);
        (if MAX { lhs.max(rhs) } else { lhs.min(rhs) }) as u32
    });
}

instruction_variant! {
    [impl<Shape, Num, const PACKED: bool, Access, Reduction>] ld_spec, variant::Ld<Shape, Num, PACKED, Access, Reduction>
    where [
        Shape: LdstShape,
        Num: ValidNum<Shape>,
        Access: AccessMode,
        Reduction: LoadReduction,
    ],
    (
        BufferHandle<Tmem>,
        R<u32>,
        R<i64>,
        R<i64>,
        Vec<Address<Register>>,
    ) => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        execute_ld_entry(
            warp,
            context,
            site,
            args,
            Shape::VALUE,
            Shape::REGISTERS_PER_NUM,
            Num::VALUE,
            PACKED,
            Access::VALUE,
            Reduction::REDUCE,
            Reduction::COMPRESS,
        )
    }
}

#[inline(never)]
fn execute_ld_entry(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    (anchor, address, row, column, mut destinations): (
        BufferHandle<Tmem>,
        R<u32>,
        R<i64>,
        R<i64>,
        Vec<Address<Register>>,
    ),
    shape: RawTcgenLdstShape,
    registers_per_num: usize,
    num: usize,
    packed: bool,
    access: TmemAccessMode,
    reduction: Option<fn(u32, u32) -> u32>,
    compression: Option<(bool, bool)>,
) -> Result<(), EngineError> {
    crate::runtime::require_full_warp_sync(context.active_mask().into_inner(), "tcgen05.ld")?;
    let expected = registers_per_num
        .checked_mul(num)
        .ok_or_else(|| EngineError::message("tcgen05.ld register count overflow"))?;
    if reduction.is_some()
        && (packed
            || num < 2
            || !matches!(
                shape,
                RawTcgenLdstShape::Shape32x32b | RawTcgenLdstShape::Shape16x32bx2(_)
            ))
    {
        return Err(EngineError::message(
            "tcgen05.ld.red requires an unpacked 32x32b/16x32bx2 load with at least x2",
        ));
    }
    if compression.is_some()
        && (packed || num < 4 || !matches!(shape, RawTcgenLdstShape::Shape32x32b))
    {
        return Err(EngineError::message(
            "tcgen05.ld.spcompress requires unpacked 32x32b with at least x4",
        ));
    }
    let expected_destinations = if compression.is_some() {
        num.div_ceil(32) + num / 2
    } else {
        expected
    } + usize::from(reduction.is_some());
    if destinations.len() != expected_destinations {
        return Err(EngineError::message(format!(
            "tcgen05.ld variant requires {expected_destinations} destination registers, got {}",
            destinations.len()
        )));
    }
    let reduction_destination = if reduction.is_some() {
        destinations.pop()
    } else {
        None
    };
    let address = uniform(&address, context, "tcgen05.ld address")?;
    let row = uniform(&row, context, "tcgen05.ld row offset")?;
    let column = uniform(&column, context, "tcgen05.ld column offset")?;
    let base_context = context.into_inner();
    let issuer_lane = context
        .active_mask()
        .first_active()
        .expect("full-warp TCGEN load has an issuer");
    let issue_context = ExecCtx::from_inner(
        base_context.with_active_mask(crate::WarpMask::from_bits(1_u32 << issuer_lane)),
    );
    let numeric_context = base_context;
    let physical = engine(warp).kernel().physical().clone();
    let lifecycle = engine(warp).kernel().services().tcgen();
    let footprint_anchor = raw_ldst_footprint_anchor(anchor.inner(), if packed { 2 } else { 4 })?;
    let logical_buffer = anchor
        .logical_buffer()
        .unwrap_or("raw_tcgen05_ld_source")
        .to_owned();
    let footprints = raw_ldst_footprints(
        &numeric_context,
        issuer_lane,
        address,
        row,
        column,
        shape,
        packed,
        expected,
    )?;
    let operation = begin(warp, issue_context, site, OperationKind::TcgenWork, true)?;
    engine(warp).tcgen_instruction_issue(
        operation.as_ref(),
        1,
        TcgenPipelineOperation::Load,
        None,
        |_, record_tmem| {
            record_tmem(
                OperationKind::Load,
                &footprint_anchor,
                &logical_buffer,
                access,
                &footprints,
            )
        },
        || {
            if let Some((maximum, absolute)) = compression {
                let mut values = vec![[0_u32; 32]; num];
                let mut valid = vec![[false; 32]; num];
                for index in 0..num {
                    crate::runtime::raw_tcgen05_read_register(
                        &physical,
                        &numeric_context,
                        &lifecycle,
                        access,
                        anchor.inner(),
                        address,
                        row,
                        column,
                        shape,
                        packed,
                        index,
                        numeric_context.active_mask(),
                        |lane, bytes, validity| {
                            values[index][lane] = u32::from_le_bytes(bytes);
                            valid[index][lane] = validity.iter().all(|v| *v);
                            Ok(())
                        },
                    )?;
                }
                for lane in numeric_context.active_mask() {
                    let metadata_count = num.div_ceil(32);
                    let mut output = vec![0_u32; destinations.len()];
                    let mut output_valid = vec![true; destinations.len()];
                    for group in 0..num / 4 {
                        let group_valid = (0..4).all(|i| valid[group * 4 + i][lane]);
                        let pair = super::reg::sparse_pair_indices(
                            std::array::from_fn(|i| {
                                let value = f32::from_bits(values[group * 4 + i][lane]);
                                if absolute {
                                    value.abs()
                                } else {
                                    value
                                }
                            }),
                            maximum,
                        );
                        for (j, index) in pair.into_iter().enumerate() {
                            let element = group * 2 + j;
                            output[element / 16] |= (index as u32) << ((element % 16) * 2);
                            output_valid[element / 16] &= group_valid;
                            output[metadata_count + element] = values[group * 4 + index][lane];
                            output_valid[metadata_count + element] = group_valid;
                        }
                    }
                    for (index, destination) in destinations.iter().enumerate() {
                        let destination = destination.inner();
                        crate::runtime::write_runtime_bytes_with_validity(
                            &physical,
                            &numeric_context,
                            destination.buffer(),
                            lane,
                            destination.lane_write_byte_offset(lane, 4)?,
                            &output[index].to_le_bytes(),
                            &[output_valid[index]; 4],
                        )?;
                    }
                    if let (Some(reduce), Some(destination)) = (reduction, &reduction_destination) {
                        let value = values.iter().map(|v| v[lane]).reduce(reduce).unwrap();
                        let initialized = valid.iter().all(|v| v[lane]);
                        let destination = destination.inner();
                        crate::runtime::write_runtime_bytes_with_validity(
                            &physical,
                            &numeric_context,
                            destination.buffer(),
                            lane,
                            destination.lane_write_byte_offset(lane, 4)?,
                            &value.to_le_bytes(),
                            &[initialized; 4],
                        )?;
                    }
                }
                return Ok(());
            }
            for (register_index, destination) in destinations.iter().enumerate() {
                raw_tcgen05_ld_register(
                    &physical,
                    &numeric_context,
                    &lifecycle,
                    access,
                    anchor.inner(),
                    destination.inner(),
                    address,
                    row,
                    column,
                    shape,
                    packed,
                    register_index,
                    numeric_context.active_mask(),
                )?;
            }
            if let (Some(reduce), Some(destination)) = (reduction, &reduction_destination) {
                // Reduction consumes the values just loaded, not a second
                // TMEM read. It shares the load's footprint and lifetime.
                for lane in numeric_context.active_mask() {
                    let mut result = None;
                    for source in &destinations {
                        let source = source.inner();
                        let mut bytes = [0; 4];
                        crate::runtime::read_runtime_bytes_into(
                            &physical,
                            &numeric_context,
                            source.buffer(),
                            lane,
                            source.lane_read_byte_offset(lane, 4)?,
                            &mut bytes,
                        )?;
                        let value = u32::from_le_bytes(bytes);
                        result = Some(result.map_or(value, |prior| reduce(prior, value)));
                    }
                    let destination = destination.inner();
                    crate::runtime::write_runtime_bytes(
                        &physical,
                        &numeric_context,
                        destination.buffer(),
                        lane,
                        destination.lane_write_byte_offset(lane, 4)?,
                        &result.expect("at least two reduction inputs").to_le_bytes(),
                    )?;
                }
            }
            Ok(())
        },
    )?;
    finish(warp, &operation)
}

instruction_variant! {
    [impl<Shape, Num, const UNPACKED: bool, Access>] st_spec, variant::St<Shape, Num, UNPACKED, Access>
    where [
        Shape: LdstShape,
        Num: ValidNum<Shape>,
        Access: AccessMode,
    ],
    (BufferHandle<Tmem>, R<u32>, R<i64>, R<i64>, Vec<R<u32>>) => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        execute_st_entry(
            warp,
            context,
            site,
            args,
            Shape::VALUE,
            Shape::REGISTERS_PER_NUM,
            Num::VALUE,
            UNPACKED,
            Access::VALUE,
        )
    }
}

#[inline(never)]
fn execute_st_entry(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    (anchor, address, row, column, sources): (
        BufferHandle<Tmem>,
        R<u32>,
        R<i64>,
        R<i64>,
        Vec<R<u32>>,
    ),
    shape: RawTcgenLdstShape,
    registers_per_num: usize,
    num: usize,
    unpacked: bool,
    access: TmemAccessMode,
) -> Result<(), EngineError> {
    crate::runtime::require_full_warp_sync(context.active_mask().into_inner(), "tcgen05.st")?;
    let expected = registers_per_num
        .checked_mul(num)
        .ok_or_else(|| EngineError::message("tcgen05.st register count overflow"))?;
    if sources.len() != expected {
        return Err(EngineError::message(format!(
            "tcgen05.st variant requires {expected} source registers, got {}",
            sources.len()
        )));
    }
    let address = uniform(&address, context, "tcgen05.st address")?;
    let row = uniform(&row, context, "tcgen05.st row offset")?;
    let column = uniform(&column, context, "tcgen05.st column offset")?;
    let base_context = context.into_inner();
    let issuer_lane = context
        .active_mask()
        .first_active()
        .expect("full-warp TCGEN store has an issuer");
    let issue_context = ExecCtx::from_inner(
        base_context.with_active_mask(crate::WarpMask::from_bits(1_u32 << issuer_lane)),
    );
    let numeric_context = base_context;
    let physical = engine(warp).kernel().physical().clone();
    let lifecycle = engine(warp).kernel().services().tcgen();
    let footprint_anchor = raw_ldst_footprint_anchor(anchor.inner(), if unpacked { 2 } else { 4 })?;
    let logical_buffer = anchor
        .logical_buffer()
        .unwrap_or("raw_tcgen05_st_destination")
        .to_owned();
    let footprints = raw_ldst_footprints(
        &numeric_context,
        issuer_lane,
        address,
        row,
        column,
        shape,
        unpacked,
        sources.len(),
    )?;
    let operation = begin(warp, issue_context, site, OperationKind::TcgenWork, true)?;
    engine(warp).tcgen_instruction_issue(
        operation.as_ref(),
        1,
        TcgenPipelineOperation::Store,
        None,
        |_, record_tmem| {
            record_tmem(
                OperationKind::Store,
                &footprint_anchor,
                &logical_buffer,
                access,
                &footprints,
            )
        },
        || {
            for (register_index, source) in sources.iter().enumerate() {
                raw_tcgen05_st_register(
                    &physical,
                    &numeric_context,
                    &lifecycle,
                    access,
                    anchor.inner(),
                    source.inner(),
                    address,
                    row,
                    column,
                    shape,
                    unpacked,
                    register_index,
                    numeric_context.active_mask(),
                )?;
            }
            Ok(())
        },
    )?;
    finish(warp, &operation)
}

trait CpShape {
    const CODE: u8;
}

macro_rules! cp_shape {
    ($marker:ty, $code:literal) => {
        impl CpShape for $marker {
            const CODE: u8 = $code;
        }
    };
}

cp_shape!(variant::Cp32x128bWarpx4, 0);
cp_shape!(variant::Cp64x128bWarpx2_02_13, 1);
cp_shape!(variant::Cp128x128b, 2);
cp_shape!(variant::Cp128x256b, 3);
cp_shape!(variant::Cp4x256b, 4);
cp_shape!(variant::Cp64x128bWarpx2_01_23, 5);

trait Decompress {
    const CODE: u8;
}

impl Decompress for variant::NoDecompress {
    const CODE: u8 = 0;
}
impl Decompress for variant::DecompressB4 {
    const CODE: u8 = 1;
}
impl Decompress for variant::DecompressB6 {
    const CODE: u8 = 2;
}

struct CtaGroup<const VALUE: usize>;
trait ValidCtaGroup {}
impl ValidCtaGroup for CtaGroup<1> {}
impl ValidCtaGroup for CtaGroup<2> {}

instruction_variant! {
    [impl<
        Shape,
        Decompression,
        const CTA_GROUP: usize,
        Access,
        DescriptorLayout: MatrixDescriptorLayout,
    >] cp_spec, variant::Cp<Shape, Decompression, CTA_GROUP, Access, DescriptorLayout>
    where [
        Shape: CpShape,
        Decompression: Decompress,
        Access: AccessMode,
        CtaGroup<CTA_GROUP>: ValidCtaGroup,
    ],
    (
        BufferHandle<Tmem>,
        DescriptorDomain<Shared>,
        R<u32>,
        R<u64>,
        R<i64>,
        R<i64>,
    ) => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        execute_cp_entry(
            warp,
            context,
            site,
            args,
            Shape::CODE,
            Decompression::CODE,
            CTA_GROUP,
            Access::VALUE,
            DescriptorLayout::VALUE,
        )
    }
}

#[inline(never)]
fn execute_cp_entry(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    (anchor, shared, address, descriptor, row, column): (
        BufferHandle<Tmem>,
        DescriptorDomain<Shared>,
        R<u32>,
        R<u64>,
        R<i64>,
        R<i64>,
    ),
    shape: u8,
    decompression: u8,
    cta_group: usize,
    access: TmemAccessMode,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
) -> Result<(), EngineError> {
    let Some((lane, issue_context)) = require_single_issue_lane(context, None, "tcgen05.cp")?
    else {
        unreachable!()
    };
    let address = address[lane];
    let descriptor = descriptor[lane];
    let row = row[lane];
    let column = column[lane];
    let numeric_context = issue_context.into_inner();
    let physical = engine(warp).kernel().physical().clone();
    let lifecycle = engine(warp).kernel().services().tcgen();
    let operation = begin(warp, issue_context, site, OperationKind::TcgenWork, true)?;
    let shared_candidates = shared.inner().iter().collect::<Vec<_>>();
    let destination_logical_buffer = anchor
        .logical_buffer()
        .unwrap_or("raw_tcgen05_cp_destination")
        .to_owned();
    engine(warp).tcgen_instruction_issue(
        operation.as_ref(),
        cta_group as u32,
        if shape == 4 {
            TcgenPipelineOperation::Copy4x256b
        } else {
            TcgenPipelineOperation::Copy
        },
        None,
        |record_runtime, record_tmem| {
            let footprints = raw_tcgen05_cp_footprints(
                &numeric_context,
                &shared_candidates,
                address,
                descriptor,
                row,
                column,
                shape,
                decompression,
                cta_group,
                lane,
                descriptor_layout,
            )?;
            record_runtime(
                false,
                OperationKind::Load,
                &footprints.source,
                None,
                &footprints.source_accesses,
            )?;
            let destination_footprint_anchor =
                raw_tmem_footprint_anchor(anchor.inner(), 4, footprints.words)?;
            record_tmem(
                OperationKind::Store,
                &destination_footprint_anchor,
                &destination_logical_buffer,
                access,
                &footprints.destination_accesses,
            )
        },
        || {
            raw_tcgen05_cp(
                &physical,
                &numeric_context,
                &lifecycle,
                access,
                anchor.inner(),
                &shared_candidates,
                address,
                descriptor,
                row,
                column,
                shape,
                decompression,
                cta_group,
                descriptor_layout,
            )
        },
    )?;
    finish(warp, &operation)
}

trait B16 {
    const BF16: bool;
}
impl B16 for variant::Fp16 {
    const BF16: bool = false;
}
impl B16 for variant::Bf16 {
    const BF16: bool = true;
}

/// One operand codec is shared with block-scaled MMA.
trait NarrowFloat {
    const FORMAT: RawTcgenNarrowFormat;
}
impl NarrowFloat for variant::E4m3 {
    const FORMAT: RawTcgenNarrowFormat = RawTcgenNarrowFormat::E4M3;
}
impl NarrowFloat for variant::E5m2 {
    const FORMAT: RawTcgenNarrowFormat = RawTcgenNarrowFormat::E5M2;
}
impl NarrowFloat for variant::E2m3 {
    const FORMAT: RawTcgenNarrowFormat = RawTcgenNarrowFormat::E2M3;
}
impl NarrowFloat for variant::E3m2 {
    const FORMAT: RawTcgenNarrowFormat = RawTcgenNarrowFormat::E3M2;
}
impl NarrowFloat for variant::E2m1 {
    const FORMAT: RawTcgenNarrowFormat = RawTcgenNarrowFormat::E2M1;
}

trait MatrixDescriptorLayout {
    const VALUE: RawTcgenMatrixDescriptorLayout;
}
impl MatrixDescriptorLayout for variant::MatrixDescriptorSm100 {
    const VALUE: RawTcgenMatrixDescriptorLayout = RawTcgenMatrixDescriptorLayout::Sm100;
}
impl MatrixDescriptorLayout for variant::MatrixDescriptorSm103 {
    const VALUE: RawTcgenMatrixDescriptorLayout = RawTcgenMatrixDescriptorLayout::Sm103;
}
impl MatrixDescriptorLayout for variant::MatrixDescriptorSm107 {
    const VALUE: RawTcgenMatrixDescriptorLayout = RawTcgenMatrixDescriptorLayout::Sm107;
}

trait ValidScale {
    const VALUE: usize;
}

macro_rules! scales {
    ($($value:literal),+ $(,)?) => {
        $(impl ValidScale for variant::Scale<$value> {
            const VALUE: usize = $value;
        })+
    };
}

scales!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

type WsMmaArgs<A> = (
    BufferHandle<Tmem>,
    DescriptorDomain<Shared>,
    R<u32>,
    R<A>,
    R<u64>,
    R<u32>,
    R<bool>,
    R<u64>,
);

type MmaArgs<A, const MASKS: usize> = (
    BufferHandle<Tmem>,
    DescriptorDomain<Shared>,
    R<u32>,
    R<A>,
    R<u64>,
    R<u32>,
    R<bool>,
    [R<u32>; MASKS],
);
type MmaPredArgs<A, const MASKS: usize> = (
    BufferHandle<Tmem>,
    DescriptorDomain<Shared>,
    R<u32>,
    R<A>,
    R<u64>,
    R<u32>,
    R<bool>,
    [R<u32>; MASKS],
    R<u32>,
);

trait IntegerKind {
    const VALUE: crate::runtime::RawTcgenIntegerKind;
}
impl IntegerKind for variant::Ti16 {
    const VALUE: crate::runtime::RawTcgenIntegerKind = crate::runtime::RawTcgenIntegerKind::Ti16;
}
impl IntegerKind for variant::I8 {
    const VALUE: crate::runtime::RawTcgenIntegerKind = crate::runtime::RawTcgenIntegerKind::I8;
}

trait SparseKind {
    fn format(metadata: u32, ws_mask: Option<u64>) -> DenseMmaFormat;
}

impl SparseKind for variant::Ti16 {
    fn format(metadata: u32, ws_mask: Option<u64>) -> DenseMmaFormat {
        DenseMmaFormat::Integer {
            kind: Self::VALUE,
            ws_mask,
            metadata: Some(metadata),
        }
    }
}

impl SparseKind for super::reg::variant::Tf32 {
    fn format(metadata: u32, ws_mask: Option<u64>) -> DenseMmaFormat {
        DenseMmaFormat::Float {
            kind: RawTcgenFloatKind::Tf32,
            scale: 0,
            ws_mask,
            metadata: Some(metadata),
        }
    }
}

impl<A: B16, B: B16> SparseKind for (A, B) {
    fn format(metadata: u32, ws_mask: Option<u64>) -> DenseMmaFormat {
        DenseMmaFormat::Float {
            kind: RawTcgenFloatKind::B16 {
                a_bf16: A::BF16,
                b_bf16: B::BF16,
            },
            scale: 0,
            ws_mask,
            metadata: Some(metadata),
        }
    }
}

impl<A: NarrowFloat, B: NarrowFloat, Layout: MatrixDescriptorLayout> SparseKind for (A, B, Layout) {
    fn format(metadata: u32, ws_mask: Option<u64>) -> DenseMmaFormat {
        DenseMmaFormat::Float {
            kind: RawTcgenFloatKind::SparseNarrow {
                a_format: A::FORMAT,
                b_format: B::FORMAT,
                descriptor_layout: Layout::VALUE,
            },
            scale: 0,
            ws_mask,
            metadata: Some(metadata),
        }
    }
}

#[derive(Clone, Copy)]
enum DenseMmaFormat {
    Float {
        kind: RawTcgenFloatKind,
        scale: usize,
        ws_mask: Option<u64>,
        metadata: Option<u32>,
    },
    Integer {
        kind: crate::runtime::RawTcgenIntegerKind,
        ws_mask: Option<u64>,
        metadata: Option<u32>,
    },
}

impl DenseMmaFormat {
    fn ws_mask(self) -> Option<u64> {
        match self {
            Self::Integer { ws_mask, .. } | Self::Float { ws_mask, .. } => ws_mask,
        }
    }
    fn metadata(self) -> Option<u32> {
        match self {
            Self::Integer { metadata, .. } | Self::Float { metadata, .. } => metadata,
        }
    }
    fn metadata_layout(self, descriptor: u32) -> crate::runtime::RawTcgenSparseMetadataLayout {
        match self {
            Self::Float { kind, .. } => kind.metadata_layout(descriptor),
            Self::Integer { .. } => crate::runtime::RawTcgenSparseMetadataLayout::B16 {
                selector: (descriptor & 1) as usize,
            },
        }
    }
    fn float_kind(self) -> Option<(RawTcgenFloatKind, usize)> {
        match self {
            Self::Float { kind, scale, .. } => Some((kind, scale)),
            Self::Integer { .. } => None,
        }
    }
    fn integer_kind(self) -> crate::runtime::RawTcgenIntegerKind {
        match self {
            Self::Integer { kind, .. } => kind,
            Self::Float { .. } => unreachable!("integer-only MMA path"),
        }
    }
    fn packed_k(self, descriptor: u32) -> usize {
        match self.float_kind() {
            Some((kind, _)) => kind.packed_k(descriptor),
            None => self.integer_kind().packed_k(),
        }
    }
    fn k(self, descriptor: u32) -> usize {
        self.packed_k(descriptor) * if self.metadata().is_some() { 2 } else { 1 }
    }
    fn tmem_a_columns(self, descriptor: u32) -> usize {
        match self.float_kind() {
            Some((kind, _)) => kind.tmem_a_columns(descriptor),
            None => 8,
        }
    }
    fn weight_stationary(self) -> bool {
        self.ws_mask().is_some()
    }
    fn shape(
        self,
        descriptor: u32,
        cta_group: usize,
    ) -> Result<(usize, usize, bool, bool), crate::EngineError> {
        match self {
            Self::Float { kind, .. } => crate::runtime::raw_tcgen05_float_shape(
                kind,
                descriptor,
                cta_group,
                self.weight_stationary(),
                self.metadata().is_some(),
            ),
            Self::Integer {
                kind,
                ws_mask,
                metadata,
            } => crate::runtime::raw_tcgen05_integer_shape(
                kind,
                descriptor,
                cta_group,
                ws_mask.is_some(),
                metadata.is_some(),
            ),
        }
    }
    fn accumulator(self, descriptor: u32) -> TcgenAccumulatorDtype {
        match self {
            Self::Float { .. } if descriptor & (1 << 4) == 0 => TcgenAccumulatorDtype::F16,
            Self::Float { .. } => TcgenAccumulatorDtype::F32,
            Self::Integer { .. } => TcgenAccumulatorDtype::I32,
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn shared_footprints(
        self,
        context: &crate::WarpContext,
        candidates: &[&crate::runtime::RuntimeBuffer],
        bits: u64,
        rows: usize,
        columns: usize,
        transpose: bool,
        lane: usize,
        cta_group: usize,
        mask: Option<crate::runtime::RawTcgenColumnMask>,
        is_b: bool,
    ) -> Result<
        (
            crate::runtime::RuntimeBuffer,
            Vec<(usize, Option<usize>, usize, usize)>,
        ),
        crate::EngineError,
    > {
        match self {
            Self::Integer { kind, .. } => crate::runtime::raw_tcgen05_integer_shared_footprints(
                kind, context, candidates, bits, rows, columns, transpose, lane, cta_group, mask,
            ),
            Self::Float { kind, .. } => crate::runtime::raw_tcgen05_float_shared_footprints(
                kind, context, candidates, bits, rows, columns, transpose, lane, cta_group, mask,
                is_b,
            ),
        }
    }
}

fn dense_mma_destination_footprints<const MASKS: usize>(
    context: &crate::WarpContext,
    destination: u32,
    m: usize,
    n: usize,
    layout: crate::runtime::tcgen_ops::RawTcgenDenseTmemLayout,
    masks: [u32; MASKS],
    lane: usize,
) -> Result<Vec<crate::runtime::tcgen_ops::RawTcgenTmemAccess>, crate::EngineError> {
    match MASKS {
        4 => raw_tcgen05_dense_f32_tmem_footprints(
            destination,
            m,
            n,
            layout,
            std::array::from_fn(|i| masks[i]),
            lane,
            lane,
        ),
        8 => crate::runtime::raw_tcgen05_cta2_layout_tmem_footprints(
            context,
            destination,
            m,
            n,
            layout,
            std::array::from_fn(|i| masks[i]),
            lane,
            lane,
        ),
        _ => unreachable!("dense MMA has four output masks per CTA and one or two CTAs"),
    }
}

#[inline(never)]
fn issue_mma_dense_ss<const MASKS: usize>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    args: MmaArgs<u64, MASKS>,
    predicate: Option<R<u32>>,
    format: DenseMmaFormat,
    access: TmemAccessMode,
) -> Result<(), EngineError> {
    issue_mma_dense::<_, false, MASKS>(
        warp,
        context,
        site,
        args,
        predicate,
        format,
        access,
        RawTcgenMmaA::Shared,
    )
}

macro_rules! dense_mma_variants {
    ($plain:ident, $predicated:ident, <$($param:ident : $bound:path),*>,
     $args:ty, $pred_args:ty, $issue:path, $format:expr) => {
        instruction_variant! {
            [impl<$($param: $bound,)* Access: AccessMode>]
            mma_spec, variant::$plain<$($param,)* Access>,
            $args => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                $issue(
                    warp,
                    context,
                    site,
                    args,
                    None,
                    $format,
                    Access::VALUE,
                )
            }
        }

        instruction_variant! {
            [impl<$($param: $bound,)* Access: AccessMode>]
            mma_spec, variant::$predicated<$($param,)* Access>,
            $pred_args => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                let (anchor, shared, destination, a, b, descriptor, enable_d, masks, predicate) =
                    args;
                $issue(
                    warp,
                    context,
                    site,
                    (
                        anchor,
                        shared,
                        destination,
                        a,
                        b,
                        descriptor,
                        enable_d,
                        masks,
                    ),
                    Some(predicate),
                    $format,
                    Access::VALUE,
                )
            }
        }
    };
}

macro_rules! f16_mma_variants {
    ($plain:ident, $predicated:ident, $args:ty, $pred_args:ty, $issue:path) => {
        dense_mma_variants!(
            $plain, $predicated, <A: B16, B: B16, Scale: ValidScale>,
            $args, $pred_args, $issue,
            DenseMmaFormat::Float {
                kind: RawTcgenFloatKind::B16 { a_bf16: A::BF16, b_bf16: B::BF16 },
                scale: Scale::VALUE,
                ws_mask: None,
                metadata: None,
            }
        );
    };
}

macro_rules! integer_mma_variants {
    ($plain:ident, $predicated:ident, $args:ty, $pred_args:ty, $issue:path) => {
        dense_mma_variants!(
            $plain, $predicated, <Kind: IntegerKind>, $args, $pred_args, $issue,
            DenseMmaFormat::Integer { kind: Kind::VALUE, ws_mask: None, metadata: None }
        );
    };
}

macro_rules! tf32_mma_variants {
    ($plain:ident, $predicated:ident, $issue:path, $args:ty, $pred_args:ty) => {
        dense_mma_variants!(
            $plain, $predicated, <Scale: ValidScale>, $args, $pred_args, $issue,
            DenseMmaFormat::Float {
                kind: RawTcgenFloatKind::Tf32, scale: Scale::VALUE,
                ws_mask: None, metadata: None,
            }
        );
    };
}

f16_mma_variants!(
    MmaF16SsCta1,
    MmaF16SsCta1Pred,
    MmaArgs<u64, 4>,
    MmaPredArgs<u64, 4>,
    issue_mma_dense_ss
);

macro_rules! dense_ws_mma_variant {
    ($marker:ident, <$($param:ident: $bound:path),*>, $issue:path, $a:ty,
     |$mask:ident| $format:expr) => {
        instruction_variant! {
            [impl<$($param: $bound,)* Access: AccessMode>]
            mma_spec, variant::$marker<$($param,)* Access>,
            WsMmaArgs<$a> => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                (anchor, shared, destination, a, b, descriptor, enable_d, mask): Self::Args,
            ) -> Result<(), EngineError> {
                let Some((lane, issue_context)) =
                    require_single_issue_lane(context, None, "tcgen05.mma.ws")?
                else {
                    return Ok(());
                };
                let $mask = mask[lane];
                $issue(
                    warp,
                    issue_context,
                    site,
                    (
                        anchor,
                        shared,
                        destination,
                        a,
                        b,
                        descriptor,
                        enable_d,
                        std::array::from_fn(|_| R::splat(0)),
                    ),
                    None,
                    $format,
                    Access::VALUE,
                )
            }
        }
    };
}

macro_rules! integer_ws_variant {
    ($marker:ident, $issue:path, $a:ty) => {
        dense_ws_mma_variant!(
            $marker, <Kind: IntegerKind>, $issue, $a, |mask|
            DenseMmaFormat::Integer {
                kind: Kind::VALUE, ws_mask: Some(mask), metadata: None
            }
        );
    };
}

macro_rules! f16_ws_variant {
    ($marker:ident, $issue:path, $a:ty) => {
        dense_ws_mma_variant!(
            $marker, <A: B16, B: B16, Scale: ValidScale>, $issue, $a, |mask|
            DenseMmaFormat::Float {
                kind: RawTcgenFloatKind::B16 { a_bf16: A::BF16, b_bf16: B::BF16 },
                scale: Scale::VALUE, ws_mask: Some(mask), metadata: None,
            }
        );
    };
}
integer_ws_variant!(MmaIntegerSsCta1Ws, issue_mma_dense_ss::<4>, u64);
integer_ws_variant!(MmaIntegerTsCta1Ws, issue_mma_dense_ts::<false, 4>, u32);

macro_rules! sparse_variant {
    ($marker:ident, $issue:path, $a:ty, $ws:literal, $masks:literal) => {
        instruction_variant! {
            [impl<Kind: SparseKind, Access: AccessMode>] mma_spec, variant::$marker<Kind, Access>,
            (
                BufferHandle<Tmem>,
                DescriptorDomain<Shared>,
                R<u32>,
                R<$a>,
                R<u64>,
                R<u32>,
                R<bool>,
                [R<u32>; $masks],
                R<u64>,
                R<u32>,
            ) => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
            ) -> Result<(), EngineError> {
                let (
                    anchor,
                    shared,
                    destination,
                    a,
                    b,
                    descriptor,
                    enable_d,
                    masks,
                    ws_mask,
                    metadata,
                ) = args;
                let Some((lane, _)) = require_single_issue_lane(context, None, "tcgen05.mma.sp")?
                else {
                    return Ok(());
                };
                $issue(
                    warp,
                    context,
                    site,
                    (
                        anchor,
                        shared,
                        destination,
                        a,
                        b,
                        descriptor,
                        enable_d,
                        masks,
                    ),
                    None,
                    Kind::format(metadata[lane], $ws.then_some(ws_mask[lane])),
                    Access::VALUE,
                )
            }
        }
    };
}
sparse_variant!(MmaSparseSsCta1, issue_mma_dense_ss, u64, false, 4);
sparse_variant!(MmaSparseTsCta1, issue_mma_dense_ts::<false, 4>, u32, false, 4);
sparse_variant!(MmaSparseSsCta1Ws, issue_mma_dense_ss, u64, true, 4);
sparse_variant!(MmaSparseTsCta1Ws, issue_mma_dense_ts::<false, 4>, u32, true, 4);

sparse_variant!(MmaSparseSsCta2, issue_mma_dense_ss, u64, false, 8);
sparse_variant!(
    MmaSparseTsCta2,
    issue_mma_dense_ts::<false, 8>,
    u32,
    false,
    8
);

integer_mma_variants!(
    MmaIntegerSsCta1,
    MmaIntegerSsCta1Pred,
    MmaArgs<u64, 4>,
    MmaPredArgs<u64, 4>,
    issue_mma_dense_ss
);

f16_ws_variant!(MmaF16SsCta1Ws, issue_mma_dense_ss::<4>, u64);

#[inline(never)]
fn issue_mma_dense_ts<const ASHIFT: bool, const MASKS: usize>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    args: MmaArgs<u32, MASKS>,
    predicate: Option<R<u32>>,
    format: DenseMmaFormat,
    access: TmemAccessMode,
) -> Result<(), EngineError> {
    issue_mma_dense::<_, ASHIFT, MASKS>(
        warp,
        context,
        site,
        args,
        predicate,
        format,
        access,
        RawTcgenMmaA::Tmem,
    )
}

#[inline(never)]
fn issue_mma_dense<A: Copy, const ASHIFT: bool, const MASKS: usize>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    args: MmaArgs<A, MASKS>,
    predicate: Option<R<u32>>,
    format: DenseMmaFormat,
    access: TmemAccessMode,
    source: impl FnOnce(A) -> RawTcgenMmaA,
) -> Result<(), EngineError> {
    let (anchor, shared, destination, a, b, descriptor, enable_d, masks) = args;
    let Some((lane, issue_context)) =
        require_single_issue_lane(context, predicate.as_ref(), "tcgen05.mma")?
    else {
        return Ok(());
    };
    let a_source = source(a[lane]);
    // The instruction ABI carries four output masks per CTA.
    let cta_group = MASKS / 4;
    let masks: [u32; MASKS] = std::array::from_fn(|index| masks[index][lane]);
    let numeric_context = issue_context.into_inner();
    let physical = engine(warp).kernel().physical().clone();
    let lifecycle = engine(warp).kernel().services().tcgen();
    let shared_candidates = shared.inner().iter().collect::<Vec<_>>();
    let shape = format.shape(descriptor[lane], cta_group).and_then(|shape| {
        if matches!(a_source, RawTcgenMmaA::Tmem(_)) {
            crate::runtime::raw_tcgen05_validate_tmem_a_transpose(shape.2)?;
        }
        if ASHIFT && cta_group == 1 && shape.0 != 128 {
            return Err(crate::EngineError::message(
                "tcgen05.mma.ashift requires M=128 for cta_group::1",
            ));
        }
        if ASHIFT && cta_group == 2 && shape.0 != 256 && format.metadata().is_some() {
            // PTX lists M128/M256, but sparse M128 traps in SM100 probes.
            // That does not establish invalidity on every modeled target.
            return Err(crate::EngineError::analysis_incomplete(
                "tcgen_sparse_m128_ashift_unmodeled",
            ));
        }
        Ok(shape)
    });
    let (m, n, transpose_a, transpose_b) = match shape {
        Ok(shape) => shape,
        Err(error) => {
            // Descriptor failures occur before the ordinary issue operation.
            // Attribute only the failing path, without recording extra work
            // for successful or predicated-off instructions.
            let operation = engine(warp).begin_current_operation(
                numeric_context,
                crate::StaticOpId::new(site.get()),
                OperationKind::TcgenWork,
            )?;
            return Err(error.with_operation_context(&operation).into());
        }
    };
    let layout = raw_tcgen05_cta1_dense_tmem_layout(
        m / cta_group,
        if cta_group == 1 {
            format.weight_stationary()
        } else {
            format.metadata().is_none()
        },
    )?;
    let a_accesses = match a_source {
        RawTcgenMmaA::Shared(bits) => {
            let (source, accesses) = format.shared_footprints(
                &numeric_context,
                &shared_candidates,
                bits,
                m / cta_group,
                format.packed_k(descriptor[lane]),
                transpose_a,
                lane,
                cta_group,
                None,
                false,
            )?;
            RawTcgenMmaAAccess::Shared(source, accesses)
        }
        RawTcgenMmaA::Tmem(address) => {
            let columns = format.tmem_a_columns(descriptor[lane]);
            let accesses = if cta_group == 1 {
                raw_tcgen05_packed_tmem_a_column_footprints(
                    address, m, layout, columns, lane, lane,
                )?
            } else {
                crate::runtime::raw_tcgen05_cta2_packed_tmem_a_footprints(
                    &numeric_context,
                    address,
                    m,
                    layout,
                    columns,
                    lane,
                    lane,
                )?
            };
            RawTcgenMmaAAccess::Tmem(accesses)
        }
    };
    let column_mask = format
        .ws_mask()
        .map(|bits| crate::runtime::RawTcgenColumnMask::new(bits, m, n, descriptor[lane]))
        .transpose()?;
    let (b_source, b_footprints) = format.shared_footprints(
        &numeric_context,
        &shared_candidates,
        b[lane],
        n / cta_group,
        format.k(descriptor[lane]),
        transpose_b,
        lane,
        cta_group,
        column_mask,
        true,
    )?;
    let tmem_anchor = raw_ldst_footprint_anchor(anchor.inner(), 4)?;
    let tmem_logical_buffer = anchor
        .logical_buffer()
        .unwrap_or("raw_tcgen05_tmem")
        .to_owned();
    let destination_footprints = dense_mma_destination_footprints(
        &numeric_context,
        destination[lane],
        m,
        n,
        layout,
        masks,
        lane,
    )?;
    let metadata_footprints = format
        .metadata()
        .map(|address| {
            crate::runtime::raw_tcgen05_sparse_metadata_footprints(
                &numeric_context,
                address,
                format.metadata_layout(descriptor[lane]),
                m / cta_group,
                layout,
                lane,
                cta_group,
            )
        })
        .transpose()?
        .unwrap_or_default();
    let shift_writes = match &a_accesses {
        RawTcgenMmaAAccess::Tmem(accesses) if ASHIFT => accesses
            .iter()
            .copied()
            .filter(|access| access.3 % 32 != 31)
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    };
    let pipeline_class = crate::runtime::TcgenMmaPipelineClass::new(
        m,
        n,
        format.k(descriptor[lane]),
        format.accumulator(descriptor[lane]),
    );
    let operation = begin(warp, issue_context, site, OperationKind::TcgenWork, true)?;
    engine(warp).tcgen_instruction_issue(
        operation.as_ref(),
        cta_group as u32,
        TcgenPipelineOperation::Mma,
        Some(pipeline_class),
        |record_runtime, record_tmem| {
            match &a_accesses {
                RawTcgenMmaAAccess::Shared(source, accesses) => {
                    record_runtime(true, OperationKind::Load, source, None, accesses)?;
                }
                RawTcgenMmaAAccess::Tmem(accesses) => {
                    record_tmem(
                        OperationKind::Load,
                        &tmem_anchor,
                        &tmem_logical_buffer,
                        access,
                        accesses,
                    )?;
                    if ASHIFT {
                        record_tmem(
                            OperationKind::Store,
                            &tmem_anchor,
                            &tmem_logical_buffer,
                            access,
                            &shift_writes,
                        )?;
                    }
                }
            }
            record_runtime(false, OperationKind::Load, &b_source, None, &b_footprints)?;
            if !metadata_footprints.is_empty() {
                record_tmem(
                    OperationKind::Load,
                    &tmem_anchor,
                    &tmem_logical_buffer,
                    access,
                    &metadata_footprints,
                )?;
            }
            if enable_d[lane] && !destination_footprints.is_empty() {
                record_tmem(
                    OperationKind::Load,
                    &tmem_anchor,
                    &tmem_logical_buffer,
                    access,
                    &destination_footprints,
                )?;
            }
            if !destination_footprints.is_empty() {
                record_tmem(
                    OperationKind::Store,
                    &tmem_anchor,
                    &tmem_logical_buffer,
                    access,
                    &destination_footprints,
                )?;
            }
            Ok(())
        },
        || {
            (|| {
                let Some((kind, scale)) = format.float_kind() else {
                    return crate::runtime::raw_tcgen05_mma_integer(
                        format.integer_kind(),
                        &physical,
                        &numeric_context,
                        &lifecycle,
                        access,
                        anchor.inner(),
                        &shared_candidates,
                        destination[lane],
                        a_source,
                        b[lane],
                        descriptor[lane],
                        enable_d[lane],
                        masks,
                        lane,
                        format.ws_mask(),
                        format.metadata(),
                    );
                };
                raw_tcgen05_mma_float(
                    &physical,
                    &numeric_context,
                    &lifecycle,
                    access,
                    anchor.inner(),
                    &shared_candidates,
                    destination[lane],
                    a_source,
                    b[lane],
                    descriptor[lane],
                    enable_d[lane],
                    scale,
                    &masks,
                    lane,
                    cta_group,
                    format.ws_mask(),
                    format.metadata(),
                    kind,
                )
            })()?;
            if ASHIFT {
                let RawTcgenMmaA::Tmem(address) = a_source else {
                    unreachable!("only TS variants shift A");
                };
                let shift = if format.tmem_a_columns(descriptor[lane]) == 16 {
                    raw_tcgen05_shift::<16>
                } else {
                    raw_tcgen05_shift::<8>
                };
                shift(
                    &physical,
                    &numeric_context,
                    &lifecycle,
                    access,
                    anchor.inner(),
                    address,
                    cta_group,
                    lane,
                )?;
            }
            Ok(())
        },
    )?;
    finish(warp, &operation)
}

f16_mma_variants!(
    MmaF16TsCta1,
    MmaF16TsCta1Pred,
    MmaArgs<u32, 4>,
    MmaPredArgs<u32, 4>,
    issue_mma_dense_ts::<false, 4>
);

integer_mma_variants!(
    MmaIntegerTsCta1,
    MmaIntegerTsCta1Pred,
    MmaArgs<u32, 4>,
    MmaPredArgs<u32, 4>,
    issue_mma_dense_ts::<false, 4>
);

f16_ws_variant!(MmaF16TsCta1Ws, issue_mma_dense_ts::<false, 4>, u32);




f16_mma_variants!(
    MmaF16SsCta2,
    MmaF16SsCta2Pred,
    MmaArgs<u64, 8>,
    MmaPredArgs<u64, 8>,
    issue_mma_dense_ss
);
f16_mma_variants!(
    MmaF16TsCta2,
    MmaF16TsCta2Pred,
    MmaArgs<u32, 8>,
    MmaPredArgs<u32, 8>,
    issue_mma_dense_ts::<false, 8>
);

integer_mma_variants!(
    MmaIntegerSsCta2,
    MmaIntegerSsCta2Pred,
    MmaArgs<u64, 8>,
    MmaPredArgs<u64, 8>,
    issue_mma_dense_ss
);
integer_mma_variants!(
    MmaIntegerTsCta2,
    MmaIntegerTsCta2Pred,
    MmaArgs<u32, 8>,
    MmaPredArgs<u32, 8>,
    issue_mma_dense_ts::<false, 8>
);

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn issue_mma_f8f6f4<const ASHIFT: bool>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    args: MmaArgs<u64, 4>,
    predicate: Option<R<u32>>,
    access: TmemAccessMode,
    a_format: RawTcgenNarrowFormat,
    b_format: RawTcgenNarrowFormat,
    d_f16: bool,
    a_in_tmem: bool,
    ws_mask: Option<R<u64>>,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
    lut_b: Option<R<u32>>,
) -> Result<(), EngineError> {
    let (anchor, shared, destination, a, b, descriptor, enable_d, masks) = args;
    let Some((lane, issue_context)) =
        require_single_issue_lane(context, predicate.as_ref(), "tcgen05.mma")?
    else {
        return Ok(());
    };
    let ws_mask = ws_mask.as_ref().map(|mask| mask[lane]);
    let lut_b = lut_b.as_ref().map(|address| address[lane]);
    let masks = std::array::from_fn(|index| masks[index][lane]);
    let numeric_context = issue_context.into_inner();
    let physical = engine(warp).kernel().physical().clone();
    let lifecycle = engine(warp).kernel().services().tcgen();
    let operation = begin(warp, issue_context, site, OperationKind::TcgenWork, true)?;
    let shared_candidates = shared.inner().iter().collect::<Vec<_>>();
    let tmem_logical_buffer = anchor
        .logical_buffer()
        .unwrap_or("raw_tcgen05_mma_tmem")
        .to_owned();
    let (m, n, k, transpose_a, transpose_b) = raw_tcgen05_mma_f8f6f4_cta1_shape(
        descriptor[lane],
        a_format,
        b_format,
        d_f16,
        ws_mask.is_some(),
        descriptor_layout,
    )?;
    let layout = raw_tcgen05_cta1_dense_tmem_layout(m, ws_mask.is_some())?;
    let column_mask = ws_mask
        .map(|bits| RawTcgenColumnMask::new(bits, m, n, descriptor[lane]))
        .transpose()?;
    if ASHIFT && (!a_in_tmem || !matches!(m, 128 | 256)) {
        return Err(EngineError::message(
            "tcgen05.mma.ashift requires TMEM A and M=128/256",
        ));
    }
    let pipeline_class = crate::runtime::TcgenMmaPipelineClass::new(
        m,
        n,
        k,
        if d_f16 {
            crate::runtime::TcgenAccumulatorDtype::F16
        } else {
            crate::runtime::TcgenAccumulatorDtype::F32
        },
    );
    engine(warp).tcgen_instruction_issue(
        operation.as_ref(),
        1,
        TcgenPipelineOperation::Mma,
        Some(pipeline_class),
        |record_runtime, record_tmem| {
            let tmem_anchor = raw_ldst_footprint_anchor(anchor.inner(), 4)?;
            if let Some(address) = lut_b {
                let accesses =
                    raw_tcgen05_lut_b_tmem_footprints(&numeric_context, address, n, 1, lane)?;
                record_tmem(
                    OperationKind::Load,
                    &tmem_anchor,
                    &tmem_logical_buffer,
                    access,
                    &accesses,
                )?;
            }
            if a_in_tmem {
                let address = raw_tcgen05_f8_tmem_a_address(a[lane], transpose_a)?;
                let a_accesses = raw_tcgen05_packed_tmem_a_column_footprints(
                    address,
                    m,
                    layout,
                    k / 4,
                    lane,
                    lane,
                )?;
                record_tmem(
                    OperationKind::Load,
                    &tmem_anchor,
                    &tmem_logical_buffer,
                    access,
                    &a_accesses,
                )?;
                if ASHIFT {
                    let writes = a_accesses
                        .iter()
                        .copied()
                        .filter(|access| access.3 % 32 != 31)
                        .collect::<Vec<_>>();
                    record_tmem(
                        OperationKind::Store,
                        &tmem_anchor,
                        &tmem_logical_buffer,
                        access,
                        &writes,
                    )?;
                }
            } else {
                let (a_source, a_accesses) = raw_tcgen05_f8_shared_footprints(
                    &numeric_context,
                    &shared_candidates,
                    a[lane],
                    descriptor_layout,
                    m,
                    k,
                    a_format,
                    transpose_a,
                    lane,
                    None,
                    None,
                )?;
                record_runtime(true, OperationKind::Load, &a_source, None, &a_accesses)?;
            }
            let (b_source, b_accesses) = raw_tcgen05_f8_shared_footprints(
                &numeric_context,
                &shared_candidates,
                b[lane],
                descriptor_layout,
                n,
                k,
                b_format,
                transpose_b,
                lane,
                column_mask,
                lut_b,
            )?;
            record_runtime(false, OperationKind::Load, &b_source, None, &b_accesses)?;
            // F16 and F32 destinations both occupy a whole 32-bit cell.
            let accumulator_accesses = raw_tcgen05_dense_f32_tmem_footprints(
                destination[lane],
                m,
                n,
                layout,
                masks,
                lane,
                lane,
            )?;
            if enable_d[lane] {
                record_tmem(
                    OperationKind::Load,
                    &tmem_anchor,
                    &tmem_logical_buffer,
                    access,
                    &accumulator_accesses,
                )?;
            }
            record_tmem(
                OperationKind::Store,
                &tmem_anchor,
                &tmem_logical_buffer,
                access,
                &accumulator_accesses,
            )
        },
        || {
            raw_tcgen05_mma_f8f6f4_cta1(
                &physical,
                &numeric_context,
                &lifecycle,
                access,
                anchor.inner(),
                &shared_candidates,
                destination[lane],
                a[lane],
                b[lane],
                descriptor[lane],
                enable_d[lane],
                masks,
                lane,
                1,
                a_format,
                b_format,
                d_f16,
                a_in_tmem,
                ws_mask,
                descriptor_layout,
                lut_b,
            )?;
            if ASHIFT {
                let address = raw_tcgen05_f8_tmem_a_address(a[lane], false)?;
                if k == 64 {
                    raw_tcgen05_shift::<16>(
                        &physical,
                        &numeric_context,
                        &lifecycle,
                        access,
                        anchor.inner(),
                        address,
                        1,
                        lane,
                    )?;
                } else {
                    raw_tcgen05_shift::<8>(
                        &physical,
                        &numeric_context,
                        &lifecycle,
                        access,
                        anchor.inner(),
                        address,
                        1,
                        lane,
                    )?;
                }
            }
            Ok(())
        },
    )?;
    finish(warp, &operation)
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn issue_mma_f8f6f4_cta2<const ASHIFT: bool>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    args: MmaArgs<u64, 8>,
    predicate: Option<R<u32>>,
    access: TmemAccessMode,
    a_format: RawTcgenNarrowFormat,
    b_format: RawTcgenNarrowFormat,
    d_f16: bool,
    a_in_tmem: bool,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
    lut_b: Option<R<u32>>,
) -> Result<(), EngineError> {
    let (anchor, shared, destination, a, b, descriptor, enable_d, masks) = args;
    let Some((lane, issue_context)) =
        require_single_issue_lane(context, predicate.as_ref(), "tcgen05.mma")?
    else {
        return Ok(());
    };
    let lut_b = lut_b.as_ref().map(|address| address[lane]);
    let masks = std::array::from_fn(|index| masks[index][lane]);
    let numeric_context = issue_context.into_inner();
    let physical = engine(warp).kernel().physical().clone();
    let lifecycle = engine(warp).kernel().services().tcgen();
    let shared_candidates = shared.inner().iter().collect::<Vec<_>>();
    let tmem_anchor = raw_ldst_footprint_anchor(anchor.inner(), 4)?;
    let tmem_logical_buffer = anchor
        .logical_buffer()
        .unwrap_or("raw_tcgen05_mma_tmem")
        .to_owned();
    let (m, n, k, _transpose_a, _transpose_b) = raw_tcgen05_mma_f8f6f4_cta2_shape(
        descriptor[lane],
        a_format,
        b_format,
        d_f16,
        descriptor_layout,
    )?;
    if ASHIFT && (!a_in_tmem || !matches!(m, 128 | 256)) {
        return Err(EngineError::message(
            "tcgen05.mma.ashift requires TMEM A and M=128/256",
        ));
    }
    let pipeline_class = TcgenMmaPipelineClass::new(
        m,
        n,
        k,
        if d_f16 {
            TcgenAccumulatorDtype::F16
        } else {
            TcgenAccumulatorDtype::F32
        },
    );
    let operation = begin(warp, issue_context, site, OperationKind::TcgenWork, true)?;
    engine(warp).tcgen_instruction_issue(
        operation.as_ref(),
        2,
        TcgenPipelineOperation::Mma,
        Some(pipeline_class),
        |record_runtime, record_tmem| {
            let footprints = raw_tcgen05_mma_f8f6f4_cta2_footprints(
                &numeric_context,
                &shared_candidates,
                destination[lane],
                a[lane],
                b[lane],
                descriptor[lane],
                masks,
                lane,
                a_format,
                b_format,
                d_f16,
                a_in_tmem,
                descriptor_layout,
                lut_b,
            )?;
            if let Some(address) = lut_b {
                let accesses =
                    raw_tcgen05_lut_b_tmem_footprints(&numeric_context, address, n / 2, 2, lane)?;
                record_tmem(
                    OperationKind::Load,
                    &tmem_anchor,
                    &tmem_logical_buffer,
                    access,
                    &accesses,
                )?;
            }
            match &footprints.a {
                RawTcgenMmaAAccess::Shared(source, accesses) => {
                    record_runtime(true, OperationKind::Load, source, None, accesses)?;
                }
                RawTcgenMmaAAccess::Tmem(accesses) => {
                    record_tmem(
                        OperationKind::Load,
                        &tmem_anchor,
                        &tmem_logical_buffer,
                        access,
                        accesses,
                    )?;
                    if ASHIFT {
                        let writes = accesses
                            .iter()
                            .copied()
                            .filter(|access| access.3 % 32 != 31)
                            .collect::<Vec<_>>();
                        record_tmem(
                            OperationKind::Store,
                            &tmem_anchor,
                            &tmem_logical_buffer,
                            access,
                            &writes,
                        )?;
                    }
                }
            }
            record_runtime(
                false,
                OperationKind::Load,
                &footprints.b_source,
                None,
                &footprints.b_accesses,
            )?;
            if enable_d[lane] && !footprints.accumulator_accesses.is_empty() {
                record_tmem(
                    OperationKind::Load,
                    &tmem_anchor,
                    &tmem_logical_buffer,
                    access,
                    &footprints.accumulator_accesses,
                )?;
            }
            if !footprints.accumulator_accesses.is_empty() {
                record_tmem(
                    OperationKind::Store,
                    &tmem_anchor,
                    &tmem_logical_buffer,
                    access,
                    &footprints.accumulator_accesses,
                )?;
            }
            Ok(())
        },
        || {
            raw_tcgen05_mma_f8f6f4_cta2(
                &physical,
                &numeric_context,
                &lifecycle,
                access,
                anchor.inner(),
                &shared_candidates,
                destination[lane],
                a[lane],
                b[lane],
                descriptor[lane],
                enable_d[lane],
                masks,
                lane,
                1,
                a_format,
                b_format,
                d_f16,
                a_in_tmem,
                descriptor_layout,
                lut_b,
            )?;
            if ASHIFT {
                let address = raw_tcgen05_f8_tmem_a_address(a[lane], false)?;
                if k == 64 {
                    raw_tcgen05_shift::<16>(
                        &physical,
                        &numeric_context,
                        &lifecycle,
                        access,
                        anchor.inner(),
                        address,
                        2,
                        lane,
                    )?;
                } else {
                    raw_tcgen05_shift::<8>(
                        &physical,
                        &numeric_context,
                        &lifecycle,
                        access,
                        anchor.inner(),
                        address,
                        2,
                        lane,
                    )?;
                }
            }
            Ok(())
        },
    )?;
    finish(warp, &operation)
}

macro_rules! f8f6f4_mma_variants {
    ($spec:ident, $plain:ident, $predicated:ident, $d_f16:literal, $a_in_tmem:literal, $args:ty, $pred_args:ty, $issue:path $(, ws = $ws:expr)?) => {
        instruction_variant! {
            [impl<A: NarrowFloat, B: NarrowFloat, DescriptorLayout: MatrixDescriptorLayout, Access: AccessMode>]
            $spec, variant::$plain<A, B, DescriptorLayout, Access>,
            $args => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                (anchor, shared, destination, a, b, descriptor, enable_d, masks): Self::Args,
            ) -> Result<Self::Output, EngineError> {
                $issue(
                    warp,
                    context,
                    site,
                    (anchor, shared, destination, a.map(|_, v| u64::from(v)), b, descriptor, enable_d, masks),
                    None,
                    Access::VALUE,
                    A::FORMAT,
                    B::FORMAT,
                    $d_f16,
                    $a_in_tmem,
                    $($ws,)?
                    DescriptorLayout::VALUE,
                    None,
                )
            }
        }

        instruction_variant! {
            [impl<A: NarrowFloat, B: NarrowFloat, DescriptorLayout: MatrixDescriptorLayout, Access: AccessMode>]
            $spec, variant::$predicated<A, B, DescriptorLayout, Access>,
            $pred_args => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                (anchor, shared, destination, a, b, descriptor, enable_d, masks, predicate): Self::Args,
            ) -> Result<Self::Output, EngineError> {
                $issue(
                    warp,
                    context,
                    site,
                    (anchor, shared, destination, a.map(|_, v| u64::from(v)), b, descriptor, enable_d, masks),
                    Some(predicate),
                    Access::VALUE,
                    A::FORMAT,
                    B::FORMAT,
                    $d_f16,
                    $a_in_tmem,
                    $($ws,)?
                    DescriptorLayout::VALUE,
                    None,
                )
            }
        }
    };
}

f8f6f4_mma_variants!(
    mma_spec,
    MmaF8f6f4F32SsCta1,
    MmaF8f6f4F32SsCta1Pred,
    false,
    false,
    MmaArgs<u64, 4>,
    MmaPredArgs<u64, 4>,
    issue_mma_f8f6f4::<false>, ws = None
);
f8f6f4_mma_variants!(
    mma_spec,
    MmaF8f6f4F16SsCta1,
    MmaF8f6f4F16SsCta1Pred,
    true,
    false,
    MmaArgs<u64, 4>,
    MmaPredArgs<u64, 4>,
    issue_mma_f8f6f4::<false>, ws = None
);
f8f6f4_mma_variants!(
    mma_spec,
    MmaF8f6f4F32TsCta1,
    MmaF8f6f4F32TsCta1Pred,
    false,
    true,
    MmaArgs<u32, 4>,
    MmaPredArgs<u32, 4>,
    issue_mma_f8f6f4::<false>, ws = None
);
f8f6f4_mma_variants!(
    mma_spec,
    MmaF8f6f4F16TsCta1,
    MmaF8f6f4F16TsCta1Pred,
    true,
    true,
    MmaArgs<u32, 4>,
    MmaPredArgs<u32, 4>,
    issue_mma_f8f6f4::<false>, ws = None
);

macro_rules! f8f6f4_ws_variant {
    ($variant:ident, $a:ty, $d_f16:literal, $a_in_tmem:literal) => {
        instruction_variant! {
            [impl<
                A: NarrowFloat,
                B: NarrowFloat,
                DescriptorLayout: MatrixDescriptorLayout,
                Access: AccessMode,
            >]
            mma_spec, variant::$variant<A, B, DescriptorLayout, Access>,
            WsMmaArgs<$a> => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                (anchor, shared, destination, a, b, descriptor, enable_d, mask): Self::Args,
            ) -> Result<(), EngineError> {
                issue_mma_f8f6f4::<false>(
                    warp,
                    context,
                    site,
                    (
                        anchor,
                        shared,
                        destination,
                        a.map(|_, v| u64::from(v)),
                        b,
                        descriptor,
                        enable_d,
                        std::array::from_fn(|_| R::splat(0)),
                    ),
                    None,
                    Access::VALUE,
                    A::FORMAT,
                    B::FORMAT,
                    $d_f16,
                    $a_in_tmem,
                    Some(mask),
                    DescriptorLayout::VALUE,
                    None,
                )
            }
        }
    };
}
f8f6f4_ws_variant!(MmaF8f6f4F32SsCta1Ws, u64, false, false);
f8f6f4_ws_variant!(MmaF8f6f4F16SsCta1Ws, u64, true, false);
f8f6f4_ws_variant!(MmaF8f6f4F32TsCta1Ws, u32, false, true);
f8f6f4_ws_variant!(MmaF8f6f4F16TsCta1Ws, u32, true, true);


f8f6f4_mma_variants!(
    mma_spec,
    MmaF8f6f4F32SsCta2,
    MmaF8f6f4F32SsCta2Pred,
    false,
    false,
    MmaArgs<u64, 8>,
    MmaPredArgs<u64, 8>,
    issue_mma_f8f6f4_cta2::<false>
);
f8f6f4_mma_variants!(
    mma_spec,
    MmaF8f6f4F16SsCta2,
    MmaF8f6f4F16SsCta2Pred,
    true,
    false,
    MmaArgs<u64, 8>,
    MmaPredArgs<u64, 8>,
    issue_mma_f8f6f4_cta2::<false>
);
f8f6f4_mma_variants!(
    mma_spec,
    MmaF8f6f4F32TsCta2,
    MmaF8f6f4F32TsCta2Pred,
    false,
    true,
    MmaArgs<u32, 8>,
    MmaPredArgs<u32, 8>,
    issue_mma_f8f6f4_cta2::<false>
);
f8f6f4_mma_variants!(
    mma_spec,
    MmaF8f6f4F16TsCta2,
    MmaF8f6f4F16TsCta2Pred,
    true,
    true,
    MmaArgs<u32, 8>,
    MmaPredArgs<u32, 8>,
    issue_mma_f8f6f4_cta2::<false>
);


tf32_mma_variants!(
    MmaTf32SsCta1,
    MmaTf32SsCta1Pred,
    issue_mma_dense_ss,
    MmaArgs<u64, 4>,
    MmaPredArgs<u64, 4>
);
tf32_mma_variants!(
    MmaTf32TsCta1,
    MmaTf32TsCta1Pred,
    issue_mma_dense_ts::<false, 4>,
    MmaArgs<u32, 4>,
    MmaPredArgs<u32, 4>
);
tf32_mma_variants!(
    MmaTf32SsCta2,
    MmaTf32SsCta2Pred,
    issue_mma_dense_ss,
    MmaArgs<u64, 8>,
    MmaPredArgs<u64, 8>
);
tf32_mma_variants!(
    MmaTf32TsCta2,
    MmaTf32TsCta2Pred,
    issue_mma_dense_ts::<false, 8>,
    MmaArgs<u32, 8>,
    MmaPredArgs<u32, 8>
);

macro_rules! tf32_ws_variant {
    ($marker:ident, $issue:path, $a:ty) => {
        dense_ws_mma_variant!(
            $marker, <Scale: ValidScale>, $issue, $a, |mask|
            DenseMmaFormat::Float {
                kind: RawTcgenFloatKind::Tf32,
                scale: Scale::VALUE, ws_mask: Some(mask), metadata: None,
            }
        );
    };
}
tf32_ws_variant!(MmaTf32SsCta1Ws, issue_mma_dense_ss::<4>, u64);
tf32_ws_variant!(MmaTf32TsCta1Ws, issue_mma_dense_ts::<false, 4>, u32);

type BlockMmaArgs<A = u64> = (
    BufferHandle<Tmem>,
    DescriptorDomain<Shared>,
    R<u32>,
    R<A>,
    R<u64>,
    R<u32>,
    R<u32>,
    R<u32>,
    R<bool>,
);

/// The closed set of raw `tcgen05.mma.block_scale` forms this module issues.
///
/// One row fixes the CTA group, the instruction K, and which decode, footprint,
/// and execution entries the form uses; nothing else about the issue path
/// varies, so the forms share one body.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockMmaForm {
    /// MXF4 block32, shared or TMEM A.
    Mxf4E8m0Cta1,
    Mxf4E8m0Cta2,
    /// NVF4 block32, shared or TMEM A.
    Mxf4nvf4Vec2Cta1,
    Mxf4nvf4Vec2Cta2,
    /// NVF4 block16, shared or TMEM A.
    Mxf4nvf4E2m1Cta1,
    Mxf4nvf4E2m1Cta2,
    /// `.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`, SS or TS.
    Mxf8f6f4E8m0Cta1,
    /// `.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X`, SS or TS.
    Mxf8f6f4E8m0Cta2,
}

impl BlockMmaForm {
    fn fp4_scale(self, descriptor: u32) -> RawTcgenMxf4ScaleSpelling {
        match self {
            Self::Mxf4E8m0Cta1 | Self::Mxf4E8m0Cta2 => RawTcgenMxf4ScaleSpelling::Ue8m0Vec2x,
            Self::Mxf4nvf4Vec2Cta1 | Self::Mxf4nvf4Vec2Cta2 => {
                raw_tcgen05_mxf4nvf4_vec2x_scale(descriptor)
            }
            Self::Mxf4nvf4E2m1Cta1 | Self::Mxf4nvf4E2m1Cta2 => {
                raw_tcgen05_mxf4nvf4_vec4x_scale(descriptor)
            }
            _ => unreachable!("FP4 scale requested for MXF8"),
        }
    }
    fn is_mxf8(self) -> bool {
        matches!(self, Self::Mxf8f6f4E8m0Cta1 | Self::Mxf8f6f4E8m0Cta2)
    }

    fn cta_group(self) -> u32 {
        match self {
            Self::Mxf4E8m0Cta1
            | Self::Mxf4nvf4Vec2Cta1
            | Self::Mxf4nvf4E2m1Cta1
            | Self::Mxf8f6f4E8m0Cta1 => 1,
            Self::Mxf4E8m0Cta2
            | Self::Mxf4nvf4Vec2Cta2
            | Self::Mxf4nvf4E2m1Cta2
            | Self::Mxf8f6f4E8m0Cta2 => 2,
        }
    }
}

#[inline(never)]
fn issue_block_mma<A: Copy>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    args: BlockMmaArgs<A>,
    form: BlockMmaForm,
    access: TmemAccessMode,
    descriptor_layout: RawTcgenMatrixDescriptorLayout,
    fixed_vectors: bool,
    resolve_a: fn(A) -> RawTcgenMmaA,
    lut_b: Option<R<u32>>,
    metadata: Option<R<u32>>,
) -> Result<(), EngineError> {
    let cta_group = form.cta_group();
    let (anchor, shared, destination, a, b, sfa, sfb, descriptor, enable_d) = args;
    let Some((lane, issue_context)) =
        require_single_issue_lane(context, None, "tcgen05.mma.block_scale")?
    else {
        unreachable!()
    };
    let numeric_context = issue_context.into_inner();
    let a = resolve_a(a[lane]);
    let lut_b = lut_b.map(|address| address[lane]);
    let metadata = metadata.map(|address| address[lane]);
    let physical = engine(warp).kernel().physical().clone();
    let lifecycle = engine(warp).kernel().services().tcgen();
    let operation = begin(warp, issue_context, site, OperationKind::TcgenWork, true)?;
    let shared_candidates = shared.inner().iter().collect::<Vec<_>>();
    let tmem_logical_buffer = anchor
        .logical_buffer()
        .unwrap_or("raw_tcgen05_mma_tmem")
        .to_owned();
    let (m, n, k) = match form {
        BlockMmaForm::Mxf8f6f4E8m0Cta1 | BlockMmaForm::Mxf8f6f4E8m0Cta2 => {
            raw_tcgen05_mma_block_mxf8f6f4_shape(
                descriptor[lane],
                cta_group as usize,
                descriptor_layout,
            )?
        }
        _ => raw_tcgen05_mma_block_mxf4_shape(
            descriptor[lane],
            form.fp4_scale(descriptor[lane]),
            cta_group as usize,
            descriptor_layout,
            fixed_vectors,
        )?,
    };
    let pipeline_class = crate::runtime::TcgenMmaPipelineClass::new(
        m,
        n,
        k,
        crate::runtime::TcgenAccumulatorDtype::F32,
    );
    let issue_memo = crate::kernel_engine::tcgen_issue_memo_seed(
        site,
        form as u8,
        [
            u64::from(destination[lane]),
            match a {
                RawTcgenMmaA::Shared(bits) => bits,
                RawTcgenMmaA::Tmem(column) => u64::from(column),
            },
            b[lane],
            u64::from(sfa[lane]) | (u64::from(sfb[lane]) << 32),
            u64::from(descriptor[lane]),
            u64::from(enable_d[lane]) | ((lane as u64) << 1),
            lut_b.map(u64::from).unwrap_or(1_u64 << 32),
            metadata.map(u64::from).unwrap_or(1_u64 << 32),
        ],
        std::iter::once(anchor.inner()).chain(shared_candidates.iter().copied()),
    );
    engine(warp).tcgen_instruction_issue_memoized(
        issue_memo,
        operation.as_ref(),
        cta_group,
        TcgenPipelineOperation::Mma,
        Some(pipeline_class),
        |record_runtime, record_tmem| {
            let footprints = if form.is_mxf8() {
                raw_tcgen05_mma_block_mxf8f6f4_footprints(
                    &numeric_context,
                    &shared_candidates,
                    destination[lane],
                    a,
                    b[lane],
                    sfa[lane],
                    sfb[lane],
                    descriptor[lane],
                    lane,
                    descriptor_layout,
                    cta_group as usize,
                    lut_b,
                    metadata,
                )?
            } else {
                raw_tcgen05_mma_block_mxf4_footprints(
                    &numeric_context,
                    &shared_candidates,
                    destination[lane],
                    a,
                    b[lane],
                    sfa[lane],
                    sfb[lane],
                    descriptor[lane],
                    lane,
                    form.fp4_scale(descriptor[lane]),
                    descriptor_layout,
                    fixed_vectors,
                    cta_group as usize,
                )?
            };
            if let Some(address) = metadata {
                let accesses = crate::runtime::raw_tcgen05_sparse_metadata_footprints(
                    &numeric_context,
                    address,
                    crate::runtime::RawTcgenSparseMetadataLayout::Narrow { k },
                    m / cta_group as usize,
                    raw_tcgen05_cta1_dense_tmem_layout(m / cta_group as usize, false)?,
                    lane,
                    cta_group as usize,
                )?;
                let metadata_anchor = raw_ldst_footprint_anchor(anchor.inner(), 4)?;
                record_tmem(
                    OperationKind::Load,
                    &metadata_anchor,
                    &tmem_logical_buffer,
                    access,
                    &accesses,
                )?;
            }
            if let Some(address) = lut_b {
                let accesses = raw_tcgen05_lut_b_tmem_footprints(
                    &numeric_context,
                    address,
                    n / cta_group as usize,
                    cta_group as usize,
                    lane,
                )?;
                let lookup_anchor = raw_ldst_footprint_anchor(anchor.inner(), 4)?;
                record_tmem(
                    OperationKind::Load,
                    &lookup_anchor,
                    &tmem_logical_buffer,
                    access,
                    &accesses,
                )?;
            }
            match &footprints.a {
                RawTcgenMmaAAccess::Shared(source, accesses) => {
                    record_runtime(true, OperationKind::Load, source, None, accesses)?;
                }
                RawTcgenMmaAAccess::Tmem(accesses) => {
                    let a_anchor = raw_ldst_footprint_anchor(anchor.inner(), 4)?;
                    record_tmem(
                        OperationKind::Load,
                        &a_anchor,
                        &tmem_logical_buffer,
                        access,
                        accesses,
                    )?;
                }
            }
            record_runtime(
                false,
                OperationKind::Load,
                &footprints.b_source,
                None,
                &footprints.b_accesses,
            )?;
            let scale_anchor = raw_tmem_footprint_anchor(anchor.inner(), 1, 4)?;
            record_tmem(
                OperationKind::Load,
                &scale_anchor,
                &tmem_logical_buffer,
                access,
                &footprints.sfa_accesses,
            )?;
            record_tmem(
                OperationKind::Load,
                &scale_anchor,
                &tmem_logical_buffer,
                access,
                &footprints.sfb_accesses,
            )?;
            let accumulator_anchor = raw_tmem_footprint_anchor(anchor.inner(), 4, footprints.n)?;
            if enable_d[lane] {
                record_tmem(
                    OperationKind::Load,
                    &accumulator_anchor,
                    &tmem_logical_buffer,
                    access,
                    &footprints.accumulator_accesses,
                )?;
            }
            record_tmem(
                OperationKind::Store,
                &accumulator_anchor,
                &tmem_logical_buffer,
                access,
                &footprints.accumulator_accesses,
            )
        },
        || {
            if form.is_mxf8() {
                return raw_tcgen05_mma_block_scale_mxf8f6f4(
                    &physical,
                    &numeric_context,
                    &lifecycle,
                    access,
                    anchor.inner(),
                    &shared_candidates,
                    destination[lane],
                    a,
                    b[lane],
                    sfa[lane],
                    sfb[lane],
                    descriptor[lane],
                    enable_d[lane],
                    lane,
                    cta_group as usize,
                    descriptor_layout,
                    lut_b,
                    metadata,
                );
            }
            raw_tcgen05_mma_block_scale_mxf4(
                &physical,
                &numeric_context,
                &lifecycle,
                access,
                anchor.inner(),
                &shared_candidates,
                destination[lane],
                a,
                b[lane],
                sfa[lane],
                sfb[lane],
                descriptor[lane],
                enable_d[lane],
                lane,
                form.fp4_scale(descriptor[lane]),
                cta_group as usize,
                descriptor_layout,
                fixed_vectors,
            )
        },
    )?;
    finish(warp, &operation)
}

macro_rules! block_metadata_variant {
    ($marker:ident, $form:ident, $source:ident, $trait:ident, $execute:ident, $sparse:literal) => {
        impl<Access: AccessMode, Layout: MatrixDescriptorLayout, const FIXED_VECTORS: bool> $trait
            for variant::$marker<Access, Layout, FIXED_VECTORS>
        {
            fn $execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
                lookup: R<u32>,
            ) -> Result<(), EngineError> {
                let (lookup, metadata) = if $sparse {
                    (None, Some(lookup))
                } else {
                    (Some(lookup), None)
                };
                issue_block_mma(
                    warp,
                    context,
                    site,
                    args,
                    BlockMmaForm::$form,
                    Access::VALUE,
                    Layout::VALUE,
                    FIXED_VECTORS,
                    RawTcgenMmaA::$source,
                    lookup,
                    metadata,
                )
            }
        }
    };
}
block_metadata_variant!(
    MmaBlockMxf8f6f4E8m0SsCta1,
    Mxf8f6f4E8m0Cta1,
    Shared,
    LutBMma,
    execute_lut_b,
    false
);
block_metadata_variant!(
    MmaBlockMxf8f6f4E8m0SsCta2,
    Mxf8f6f4E8m0Cta2,
    Shared,
    LutBMma,
    execute_lut_b,
    false
);
block_metadata_variant!(
    MmaBlockMxf8f6f4E8m0TsCta1,
    Mxf8f6f4E8m0Cta1,
    Tmem,
    LutBMma,
    execute_lut_b,
    false
);
block_metadata_variant!(
    MmaBlockMxf8f6f4E8m0TsCta2,
    Mxf8f6f4E8m0Cta2,
    Tmem,
    LutBMma,
    execute_lut_b,
    false
);
block_metadata_variant!(
    MmaBlockMxf8f6f4E8m0SsCta1,
    Mxf8f6f4E8m0Cta1,
    Shared,
    SparseBlockMma,
    execute_sparse_block,
    true
);
block_metadata_variant!(
    MmaBlockMxf8f6f4E8m0SsCta2,
    Mxf8f6f4E8m0Cta2,
    Shared,
    SparseBlockMma,
    execute_sparse_block,
    true
);
block_metadata_variant!(
    MmaBlockMxf8f6f4E8m0TsCta1,
    Mxf8f6f4E8m0Cta1,
    Tmem,
    SparseBlockMma,
    execute_sparse_block,
    true
);
block_metadata_variant!(
    MmaBlockMxf8f6f4E8m0TsCta2,
    Mxf8f6f4E8m0Cta2,
    Tmem,
    SparseBlockMma,
    execute_sparse_block,
    true
);

macro_rules! block_mma_variant {
    ($marker:ident, $form:ident, $a:ty, $source:ident) => {
        instruction_variant! {
            [impl<Access: AccessMode, Layout: MatrixDescriptorLayout, const FIXED_VECTORS: bool>]
            mma_spec, variant::$marker<Access, Layout, FIXED_VECTORS>,
            BlockMmaArgs<$a> => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
            ) -> Result<(), EngineError> {
                issue_block_mma(
                    warp,
                    context,
                    site,
                    args,
                    BlockMmaForm::$form,
                    Access::VALUE,
                    Layout::VALUE,
                    FIXED_VECTORS,
                    RawTcgenMmaA::$source,
                    None,
                    None,
                )
            }
        }
    };
}
block_mma_variant!(MmaBlockMxf4E8m0SsCta1, Mxf4E8m0Cta1, u64, Shared);
block_mma_variant!(MmaBlockMxf4E8m0SsCta2, Mxf4E8m0Cta2, u64, Shared);
block_mma_variant!(MmaBlockMxf4E8m0TsCta2, Mxf4E8m0Cta2, u32, Tmem);
block_mma_variant!(MmaBlockMxf4nvf4Vec2SsCta1, Mxf4nvf4Vec2Cta1, u64, Shared);
block_mma_variant!(MmaBlockMxf4nvf4Vec2SsCta2, Mxf4nvf4Vec2Cta2, u64, Shared);
block_mma_variant!(MmaBlockMxf4nvf4Vec2TsCta1, Mxf4nvf4Vec2Cta1, u32, Tmem);
block_mma_variant!(MmaBlockMxf4nvf4Vec2TsCta2, Mxf4nvf4Vec2Cta2, u32, Tmem);
block_mma_variant!(MmaBlockMxf4E8m0TsCta1, Mxf4E8m0Cta1, u32, Tmem);
block_mma_variant!(MmaBlockMxf4nvf4E2m1TsCta1, Mxf4nvf4E2m1Cta1, u32, Tmem);
block_mma_variant!(MmaBlockMxf4nvf4E2m1TsCta2, Mxf4nvf4E2m1Cta2, u32, Tmem);
block_mma_variant!(MmaBlockMxf4nvf4E2m1SsCta1, Mxf4nvf4E2m1Cta1, u64, Shared);
block_mma_variant!(MmaBlockMxf4nvf4E2m1SsCta2, Mxf4nvf4E2m1Cta2, u64, Shared);
block_mma_variant!(MmaBlockMxf8f6f4E8m0SsCta1, Mxf8f6f4E8m0Cta1, u64, Shared);
block_mma_variant!(MmaBlockMxf8f6f4E8m0SsCta2, Mxf8f6f4E8m0Cta2, u64, Shared);
block_mma_variant!(MmaBlockMxf8f6f4E8m0TsCta1, Mxf8f6f4E8m0Cta1, u32, Tmem);
block_mma_variant!(MmaBlockMxf8f6f4E8m0TsCta2, Mxf8f6f4E8m0Cta2, u32, Tmem);

type SparseBlockMmaArgs = (
    BufferHandle<Tmem>,
    DescriptorDomain<Shared>,
    R<u32>,
    R<u64>,
    R<u64>,
    R<u32>,
    R<u32>,
    R<u32>,
    R<u32>,
    R<bool>,
);

instruction_variant! {
    [impl<Access: AccessMode>] mma_sp_spec, variant::MmaSpBlockMxf4E8m0SsCta1<Access>,
    SparseBlockMmaArgs => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        issue_sparse_block_mma(warp, context, site, args, Access::VALUE)
    }
}

#[inline(never)]
fn issue_sparse_block_mma(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    (anchor, shared, destination, a, b, sfa, sfb, metadata, descriptor, enable_d): SparseBlockMmaArgs,
    access: TmemAccessMode,
) -> Result<(), EngineError> {
    let Some((lane, issue_context)) =
        require_single_issue_lane(context, None, "tcgen05.mma.sp.block_scale")?
    else {
        unreachable!()
    };
    let numeric_context = issue_context.into_inner();
    let physical = engine(warp).kernel().physical().clone();
    let lifecycle = engine(warp).kernel().services().tcgen();
    let operation = begin_raw_gap(
        warp,
        issue_context,
        site,
        crate::AnalysisGapKind::TcgenMma,
        1,
    )?;
    let shared_candidates = shared.inner().iter().collect::<Vec<_>>();
    engine(warp).tcgen_instruction_issue(
        operation.as_ref(),
        1,
        TcgenPipelineOperation::Mma,
        None,
        |_, _| Ok(()),
        || {
            raw_tcgen05_mma_sp_block_scale_mxf4_e8m0_ss_cta1(
                &physical,
                &numeric_context,
                &lifecycle,
                access,
                anchor.inner(),
                &shared_candidates,
                destination[lane],
                a[lane],
                b[lane],
                sfa[lane],
                sfb[lane],
                metadata[lane],
                descriptor[lane],
                enable_d[lane],
                lane,
            )
        },
    )?;
    finish(warp, &operation)
}

instruction_variant! {
    [impl<const CTA_GROUP: usize, Access: AccessMode>] shift_spec, variant::Shift<CTA_GROUP, Access>
    where [
        CtaGroup<CTA_GROUP>: ValidCtaGroup,
    ],
    (BufferHandle<Tmem>, R<u32>) => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        shift_entry(warp, context, site, args, CTA_GROUP, Access::VALUE)
    }
}

#[inline(never)]
fn shift_entry(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    (anchor, address): (BufferHandle<Tmem>, R<u32>),
    cta_group: usize,
    access: TmemAccessMode,
) -> Result<(), EngineError> {
    let Some((lane, issue_context)) = require_single_issue_lane(context, None, "tcgen05.shift")?
    else {
        unreachable!()
    };
    let numeric_context = issue_context.into_inner();
    let physical = engine(warp).kernel().physical().clone();
    let lifecycle = engine(warp).kernel().services().tcgen();
    let address = address[lane];
    // The instruction participates in TCGEN ordering, but its in-place TMEM
    // read/write footprint is not yet exposed to Racecheck. Keep the ordering
    // token while failing closed if a checked kernel actually executes shift.
    let operation = begin_raw_gap(
        warp,
        issue_context,
        site,
        crate::AnalysisGapKind::TcgenShift,
        cta_group as u32,
    )?;
    engine(warp).tcgen_instruction_issue(
        operation.as_ref(),
        cta_group as u32,
        TcgenPipelineOperation::Shift,
        None,
        |_, _| Ok(()),
        || {
            raw_tcgen05_shift::<8>(
                &physical,
                &numeric_context,
                &lifecycle,
                access,
                anchor.inner(),
                address,
                cta_group,
                lane,
            )
        },
    )?;
    finish(warp, &operation)
}

#[cfg(test)]
mod raw_variant_tests {
    use super::*;

    fn assert_ld<V: LdVariant>() {}
    fn assert_st<V: StVariant>() {}
    fn assert_cp<V: CpVariant>() {}
    fn assert_mma<V: MmaVariant>() {}

    #[test]
    fn existing_raw_tcgen_forms_have_callable_static_variants() {
        assert_ld::<variant::Ld<variant::Shape16x256b, variant::Num<32>, true, variant::DynamicTmem>>(
        );
        assert_st::<variant::St<variant::Shape16x128b, variant::Num<64>, false, variant::StaticTmem>>(
        );
        assert_cp::<
            variant::Cp<
                variant::Cp64x128bWarpx2_02_13,
                variant::DecompressB6,
                2,
                variant::DynamicTmem,
            >,
        >();
        assert_mma::<
            variant::MmaF16SsCta1<
                variant::Bf16,
                variant::Fp16,
                variant::Scale<15>,
                variant::DynamicTmem,
            >,
        >();
        assert_mma::<
            variant::MmaSparseSsCta1<(variant::Fp16, variant::Bf16), variant::StaticTmem>,
        >();
    }
}
