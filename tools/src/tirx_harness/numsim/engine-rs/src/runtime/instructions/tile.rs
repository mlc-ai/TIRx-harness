//! Engine implementation of v2 typed whole-tile operations.
//!
//! A frontend owns layout compilation through [`MappedView`].  These calls own
//! instruction partitioning, effect construction, numeric execution, and
//! completion.  Raw descriptor calls and typed calls meet below this module at
//! the engine's async/TCGEN transaction cores; typed calls never manufacture a
//! PTX descriptor.

use std::marker::PhantomData;

use super::async_copy::StaticReduction;
use super::instruction::{async_mapped_instruction, mapped_instruction};
use super::mode_axis::for_each_engine_mode;
use super::transport::{engine, ElementLocation};
use super::{
    begin, finish, singleton_issue_lane, Address, EngineError, ExecCtx, Global, LaneId, LaneMask,
    Local, LogicalCoord, MappedView, MemorySpace, Register, Shared, SiteId, WarpHandle, R,
};
use crate::runtime::AsyncSourceFill;
use crate::typed_copy::TypedTileCopyElement;
use crate::{AsyncGroupDomain, DeferredGlobalReduction, OperationKind, RuntimeScalar, WarpMask};

// `copy` executes one complete synchronous source-level tile copy. The
// frontend supplies only two pure mapped views. The engine validates the maps,
// gathers the complete source snapshot, records one union footprint per
// source/destination side, applies the scope protocol, and then restores the
// destination. This is intentionally not expressed as an `ld`/`st` loop: doing
// so would change the source operation's atomic snapshot and checker footprint
// semantics.
async_mapped_instruction!(
    copy_spec,
    CopyVariant,
    copy,
    views: [destination: Destination, source: Source]
);

// `cp_async` executes a complete logical tile using classic `cp.async`
// semantics.
mapped_instruction!(
    classic_spec,
    CpAsyncVariant,
    cp_async,
    views: [destination: Destination, source: Source],
    args
);

// `cp_async_bulk` executes a complete logical tile using `cp.async.bulk`.
mapped_instruction!(
    bulk_spec,
    BulkCopyVariant,
    cp_async_bulk,
    views: [destination: Destination, source: Source],
    args
);

// `cp_async_bulk_tensor` executes a complete mapped tile using
// `cp.async.bulk.tensor`. No tensor-map descriptor is encoded; the engine
// gathers the two pure maps directly.
mapped_instruction!(
    tensor_spec,
    TensorCopyVariant,
    cp_async_bulk_tensor,
    views: [destination: Destination, source: Source],
    args
);

// `cp_reduce_async_bulk_tensor` executes a complete mapped tile using
// `cp.reduce.async.bulk.tensor`.
mapped_instruction!(
    tensor_reduce_spec,
    TensorReduceVariant,
    cp_reduce_async_bulk_tensor,
    views: [destination: Destination, source: Source],
    args
);

// `tcgen05_cp` copies a complete mapped shared-memory tile into TMEM using the
// statically selected `tcgen05.cp` form.
mapped_instruction!(
    tcgen_cp_spec,
    Tcgen05CpVariant,
    tcgen05_cp,
    views: [destination: Destination, source: Source]
);

// `tcgen05_ld` loads a complete mapped TMEM tile into lane-private local
// storage.
mapped_instruction!(
    tcgen_ld_spec,
    Tcgen05LdVariant,
    tcgen05_ld,
    views: [destination: Destination, source: Source],
    args
);

// `tcgen05_st` stores one complete lane-private tile into TMEM.
mapped_instruction!(
    tcgen_st_spec,
    Tcgen05StVariant,
    tcgen05_st,
    views: [destination: Destination, source: Source],
    args
);

// `gemm` executes one complete register-resident logical tile using a
// statically selected family of `mma.sync.m16n8k{8,16}` instructions. `M/N/K`
// describe the source tile, not a generic repeat count. The frontend owns the
// pure element maps and may specialize a proven canonical fragment; the engine
// owns validation, numeric execution, and writes.
mapped_instruction!(
    warp_gemm_spec,
    GemmVariant,
    gemm,
    views: [destination: Destination, a: A, b: B, c: C]
);

/// Static tile shapes, execution partitions, and instruction forms.
pub mod variant {
    use super::PhantomData;

    pub use super::super::async_copy::variant::{
        NoFill, ReduceAdd, ReduceAnd, ReduceDec, ReduceInc, ReduceMax, ReduceMin, ReduceOr,
        ReduceXor, ZeroFill,
    };

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Shape1<const X: usize>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Shape2<const X: usize, const Y: usize>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Shape3<const X: usize, const Y: usize, const Z: usize>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Shape4<const A: usize, const B: usize, const C: usize, const D: usize>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Shape5<
        const A: usize,
        const B: usize,
        const C: usize,
        const D: usize,
        const E: usize,
    >;

    /// Every active lane owns one instance of the logical shape.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Thread;
    /// Logical elements are assigned round-robin to lanes of one warp.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Warp;
    /// Logical elements are assigned round-robin to threads of one warpgroup.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Warpgroup<const WARPS: usize>;
    /// Logical elements are assigned round-robin to all threads of one CTA.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Cta;

    /// One complete synchronous source-level tile copy. `Src`/`Dst`, shape,
    /// physical element type, execution scope, snapshot protocol, and source
    /// fill are all frontend-known specializations; only mapped addresses are
    /// runtime data.
    pub struct Copy<Shape, T, Src, Dst, Scope, Sync = SnapshotSync, Fill = NoFill>(
        PhantomData<fn() -> (Shape, T, Src, Dst, Scope, Sync, Fill)>,
    );

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SnapshotSync;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct NoSnapshotSync;

    /// A complete logical tile implemented with classic `cp.async`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CpAsync<Shape, T, Scope, const BYTES: usize, Fill = NoFill>(
        PhantomData<fn() -> (Shape, T, Scope, Fill)>,
    );

    /// G2S/cluster `cp.async.bulk`; completion is an mbarrier operand.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BulkG2s<Shape, T, Scope, const CTA_GROUP: u32>(
        PhantomData<fn() -> (Shape, T, Scope)>,
    );
    /// Multicast G2S/cluster `cp.async.bulk`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BulkG2sMulticast<Shape, T, Scope, const CTA_GROUP: u32>(
        PhantomData<fn() -> (Shape, T, Scope)>,
    );
    /// S2G `cp.async.bulk`, completed through the bulk async group.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BulkS2g<Shape, T, Scope>(PhantomData<fn() -> (Shape, T, Scope)>);
    /// S2S cluster `cp.async.bulk`; a mapped destination may select one CTA.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BulkS2sCluster<Shape, T, Scope>(PhantomData<fn() -> (Shape, T, Scope)>);

    /// Typed G2S `cp.async.bulk.tensor` without descriptor encoding.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorG2s<
        Shape,
        T,
        Scope,
        const CTA_GROUP: u32,
        Fill = ZeroFill,
        Conversion = NoTensorConversion,
    >(PhantomData<fn() -> (Shape, T, Scope, Fill, Conversion)>);
    /// Typed multicast G2S `cp.async.bulk.tensor`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorG2sMulticast<
        Shape,
        T,
        Scope,
        const CTA_GROUP: u32,
        Fill = ZeroFill,
        Conversion = NoTensorConversion,
    >(PhantomData<fn() -> (Shape, T, Scope, Fill, Conversion)>);
    /// Typed S2G `cp.async.bulk.tensor`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorS2g<Shape, T, Scope>(PhantomData<fn() -> (Shape, T, Scope)>);
    /// Typed `cp.reduce.async.bulk.tensor`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorS2gReduce<Shape, T, Scope, Op>(PhantomData<fn() -> (Shape, T, Scope, Op)>);

    /// Complete mapped tile implemented by one or more statically selected
    /// `tcgen05.cp` instructions. `Raw` is a legal raw CP specialization.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Tcgen05Cp<Shape, SourceT, DestinationT, Raw>(
        PhantomData<fn() -> (Shape, SourceT, DestinationT, Raw)>,
    );
    /// Complete tile implemented by `tcgen05.ld`. `Mapping` selects either
    /// the general frontend map or a frontend-proven canonical PTX mapping.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Tcgen05Ld<Shape, SourceT, DestinationT, Raw, Mapping = Mapped>(
        PhantomData<fn() -> (Shape, SourceT, DestinationT, Raw, Mapping)>,
    );
    /// Complete tile implemented by `tcgen05.st`. `Mapping` selects either
    /// the general frontend map or a frontend-proven canonical PTX mapping.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Tcgen05St<Shape, SourceT, DestinationT, Raw, Mapping = Mapped>(
        PhantomData<fn() -> (Shape, SourceT, DestinationT, Raw, Mapping)>,
    );

    /// General frontend-supplied element mapping.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Mapped;
    /// General element maps with the exact two-half Layout-E interpretation
    /// used by the supported `tcgen05.mma.ws` form.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MappedWsBatched;
    /// General element maps with two TMEM A banks per target CTA in CTA2.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MappedCta2BankedA;
    /// Canonical f32 m64 mapping for `.16x256b.x8` `tcgen05.ld`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CanonicalF32M64;
    /// Canonical warpgroup register/TMEM mapping for `.32x32b` `tcgen05.st`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Canonical32x32b;

    /// Frontend-proven canonical register fragments for one
    /// `mma.sync.m16n8k{8,16}` instruction.  The marker lets the engine use
    /// its packed fragment access without moving layout evaluation into the
    /// ABI or weakening the general mapped form.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CanonicalMmaM16N8;

    /// Frontend-proven whole-buffer BF16 shared-memory mapping for dense
    /// CTA1 `tcgen05.mma`. The swizzle parameters are compile-time layout
    /// facts; the destination TMEM origin remains a runtime operand.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CanonicalBf16SsCta1<
        const A_ATOM_COLUMNS: usize,
        const A_PER_ELEMENT_SHIFT: u32,
        const A_OUTER_MASK: usize,
        const A_ATOM_SHIFT: u32,
        const B_ATOM_COLUMNS: usize,
        const B_PER_ELEMENT_SHIFT: u32,
        const B_OUTER_MASK: usize,
        const B_ATOM_SHIFT: u32,
        const REUSE_A_AS_B: bool,
    >;

    /// One complete register-resident tile implemented by the statically
    /// selected `mma.sync.m16n8k{8,16}` form.
    pub struct MmaSync<
        Input,
        const M: usize,
        const N: usize,
        const K: usize,
        const MMA_K: usize,
        const TRANS_A: bool,
        const TRANS_B: bool,
        const ACCUMULATE: bool,
        Mapping = Mapped,
    >(PhantomData<fn() -> (Input, Mapping)>);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct AShared;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ATmem;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Tf32;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct E4m3;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct E2m1;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct E8m0;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct OobNan;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct NoTensorConversion;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorTf32;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Dense;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DensePredicated;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DenseDescriptor;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DenseDescriptorPredicated;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BlockScaled<Scale>(PhantomData<fn() -> Scale>);
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BlockScaledPredicated<Scale>(PhantomData<fn() -> Scale>);
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BlockScaledDescriptor<Scale>(PhantomData<fn() -> Scale>);
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BlockScaledDescriptorPredicated<Scale>(PhantomData<fn() -> Scale>);

    /// One complete mapped TCGEN GEMM. The function (`gemm_async` versus
    /// `gemm_async_ws`) selects the PTX mnemonic; these type/const arguments
    /// specialize every compile-time operand and descriptor fact.
    pub struct Gemm<
        Mode,
        AInput,
        BInput,
        APlace,
        Access,
        const M: usize,
        const N: usize,
        const K: usize,
        const INSTR_M: usize,
        const INSTR_N: usize,
        const INSTR_K: usize,
        const CTA_GROUP: u32,
        const TRANS_A: bool,
        const TRANS_B: bool,
        const SCALE_VECTOR: usize = 1,
        const EXPECTED_DESCRIPTOR: u32 = 0,
        const DESCRIPTOR_MASK: u32 = { u32::MAX },
        Mapping = Mapped,
    >(PhantomData<fn() -> (Mode, AInput, BInput, APlace, Access, Mapping)>);
}

mod sealed {
    pub trait Shape {}
    pub trait Scope {}
    pub trait TcgenLdMapping {}
    pub trait TcgenStMapping {}
    pub trait WarpGemmMapping {}
    pub trait Gemm {}
    pub trait GemmWs {}
    pub trait GemmMode {}
    pub trait GemmAPlacement {}
    pub trait GemmMapping {}
}

trait StaticShape: sealed::Shape {
    const EXTENTS: &'static [usize];
}

impl<const X: usize> sealed::Shape for variant::Shape1<X> {}
impl<const X: usize> StaticShape for variant::Shape1<X> {
    const EXTENTS: &'static [usize] = &[X];
}
impl<const X: usize, const Y: usize> sealed::Shape for variant::Shape2<X, Y> {}
impl<const X: usize, const Y: usize> StaticShape for variant::Shape2<X, Y> {
    const EXTENTS: &'static [usize] = &[X, Y];
}
impl<const X: usize, const Y: usize, const Z: usize> sealed::Shape for variant::Shape3<X, Y, Z> {}
impl<const X: usize, const Y: usize, const Z: usize> StaticShape for variant::Shape3<X, Y, Z> {
    const EXTENTS: &'static [usize] = &[X, Y, Z];
}
impl<const A: usize, const B: usize, const C: usize, const D: usize> sealed::Shape
    for variant::Shape4<A, B, C, D>
{
}
impl<const A: usize, const B: usize, const C: usize, const D: usize> StaticShape
    for variant::Shape4<A, B, C, D>
{
    const EXTENTS: &'static [usize] = &[A, B, C, D];
}
impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize> sealed::Shape
    for variant::Shape5<A, B, C, D, E>
{
}
impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize> StaticShape
    for variant::Shape5<A, B, C, D, E>
{
    const EXTENTS: &'static [usize] = &[A, B, C, D, E];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TileScopeKind {
    Thread,
    Warp,
    Warpgroup,
    Cta,
}

trait StaticScope: sealed::Scope {
    const KIND: TileScopeKind;
    fn lanes(context: ExecCtx, linear: usize) -> Result<Vec<usize>, EngineError> {
        scope_lanes(Self::KIND, context, linear)
    }
}

impl sealed::Scope for variant::Thread {}
impl StaticScope for variant::Thread {
    const KIND: TileScopeKind = TileScopeKind::Thread;
}

impl sealed::Scope for variant::Warp {}
impl StaticScope for variant::Warp {
    const KIND: TileScopeKind = TileScopeKind::Warp;
}

impl sealed::Scope for variant::Warpgroup<4> {}
impl StaticScope for variant::Warpgroup<4> {
    const KIND: TileScopeKind = TileScopeKind::Warpgroup;
}

impl sealed::Scope for variant::Cta {}
impl StaticScope for variant::Cta {
    const KIND: TileScopeKind = TileScopeKind::Cta;
}

fn coordinates_for(extents: &[usize], linear: usize) -> Result<Vec<i64>, EngineError> {
    if extents.is_empty() || extents.iter().any(|&extent| extent == 0) {
        return Err(EngineError::message(
            "typed tile shape must have positive rank and extents",
        ));
    }
    let mut remainder = linear;
    let mut result = vec![0_i64; extents.len()];
    for axis in (0..extents.len()).rev() {
        let extent = extents[axis];
        result[axis] = i64::try_from(remainder % extent)
            .map_err(|_| EngineError::message("typed tile coordinate exceeds i64"))?;
        remainder /= extent;
    }
    Ok(result)
}

fn coordinates<Shape: StaticShape>(linear: usize) -> Result<Vec<i64>, EngineError> {
    coordinates_for(Shape::EXTENTS, linear)
}

fn element_count_for(extents: &[usize]) -> Result<usize, EngineError> {
    extents.iter().try_fold(1_usize, |count, &extent| {
        if extent == 0 {
            return Err(EngineError::message("typed tile extent must be positive"));
        }
        count
            .checked_mul(extent)
            .ok_or_else(|| EngineError::message("typed tile element count overflow"))
    })
}

fn element_count<Shape: StaticShape>() -> Result<usize, EngineError> {
    element_count_for(Shape::EXTENTS)
}

fn scope_lanes(
    scope: TileScopeKind,
    context: ExecCtx,
    linear: usize,
) -> Result<Vec<usize>, EngineError> {
    match scope {
        TileScopeKind::Thread => Ok(context.active_mask().into_iter().collect()),
        TileScopeKind::Warp => {
            let lane = linear % crate::WARP_SIZE;
            Ok(context
                .active_mask()
                .contains(lane)
                .then_some(lane)
                .into_iter()
                .collect())
        }
        TileScopeKind::Warpgroup => {
            let inner = context.into_inner();
            let owner = linear % (4 * crate::WARP_SIZE);
            let relative_warp = inner.warp_id_in_cta() % 4;
            let lane = owner % crate::WARP_SIZE;
            Ok(
                (owner / crate::WARP_SIZE == relative_warp && context.active_mask().contains(lane))
                    .then_some(lane)
                    .into_iter()
                    .collect(),
            )
        }
        TileScopeKind::Cta => {
            let inner = context.into_inner();
            let threads = inner
                .topology()
                .warps_per_cta()
                .checked_mul(crate::WARP_SIZE)
                .ok_or_else(|| EngineError::message("CTA tile thread count overflow"))?;
            let owner = linear % threads;
            let lane = owner % crate::WARP_SIZE;
            Ok((owner / crate::WARP_SIZE == inner.warp_id_in_cta()
                && context.active_mask().contains(lane))
            .then_some(lane)
            .into_iter()
            .collect())
        }
    }
}

trait TileElement: super::mem::MemoryType {}
impl<T: super::mem::MemoryType> TileElement for T {}

trait TcgenTileElement {
    type Physical: RuntimeScalar + Copy + Send + Sync + 'static;
}

/// Concrete TCGEN transfer entries compiled into the engine crate.
///
/// The public tile ABI remains generic over the frontend's static element
/// markers, but its monomorphized adapter only selects one of these methods.
/// Layout traversal, validation, and memory traffic stay in the
/// already-compiled engine artifact.
trait TcgenTransferEntry<DestinationT>: TcgenTileElement
where
    DestinationT: TcgenTileElement<Physical = Self::Physical>,
{
    fn mapped_cp(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<super::Tmem>,
        source: &MappedView<Shared>,
        extents: &[usize],
        cta_group: u32,
        access: crate::TmemAccessMode,
        pipeline_operation: crate::runtime::TcgenPipelineOperation,
    ) -> Result<(), EngineError>;

    fn mapped_ld(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<Register>,
        source: &MappedView<super::Tmem>,
        extents: &[usize],
        access: crate::TmemAccessMode,
    ) -> Result<(), EngineError>;

    fn mapped_st(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<super::Tmem>,
        source: &MappedView<Register>,
        extents: &[usize],
        access: crate::TmemAccessMode,
    ) -> Result<(), EngineError>;

    fn canonical_32x32b_ld(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<Register>,
        source: &MappedView<super::Tmem>,
        access: crate::TmemAccessMode,
        elements_per_lane: usize,
        base_lane: R<i64>,
        base_tcol: R<i64>,
        allocated_addr: R<i64>,
    ) -> Result<(), EngineError>;

    fn canonical_32x32b_st(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<super::Tmem>,
        source: &MappedView<Register>,
        access: crate::TmemAccessMode,
        elements_per_lane: usize,
        base_lane: R<i64>,
        base_tcol: R<i64>,
        allocated_addr: R<i64>,
    ) -> Result<(), EngineError>;
}

macro_rules! direct_tile_elements {
    ($($marker:ty => $scalar:ty),+ $(,)?) => {
        $(
            impl TcgenTileElement for $marker {
                type Physical = $scalar;
            }
        )+
    };
}

direct_tile_elements!(
    super::mem::variant::Bool => bool,
    super::reg::variant::I8 => i8,
    super::reg::variant::I16 => i16,
    super::reg::variant::I32 => i32,
    super::reg::variant::I64 => i64,
    super::reg::variant::U8 => u8,
    super::reg::variant::U16 => u16,
    super::reg::variant::U32 => u32,
    super::reg::variant::U64 => u64,
    super::reg::variant::B32 => u32,
    super::reg::variant::B64 => u64,
    super::reg::variant::F32 => f32,
);

impl TcgenTileElement for super::reg::variant::F16 {
    type Physical = u16;
}

impl TcgenTileElement for super::reg::variant::Bf16 {
    type Physical = u16;
}

impl TcgenTileElement for super::reg::variant::F64 {
    type Physical = f64;
}

impl TcgenTileElement for variant::E4m3 {
    type Physical = u8;
}

impl TcgenTileElement for variant::E8m0 {
    type Physical = u8;
}

macro_rules! tcgen_transfer_entry {
    ($source:ty, $destination:ty) => {
        impl TcgenTransferEntry<$destination> for $source {
            fn mapped_cp(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                destination: &MappedView<super::Tmem>,
                source: &MappedView<Shared>,
                extents: &[usize],
                cta_group: u32,
                access: crate::TmemAccessMode,
                pipeline_operation: crate::runtime::TcgenPipelineOperation,
            ) -> Result<(), EngineError> {
                execute_mapped_tcgen_transfer::<$source, $destination, Shared, super::Tmem>(
                    warp,
                    context,
                    site,
                    destination,
                    source,
                    extents,
                    pipeline_operation,
                    cta_group,
                    access,
                    false,
                    "tile tcgen05.cp source map",
                    "tile tcgen05.cp destination map",
                )
            }

            fn mapped_ld(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                destination: &MappedView<Register>,
                source: &MappedView<super::Tmem>,
                extents: &[usize],
                access: crate::TmemAccessMode,
            ) -> Result<(), EngineError> {
                execute_mapped_tcgen_transfer::<$source, $destination, super::Tmem, Register>(
                    warp,
                    context,
                    site,
                    destination,
                    source,
                    extents,
                    crate::runtime::TcgenPipelineOperation::Load,
                    1,
                    access,
                    true,
                    "tile tcgen05.ld source map",
                    "tile tcgen05.ld destination map",
                )
            }

            fn mapped_st(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                destination: &MappedView<super::Tmem>,
                source: &MappedView<Register>,
                extents: &[usize],
                access: crate::TmemAccessMode,
            ) -> Result<(), EngineError> {
                execute_mapped_tcgen_transfer::<$source, $destination, Register, super::Tmem>(
                    warp,
                    context,
                    site,
                    destination,
                    source,
                    extents,
                    crate::runtime::TcgenPipelineOperation::Store,
                    1,
                    access,
                    false,
                    "tile tcgen05.st source map",
                    "tile tcgen05.st destination map",
                )
            }

            fn canonical_32x32b_ld(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                destination: &MappedView<Register>,
                source: &MappedView<super::Tmem>,
                access: crate::TmemAccessMode,
                elements_per_lane: usize,
                base_lane: R<i64>,
                base_tcol: R<i64>,
                allocated_addr: R<i64>,
            ) -> Result<(), EngineError> {
                issue_canonical_32x32b::<$source>(
                    warp,
                    context,
                    site,
                    source,
                    destination,
                    Canonical32x32bDirection::Load,
                    access,
                    elements_per_lane,
                    base_lane,
                    base_tcol,
                    allocated_addr,
                )
            }

            fn canonical_32x32b_st(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                destination: &MappedView<super::Tmem>,
                source: &MappedView<Register>,
                access: crate::TmemAccessMode,
                elements_per_lane: usize,
                base_lane: R<i64>,
                base_tcol: R<i64>,
                allocated_addr: R<i64>,
            ) -> Result<(), EngineError> {
                issue_canonical_32x32b::<$source>(
                    warp,
                    context,
                    site,
                    destination,
                    source,
                    Canonical32x32bDirection::Store,
                    access,
                    elements_per_lane,
                    base_lane,
                    base_tcol,
                    allocated_addr,
                )
            }
        }
    };
}

// Every pair accepted by the old physical-type equality contract gets a
// concrete entry. This preserves reinterpret variants without leaving
// their transfer loops for downstream rustc to instantiate.
tcgen_transfer_entry!(super::mem::variant::Bool, super::mem::variant::Bool);
tcgen_transfer_entry!(super::reg::variant::I8, super::reg::variant::I8);
tcgen_transfer_entry!(super::reg::variant::I16, super::reg::variant::I16);
tcgen_transfer_entry!(super::reg::variant::I32, super::reg::variant::I32);
tcgen_transfer_entry!(super::reg::variant::I64, super::reg::variant::I64);
tcgen_transfer_entry!(super::reg::variant::F32, super::reg::variant::F32);
tcgen_transfer_entry!(super::reg::variant::F64, super::reg::variant::F64);

macro_rules! tcgen_u8_entries {
    ($source:ty) => {
        tcgen_transfer_entry!($source, super::reg::variant::U8);
        tcgen_transfer_entry!($source, variant::E4m3);
        tcgen_transfer_entry!($source, variant::E8m0);
    };
}
tcgen_u8_entries!(super::reg::variant::U8);
tcgen_u8_entries!(variant::E4m3);
tcgen_u8_entries!(variant::E8m0);

macro_rules! tcgen_u16_entries {
    ($source:ty) => {
        tcgen_transfer_entry!($source, super::reg::variant::U16);
        tcgen_transfer_entry!($source, super::reg::variant::F16);
        tcgen_transfer_entry!($source, super::reg::variant::Bf16);
    };
}
tcgen_u16_entries!(super::reg::variant::U16);
tcgen_u16_entries!(super::reg::variant::F16);
tcgen_u16_entries!(super::reg::variant::Bf16);

macro_rules! tcgen_u32_entries {
    ($source:ty) => {
        tcgen_transfer_entry!($source, super::reg::variant::U32);
        tcgen_transfer_entry!($source, super::reg::variant::B32);
    };
}
tcgen_u32_entries!(super::reg::variant::U32);
tcgen_u32_entries!(super::reg::variant::B32);

macro_rules! tcgen_u64_entries {
    ($source:ty) => {
        tcgen_transfer_entry!($source, super::reg::variant::U64);
        tcgen_transfer_entry!($source, super::reg::variant::B64);
    };
}
tcgen_u64_entries!(super::reg::variant::U64);
tcgen_u64_entries!(super::reg::variant::B64);

mod operands {
    use super::{AsyncSourceFill, LaneMask};

    pub trait ClassicFill {
        type Args;
        const SOURCE_FILL: AsyncSourceFill;
        fn lane_enabled(args: &Self::Args, lane: usize) -> bool;

        fn enabled_mask(args: &Self::Args, active: LaneMask) -> LaneMask {
            LaneMask::from_predicate(|lane| active.contains(lane) && Self::lane_enabled(args, lane))
        }
    }
}

use operands::ClassicFill;

struct ClassicBytes<const VALUE: usize>;
trait ValidClassicBytes {}
impl ValidClassicBytes for ClassicBytes<4> {}
impl ValidClassicBytes for ClassicBytes<8> {}
impl ValidClassicBytes for ClassicBytes<16> {}

struct TileCtaGroup<const VALUE: u32>;
trait ValidTileCtaGroup {}
impl ValidTileCtaGroup for TileCtaGroup<1> {}
impl ValidTileCtaGroup for TileCtaGroup<2> {}

impl ClassicFill for variant::NoFill {
    type Args = ();
    const SOURCE_FILL: AsyncSourceFill = AsyncSourceFill::None;
    fn lane_enabled(_args: &Self::Args, _lane: usize) -> bool {
        true
    }
}

impl ClassicFill for variant::ZeroFill {
    type Args = R<bool>;
    const SOURCE_FILL: AsyncSourceFill = AsyncSourceFill::Zero;
    fn lane_enabled(args: &Self::Args, lane: usize) -> bool {
        args[lane]
    }
}

struct CopyPlan {
    elements: Vec<TypedTileCopyElement>,
    destination_rank: Option<usize>,
}

trait SyncCopySpace: MemorySpace {
    const LANE_PRIVATE: bool;
}

impl SyncCopySpace for Global {
    const LANE_PRIVATE: bool = false;
}

impl SyncCopySpace for Shared {
    const LANE_PRIVATE: bool = false;
}

impl SyncCopySpace for Local {
    const LANE_PRIVATE: bool = true;
}

impl SyncCopySpace for Register {
    const LANE_PRIVATE: bool = true;
}

fn byte_index<S: MemorySpace>(
    reference: super::ElementRef<S>,
    itemsize: usize,
    label: &str,
) -> Result<(i64, bool), EngineError> {
    if !reference.is_in_bounds() {
        return Ok((0, false));
    }
    let ElementLocation::ByteOffset(offset) = reference.location() else {
        return Err(EngineError::message(format!(
            "{label} produced a TMEM coordinate for a byte-addressed copy"
        )));
    };
    if offset < 0 {
        return Err(EngineError::message(format!(
            "{label} produced negative in-bounds byte offset {offset}"
        )));
    }
    let itemsize = i128::try_from(itemsize)
        .map_err(|_| EngineError::message("typed copy itemsize exceeds i128"))?;
    if offset % itemsize != 0 {
        return Err(EngineError::message(format!(
            "{label} byte offset {offset} is not aligned to {itemsize} bytes"
        )));
    }
    let index = i64::try_from(offset / itemsize)
        .map_err(|_| EngineError::message(format!("{label} element index exceeds i64")))?;
    Ok((index, true))
}

fn mapped_copy_plan<Src, Dst>(
    context: ExecCtx,
    source: &MappedView<Src>,
    destination: &MappedView<Dst>,
    extents: &[usize],
    scope: TileScopeKind,
    itemsize: usize,
) -> Result<CopyPlan, EngineError>
where
    Src: MemorySpace,
    Dst: MemorySpace,
{
    let count = element_count_for(extents)?;
    let mut elements = Vec::new();
    let mut destination_rank = None;
    for linear in 0..count {
        let coordinates = coordinates_for(extents, linear)?;
        for lane in scope_lanes(scope, context, linear)? {
            let lane_id = LaneId::from_index(lane);
            let logical = LogicalCoord::new(&coordinates);
            let source_ref = source
                .map(logical, lane_id)
                .map_err(|error| EngineError::message(format!("typed copy source map: {error}")))?;
            let destination_ref = destination.map(logical, lane_id).map_err(|error| {
                EngineError::message(format!("typed copy destination map: {error}"))
            })?;
            if source_ref.target_rank().is_some() {
                return Err(EngineError::message(
                    "typed copy source map cannot select a remote CTA",
                ));
            }
            if let Some(rank) = destination_ref.target_rank() {
                let rank = usize::try_from(rank)
                    .map_err(|_| EngineError::message("mapped CTA rank exceeds usize"))?;
                match destination_rank {
                    Some(previous) if previous != rank => {
                        return Err(EngineError::message(
                            "one typed copy cannot target multiple non-multicast CTA ranks",
                        ));
                    }
                    None => destination_rank = Some(rank),
                    _ => {}
                }
            }
            let (source_index, source_in_bounds) =
                byte_index(source_ref, itemsize, "typed copy source map")?;
            let (destination_index, destination_in_bounds) =
                byte_index(destination_ref, itemsize, "typed copy destination map")?;
            elements.push(TypedTileCopyElement {
                source_index,
                source_in_bounds,
                source_lane: lane,
                destination_index,
                destination_in_bounds,
                destination_lane: lane,
            });
        }
    }
    Ok(CopyPlan {
        elements,
        destination_rank,
    })
}

fn mapped_sync_copy_plan<Src, Dst>(
    context: ExecCtx,
    source: &MappedView<Src>,
    destination: &MappedView<Dst>,
    extents: &[usize],
    scope: TileScopeKind,
    itemsize: usize,
) -> Result<CopyPlan, EngineError>
where
    Src: SyncCopySpace,
    Dst: SyncCopySpace,
{
    let count = element_count_for(extents)?;
    let active_lanes: Vec<_> = context.active_mask().into_iter().collect();
    let mut elements = Vec::new();
    let mut destination_rank = None;

    for linear in 0..count {
        let coordinates = coordinates_for(extents, linear)?;
        let logical = LogicalCoord::new(&coordinates);
        let mut source_candidates = Vec::new();
        let mut destination_candidates = Vec::new();

        for &lane in &active_lanes {
            let lane_id = LaneId::from_index(lane);
            let source_ref = source
                .map(logical, lane_id)
                .map_err(|error| EngineError::message(format!("typed copy source map: {error}")))?;
            if source_ref.target_rank().is_some() {
                return Err(EngineError::message(
                    "typed copy source map cannot select a remote CTA",
                ));
            }
            if source_ref.is_owned() {
                let (source_index, source_in_bounds) =
                    byte_index(source_ref, itemsize, "typed copy source map")?;
                source_candidates.push((lane, source_index, source_in_bounds));
            }

            let destination_ref = destination.map(logical, lane_id).map_err(|error| {
                EngineError::message(format!("typed copy destination map: {error}"))
            })?;
            if let Some(rank) = destination_ref.target_rank() {
                let rank = usize::try_from(rank)
                    .map_err(|_| EngineError::message("mapped CTA rank exceeds usize"))?;
                match destination_rank {
                    Some(previous) if previous != rank => {
                        return Err(EngineError::message(
                            "one typed copy cannot target multiple non-multicast CTA ranks",
                        ));
                    }
                    None => destination_rank = Some(rank),
                    _ => {}
                }
            }
            if destination_ref.is_owned() {
                let (destination_index, destination_in_bounds) =
                    byte_index(destination_ref, itemsize, "typed copy destination map")?;
                destination_candidates.push((lane, destination_index, destination_in_bounds));
            }
        }

        let source_replicated =
            !active_lanes.is_empty() && source_candidates.len() == active_lanes.len();
        let destination_replicated =
            !active_lanes.is_empty() && destination_candidates.len() == active_lanes.len();
        let default_lanes = scope_lanes(scope, context, linear)?;

        if source_candidates.is_empty() || destination_candidates.is_empty() {
            let collective_scope = matches!(scope, TileScopeKind::Warpgroup | TileScopeKind::Cta);
            let owned_in_another_warp = collective_scope
                && (source_candidates.is_empty() && destination_replicated
                    || destination_candidates.is_empty() && source_replicated
                    || source_candidates.is_empty() && destination_candidates.is_empty());
            if owned_in_another_warp {
                continue;
            }
            return Err(EngineError::message(format!(
                "typed copy logical element {linear} has no source or destination owner in this execution scope"
            )));
        }

        let mut append = |source: (usize, i64, bool), destination: (usize, i64, bool)| {
            let (source_lane, source_index, source_in_bounds) = source;
            let (destination_lane, destination_index, destination_in_bounds) = destination;
            elements.push(TypedTileCopyElement {
                source_index,
                source_in_bounds,
                source_lane,
                destination_index,
                destination_in_bounds,
                destination_lane,
            });
        };

        let source_at = |lane: usize| {
            source_candidates
                .iter()
                .copied()
                .find(|candidate| candidate.0 == lane)
        };
        let destination_at = |lane: usize| {
            destination_candidates
                .iter()
                .copied()
                .find(|candidate| candidate.0 == lane)
        };

        match (source_replicated, destination_replicated) {
            (true, true) => {
                let owner_lanes = if Dst::LANE_PRIVATE {
                    &active_lanes
                } else {
                    &default_lanes
                };
                for &lane in owner_lanes {
                    append(
                        source_at(lane).expect("replicated source includes every active lane"),
                        destination_at(lane)
                            .expect("replicated destination includes every active lane"),
                    );
                }
            }
            (true, false) => {
                for destination in destination_candidates.iter().copied() {
                    append(
                        source_at(destination.0)
                            .expect("replicated source includes destination owner lane"),
                        destination,
                    );
                }
            }
            (false, true) => {
                for source in source_candidates.iter().copied() {
                    append(
                        source,
                        destination_at(source.0)
                            .expect("replicated destination includes source owner lane"),
                    );
                }
            }
            (false, false) if source_candidates.len() == 1 => {
                let source = source_candidates[0];
                for destination in destination_candidates.iter().copied() {
                    append(source, destination);
                }
            }
            (false, false) if destination_candidates.len() == 1 => {
                let destination = destination_candidates[0];
                let source = source_at(destination.0).unwrap_or(source_candidates[0]);
                append(source, destination);
            }
            (false, false) if source_candidates.len() == destination_candidates.len() => {
                for (source, destination) in source_candidates
                    .iter()
                    .copied()
                    .zip(destination_candidates.iter().copied())
                {
                    append(source, destination);
                }
            }
            _ => {
                return Err(EngineError::message(format!(
                    "typed copy logical element {linear} has incompatible source/destination owner counts {} and {}",
                    source_candidates.len(),
                    destination_candidates.len()
                )));
            }
        }
    }

    Ok(CopyPlan {
        elements,
        destination_rank,
    })
}

trait CopySyncProtocol {
    const ENABLED: bool;
}

impl CopySyncProtocol for variant::SnapshotSync {
    const ENABLED: bool = true;
}

impl CopySyncProtocol for variant::NoSnapshotSync {
    const ENABLED: bool = false;
}

trait CopyFill {
    const ZERO_FILL: bool;
}

impl CopyFill for variant::NoFill {
    const ZERO_FILL: bool = false;
}

impl CopyFill for variant::ZeroFill {
    const ZERO_FILL: bool = true;
}

impl<Shape, T, Src, Dst, Scope, Sync, Fill> copy_spec::sealed::Sealed
    for variant::Copy<Shape, T, Src, Dst, Scope, Sync, Fill>
where
    Shape: StaticShape,
    T: TileElement,
    Src: MemorySpace,
    Dst: MemorySpace,
    Scope: StaticScope,
    Sync: CopySyncProtocol,
    Fill: CopyFill,
{
}

impl<Shape, T, Src, Dst, Scope, Sync, Fill> copy_spec::Variant
    for variant::Copy<Shape, T, Src, Dst, Scope, Sync, Fill>
where
    Shape: StaticShape,
    T: TileElement,
    Src: MemorySpace,
    Dst: MemorySpace,
    Scope: StaticScope,
    Sync: CopySyncProtocol,
    Fill: CopyFill,
{
    type Source = Src;
    type Destination = Dst;
    type Output = ();
}

macro_rules! copy_execute_for_mode_and_spaces {
    ($warp:ty, $source:ty, $destination:ty) => {
        const _: () = {
            #[inline(never)]
            async fn entry(
                warp: &mut $warp,
                context: ExecCtx,
                site: SiteId,
                destination: &MappedView<$destination>,
                source: &MappedView<$source>,
                extents: &[usize],
                scope: TileScopeKind,
                itemsize: usize,
                sync: bool,
                zero_fill: bool,
            ) -> Result<(), EngineError> {
                execute_typed_copy(
                    warp,
                    context,
                    site,
                    destination,
                    source,
                    extents,
                    scope,
                    itemsize,
                    sync,
                    zero_fill,
                )
                .await
            }

            impl<Shape, T, Scope, Sync, Fill> copy_spec::sealed::Execute<$warp>
                for variant::Copy<Shape, T, $source, $destination, Scope, Sync, Fill>
            where
                Shape: StaticShape,
                T: TileElement,
                Scope: StaticScope,
                Sync: CopySyncProtocol,
                Fill: CopyFill,
            {
                #[inline(always)]
                async fn execute(
                    warp: &mut $warp,
                    context: ExecCtx,
                    site: SiteId,
                    destination: &MappedView<$destination>,
                    source: &MappedView<$source>,
                ) -> Result<Self::Output, EngineError> {
                    entry(
                        warp,
                        context,
                        site,
                        destination,
                        source,
                        Shape::EXTENTS,
                        Scope::KIND,
                        <<T as super::mem::MemoryType>::Storage as RuntimeScalar>::BYTE_LEN,
                        Sync::ENABLED,
                        Fill::ZERO_FILL,
                    )
                    .await
                }
            }
        };
    };
}

macro_rules! copy_execute_for_mode {
    ($warp:ty) => {
        copy_execute_for_mode_and_spaces!($warp, Global, Global);
        copy_execute_for_mode_and_spaces!($warp, Global, Shared);
        copy_execute_for_mode_and_spaces!($warp, Global, Local);
        copy_execute_for_mode_and_spaces!($warp, Global, Register);
        copy_execute_for_mode_and_spaces!($warp, Shared, Global);
        copy_execute_for_mode_and_spaces!($warp, Shared, Shared);
        copy_execute_for_mode_and_spaces!($warp, Shared, Local);
        copy_execute_for_mode_and_spaces!($warp, Shared, Register);
        copy_execute_for_mode_and_spaces!($warp, Local, Global);
        copy_execute_for_mode_and_spaces!($warp, Local, Shared);
        copy_execute_for_mode_and_spaces!($warp, Local, Local);
        copy_execute_for_mode_and_spaces!($warp, Local, Register);
        copy_execute_for_mode_and_spaces!($warp, Register, Global);
        copy_execute_for_mode_and_spaces!($warp, Register, Shared);
        copy_execute_for_mode_and_spaces!($warp, Register, Local);
        copy_execute_for_mode_and_spaces!($warp, Register, Register);
    };
}

for_each_engine_mode!(test_visible, copy_execute_for_mode);

fn copy_source_mask(plan: &CopyPlan) -> WarpMask {
    plan.elements
        .iter()
        .filter(|element| element.source_in_bounds)
        .fold(WarpMask::EMPTY, |mask, element| {
            mask | WarpMask::from_bits(1_u32 << element.source_lane)
        })
}

fn copy_destination_mask(plan: &CopyPlan) -> WarpMask {
    plan.elements
        .iter()
        .filter(|element| element.destination_in_bounds)
        .fold(WarpMask::EMPTY, |mask, element| {
            mask | WarpMask::from_bits(1_u32 << element.destination_lane)
        })
}

async fn copy_scope_sync<W>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    scope: TileScopeKind,
) -> Result<(), EngineError>
where
    W: WarpHandle + Send,
{
    match scope {
        TileScopeKind::Thread => Ok(()),
        TileScopeKind::Warp => {
            super::warp::execute_bar_warp_sync(warp, context, site, R::splat(u32::MAX)).await
        }
        TileScopeKind::Warpgroup => {
            super::sync::execute_bar_sync(warp, context, site, 8, 4 * crate::WARP_SIZE as i64).await
        }
        TileScopeKind::Cta => {
            let threads = context
                .into_inner()
                .topology()
                .warps_per_cta()
                .checked_mul(crate::WARP_SIZE)
                .ok_or_else(|| EngineError::message("CTA tile copy thread count overflow"))?;
            super::sync::execute_bar_sync(
                warp,
                context,
                site,
                0,
                i64::try_from(threads)
                    .map_err(|_| EngineError::message("CTA thread count exceeds i64"))?,
            )
            .await
        }
    }
}

#[inline(never)]
async fn execute_typed_copy<Src, Dst, W>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    destination: &MappedView<Dst>,
    source: &MappedView<Src>,
    extents: &[usize],
    scope: TileScopeKind,
    itemsize: usize,
    sync: bool,
    zero_fill: bool,
) -> Result<(), EngineError>
where
    Src: SyncCopySpace,
    Dst: SyncCopySpace,
    W: WarpHandle + Send,
{
    let plan = mapped_sync_copy_plan(context, source, destination, extents, scope, itemsize)?;
    let source_mask = copy_source_mask(&plan);
    let source_operation = if source_mask.is_empty() {
        None
    } else {
        engine(warp).begin_optional_operation(
            context.into_inner().with_active_mask(source_mask),
            site.get(),
            OperationKind::Load,
            false,
        )?
    };
    let snapshots = engine(warp).typed_tile_copy_snapshot(
        source_operation.as_ref(),
        &context.into_inner(),
        itemsize,
        source.allocation().inner(),
        source.allocation().logical_buffer(),
        zero_fill,
        &plan.elements,
    )?;
    engine(warp).finish_optional_operation(&source_operation)?;

    if sync {
        copy_scope_sync(warp, context, site, scope).await?;
    }

    let destination_mask = copy_destination_mask(&plan);
    let destination_operation = if destination_mask.is_empty() {
        None
    } else {
        engine(warp).begin_optional_operation(
            context.into_inner().with_active_mask(destination_mask),
            site.get(),
            OperationKind::Store,
            false,
        )?
    };
    engine(warp).typed_tile_copy_restore(
        destination_operation.as_ref(),
        &context.into_inner(),
        itemsize,
        destination.allocation().inner(),
        destination.allocation().logical_buffer(),
        plan.destination_rank,
        &plan.elements,
        &snapshots,
    )?;
    engine(warp).finish_optional_operation(&destination_operation)?;

    if sync {
        copy_scope_sync(warp, context, site, scope).await?;
    }
    Ok(())
}

fn issue_group_copy<W: WarpHandle, Src: MemorySpace, Dst: MemorySpace>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    domain: AsyncGroupDomain,
    defer_global_writes: bool,
    itemsize: usize,
    source: &MappedView<Src>,
    destination: &MappedView<Dst>,
    plan: &CopyPlan,
    source_fill: AsyncSourceFill,
    reduction: Option<DeferredGlobalReduction>,
) -> Result<(), EngineError> {
    if plan
        .elements
        .iter()
        .any(|element| element.source_lane != element.destination_lane)
    {
        return Err(EngineError::message(
            "asynchronous mapped copy requires identical source/destination execution lanes",
        ));
    }
    let operation = begin(warp, context, site, OperationKind::AsyncIssue, true)?;
    let mut replay =
        |emit: &mut dyn FnMut(i64, bool, i64, bool, usize) -> Result<(), crate::EngineError>| {
            for element in &plan.elements {
                emit(
                    element.source_index,
                    element.source_in_bounds,
                    element.destination_index,
                    element.destination_in_bounds,
                    element.source_lane,
                )?;
            }
            Ok(())
        };
    engine(warp).typed_async_group_issue_elements(
        operation.as_ref(),
        &context.into_inner(),
        domain,
        defer_global_writes,
        context.active_mask().into_inner(),
        itemsize,
        source.allocation().inner(),
        destination.allocation().inner(),
        None,
        plan.destination_rank,
        source_fill,
        false,
        reduction,
        &mut replay,
    )?;
    finish(warp, &operation)
}

#[derive(Clone)]
struct PayloadControl {
    barrier: Address<Shared>,
    cta_group: i64,
    cta_mask: u64,
    multicast: bool,
}

fn issue_payload_copy<W: WarpHandle, Src: MemorySpace, Dst: MemorySpace>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    itemsize: usize,
    source: &MappedView<Src>,
    destination: &MappedView<Dst>,
    plan: &CopyPlan,
    control: PayloadControl,
    source_fill: AsyncSourceFill,
    round_to_tf32: bool,
) -> Result<(), EngineError> {
    let issuer_lane = singleton_issue_lane(context, "typed mbarrier-backed copy")?;
    if plan.elements.iter().any(|element| {
        element.source_lane != issuer_lane || element.destination_lane != issuer_lane
    }) {
        return Err(EngineError::message(
            "typed mbarrier-backed copy map assigned work outside its issuing lane",
        ));
    }
    let inner = context.into_inner();
    let source_barrier = control.barrier.inner().resolve_shared_barrier(
        &inner,
        context.active_mask().into_inner(),
        plan.destination_rank,
    )?;
    let operation = begin(warp, context, site, OperationKind::AsyncIssue, true)?;
    let mut replay = |emit: &mut dyn FnMut(
        &[(i64, i64, bool, bool); crate::WARP_SIZE],
        usize,
    ) -> Result<(), crate::EngineError>| {
        for chunk in plan.elements.chunks(crate::WARP_SIZE) {
            let mut batch = [(0_i64, 0_i64, false, false); crate::WARP_SIZE];
            for (slot, element) in chunk.iter().enumerate() {
                batch[slot] = (
                    element.source_index,
                    element.destination_index,
                    element.source_in_bounds,
                    element.destination_in_bounds,
                );
            }
            emit(&batch, chunk.len())?;
        }
        Ok(())
    };
    engine(warp).async_payload_issue_batches(
        operation.as_ref(),
        &inner,
        control.barrier.inner(),
        context.active_mask().into_inner(),
        source_barrier,
        control.cta_group,
        control.cta_mask,
        control.multicast,
        itemsize,
        source.allocation().inner(),
        destination.allocation().inner(),
        control.multicast.then_some(control.cta_mask),
        plan.destination_rank,
        source_fill,
        round_to_tf32,
        None,
        None,
        &mut replay,
    )?;
    finish(warp, &operation)
}

impl<Shape, T, Scope, const BYTES: usize, Fill> classic_spec::sealed::Sealed
    for variant::CpAsync<Shape, T, Scope, BYTES, Fill>
where
    Shape: StaticShape,
    T: TileElement,
    Scope: StaticScope,
    Fill: ClassicFill,
    ClassicBytes<BYTES>: ValidClassicBytes,
{
}

impl<Shape, T, Scope, const BYTES: usize, Fill> classic_spec::Variant
    for variant::CpAsync<Shape, T, Scope, BYTES, Fill>
where
    Shape: StaticShape,
    T: TileElement,
    Scope: StaticScope,
    Fill: ClassicFill,
    ClassicBytes<BYTES>: ValidClassicBytes,
{
    type Args = Fill::Args;
    type Source = Global;
    type Destination = Shared;
    type Output = ();
}

macro_rules! classic_execute_for_mode {
    ($warp:ty) => {
        impl<Shape, T, Scope, const BYTES: usize, Fill> classic_spec::sealed::Execute<$warp>
            for variant::CpAsync<Shape, T, Scope, BYTES, Fill>
        where
            Shape: StaticShape,
            T: TileElement,
            Scope: StaticScope,
            Fill: ClassicFill,
            ClassicBytes<BYTES>: ValidClassicBytes,
        {
            #[inline(always)]
            fn execute(
                warp: &mut $warp,
                context: ExecCtx,
                site: SiteId,
                destination: &MappedView<Shared>,
                source: &MappedView<Global>,
                args: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                let fill = args;
                execute_classic_cp_async_core(
                    warp,
                    context,
                    site,
                    destination,
                    source,
                    Shape::EXTENTS,
                    Scope::KIND,
                    <<T as super::mem::MemoryType>::Scalar as RuntimeScalar>::BYTE_LEN,
                    BYTES,
                    Fill::enabled_mask(&fill, context.active_mask()),
                    Fill::SOURCE_FILL,
                )
            }
        }
    };
}

for_each_engine_mode!(test_visible, classic_execute_for_mode);

#[inline(never)]
fn execute_classic_cp_async_core<W: WarpHandle>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    destination: &MappedView<Shared>,
    source: &MappedView<Global>,
    extents: &[usize],
    scope: TileScopeKind,
    itemsize: usize,
    bytes: usize,
    enabled: LaneMask,
    source_fill: AsyncSourceFill,
) -> Result<(), EngineError> {
    if bytes % itemsize != 0 {
        return Err(EngineError::message(format!(
            "tile cp.async width {} is not divisible by element width {itemsize}",
            bytes
        )));
    }
    let mut plan = mapped_copy_plan(context, source, destination, extents, scope, itemsize)?;
    for element in &mut plan.elements {
        element.source_in_bounds &= enabled.contains(element.source_lane);
        if source_fill == AsyncSourceFill::None
            && (!element.source_in_bounds || !element.destination_in_bounds)
        {
            return Err(EngineError::message(
                "non-zfill tile cp.async contains an out-of-bounds element",
            ));
        }
    }
    let elements_per_instruction = bytes / itemsize;
    for lane in context.active_mask() {
        let lane_elements = plan
            .elements
            .iter()
            .filter(|element| element.source_lane == lane)
            .count();
        if lane_elements % elements_per_instruction != 0 {
            return Err(EngineError::message(format!(
                "tile cp.async lane {lane} maps {lane_elements} elements, not a multiple of its {}-element PTX vector",
                elements_per_instruction
            )));
        }
    }
    issue_group_copy(
        warp,
        context,
        site,
        AsyncGroupDomain::CpAsync,
        false,
        itemsize,
        source,
        destination,
        &plan,
        source_fill,
        None,
    )
}

#[derive(Clone, Copy)]
enum CopyDirection {
    G2s,
    S2g,
    S2s,
}

struct BulkRuntimeArgs {
    barrier: Option<Address<Shared>>,
    cta_mask: u64,
    multicast: bool,
}

trait BulkForm: BulkCopyVariant {
    type Shape: StaticShape;
    type Scope: StaticScope;
    type Element: TileElement;
    const DIRECTION: CopyDirection;
    const CTA_GROUP: u32;
    fn split(args: Self::Args, lane: usize) -> Result<BulkRuntimeArgs, EngineError>;
}

macro_rules! impl_bulk_form {
    (
        impl($($generic:tt)*) $marker:ty;
        shape=$shape:ty, element=$element:ty, scope=$scope:ty;
        spaces=$source:ty => $destination:ty;
        direction=$direction:ident, group=$group:expr;
        base=$base:ty;
        split=|$args:ident, $lane:ident| $split:expr;
        where $($bounds:tt)*
    ) => {
        impl<$($generic)*> bulk_spec::sealed::Sealed for $marker where $($bounds)* {}
        impl<$($generic)*> bulk_spec::Variant for $marker where $($bounds)* {
            type Args = $base;
            type Source = $source;
            type Destination = $destination;
            type Output = ();
        }
        impl<$($generic)*> BulkForm for $marker where $($bounds)* {
            type Shape = $shape;
            type Scope = $scope;
            type Element = $element;
            const DIRECTION: CopyDirection = CopyDirection::$direction;
            const CTA_GROUP: u32 = $group;
            fn split(args: Self::Args, lane: usize) -> Result<BulkRuntimeArgs, EngineError> {
                let $args: $base = args;
                let $lane = lane;
                $split
            }
        }
    };
}

impl_bulk_form!(
    impl(Shape, T, Scope, const CTA_GROUP: u32)
        variant::BulkG2s<Shape, T, Scope, CTA_GROUP>;
    shape=Shape, element=T, scope=Scope;
    spaces=Global => Shared;
    direction=G2s, group=CTA_GROUP;
    base=Address<Shared>;
    split=|barrier, _lane| Ok(BulkRuntimeArgs {
        barrier: Some(barrier), cta_mask: 0, multicast: false,
    });
    where Shape: StaticShape, T: TileElement, Scope: StaticScope,
          TileCtaGroup<CTA_GROUP>: ValidTileCtaGroup
);

impl_bulk_form!(
    impl(Shape, T, Scope, const CTA_GROUP: u32)
        variant::BulkG2sMulticast<Shape, T, Scope, CTA_GROUP>;
    shape=Shape, element=T, scope=Scope;
    spaces=Global => Shared;
    direction=G2s, group=CTA_GROUP;
    base=(Address<Shared>, R<i64>);
    split=|values, lane| {
        let (barrier, masks) = values;
        let cta_mask = u64::try_from(masks[lane])
            .map_err(|_| EngineError::message("negative typed bulk multicast CTA mask"))?;
        Ok(BulkRuntimeArgs { barrier: Some(barrier), cta_mask, multicast: true })
    };
    where Shape: StaticShape, T: TileElement, Scope: StaticScope,
          TileCtaGroup<CTA_GROUP>: ValidTileCtaGroup
);

impl_bulk_form!(
    impl(Shape, T, Scope) variant::BulkS2g<Shape, T, Scope>;
    shape=Shape, element=T, scope=Scope;
    spaces=Shared => Global;
    direction=S2g, group=1;
    base=();
    split=|_values, _lane| Ok(BulkRuntimeArgs {
        barrier: None, cta_mask: 0, multicast: false,
    });
    where Shape: StaticShape, T: TileElement, Scope: StaticScope
);

impl_bulk_form!(
    impl(Shape, T, Scope) variant::BulkS2sCluster<Shape, T, Scope>;
    shape=Shape, element=T, scope=Scope;
    spaces=Shared => Shared;
    direction=S2s, group=1;
    base=Address<Shared>;
    split=|barrier, _lane| Ok(BulkRuntimeArgs {
        barrier: Some(barrier), cta_mask: 0, multicast: false,
    });
    where Shape: StaticShape, T: TileElement, Scope: StaticScope
);

macro_rules! bulk_g2s_execute_for_mode {
    ($warp:ty, $marker:ident, $base:ty) => {
        const _: () = {
            #[inline(never)]
            fn entry(
                warp: &mut $warp,
                context: ExecCtx,
                site: SiteId,
                destination: &MappedView<Shared>,
                source: &MappedView<Global>,
                extents: &[usize],
                scope: TileScopeKind,
                itemsize: usize,
                runtime: BulkRuntimeArgs,
                cta_group: u32,
            ) -> Result<(), EngineError> {
                execute_typed_bulk(
                    warp,
                    context,
                    site,
                    destination,
                    source,
                    extents,
                    scope,
                    itemsize,
                    runtime,
                    CopyDirection::G2s,
                    cta_group,
                )
            }

            impl<Shape, T, Scope, const CTA_GROUP: u32> bulk_spec::sealed::Execute<$warp>
                for variant::$marker<Shape, T, Scope, CTA_GROUP>
            where
                Shape: StaticShape,
                T: TileElement,
                Scope: StaticScope,
                TileCtaGroup<CTA_GROUP>: ValidTileCtaGroup,
                variant::$marker<Shape, T, Scope, CTA_GROUP>: BulkForm
                    + BulkCopyVariant<
                        Args = $base,
                        Source = Global,
                        Destination = Shared,
                        Output = (),
                    >,
            {
                #[inline(always)]
                fn execute(
                    warp: &mut $warp,
                    context: ExecCtx,
                    site: SiteId,
                    destination: &MappedView<Shared>,
                    source: &MappedView<Global>,
                    args: Self::Args,
                ) -> Result<Self::Output, EngineError> {
                    let lane = singleton_issue_lane(context, "typed cp.async.bulk")?;
                    entry(
                        warp,
                        context,
                        site,
                        destination,
                        source,
                        Shape::EXTENTS,
                        Scope::KIND,
                        <<T as super::mem::MemoryType>::Scalar as RuntimeScalar>::BYTE_LEN,
                        <Self as BulkForm>::split(args, lane)?,
                        CTA_GROUP,
                    )
                }
            }
        };
    };
}

macro_rules! bulk_s2g_execute_for_mode {
    ($warp:ty) => {
        const _: () = {
            #[inline(never)]
            fn entry(
                warp: &mut $warp,
                context: ExecCtx,
                site: SiteId,
                destination: &MappedView<Global>,
                source: &MappedView<Shared>,
                extents: &[usize],
                scope: TileScopeKind,
                itemsize: usize,
            ) -> Result<(), EngineError> {
                execute_typed_bulk(
                    warp,
                    context,
                    site,
                    destination,
                    source,
                    extents,
                    scope,
                    itemsize,
                    BulkRuntimeArgs {
                        barrier: None,
                        cta_mask: 0,
                        multicast: false,
                    },
                    CopyDirection::S2g,
                    1,
                )
            }

            impl<Shape, T, Scope> bulk_spec::sealed::Execute<$warp>
                for variant::BulkS2g<Shape, T, Scope>
            where
                Shape: StaticShape,
                T: TileElement,
                Scope: StaticScope,
                variant::BulkS2g<Shape, T, Scope>:
                    BulkCopyVariant<Args = (), Source = Shared, Destination = Global, Output = ()>,
            {
                #[inline(always)]
                fn execute(
                    warp: &mut $warp,
                    context: ExecCtx,
                    site: SiteId,
                    destination: &MappedView<Global>,
                    source: &MappedView<Shared>,
                    (): Self::Args,
                ) -> Result<Self::Output, EngineError> {
                    entry(
                        warp,
                        context,
                        site,
                        destination,
                        source,
                        Shape::EXTENTS,
                        Scope::KIND,
                        <<T as super::mem::MemoryType>::Scalar as RuntimeScalar>::BYTE_LEN,
                    )
                }
            }
        };
    };
}

macro_rules! bulk_s2s_execute_for_mode {
    ($warp:ty) => {
        const _: () = {
            #[inline(never)]
            fn entry(
                warp: &mut $warp,
                context: ExecCtx,
                site: SiteId,
                destination: &MappedView<Shared>,
                source: &MappedView<Shared>,
                extents: &[usize],
                scope: TileScopeKind,
                itemsize: usize,
                barrier: Address<Shared>,
            ) -> Result<(), EngineError> {
                execute_typed_bulk(
                    warp,
                    context,
                    site,
                    destination,
                    source,
                    extents,
                    scope,
                    itemsize,
                    BulkRuntimeArgs {
                        barrier: Some(barrier),
                        cta_mask: 0,
                        multicast: false,
                    },
                    CopyDirection::S2s,
                    1,
                )
            }

            impl<Shape, T, Scope> bulk_spec::sealed::Execute<$warp>
                for variant::BulkS2sCluster<Shape, T, Scope>
            where
                Shape: StaticShape,
                T: TileElement,
                Scope: StaticScope,
                variant::BulkS2sCluster<Shape, T, Scope>: BulkCopyVariant<
                    Args = Address<Shared>,
                    Source = Shared,
                    Destination = Shared,
                    Output = (),
                >,
            {
                #[inline(always)]
                fn execute(
                    warp: &mut $warp,
                    context: ExecCtx,
                    site: SiteId,
                    destination: &MappedView<Shared>,
                    source: &MappedView<Shared>,
                    barrier: Self::Args,
                ) -> Result<Self::Output, EngineError> {
                    entry(
                        warp,
                        context,
                        site,
                        destination,
                        source,
                        Shape::EXTENTS,
                        Scope::KIND,
                        <<T as super::mem::MemoryType>::Scalar as RuntimeScalar>::BYTE_LEN,
                        barrier,
                    )
                }
            }
        };
    };
}

macro_rules! bulk_execute_for_mode {
    ($warp:ty) => {
        bulk_g2s_execute_for_mode!($warp, BulkG2s, Address<Shared>);
        bulk_g2s_execute_for_mode!($warp, BulkG2sMulticast, (Address<Shared>, R<i64>));
        bulk_s2g_execute_for_mode!($warp);
        bulk_s2s_execute_for_mode!($warp);
    };
}

for_each_engine_mode!(test_visible, bulk_execute_for_mode);

#[inline(never)]
fn execute_typed_bulk<Src, Dst, W>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    destination: &MappedView<Dst>,
    source: &MappedView<Src>,
    extents: &[usize],
    scope: TileScopeKind,
    itemsize: usize,
    runtime: BulkRuntimeArgs,
    direction: CopyDirection,
    cta_group: u32,
) -> Result<(), EngineError>
where
    Src: MemorySpace,
    Dst: MemorySpace,
    W: WarpHandle,
{
    let plan = mapped_copy_plan(context, source, destination, extents, scope, itemsize)?;
    match direction {
        CopyDirection::S2g => issue_group_copy(
            warp,
            context,
            site,
            AsyncGroupDomain::Bulk,
            true,
            itemsize,
            source,
            destination,
            &plan,
            AsyncSourceFill::None,
            None,
        ),
        CopyDirection::G2s | CopyDirection::S2s => {
            let barrier = runtime.barrier.ok_or_else(|| {
                EngineError::message("mbarrier-backed typed bulk copy has no barrier")
            })?;
            issue_payload_copy(
                warp,
                context,
                site,
                itemsize,
                source,
                destination,
                &plan,
                PayloadControl {
                    barrier,
                    cta_group: i64::from(cta_group),
                    cta_mask: runtime.cta_mask,
                    multicast: runtime.multicast,
                },
                if matches!(direction, CopyDirection::G2s) {
                    AsyncSourceFill::Zero
                } else {
                    AsyncSourceFill::None
                },
                false,
            )
        }
    }
}

trait TensorForm: TensorCopyVariant {
    type Shape: StaticShape;
    type Scope: StaticScope;
    type Element: TileElement;
    const DIRECTION: CopyDirection;
    const CTA_GROUP: u32;
    const SOURCE_FILL: AsyncSourceFill;
    const ROUND_TO_TF32: bool;
    fn split(args: Self::Args, lane: usize) -> Result<BulkRuntimeArgs, EngineError>;
}

trait TensorFill {
    const VALUE: AsyncSourceFill;
}

impl TensorFill for variant::ZeroFill {
    const VALUE: AsyncSourceFill = AsyncSourceFill::Zero;
}

impl TensorFill for variant::OobNan {
    const VALUE: AsyncSourceFill = AsyncSourceFill::OobNan;
}

trait TensorConversion {
    const ROUND_TO_TF32: bool;
}

impl TensorConversion for variant::NoTensorConversion {
    const ROUND_TO_TF32: bool = false;
}

impl TensorConversion for variant::TensorTf32 {
    const ROUND_TO_TF32: bool = true;
}

macro_rules! tensor_form {
    (
        $marker:ident, $source:ty => $destination:ty, $direction:ident,
        args=$base:ty, split=|$args:ident, $lane:ident| $split:expr
    ) => {
        impl<Shape, T, Scope, const CTA_GROUP: u32, Fill, Conversion> tensor_spec::sealed::Sealed
            for variant::$marker<Shape, T, Scope, CTA_GROUP, Fill, Conversion>
        where
            Shape: StaticShape,
            T: TileElement,
            Scope: StaticScope,
            Fill: TensorFill,
            Conversion: TensorConversion,
            TileCtaGroup<CTA_GROUP>: ValidTileCtaGroup,
        {
        }
        impl<Shape, T, Scope, const CTA_GROUP: u32, Fill, Conversion> tensor_spec::Variant
            for variant::$marker<Shape, T, Scope, CTA_GROUP, Fill, Conversion>
        where
            Shape: StaticShape,
            T: TileElement,
            Scope: StaticScope,
            Fill: TensorFill,
            Conversion: TensorConversion,
            TileCtaGroup<CTA_GROUP>: ValidTileCtaGroup,
        {
            type Args = $base;
            type Source = $source;
            type Destination = $destination;
            type Output = ();
        }
        impl<Shape, T, Scope, const CTA_GROUP: u32, Fill, Conversion> TensorForm
            for variant::$marker<Shape, T, Scope, CTA_GROUP, Fill, Conversion>
        where
            Shape: StaticShape,
            T: TileElement,
            Scope: StaticScope,
            Fill: TensorFill,
            Conversion: TensorConversion,
            TileCtaGroup<CTA_GROUP>: ValidTileCtaGroup,
        {
            type Shape = Shape;
            type Scope = Scope;
            type Element = T;
            const DIRECTION: CopyDirection = CopyDirection::$direction;
            const CTA_GROUP: u32 = CTA_GROUP;
            const SOURCE_FILL: AsyncSourceFill = Fill::VALUE;
            const ROUND_TO_TF32: bool = Conversion::ROUND_TO_TF32;
            fn split(args: Self::Args, lane: usize) -> Result<BulkRuntimeArgs, EngineError> {
                let $args: $base = args;
                let $lane = lane;
                $split
            }
        }
    };
}

tensor_form!(TensorG2s, Global => Shared, G2s, args=Address<Shared>, split=|barrier, _lane| {
    Ok(BulkRuntimeArgs { barrier: Some(barrier), cta_mask: 0, multicast: false })
});
tensor_form!(
    TensorG2sMulticast, Global => Shared, G2s,
    args=(Address<Shared>, R<i64>), split=|values, lane| {
        let (barrier, masks) = values;
        let cta_mask = u64::try_from(masks[lane])
            .map_err(|_| EngineError::message("negative typed TMA multicast CTA mask"))?;
        Ok(BulkRuntimeArgs { barrier: Some(barrier), cta_mask, multicast: true })
    }
);

impl<Shape, T, Scope> tensor_spec::sealed::Sealed for variant::TensorS2g<Shape, T, Scope>
where
    Shape: StaticShape,
    T: TileElement,
    Scope: StaticScope,
{
}
impl<Shape, T, Scope> tensor_spec::Variant for variant::TensorS2g<Shape, T, Scope>
where
    Shape: StaticShape,
    T: TileElement,
    Scope: StaticScope,
{
    type Args = ();
    type Source = Shared;
    type Destination = Global;
    type Output = ();
}
impl<Shape, T, Scope> TensorForm for variant::TensorS2g<Shape, T, Scope>
where
    Shape: StaticShape,
    T: TileElement,
    Scope: StaticScope,
{
    type Shape = Shape;
    type Scope = Scope;
    type Element = T;
    const DIRECTION: CopyDirection = CopyDirection::S2g;
    const CTA_GROUP: u32 = 1;
    const SOURCE_FILL: AsyncSourceFill = AsyncSourceFill::None;
    const ROUND_TO_TF32: bool = false;
    fn split((): Self::Args, _lane: usize) -> Result<BulkRuntimeArgs, EngineError> {
        Ok(BulkRuntimeArgs {
            barrier: None,
            cta_mask: 0,
            multicast: false,
        })
    }
}

macro_rules! tensor_g2s_execute_for_mode {
    ($warp:ty, $marker:ident, $base:ty) => {
        const _: () = {
            #[inline(never)]
            fn entry(
                warp: &mut $warp,
                context: ExecCtx,
                site: SiteId,
                destination: &MappedView<Shared>,
                source: &MappedView<Global>,
                extents: &[usize],
                scope: TileScopeKind,
                itemsize: usize,
                runtime: BulkRuntimeArgs,
                cta_group: u32,
                source_fill: AsyncSourceFill,
                round_to_tf32: bool,
            ) -> Result<(), EngineError> {
                execute_typed_tensor_copy(
                    warp,
                    context,
                    site,
                    destination,
                    source,
                    extents,
                    scope,
                    itemsize,
                    runtime,
                    CopyDirection::G2s,
                    cta_group,
                    source_fill,
                    round_to_tf32,
                )
            }

            impl<Shape, T, Scope, const CTA_GROUP: u32, Fill, Conversion>
                tensor_spec::sealed::Execute<$warp>
                for variant::$marker<Shape, T, Scope, CTA_GROUP, Fill, Conversion>
            where
                Shape: StaticShape,
                T: TileElement,
                Scope: StaticScope,
                Fill: TensorFill,
                Conversion: TensorConversion,
                TileCtaGroup<CTA_GROUP>: ValidTileCtaGroup,
                variant::$marker<Shape, T, Scope, CTA_GROUP, Fill, Conversion>: TensorForm
                    + TensorCopyVariant<Source = Global, Destination = Shared, Output = ()>,
            {
                #[inline(always)]
                fn execute(
                    warp: &mut $warp,
                    context: ExecCtx,
                    site: SiteId,
                    destination: &MappedView<Shared>,
                    source: &MappedView<Global>,
                    args: Self::Args,
                ) -> Result<Self::Output, EngineError> {
                    let lane = singleton_issue_lane(context, "typed cp.async.bulk.tensor")?;
                    let runtime = <Self as TensorForm>::split(args, lane)?;
                    entry(
                        warp,
                        context,
                        site,
                        destination,
                        source,
                        Shape::EXTENTS,
                        Scope::KIND,
                        <<T as super::mem::MemoryType>::Scalar as RuntimeScalar>::BYTE_LEN,
                        runtime,
                        CTA_GROUP,
                        Fill::VALUE,
                        Conversion::ROUND_TO_TF32,
                    )
                }
            }
        };
    };
}

macro_rules! tensor_s2g_execute_for_mode {
    ($warp:ty) => {
        const _: () = {
            #[inline(never)]
            fn entry(
                warp: &mut $warp,
                context: ExecCtx,
                site: SiteId,
                destination: &MappedView<Global>,
                source: &MappedView<Shared>,
                extents: &[usize],
                scope: TileScopeKind,
                itemsize: usize,
            ) -> Result<(), EngineError> {
                execute_typed_tensor_copy(
                    warp,
                    context,
                    site,
                    destination,
                    source,
                    extents,
                    scope,
                    itemsize,
                    BulkRuntimeArgs {
                        barrier: None,
                        cta_mask: 0,
                        multicast: false,
                    },
                    CopyDirection::S2g,
                    1,
                    AsyncSourceFill::None,
                    false,
                )
            }

            impl<Shape, T, Scope> tensor_spec::sealed::Execute<$warp>
                for variant::TensorS2g<Shape, T, Scope>
            where
                Shape: StaticShape,
                T: TileElement,
                Scope: StaticScope,
                variant::TensorS2g<Shape, T, Scope>: TensorCopyVariant<
                    Args = (),
                    Source = Shared,
                    Destination = Global,
                    Output = (),
                >,
            {
                #[inline(always)]
                fn execute(
                    warp: &mut $warp,
                    context: ExecCtx,
                    site: SiteId,
                    destination: &MappedView<Global>,
                    source: &MappedView<Shared>,
                    (): Self::Args,
                ) -> Result<Self::Output, EngineError> {
                    entry(
                        warp,
                        context,
                        site,
                        destination,
                        source,
                        Shape::EXTENTS,
                        Scope::KIND,
                        <<T as super::mem::MemoryType>::Scalar as RuntimeScalar>::BYTE_LEN,
                    )
                }
            }
        };
    };
}

macro_rules! tensor_execute_for_mode {
    ($warp:ty) => {
        tensor_g2s_execute_for_mode!($warp, TensorG2s, Address<Shared>);
        tensor_g2s_execute_for_mode!($warp, TensorG2sMulticast, (Address<Shared>, R<i64>));
        tensor_s2g_execute_for_mode!($warp);
    };
}

for_each_engine_mode!(test_visible, tensor_execute_for_mode);

#[inline(never)]
fn execute_typed_tensor_copy<Src, Dst, W>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    destination: &MappedView<Dst>,
    source: &MappedView<Src>,
    extents: &[usize],
    scope: TileScopeKind,
    itemsize: usize,
    runtime: BulkRuntimeArgs,
    direction: CopyDirection,
    cta_group: u32,
    source_fill: AsyncSourceFill,
    round_to_tf32: bool,
) -> Result<(), EngineError>
where
    Src: MemorySpace,
    Dst: MemorySpace,
    W: WarpHandle,
{
    let plan = mapped_copy_plan(context, source, destination, extents, scope, itemsize)?;
    match direction {
        CopyDirection::S2g => issue_group_copy(
            warp,
            context,
            site,
            AsyncGroupDomain::Bulk,
            true,
            itemsize,
            source,
            destination,
            &plan,
            source_fill,
            None,
        ),
        CopyDirection::G2s => issue_payload_copy(
            warp,
            context,
            site,
            itemsize,
            source,
            destination,
            &plan,
            PayloadControl {
                barrier: runtime.barrier.ok_or_else(|| {
                    EngineError::message("typed tensor copy has no completion barrier")
                })?,
                cta_group: i64::from(cta_group),
                cta_mask: runtime.cta_mask,
                multicast: runtime.multicast,
            },
            source_fill,
            round_to_tf32,
        ),
        CopyDirection::S2s => unreachable!("tensor copy has no S2S specialization"),
    }
}

impl<Shape, T, Scope, Op> tensor_reduce_spec::sealed::Sealed
    for variant::TensorS2gReduce<Shape, T, Scope, Op>
where
    Shape: StaticShape,
    T: TileElement,
    Scope: StaticScope,
    Op: StaticReduction<T, true>,
{
}
impl<Shape, T, Scope, Op> tensor_reduce_spec::Variant
    for variant::TensorS2gReduce<Shape, T, Scope, Op>
where
    Shape: StaticShape,
    T: TileElement,
    Scope: StaticScope,
    Op: StaticReduction<T, true>,
{
    type Args = ();
    type Source = Shared;
    type Destination = Global;
    type Output = ();
}
macro_rules! tensor_reduce_execute_for_mode {
    ($warp:ty) => {
        const _: () = {
            #[inline(never)]
            fn entry(
                warp: &mut $warp,
                context: ExecCtx,
                site: SiteId,
                destination: &MappedView<Global>,
                source: &MappedView<Shared>,
                extents: &[usize],
                scope: TileScopeKind,
                itemsize: usize,
                reduction: DeferredGlobalReduction,
            ) -> Result<(), EngineError> {
                execute_typed_tensor_reduce(
                    warp,
                    context,
                    site,
                    destination,
                    source,
                    extents,
                    scope,
                    itemsize,
                    reduction,
                )
            }

            impl<Shape, T, Scope, Op> tensor_reduce_spec::sealed::Execute<$warp>
                for variant::TensorS2gReduce<Shape, T, Scope, Op>
            where
                Shape: StaticShape,
                T: TileElement,
                Scope: StaticScope,
                Op: StaticReduction<T, true>,
                variant::TensorS2gReduce<Shape, T, Scope, Op>: TensorReduceVariant<
                    Args = (),
                    Source = Shared,
                    Destination = Global,
                    Output = (),
                >,
            {
                #[inline(always)]
                fn execute(
                    warp: &mut $warp,
                    context: ExecCtx,
                    site: SiteId,
                    destination: &MappedView<Self::Destination>,
                    source: &MappedView<Self::Source>,
                    (): Self::Args,
                ) -> Result<Self::Output, EngineError> {
                    entry(
                        warp,
                        context,
                        site,
                        destination,
                        source,
                        Shape::EXTENTS,
                        Scope::KIND,
                        <<T as super::mem::MemoryType>::Scalar as RuntimeScalar>::BYTE_LEN,
                        Op::VALUE,
                    )
                }
            }
        };
    };
}

for_each_engine_mode!(test_visible, tensor_reduce_execute_for_mode);

#[inline(never)]
fn execute_typed_tensor_reduce<W: WarpHandle>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    destination: &MappedView<Global>,
    source: &MappedView<Shared>,
    extents: &[usize],
    scope: TileScopeKind,
    itemsize: usize,
    reduction: DeferredGlobalReduction,
) -> Result<(), EngineError> {
    let plan = mapped_copy_plan(context, source, destination, extents, scope, itemsize)?;
    issue_group_copy(
        warp,
        context,
        site,
        AsyncGroupDomain::Bulk,
        true,
        itemsize,
        source,
        destination,
        &plan,
        AsyncSourceFill::None,
        Some(reduction),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct MappedElement {
    execution_lane: usize,
    target_cta: Option<usize>,
    location: ElementLocation,
}

/// TCGEN register ownership is part of the frontend layout (for example the
/// `.16x*b` atom layouts are not a row-major round-robin partition).  Probe
/// every active lane in this warp and let an out-of-bounds map entry mean
/// "this lane does not own this logical element".  Both operands are checked
/// for identical execution-lane ownership by `issue_tcgen_transfer` below.
fn mapped_tcgen_elements<S>(
    context: ExecCtx,
    view: &MappedView<S>,
    extents: &[usize],
    label: &str,
) -> Result<Vec<MappedElement>, EngineError>
where
    S: MemorySpace,
{
    let mut result = Vec::new();
    for linear in 0..element_count_for(extents)? {
        let coordinates = coordinates_for(extents, linear)?;
        let owners = view
            .owners(LogicalCoord::new(&coordinates))
            .map_err(|error| EngineError::message(format!("{label}: {error}")))?
            & context.active_mask();
        for lane in owners {
            let reference = view
                .map(LogicalCoord::new(&coordinates), LaneId::from_index(lane))
                .map_err(|error| EngineError::message(format!("{label}: {error}")))?;
            if !reference.is_in_bounds() {
                continue;
            }
            result.push(MappedElement {
                execution_lane: lane,
                target_cta: reference.target_rank().map(|rank| rank as usize),
                location: reference.location(),
            });
        }
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn execute_mapped_tcgen_transfer<SourceT, DestinationT, Source, Destination>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    destination: &MappedView<Destination>,
    source: &MappedView<Source>,
    extents: &[usize],
    pipeline_operation: crate::runtime::TcgenPipelineOperation,
    cta_group: u32,
    access: crate::TmemAccessMode,
    source_is_tmem: bool,
    source_label: &str,
    destination_label: &str,
) -> Result<(), EngineError>
where
    SourceT: TcgenTileElement,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
    Source: MemorySpace,
    Destination: MemorySpace,
{
    let source_elements = mapped_tcgen_elements(context, source, extents, source_label)?;
    let destination_elements =
        mapped_tcgen_elements(context, destination, extents, destination_label)?;
    issue_tcgen_transfer::<SourceT, DestinationT>(
        warp,
        context,
        site,
        cta_group,
        pipeline_operation,
        access,
        source.allocation().inner().clone(),
        source
            .allocation()
            .logical_buffer()
            .unwrap_or("tile_source"),
        destination.allocation().inner().clone(),
        destination
            .allocation()
            .logical_buffer()
            .unwrap_or("tile_destination"),
        source_elements,
        destination_elements,
        source_is_tmem,
    )
}

trait TileTmemAccess {
    const VALUE: crate::TmemAccessMode;
}

impl TileTmemAccess for super::tcgen05::variant::StaticTmem {
    const VALUE: crate::TmemAccessMode = crate::TmemAccessMode::Static;
}

impl TileTmemAccess for super::tcgen05::variant::DynamicTmem {
    const VALUE: crate::TmemAccessMode = crate::TmemAccessMode::Dynamic;
}

trait TileRawCpShape {
    const PIPELINE_OPERATION: crate::runtime::TcgenPipelineOperation;
}

macro_rules! tile_raw_cp_shape {
    ($marker:ty, $operation:expr) => {
        impl TileRawCpShape for $marker {
            const PIPELINE_OPERATION: crate::runtime::TcgenPipelineOperation = $operation;
        }
    };
}

tile_raw_cp_shape!(
    super::tcgen05::variant::Cp32x128bWarpx4,
    crate::runtime::TcgenPipelineOperation::Copy
);
tile_raw_cp_shape!(
    super::tcgen05::variant::Cp64x128bWarpx2_02_13,
    crate::runtime::TcgenPipelineOperation::Copy
);
tile_raw_cp_shape!(
    super::tcgen05::variant::Cp128x128b,
    crate::runtime::TcgenPipelineOperation::Copy
);
tile_raw_cp_shape!(
    super::tcgen05::variant::Cp128x256b,
    crate::runtime::TcgenPipelineOperation::Copy
);
tile_raw_cp_shape!(
    super::tcgen05::variant::Cp4x256b,
    crate::runtime::TcgenPipelineOperation::Copy4x256b
);
tile_raw_cp_shape!(
    super::tcgen05::variant::Cp64x128bWarpx2_01_23,
    crate::runtime::TcgenPipelineOperation::Copy
);

trait RawCpForm {
    const CTA_GROUP: u32;
    const ACCESS: crate::TmemAccessMode;
    const PIPELINE_OPERATION: crate::runtime::TcgenPipelineOperation;
}

impl<PhysicalShape, Decompress, const CTA_GROUP: usize, Access> RawCpForm
    for super::tcgen05::variant::Cp<PhysicalShape, Decompress, CTA_GROUP, Access>
where
    Self: super::tcgen05::CpVariant,
    PhysicalShape: TileRawCpShape,
    Access: TileTmemAccess,
{
    const CTA_GROUP: u32 = CTA_GROUP as u32;
    const ACCESS: crate::TmemAccessMode = Access::VALUE;
    const PIPELINE_OPERATION: crate::runtime::TcgenPipelineOperation =
        PhysicalShape::PIPELINE_OPERATION;
}

trait RawLdForm {
    const ACCESS: crate::TmemAccessMode;
}

impl<PhysicalShape, Num, const PACKED: bool, Access> RawLdForm
    for super::tcgen05::variant::Ld<PhysicalShape, Num, PACKED, Access>
where
    Self: super::tcgen05::LdVariant,
    Access: TileTmemAccess,
{
    const ACCESS: crate::TmemAccessMode = Access::VALUE;
}

trait RawCanonical32x32bLdForm: RawLdForm {
    const NUM: usize;
}

impl<const N: usize, Access> RawCanonical32x32bLdForm
    for super::tcgen05::variant::Ld<
        super::tcgen05::variant::Shape32x32b,
        super::tcgen05::variant::Num<N>,
        false,
        Access,
    >
where
    Self: RawLdForm,
{
    const NUM: usize = N;
}

trait RawStForm {
    const ACCESS: crate::TmemAccessMode;
}

impl<PhysicalShape, Num, const UNPACKED: bool, Access> RawStForm
    for super::tcgen05::variant::St<PhysicalShape, Num, UNPACKED, Access>
where
    Self: super::tcgen05::StVariant,
    Access: TileTmemAccess,
{
    const ACCESS: crate::TmemAccessMode = Access::VALUE;
}

trait RawCanonical32x32bStForm: RawStForm {
    const NUM: usize;
}

impl<const N: usize, Access> RawCanonical32x32bStForm
    for super::tcgen05::variant::St<
        super::tcgen05::variant::Shape32x32b,
        super::tcgen05::variant::Num<N>,
        false,
        Access,
    >
where
    Self: RawStForm,
{
    const NUM: usize = N;
}

fn byte_offset(
    location: MappedElement,
    itemsize: usize,
    label: &str,
) -> Result<usize, EngineError> {
    let ElementLocation::ByteOffset(offset) = location.location else {
        return Err(EngineError::message(format!(
            "{label} expected a byte-addressed element"
        )));
    };
    let offset = usize::try_from(offset)
        .map_err(|_| EngineError::message(format!("{label} has a negative/oversized offset")))?;
    if offset % itemsize != 0 {
        return Err(EngineError::message(format!(
            "{label} byte offset {offset} is not aligned to itemsize {itemsize}"
        )));
    }
    Ok(offset)
}

fn tmem_coordinates(location: MappedElement, label: &str) -> Result<(i64, i64, i64), EngineError> {
    let ElementLocation::Tmem {
        mapped_lane,
        tcol_element,
        allocated_addr,
        bit_offset: 0,
    } = location.location
    else {
        return Err(EngineError::message(format!(
            "{label} expected a TMEM coordinate"
        )));
    };
    Ok((mapped_lane, tcol_element, allocated_addr))
}

fn regular_footprints(
    elements: &[MappedElement],
    itemsize: usize,
    label: &str,
) -> Result<Vec<(usize, Option<usize>, usize, usize)>, EngineError> {
    elements
        .iter()
        .copied()
        .map(|element| {
            Ok((
                element.execution_lane,
                element.target_cta,
                byte_offset(element, itemsize, label)?,
                itemsize,
            ))
        })
        .collect()
}

fn tmem_footprints(
    elements: &[MappedElement],
    provenance_lane: usize,
    itemsize: usize,
    label: &str,
) -> Result<Vec<(usize, usize, Option<usize>, i64, i64, i64, usize)>, EngineError> {
    elements
        .iter()
        .copied()
        .map(|element| {
            let (mapped_lane, tcol_element, allocated_addr) = tmem_coordinates(element, label)?;
            Ok((
                provenance_lane,
                element.execution_lane,
                element.target_cta,
                mapped_lane,
                tcol_element,
                allocated_addr,
                itemsize,
            ))
        })
        .collect()
}

fn read_regular<T: RuntimeScalar>(
    physical: &crate::PhysicalMemory,
    context: &crate::WarpContext,
    buffer: &crate::runtime::RuntimeBuffer,
    element: MappedElement,
) -> Result<T, EngineError> {
    let byte_offset = byte_offset(element, T::BYTE_LEN, "typed TCGEN regular source")?;
    let index = i64::try_from(byte_offset / T::BYTE_LEN)
        .map_err(|_| EngineError::message("typed TCGEN source index exceeds i64"))?;
    let result = match element.target_cta {
        Some(target) => crate::runtime::read_shared_scalar_at_cta(
            physical,
            context,
            buffer,
            target,
            index,
            element.execution_lane,
        ),
        None => crate::runtime::load_scalar_lane(
            physical,
            context,
            buffer,
            index,
            element.execution_lane,
        ),
    };
    result.map_err(Into::into)
}

fn write_regular<T: RuntimeScalar>(
    physical: &crate::PhysicalMemory,
    context: &crate::WarpContext,
    buffer: &crate::runtime::RuntimeBuffer,
    element: MappedElement,
    value: T,
) -> Result<(), EngineError> {
    if element.target_cta.is_some() {
        return Err(EngineError::message(
            "typed TCGEN register/local destination cannot select a remote CTA",
        ));
    }
    let byte_offset = byte_offset(element, T::BYTE_LEN, "typed TCGEN regular destination")?;
    let index = i64::try_from(byte_offset / T::BYTE_LEN)
        .map_err(|_| EngineError::message("typed TCGEN destination index exceeds i64"))?;
    let indices = crate::WarpValue::splat(index);
    let values = crate::WarpValue::splat(value);
    crate::runtime::store_scalar_warp(
        physical,
        context,
        buffer,
        &indices,
        &values,
        WarpMask::from_bits(1_u32 << element.execution_lane),
    )
    .map_err(Into::into)
}

fn read_tmem<T: RuntimeScalar>(
    physical: &crate::PhysicalMemory,
    context: &crate::WarpContext,
    lifecycle: &crate::TcgenLifecycleHub,
    access: crate::TmemAccessMode,
    buffer: &crate::runtime::RuntimeBuffer,
    element: MappedElement,
) -> Result<T, EngineError> {
    let (mapped_lane, tcol_element, allocated_addr) =
        tmem_coordinates(element, "typed TCGEN source")?;
    crate::runtime::load_tmem_scalar_at_cta(
        physical,
        context,
        lifecycle,
        access,
        buffer,
        element
            .target_cta
            .unwrap_or_else(|| context.cta_id_in_cluster()),
        mapped_lane,
        tcol_element,
        allocated_addr,
        element.execution_lane,
    )
    .map_err(Into::into)
}

fn write_tmem<T: RuntimeScalar>(
    physical: &crate::PhysicalMemory,
    context: &crate::WarpContext,
    lifecycle: &crate::TcgenLifecycleHub,
    access: crate::TmemAccessMode,
    buffer: &crate::runtime::RuntimeBuffer,
    element: MappedElement,
    value: T,
) -> Result<(), EngineError> {
    let (mapped_lane, tcol_element, allocated_addr) =
        tmem_coordinates(element, "typed TCGEN destination")?;
    crate::runtime::store_tmem_scalar_at_cta(
        physical,
        context,
        lifecycle,
        access,
        buffer,
        element
            .target_cta
            .unwrap_or_else(|| context.cta_id_in_cluster()),
        mapped_lane,
        tcol_element,
        allocated_addr,
        value,
        element.execution_lane,
    )
    .map_err(Into::into)
}

#[derive(Clone, Copy)]
struct FastTmemF32M64Load {
    base_lane: i64,
    base_tcol: i64,
    allocated_addr: i64,
}

/// Recognize the physical mapping of one complete m64 `.16x256b.x8`
/// `tcgen05.ld` warp slice.  This is deliberately an engine-private numeric
/// optimization: the public contract remains the two frontend-owned maps,
/// and an arbitrary map continues through the scalar reference path.
fn fast_tmem_f32_m64_load(
    context: &crate::WarpContext,
    source: &[MappedElement],
    destination: &[MappedElement],
) -> Option<FastTmemF32M64Load> {
    const ELEMENTS_PER_LANE: usize = 32;
    const ELEMENT_COUNT: usize = crate::WARP_SIZE * ELEMENTS_PER_LANE;

    if context.active_mask() != WarpMask::FULL
        || source.len() != ELEMENT_COUNT
        || destination.len() != ELEMENT_COUNT
    {
        return None;
    }

    let warp_lane_offset = (context.warp_id_in_cta() % 4).checked_mul(crate::WARP_SIZE)?;
    let mut seen = [false; ELEMENT_COUNT];
    let mut base: Option<FastTmemF32M64Load> = None;
    for (&source, &destination) in source.iter().zip(destination) {
        if source.execution_lane != destination.execution_lane
            || source.target_cta.is_some()
            || destination.target_cta.is_some()
        {
            return None;
        }
        let execution_lane = source.execution_lane;
        let ElementLocation::ByteOffset(destination_offset) = destination.location else {
            return None;
        };
        let destination_offset = usize::try_from(destination_offset).ok()?;
        if destination_offset % std::mem::size_of::<f32>() != 0 {
            return None;
        }
        let slot = destination_offset / std::mem::size_of::<f32>();
        if execution_lane >= crate::WARP_SIZE || slot >= ELEMENTS_PER_LANE {
            return None;
        }
        let seen_slot = execution_lane * ELEMENTS_PER_LANE + slot;
        if std::mem::replace(&mut seen[seen_slot], true) {
            return None;
        }

        let (mapped_lane, tcol, allocated_addr) = tmem_coordinates(source, "fast m64 load").ok()?;
        let lane_group = execution_lane / 4;
        let lane_in_group = execution_lane % 4;
        let block = slot / 4;
        let lane_bank = (slot % 4) / 2;
        let pair = slot % 2;
        let expected_lane_offset = warp_lane_offset + lane_bank * 8 + lane_group;
        let expected_tcol_offset = block * 8 + lane_in_group * 2 + pair;
        let candidate = FastTmemF32M64Load {
            base_lane: mapped_lane.checked_sub(i64::try_from(expected_lane_offset).ok()?)?,
            base_tcol: tcol.checked_sub(i64::try_from(expected_tcol_offset).ok()?)?,
            allocated_addr,
        };
        match base {
            Some(base)
                if base.base_lane != candidate.base_lane
                    || base.base_tcol != candidate.base_tcol
                    || base.allocated_addr != candidate.allocated_addr =>
            {
                return None;
            }
            None => base = Some(candidate),
            _ => {}
        }
    }
    seen.into_iter().all(|value| value).then_some(base?)
}

#[allow(clippy::too_many_arguments)]
fn issue_tcgen_transfer<SourceT, DestinationT>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    cta_group: u32,
    pipeline_operation: crate::runtime::TcgenPipelineOperation,
    access: crate::TmemAccessMode,
    source_buffer: crate::runtime::RuntimeBuffer,
    source_logical_buffer: &str,
    destination_buffer: crate::runtime::RuntimeBuffer,
    destination_logical_buffer: &str,
    source_elements: Vec<MappedElement>,
    destination_elements: Vec<MappedElement>,
    source_is_tmem: bool,
) -> Result<(), EngineError>
where
    SourceT: TcgenTileElement,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
{
    if source_elements.len() != destination_elements.len() {
        return Err(EngineError::message(
            "typed TCGEN source/destination maps have different element counts",
        ));
    }
    if source_elements
        .iter()
        .zip(&destination_elements)
        .any(|(source, destination)| source.execution_lane != destination.execution_lane)
    {
        return Err(EngineError::message(
            "typed TCGEN source/destination maps disagree on execution lane",
        ));
    }

    let issuer_lane = context
        .active_mask()
        .first_active()
        .ok_or_else(|| EngineError::message("typed TCGEN transfer has no issuing lane"))?;
    let kind = pipeline_operation.work_kind();
    if kind == crate::runtime::TcgenWorkKind::Commit && context.active_mask().len() != 1 {
        return Err(EngineError::message(
            "typed tcgen05.cp requires exactly one issuing lane",
        ));
    }
    if matches!(
        kind,
        crate::runtime::TcgenWorkKind::Load | crate::runtime::TcgenWorkKind::Store
    ) {
        crate::runtime::require_full_warp_sync(
            context.active_mask().into_inner(),
            "typed tcgen05.ld/st",
        )?;
    }
    let issue_mask = LaneMask::single(issuer_lane)?;
    let issue_context = ExecCtx::from_inner(
        context
            .into_inner()
            .with_active_mask(issue_mask.into_inner()),
    );
    let operation = begin(warp, issue_context, site, OperationKind::TcgenWork, false)?;
    let physical = engine(warp).kernel().physical().clone();
    let lifecycle = engine(warp).kernel().services().tcgen();
    let inner = context.into_inner();
    let source_regular = (kind == crate::runtime::TcgenWorkKind::Commit)
        .then(|| {
            regular_footprints(
                &source_elements,
                SourceT::Physical::BYTE_LEN,
                "typed TCGEN source",
            )
        })
        .transpose()?;
    let source_tmem = source_is_tmem
        .then(|| {
            tmem_footprints(
                &source_elements,
                issuer_lane,
                SourceT::Physical::BYTE_LEN,
                "typed TCGEN source",
            )
        })
        .transpose()?;
    let destination_tmem = (!source_is_tmem)
        .then(|| {
            tmem_footprints(
                &destination_elements,
                issuer_lane,
                SourceT::Physical::BYTE_LEN,
                "typed TCGEN destination",
            )
        })
        .transpose()?;

    let fast_f32_m64_load = (source_is_tmem
        && std::any::TypeId::of::<SourceT::Physical>() == std::any::TypeId::of::<f32>())
    .then(|| fast_tmem_f32_m64_load(&inner, &source_elements, &destination_elements))
    .flatten();
    engine(warp).tcgen_instruction_issue(
        operation.as_ref(),
        cta_group,
        pipeline_operation,
        None,
        |record_regular, record_tmem| {
            if let Some(accesses) = source_regular.as_deref() {
                record_regular(
                    false,
                    OperationKind::Load,
                    &source_buffer,
                    Some(source_logical_buffer),
                    accesses,
                )?;
            }
            if let Some(accesses) = source_tmem.as_deref() {
                record_tmem(
                    OperationKind::Load,
                    &source_buffer,
                    source_logical_buffer,
                    access,
                    accesses,
                )?;
            }
            if let Some(accesses) = destination_tmem.as_deref() {
                record_tmem(
                    OperationKind::Store,
                    &destination_buffer,
                    destination_logical_buffer,
                    access,
                    accesses,
                )?;
            }
            Ok(())
        },
        || {
            if let Some(fast) = fast_f32_m64_load {
                return crate::runtime::copy_tmem_f32_m64_to_local_warp(
                    &physical,
                    &inner,
                    &lifecycle,
                    access,
                    &source_buffer,
                    &destination_buffer,
                    &crate::WarpValue::splat(fast.base_lane),
                    &crate::WarpValue::splat(fast.base_tcol),
                    &crate::WarpValue::splat(fast.allocated_addr),
                    8,
                    inner.active_mask(),
                );
            }
            for (&source, &destination) in source_elements.iter().zip(&destination_elements) {
                let value = if source_is_tmem {
                    read_tmem::<SourceT::Physical>(
                        &physical,
                        &inner,
                        &lifecycle,
                        access,
                        &source_buffer,
                        source,
                    )?
                } else {
                    read_regular::<SourceT::Physical>(&physical, &inner, &source_buffer, source)?
                };
                if source_is_tmem {
                    write_regular::<SourceT::Physical>(
                        &physical,
                        &inner,
                        &destination_buffer,
                        destination,
                        value,
                    )?;
                } else {
                    write_tmem::<SourceT::Physical>(
                        &physical,
                        &inner,
                        &lifecycle,
                        access,
                        &destination_buffer,
                        destination,
                        value,
                    )?;
                }
            }
            Ok(())
        },
    )?;
    finish(warp, &operation)
}

/// Execute the frontend-proven canonical m64 f32 mapping without expanding
/// its fixed PTX register/TMEM permutation into thousands of map entries.
#[allow(clippy::too_many_arguments)]
fn issue_canonical_f32_m64_ld(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    destination: &MappedView<Register>,
    source: &MappedView<super::Tmem>,
    access: crate::TmemAccessMode,
    base_lane: R<i64>,
    base_tcol: R<i64>,
    allocated_addr: R<i64>,
    num: usize,
) -> Result<(), EngineError> {
    crate::runtime::require_full_warp_sync(
        context.active_mask().into_inner(),
        "canonical f32 m64 tcgen05.ld",
    )?;
    let issuer_lane = context
        .active_mask()
        .first_active()
        .ok_or_else(|| EngineError::message("canonical tcgen05.ld has no issuing lane"))?;
    let issue_mask = LaneMask::single(issuer_lane)?;
    let issue_context = ExecCtx::from_inner(
        context
            .into_inner()
            .with_active_mask(issue_mask.into_inner()),
    );
    let operation = begin(warp, issue_context, site, OperationKind::TcgenWork, false)?;

    let physical = engine(warp).kernel().physical().clone();
    let lifecycle = engine(warp).kernel().services().tcgen();
    let inner = context.into_inner();
    let source_buffer = source.allocation().inner().clone();
    let destination_buffer = destination.allocation().inner().clone();
    let source_logical_buffer = source
        .allocation()
        .logical_buffer()
        .unwrap_or("tile_source");
    let warp_lane_offset = (inner.warp_id_in_cta() % 4)
        .checked_mul(crate::WARP_SIZE)
        .ok_or_else(|| EngineError::message("canonical tcgen05.ld warp offset overflow"))?;
    let mapped_lane =
        base_lane[issuer_lane]
            .checked_add(i64::try_from(warp_lane_offset).map_err(|_| {
                EngineError::message("canonical tcgen05.ld warp offset exceeds i64")
            })?)
            .ok_or_else(|| EngineError::message("canonical tcgen05.ld lane overflow"))?;
    if !matches!(num, 1 | 8) {
        return Err(EngineError::message(format!(
            "canonical m64 tcgen05.ld requires .x1 or .x8, got .x{num}"
        )));
    }
    let row_bytes = num
        .checked_mul(32)
        .ok_or_else(|| EngineError::message("canonical m64 TCGEN row width overflow"))?;
    let mut footprint = Vec::with_capacity(16);
    for row in 0_i64..16_i64 {
        footprint.push((
            issuer_lane,
            issuer_lane,
            None,
            mapped_lane + row,
            base_tcol[issuer_lane],
            allocated_addr[issuer_lane],
            row_bytes,
        ));
    }

    engine(warp).tcgen_instruction_issue(
        operation.as_ref(),
        1,
        crate::runtime::TcgenPipelineOperation::Load,
        None,
        |_record_regular, record_tmem| {
            record_tmem(
                OperationKind::Load,
                &source_buffer,
                source_logical_buffer,
                access,
                &footprint,
            )
        },
        || {
            crate::runtime::copy_tmem_f32_m64_to_local_warp(
                &physical,
                &inner,
                &lifecycle,
                access,
                &source_buffer,
                &destination_buffer,
                base_lane.inner(),
                base_tcol.inner(),
                allocated_addr.inner(),
                num,
                inner.active_mask(),
            )
        },
    )?;
    finish(warp, &operation)
}

/// Which way a canonical `.32x32b` transfer runs.
///
/// The load and store forms are the same instruction read in two directions:
/// they take the same steps in the same order and differ only in which view is
/// the TMEM side, which diagnostics they carry, and which warp copy they end
/// in. This is the row; [`issue_canonical_32x32b`] is the driver.
#[derive(Clone, Copy)]
enum Canonical32x32bDirection {
    Load,
    Store,
}

/// Everything a `.32x32b` direction contributes except which warp copy runs.
struct Canonical32x32bRow {
    /// Label for the full-warp-sync requirement.
    sync_label: &'static str,
    /// Stem every operand diagnostic is built from.
    mnemonic: &'static str,
    pipeline_operation: crate::runtime::TcgenPipelineOperation,
    operation_kind: OperationKind,
    /// Fallback name for the TMEM-side logical buffer.
    tmem_logical_default: &'static str,
}

impl Canonical32x32bDirection {
    const fn row(self) -> Canonical32x32bRow {
        match self {
            Self::Load => Canonical32x32bRow {
                sync_label: "canonical tcgen05.ld.32x32b",
                mnemonic: "canonical tcgen05.ld",
                pipeline_operation: crate::runtime::TcgenPipelineOperation::Load,
                operation_kind: OperationKind::Load,
                tmem_logical_default: "tile_source",
            },
            Self::Store => Canonical32x32bRow {
                sync_label: "canonical tcgen05.st.32x32b",
                mnemonic: "canonical tcgen05.st",
                pipeline_operation: crate::runtime::TcgenPipelineOperation::Store,
                operation_kind: OperationKind::Store,
                tmem_logical_default: "tile_destination",
            },
        }
    }
}

/// Execute a frontend-proven `.32x32b` TMEM/register mapping without
/// evaluating a layout map for every transferred scalar.
///
/// `tmem` and `register` name the two views by *space* rather than by role;
/// `direction` says which is the source.
#[allow(clippy::too_many_arguments)]
fn issue_canonical_32x32b<SourceT>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    tmem: &MappedView<super::Tmem>,
    register: &MappedView<Register>,
    direction: Canonical32x32bDirection,
    access: crate::TmemAccessMode,
    elements_per_lane: usize,
    base_lane: R<i64>,
    base_tcol: R<i64>,
    allocated_addr: R<i64>,
) -> Result<(), EngineError>
where
    SourceT: TcgenTileElement,
{
    let row = direction.row();
    let mnemonic = row.mnemonic;
    crate::runtime::require_full_warp_sync(context.active_mask().into_inner(), row.sync_label)?;
    let issuer_lane = context
        .active_mask()
        .first_active()
        .ok_or_else(|| EngineError::message(format!("{mnemonic} has no issuing lane")))?;
    for lane in context.active_mask() {
        if base_lane[lane] != base_lane[issuer_lane]
            || base_tcol[lane] != base_tcol[issuer_lane]
            || allocated_addr[lane] != allocated_addr[issuer_lane]
        {
            return Err(EngineError::message(format!(
                "{mnemonic} TMEM origin must be uniform"
            )));
        }
    }
    let row_bytes = elements_per_lane
        .checked_mul(SourceT::Physical::BYTE_LEN)
        .ok_or_else(|| EngineError::message(format!("{mnemonic} row size overflows usize")))?;
    let issue_mask = LaneMask::single(issuer_lane)?;
    let issue_context = ExecCtx::from_inner(
        context
            .into_inner()
            .with_active_mask(issue_mask.into_inner()),
    );
    let operation = begin(warp, issue_context, site, OperationKind::TcgenWork, false)?;

    let physical = engine(warp).kernel().physical().clone();
    let lifecycle = engine(warp).kernel().services().tcgen();
    let inner = context.into_inner();
    let tmem_buffer = tmem.allocation().inner().clone();
    let register_buffer = register.allocation().inner().clone();
    let tmem_logical_buffer = tmem
        .allocation()
        .logical_buffer()
        .unwrap_or(row.tmem_logical_default);
    let warp_lane_offset = (inner.warp_id_in_cta() % 4)
        .checked_mul(crate::WARP_SIZE)
        .ok_or_else(|| EngineError::message(format!("{mnemonic} warp offset overflow")))?;
    let mut tmem_footprint = Vec::with_capacity(crate::WARP_SIZE);
    for execution_lane in context.active_mask() {
        let lane_offset = warp_lane_offset
            .checked_add(execution_lane)
            .ok_or_else(|| EngineError::message(format!("{mnemonic} lane offset overflow")))?;
        let mapped_lane =
            base_lane[issuer_lane]
                .checked_add(i64::try_from(lane_offset).map_err(|_| {
                    EngineError::message(format!("{mnemonic} lane offset exceeds i64"))
                })?)
                .ok_or_else(|| EngineError::message(format!("{mnemonic} TLane overflow")))?;
        tmem_footprint.push((
            issuer_lane,
            execution_lane,
            None,
            mapped_lane,
            base_tcol[issuer_lane],
            allocated_addr[issuer_lane],
            row_bytes,
        ));
    }

    // The copy helpers take `(source, destination)` in transfer order, which is
    // the one place the two directions disagree about which buffer is which.
    let (source_buffer, destination_buffer) = match direction {
        Canonical32x32bDirection::Load => (&tmem_buffer, &register_buffer),
        Canonical32x32bDirection::Store => (&register_buffer, &tmem_buffer),
    };
    engine(warp).tcgen_instruction_issue(
        operation.as_ref(),
        1,
        row.pipeline_operation,
        None,
        |_record_regular, record_tmem| {
            record_tmem(
                row.operation_kind,
                &tmem_buffer,
                tmem_logical_buffer,
                access,
                &tmem_footprint,
            )
        },
        || match direction {
            Canonical32x32bDirection::Load => {
                crate::runtime::copy_tmem_32x32b_to_register_warp::<SourceT::Physical>(
                    &physical,
                    &inner,
                    &lifecycle,
                    access,
                    source_buffer,
                    destination_buffer,
                    base_lane.inner(),
                    base_tcol.inner(),
                    allocated_addr.inner(),
                    elements_per_lane,
                    inner.active_mask(),
                )
            }
            Canonical32x32bDirection::Store => {
                crate::runtime::copy_register_32x32b_to_tmem_warp::<SourceT::Physical>(
                    &physical,
                    &inner,
                    &lifecycle,
                    access,
                    source_buffer,
                    destination_buffer,
                    base_lane.inner(),
                    base_tcol.inner(),
                    allocated_addr.inner(),
                    elements_per_lane,
                    inner.active_mask(),
                )
            }
        },
    )?;
    finish(warp, &operation)
}

impl<Shape, SourceT, DestinationT, Raw> tcgen_cp_spec::sealed::Sealed
    for variant::Tcgen05Cp<Shape, SourceT, DestinationT, Raw>
where
    Shape: StaticShape,
    SourceT: TcgenTileElement + TcgenTransferEntry<DestinationT>,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
    Raw: RawCpForm,
{
}
impl<Shape, SourceT, DestinationT, Raw> tcgen_cp_spec::Variant
    for variant::Tcgen05Cp<Shape, SourceT, DestinationT, Raw>
where
    Shape: StaticShape,
    SourceT: TcgenTileElement + TcgenTransferEntry<DestinationT>,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
    Raw: RawCpForm,
{
    type Source = Shared;
    type Destination = super::Tmem;
    type Output = ();
}

impl<Shape, SourceT, DestinationT, Raw> tcgen_cp_spec::sealed::Execute<super::Engine>
    for variant::Tcgen05Cp<Shape, SourceT, DestinationT, Raw>
where
    Shape: StaticShape,
    SourceT: TcgenTileElement + TcgenTransferEntry<DestinationT>,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
    Raw: RawCpForm,
{
    #[inline(always)]
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<super::Tmem>,
        source: &MappedView<Shared>,
    ) -> Result<Self::Output, EngineError> {
        <SourceT as TcgenTransferEntry<DestinationT>>::mapped_cp(
            warp,
            context,
            site,
            destination,
            source,
            Shape::EXTENTS,
            Raw::CTA_GROUP,
            Raw::ACCESS,
            Raw::PIPELINE_OPERATION,
        )
    }
}
/// Closed specialization contract for a `tcgen05.ld` operand mapping.
#[allow(private_bounds)]
pub trait Tcgen05LdMapping: sealed::TcgenLdMapping {
    type Args;
}

impl sealed::TcgenLdMapping for variant::Mapped {}
impl Tcgen05LdMapping for variant::Mapped {
    type Args = ();
}

impl sealed::TcgenLdMapping for variant::CanonicalF32M64 {}
impl Tcgen05LdMapping for variant::CanonicalF32M64 {
    /// Runtime TMEM `(base_lane, base_tcol, allocated_addr)` operands.
    type Args = (R<i64>, R<i64>, R<i64>);
}

impl sealed::TcgenLdMapping for variant::Canonical32x32b {}
impl Tcgen05LdMapping for variant::Canonical32x32b {
    /// Runtime TMEM `(base_lane, base_tcol, allocated_addr)` operands.
    type Args = (R<i64>, R<i64>, R<i64>);
}

impl<Shape, SourceT, DestinationT, Raw, Mapping> tcgen_ld_spec::sealed::Sealed
    for variant::Tcgen05Ld<Shape, SourceT, DestinationT, Raw, Mapping>
where
    Shape: StaticShape,
    SourceT: TcgenTileElement + TcgenTransferEntry<DestinationT>,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
    Raw: RawLdForm,
    Mapping: TcgenLdMappingForm<Shape, SourceT, DestinationT, Raw>,
{
}
impl<Shape, SourceT, DestinationT, Raw, Mapping> tcgen_ld_spec::Variant
    for variant::Tcgen05Ld<Shape, SourceT, DestinationT, Raw, Mapping>
where
    Shape: StaticShape,
    SourceT: TcgenTileElement + TcgenTransferEntry<DestinationT>,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
    Raw: RawLdForm,
    Mapping: TcgenLdMappingForm<Shape, SourceT, DestinationT, Raw>,
{
    type Args = Mapping::Args;
    type Source = super::Tmem;
    type Destination = Register;
    type Output = ();
}

impl<Shape, SourceT, DestinationT, Raw, Mapping> tcgen_ld_spec::sealed::Execute<super::Engine>
    for variant::Tcgen05Ld<Shape, SourceT, DestinationT, Raw, Mapping>
where
    Shape: StaticShape,
    SourceT: TcgenTileElement + TcgenTransferEntry<DestinationT>,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
    Raw: RawLdForm,
    Mapping: TcgenLdMappingForm<Shape, SourceT, DestinationT, Raw>,
{
    #[inline(always)]
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<Register>,
        source: &MappedView<super::Tmem>,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        Mapping::issue(warp, context, site, destination, source, args)
    }
}
trait TcgenLdMappingForm<Shape, SourceT, DestinationT, Raw>: Tcgen05LdMapping
where
    Shape: StaticShape,
    SourceT: TcgenTileElement + TcgenTransferEntry<DestinationT>,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
    Raw: RawLdForm,
{
    fn issue(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<Register>,
        source: &MappedView<super::Tmem>,
        args: Self::Args,
    ) -> Result<(), EngineError>;
}

impl<Shape, SourceT, DestinationT, Raw> TcgenLdMappingForm<Shape, SourceT, DestinationT, Raw>
    for variant::Mapped
where
    Shape: StaticShape,
    SourceT: TcgenTileElement + TcgenTransferEntry<DestinationT>,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
    Raw: RawLdForm,
{
    fn issue(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<Register>,
        source: &MappedView<super::Tmem>,
        (): Self::Args,
    ) -> Result<(), EngineError> {
        <SourceT as TcgenTransferEntry<DestinationT>>::mapped_ld(
            warp,
            context,
            site,
            destination,
            source,
            Shape::EXTENTS,
            Raw::ACCESS,
        )
    }
}

macro_rules! canonical_f32_m64_ld_form {
    ($elements:literal, $num:literal) => {
        impl<Access>
            TcgenLdMappingForm<
                variant::Shape1<$elements>,
                super::reg::variant::F32,
                super::reg::variant::F32,
                super::tcgen05::variant::Ld<
                    super::tcgen05::variant::Shape16x256b,
                    super::tcgen05::variant::Num<$num>,
                    false,
                    Access,
                >,
            > for variant::CanonicalF32M64
        where
            Access: TileTmemAccess,
            super::tcgen05::variant::Ld<
                super::tcgen05::variant::Shape16x256b,
                super::tcgen05::variant::Num<$num>,
                false,
                Access,
            >: RawLdForm,
        {
            fn issue(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                destination: &MappedView<Register>,
                source: &MappedView<super::Tmem>,
                (base_lane, base_tcol, allocated_addr): Self::Args,
            ) -> Result<(), EngineError> {
                issue_canonical_f32_m64_ld(
                    warp,
                    context,
                    site,
                    destination,
                    source,
                    Access::VALUE,
                    base_lane,
                    base_tcol,
                    allocated_addr,
                    $num,
                )
            }
        }
    };
}

canonical_f32_m64_ld_form!(512, 1);
canonical_f32_m64_ld_form!(4096, 8);

impl<Shape, SourceT, DestinationT, Raw> TcgenLdMappingForm<Shape, SourceT, DestinationT, Raw>
    for variant::Canonical32x32b
where
    Shape: StaticShape,
    SourceT: TcgenTileElement + TcgenTransferEntry<DestinationT>,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
    Raw: RawCanonical32x32bLdForm,
{
    fn issue(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<Register>,
        source: &MappedView<super::Tmem>,
        (base_lane, base_tcol, allocated_addr): Self::Args,
    ) -> Result<(), EngineError> {
        let row_bytes = Raw::NUM
            .checked_mul(4)
            .ok_or_else(|| EngineError::message("canonical tcgen05.ld register count overflows"))?;
        if row_bytes % SourceT::Physical::BYTE_LEN != 0 {
            return Err(EngineError::message(
                "canonical tcgen05.ld register width is not divisible by element width",
            ));
        }
        let elements_per_lane = row_bytes / SourceT::Physical::BYTE_LEN;
        let expected_elements = 4_usize
            .checked_mul(crate::WARP_SIZE)
            .and_then(|threads| threads.checked_mul(elements_per_lane))
            .ok_or_else(|| EngineError::message("canonical tcgen05.ld shape overflows usize"))?;
        let actual_elements = element_count::<Shape>()?;
        if actual_elements != expected_elements {
            return Err(EngineError::message(format!(
                "canonical tcgen05.ld shape has {actual_elements} elements, expected {expected_elements}"
            )));
        }
        <SourceT as TcgenTransferEntry<DestinationT>>::canonical_32x32b_ld(
            warp,
            context,
            site,
            destination,
            source,
            Raw::ACCESS,
            elements_per_lane,
            base_lane,
            base_tcol,
            allocated_addr,
        )
    }
}

/// Closed specialization contract for a `tcgen05.st` operand mapping.
#[allow(private_bounds)]
pub trait Tcgen05StMapping: sealed::TcgenStMapping {
    type Args;
}

impl sealed::TcgenStMapping for variant::Mapped {}
impl Tcgen05StMapping for variant::Mapped {
    type Args = ();
}

impl sealed::TcgenStMapping for variant::Canonical32x32b {}
impl Tcgen05StMapping for variant::Canonical32x32b {
    /// Runtime TMEM `(base_lane, base_tcol, allocated_addr)` operands.
    type Args = (R<i64>, R<i64>, R<i64>);
}

impl<Shape, SourceT, DestinationT, Raw, Mapping> tcgen_st_spec::sealed::Sealed
    for variant::Tcgen05St<Shape, SourceT, DestinationT, Raw, Mapping>
where
    Shape: StaticShape,
    SourceT: TcgenTileElement + TcgenTransferEntry<DestinationT>,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
    Raw: RawStForm,
    Mapping: TcgenStMappingForm<Shape, SourceT, DestinationT, Raw>,
{
}
impl<Shape, SourceT, DestinationT, Raw, Mapping> tcgen_st_spec::Variant
    for variant::Tcgen05St<Shape, SourceT, DestinationT, Raw, Mapping>
where
    Shape: StaticShape,
    SourceT: TcgenTileElement + TcgenTransferEntry<DestinationT>,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
    Raw: RawStForm,
    Mapping: TcgenStMappingForm<Shape, SourceT, DestinationT, Raw>,
{
    type Args = Mapping::Args;
    type Source = Register;
    type Destination = super::Tmem;
    type Output = ();
}

impl<Shape, SourceT, DestinationT, Raw, Mapping> tcgen_st_spec::sealed::Execute<super::Engine>
    for variant::Tcgen05St<Shape, SourceT, DestinationT, Raw, Mapping>
where
    Shape: StaticShape,
    SourceT: TcgenTileElement + TcgenTransferEntry<DestinationT>,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
    Raw: RawStForm,
    Mapping: TcgenStMappingForm<Shape, SourceT, DestinationT, Raw>,
{
    #[inline(always)]
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<super::Tmem>,
        source: &MappedView<Register>,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        Mapping::issue(warp, context, site, destination, source, args)
    }
}
trait TcgenStMappingForm<Shape, SourceT, DestinationT, Raw>: Tcgen05StMapping
where
    Shape: StaticShape,
    SourceT: TcgenTileElement + TcgenTransferEntry<DestinationT>,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
    Raw: RawStForm,
{
    fn issue(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<super::Tmem>,
        source: &MappedView<Register>,
        args: Self::Args,
    ) -> Result<(), EngineError>;
}

impl<Shape, SourceT, DestinationT, Raw> TcgenStMappingForm<Shape, SourceT, DestinationT, Raw>
    for variant::Mapped
where
    Shape: StaticShape,
    SourceT: TcgenTileElement + TcgenTransferEntry<DestinationT>,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
    Raw: RawStForm,
{
    fn issue(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<super::Tmem>,
        source: &MappedView<Register>,
        (): Self::Args,
    ) -> Result<(), EngineError> {
        <SourceT as TcgenTransferEntry<DestinationT>>::mapped_st(
            warp,
            context,
            site,
            destination,
            source,
            Shape::EXTENTS,
            Raw::ACCESS,
        )
    }
}

impl<Shape, SourceT, DestinationT, Raw> TcgenStMappingForm<Shape, SourceT, DestinationT, Raw>
    for variant::Canonical32x32b
where
    Shape: StaticShape,
    SourceT: TcgenTileElement + TcgenTransferEntry<DestinationT>,
    DestinationT: TcgenTileElement<Physical = SourceT::Physical>,
    Raw: RawCanonical32x32bStForm,
{
    fn issue(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<super::Tmem>,
        source: &MappedView<Register>,
        (base_lane, base_tcol, allocated_addr): Self::Args,
    ) -> Result<(), EngineError> {
        let row_bytes = Raw::NUM
            .checked_mul(4)
            .ok_or_else(|| EngineError::message("canonical tcgen05.st register count overflows"))?;
        if row_bytes % SourceT::Physical::BYTE_LEN != 0 {
            return Err(EngineError::message(
                "canonical tcgen05.st register width is not divisible by element width",
            ));
        }
        let elements_per_lane = row_bytes / SourceT::Physical::BYTE_LEN;
        let expected_elements = 4_usize
            .checked_mul(crate::WARP_SIZE)
            .and_then(|threads| threads.checked_mul(elements_per_lane))
            .ok_or_else(|| EngineError::message("canonical tcgen05.st shape overflows usize"))?;
        if element_count::<Shape>()? != expected_elements {
            return Err(EngineError::message(format!(
                "canonical tcgen05.st shape has {} elements, expected {expected_elements}",
                element_count::<Shape>()?
            )));
        }
        <SourceT as TcgenTransferEntry<DestinationT>>::canonical_32x32b_st(
            warp,
            context,
            site,
            destination,
            source,
            Raw::ACCESS,
            elements_per_lane,
            base_lane,
            base_tcol,
            allocated_addr,
        )
    }
}

trait GemmInput {
    type Storage: RuntimeScalar;
    const BITS: u8;
    const SPEC: GemmInputSpec;
    fn decode(value: Self::Storage, bit_offset: u8) -> Result<f32, EngineError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GemmInputFormat {
    F16,
    Bf16,
    F32,
    Tf32,
    E4m3,
    E8m0,
    E2m1,
    NoScale,
}

#[derive(Clone, Copy)]
struct GemmInputSpec {
    format: GemmInputFormat,
    bits: u8,
    byte_len: usize,
    read: fn(
        &crate::PhysicalMemory,
        &crate::WarpContext,
        &crate::TcgenLifecycleHub,
        crate::TmemAccessMode,
        &crate::runtime::RuntimeBuffer,
        MappedElement,
        bool,
        &str,
    ) -> Result<f32, EngineError>,
}

impl GemmInput for super::reg::variant::F16 {
    type Storage = u16;
    const BITS: u8 = 16;
    const SPEC: GemmInputSpec = GemmInputSpec {
        format: GemmInputFormat::F16,
        bits: Self::BITS,
        byte_len: Self::Storage::BYTE_LEN,
        read: read_gemm_f16,
    };
    fn decode(value: Self::Storage, bit_offset: u8) -> Result<f32, EngineError> {
        if bit_offset != 0 {
            return Err(EngineError::message(
                "fp16 GEMM element has a sub-byte offset",
            ));
        }
        Ok(crate::fp16_bits_to_f32(value))
    }
}

impl GemmInput for super::reg::variant::Bf16 {
    type Storage = u16;
    const BITS: u8 = 16;
    const SPEC: GemmInputSpec = GemmInputSpec {
        format: GemmInputFormat::Bf16,
        bits: Self::BITS,
        byte_len: Self::Storage::BYTE_LEN,
        read: read_gemm_bf16,
    };
    fn decode(value: Self::Storage, bit_offset: u8) -> Result<f32, EngineError> {
        if bit_offset != 0 {
            return Err(EngineError::message(
                "bf16 GEMM element has a sub-byte offset",
            ));
        }
        Ok(crate::bf16_bits_to_f32(value))
    }
}

impl GemmInput for super::reg::variant::F32 {
    type Storage = f32;
    const BITS: u8 = 32;
    const SPEC: GemmInputSpec = GemmInputSpec {
        format: GemmInputFormat::F32,
        bits: Self::BITS,
        byte_len: Self::Storage::BYTE_LEN,
        read: read_gemm_f32,
    };
    fn decode(value: Self::Storage, bit_offset: u8) -> Result<f32, EngineError> {
        if bit_offset != 0 {
            return Err(EngineError::message(
                "f32 GEMM element has a sub-byte offset",
            ));
        }
        Ok(value)
    }
}

impl GemmInput for variant::Tf32 {
    type Storage = f32;
    const BITS: u8 = 32;
    const SPEC: GemmInputSpec = GemmInputSpec {
        format: GemmInputFormat::Tf32,
        bits: Self::BITS,
        byte_len: Self::Storage::BYTE_LEN,
        read: read_gemm_tf32,
    };
    fn decode(value: Self::Storage, bit_offset: u8) -> Result<f32, EngineError> {
        if bit_offset != 0 {
            return Err(EngineError::message(
                "tf32 GEMM element has a sub-byte offset",
            ));
        }
        Ok(crate::numpy_backend::f32_to_tf32(value))
    }
}

impl GemmInput for variant::E4m3 {
    type Storage = u8;
    const BITS: u8 = 8;
    const SPEC: GemmInputSpec = GemmInputSpec {
        format: GemmInputFormat::E4m3,
        bits: Self::BITS,
        byte_len: Self::Storage::BYTE_LEN,
        read: read_gemm_e4m3,
    };
    fn decode(value: Self::Storage, bit_offset: u8) -> Result<f32, EngineError> {
        if bit_offset != 0 {
            return Err(EngineError::message(
                "e4m3 GEMM element has a sub-byte offset",
            ));
        }
        Ok(crate::float8_e4m3fn_bits_to_f32(value))
    }
}

impl GemmInput for variant::E8m0 {
    type Storage = u8;
    const BITS: u8 = 8;
    const SPEC: GemmInputSpec = GemmInputSpec {
        format: GemmInputFormat::E8m0,
        bits: Self::BITS,
        byte_len: Self::Storage::BYTE_LEN,
        read: read_gemm_e8m0,
    };
    fn decode(value: Self::Storage, bit_offset: u8) -> Result<f32, EngineError> {
        if bit_offset != 0 {
            return Err(EngineError::message(
                "e8m0 GEMM element has a sub-byte offset",
            ));
        }
        Ok(crate::float8_e8m0fnu_bits_to_f32(value))
    }
}

impl GemmInput for variant::E2m1 {
    type Storage = u8;
    const BITS: u8 = 4;
    const SPEC: GemmInputSpec = GemmInputSpec {
        format: GemmInputFormat::E2m1,
        bits: Self::BITS,
        byte_len: Self::Storage::BYTE_LEN,
        read: read_gemm_e2m1,
    };
    fn decode(value: Self::Storage, bit_offset: u8) -> Result<f32, EngineError> {
        if !matches!(bit_offset, 0 | 4) {
            return Err(EngineError::message(format!(
                "e2m1 GEMM bit offset {bit_offset} must be 0 or 4"
            )));
        }
        Ok(crate::float4_e2m1fn_bits_to_f32(
            (value >> bit_offset) & 0x0f,
        ))
    }
}

/// Closed specialization contract for the state space of GEMM operand A.
///
/// It is public because that state space occurs in [`gemm_async`] and
/// [`gemm_async_ws`] signatures; placement validation remains engine-owned.
#[allow(private_bounds)]
pub trait GemmAPlacement: sealed::GemmAPlacement {
    type Space: MemorySpace;
}

trait APlacement: GemmAPlacement {
    const IS_TMEM: bool;
}

impl sealed::GemmAPlacement for variant::AShared {}
impl GemmAPlacement for variant::AShared {
    type Space = Shared;
}
impl APlacement for variant::AShared {
    const IS_TMEM: bool = false;
}

impl sealed::GemmAPlacement for variant::ATmem {}
impl GemmAPlacement for variant::ATmem {
    type Space = super::Tmem;
}
impl APlacement for variant::ATmem {
    const IS_TMEM: bool = true;
}

#[derive(Clone)]
struct GemmRuntimeArgs {
    scale_a: Option<MappedView<super::Tmem>>,
    scale_b: Option<MappedView<super::Tmem>>,
    accumulate: R<bool>,
    descriptor: Option<R<u32>>,
    predicate: Option<R<bool>>,
}

/// Closed specialization contract for the runtime operands carried by a GEMM
/// form. Descriptor and predicate presence are therefore compile-time facts.
#[allow(private_bounds)]
pub trait GemmMode: sealed::GemmMode {
    type Args;
}

mod gemm_operands {
    use super::{GemmInput, GemmMode, GemmRuntimeArgs};

    pub struct NoScale;

    pub(super) trait Form: GemmMode {
        type Scale: GemmInput;
        const SCALED: bool;
        const HAS_DESCRIPTOR: bool;
        fn split(args: Self::Args) -> GemmRuntimeArgs;
    }
}

use gemm_operands::{Form as GemmModeForm, NoScale};

macro_rules! gemm_mode_args {
    (impl<$scale:ident> $mode:ty => $args:ty) => {
        impl<$scale: GemmInput> sealed::GemmMode for $mode {}
        impl<$scale: GemmInput> GemmMode for $mode {
            type Args = $args;
        }
    };
    ($mode:ty => $args:ty) => {
        impl sealed::GemmMode for $mode {}
        impl GemmMode for $mode {
            type Args = $args;
        }
    };
}

gemm_mode_args!(variant::Dense => R<bool>);
gemm_mode_args!(variant::DensePredicated => (R<bool>, R<bool>));
gemm_mode_args!(variant::DenseDescriptor => (R<bool>, R<u32>));
gemm_mode_args!(variant::DenseDescriptorPredicated => (R<bool>, R<u32>, R<bool>));
gemm_mode_args!(impl<Scale> variant::BlockScaled<Scale> => (
    MappedView<super::Tmem>,
    MappedView<super::Tmem>,
    R<bool>,
));
gemm_mode_args!(impl<Scale> variant::BlockScaledPredicated<Scale> => (
    MappedView<super::Tmem>,
    MappedView<super::Tmem>,
    R<bool>,
    R<bool>,
));
gemm_mode_args!(impl<Scale> variant::BlockScaledDescriptor<Scale> => (
    MappedView<super::Tmem>,
    MappedView<super::Tmem>,
    R<bool>,
    R<u32>,
));
gemm_mode_args!(impl<Scale> variant::BlockScaledDescriptorPredicated<Scale> => (
    MappedView<super::Tmem>,
    MappedView<super::Tmem>,
    R<bool>,
    R<u32>,
    R<bool>,
));

impl GemmInput for NoScale {
    type Storage = u8;
    const BITS: u8 = 8;
    const SPEC: GemmInputSpec = GemmInputSpec {
        format: GemmInputFormat::NoScale,
        bits: Self::BITS,
        byte_len: Self::Storage::BYTE_LEN,
        read: read_gemm_no_scale,
    };
    fn decode(_value: Self::Storage, _bit_offset: u8) -> Result<f32, EngineError> {
        Err(EngineError::message("dense GEMM has no scale operand"))
    }
}

impl GemmModeForm for variant::Dense {
    type Scale = NoScale;
    const SCALED: bool = false;
    const HAS_DESCRIPTOR: bool = false;
    fn split(accumulate: Self::Args) -> GemmRuntimeArgs {
        GemmRuntimeArgs {
            scale_a: None,
            scale_b: None,
            accumulate,
            descriptor: None,
            predicate: None,
        }
    }
}

impl GemmModeForm for variant::DensePredicated {
    type Scale = NoScale;
    const SCALED: bool = false;
    const HAS_DESCRIPTOR: bool = false;
    fn split((accumulate, predicate): Self::Args) -> GemmRuntimeArgs {
        GemmRuntimeArgs {
            scale_a: None,
            scale_b: None,
            accumulate,
            descriptor: None,
            predicate: Some(predicate),
        }
    }
}

impl GemmModeForm for variant::DenseDescriptor {
    type Scale = NoScale;
    const SCALED: bool = false;
    const HAS_DESCRIPTOR: bool = true;
    fn split((accumulate, descriptor): Self::Args) -> GemmRuntimeArgs {
        GemmRuntimeArgs {
            scale_a: None,
            scale_b: None,
            accumulate,
            descriptor: Some(descriptor),
            predicate: None,
        }
    }
}

impl GemmModeForm for variant::DenseDescriptorPredicated {
    type Scale = NoScale;
    const SCALED: bool = false;
    const HAS_DESCRIPTOR: bool = true;
    fn split((accumulate, descriptor, predicate): Self::Args) -> GemmRuntimeArgs {
        GemmRuntimeArgs {
            scale_a: None,
            scale_b: None,
            accumulate,
            descriptor: Some(descriptor),
            predicate: Some(predicate),
        }
    }
}

impl<Scale: GemmInput> GemmModeForm for variant::BlockScaled<Scale> {
    type Scale = Scale;
    const SCALED: bool = true;
    const HAS_DESCRIPTOR: bool = false;
    fn split((scale_a, scale_b, accumulate): Self::Args) -> GemmRuntimeArgs {
        GemmRuntimeArgs {
            scale_a: Some(scale_a),
            scale_b: Some(scale_b),
            accumulate,
            descriptor: None,
            predicate: None,
        }
    }
}

impl<Scale: GemmInput> GemmModeForm for variant::BlockScaledPredicated<Scale> {
    type Scale = Scale;
    const SCALED: bool = true;
    const HAS_DESCRIPTOR: bool = false;
    fn split((scale_a, scale_b, accumulate, predicate): Self::Args) -> GemmRuntimeArgs {
        GemmRuntimeArgs {
            scale_a: Some(scale_a),
            scale_b: Some(scale_b),
            accumulate,
            descriptor: None,
            predicate: Some(predicate),
        }
    }
}

impl<Scale: GemmInput> GemmModeForm for variant::BlockScaledDescriptor<Scale> {
    type Scale = Scale;
    const SCALED: bool = true;
    const HAS_DESCRIPTOR: bool = true;
    fn split((scale_a, scale_b, accumulate, descriptor): Self::Args) -> GemmRuntimeArgs {
        GemmRuntimeArgs {
            scale_a: Some(scale_a),
            scale_b: Some(scale_b),
            accumulate,
            descriptor: Some(descriptor),
            predicate: None,
        }
    }
}

impl<Scale: GemmInput> GemmModeForm for variant::BlockScaledDescriptorPredicated<Scale> {
    type Scale = Scale;
    const SCALED: bool = true;
    const HAS_DESCRIPTOR: bool = true;
    fn split((scale_a, scale_b, accumulate, descriptor, predicate): Self::Args) -> GemmRuntimeArgs {
        GemmRuntimeArgs {
            scale_a: Some(scale_a),
            scale_b: Some(scale_b),
            accumulate,
            descriptor: Some(descriptor),
            predicate: Some(predicate),
        }
    }
}

mod gemm_static {
    use super::{APlacement, GemmAEntry, GemmInput, GemmModeForm, TileTmemAccess};

    pub(super) trait StaticGemm {
        type Mode: GemmModeForm;
        type AInput: GemmInput;
        type BInput: GemmInput;
        type APlace: APlacement + GemmAEntry;
        type Access: TileTmemAccess;
        type Mapping;
        const M: usize;
        const N: usize;
        const K: usize;
        const INSTR_M: usize;
        const INSTR_N: usize;
        const INSTR_K: usize;
        const CTA_GROUP: u32;
        const TRANS_A: bool;
        const TRANS_B: bool;
        const SCALE_VECTOR: usize;
        const EXPECTED_DESCRIPTOR: u32;
        const DESCRIPTOR_MASK: u32;
    }
}

use gemm_static::StaticGemm;

impl<
        Mode,
        AInput,
        BInput,
        APlace,
        Access,
        const M: usize,
        const N: usize,
        const K: usize,
        const INSTR_M: usize,
        const INSTR_N: usize,
        const INSTR_K: usize,
        const CTA_GROUP: u32,
        const TRANS_A: bool,
        const TRANS_B: bool,
        const SCALE_VECTOR: usize,
        const EXPECTED_DESCRIPTOR: u32,
        const DESCRIPTOR_MASK: u32,
        Mapping,
    > StaticGemm
    for variant::Gemm<
        Mode,
        AInput,
        BInput,
        APlace,
        Access,
        M,
        N,
        K,
        INSTR_M,
        INSTR_N,
        INSTR_K,
        CTA_GROUP,
        TRANS_A,
        TRANS_B,
        SCALE_VECTOR,
        EXPECTED_DESCRIPTOR,
        DESCRIPTOR_MASK,
        Mapping,
    >
where
    Mode: GemmModeForm,
    AInput: GemmInput,
    BInput: GemmInput,
    APlace: APlacement + GemmAEntry,
    Access: TileTmemAccess,
{
    type Mode = Mode;
    type AInput = AInput;
    type BInput = BInput;
    type APlace = APlace;
    type Access = Access;
    type Mapping = Mapping;
    const M: usize = M;
    const N: usize = N;
    const K: usize = K;
    const INSTR_M: usize = INSTR_M;
    const INSTR_N: usize = INSTR_N;
    const INSTR_K: usize = INSTR_K;
    const CTA_GROUP: u32 = CTA_GROUP;
    const TRANS_A: bool = TRANS_A;
    const TRANS_B: bool = TRANS_B;
    const SCALE_VECTOR: usize = SCALE_VECTOR;
    const EXPECTED_DESCRIPTOR: u32 = EXPECTED_DESCRIPTOR;
    const DESCRIPTOR_MASK: u32 = DESCRIPTOR_MASK;
}

/// Engine-private constants selected by an exact typed GEMM specialization.
/// This is not a frontend IR or a runtime opcode: the inline specialization
/// adapter constructs it from associated constants before entering the shared
/// numeric/effect implementation.
#[derive(Clone, Copy)]
struct MappedGemmConfig {
    m: usize,
    n: usize,
    k: usize,
    instr_m: usize,
    instr_n: usize,
    instr_k: usize,
    cta_group: u32,
    trans_a: bool,
    trans_b: bool,
    scale_vector: usize,
    expected_descriptor: u32,
    descriptor_mask: u32,
    access: crate::TmemAccessMode,
    a_is_tmem: bool,
    scaled: bool,
    has_descriptor: bool,
    weight_stationary: bool,
    ws_batched: bool,
    cta2_banked_a: bool,
    pipeline_class: Option<crate::runtime::TcgenMmaPipelineClass>,
}

impl MappedGemmConfig {
    #[inline(always)]
    fn from_variant<V: StaticGemm>(weight_stationary: bool) -> Self
    where
        V::Mapping: GemmMapping<V::Mode>,
    {
        Self {
            m: V::M,
            n: V::N,
            k: V::K,
            instr_m: V::INSTR_M,
            instr_n: V::INSTR_N,
            instr_k: V::INSTR_K,
            cta_group: V::CTA_GROUP,
            trans_a: V::TRANS_A,
            trans_b: V::TRANS_B,
            scale_vector: V::SCALE_VECTOR,
            expected_descriptor: V::EXPECTED_DESCRIPTOR,
            descriptor_mask: V::DESCRIPTOR_MASK,
            access: V::Access::VALUE,
            a_is_tmem: V::APlace::IS_TMEM,
            scaled: V::Mode::SCALED,
            has_descriptor: V::Mode::HAS_DESCRIPTOR,
            weight_stationary,
            ws_batched: V::Mapping::WS_BATCHED,
            cta2_banked_a: V::Mapping::CTA2_BANKED_A,
            // The tile frontend accepts only float32 TMEM destinations for
            // TCGEN GEMM, so its accumulator dtype is exact here rather than
            // inferred from the A/B instruction kind.
            pipeline_class: Some(crate::runtime::TcgenMmaPipelineClass::new(
                V::INSTR_M,
                V::INSTR_N,
                V::INSTR_K,
                crate::runtime::TcgenAccumulatorDtype::F32,
            )),
        }
    }
}

/// Placement-specific concrete GEMM entry.  Only the frontend-owned A view
/// type differs; all input formats and instruction qualifiers are erased into
/// small immutable specs before entering the numeric/effect core.
trait GemmAEntry: APlacement {
    fn execute_mapped(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<super::Tmem>,
        a: &MappedView<Self::Space>,
        b: &MappedView<Shared>,
        runtime: GemmRuntimeArgs,
        config: MappedGemmConfig,
        a_input: GemmInputSpec,
        b_input: GemmInputSpec,
        scale: GemmInputSpec,
    ) -> Result<(), EngineError>;
}

impl GemmAEntry for variant::AShared {
    fn execute_mapped(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<super::Tmem>,
        a: &MappedView<Shared>,
        b: &MappedView<Shared>,
        runtime: GemmRuntimeArgs,
        config: MappedGemmConfig,
        a_input: GemmInputSpec,
        b_input: GemmInputSpec,
        scale: GemmInputSpec,
    ) -> Result<(), EngineError> {
        execute_mapped_gemm::<Shared>(
            warp,
            context,
            site,
            destination,
            a,
            b,
            runtime,
            config,
            a_input,
            b_input,
            scale,
        )
    }
}

impl GemmAEntry for variant::ATmem {
    fn execute_mapped(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<super::Tmem>,
        a: &MappedView<super::Tmem>,
        b: &MappedView<Shared>,
        runtime: GemmRuntimeArgs,
        config: MappedGemmConfig,
        a_input: GemmInputSpec,
        b_input: GemmInputSpec,
        scale: GemmInputSpec,
    ) -> Result<(), EngineError> {
        execute_mapped_gemm::<super::Tmem>(
            warp,
            context,
            site,
            destination,
            a,
            b,
            runtime,
            config,
            a_input,
            b_input,
            scale,
        )
    }
}

fn map_gemm_element<S: MemorySpace>(
    view: &MappedView<S>,
    coordinates: &[i64],
    lane: usize,
    instruction_target: Option<usize>,
    label: &str,
) -> Result<MappedElement, EngineError> {
    let reference = view
        .map(LogicalCoord::new(coordinates), LaneId::from_index(lane))
        .map_err(|error| EngineError::message(format!("{label}: {error}")))?;
    if !reference.is_in_bounds() {
        return Err(EngineError::message(format!(
            "{label} produced an out-of-bounds element"
        )));
    }
    let mapped_target = reference.target_rank().map(|rank| rank as usize);
    if let (Some(mapped), Some(instruction)) = (mapped_target, instruction_target) {
        if mapped != instruction {
            return Err(EngineError::message(format!(
                "{label} selected CTA {mapped}, but the instruction targets CTA {instruction}"
            )));
        }
    }
    Ok(MappedElement {
        execution_lane: lane,
        target_cta: instruction_target.or(mapped_target),
        location: reference.location(),
    })
}

fn map_gemm_matrix<S: MemorySpace>(
    view: &MappedView<S>,
    rows: usize,
    columns: usize,
    transpose_storage: bool,
    lane: usize,
    target_cta: Option<usize>,
    row_offset: usize,
    label: &str,
) -> Result<Vec<MappedElement>, EngineError> {
    let mut elements = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or_else(|| EngineError::message("GEMM matrix element count overflow"))?,
    );
    for row in 0..rows {
        for column in 0..columns {
            let row = row
                .checked_add(row_offset)
                .ok_or_else(|| EngineError::message("GEMM row offset overflow"))?;
            let row =
                i64::try_from(row).map_err(|_| EngineError::message("GEMM row exceeds i64"))?;
            let column = i64::try_from(column)
                .map_err(|_| EngineError::message("GEMM column exceeds i64"))?;
            let coordinates = if transpose_storage {
                [column, row]
            } else {
                [row, column]
            };
            elements.push(map_gemm_element(
                view,
                &coordinates,
                lane,
                target_cta,
                label,
            )?);
        }
    }
    Ok(elements)
}

/// Map one register-resident logical matrix without teaching the engine a
/// frontend layout.  The frontend map marks exactly one owning lane for each
/// logical element; the engine only validates and consumes that pure result.
fn map_register_gemm_matrix(
    view: &MappedView<Register>,
    context: ExecCtx,
    rows: usize,
    columns: usize,
    transpose_storage: bool,
    label: &str,
) -> Result<Vec<MappedElement>, EngineError> {
    let mut elements = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or_else(|| EngineError::message("warp GEMM matrix element count overflow"))?,
    );
    for row in 0..rows {
        for column in 0..columns {
            let row =
                i64::try_from(row).map_err(|_| EngineError::message("GEMM row exceeds i64"))?;
            let column = i64::try_from(column)
                .map_err(|_| EngineError::message("GEMM column exceeds i64"))?;
            let coordinates = if transpose_storage {
                [column, row]
            } else {
                [row, column]
            };
            let mut owner = None;
            for lane in context.active_mask() {
                let reference = view
                    .map(LogicalCoord::new(&coordinates), LaneId::from_index(lane))
                    .map_err(|error| EngineError::message(format!("{label}: {error}")))?;
                if !reference.is_in_bounds() {
                    continue;
                }
                if reference.target_rank().is_some() {
                    return Err(EngineError::message(format!(
                        "{label} register map selected a CTA rank"
                    )));
                }
                if owner.is_some() {
                    return Err(EngineError::message(format!(
                        "{label} has more than one owning lane at ({row}, {column})"
                    )));
                }
                owner = Some(MappedElement {
                    execution_lane: lane,
                    target_cta: None,
                    location: reference.location(),
                });
            }
            elements.push(owner.ok_or_else(|| {
                EngineError::message(format!("{label} has no owning lane at ({row}, {column})"))
            })?);
        }
    }
    Ok(elements)
}

fn map_gemm_scale_matrix(
    view: &MappedView<super::Tmem>,
    rows: usize,
    columns: usize,
    lane: usize,
    target_cta: Option<usize>,
    row_offset: usize,
    descriptor_selector: Option<usize>,
    values_per_instruction: usize,
    label: &str,
) -> Result<Vec<MappedElement>, EngineError> {
    let mut elements = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or_else(|| EngineError::message("GEMM scale element count overflow"))?,
    );
    for row in 0..rows {
        for column in 0..columns {
            let mapped_row = row
                .checked_add(row_offset)
                .ok_or_else(|| EngineError::message("GEMM scale row offset overflow"))?;
            let coordinates = [
                i64::try_from(mapped_row)
                    .map_err(|_| EngineError::message("GEMM scale row exceeds i64"))?,
                i64::try_from(column)
                    .map_err(|_| EngineError::message("GEMM scale column exceeds i64"))?,
            ];
            elements.push(map_gemm_element(
                view,
                &coordinates,
                lane,
                target_cta,
                label,
            )?);
        }
    }
    if let Some(selector) = descriptor_selector {
        if values_per_instruction == 0 {
            return Err(EngineError::message(
                "GEMM runtime scale selector has no values per instruction",
            ));
        }
        let reuses_descriptor_cell = columns > values_per_instruction
            && (0..rows).all(|row| {
                (values_per_instruction..columns).all(|column| {
                    elements[row * columns + column].location
                        == elements[row * columns + column % values_per_instruction].location
                })
            });
        if reuses_descriptor_cell {
            for row in 0..rows {
                for column in 0..columns {
                    let slot = selector
                        .checked_add(column % values_per_instruction)
                        .ok_or_else(|| EngineError::message("GEMM scale selector overflow"))?;
                    let slot = i64::try_from(slot)
                        .map_err(|_| EngineError::message("GEMM scale selector exceeds i64"))?;
                    let element = &mut elements[row * columns + column];
                    element.location = match element.location {
                        ElementLocation::Tmem {
                            mapped_lane,
                            tcol_element,
                            allocated_addr,
                            bit_offset,
                        } => ElementLocation::Tmem {
                            mapped_lane,
                            tcol_element: tcol_element.checked_add(slot).ok_or_else(|| {
                                EngineError::message("GEMM scale TMEM column overflow")
                            })?,
                            allocated_addr,
                            bit_offset,
                        },
                        _ => {
                            return Err(EngineError::message(format!(
                                "{label} expected a TMEM scale coordinate"
                            )))
                        }
                    };
                }
            }
        }
    }
    Ok(elements)
}

fn storage_location(
    element: MappedElement,
    label: &str,
) -> Result<(Option<usize>, Option<(i64, i64, i64)>, u8), EngineError> {
    match element.location {
        ElementLocation::ByteOffset(offset) => Ok((
            Some(usize::try_from(offset).map_err(|_| {
                EngineError::message(format!("{label} has a negative/oversized byte offset"))
            })?),
            None,
            0,
        )),
        ElementLocation::BitOffset {
            byte_offset,
            bit_offset,
        } => Ok((
            Some(usize::try_from(byte_offset).map_err(|_| {
                EngineError::message(format!("{label} has a negative/oversized byte offset"))
            })?),
            None,
            bit_offset,
        )),
        ElementLocation::Tmem {
            mapped_lane,
            tcol_element,
            allocated_addr,
            bit_offset,
        } => Ok((
            None,
            Some((mapped_lane, tcol_element, allocated_addr)),
            bit_offset,
        )),
    }
}

fn read_gemm_value<I: GemmInput>(
    physical: &crate::PhysicalMemory,
    context: &crate::WarpContext,
    lifecycle: &crate::TcgenLifecycleHub,
    access: crate::TmemAccessMode,
    buffer: &crate::runtime::RuntimeBuffer,
    element: MappedElement,
    expect_tmem: bool,
    label: &str,
) -> Result<f32, EngineError> {
    let (byte_offset, tmem, bit_offset) = storage_location(element, label)?;
    let storage = if expect_tmem {
        let (mapped_lane, tcol_element, allocated_addr) = tmem
            .ok_or_else(|| EngineError::message(format!("{label} expected a TMEM coordinate")))?;
        crate::runtime::load_tmem_scalar_at_cta::<I::Storage>(
            physical,
            context,
            lifecycle,
            access,
            buffer,
            element
                .target_cta
                .unwrap_or_else(|| context.cta_id_in_cluster()),
            mapped_lane,
            tcol_element,
            allocated_addr,
            element.execution_lane,
        )
        .map_err(EngineError::from)?
    } else {
        let byte_offset = byte_offset.ok_or_else(|| {
            EngineError::message(format!("{label} expected a byte-addressed element"))
        })?;
        if byte_offset % I::Storage::BYTE_LEN != 0 {
            return Err(EngineError::message(format!(
                "{label} byte offset {byte_offset} is not aligned to {}",
                I::Storage::BYTE_LEN
            )));
        }
        let index = i64::try_from(byte_offset / I::Storage::BYTE_LEN)
            .map_err(|_| EngineError::message(format!("{label} index exceeds i64")))?;
        match element.target_cta {
            Some(target) => crate::runtime::read_shared_scalar_at_cta::<I::Storage>(
                physical,
                context,
                buffer,
                target,
                index,
                element.execution_lane,
            ),
            None => crate::runtime::load_scalar_lane::<I::Storage>(
                physical,
                context,
                buffer,
                index,
                element.execution_lane,
            ),
        }
        .map_err(EngineError::from)?
    };
    I::decode(storage, bit_offset)
}

macro_rules! concrete_gemm_reader {
    ($function:ident, $input:ty) => {
        #[inline(never)]
        fn $function(
            physical: &crate::PhysicalMemory,
            context: &crate::WarpContext,
            lifecycle: &crate::TcgenLifecycleHub,
            access: crate::TmemAccessMode,
            buffer: &crate::runtime::RuntimeBuffer,
            element: MappedElement,
            expect_tmem: bool,
            label: &str,
        ) -> Result<f32, EngineError> {
            read_gemm_value::<$input>(
                physical,
                context,
                lifecycle,
                access,
                buffer,
                element,
                expect_tmem,
                label,
            )
        }
    };
}

concrete_gemm_reader!(read_gemm_f16, super::reg::variant::F16);
concrete_gemm_reader!(read_gemm_bf16, super::reg::variant::Bf16);
concrete_gemm_reader!(read_gemm_f32, super::reg::variant::F32);
concrete_gemm_reader!(read_gemm_tf32, variant::Tf32);
concrete_gemm_reader!(read_gemm_e4m3, variant::E4m3);
concrete_gemm_reader!(read_gemm_e8m0, variant::E8m0);
concrete_gemm_reader!(read_gemm_e2m1, variant::E2m1);

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn read_gemm_no_scale(
    _physical: &crate::PhysicalMemory,
    _context: &crate::WarpContext,
    _lifecycle: &crate::TcgenLifecycleHub,
    _access: crate::TmemAccessMode,
    _buffer: &crate::runtime::RuntimeBuffer,
    _element: MappedElement,
    _expect_tmem: bool,
    _label: &str,
) -> Result<f32, EngineError> {
    Err(EngineError::message("dense GEMM has no scale operand"))
}

fn gemm_regular_footprints(
    elements: &[MappedElement],
    label: &str,
    input: GemmInputSpec,
) -> Result<Vec<(usize, Option<usize>, usize, usize)>, EngineError> {
    elements
        .iter()
        .copied()
        .map(|element| {
            let (byte_offset, tmem, bit_offset) = storage_location(element, label)?;
            if tmem.is_some() {
                return Err(EngineError::message(format!(
                    "{label} expected byte-addressed elements"
                )));
            }
            if input.bits != 4 && bit_offset != 0 {
                return Err(EngineError::message(format!(
                    "{label} has a sub-byte offset for a byte-sized type"
                )));
            }
            Ok((
                element.execution_lane,
                element.target_cta,
                byte_offset.expect("validated byte location"),
                input.byte_len,
            ))
        })
        .collect()
}

fn gemm_tmem_footprints(
    elements: &[MappedElement],
    label: &str,
    input: GemmInputSpec,
) -> Result<Vec<(usize, usize, Option<usize>, i64, i64, i64, usize)>, EngineError> {
    let mut footprints = Vec::with_capacity(elements.len());
    for &element in elements {
        let (byte_offset, tmem, bit_offset) = storage_location(element, label)?;
        if byte_offset.is_some() {
            return Err(EngineError::message(format!(
                "{label} expected TMEM elements"
            )));
        }
        if input.bits != 4 && bit_offset != 0 {
            return Err(EngineError::message(format!(
                "{label} has a sub-byte offset for a byte-sized type"
            )));
        }
        let (mapped_lane, tcol_element, allocated_addr) = tmem.expect("validated TMEM location");
        footprints.push((
            element.execution_lane,
            element.execution_lane,
            element.target_cta,
            mapped_lane,
            tcol_element,
            allocated_addr,
            input.byte_len,
        ));
    }
    Ok(footprints)
}

#[allow(clippy::type_complexity)]
fn map_gemm_operands<A: MemorySpace>(
    context: ExecCtx,
    lane: usize,
    destination: &MappedView<super::Tmem>,
    a: &MappedView<A>,
    b: &MappedView<Shared>,
    runtime: &GemmRuntimeArgs,
    config: MappedGemmConfig,
) -> Result<
    (
        Vec<MappedElement>,
        Vec<MappedElement>,
        Vec<MappedElement>,
        Option<Vec<MappedElement>>,
        Option<Vec<MappedElement>>,
    ),
    EngineError,
> {
    let group = usize::try_from(config.cta_group)
        .map_err(|_| EngineError::message("GEMM CTA group exceeds usize"))?;
    if config.n % group != 0 {
        return Err(EngineError::message(format!(
            "typed TCGEN GEMM N={} is not divisible by cta_group={group}",
            config.n
        )));
    }
    let b_rows_per_cta = config.n / group;
    let pair_base = context.into_inner().cta_id_in_cluster() & !1_usize;
    let targets: Vec<Option<usize>> = match config.cta_group {
        1 => vec![None],
        2 => vec![Some(pair_base), Some(pair_base + 1)],
        _ => unreachable!("validated TCGEN CTA group"),
    };
    let mut a_elements = Vec::new();
    let mut b_elements = Vec::new();
    let mut d_elements = Vec::new();
    for &target in &targets {
        let a_rows = if config.ws_batched || config.cta2_banked_a {
            config
                .m
                .checked_mul(2)
                .ok_or_else(|| EngineError::message("batched WS GEMM A rows overflow"))?
        } else {
            config.m
        };
        a_elements.extend(map_gemm_matrix(
            a,
            a_rows,
            config.k,
            config.trans_a,
            lane,
            target,
            0,
            "gemm A map",
        )?);
        b_elements.extend(map_gemm_matrix(
            b,
            b_rows_per_cta,
            config.k,
            config.trans_b,
            lane,
            target,
            0,
            "gemm B map",
        )?);
        d_elements.extend(map_gemm_matrix(
            destination,
            config.m,
            config.n,
            false,
            lane,
            target,
            0,
            "gemm D map",
        )?);
    }
    let (scale_a, scale_b) = if config.scaled {
        if config.scale_vector == 0 || config.k % config.scale_vector != 0 {
            return Err(EngineError::message(format!(
                "scaled GEMM K={} is not divisible by scale vector {}",
                config.k, config.scale_vector
            )));
        }
        let scale_columns = config.k / config.scale_vector;
        let scale_a = runtime
            .scale_a
            .as_ref()
            .ok_or_else(|| EngineError::message("scaled GEMM is missing its A scale view"))?;
        let scale_b = runtime
            .scale_b
            .as_ref()
            .ok_or_else(|| EngineError::message("scaled GEMM is missing its B scale view"))?;
        let mut a_scales = Vec::new();
        let mut b_scales = Vec::new();
        let descriptor = runtime.descriptor.as_ref().map(|values| values[lane]);
        let a_selector = descriptor.map(|value| ((value >> 29) & 0x3) as usize);
        let b_selector = descriptor.map(|value| ((value >> 4) & 0x3) as usize);
        let values_per_instruction = config.instr_k / config.scale_vector;
        for (target_index, &target) in targets.iter().enumerate() {
            a_scales.extend(map_gemm_scale_matrix(
                scale_a,
                config.m,
                scale_columns,
                lane,
                target,
                0,
                a_selector,
                values_per_instruction,
                "gemm SFA map",
            )?);
            b_scales.extend(map_gemm_scale_matrix(
                scale_b,
                b_rows_per_cta,
                scale_columns,
                lane,
                target,
                target_index * b_rows_per_cta,
                b_selector,
                values_per_instruction,
                "gemm SFB map",
            )?);
        }
        (Some(a_scales), Some(b_scales))
    } else {
        (None, None)
    };
    Ok((a_elements, b_elements, d_elements, scale_a, scale_b))
}

#[inline(never)]
fn execute_mapped_gemm<A>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    destination: &MappedView<super::Tmem>,
    a: &MappedView<A>,
    b: &MappedView<Shared>,
    runtime: GemmRuntimeArgs,
    config: MappedGemmConfig,
    a_input: GemmInputSpec,
    b_input: GemmInputSpec,
    scale: GemmInputSpec,
) -> Result<(), EngineError>
where
    A: MemorySpace,
{
    if !matches!(config.m, 64 | 128) || !matches!(config.cta_group, 1 | 2) {
        return Err(EngineError::message(format!(
            "typed TCGEN GEMM requires M=64/128 and cta_group=1/2, got M={} group={}",
            config.m, config.cta_group
        )));
    }
    if config.k == 0 || config.n == 0 {
        return Err(EngineError::message(
            "typed TCGEN GEMM N and K must be positive",
        ));
    }
    let total_m = config
        .m
        .checked_mul(config.cta_group as usize)
        .ok_or_else(|| EngineError::message("typed TCGEN GEMM total M overflows"))?;
    if config.instr_m == 0
        || config.instr_n == 0
        || config.instr_k == 0
        || total_m % config.instr_m != 0
        || config.n % config.instr_n != 0
        || config.k % config.instr_k != 0
    {
        return Err(EngineError::message(format!(
            "typed TCGEN GEMM tile ({total_m}, {}, {}) is not divisible by instruction ({}, {}, {})",
            config.n,
            config.k,
            config.instr_m,
            config.instr_n,
            config.instr_k,
        )));
    }
    if config.weight_stationary && config.cta_group != 1 {
        return Err(EngineError::message(
            "tcgen05.mma.ws typed GEMM requires cta_group=1",
        ));
    }
    if config.ws_batched
        && (!config.weight_stationary
            || config.m != 64
            || config.n % 2 != 0
            || !config.a_is_tmem
            || config.scaled)
    {
        return Err(EngineError::message(
            "batched tcgen05.mma.ws requires dense TMEM A, M=64, even N, and cta_group=1",
        ));
    }
    if config.cta2_banked_a
        && (config.cta_group != 2
            || config.m != 64
            || config.n % 2 != 0
            || !config.a_is_tmem
            || config.scaled)
    {
        return Err(EngineError::message(
            "CTA2 bank-batched GEMM requires dense TMEM A, M=64, even N, and cta_group=2",
        ));
    }
    if config.a_is_tmem && config.trans_a {
        return Err(EngineError::message(
            "TMEM A cannot use transA in typed TCGEN GEMM",
        ));
    }
    let joint_m = config
        .m
        .checked_mul(config.cta_group as usize)
        .ok_or_else(|| EngineError::message("typed TCGEN GEMM joint M overflows"))?;
    let selected = runtime
        .predicate
        .as_ref()
        .map(|predicate| predicate.to_mask(|_, value| *value))
        .unwrap_or(LaneMask::FULL)
        & context.active_mask();
    let Some(lane) = selected.first_active() else {
        return Ok(());
    };
    let issue_mask = LaneMask::single(lane)?;
    let issue_context = ExecCtx::from_inner(
        context
            .into_inner()
            .with_active_mask(issue_mask.into_inner()),
    );
    let accumulate = runtime.accumulate[lane];
    if let Some(descriptor) = runtime.descriptor.as_ref() {
        let actual = descriptor[lane];
        if actual & config.descriptor_mask != config.expected_descriptor & config.descriptor_mask {
            return Err(EngineError::message(format!(
                "typed GEMM runtime descriptor {actual:#010x} does not match the typed TCGEN ABI"
            )));
        }
        if config.scaled {
            if config.scale_vector == 0 || config.instr_k % config.scale_vector != 0 {
                return Err(EngineError::message(
                    "typed scaled GEMM instruction K is not divisible by its scale vector",
                ));
            }
            let values_per_instruction = config.instr_k / config.scale_vector;
            let a_selector = ((actual >> 29) & 0x3) as usize;
            let b_selector = ((actual >> 4) & 0x3) as usize;
            if a_selector + values_per_instruction > 4 || b_selector + values_per_instruction > 4 {
                return Err(EngineError::message(
                    "typed scaled GEMM descriptor selector crosses a 32-bit TMEM cell",
                ));
            }
        }
    } else if config.has_descriptor {
        return Err(EngineError::message(
            "typed GEMM descriptor specialization is missing its runtime descriptor",
        ));
    }

    let (a_elements, b_elements, d_elements, scale_a_elements, scale_b_elements) =
        map_gemm_operands(issue_context, lane, destination, a, b, &runtime, config)?;
    let a_buffer = a.allocation().inner().clone();
    let b_buffer = b.allocation().inner().clone();
    let d_buffer = destination.allocation().inner().clone();
    let a_logical_buffer = a.allocation().logical_buffer().unwrap_or("gemm_A");
    let b_logical_buffer = b.allocation().logical_buffer().unwrap_or("gemm_B");
    let d_logical_buffer = destination
        .allocation()
        .logical_buffer()
        .unwrap_or("gemm_D");
    let scale_a_buffer = runtime
        .scale_a
        .as_ref()
        .map(|view| view.allocation().inner().clone());
    let scale_b_buffer = runtime
        .scale_b
        .as_ref()
        .map(|view| view.allocation().inner().clone());
    let scale_a_logical_buffer = runtime
        .scale_a
        .as_ref()
        .and_then(|view| view.allocation().logical_buffer())
        .unwrap_or("gemm_SFA");
    let scale_b_logical_buffer = runtime
        .scale_b
        .as_ref()
        .and_then(|view| view.allocation().logical_buffer())
        .unwrap_or("gemm_SFB");
    let a_regular_accesses = (!config.a_is_tmem)
        .then(|| gemm_regular_footprints(&a_elements, "gemm A footprint", a_input))
        .transpose()?;
    let a_tmem_accesses = config
        .a_is_tmem
        .then(|| gemm_tmem_footprints(&a_elements, "gemm A footprint", a_input))
        .transpose()?;
    let b_accesses = gemm_regular_footprints(&b_elements, "gemm B footprint", b_input)?;
    let d_accesses = gemm_tmem_footprints(
        &d_elements,
        "gemm D footprint",
        super::reg::variant::F32::SPEC,
    )?;
    let scale_a_accesses = scale_a_elements
        .as_deref()
        .map(|elements| gemm_tmem_footprints(elements, "gemm SFA footprint", scale))
        .transpose()?;
    let scale_b_accesses = scale_b_elements
        .as_deref()
        .map(|elements| gemm_tmem_footprints(elements, "gemm SFB footprint", scale))
        .transpose()?;

    engine(warp).discard_mma_collectors(lane, config.weight_stationary);
    let operation = begin(warp, issue_context, site, OperationKind::TcgenWork, false)?;
    let physical = engine(warp).kernel().physical().clone();
    let lifecycle = engine(warp).kernel().services().tcgen();
    let inner = issue_context.into_inner();
    engine(warp).tcgen_instruction_issue(
        operation.as_ref(),
        config.cta_group,
        crate::runtime::TcgenPipelineOperation::Mma,
        config.pipeline_class,
        |record_regular, record_tmem| {
            if let Some(accesses) = a_regular_accesses.as_deref() {
                record_regular(
                    true,
                    OperationKind::Load,
                    &a_buffer,
                    Some(a_logical_buffer),
                    accesses,
                )?;
            }
            if let Some(accesses) = a_tmem_accesses.as_deref() {
                record_tmem(
                    OperationKind::Load,
                    &a_buffer,
                    a_logical_buffer,
                    config.access,
                    accesses,
                )?;
            }
            record_regular(
                false,
                OperationKind::Load,
                &b_buffer,
                Some(b_logical_buffer),
                &b_accesses,
            )?;
            if let (Some(buffer), Some(accesses)) =
                (scale_a_buffer.as_ref(), scale_a_accesses.as_deref())
            {
                record_tmem(
                    OperationKind::Load,
                    buffer,
                    scale_a_logical_buffer,
                    config.access,
                    accesses,
                )?;
            }
            if let (Some(buffer), Some(accesses)) =
                (scale_b_buffer.as_ref(), scale_b_accesses.as_deref())
            {
                record_tmem(
                    OperationKind::Load,
                    buffer,
                    scale_b_logical_buffer,
                    config.access,
                    accesses,
                )?;
            }
            if accumulate {
                record_tmem(
                    OperationKind::Load,
                    &d_buffer,
                    d_logical_buffer,
                    config.access,
                    &d_accesses,
                )?;
            }
            record_tmem(
                OperationKind::Store,
                &d_buffer,
                d_logical_buffer,
                config.access,
                &d_accesses,
            )?;
            Ok(())
        },
        || {
            let a_rows = if config.ws_batched || config.cta2_banked_a {
                joint_m
                    .checked_mul(2)
                    .ok_or_else(|| EngineError::message("batched WS GEMM A rows overflow"))?
            } else {
                joint_m
            };
            let mut a_values = Vec::with_capacity(a_rows * config.k);
            for &element in &a_elements {
                a_values.push((a_input.read)(
                    &physical,
                    &inner,
                    &lifecycle,
                    config.access,
                    &a_buffer,
                    element,
                    config.a_is_tmem,
                    "gemm A",
                )?);
            }
            let mut b_values = Vec::with_capacity(config.n * config.k);
            for &element in &b_elements {
                b_values.push((b_input.read)(
                    &physical,
                    &inner,
                    &lifecycle,
                    config.access,
                    &b_buffer,
                    element,
                    false,
                    "gemm B",
                )?);
            }
            if let (Some(buffer), Some(elements)) =
                (scale_a_buffer.as_ref(), scale_a_elements.as_deref())
            {
                let mut scales = Vec::with_capacity(elements.len());
                for &element in elements {
                    scales.push((scale.read)(
                        &physical,
                        &inner,
                        &lifecycle,
                        config.access,
                        buffer,
                        element,
                        true,
                        "gemm SFA",
                    )?);
                }
                for (index, value) in a_values.iter_mut().enumerate() {
                    let row = index / config.k;
                    let k = index % config.k;
                    *value *=
                        scales[row * (config.k / config.scale_vector) + k / config.scale_vector];
                }
            }
            if let (Some(buffer), Some(elements)) =
                (scale_b_buffer.as_ref(), scale_b_elements.as_deref())
            {
                let mut scales = Vec::with_capacity(elements.len());
                for &element in elements {
                    scales.push((scale.read)(
                        &physical,
                        &inner,
                        &lifecycle,
                        config.access,
                        buffer,
                        element,
                        true,
                        "gemm SFB",
                    )?);
                }
                for (index, value) in b_values.iter_mut().enumerate() {
                    let row = index / config.k;
                    let k = index % config.k;
                    *value *=
                        scales[row * (config.k / config.scale_vector) + k / config.scale_vector];
                }
            }
            let mut input_d = None;
            let mut d_values = Vec::new();
            if accumulate {
                d_values.reserve(d_elements.len());
                for &element in &d_elements {
                    d_values.push(read_tmem::<f32>(
                        &physical,
                        &inner,
                        &lifecycle,
                        config.access,
                        &d_buffer,
                        element,
                    )?);
                }
                input_d = Some((d_values.as_slice(), 1.0_f32));
            }
            let output = if config.cta2_banked_a {
                let half_n = config.n / 2;
                let target_rows = config.m;
                let mut output = vec![0.0_f32; joint_m * config.n];
                for target in 0..2 {
                    for bank in 0..2 {
                        let a_bank = target * 2 + bank;
                        let a_start = a_bank * target_rows * config.k;
                        let b_start = bank * half_n * config.k;
                        let d_half = accumulate.then(|| {
                            let mut values = Vec::with_capacity(target_rows * half_n);
                            for row in 0..target_rows {
                                let start = (target * target_rows + row) * config.n + bank * half_n;
                                values.extend_from_slice(&d_values[start..start + half_n]);
                            }
                            values
                        });
                        let bank_output = crate::runtime::mma_f32_abt_increasing_k(
                            target_rows,
                            half_n,
                            config.k,
                            &a_values[a_start..a_start + target_rows * config.k],
                            &b_values[b_start..b_start + half_n * config.k],
                            d_half.as_deref().map(|values| (values, 1.0_f32)),
                        )
                        .map_err(EngineError::from)?;
                        for row in 0..target_rows {
                            let destination =
                                (target * target_rows + row) * config.n + bank * half_n;
                            let source = row * half_n;
                            output[destination..destination + half_n]
                                .copy_from_slice(&bank_output[source..source + half_n]);
                        }
                    }
                }
                output
            } else if config.ws_batched {
                let half_n = config.n / 2;
                let mut output = vec![0.0_f32; joint_m * config.n];
                for half in 0..2 {
                    let a_start = half * joint_m * config.k;
                    let b_start = half * half_n * config.k;
                    let d_half = accumulate.then(|| {
                        let mut values = Vec::with_capacity(joint_m * half_n);
                        for row in 0..joint_m {
                            let start = row * config.n + half * half_n;
                            values.extend_from_slice(&d_values[start..start + half_n]);
                        }
                        values
                    });
                    let half_output = crate::runtime::mma_f32_abt_increasing_k(
                        joint_m,
                        half_n,
                        config.k,
                        &a_values[a_start..a_start + joint_m * config.k],
                        &b_values[b_start..b_start + half_n * config.k],
                        d_half.as_deref().map(|values| (values, 1.0_f32)),
                    )
                    .map_err(EngineError::from)?;
                    for row in 0..joint_m {
                        let destination = row * config.n + half * half_n;
                        let source = row * half_n;
                        output[destination..destination + half_n]
                            .copy_from_slice(&half_output[source..source + half_n]);
                    }
                }
                output
            } else {
                crate::runtime::mma_f32_abt_increasing_k(
                    joint_m, config.n, config.k, &a_values, &b_values, input_d,
                )
                .map_err(EngineError::from)?
            };
            for (&element, &value) in d_elements.iter().zip(&output) {
                write_tmem::<f32>(
                    &physical,
                    &inner,
                    &lifecycle,
                    config.access,
                    &d_buffer,
                    element,
                    value,
                )?;
            }
            Ok(())
        },
    )?;
    finish(warp, &operation)
}

/// Closed specialization contract for the mapping carried by a typed GEMM.
/// `Mapped` consumes frontend element maps. Canonical forms carry only the
/// runtime operands that could not be specialized by the frontend.
#[allow(private_bounds)]
pub trait GemmMapping<Mode: GemmMode>: sealed::GemmMapping {
    type Args;
    const WS_BATCHED: bool;
    const CTA2_BANKED_A: bool;
}

/// Mapping specializations legal for the ordinary and WS PTX mnemonics.
trait GemmRegularMapping: sealed::GemmMapping {}
trait GemmWsMapping: sealed::GemmMapping {}

macro_rules! mapped_gemm_variants {
    ($($marker:ty => {
        ws_batched: $ws_batched:literal,
        cta2_banked_a: $cta2_banked_a:literal
    }),+ $(,)?) => {
        $(
            impl sealed::GemmMapping for $marker {}
            impl<Mode: GemmMode> GemmMapping<Mode> for $marker {
                type Args = Mode::Args;
                const WS_BATCHED: bool = $ws_batched;
                const CTA2_BANKED_A: bool = $cta2_banked_a;
            }
        )+
    };
}

mapped_gemm_variants!(
    variant::Mapped => {
        ws_batched: false,
        cta2_banked_a: false
    },
    variant::MappedWsBatched => {
        ws_batched: true,
        cta2_banked_a: false
    },
    variant::MappedCta2BankedA => {
        ws_batched: false,
        cta2_banked_a: true
    },
);

impl GemmRegularMapping for variant::Mapped {}
impl GemmRegularMapping for variant::MappedCta2BankedA {}
impl GemmWsMapping for variant::Mapped {}
impl GemmWsMapping for variant::MappedWsBatched {}

impl<
        const A_ATOM_COLUMNS: usize,
        const A_PER_ELEMENT_SHIFT: u32,
        const A_OUTER_MASK: usize,
        const A_ATOM_SHIFT: u32,
        const B_ATOM_COLUMNS: usize,
        const B_PER_ELEMENT_SHIFT: u32,
        const B_OUTER_MASK: usize,
        const B_ATOM_SHIFT: u32,
        const REUSE_A_AS_B: bool,
    > sealed::GemmMapping
    for variant::CanonicalBf16SsCta1<
        A_ATOM_COLUMNS,
        A_PER_ELEMENT_SHIFT,
        A_OUTER_MASK,
        A_ATOM_SHIFT,
        B_ATOM_COLUMNS,
        B_PER_ELEMENT_SHIFT,
        B_OUTER_MASK,
        B_ATOM_SHIFT,
        REUSE_A_AS_B,
    >
{
}

impl<
        Mode: GemmMode,
        const A_ATOM_COLUMNS: usize,
        const A_PER_ELEMENT_SHIFT: u32,
        const A_OUTER_MASK: usize,
        const A_ATOM_SHIFT: u32,
        const B_ATOM_COLUMNS: usize,
        const B_PER_ELEMENT_SHIFT: u32,
        const B_OUTER_MASK: usize,
        const B_ATOM_SHIFT: u32,
        const REUSE_A_AS_B: bool,
    > GemmMapping<Mode>
    for variant::CanonicalBf16SsCta1<
        A_ATOM_COLUMNS,
        A_PER_ELEMENT_SHIFT,
        A_OUTER_MASK,
        A_ATOM_SHIFT,
        B_ATOM_COLUMNS,
        B_PER_ELEMENT_SHIFT,
        B_OUTER_MASK,
        B_ATOM_SHIFT,
        REUSE_A_AS_B,
    >
{
    /// GEMM-mode operands followed by runtime destination `(tcol, allocation)`.
    type Args = (Mode::Args, R<i64>, R<i64>);
    const WS_BATCHED: bool = false;
    const CTA2_BANKED_A: bool = false;
}

impl<
        const A_ATOM_COLUMNS: usize,
        const A_PER_ELEMENT_SHIFT: u32,
        const A_OUTER_MASK: usize,
        const A_ATOM_SHIFT: u32,
        const B_ATOM_COLUMNS: usize,
        const B_PER_ELEMENT_SHIFT: u32,
        const B_OUTER_MASK: usize,
        const B_ATOM_SHIFT: u32,
        const REUSE_A_AS_B: bool,
    > GemmRegularMapping
    for variant::CanonicalBf16SsCta1<
        A_ATOM_COLUMNS,
        A_PER_ELEMENT_SHIFT,
        A_OUTER_MASK,
        A_ATOM_SHIFT,
        B_ATOM_COLUMNS,
        B_PER_ELEMENT_SHIFT,
        B_OUTER_MASK,
        B_ATOM_SHIFT,
        REUSE_A_AS_B,
    >
{
}

/// Every static fact a GEMM issue path needs, erased from the variant type
/// into values.
///
/// This is what lets [`GemmMappingForm`] be selected by the qualifiers that
/// actually gate a mapping instead of by the whole variant: a bound naming the
/// variant would name the mapping the variant carries, which is the mapping
/// being bounded.
#[derive(Clone, Copy)]
struct GemmDescriptor {
    config: MappedGemmConfig,
    a_input: GemmInputSpec,
    b_input: GemmInputSpec,
    scale: GemmInputSpec,
}

impl GemmDescriptor {
    #[inline(always)]
    fn from_variant<V: StaticGemm>(weight_stationary: bool) -> Self
    where
        V::Mapping: GemmMapping<V::Mode>,
    {
        Self {
            config: MappedGemmConfig::from_variant::<V>(weight_stationary),
            a_input: V::AInput::SPEC,
            b_input: V::BInput::SPEC,
            scale: <V::Mode as GemmModeForm>::Scale::SPEC,
        }
    }
}

/// Asserts a mapping is legal for a variant's instruction qualifiers.
///
/// These are const-generic, so only the variant's own impl can state them --
/// a caller would have to pass `V::CTA_GROUP` and friends as const-generic
/// arguments, which associated constants cannot be. Keeping the qualifier
/// guard here and off [`GemmMappingForm`] is what lets a caller name a
/// mapping without naming the variant that carries it.
trait GemmMappingQualifiers<
    const CTA_GROUP: u32,
    const TRANS_A: bool,
    const TRANS_B: bool,
    const SCALE_VECTOR: usize,
>
{
}

/// `Mapped` interprets the operand maps directly, so it is legal everywhere.
impl<const CTA_GROUP: u32, const TRANS_A: bool, const TRANS_B: bool, const SCALE_VECTOR: usize>
    GemmMappingQualifiers<CTA_GROUP, TRANS_A, TRANS_B, SCALE_VECTOR> for variant::Mapped
{
}

impl<const TRANS_B: bool> GemmMappingQualifiers<1, false, TRANS_B, 1> for variant::MappedWsBatched {}

impl<const TRANS_B: bool> GemmMappingQualifiers<2, false, TRANS_B, 1>
    for variant::MappedCta2BankedA
{
}

/// The canonical BF16 shared/shared form is proven only for `cta_group=1`,
/// untransposed operands, and an unscaled (`SCALE_VECTOR = 1`) instruction.
impl<
        const A_ATOM_COLUMNS: usize,
        const A_PER_ELEMENT_SHIFT: u32,
        const A_OUTER_MASK: usize,
        const A_ATOM_SHIFT: u32,
        const B_ATOM_COLUMNS: usize,
        const B_PER_ELEMENT_SHIFT: u32,
        const B_OUTER_MASK: usize,
        const B_ATOM_SHIFT: u32,
        const REUSE_A_AS_B: bool,
    > GemmMappingQualifiers<1, false, false, 1>
    for variant::CanonicalBf16SsCta1<
        A_ATOM_COLUMNS,
        A_PER_ELEMENT_SHIFT,
        A_OUTER_MASK,
        A_ATOM_SHIFT,
        B_ATOM_COLUMNS,
        B_PER_ELEMENT_SHIFT,
        B_OUTER_MASK,
        B_ATOM_SHIFT,
        REUSE_A_AS_B,
    >
{
}

/// Closed dispatch from a GEMM mapping to the path that issues it.
///
/// The parameters are exactly the variant qualifiers that gate which mappings
/// are legal — everything else a mapping needs arrives as a [`GemmDescriptor`]
/// value. Selecting on qualifiers rather than on the variant keeps the bound
/// free of the mapping it constrains.
trait GemmMappingForm<Mode: GemmModeForm, AInput: GemmInput, BInput: GemmInput, APlace: GemmAEntry>:
    GemmMapping<Mode>
{
    fn issue(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<super::Tmem>,
        a: &MappedView<<APlace as GemmAPlacement>::Space>,
        b: &MappedView<Shared>,
        args: Self::Args,
        descriptor: GemmDescriptor,
    ) -> Result<(), EngineError>;
}

macro_rules! mapped_gemm_forms {
    ($($mapping:ty),+ $(,)?) => {
        $(
            impl<Mode: GemmModeForm, AInput: GemmInput, BInput: GemmInput, APlace: GemmAEntry>
                GemmMappingForm<Mode, AInput, BInput, APlace> for $mapping
            {
                fn issue(
                    warp: &mut super::Engine,
                    context: ExecCtx,
                    site: SiteId,
                    destination: &MappedView<super::Tmem>,
                    a: &MappedView<<APlace as GemmAPlacement>::Space>,
                    b: &MappedView<Shared>,
                    args: Self::Args,
                    descriptor: GemmDescriptor,
                ) -> Result<(), EngineError> {
                    <APlace as GemmAEntry>::execute_mapped(
                        warp,
                        context,
                        site,
                        destination,
                        a,
                        b,
                        Mode::split(args),
                        descriptor.config,
                        descriptor.a_input,
                        descriptor.b_input,
                        descriptor.scale,
                    )
                }
            }
        )+
    };
}

mapped_gemm_forms!(
    variant::Mapped,
    variant::MappedWsBatched,
    variant::MappedCta2BankedA,
);

#[derive(Clone, Copy)]
struct CanonicalBf16GemmConfig {
    m: usize,
    n: usize,
    k: usize,
    access: crate::TmemAccessMode,
    pipeline_class: crate::runtime::TcgenMmaPipelineClass,
    a_atom_columns: usize,
    a_per_element_shift: u32,
    a_outer_mask: usize,
    a_atom_shift: u32,
    b_atom_columns: usize,
    b_per_element_shift: u32,
    b_outer_mask: usize,
    b_atom_shift: u32,
    reuse_a_as_b: bool,
}

#[allow(clippy::too_many_arguments)]
fn issue_canonical_bf16_ss_cta1_gemm(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    destination: &MappedView<super::Tmem>,
    a: &MappedView<Shared>,
    b: &MappedView<Shared>,
    accumulate: R<bool>,
    destination_base_tcol: R<i64>,
    destination_allocated_addr: R<i64>,
    config: CanonicalBf16GemmConfig,
) -> Result<(), EngineError> {
    if !matches!(config.m, 64 | 128) || config.n < 64 || config.k < 64 {
        return Err(EngineError::message(
            "canonical BF16 shared/shared GEMM requires dense non-transposed CTA1 M64/M128 with N,K >= 64",
        ));
    }
    let Some(lane) = context.active_mask().first_active() else {
        return Ok(());
    };
    if accumulate[lane] {
        return Err(EngineError::message(
            "canonical BF16 shared/shared GEMM does not support accumulation",
        ));
    }
    let issue_mask = LaneMask::single(lane)?;
    let issue_context = ExecCtx::from_inner(
        context
            .into_inner()
            .with_active_mask(issue_mask.into_inner()),
    );
    engine(warp).discard_mma_collectors(lane, false);
    let operation = begin(warp, issue_context, site, OperationKind::TcgenWork, false)?;

    let physical = engine(warp).kernel().physical().clone();
    let lifecycle = engine(warp).kernel().services().tcgen();
    let inner = issue_context.into_inner();
    let a_buffer = a.allocation().inner().clone();
    let b_buffer = b.allocation().inner().clone();
    let destination_buffer = destination.allocation().inner().clone();
    let a_logical_buffer = a.allocation().logical_buffer().unwrap_or("gemm_A");
    let b_logical_buffer = b.allocation().logical_buffer().unwrap_or("gemm_B");
    let destination_logical_buffer = destination
        .allocation()
        .logical_buffer()
        .unwrap_or("gemm_D");
    let a_footprint = [(
        lane,
        None,
        0_usize,
        crate::runtime::runtime_buffer_byte_len(&a_buffer),
    )];
    let b_footprint = [(
        lane,
        None,
        0_usize,
        crate::runtime::runtime_buffer_byte_len(&b_buffer),
    )];
    let row_bytes = config
        .n
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| EngineError::message("canonical GEMM destination row size overflows"))?;
    let base_tcol = destination_base_tcol[lane];
    let allocated_addr = destination_allocated_addr[lane];
    let target_cta = inner.cta_id_in_cluster();
    let mut destination_footprint = Vec::with_capacity(config.m);
    for row in 0..config.m {
        let mapped_row = if config.m == 128 {
            row
        } else {
            (row / 16) * 32 + row % 16
        };
        destination_footprint.push((
            lane,
            lane,
            Some(target_cta),
            i64::try_from(mapped_row)
                .map_err(|_| EngineError::message("canonical GEMM row exceeds i64"))?,
            base_tcol,
            allocated_addr,
            row_bytes,
        ));
    }
    let descriptor = crate::runtime::TileGemmBf16Descriptor::new(
        config.m,
        config.n,
        config.k,
        crate::runtime::TileGemmOperandLayout::new(
            config.a_atom_columns,
            config.a_per_element_shift,
            config.a_outer_mask,
            config.a_atom_shift,
        ),
        crate::runtime::TileGemmOperandLayout::new(
            config.b_atom_columns,
            config.b_per_element_shift,
            config.b_outer_mask,
            config.b_atom_shift,
        ),
        config.reuse_a_as_b,
    );
    let observes_operations = engine(warp).observes_operations();
    engine(warp).tcgen_instruction_issue(
        operation.as_ref(),
        1,
        crate::runtime::TcgenPipelineOperation::Mma,
        Some(config.pipeline_class),
        |record_regular, record_tmem| {
            record_regular(
                true,
                OperationKind::Load,
                &a_buffer,
                Some(a_logical_buffer),
                &a_footprint,
            )?;
            record_regular(
                false,
                OperationKind::Load,
                &b_buffer,
                Some(b_logical_buffer),
                &b_footprint,
            )?;
            record_tmem(
                OperationKind::Store,
                &destination_buffer,
                destination_logical_buffer,
                config.access,
                &destination_footprint,
            )
        },
        || {
            if observes_operations {
                crate::runtime::tile_gemm_bf16_f32_ss_cta1_increasing_k(
                    &physical,
                    &inner,
                    &lifecycle,
                    config.access,
                    &destination_buffer,
                    &a_buffer,
                    &b_buffer,
                    base_tcol,
                    allocated_addr,
                    descriptor,
                    lane,
                )?;
            } else {
                crate::runtime::tile_gemm_bf16_f32_ss_cta1(
                    &physical,
                    &inner,
                    &lifecycle,
                    config.access,
                    &destination_buffer,
                    &a_buffer,
                    &b_buffer,
                    base_tcol,
                    allocated_addr,
                    descriptor,
                    lane,
                )?;
            }
            Ok(())
        },
    )?;
    finish(warp, &operation)
}

impl<
        const A_ATOM_COLUMNS: usize,
        const A_PER_ELEMENT_SHIFT: u32,
        const A_OUTER_MASK: usize,
        const A_ATOM_SHIFT: u32,
        const B_ATOM_COLUMNS: usize,
        const B_PER_ELEMENT_SHIFT: u32,
        const B_OUTER_MASK: usize,
        const B_ATOM_SHIFT: u32,
        const REUSE_A_AS_B: bool,
    >
    GemmMappingForm<
        variant::Dense,
        super::reg::variant::Bf16,
        super::reg::variant::Bf16,
        variant::AShared,
    >
    for variant::CanonicalBf16SsCta1<
        A_ATOM_COLUMNS,
        A_PER_ELEMENT_SHIFT,
        A_OUTER_MASK,
        A_ATOM_SHIFT,
        B_ATOM_COLUMNS,
        B_PER_ELEMENT_SHIFT,
        B_OUTER_MASK,
        B_ATOM_SHIFT,
        REUSE_A_AS_B,
    >
{
    fn issue(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<super::Tmem>,
        a: &MappedView<Shared>,
        b: &MappedView<Shared>,
        (accumulate, destination_base_tcol, destination_allocated_addr): Self::Args,
        descriptor: GemmDescriptor,
    ) -> Result<(), EngineError> {
        if descriptor.config.weight_stationary {
            return Err(EngineError::message(
                "canonical BF16 shared/shared mapping is not a tcgen05.mma.ws variant",
            ));
        }
        issue_canonical_bf16_ss_cta1_gemm(
            warp,
            context,
            site,
            destination,
            a,
            b,
            accumulate,
            destination_base_tcol,
            destination_allocated_addr,
            CanonicalBf16GemmConfig {
                m: descriptor.config.m,
                n: descriptor.config.n,
                k: descriptor.config.k,
                access: descriptor.config.access,
                pipeline_class: descriptor.config.pipeline_class.ok_or_else(|| {
                    EngineError::message(
                        "canonical BF16 shared/shared GEMM has no TCGEN MMA pipeline class",
                    )
                })?,
                a_atom_columns: A_ATOM_COLUMNS,
                a_per_element_shift: A_PER_ELEMENT_SHIFT,
                a_outer_mask: A_OUTER_MASK,
                a_atom_shift: A_ATOM_SHIFT,
                b_atom_columns: B_ATOM_COLUMNS,
                b_per_element_shift: B_PER_ELEMENT_SHIFT,
                b_outer_mask: B_OUTER_MASK,
                b_atom_shift: B_ATOM_SHIFT,
                reuse_a_as_b: REUSE_A_AS_B,
            },
        )
    }
}

trait WarpGemmInput: GemmInput {
    fn execute_mapped(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<Register>,
        a: &MappedView<Register>,
        b: &MappedView<Register>,
        c: &MappedView<Register>,
        config: WarpGemmConfig,
    ) -> Result<(), EngineError>;

    fn execute_canonical(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<Register>,
        a: &MappedView<Register>,
        b: &MappedView<Register>,
        c: &MappedView<Register>,
        mma_k: usize,
        accumulate: bool,
    ) -> Result<(), EngineError>;
}

macro_rules! warp_gemm_input_entry {
    ($input:ty) => {
        impl WarpGemmInput for $input {
            fn execute_mapped(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                destination: &MappedView<Register>,
                a: &MappedView<Register>,
                b: &MappedView<Register>,
                c: &MappedView<Register>,
                config: WarpGemmConfig,
            ) -> Result<(), EngineError> {
                execute_mapped_warp_gemm::<$input>(
                    warp,
                    context,
                    site,
                    destination,
                    a,
                    b,
                    c,
                    config,
                )
            }

            fn execute_canonical(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                destination: &MappedView<Register>,
                a: &MappedView<Register>,
                b: &MappedView<Register>,
                c: &MappedView<Register>,
                mma_k: usize,
                accumulate: bool,
            ) -> Result<(), EngineError> {
                execute_canonical_mma_m16n8::<$input>(
                    warp,
                    context,
                    site,
                    destination,
                    a,
                    b,
                    c,
                    mma_k,
                    accumulate,
                )
            }
        }
    };
}

warp_gemm_input_entry!(super::reg::variant::F16);
warp_gemm_input_entry!(super::reg::variant::Bf16);

trait StaticWarpGemm {
    type Input: WarpGemmInput;
    type Mapping;
    const M: usize;
    const N: usize;
    const K: usize;
    const MMA_K: usize;
    const TRANS_A: bool;
    const TRANS_B: bool;
    const ACCUMULATE: bool;
}

impl<
        Input,
        const M: usize,
        const N: usize,
        const K: usize,
        const MMA_K: usize,
        const TRANS_A: bool,
        const TRANS_B: bool,
        const ACCUMULATE: bool,
        Mapping,
    > StaticWarpGemm
    for variant::MmaSync<Input, M, N, K, MMA_K, TRANS_A, TRANS_B, ACCUMULATE, Mapping>
where
    Input: WarpGemmInput,
{
    type Input = Input;
    type Mapping = Mapping;
    const M: usize = M;
    const N: usize = N;
    const K: usize = K;
    const MMA_K: usize = MMA_K;
    const TRANS_A: bool = TRANS_A;
    const TRANS_B: bool = TRANS_B;
    const ACCUMULATE: bool = ACCUMULATE;
}

#[derive(Clone, Copy)]
struct WarpGemmConfig {
    m: usize,
    n: usize,
    k: usize,
    mma_k: usize,
    trans_a: bool,
    trans_b: bool,
    accumulate: bool,
}

impl WarpGemmConfig {
    #[inline(always)]
    fn from_variant<V: StaticWarpGemm>() -> Self {
        Self {
            m: V::M,
            n: V::N,
            k: V::K,
            mma_k: V::MMA_K,
            trans_a: V::TRANS_A,
            trans_b: V::TRANS_B,
            accumulate: V::ACCUMULATE,
        }
    }
}

trait WarpGemmMappingForm<V>: sealed::WarpGemmMapping
where
    V: StaticWarpGemm,
{
    fn issue(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<Register>,
        a: &MappedView<Register>,
        b: &MappedView<Register>,
        c: &MappedView<Register>,
    ) -> Result<(), EngineError>;
}

impl sealed::WarpGemmMapping for variant::Mapped {}
impl<V> WarpGemmMappingForm<V> for variant::Mapped
where
    V: StaticWarpGemm,
{
    fn issue(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<Register>,
        a: &MappedView<Register>,
        b: &MappedView<Register>,
        c: &MappedView<Register>,
    ) -> Result<(), EngineError> {
        V::Input::execute_mapped(
            warp,
            context,
            site,
            destination,
            a,
            b,
            c,
            WarpGemmConfig::from_variant::<V>(),
        )
    }
}

impl<
        Input,
        const M: usize,
        const N: usize,
        const K: usize,
        const MMA_K: usize,
        const TRANS_A: bool,
        const TRANS_B: bool,
        const ACCUMULATE: bool,
        Mapping,
    > warp_gemm_spec::sealed::Sealed
    for variant::MmaSync<Input, M, N, K, MMA_K, TRANS_A, TRANS_B, ACCUMULATE, Mapping>
where
    Input: WarpGemmInput,
    Mapping: WarpGemmMappingForm<
        variant::MmaSync<Input, M, N, K, MMA_K, TRANS_A, TRANS_B, ACCUMULATE, Mapping>,
    >,
{
}

impl<
        Input,
        const M: usize,
        const N: usize,
        const K: usize,
        const MMA_K: usize,
        const TRANS_A: bool,
        const TRANS_B: bool,
        const ACCUMULATE: bool,
        Mapping,
    > warp_gemm_spec::Variant
    for variant::MmaSync<Input, M, N, K, MMA_K, TRANS_A, TRANS_B, ACCUMULATE, Mapping>
where
    Input: WarpGemmInput,
    Mapping: WarpGemmMappingForm<
        variant::MmaSync<Input, M, N, K, MMA_K, TRANS_A, TRANS_B, ACCUMULATE, Mapping>,
    >,
{
    type Destination = Register;
    type A = Register;
    type B = Register;
    type C = Register;
    type Output = ();
}

impl<
        Input,
        const M: usize,
        const N: usize,
        const K: usize,
        const MMA_K: usize,
        const TRANS_A: bool,
        const TRANS_B: bool,
        const ACCUMULATE: bool,
        Mapping,
    > warp_gemm_spec::sealed::Execute<super::Engine>
    for variant::MmaSync<Input, M, N, K, MMA_K, TRANS_A, TRANS_B, ACCUMULATE, Mapping>
where
    Input: WarpGemmInput,
    Mapping: WarpGemmMappingForm<
        variant::MmaSync<Input, M, N, K, MMA_K, TRANS_A, TRANS_B, ACCUMULATE, Mapping>,
    >,
{
    #[inline(always)]
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<Register>,
        a: &MappedView<Register>,
        b: &MappedView<Register>,
        c: &MappedView<Register>,
    ) -> Result<Self::Output, EngineError> {
        <Mapping as WarpGemmMappingForm<Self>>::issue(warp, context, site, destination, a, b, c)
    }
}

#[inline(never)]
fn execute_mapped_warp_gemm<Input>(
    warp: &mut super::Engine,
    context: ExecCtx,
    _site: SiteId,
    destination: &MappedView<Register>,
    a: &MappedView<Register>,
    b: &MappedView<Register>,
    c: &MappedView<Register>,
    config: WarpGemmConfig,
) -> Result<(), EngineError>
where
    Input: WarpGemmInput,
{
    crate::runtime::require_full_warp_sync(context.active_mask().into_inner(), "tirx.tile.gemm")
        .map_err(EngineError::from)?;
    if config.m == 0
        || config.n == 0
        || config.k == 0
        || config.m % 16 != 0
        || config.n % 8 != 0
        || !matches!(config.mma_k, 8 | 16)
        || config.k % config.mma_k != 0
    {
        return Err(EngineError::message(format!(
            "typed warp GEMM ({}, {}, {}) is not covered by mma.sync.m16n8k{}",
            config.m, config.n, config.k, config.mma_k,
        )));
    }

    let a_elements =
        map_register_gemm_matrix(a, context, config.m, config.k, config.trans_a, "gemm A map")?;
    let b_elements = map_register_gemm_matrix(
        b,
        context,
        config.n,
        config.k,
        !config.trans_b,
        "gemm B map",
    )?;
    let d_elements = map_register_gemm_matrix(
        destination,
        context,
        config.m,
        config.n,
        false,
        "gemm D map",
    )?;
    let c_elements = if config.accumulate {
        Some(map_register_gemm_matrix(
            c,
            context,
            config.m,
            config.n,
            false,
            "gemm C map",
        )?)
    } else {
        None
    };

    let physical = engine(warp).kernel().physical().clone();
    let lifecycle = engine(warp).kernel().services().tcgen();
    let inner = context.into_inner();
    let a_buffer = a.allocation().inner().clone();
    let b_buffer = b.allocation().inner().clone();
    let c_buffer = c.allocation().inner().clone();
    let d_buffer = destination.allocation().inner().clone();

    let mut a_values = Vec::with_capacity(a_elements.len());
    for &element in &a_elements {
        a_values.push(read_gemm_value::<Input>(
            &physical,
            &inner,
            &lifecycle,
            crate::TmemAccessMode::Static,
            &a_buffer,
            element,
            false,
            "gemm A",
        )?);
    }

    let mut b_values = Vec::with_capacity(b_elements.len());
    for &element in &b_elements {
        b_values.push(read_gemm_value::<Input>(
            &physical,
            &inner,
            &lifecycle,
            crate::TmemAccessMode::Static,
            &b_buffer,
            element,
            false,
            "gemm B",
        )?);
    }

    let mut c_values = Vec::new();
    if let Some(elements) = c_elements.as_deref() {
        c_values.reserve(elements.len());
        for &element in elements {
            c_values.push(read_gemm_value::<super::reg::variant::F32>(
                &physical,
                &inner,
                &lifecycle,
                crate::TmemAccessMode::Static,
                &c_buffer,
                element,
                false,
                "gemm C",
            )?);
        }
    }
    let input_d = config.accumulate.then_some((c_values.as_slice(), 1.0_f32));
    let output = crate::runtime::mma_f32_abt_increasing_k(
        config.m, config.n, config.k, &a_values, &b_values, input_d,
    )
    .map_err(EngineError::from)?;
    for (&element, &value) in d_elements.iter().zip(&output) {
        write_regular::<f32>(&physical, &inner, &d_buffer, element, value)?;
    }

    Ok(())
}

#[inline(never)]
fn execute_canonical_mma_m16n8<Input>(
    warp: &mut super::Engine,
    context: ExecCtx,
    _site: SiteId,
    destination: &MappedView<Register>,
    a: &MappedView<Register>,
    b: &MappedView<Register>,
    c: &MappedView<Register>,
    mma_k: usize,
    accumulate: bool,
) -> Result<(), EngineError>
where
    Input: WarpGemmInput,
{
    crate::runtime::require_full_warp_sync(context.active_mask().into_inner(), "tirx.tile.gemm")
        .map_err(EngineError::from)?;
    if !matches!(mma_k, 8 | 16) {
        return Err(EngineError::message(format!(
            "canonical warp GEMM must be one mma.sync.m16n8k{{8,16}}, got instruction K={mma_k}",
        )));
    }

    let physical = engine(warp).kernel().physical().clone();
    let inner = context.into_inner();
    let a_physical = crate::runtime::load_warp_mma_m16n8_fragment::<<Input as GemmInput>::Storage>(
        &physical,
        &inner,
        a.allocation().inner(),
        crate::runtime::WarpMmaFragmentRole::A,
        mma_k,
    )?;
    let a_values = a_physical
        .into_iter()
        .map(|value| <Input as GemmInput>::decode(value, 0))
        .collect::<Result<Vec<_>, _>>()?;
    let b_physical = crate::runtime::load_warp_mma_m16n8_fragment::<<Input as GemmInput>::Storage>(
        &physical,
        &inner,
        b.allocation().inner(),
        crate::runtime::WarpMmaFragmentRole::B,
        mma_k,
    )?;
    let b_values = b_physical
        .into_iter()
        .map(|value| <Input as GemmInput>::decode(value, 0))
        .collect::<Result<Vec<_>, _>>()?;
    let c_values = if accumulate {
        crate::runtime::load_warp_mma_m16n8_fragment::<f32>(
            &physical,
            &inner,
            c.allocation().inner(),
            crate::runtime::WarpMmaFragmentRole::C,
            mma_k,
        )?
    } else {
        Vec::new()
    };
    let input_d = accumulate.then_some((c_values.as_slice(), 1.0_f32));
    let output =
        crate::runtime::mma_f32_abt_increasing_k(16, 8, mma_k, &a_values, &b_values, input_d)
            .map_err(EngineError::from)?;
    crate::runtime::store_warp_mma_m16n8_f32_fragment(
        &physical,
        &inner,
        destination.allocation().inner(),
        &output,
    )?;
    Ok(())
}

impl sealed::WarpGemmMapping for variant::CanonicalMmaM16N8 {}
impl<
        Input,
        const MMA_K: usize,
        const TRANS_A: bool,
        const TRANS_B: bool,
        const ACCUMULATE: bool,
    >
    WarpGemmMappingForm<
        variant::MmaSync<
            Input,
            16,
            8,
            MMA_K,
            MMA_K,
            TRANS_A,
            TRANS_B,
            ACCUMULATE,
            variant::CanonicalMmaM16N8,
        >,
    > for variant::CanonicalMmaM16N8
where
    Input: WarpGemmInput,
{
    fn issue(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        destination: &MappedView<Register>,
        a: &MappedView<Register>,
        b: &MappedView<Register>,
        c: &MappedView<Register>,
    ) -> Result<(), EngineError> {
        Input::execute_canonical(warp, context, site, destination, a, b, c, MMA_K, ACCUMULATE)
    }
}

/// Static `tcgen05.mma` specialization accepted by [`gemm_async`].
#[allow(private_bounds)]
pub trait GemmAsyncVariant: sealed::Gemm {
    type Args;
    type A: MemorySpace;
}

macro_rules! impl_public_mapped_gemm_variant {
    ($sealed:ident, $public:ident) => {
        impl<
                Mode,
                AInput,
                BInput,
                APlace,
                Access,
                const M: usize,
                const N: usize,
                const K: usize,
                const INSTR_M: usize,
                const INSTR_N: usize,
                const INSTR_K: usize,
                const CTA_GROUP: u32,
                const TRANS_A: bool,
                const TRANS_B: bool,
                const SCALE_VECTOR: usize,
                const EXPECTED_DESCRIPTOR: u32,
                const DESCRIPTOR_MASK: u32,
                Mapping,
            > sealed::$sealed
            for variant::Gemm<
                Mode,
                AInput,
                BInput,
                APlace,
                Access,
                M,
                N,
                K,
                INSTR_M,
                INSTR_N,
                INSTR_K,
                CTA_GROUP,
                TRANS_A,
                TRANS_B,
                SCALE_VECTOR,
                EXPECTED_DESCRIPTOR,
                DESCRIPTOR_MASK,
                Mapping,
            >
        where
            Mode: GemmModeForm,
            AInput: GemmInput,
            BInput: GemmInput,
            APlace: APlacement + GemmAEntry,
            Access: TileTmemAccess,
            Mapping: GemmMappingForm<Mode, AInput, BInput, APlace>
                + GemmMappingQualifiers<CTA_GROUP, TRANS_A, TRANS_B, SCALE_VECTOR>
                + GemmMapping<Mode>
                + GemmWsMapping,
        {
        }

        impl<
                Mode,
                AInput,
                BInput,
                APlace,
                Access,
                const M: usize,
                const N: usize,
                const K: usize,
                const INSTR_M: usize,
                const INSTR_N: usize,
                const INSTR_K: usize,
                const CTA_GROUP: u32,
                const TRANS_A: bool,
                const TRANS_B: bool,
                const SCALE_VECTOR: usize,
                const EXPECTED_DESCRIPTOR: u32,
                const DESCRIPTOR_MASK: u32,
                Mapping,
            > $public
            for variant::Gemm<
                Mode,
                AInput,
                BInput,
                APlace,
                Access,
                M,
                N,
                K,
                INSTR_M,
                INSTR_N,
                INSTR_K,
                CTA_GROUP,
                TRANS_A,
                TRANS_B,
                SCALE_VECTOR,
                EXPECTED_DESCRIPTOR,
                DESCRIPTOR_MASK,
                Mapping,
            >
        where
            Mode: GemmModeForm,
            AInput: GemmInput,
            BInput: GemmInput,
            APlace: APlacement + GemmAEntry,
            Access: TileTmemAccess,
            Mapping: GemmMappingForm<Mode, AInput, BInput, APlace>
                + GemmMappingQualifiers<CTA_GROUP, TRANS_A, TRANS_B, SCALE_VECTOR>
                + GemmMapping<Mode>
                + GemmWsMapping,
        {
            type Args = <Mapping as GemmMapping<Mode>>::Args;
            type A = <APlace as GemmAPlacement>::Space;
        }
    };
}

impl<
        Mode,
        AInput,
        BInput,
        APlace,
        Access,
        const M: usize,
        const N: usize,
        const K: usize,
        const INSTR_M: usize,
        const INSTR_N: usize,
        const INSTR_K: usize,
        const CTA_GROUP: u32,
        const TRANS_A: bool,
        const TRANS_B: bool,
        const SCALE_VECTOR: usize,
        const EXPECTED_DESCRIPTOR: u32,
        const DESCRIPTOR_MASK: u32,
        Mapping,
    > sealed::Gemm
    for variant::Gemm<
        Mode,
        AInput,
        BInput,
        APlace,
        Access,
        M,
        N,
        K,
        INSTR_M,
        INSTR_N,
        INSTR_K,
        CTA_GROUP,
        TRANS_A,
        TRANS_B,
        SCALE_VECTOR,
        EXPECTED_DESCRIPTOR,
        DESCRIPTOR_MASK,
        Mapping,
    >
where
    Mode: GemmModeForm,
    AInput: GemmInput,
    BInput: GemmInput,
    APlace: APlacement + GemmAEntry,
    Access: TileTmemAccess,
    Mapping: GemmMappingForm<Mode, AInput, BInput, APlace>
        + GemmMappingQualifiers<CTA_GROUP, TRANS_A, TRANS_B, SCALE_VECTOR>
        + GemmMapping<Mode>
        + GemmRegularMapping,
{
}

impl<
        Mode,
        AInput,
        BInput,
        APlace,
        Access,
        const M: usize,
        const N: usize,
        const K: usize,
        const INSTR_M: usize,
        const INSTR_N: usize,
        const INSTR_K: usize,
        const CTA_GROUP: u32,
        const TRANS_A: bool,
        const TRANS_B: bool,
        const SCALE_VECTOR: usize,
        const EXPECTED_DESCRIPTOR: u32,
        const DESCRIPTOR_MASK: u32,
        Mapping,
    > GemmAsyncVariant
    for variant::Gemm<
        Mode,
        AInput,
        BInput,
        APlace,
        Access,
        M,
        N,
        K,
        INSTR_M,
        INSTR_N,
        INSTR_K,
        CTA_GROUP,
        TRANS_A,
        TRANS_B,
        SCALE_VECTOR,
        EXPECTED_DESCRIPTOR,
        DESCRIPTOR_MASK,
        Mapping,
    >
where
    Mode: GemmModeForm,
    AInput: GemmInput,
    BInput: GemmInput,
    APlace: APlacement + GemmAEntry,
    Access: TileTmemAccess,
    Mapping: GemmMappingForm<Mode, AInput, BInput, APlace>
        + GemmMappingQualifiers<CTA_GROUP, TRANS_A, TRANS_B, SCALE_VECTOR>
        + GemmMapping<Mode>
        + GemmRegularMapping,
{
    type Args = <Mapping as GemmMapping<Mode>>::Args;
    type A = <APlace as GemmAPlacement>::Space;
}

/// The body `gemm_async` and `gemm_async_ws` share: build the variant's
/// descriptor and hand it to the mapping. The two differ only in the mnemonic
/// they name, which reaches the numeric core as `weight_stationary`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn issue_mapped_gemm_tile<V>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    destination: &MappedView<super::Tmem>,
    a: &MappedView<<V::APlace as GemmAPlacement>::Space>,
    b: &MappedView<Shared>,
    args: <V::Mapping as GemmMapping<V::Mode>>::Args,
    weight_stationary: bool,
) -> Result<(), EngineError>
where
    V: StaticGemm,
    V::Mapping: GemmMappingForm<V::Mode, V::AInput, V::BInput, V::APlace>,
{
    V::Mapping::issue(
        warp,
        context,
        site,
        destination,
        a,
        b,
        args,
        GemmDescriptor::from_variant::<V>(weight_stationary),
    )
}

/// Execute one complete logical tile with `tcgen05.mma` variants.
#[allow(private_bounds)]
#[inline(always)]
pub fn gemm_async<V>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    destination: &MappedView<super::Tmem>,
    a: &MappedView<V::A>,
    b: &MappedView<Shared>,
    args: V::Args,
) -> Result<(), EngineError>
where
    V: GemmAsyncVariant<A = <<V as StaticGemm>::APlace as GemmAPlacement>::Space> + StaticGemm,
    V::Mapping: GemmMappingForm<V::Mode, V::AInput, V::BInput, V::APlace>
        + GemmMapping<V::Mode, Args = <V as GemmAsyncVariant>::Args>,
{
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(context, site, std::any::type_name::<V>());
    issue_mapped_gemm_tile::<V>(warp, context, site, destination, a, b, args, false)
}

/// Static `tcgen05.mma.ws` specialization accepted by [`gemm_async_ws`].
#[allow(private_bounds)]
pub trait GemmAsyncWsVariant: sealed::GemmWs {
    type Args;
    type A: MemorySpace;
}

impl_public_mapped_gemm_variant!(GemmWs, GemmAsyncWsVariant);

/// Execute one complete logical tile with `tcgen05.mma.ws`. It is a distinct
/// function because WS is a distinct PTX mnemonic, not a runtime flag.
#[allow(private_bounds)]
#[inline(always)]
pub fn gemm_async_ws<V>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    destination: &MappedView<super::Tmem>,
    a: &MappedView<V::A>,
    b: &MappedView<Shared>,
    args: V::Args,
) -> Result<(), EngineError>
where
    V: GemmAsyncWsVariant<
            Args = <<V as StaticGemm>::Mode as GemmMode>::Args,
            A = <<V as StaticGemm>::APlace as GemmAPlacement>::Space,
        > + StaticGemm,
    V::Mapping: GemmMappingForm<V::Mode, V::AInput, V::BInput, V::APlace>
        + GemmMapping<V::Mode, Args = <<V as StaticGemm>::Mode as GemmMode>::Args>,
{
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(context, site, std::any::type_name::<V>());
    issue_mapped_gemm_tile::<V>(warp, context, site, destination, a, b, args, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    type Shape = variant::Shape2<4, 8>;

    #[test]
    fn static_shape_is_the_complete_logical_tile_not_a_repeat_count() {
        assert_eq!(Shape::EXTENTS, &[4, 8]);
        assert_eq!(element_count::<Shape>().unwrap(), 32);
        assert_eq!(coordinates::<Shape>(19).unwrap(), vec![2, 3]);
    }
}
