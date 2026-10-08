//! Engine implementation of v2 async-copy instruction specializations.

use super::instruction::{
    async_instruction, instruction_variant, sync_instruction, sync_instruction_generic_args,
};
use super::mode_axis::for_each_engine_mode;
use super::transport::engine;
use super::{
    Address, EngineError, ExecCtx, Global, LaneMask, R, Shared, SiteId, TensorMapHandle,
    WarpHandle, begin, finish,
};
use crate::runtime::PtxStateSpace;
use crate::{AsyncGroupDomain, MemoryAccessSemantics, OperationKind};

sync_instruction_generic_args!(cp_async_spec, CpAsyncVariant, cp_async);
async_instruction!(
    cp_async_wait_group_spec,
    CpAsyncWaitGroupVariant,
    cp_async_wait_group
);
sync_instruction!(
    cp_async_mbarrier_arrive_spec,
    CpAsyncMbarrierArriveVariant,
    cp_async_mbarrier_arrive
);
sync_instruction!(cp_async_bulk_spec, CpAsyncBulkVariant, cp_async_bulk);
sync_instruction!(
    cp_reduce_async_bulk_spec,
    CpReduceAsyncBulkVariant,
    cp_reduce_async_bulk
);
sync_instruction!(st_async_spec, StAsyncVariant, st_async);
sync_instruction!(red_async_spec, RedAsyncVariant, red_async);
async_instruction!(
    cp_async_bulk_wait_group_spec,
    CpAsyncBulkWaitGroupVariant,
    cp_async_bulk_wait_group
);
sync_instruction!(
    cp_async_bulk_tensor_spec,
    CpAsyncBulkTensorVariant,
    cp_async_bulk_tensor
);
sync_instruction!(
    cp_reduce_async_bulk_tensor_spec,
    CpReduceAsyncBulkTensorVariant,
    cp_reduce_async_bulk_tensor
);
sync_instruction!(
    cp_async_bulk_prefetch_tensor_spec,
    CpAsyncBulkPrefetchTensorVariant,
    cp_async_bulk_prefetch_tensor
);
sync_instruction!(applypriority_spec, ApplyPriorityVariant, applypriority);
sync_instruction!(discard_spec, DiscardVariant, discard);

/// Resolve the ephemeral TensorMap used by override-qualified instructions.
#[allow(clippy::too_many_arguments)]
pub fn override_tensor_map<W: WarpHandle>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    tensor_map: TensorMapHandle,
    address: Address<Global>,
    dimensions: &[i64],
    lower_strides: &[i64],
    upper_strides: i64,
    coordinates: &[i64],
) -> Result<TensorMapHandle, EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(
        context,
        site,
        std::any::type_name_of_val(&override_tensor_map::<W>),
    );
    let context = context.into_inner();
    let result = (|| {
        let physical = engine(warp).kernel().physical();
        let view = address.inner().resolve_uniform_global_remainder_view(
            physical,
            &context,
            context.active_mask(),
        )?;
        tensor_map.inner().with_overrides(
            physical.global(),
            view,
            dimensions,
            lower_strides,
            upper_strides,
            coordinates,
        )
    })();
    match result {
        Ok(overridden) => Ok(TensorMapHandle::from_inner(std::sync::Arc::new(overridden))),
        Err(error) => {
            // Successful preparation is not another async issue. A failure
            // still belongs to this instruction, with its current loop/lane context.
            let operation = engine(warp).begin_current_operation(
                context,
                crate::StaticOpId::new(site.get()),
                OperationKind::AsyncIssue,
            )?;
            Err(error.with_operation_context(&operation).into())
        }
    }
}

/// Static PTX forms. Only qualifiers that change an instruction spelling are
/// represented here; runtime register operands remain in `Args`.
pub mod variant {
    use std::marker::PhantomData;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct RedAsync<T, Op>(PhantomData<fn(T, Op)>);
    pub struct StRelease<Scope, const BITS: usize, const MMIO: bool>(PhantomData<Scope>);
    pub struct RedRelease<Scope, const BITS: usize, const MMIO: bool>(PhantomData<Scope>);
    pub struct BulkS2cReduce<T, Op, Scope>(PhantomData<fn(T, Op, Scope)>);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct NoFill;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ZeroFill;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SourceSize;

    /// One `cp.async` spelling. `BYTES` is a PTX immediate; `Fill` closes
    /// the optional source-size or copy-predicate operand.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CpAsync<const BYTES: usize, Fill = NoFill>(PhantomData<fn(Fill)>);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BulkG2sCluster<const REPORT_PATTERN: u32 = 0, Sem = super::super::mem::variant::Plain>(
        PhantomData<Sem>,
    );
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BulkG2sClusterMulticast<
        const REPORT_PATTERN: u32 = 0,
        Sem = super::super::mem::variant::Plain,
    >(PhantomData<Sem>);
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BulkG2sCta<const REPORT_PATTERN: u32 = 0, Sem = super::super::mem::variant::Plain>(
        PhantomData<Sem>,
    );
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BulkG2sCtaIgnoreOob<Sem = super::super::mem::variant::Plain>(PhantomData<Sem>);
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BulkSharedToCluster<Sem = super::super::mem::variant::Plain>(PhantomData<Sem>);
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BulkS2g<Sem = super::super::mem::variant::Plain>(PhantomData<Sem>);
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BulkS2gMasked<Sem = super::super::mem::variant::Plain>(PhantomData<Sem>);
    /// Non-tensor shared-to-global bulk reductions, using the shared element math.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BulkS2gReduce<T, Op, Scope>(PhantomData<fn(T, Op, Scope)>);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorG2sCta<const RANK: usize, const CTA_GROUP: u32, const REPORT: u32 = 0>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorG2sCluster<const RANK: usize, const CTA_GROUP: u32, const REPORT: u32 = 0>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorG2sClusterMulticast<
        const RANK: usize,
        const CTA_GROUP: u32,
        const REPORT: u32 = 0,
    >;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorGather4Cta<const CTA_GROUP: u32, const REPORT_PATTERN: u32 = 0>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorIm2col<
        const RANK: usize,
        const MODE: u32,
        const CTA_GROUP: u32,
        const MULTICAST: bool,
        const REPORT: u32 = 0,
    >;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorGather4Cluster<const CTA_GROUP: u32, const REPORT_PATTERN: u32 = 0>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorGather4ClusterMulticast<const CTA_GROUP: u32, const REPORT_PATTERN: u32 = 0>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorS2g<const RANK: usize, const MODE: u32 = 0>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorS2gReduce<const RANK: usize, Op, const MODE: u32 = 0>(PhantomData<fn(Op)>);
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorPrefetch<const RANK: usize>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorPrefetchGather4;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorPrefetchEvictLast<const RANK: usize>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorPrefetchEvictLastGather4;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorApplyPriority<const RANK: usize>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TensorApplyPriorityGather4;

    /// `applypriority{.global}.L2::evict_normal`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ApplyPriority;
    /// `discard{.global}.L2`, represented by one concrete indeterminate-value witness.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Discard;
    /// Non-tensor `applypriority.async.bulk...bulk_group`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BulkApplyPriority;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ReduceAdd;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ReduceMin;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ReduceMax;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ReduceInc;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ReduceDec;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ReduceAnd;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ReduceOr;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ReduceXor;

    /// `cp.async.wait_group N`. `N` is a plain instruction immediate the
    /// engine only range-checks, so it rides in `Args` instead of splitting
    /// the spelling into eight monomorphizations plus a dynamic twin.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CpAsyncWaitGroup;

    /// `cp.async.bulk.wait_group N` (full completion); `N` rides in `Args`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BulkWaitGroup;

    /// `cp.async.bulk.wait_group.read N` (read completion only). Read-only
    /// completion is a consumed semantic axis, so it stays a spelling.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BulkWaitGroupRead;

    /// Legacy `cp.async.mbarrier.arrive` form that increments pending arrivals.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CpAsyncMbarrierArrive;

    /// Legacy `cp.async.mbarrier.arrive.noinc` form.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CpAsyncMbarrierArriveNoInc;

    /// `st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct StAsyncClusterCompleteTxBytesU32x4;

    /// Raw `st.async.shared::cluster...u32.v4` with addresses already mapped by `mapa`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct StAsyncClusterMappedCompleteTxBytes<const WORDS: usize>;
}

impl st_async_spec::sealed::Sealed for variant::StAsyncClusterCompleteTxBytesU32x4 {}
macro_rules! async_release_variant {
    ($spec:ident, $variant:ident, $reduction:literal) => {
        instruction_variant! {
            [impl<S: super::mem::StaticScope, const BITS: usize, const MMIO: bool>]
            $spec, variant::$variant<S, BITS, MMIO>,
            (Address<Global>, R<u64>) => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                (address, value): Self::Args,
            ) -> Result<(), EngineError> {
                if !matches!(BITS, 8 | 16 | 32 | 64) {
                    return Err(EngineError::message("invalid async release bit width"));
                }
                let operation = begin(warp, context, site, OperationKind::AsyncIssue, true)?;
                engine(warp).async_release_issue(
                    operation.as_ref(),
                    &context.into_inner(),
                    address.inner(),
                    value.inner(),
                    BITS / 8,
                    S::VALUE,
                    MMIO,
                    $reduction,
                )?;
                finish(warp, &operation)
            }
        }
    };
}
async_release_variant!(st_async_spec, StRelease, false);
async_release_variant!(red_async_spec, RedRelease, true);
macro_rules! red_async_variant {
    ($scalar:ty, $marker:ident, $operation:ident) => {
        instruction_variant! {
            [impl] red_async_spec, variant::RedAsync<$scalar, super::mem::variant::$marker>,
            (Address<Shared>, R<$scalar>, Address<Shared>) => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                (address, value, barrier): Self::Args,
            ) -> Result<(), EngineError> {
                for lane in context.active_mask().into_inner() {
                    let issue_context = context.with_active_mask(LaneMask::single(lane)?);
                    let operation =
                        begin(warp, issue_context, site, OperationKind::AsyncIssue, true)?;
                    engine(warp).red_async_cluster_issue(
                        operation.as_ref(),
                        &issue_context.into_inner(),
                        address.inner(),
                        barrier.inner(),
                        value.inner(),
                        issue_context.active_mask().into_inner(),
                        crate::runtime::RawAtomicOperation::$operation,
                    )?;
                    finish(warp, &operation)?;
                }
                Ok(())
            }
        }
        instruction_variant! {
            [impl<Scope: super::mem::StaticScope>]
            cp_reduce_async_bulk_spec, variant::BulkS2cReduce<$scalar, super::mem::variant::$marker, Scope>,
            (Address<Shared>, Address<Shared>, R<i64>, Address<Shared>) => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                (destination, source, size, barrier): Self::Args,
            ) -> Result<(), EngineError> {
                for lane in context.active_mask().into_inner() {
                    let issue_context = context.with_active_mask(LaneMask::single(lane)?);
                    let operation =
                        begin(warp, issue_context, site, OperationKind::AsyncIssue, true)?;
                    engine(warp).raw_bulk_reduce_s2c_issue::<$scalar>(
                        operation.as_ref(),
                        &issue_context.into_inner(),
                        destination.inner(),
                        source.inner(),
                        size[lane],
                        issue_context.active_mask().into_inner(),
                        barrier.inner(),
                        crate::runtime::RawAtomicOperation::$operation,
                        Scope::VALUE,
                    )?;
                    finish(warp, &operation)?;
                }
                Ok(())
            }
        }
    };
}
red_async_variant!(u32, Add, Add);
red_async_variant!(i32, Add, Add);
red_async_variant!(u64, Add, Add);
red_async_variant!(u32, Minimum, Minimum);
red_async_variant!(i32, Minimum, Minimum);
red_async_variant!(u32, Maximum, Maximum);
red_async_variant!(i32, Maximum, Maximum);
red_async_variant!(u32, BitAnd, BitAnd);
red_async_variant!(u32, BitOr, BitOr);
red_async_variant!(u32, BitXor, BitXor);
red_async_variant!(u32, Increment, Increment);
red_async_variant!(u32, Decrement, Decrement);
impl st_async_spec::Variant for variant::StAsyncClusterCompleteTxBytesU32x4 {
    type Args = (Address<Shared>, Address<Shared>, R<i64>, [R<u32>; 4]);
    type Output = ();
}
impl st_async_spec::sealed::Execute for variant::StAsyncClusterCompleteTxBytesU32x4 {
    #[inline(always)]
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        let (destination, barrier, target_ranks, values) = args;
        let operation = begin(warp, context, site, OperationKind::AsyncIssue, true)?;
        let value_refs = values.each_ref().map(R::inner);
        engine(warp).st_async_cluster_u32x4_issue(
            operation.as_ref(),
            &context.into_inner(),
            destination.inner(),
            barrier.inner(),
            target_ranks.inner(),
            &value_refs,
            context.active_mask().into_inner(),
        )?;
        finish(warp, &operation)
    }
}

instruction_variant! {
    [impl<const WORDS: usize>] st_async_spec, variant::StAsyncClusterMappedCompleteTxBytes<WORDS>,
    (Address<Shared>, Address<Shared>, [R<u32>; WORDS]) => ();
    #[inline(always)]
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        let (destination, barrier, values) = args;
        let operation = begin(warp, context, site, OperationKind::AsyncIssue, true)?;
        let value_refs = values.each_ref().map(R::inner);
        engine(warp).st_async_cluster_mapped_words_issue(
            operation.as_ref(),
            &context.into_inner(),
            destination.inner(),
            barrier.inner(),
            &value_refs,
            context.active_mask().into_inner(),
        )?;
        finish(warp, &operation)
    }
}

mod cp_async_operands {
    use super::*;

    pub trait Fill: 'static {
        type Args;
        fn split<const BYTES: usize>(
            args: Self::Args,
        ) -> (Address<Shared>, Address<Global>, R<u32>);
    }

    impl Fill for variant::NoFill {
        type Args = (Address<Shared>, Address<Global>);

        fn split<const BYTES: usize>(
            (destination, source): Self::Args,
        ) -> (Address<Shared>, Address<Global>, R<u32>) {
            (destination, source, R::splat(BYTES as u32))
        }
    }

    impl Fill for variant::ZeroFill {
        type Args = (Address<Shared>, Address<Global>, R<bool>);

        fn split<const BYTES: usize>(
            (destination, source, predicate): Self::Args,
        ) -> (Address<Shared>, Address<Global>, R<u32>) {
            let sizes = predicate.map(|_, copy| if copy { BYTES as u32 } else { 0 });
            (destination, source, sizes)
        }
    }

    impl Fill for variant::SourceSize {
        type Args = (Address<Shared>, Address<Global>, R<u32>);

        fn split<const BYTES: usize>(args: Self::Args) -> Self::Args {
            args
        }
    }
}

struct CpAsyncBytes<const VALUE: usize>;
trait ValidCpAsyncBytes {}
impl ValidCpAsyncBytes for CpAsyncBytes<4> {}
impl ValidCpAsyncBytes for CpAsyncBytes<8> {}
impl ValidCpAsyncBytes for CpAsyncBytes<16> {}

struct TmaRank<const VALUE: usize>;
trait ValidTmaRank {}
impl ValidTmaRank for TmaRank<1> {}
impl ValidTmaRank for TmaRank<2> {}
impl ValidTmaRank for TmaRank<3> {}
impl ValidTmaRank for TmaRank<4> {}
impl ValidTmaRank for TmaRank<5> {}

struct TmaCtaGroup<const VALUE: u32>;
trait ValidTmaCtaGroup {}
impl ValidTmaCtaGroup for TmaCtaGroup<1> {}
impl ValidTmaCtaGroup for TmaCtaGroup<2> {}

impl<const BYTES: usize, Fill> cp_async_spec::sealed::Sealed for variant::CpAsync<BYTES, Fill>
where
    Fill: cp_async_operands::Fill,
    CpAsyncBytes<BYTES>: ValidCpAsyncBytes,
{
}

impl<const BYTES: usize, Fill> cp_async_spec::Variant for variant::CpAsync<BYTES, Fill>
where
    Fill: cp_async_operands::Fill,
    CpAsyncBytes<BYTES>: ValidCpAsyncBytes,
{
    type Output = ();
}

#[inline(never)]
fn cp_async_entry<W: WarpHandle>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    destination: Address<Shared>,
    source: Address<Global>,
    source_sizes: R<u32>,
    bytes: usize,
) -> Result<(), EngineError> {
    let operation = begin(warp, context, site, OperationKind::AsyncIssue, true)?;
    engine(warp).raw_cp_async_issue(
        operation.as_ref(),
        &context.into_inner(),
        destination.inner(),
        source.inner(),
        source_sizes.inner(),
        context.active_mask().into_inner(),
        bytes,
    )?;
    finish(warp, &operation)
}

// One concrete copy entry per (byte width, fill) per engine mode.
macro_rules! cp_async_semantic_entry_for_mode {
    ($warp:ty, $bytes:literal, $fill:ty) => {
        const _: () = {
            #[inline(never)]
            fn entry(
                warp: &mut $warp,
                context: ExecCtx,
                site: SiteId,
                destination: Address<Shared>,
                source: Address<Global>,
                source_sizes: R<u32>,
            ) -> Result<(), EngineError> {
                cp_async_entry(
                    warp,
                    context,
                    site,
                    destination,
                    source,
                    source_sizes,
                    $bytes,
                )
            }

            impl cp_async_spec::sealed::Execute<$warp, <$fill as cp_async_operands::Fill>::Args>
                for variant::CpAsync<$bytes, $fill>
            {
                #[inline(always)]
                fn execute(
                    warp: &mut $warp,
                    context: ExecCtx,
                    site: SiteId,
                    args: <$fill as cp_async_operands::Fill>::Args,
                ) -> Result<Self::Output, EngineError> {
                    let (destination, source, source_sizes) =
                        <$fill as cp_async_operands::Fill>::split::<$bytes>(args);
                    entry(warp, context, site, destination, source, source_sizes)
                }
            }
        };
    };
}

macro_rules! cp_async_entries_for_mode {
    ($warp:ty) => {
        cp_async_semantic_entry_for_mode!($warp, 4, variant::NoFill);
        cp_async_semantic_entry_for_mode!($warp, 8, variant::NoFill);
        cp_async_semantic_entry_for_mode!($warp, 16, variant::NoFill);
        cp_async_semantic_entry_for_mode!($warp, 4, variant::ZeroFill);
        cp_async_semantic_entry_for_mode!($warp, 8, variant::ZeroFill);
        cp_async_semantic_entry_for_mode!($warp, 16, variant::ZeroFill);
        cp_async_semantic_entry_for_mode!($warp, 4, variant::SourceSize);
        cp_async_semantic_entry_for_mode!($warp, 8, variant::SourceSize);
        cp_async_semantic_entry_for_mode!($warp, 16, variant::SourceSize);
    };
}

for_each_engine_mode!(test_visible, cp_async_entries_for_mode);

type BulkG2sBase = (Address<Shared>, Address<Global>, R<i64>, Address<Shared>);
type BulkG2sMulticastBase = (
    Address<Shared>,
    Address<Global>,
    R<i64>,
    Address<Shared>,
    R<i64>,
);
type BulkG2sIgnoreBase = (
    Address<Shared>,
    Address<Global>,
    R<i64>,
    Address<Shared>,
    R<i64>,
    R<i64>,
);
type BulkS2gBase = (Address<Global>, Address<Shared>, R<i64>);
type BulkS2gMaskedBase = (Address<Global>, Address<Shared>, R<i64>, R<i64>);

#[inline(never)]
fn execute_bulk_g2s(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    (destination, source, num_bytes, barrier): BulkG2sBase,
    destination_space: PtxStateSpace,
    report_pattern: u32,
    scope: Option<crate::MemoryScope>,
) -> Result<(), EngineError> {
    for lane in context.active_mask().into_inner() {
        let issue_context = context.with_active_mask(LaneMask::single(lane)?);
        let mask = issue_context.active_mask().into_inner();
        let inner_context = issue_context.into_inner();
        let source_barrier = barrier
            .inner()
            .resolve_shared_barrier(&inner_context, mask, None)?;
        let operation = begin(warp, issue_context, site, OperationKind::AsyncIssue, true)?;
        engine(warp).raw_bulk_copy_g2s_issue(
            operation.as_ref(),
            &inner_context,
            destination.inner(),
            source.inner(),
            num_bytes[lane],
            mask,
            destination_space,
            barrier.inner(),
            source_barrier,
            report_pattern,
            scope,
        )?;
        finish(warp, &operation)?;
    }
    Ok(())
}

macro_rules! bulk_g2s_variant {
    ($marker:ident, $space:expr) => {
        instruction_variant! {
            [impl<const REPORT: u32, Sem: BulkCopySemantics>]
            cp_async_bulk_spec, variant::$marker<REPORT, Sem>,
            BulkG2sBase => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                execute_bulk_g2s(warp, context, site, args, $space, REPORT, Sem::SCOPE)
            }
        }
    };
}

bulk_g2s_variant!(BulkG2sCluster, PtxStateSpace::SharedCluster);
bulk_g2s_variant!(BulkG2sCta, PtxStateSpace::SharedCta);
instruction_variant! {
    [impl<const REPORT: u32, Sem: BulkCopySemantics>]
    cp_async_bulk_spec, variant::BulkG2sClusterMulticast<REPORT, Sem>,
    BulkG2sMulticastBase => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (destination, source, num_bytes, barrier, cta_masks): Self::Args,
    ) -> Result<(), EngineError> {
        for lane in context.active_mask().into_inner() {
            let issue_context = context.with_active_mask(LaneMask::single(lane)?);
            let operation = begin(warp, issue_context, site, OperationKind::AsyncIssue, true)?;
            engine(warp).raw_bulk_copy_g2s_multicast_issue(
                operation.as_ref(),
                &issue_context.into_inner(),
                destination.inner(),
                source.inner(),
                num_bytes.inner(),
                issue_context.active_mask().into_inner(),
                barrier.inner(),
                cta_masks.inner(),
                REPORT,
                Sem::SCOPE,
            )?;
            finish(warp, &operation)?;
        }
        Ok(())
    }
}

type TmaG2sBase<const RANK: usize> = (
    Address<Shared>,
    TensorMapHandle,
    [R<i64>; RANK],
    Address<Shared>,
);
type TmaG2sMulticastBase<const RANK: usize> = (
    Address<Shared>,
    TensorMapHandle,
    [R<i64>; RANK],
    Address<Shared>,
    R<i64>,
);
type TmaGather4Base = (
    Address<Shared>,
    TensorMapHandle,
    [R<i64>; 5],
    Address<Shared>,
);
type TmaGather4MulticastBase = (
    Address<Shared>,
    TensorMapHandle,
    [R<i64>; 5],
    Address<Shared>,
    R<i64>,
);
type TmaS2gBase<const RANK: usize> = (Address<Shared>, TensorMapHandle, [R<i64>; RANK]);

fn tma_coordinates(values: &[R<i64>], lane: usize) -> Vec<i64> {
    values.iter().map(|axis| axis[lane]).collect()
}

fn validate_tma_descriptor_rank(
    tensor_map: &TensorMapHandle,
    rank: usize,
    label: &str,
) -> Result<(), EngineError> {
    if tensor_map.inner().rank() != rank {
        return Err(EngineError::message(format!(
            "{label} rank specialization {rank} does not match descriptor rank {}",
            tensor_map.inner().rank()
        )));
    }
    Ok(())
}

fn tma_multicast_mask(masks: Option<&R<i64>>, lane: usize) -> Result<u64, EngineError> {
    masks.map_or(Ok(0), |masks| {
        u64::try_from(masks[lane])
            .map_err(|_| EngineError::message("negative TMA multicast CTA mask"))
    })
}

#[inline(never)]
fn execute_tma_g2s(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    destination: Address<Shared>,
    tensor_map: TensorMapHandle,
    coordinate_operands: &[R<i64>],
    barrier: Address<Shared>,
    cta_group: u32,
    cta_masks: Option<R<i64>>,
    report_pattern: u32,
) -> Result<(), EngineError> {
    validate_tma_descriptor_rank(
        &tensor_map,
        coordinate_operands.len(),
        "cp.async.bulk.tensor g2s",
    )?;
    for lane in context.active_mask().into_inner() {
        let issue_context = context.with_active_mask(LaneMask::single(lane)?);
        let coordinates = tma_coordinates(coordinate_operands, lane);
        let mask = tma_multicast_mask(cta_masks.as_ref(), lane)?;
        let operation = begin(warp, issue_context, site, OperationKind::AsyncIssue, true)?;
        engine(warp).raw_tma_g2c_instruction(
            operation.as_ref(),
            &issue_context.into_inner(),
            destination.inner(),
            tensor_map.inner().as_ref(),
            &coordinates,
            mask,
            cta_masks.is_some(),
            barrier.inner(),
            i64::from(cta_group),
            report_pattern,
            None,
        )?;
        finish(warp, &operation)?;
    }
    Ok(())
}

macro_rules! tma_g2s_variant {
    ($marker:ident) => {
        instruction_variant! {
            [impl<const RANK: usize, const CTA_GROUP: u32, const REPORT: u32>] cp_async_bulk_tensor_spec, variant::$marker<RANK, CTA_GROUP, REPORT>
            where [
                TmaRank<RANK>: ValidTmaRank,
                TmaCtaGroup<CTA_GROUP>: ValidTmaCtaGroup,
            ],
            TmaG2sBase<RANK> => ();
            #[inline(always)]
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                let (destination, tensor_map, coordinates, barrier) = args;
                execute_tma_g2s(
                    warp,
                    context,
                    site,
                    destination,
                    tensor_map,
                    &coordinates,
                    barrier,
                    CTA_GROUP,
                    None,
                    REPORT,
                )
            }
        }
    };
}

tma_g2s_variant!(TensorG2sCta);
tma_g2s_variant!(TensorG2sCluster);

instruction_variant! {
    [impl<const RANK: usize, const CTA_GROUP: u32, const REPORT: u32>] cp_async_bulk_tensor_spec, variant::TensorG2sClusterMulticast<RANK, CTA_GROUP, REPORT>
    where [
        TmaRank<RANK>: ValidTmaRank,
        TmaCtaGroup<CTA_GROUP>: ValidTmaCtaGroup,
    ],
    TmaG2sMulticastBase<RANK> => ();
    #[inline(always)]
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        let (destination, tensor_map, coordinates, barrier, cta_masks) = args;
        execute_tma_g2s(
            warp,
            context,
            site,
            destination,
            tensor_map,
            &coordinates,
            barrier,
            CTA_GROUP,
            Some(cta_masks),
            REPORT,
        )
    }
}

#[inline(never)]
fn execute_tma_im2col(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    destination: Address<Shared>,
    tensor_map: TensorMapHandle,
    coordinates: &[R<i64>],
    barrier: Address<Shared>,
    info: [R<i64>; 3],
    masks: R<i64>,
    mode: u32,
    cta_group: u32,
    multicast: bool,
    report: u32,
) -> Result<(), EngineError> {
    use crate::runtime::tensor_map::Im2colMode;
    validate_tma_descriptor_rank(&tensor_map, coordinates.len(), "im2col")?;
    let (mode, count) = match mode {
        0 if (3..=5).contains(&coordinates.len()) => (Im2colMode::Spatial, coordinates.len() - 2),
        1 => (Im2colMode::Wide, 2),
        2 => (Im2colMode::Wide128, 2),
        _ => return Err(EngineError::message("invalid im2col specialization")),
    };
    for lane in context.active_mask().into_inner() {
        let issue_context = context.with_active_mask(LaneMask::single(lane)?);
        let coordinates = tma_coordinates(coordinates, lane);
        let info = tma_coordinates(&info, lane);
        if info[count..].iter().any(|value| *value != 0) {
            return Err(EngineError::message("nonzero unused im2col offsets"));
        }
        let mask = tma_multicast_mask(Some(&masks), lane)?;
        let operation = begin(warp, issue_context, site, OperationKind::AsyncIssue, true)?;
        engine(warp).raw_tma_g2c_instruction(
            operation.as_ref(),
            &issue_context.into_inner(),
            destination.inner(),
            tensor_map.inner().as_ref(),
            &coordinates,
            mask,
            multicast,
            barrier.inner(),
            i64::from(cta_group),
            report,
            Some((mode, &info[..count])),
        )?;
        finish(warp, &operation)?;
    }
    Ok(())
}

instruction_variant! {
    [impl<const N: usize, const MODE: u32, const GROUP: u32, const MULTI: bool, const REPORT: u32>] cp_async_bulk_tensor_spec, variant::TensorIm2col<N, MODE, GROUP, MULTI, REPORT>
    where [
        TmaRank<N>: ValidTmaRank,
        TmaCtaGroup<GROUP>: ValidTmaCtaGroup,
    ],
    (
        Address<Shared>,
        TensorMapHandle,
        [R<i64>; N],
        Address<Shared>,
        [R<i64>; 3],
        R<i64>,
    ) => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<(), EngineError> {
        let (destination, map, coordinates, barrier, info, mask) = args;
        execute_tma_im2col(
            warp,
            context,
            site,
            destination,
            map,
            &coordinates,
            barrier,
            info,
            mask,
            MODE,
            GROUP,
            MULTI,
            REPORT,
        )
    }
}

#[inline(never)]
fn execute_tma_gather4(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    (destination, tensor_map, coordinates, barrier): TmaGather4Base,
    cta_masks: Option<R<i64>>,
    cta_group: u32,
    report_pattern: u32,
) -> Result<(), EngineError> {
    validate_tma_descriptor_rank(&tensor_map, 2, "cp.async.bulk.tensor gather4")?;
    for lane in context.active_mask().into_inner() {
        let issue_context = context.with_active_mask(LaneMask::single(lane)?);
        let coordinates = tma_coordinates(&coordinates, lane);
        let mask = tma_multicast_mask(cta_masks.as_ref(), lane)?;
        let operation = begin(warp, issue_context, site, OperationKind::AsyncIssue, true)?;
        engine(warp).raw_tma_gather4_instruction(
            operation.as_ref(),
            &issue_context.into_inner(),
            destination.inner(),
            tensor_map.inner().as_ref(),
            coordinates[0],
            &coordinates[1..],
            mask,
            cta_masks.is_some(),
            barrier.inner(),
            i64::from(cta_group),
            report_pattern,
        )?;
        finish(warp, &operation)?;
    }
    Ok(())
}

macro_rules! tma_gather4_variant {
    ($marker:ident, $base:ty, |$args:ident| $split:expr) => {
        instruction_variant! {
            [impl<const CTA_GROUP: u32, const REPORT_PATTERN: u32>] cp_async_bulk_tensor_spec, variant::$marker<CTA_GROUP, REPORT_PATTERN>
            where [
                TmaCtaGroup<CTA_GROUP>: ValidTmaCtaGroup,
            ],
            $base => ();
            #[inline(always)]
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                let $args: $base = args;
                let (base, cta_masks) = $split;
                execute_tma_gather4(
                    warp,
                    context,
                    site,
                    base,
                    cta_masks,
                    CTA_GROUP,
                    REPORT_PATTERN,
                )
            }
        }
    };
}

tma_gather4_variant!(TensorGather4Cta, TmaGather4Base, |args| (args, None));
tma_gather4_variant!(TensorGather4Cluster, TmaGather4Base, |args| (args, None));
tma_gather4_variant!(
    TensorGather4ClusterMulticast,
    TmaGather4MulticastBase,
    |args| {
        let (destination, tensor_map, coordinates, barrier, cta_masks) = args;
        (
            (destination, tensor_map, coordinates, barrier),
            Some(cta_masks),
        )
    }
);

instruction_variant! {
    [impl<const RANK: usize, const MODE: u32>] cp_async_bulk_tensor_spec, variant::TensorS2g<RANK, MODE>
    where [
        TmaRank<RANK>: ValidTmaRank,
    ],
    TmaS2gBase<RANK> => ();
    #[inline(always)]
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        let (source, tensor_map, coordinates) = args;
        execute_tma_s2g(
            warp,
            context,
            site,
            source,
            tensor_map,
            &coordinates,
            None,
            MODE,
        )
    }
}

trait TmaReduction {
    const VALUE: crate::runtime::tensor_map::RawTmaReductionOp;
}

macro_rules! tma_reductions {
    ($($marker:ty => $value:ident),+ $(,)?) => {
        $(impl TmaReduction for $marker {
            const VALUE: crate::runtime::tensor_map::RawTmaReductionOp =
                crate::runtime::tensor_map::RawTmaReductionOp::$value;
        })+
    };
}

tma_reductions!(
    variant::ReduceAdd => Add,
    variant::ReduceMin => Min,
    variant::ReduceMax => Max,
    variant::ReduceInc => Inc,
    variant::ReduceDec => Dec,
    variant::ReduceAnd => And,
    variant::ReduceOr => Or,
    variant::ReduceXor => Xor,
);

instruction_variant! {
    [impl<const RANK: usize, Op: TmaReduction, const MODE: u32>] cp_reduce_async_bulk_tensor_spec, variant::TensorS2gReduce<RANK, Op, MODE>
    where [
        TmaRank<RANK>: ValidTmaRank,
    ],
    TmaS2gBase<RANK> => ();
    #[inline(always)]
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        let (source, tensor_map, coordinates) = args;
        execute_tma_s2g(
            warp,
            context,
            site,
            source,
            tensor_map,
            &coordinates,
            Some(Op::VALUE),
            MODE,
        )
    }
}

#[inline(never)]
fn execute_tma_s2g(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    source: Address<Shared>,
    tensor_map: TensorMapHandle,
    coordinate_operands: &[R<i64>],
    reduction: Option<crate::runtime::tensor_map::RawTmaReductionOp>,
    mode: u32,
) -> Result<(), EngineError> {
    validate_tma_descriptor_rank(
        &tensor_map,
        coordinate_operands.len(),
        "cp{.reduce}.async.bulk.tensor s2g",
    )?;
    let mode = tma_store_mode(mode)?;
    for lane in context.active_mask().into_inner() {
        let issue_context = context.with_active_mask(LaneMask::single(lane)?);
        let coordinates = tma_coordinates(coordinate_operands, lane);
        let operation = begin(warp, issue_context, site, OperationKind::AsyncIssue, true)?;
        engine(warp).raw_tma_s2g_issue(
            operation.as_ref(),
            &issue_context.into_inner(),
            source.inner(),
            tensor_map.inner().as_ref(),
            &coordinates,
            reduction,
            mode,
        )?;
        finish(warp, &operation)?;
    }
    Ok(())
}

fn tma_store_mode(
    mode: u32,
) -> Result<Option<crate::runtime::tensor_map::Im2colMode>, EngineError> {
    use crate::runtime::tensor_map::Im2colMode;
    match mode {
        0 => Ok(None),
        1 => Ok(Some(Im2colMode::Spatial)),
        2 => Ok(Some(Im2colMode::Wide)),
        _ => Err(EngineError::message("invalid TensorMap store mode")),
    }
}

fn validate_global_cache_hint_address(
    address: &Address<Global>,
    active: LaneMask,
    alignment: usize,
    valid_byte_len: usize,
    label: &str,
) -> Result<(), EngineError> {
    let active = active.into_inner();
    address
        .inner()
        .require_ptx_space_for_mask(PtxStateSpace::Global, active)?;
    for lane in active {
        let physical = address
            .inner()
            .lane_physical_byte_offset(lane, valid_byte_len)?;
        if physical % alignment != 0 {
            return Err(EngineError::message(format!(
                "{label} requires a {alignment}-byte aligned global address on lane {lane}, got byte offset {physical}"
            )));
        }
    }
    Ok(())
}

fn validate_bulk_cache_hint_range(
    address: &Address<Global>,
    sizes: &R<u32>,
    active: LaneMask,
    alignment: usize,
    label: &str,
) -> Result<(), EngineError> {
    for lane in active {
        let byte_len = usize::try_from(sizes[lane]).map_err(|_| {
            EngineError::message(format!("{label} size exceeds usize on lane {lane}"))
        })?;
        if byte_len % 16 != 0 {
            return Err(EngineError::message(format!(
                "{label} size {byte_len} must be a multiple of 16 on lane {lane}"
            )));
        }
    }
    // Unlike `.valid_addr`, bulk cache hints do not promise that the hinted
    // window is valid memory. Validate the address itself and its alignment,
    // without imposing one NumSim allocation's bounds on the whole window.
    validate_global_cache_hint_address(address, active, alignment, 0, label)
}

/// Validate the address promised valid by `prefetch.L1::32B.valid_addr`.
#[inline(never)]
pub fn prefetch_valid_address(
    _warp: &mut super::Engine,
    context: ExecCtx,
    _site: SiteId,
    address: Address<Global>,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(
        context,
        _site,
        std::any::type_name_of_val(&prefetch_valid_address),
    );
    validate_global_cache_hint_address(
        &address,
        context.active_mask(),
        1,
        1,
        "prefetch.L1::32B.valid_addr",
    )
}

/// Validate a non-tensor bulk prefetch, independent of its cache policy.
#[inline(never)]
pub fn cp_async_bulk_prefetch(
    _warp: &mut super::Engine,
    context: ExecCtx,
    _site: SiteId,
    (address, sizes): (Address<Global>, R<u32>),
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(
        context,
        _site,
        std::any::type_name_of_val(&cp_async_bulk_prefetch),
    );
    validate_bulk_cache_hint_range(
        &address,
        &sizes,
        context.active_mask(),
        16,
        "cp.async.bulk.prefetch",
    )
}

instruction_variant! {
    [impl] applypriority_spec, variant::ApplyPriority,
    Address<Global> => ();
    #[inline(always)]
    fn execute(
        _warp: &mut super::Engine,
        context: ExecCtx,
        _site: SiteId,
        address: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        validate_global_cache_hint_address(
            &address,
            context.active_mask(),
            128,
            0,
            "applypriority.L2::evict_normal",
        )
    }
}

instruction_variant! {
    [impl] applypriority_spec, variant::BulkApplyPriority,
    (Address<Global>, R<u32>) => ();
    #[inline(always)]
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        let (address, sizes) = args;
        validate_bulk_cache_hint_range(
            &address,
            &sizes,
            context.active_mask(),
            128,
            "applypriority.async.bulk",
        )?;
        let operation = begin(warp, context, site, OperationKind::AsyncIssue, true)?;
        engine(warp).bulk_async_group_issue_without_accesses(
            operation.as_ref(),
            context.active_mask().into_inner(),
        )?;
        finish(warp, &operation)
    }
}

fn execute_tma_cache_hint(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    tensor_map: TensorMapHandle,
    descriptor_rank: usize,
    coordinate_operands: &[R<i64>],
    label: &str,
    bulk_group: bool,
) -> Result<(), EngineError> {
    validate_tma_descriptor_rank(&tensor_map, descriptor_rank, label)?;
    let _ = coordinate_operands;
    if !bulk_group {
        // Tensor prefetch changes cache residency only, which NumSim does not
        // model. Its descriptor/rank/runtime operands are still validated.
        return Ok(());
    }
    let operation = begin(warp, context, site, OperationKind::AsyncIssue, true)?;
    engine(warp).bulk_async_group_issue_without_accesses(
        operation.as_ref(),
        context.active_mask().into_inner(),
    )?;
    finish(warp, &operation)
}

macro_rules! tensor_cache_hint_variant {
    ($spec:ident, $marker:ident, $rank:ident, $label:literal, $bulk_group:literal) => {
        impl<const $rank: usize> $spec::sealed::Sealed for variant::$marker<$rank> where
            TmaRank<$rank>: ValidTmaRank
        {
        }
        impl<const $rank: usize> $spec::Variant for variant::$marker<$rank>
        where
            TmaRank<$rank>: ValidTmaRank,
        {
            type Args = (TensorMapHandle, [R<i64>; $rank]);
            type Output = ();
        }
        impl<const $rank: usize> $spec::sealed::Execute for variant::$marker<$rank>
        where
            TmaRank<$rank>: ValidTmaRank,
        {
            #[inline(always)]
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                let (tensor_map, coordinates) = args;
                execute_tma_cache_hint(
                    warp,
                    context,
                    site,
                    tensor_map,
                    $rank,
                    &coordinates,
                    $label,
                    $bulk_group,
                )
            }
        }
    };
}

tensor_cache_hint_variant!(
    cp_async_bulk_prefetch_tensor_spec,
    TensorPrefetch,
    RANK,
    "cp.async.bulk.prefetch.tensor",
    false
);
tensor_cache_hint_variant!(
    cp_async_bulk_prefetch_tensor_spec,
    TensorPrefetchEvictLast,
    RANK,
    "cp.async.bulk.prefetch.tensor.L2::evict_last",
    false
);
tensor_cache_hint_variant!(
    applypriority_spec,
    TensorApplyPriority,
    RANK,
    "applypriority.async.bulk.tensor",
    true
);

macro_rules! tensor_gather4_cache_hint_variant {
    ($spec:ident, $marker:ident, $label:literal, $bulk_group:literal) => {
        instruction_variant! {
            [impl] $spec, variant::$marker,
            (TensorMapHandle, [R<i64>; 5]) => ();
            #[inline(always)]
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                let (tensor_map, coordinates) = args;
                execute_tma_cache_hint(
                    warp,
                    context,
                    site,
                    tensor_map,
                    2,
                    &coordinates,
                    $label,
                    $bulk_group,
                )
            }
        }
    };
}

tensor_gather4_cache_hint_variant!(
    cp_async_bulk_prefetch_tensor_spec,
    TensorPrefetchGather4,
    "cp.async.bulk.prefetch.tensor gather4",
    false
);
tensor_gather4_cache_hint_variant!(
    cp_async_bulk_prefetch_tensor_spec,
    TensorPrefetchEvictLastGather4,
    "cp.async.bulk.prefetch.tensor gather4.L2::evict_last",
    false
);
tensor_gather4_cache_hint_variant!(
    applypriority_spec,
    TensorApplyPriorityGather4,
    "applypriority.async.bulk.tensor gather4",
    true
);

/// `prefetch.tensormap` has no numerical or synchronization effect. The handle
/// is retained so malformed host bindings cannot be erased by the frontend.
#[inline(never)]
pub fn prefetch_tensormap(
    _warp: &mut super::Engine,
    _context: ExecCtx,
    _site: SiteId,
    _tensor_map: TensorMapHandle,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(
        _context,
        _site,
        std::any::type_name_of_val(&prefetch_tensormap),
    );
    Ok(())
}

instruction_variant! {
    [impl<Sem: BulkCopySemantics>] cp_async_bulk_spec, variant::BulkG2sCtaIgnoreOob<Sem>,
    BulkG2sIgnoreBase => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<(), EngineError> {
        let (destination, source, num_bytes, barrier, ignore_left, ignore_right) = args;
        for lane in context.active_mask().into_inner() {
            let issue_context = context.with_active_mask(LaneMask::single(lane)?);
            let mask = issue_context.active_mask().into_inner();
            let inner_context = issue_context.into_inner();
            let source_barrier =
                barrier
                    .inner()
                    .resolve_shared_barrier(&inner_context, mask, None)?;
            let operation = begin(warp, issue_context, site, OperationKind::AsyncIssue, true)?;
            engine(warp).raw_bulk_copy_g2s_ignore_oob_issue(
                operation.as_ref(),
                &inner_context,
                destination.inner(),
                source.inner(),
                num_bytes.inner(),
                ignore_left.inner(),
                ignore_right.inner(),
                mask,
                barrier.inner(),
                source_barrier,
                Sem::SCOPE,
            )?;
            finish(warp, &operation)?;
        }
        Ok(())
    }
}
#[inline(never)]
fn execute_bulk_s2s(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    args: (Address<Shared>, Address<Shared>, R<i64>, Address<Shared>),
    scope: Option<crate::MemoryScope>,
) -> Result<(), EngineError> {
    let (destination, source, num_bytes, barrier) = args;
    for lane in context.active_mask().into_inner() {
        let issue_context = context.with_active_mask(LaneMask::single(lane)?);
        let mask = issue_context.active_mask().into_inner();
        let inner_context = issue_context.into_inner();
        let source_barrier = barrier
            .inner()
            .resolve_shared_barrier(&inner_context, mask, None)?;
        let operation = begin(warp, issue_context, site, OperationKind::AsyncIssue, true)?;
        engine(warp).raw_bulk_copy_s2s_issue(
            operation.as_ref(),
            &inner_context,
            destination.inner(),
            source.inner(),
            num_bytes[lane],
            mask,
            barrier.inner(),
            source_barrier,
            scope,
        )?;
        finish(warp, &operation)?;
    }
    Ok(())
}

instruction_variant! {
    [impl<Sem: BulkCopySemantics>] cp_async_bulk_spec, variant::BulkSharedToCluster<Sem>,
    (Address<Shared>, Address<Shared>, R<i64>, Address<Shared>) => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        execute_bulk_s2s(warp, context, site, args, Sem::SCOPE)
    }
}

pub(crate) trait BulkCopySemantics {
    const SCOPE: Option<crate::MemoryScope>;
}

impl BulkCopySemantics for super::mem::variant::Plain {
    const SCOPE: Option<crate::MemoryScope> = None;
}

impl<Scope: super::mem::StaticScope> BulkCopySemantics for super::mem::variant::Relaxed<Scope> {
    const SCOPE: Option<crate::MemoryScope> = Some(Scope::VALUE);
}

instruction_variant! {
    [impl<Sem: BulkCopySemantics>] cp_async_bulk_spec, variant::BulkS2g<Sem>,
    BulkS2gBase => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (destination, source, num_bytes): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        for lane in context.active_mask().into_inner() {
            let issue_context = context.with_active_mask(LaneMask::single(lane)?);
            let operation = begin(warp, issue_context, site, OperationKind::AsyncIssue, true)?;
            engine(warp).raw_bulk_copy_s2g_issue(
                operation.as_ref(),
                &issue_context.into_inner(),
                destination.inner(),
                source.inner(),
                num_bytes[lane],
                issue_context.active_mask().into_inner(),
                Sem::SCOPE,
            )?;
            finish(warp, &operation)?;
        }
        Ok(())
    }
}

instruction_variant! {
    [impl<Sem: BulkCopySemantics>] cp_async_bulk_spec, variant::BulkS2gMasked<Sem>,
    BulkS2gMaskedBase => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<(), EngineError> {
        let (destination, source, num_bytes, byte_masks) = args;
        for lane in context.active_mask().into_inner() {
            let issue_context = context.with_active_mask(LaneMask::single(lane)?);
            let operation = begin(warp, issue_context, site, OperationKind::AsyncIssue, true)?;
            engine(warp).raw_bulk_copy_s2g_masked_issue(
                operation.as_ref(),
                &issue_context.into_inner(),
                destination.inner(),
                source.inner(),
                num_bytes.inner(),
                issue_context.active_mask().into_inner(),
                byte_masks.inner(),
                Sem::SCOPE,
            )?;
            finish(warp, &operation)?;
        }
        Ok(())
    }
}

// Tensor and non-tensor reductions share arithmetic; their type sets differ.
pub(crate) trait StaticReduction<T: super::mem::MemoryType, const TENSOR: bool = false> {
    const VALUE: crate::DeferredGlobalReduction;
}

macro_rules! reductions {
    ($op:ty, $type:ty => $value:ident) => {
        impl<const TENSOR: bool> StaticReduction<$type, TENSOR> for $op {
            const VALUE: crate::DeferredGlobalReduction = crate::DeferredGlobalReduction::$value;
        }
    };
}

reductions!(variant::ReduceAdd, super::reg::variant::U32 => AddU32);
reductions!(variant::ReduceAdd, super::reg::variant::I32 => AddI32);
reductions!(variant::ReduceAdd, super::reg::variant::U64 => AddU64);
// Bulk F32 addition preserves subnormals, including the implicit noftz form.
reductions!(variant::ReduceAdd, super::reg::variant::F32 => AddF32);
reductions!(variant::ReduceAdd, super::reg::variant::F16 => AddF16);
reductions!(variant::ReduceAdd, super::reg::variant::Bf16 => AddBf16);
reductions!(variant::ReduceMin, super::reg::variant::U32 => MinU32);
reductions!(variant::ReduceMin, super::reg::variant::I32 => MinI32);
reductions!(variant::ReduceMin, super::reg::variant::U64 => MinU64);
reductions!(variant::ReduceMin, super::reg::variant::I64 => MinI64);
reductions!(variant::ReduceMin, super::reg::variant::F16 => MinF16);
reductions!(variant::ReduceMin, super::reg::variant::Bf16 => MinBf16);
reductions!(variant::ReduceMax, super::reg::variant::U32 => MaxU32);
reductions!(variant::ReduceMax, super::reg::variant::I32 => MaxI32);
reductions!(variant::ReduceMax, super::reg::variant::U64 => MaxU64);
reductions!(variant::ReduceMax, super::reg::variant::I64 => MaxI64);
reductions!(variant::ReduceMax, super::reg::variant::F16 => MaxF16);
reductions!(variant::ReduceMax, super::reg::variant::Bf16 => MaxBf16);
reductions!(variant::ReduceInc, super::reg::variant::U32 => IncU32);
reductions!(variant::ReduceDec, super::reg::variant::U32 => DecU32);
reductions!(variant::ReduceAnd, super::reg::variant::B32 => AndB32);
reductions!(variant::ReduceAnd, super::reg::variant::B64 => AndB64);
reductions!(variant::ReduceOr, super::reg::variant::B32 => OrB32);
reductions!(variant::ReduceOr, super::reg::variant::B64 => OrB64);
reductions!(variant::ReduceXor, super::reg::variant::B32 => XorB32);
reductions!(variant::ReduceXor, super::reg::variant::B64 => XorB64);

// F64 add exists only on the non-tensor instruction. Sharing element math
// must not widen the typed tensor ABI's legal operation/type combinations.
impl StaticReduction<super::reg::variant::F64> for variant::ReduceAdd {
    const VALUE: crate::DeferredGlobalReduction = crate::DeferredGlobalReduction::AddF64;
}

instruction_variant! {
    [impl<T: super::mem::MemoryType, Op: StaticReduction<T>, Scope: super::mem::StaticScope>]
    cp_reduce_async_bulk_spec, variant::BulkS2gReduce<T, Op, Scope>,
    BulkS2gBase => ();
    #[inline(always)]
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        let (destination, source, num_bytes) = args;
        // Bulk groups belong to issuing threads. Keep each lane's payload
        // and waits on the existing exact single-thread transaction.
        for lane in context.active_mask().into_inner() {
            let issue_context = context.with_active_mask(LaneMask::single(lane)?);
            let operation = begin(warp, issue_context, site, OperationKind::AsyncIssue, true)?;
            engine(warp).raw_bulk_reduce_s2g_issue(
                operation.as_ref(),
                &issue_context.into_inner(),
                destination.inner(),
                source.inner(),
                num_bytes[lane],
                issue_context.active_mask().into_inner(),
                Scope::VALUE,
                Op::VALUE,
            )?;
            finish(warp, &operation)?;
        }
        Ok(())
    }
}

/// Commit the current classic `cp.async` group.
///
/// There is only one legal spelling, so this is intentionally not a generic
/// specialization point.
#[inline(never)]
pub fn cp_async_commit_group(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(
        context,
        site,
        std::any::type_name_of_val(&cp_async_commit_group),
    );
    let operation = begin(warp, context, site, OperationKind::Barrier, true)?;
    engine(warp).async_group_commit(
        operation.as_ref(),
        AsyncGroupDomain::CpAsync,
        context.active_mask().into_inner(),
    )?;
    finish(warp, &operation)
}

#[inline(never)]
async fn execute_cp_async_wait_group(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    pending: i64,
) -> Result<(), EngineError> {
    let operation = begin(warp, context, site, OperationKind::Barrier, true)?;
    engine(warp)
        .async_group_wait(
            operation.as_ref(),
            AsyncGroupDomain::CpAsync,
            context.active_mask().into_inner(),
            pending,
            false,
        )
        .await?;
    finish(warp, &operation)
}

instruction_variant! {
    [impl] cp_async_wait_group_spec, variant::CpAsyncWaitGroup,
    i64 => ();
    async fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        pending: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        execute_cp_async_wait_group(warp, context, site, pending).await
    }
}

/// Commit the current `cp.async.bulk` group.
#[inline(never)]
pub fn cp_async_bulk_commit_group(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(
        context,
        site,
        std::any::type_name_of_val(&cp_async_bulk_commit_group),
    );
    let operation = begin(warp, context, site, OperationKind::Barrier, true)?;
    engine(warp).async_group_commit(
        operation.as_ref(),
        AsyncGroupDomain::Bulk,
        context.active_mask().into_inner(),
    )?;
    finish(warp, &operation)
}

#[inline(never)]
async fn execute_bulk_wait_group(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    pending: i64,
    read_only: bool,
) -> Result<(), EngineError> {
    let operation = begin(warp, context, site, OperationKind::Barrier, true)?;
    engine(warp)
        .async_group_wait(
            operation.as_ref(),
            AsyncGroupDomain::Bulk,
            context.active_mask().into_inner(),
            pending,
            read_only,
        )
        .await?;
    finish(warp, &operation)
}

macro_rules! bulk_wait_variant {
    ($marker:ident, $read_only:literal) => {
        instruction_variant! {
            [impl] cp_async_bulk_wait_group_spec, variant::$marker,
            i64 => ();
            async fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                pending: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                execute_bulk_wait_group(warp, context, site, pending, $read_only).await
            }
        }
    };
}

bulk_wait_variant!(BulkWaitGroup, false);
bulk_wait_variant!(BulkWaitGroupRead, true);

/// Both `cp.async.mbarrier.arrive` spellings commit the issuing lanes' prior
/// classic `cp.async` work and bind one arrive-on per lane to that work's
/// full-completion milestone. They differ only in whether the current phase's
/// pending arrival count is first raised to budget for that arrive-on, which is
/// a static property of the spelling.
#[inline(never)]
fn cp_async_mbarrier_arrive_impl(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    barrier: Address<Shared>,
    no_increment: bool,
) -> Result<(), EngineError> {
    let operation = begin(warp, context, site, OperationKind::Barrier, true)?;
    engine(warp).cp_async_mbarrier_arrive_completion(
        operation.as_ref(),
        barrier.inner(),
        context.active_mask().into_inner(),
        !no_increment,
    )?;
    finish(warp, &operation)
}

macro_rules! cp_async_mbarrier_variant {
    ($marker:ty, $no_increment:expr) => {
        instruction_variant! {
            [impl] cp_async_mbarrier_arrive_spec, $marker,
            Address<Shared> => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                barrier: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                cp_async_mbarrier_arrive_impl(warp, context, site, barrier, $no_increment)
            }
        }
    };
}

cp_async_mbarrier_variant!(variant::CpAsyncMbarrierArrive, false);
cp_async_mbarrier_variant!(variant::CpAsyncMbarrierArriveNoInc, true);

#[cfg(test)]
mod cache_hint_tests {
    use super::{
        Address, Global, LaneMask, R, validate_bulk_cache_hint_range,
        validate_global_cache_hint_address,
    };
    use crate::runtime::{PhysicalPtr, RuntimeBuffer};
    use crate::{GlobalMemory, WarpValue};

    fn global_address(byte_len: usize, byte_offset: i64) -> (GlobalMemory, Address<Global>) {
        let global = GlobalMemory::new();
        let allocation = global.allocate_zeroed(byte_len).unwrap();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Global(global.full_view(allocation).unwrap()),
            WarpValue::splat(byte_offset),
            1,
        )
        .bounded_to_initial_view();
        (global, Address::from_inner(pointer))
    }

    #[test]
    fn valid_address_prefetch_requires_one_addressable_global_byte() {
        let (_global, address) = global_address(1, 0);
        validate_global_cache_hint_address(
            &address,
            LaneMask::from_bits(1),
            1,
            1,
            "valid-address prefetch",
        )
        .unwrap();

        let (_global, one_past_end) = global_address(1, 1);
        assert!(validate_global_cache_hint_address(
            &one_past_end,
            LaneMask::from_bits(1),
            1,
            1,
            "valid-address prefetch",
        )
        .unwrap_err()
        .to_string()
        .contains("out-of-bounds"));
    }

    #[test]
    fn address_cache_hints_enforce_alignment_and_bulk_size_only() {
        let mask = LaneMask::from_bits(1);
        let (_global, aligned) = global_address(256, 0);
        validate_bulk_cache_hint_range(&aligned, &R::splat(128), mask, 128, "bulk applypriority")
            .unwrap();

        let (_global, misaligned) = global_address(256, 16);
        assert!(validate_bulk_cache_hint_range(
            &misaligned,
            &R::splat(128),
            mask,
            128,
            "bulk applypriority",
        )
        .unwrap_err()
        .to_string()
        .contains("128-byte aligned"));

        assert!(
            validate_bulk_cache_hint_range(&aligned, &R::splat(12), mask, 16, "bulk prefetch",)
                .unwrap_err()
                .to_string()
                .contains("multiple of 16")
        );

        let (_global, too_short) = global_address(64, 0);
        validate_bulk_cache_hint_range(&too_short, &R::splat(128), mask, 16, "bulk prefetch")
            .unwrap();

        validate_bulk_cache_hint_range(&aligned, &R::splat(0), mask, 16, "bulk prefetch").unwrap();
    }
}
