//! Engine implementation of v2 source-level collective operations.

use super::EngineError;
use std::marker::PhantomData;

use super::instruction::{async_instruction, instruction_variant};
use super::transport::engine;
use super::{Address, ExecCtx, Shared, SiteId, R};
use crate::collectives::{
    Bf16Reduce, CtaReduceContribution, CtaReduceElement, CtaReduceOp, CtaReduceValue, Fp16Reduce,
};
use crate::runtime::warp_ops::{
    warp_reduce_max, warp_reduce_max_bf16, warp_reduce_max_fp16, warp_reduce_min,
    warp_reduce_min_bf16, warp_reduce_min_fp16, warp_reduce_sum, warp_reduce_sum_bf16,
    warp_reduce_sum_fp16,
};
use crate::runtime::{raw_store_physical_ptr_warp, PtxStateSpace};
use crate::{
    MemoryAccessSemantics, OccurrenceKey, OperationKind, ParticipantContract, RuntimeScalar,
    StaticOpId, WarpMask, WarpValue, WARP_SIZE,
};

// `cta_reduce` executes one complete `tirx.cuda.cta_reduce` intrinsic,
// including its architecturally visible scratch traffic and CTA
// synchronization. It is intentionally a high-level collective ABI rather than
// a fake PTX instruction: the cross-warp rendezvous and publication state
// cannot be reconstructed from independent `ld`/`st`/`bar.sync` calls, while
// the operation, scalar format, and participant count are all static variants.
async_instruction!(cta_reduce_spec, CtaReduceVariant, cta_reduce);

// `cta_vote` executes one complete `syncthreads_and` or `syncthreads_or`. It
// remains separate from `cta_reduce`: it has no scratch operand and returns a
// boolean-vote integer, so merging the two would require an invalid
// nullable address/value contract.
async_instruction!(cta_vote_spec, CtaVoteVariant, cta_vote);
async_instruction!(bar_reduce_spec, BarReduceVariant, bar_reduce);

// `participate` validates one frontend collective-participation boundary
// without creating a memory-ordering edge. This is an engine protocol, not a
// PTX instruction, and it is deliberately distinct from every barrier ABI. The
// scope is its entire operand set, hence the `no_args` form.
async_instruction!(participate_spec, ParticipateVariant, participate, no_args);

/// Frontend tile scope whose participation cannot be reconstructed after a
/// tile operation has been expanded into scalar instructions.
pub mod scope {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Warp;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Warpgroup;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Cta;
}

/// Static specializations of the high-level CUDA CTA reduction intrinsic.
pub mod variant {
    use super::PhantomData;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Sum;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Min;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Max;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct All;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Any;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BarReduce<Op, const ALIGNED: bool>(PhantomData<Op>);

    /// Semantic f16/bf16 values use f32 register carriers but round at every
    /// reduction step.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct F16;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Bf16;

    /// One `tirx.cuda.cta_reduce` variant. The participating warp count is the
    /// launch contract encoded by the frontend intrinsic; it names no type and
    /// is re-checked against the topology at runtime, so it rides in `Args`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Reduce<T, Op>(PhantomData<fn() -> (T, Op)>);
}

/// One `cta_vote` specialization: the lane-fold applied inside the warp, then
/// the CTA-wide reduction that publishes it.
macro_rules! cta_vote_variants {
    ($($marker:ty => ($fold:ident, $reduction:ident, $name:literal)),+ $(,)?) => {
        $(
            instruction_variant! {
                [impl] cta_vote_spec, $marker,
                R<bool> => R<i64>;
                #[inline(never)]
                async fn execute(
                    warp: &mut super::Engine,
                    context: ExecCtx,
                    site: SiteId,
                    predicates: Self::Args,
                ) -> Result<Self::Output, EngineError> {
                    let local = i32::from(
                        context
                            .active_mask()
                            .into_inner()
                            .into_iter()
                            .$fold(|lane| predicates[lane]),
                    );
                    execute_cta_vote(
                        warp,
                        context,
                        site,
                        CtaReduceOp::$reduction,
                        $name,
                        local,
                    )
                    .await
                }
            }
        )+
    };
}

cta_vote_variants!(
    variant::All => (all, Min, "and"),
    variant::Any => (any, Max, "or"),
);

trait ReduceOperation {
    const VALUE: CtaReduceOp;
    const NAME: &'static str;
}

macro_rules! reduce_operations {
    ($($marker:ty => ($value:ident, $name:literal)),+ $(,)?) => {
        $(
            impl ReduceOperation for $marker {
                const VALUE: CtaReduceOp = CtaReduceOp::$value;
                const NAME: &'static str = $name;
            }
        )+
    };
}

reduce_operations!(
    variant::Sum => (Sum, "sum"),
    variant::Min => (Min, "min"),
    variant::Max => (Max, "max"),
);

mod reduce_type_sealed {
    pub trait Sealed {}
}

/// Semantic register carrier selected by a CTA reduction scalar marker.
#[allow(private_bounds)]
pub trait CtaReduceType: reduce_type_sealed::Sealed {
    type Scalar: RuntimeScalar + Send + Sync + 'static;
}

trait ReduceType: CtaReduceType {
    type Storage: RuntimeScalar + Send + Sync + 'static;
    type Contribution: CtaReduceElement;

    fn warp_reduce(
        operation: CtaReduceOp,
        mask: WarpMask,
        values: &WarpValue<Self::Scalar>,
    ) -> Result<WarpValue<Self::Scalar>, crate::EngineError>;
    fn encode(value: Self::Scalar) -> Self::Storage;
    fn contribution(value: Self::Scalar) -> Self::Contribution;
    fn result(value: CtaReduceValue) -> Result<Self::Scalar, crate::EngineError>;
}

macro_rules! ordinary_reduce_types {
    ($($marker:ty => $scalar:ty),+ $(,)?) => {
        $(
            impl reduce_type_sealed::Sealed for $marker {}
            impl CtaReduceType for $marker {
                type Scalar = $scalar;
            }
            impl ReduceType for $marker {
                type Storage = $scalar;
                type Contribution = $scalar;

                fn warp_reduce(
                    operation: CtaReduceOp,
                    mask: WarpMask,
                    values: &WarpValue<Self::Scalar>,
                ) -> Result<WarpValue<Self::Scalar>, crate::EngineError> {
                    match operation {
                        CtaReduceOp::Sum => warp_reduce_sum(mask, values, WARP_SIZE),
                        CtaReduceOp::Min => warp_reduce_min(mask, values, WARP_SIZE),
                        CtaReduceOp::Max => warp_reduce_max(mask, values, WARP_SIZE),
                    }
                }

                fn encode(value: Self::Scalar) -> Self::Storage { value }
                fn contribution(value: Self::Scalar) -> Self::Contribution { value }
                fn result(value: CtaReduceValue) -> Result<Self::Scalar, crate::EngineError> {
                    Ok(<$scalar as CtaReduceElement>::from_cta_reduce_value(value)?)
                }
            }
        )+
    };
}

ordinary_reduce_types!(
    super::reg::variant::I8 => i8,
    super::reg::variant::I16 => i16,
    super::reg::variant::I32 => i32,
    super::reg::variant::I64 => i64,
    super::reg::variant::U8 => u8,
    super::reg::variant::U16 => u16,
    super::reg::variant::U32 => u32,
    super::reg::variant::U64 => u64,
    super::reg::variant::F32 => f32,
    super::reg::variant::F64 => f64,
);

impl ReduceType for variant::F16 {
    type Storage = u16;
    type Contribution = Fp16Reduce;

    fn warp_reduce(
        operation: CtaReduceOp,
        mask: WarpMask,
        values: &WarpValue<f32>,
    ) -> Result<WarpValue<f32>, crate::EngineError> {
        match operation {
            CtaReduceOp::Sum => warp_reduce_sum_fp16(mask, values, WARP_SIZE),
            CtaReduceOp::Min => warp_reduce_min_fp16(mask, values, WARP_SIZE),
            CtaReduceOp::Max => warp_reduce_max_fp16(mask, values, WARP_SIZE),
        }
    }

    fn encode(value: f32) -> u16 {
        crate::f32_to_fp16_bits(value)
    }

    fn contribution(value: f32) -> Fp16Reduce {
        Fp16Reduce::from_f32(value)
    }

    fn result(value: CtaReduceValue) -> Result<f32, crate::EngineError> {
        Ok(Fp16Reduce::from_cta_reduce_value(value)?.to_f32())
    }
}

impl reduce_type_sealed::Sealed for variant::F16 {}
impl CtaReduceType for variant::F16 {
    type Scalar = f32;
}

impl ReduceType for variant::Bf16 {
    type Storage = u16;
    type Contribution = Bf16Reduce;

    fn warp_reduce(
        operation: CtaReduceOp,
        mask: WarpMask,
        values: &WarpValue<f32>,
    ) -> Result<WarpValue<f32>, crate::EngineError> {
        match operation {
            CtaReduceOp::Sum => warp_reduce_sum_bf16(mask, values, WARP_SIZE),
            CtaReduceOp::Min => warp_reduce_min_bf16(mask, values, WARP_SIZE),
            CtaReduceOp::Max => warp_reduce_max_bf16(mask, values, WARP_SIZE),
        }
    }

    fn encode(value: f32) -> u16 {
        crate::f32_to_bf16_bits(value)
    }

    fn contribution(value: f32) -> Bf16Reduce {
        Bf16Reduce::from_f32(value)
    }

    fn result(value: CtaReduceValue) -> Result<f32, crate::EngineError> {
        Ok(Bf16Reduce::from_cta_reduce_value(value)?.to_f32())
    }
}

impl reduce_type_sealed::Sealed for variant::Bf16 {}
impl CtaReduceType for variant::Bf16 {
    type Scalar = f32;
}

impl<T: ReduceType, Op: ReduceOperation> cta_reduce_spec::sealed::Sealed
    for variant::Reduce<T, Op>
{
}

impl<T: ReduceType, Op: ReduceOperation> cta_reduce_spec::Variant for variant::Reduce<T, Op> {
    type Args = (R<<T as CtaReduceType>::Scalar>, Address<Shared>, usize);
    type Output = R<<T as CtaReduceType>::Scalar>;
}

impl<T: ReduceType + ReduceEntry<Op>, Op: ReduceOperation> cta_reduce_spec::sealed::Execute
    for variant::Reduce<T, Op>
{
    #[inline(always)]
    async fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        let (values, scratch, warps) = args;
        <T as ReduceEntry<Op>>::execute(warp, context, site, (values, scratch), warps).await
    }
}

type ReduceStorage<T> = <T as ReduceType>::Storage;

trait ReduceEntry<Op: ReduceOperation>: ReduceType {
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: (R<Self::Scalar>, Address<Shared>),
        warps: usize,
    ) -> impl std::future::Future<Output = Result<R<Self::Scalar>, EngineError>> + Send;
}

macro_rules! reduce_entry {
    ($ty:ty, $op:ty) => {
        impl ReduceEntry<$op> for $ty {
            #[inline(never)]
            async fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: (R<Self::Scalar>, Address<Shared>),
                warps: usize,
            ) -> Result<R<Self::Scalar>, EngineError> {
                execute_cta_reduce::<$ty>(
                    warp,
                    context,
                    site,
                    args,
                    <$op as ReduceOperation>::VALUE,
                    <$op as ReduceOperation>::NAME,
                    warps,
                )
                .await
            }
        }
    };
}

macro_rules! reduce_entries {
    ($ty:ty) => {
        reduce_entry!($ty, variant::Sum);
        reduce_entry!($ty, variant::Min);
        reduce_entry!($ty, variant::Max);
    };
}

reduce_entries!(super::reg::variant::I8);
reduce_entries!(super::reg::variant::I16);
reduce_entries!(super::reg::variant::I32);
reduce_entries!(super::reg::variant::I64);
reduce_entries!(super::reg::variant::U8);
reduce_entries!(super::reg::variant::U16);
reduce_entries!(super::reg::variant::U32);
reduce_entries!(super::reg::variant::U64);
reduce_entries!(super::reg::variant::F32);
reduce_entries!(super::reg::variant::F64);
reduce_entries!(variant::F16);
reduce_entries!(variant::Bf16);

#[inline(never)]
async fn execute_cta_reduce<T>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    args: (R<T::Scalar>, Address<Shared>),
    operation: CtaReduceOp,
    operation_name: &'static str,
    warps: usize,
) -> Result<R<T::Scalar>, EngineError>
where
    T: ReduceType,
{
    if warps == 0 || warps > WARP_SIZE || !warps.is_power_of_two() {
        return Err(EngineError::message(format!(
            "CTA reduction warp count must be a power of two in 1..={WARP_SIZE}, got {warps}"
        )));
    }
    let (values, scratch) = args;
    let context = context.into_inner();
    crate::runtime::require_full_warp_sync(context.active_mask(), "cuda_cta_reduce")?;
    let warp = engine(warp);
    if warp.kernel().physical().topology().warps_per_cta() != warps {
        return Err(EngineError::message(format!(
            "cuda_cta_reduce expected {warps} warps"
        )));
    }
    let physical = warp.kernel().physical().clone();
    let hub = warp.kernel().services().cta_reduce();
    let scratch = scratch.into_inner();
    let lane_zero = WarpMask::from_lanes(std::iter::once(0_usize))
        .map_err(|error| EngineError::message(error.to_string()))?;
    let storage_width = ReduceStorage::<T>::BYTE_LEN;
    let warp_partial = T::warp_reduce(operation, context.active_mask(), values.inner())?;

    let partial_offsets = WarpValue::splat(
        i64::try_from(context.warp_id_in_cta() * storage_width)
            .map_err(|_| EngineError::message("CTA reduction scratch offset exceeds i64"))?,
    );
    let partial_pointer = scratch.with_byte_offset(&partial_offsets, storage_width, lane_zero)?;
    let partial_storage = WarpValue::from_fn(|lane| T::encode(warp_partial[lane]));
    let partial_context = context.with_active_mask(lane_zero);
    let partial_store = warp.begin_current_operation(
        partial_context,
        StaticOpId::new(site.get()),
        OperationKind::Store,
    )?;
    warp.physical_pointer_access(
        Some(&partial_store),
        OperationKind::Store,
        &partial_pointer,
        None,
        lane_zero,
        storage_width,
        false,
        false,
        MemoryAccessSemantics::plain(),
        || {
            raw_store_physical_ptr_warp::<ReduceStorage<T>>(
                &physical,
                &partial_context,
                &partial_pointer,
                &partial_storage,
                lane_zero,
                PtxStateSpace::Shared,
            )
        },
    )?;
    warp.finish_operation(&partial_store)?;

    let first_barrier = warp.begin_current_operation(
        context,
        StaticOpId::new(site.get()),
        OperationKind::Collective,
    )?;
    let loop_path = first_barrier
        .id()
        .loop_frames()
        .iter()
        .map(|frame| {
            i64::try_from(frame.iteration_ordinal())
                .map_err(|_| EngineError::message("CTA reduction loop iteration exceeds i64"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    warp.named_barrier_sync_with_alignment(
        Some(&first_barrier),
        0,
        i64::try_from(warps * WARP_SIZE)
            .map_err(|_| EngineError::message("CTA reduction arrival count exceeds i64"))?,
        context.active_mask(),
        true,
    )
    .await?;
    warp.finish_operation(&first_barrier)?;

    let contract = ParticipantContract::cta(context);
    let key = OccurrenceKey::new(
        site.get(),
        format!("cta_reduce_{operation_name}"),
        loop_path,
        contract.scope().clone(),
    );
    let contribution = T::contribution(warp_partial[0]);
    let published = hub
        .collect(
            key,
            contract,
            context.global_warp_id(),
            CtaReduceContribution::new(operation, contribution),
        )
        .map_err(crate::EngineError::from)?
        .await
        .map_err(crate::EngineError::from)?;
    let scalar = T::result(*published)?;
    let result = WarpValue::splat(scalar);

    if context.warp_id_in_cta() == 0 {
        let partial_read_mask = WarpMask::from_lanes(0..warps)
            .map_err(|error| EngineError::message(error.to_string()))?;
        let partial_read_offsets =
            WarpValue::from_fn(|lane| i64::try_from(lane * storage_width).unwrap_or(i64::MAX));
        let partial_read_pointer =
            scratch.with_byte_offset(&partial_read_offsets, storage_width, partial_read_mask)?;
        let partial_read_context = context.with_active_mask(partial_read_mask);
        let partial_read = warp.begin_current_operation(
            partial_read_context,
            StaticOpId::new(site.get()),
            OperationKind::Load,
        )?;
        warp.physical_pointer_access(
            Some(&partial_read),
            OperationKind::Load,
            &partial_read_pointer,
            None,
            partial_read_mask,
            storage_width,
            false,
            false,
            MemoryAccessSemantics::plain(),
            || Ok(()),
        )?;
        warp.finish_operation(&partial_read)?;

        let result_storage = WarpValue::from_fn(|lane| T::encode(result[lane]));
        let final_store_context = context.with_active_mask(lane_zero);
        let final_store = warp.begin_current_operation(
            final_store_context,
            StaticOpId::new(site.get()),
            OperationKind::Store,
        )?;
        warp.physical_pointer_access(
            Some(&final_store),
            OperationKind::Store,
            &scratch,
            None,
            lane_zero,
            storage_width,
            false,
            false,
            MemoryAccessSemantics::plain(),
            || {
                raw_store_physical_ptr_warp::<ReduceStorage<T>>(
                    &physical,
                    &final_store_context,
                    &scratch,
                    &result_storage,
                    lane_zero,
                    PtxStateSpace::Shared,
                )
            },
        )?;
        warp.finish_operation(&final_store)?;
    }

    let final_barrier = warp.begin_current_operation(
        context,
        StaticOpId::new(site.get()),
        OperationKind::Collective,
    )?;
    warp.named_barrier_sync_with_alignment(
        Some(&final_barrier),
        0,
        i64::try_from(warps * WARP_SIZE)
            .map_err(|_| EngineError::message("CTA reduction arrival count exceeds i64"))?,
        context.active_mask(),
        true,
    )
    .await?;
    warp.finish_operation(&final_barrier)?;

    let final_read =
        warp.begin_current_operation(context, StaticOpId::new(site.get()), OperationKind::Load)?;
    warp.physical_pointer_access(
        Some(&final_read),
        OperationKind::Load,
        &scratch,
        None,
        context.active_mask(),
        storage_width,
        false,
        false,
        MemoryAccessSemantics::plain(),
        || Ok(()),
    )?;
    warp.finish_operation(&final_read)?;
    Ok(R::from_inner(result))
}

#[inline(never)]
async fn execute_cta_vote(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    vote_operation: CtaReduceOp,
    operation_name: &'static str,
    local: i32,
) -> Result<R<i64>, EngineError> {
    let context = context.into_inner();
    crate::runtime::require_full_warp_sync(
        context.active_mask(),
        if operation_name == "and" {
            "cuda_syncthreads_and"
        } else {
            "cuda_syncthreads_or"
        },
    )?;
    let warp = engine(warp);
    let hub = warp.kernel().services().cta_reduce();
    let operation = warp.begin_current_operation(
        context,
        StaticOpId::new(site.get()),
        OperationKind::Collective,
    )?;
    let loop_path = operation
        .id()
        .loop_frames()
        .iter()
        .map(|frame| {
            i64::try_from(frame.iteration_ordinal())
                .map_err(|_| EngineError::message("CTA vote loop iteration exceeds i64"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let contract = ParticipantContract::cta(context);
    let key = OccurrenceKey::new(
        site.get(),
        format!("cta_vote_{operation_name}"),
        loop_path,
        contract.scope().clone(),
    );
    let published = hub
        .collect(
            key,
            contract,
            context.global_warp_id(),
            CtaReduceContribution::new(vote_operation, local),
        )
        .map_err(crate::EngineError::from)?
        .await
        .map_err(crate::EngineError::from)?;
    let scalar = i32::from_cta_reduce_value(*published).map_err(crate::EngineError::from)?;
    warp.finish_operation(&operation)?;
    Ok(R::splat(i64::from(scalar != 0)))
}

macro_rules! bar_reduce_variants {
    ($($op:ty => $reduction:ident),+ $(,)?) => {
        $(
            instruction_variant! {
                [impl<const ALIGNED: bool>] bar_reduce_spec, variant::BarReduce<$op, ALIGNED>,
                (i64, i64, R<bool>) => R<u32>;
                async fn execute(warp: &mut super::Engine, context: ExecCtx, site: SiteId,
                                 (id, count, predicates): Self::Args) -> Result<Self::Output, EngineError> {
                    let count_true: i32 = predicates.inner().lanes().iter().map(|value| i32::from(*value)).sum();
                    let local = match CtaReduceOp::$reduction {
                        CtaReduceOp::Min => i32::from(count_true == 32),
                        CtaReduceOp::Max => i32::from(count_true != 0),
                        _ => count_true,
                    };
                    execute_bar_reduce(warp, context, site, id, count, ALIGNED, CtaReduceOp::$reduction, local).await
                }
            }
        )+
    };
}

bar_reduce_variants!(variant::Sum => Sum, variant::All => Min, variant::Any => Max);

#[inline(never)]
async fn execute_bar_reduce(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    id: i64,
    count: i64,
    aligned: bool,
    reduction: CtaReduceOp,
    local: i32,
) -> Result<R<u32>, EngineError> {
    let context = context.into_inner();
    crate::runtime::require_full_warp_sync(context.active_mask(), "barrier.red")?;
    let warp = engine(warp);
    let operation = warp.begin_current_operation(
        context,
        StaticOpId::new(site.get()),
        OperationKind::Collective,
    )?;
    let resume = warp
        .named_barrier_sync_effect(Some(&operation), id, count, context.active_mask(), aligned)
        .await?
        .ok_or_else(|| {
            EngineError::message("barrier.red requires converged participating warps")
        })?;
    let generation = i64::try_from(resume.generation())
        .map_err(|_| EngineError::message("barrier.red generation exceeds i64"))?;
    let scope = crate::ScopeInstance::Cta {
        global_cta_id: context.global_cta_id(),
    };
    let key = OccurrenceKey::new(
        resume.barrier_id().barrier_id() as u64,
        "barrier.red",
        [generation],
        scope.clone(),
    );
    let hub = warp.kernel().services().cta_reduce();
    // The last arrival snapshots before its first collective suspension. Later
    // waiters use that hub-owned contract even if another group reused the ID.
    let contract = match hub.participant_contract(&key) {
        Some(contract) => contract,
        None => ParticipantContract::explicit(
            scope,
            warp.kernel()
                .services()
                .named_barriers()
                .completed_participants(resume.barrier_id(), resume.generation())?,
        ),
    };
    let result = hub
        .collect(
            key,
            contract,
            context.global_warp_id(),
            CtaReduceContribution::new(reduction, local),
        )
        .map_err(crate::EngineError::from)?
        .await
        .map_err(crate::EngineError::from)?;
    let scalar = i32::from_cta_reduce_value(*result).map_err(crate::EngineError::from)?;
    warp.finish_operation(&operation)?;
    Ok(R::splat(scalar as u32))
}

/// One frontend participation scope: its engine scope key, its diagnostic
/// name, and whether the boundary is warp-local (validated in place) or a
/// cross-warp rendezvous.
macro_rules! participation_scopes {
    ($($marker:ty => ($scope:literal, $name:literal, $warp_local:literal)),+ $(,)?) => {
        $(
            impl participate_spec::sealed::Sealed for $marker {}

            impl participate_spec::Variant for $marker {
                type Output = ();
            }

            impl participate_spec::sealed::Execute for $marker {
                #[inline(never)]
                async fn execute(
                    warp: &mut super::Engine,
                    context: ExecCtx,
                    site: SiteId,
                ) -> Result<Self::Output, EngineError> {
                    execute_participate(warp, context, site, $scope, $name, $warp_local).await
                }
            }
        )+
    };
}

participation_scopes!(
    scope::Warp => ("participation:warp", "tile.warp participation", true),
    // `Tx.wg` selects a warpgroup-shaped layout, but it does not itself add a
    // four-warp rendezvous.  Every warp that dynamically reaches the tile op
    // must still execute it with all 32 lanes active.  Lowerings that contain
    // a real warpgroup barrier model that barrier separately.  (#594)
    scope::Warpgroup => ("participation:warpgroup", "tile.warpgroup participation", true),
    scope::Cta => ("participation:cta", "tile.cta participation", false),
);

#[inline(never)]
async fn execute_participate(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    scope: &'static str,
    name: &'static str,
    warp_local: bool,
) -> Result<(), EngineError> {
    if warp_local {
        return crate::runtime::require_full_warp_sync(context.active_mask().into_inner(), name)
            .map_err(Into::into);
    }
    let warp = engine(warp);
    let operation = warp.begin_current_operation(
        context.into_inner(),
        StaticOpId::new(site.get()),
        OperationKind::Collective,
    )?;
    warp.rendezvous_sync(Some(&operation), scope, name).await?;
    warp.finish_operation(&operation).map_err(Into::into)
}

/// Complete cooperative-grid rendezvous.
#[inline(never)]
pub async fn grid_sync(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(context, site, std::any::type_name_of_val(&grid_sync));
    let warp = engine(warp);
    let operation = warp.begin_current_operation(
        context.into_inner(),
        StaticOpId::new(site.get()),
        OperationKind::Collective,
    )?;
    warp.rendezvous_sync(Some(&operation), "grid", "grid.sync")
        .await?;
    warp.finish_operation(&operation).map_err(Into::into)
}
