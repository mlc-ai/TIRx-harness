//! Engine implementation of v2 warp-scope instruction specializations.

use std::marker::PhantomData;

use super::instruction::{instruction_variant, sync_instruction};
use super::transport::engine;
use super::{EngineError, ExecCtx, LaneMask, SiteId, WarpHandle, R};
use crate::abi::v2::mem::MemoryType;
use crate::runtime::warp_ops::{
    validate_warp_collective_participants, warp_shuffle_ptx, warp_shuffle_source_mask,
    WarpShuffleMode,
};
use crate::{OperationKind, WarpMask, WarpValue, WARP_SIZE};

/// Read the opaque execution context associated with a sealed warp handle.
#[inline(never)]
pub fn context(warp: &mut super::Engine) -> ExecCtx {
    ExecCtx::from_inner(engine(warp).context())
}

sync_instruction!(shfl_sync_spec, ShflSyncVariant, shfl_sync);
sync_instruction!(vote_sync_spec, VoteSyncVariant, vote_sync);
sync_instruction!(match_sync_spec, MatchSyncVariant, match_sync);
sync_instruction!(redux_sync_spec, ReduxSyncVariant, redux_sync);
sync_instruction!(movmatrix_spec, MovMatrixVariant, movmatrix);

/// Static variants of warp instructions. Runtime PTX operands remain in each
/// variant's `Args` tuple.
pub mod variant {
    use super::PhantomData;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Index;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Up;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Down;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Butterfly;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DataOnly;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DataAndPredicate;

    /// One `shfl.sync.mode.b32`; `T`, `Mode`, and optional destination
    /// predicate presence are all instruction spelling, not runtime flags.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Shfl<T, Mode, Dest = DataOnly>(PhantomData<fn() -> (T, Mode, Dest)>);

    /// One `match.mode.sync.type`; `T`, `Mode`, and optional destination
    /// predicate presence are all instruction spelling.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Match<T, Mode, Dest = DataOnly>(PhantomData<fn() -> (T, Mode, Dest)>);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Any;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct All;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Uniform;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Ballot;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Add;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Min;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Max;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct And;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Or;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Xor;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MinNan;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MaxNan;

    /// One `redux.sync.op.type` form.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Redux<T, Op>(PhantomData<fn() -> (T, Op)>);

    /// One `movmatrix.sync.aligned.m8n8.trans.b16` instruction.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MovMatrixB16;
}

instruction_variant! {
    [impl] movmatrix_spec, variant::MovMatrixB16,
    R<u32> => R<u32>;
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        values: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        let operation = engine(warp).begin_optional_operation(
            context.into_inner(),
            site.get(),
            OperationKind::Collective,
            false,
        )?;
        let result = engine(warp).collective_numeric_operation(operation.as_ref(), || {
            let full_members = WarpValue::splat(u32::MAX);
            validate_warp_collective_participants(
                context.active_mask().into_inner(),
                &full_members,
                &crate::DiagnosticLabel::new("movmatrix.sync.aligned.m8n8.trans.b16"),
            )?;
            let mut transposed = WarpValue::splat(0_u32);
            for destination_lane in 0..WARP_SIZE {
                let column = destination_lane / 4;
                let row_pair = destination_lane % 4;
                let source_pair = column / 2;
                let source_shift = (column % 2) * 16;
                let source_lane_low = 2 * row_pair * 4 + source_pair;
                let source_lane_high = (2 * row_pair + 1) * 4 + source_pair;
                let low = (values[source_lane_low] >> source_shift) & 0xffff;
                let high = (values[source_lane_high] >> source_shift) & 0xffff;
                transposed[destination_lane] = low | (high << 16);
            }
            Ok(R::from_inner(transposed))
        })?;
        engine(warp).finish_optional_operation(&operation)?;
        Ok(result)
    }
}

trait ShuffleType: MemoryType {}

impl ShuffleType for super::reg::variant::I32 {}
impl ShuffleType for super::reg::variant::U32 {}
impl ShuffleType for super::reg::variant::B32 {}
impl ShuffleType for super::reg::variant::F32 {}

trait ShuffleMode {
    const VALUE: WarpShuffleMode;
}

trait ShuffleExecute<T: ShuffleType>: ShuffleMode {
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: (R<u32>, R<T::Scalar>, R<u32>, R<u32>),
    ) -> Result<(R<T::Scalar>, R<bool>), EngineError>;
}

impl ShuffleMode for variant::Index {
    const VALUE: WarpShuffleMode = WarpShuffleMode::Index;
}
impl ShuffleMode for variant::Up {
    const VALUE: WarpShuffleMode = WarpShuffleMode::Up;
}
impl ShuffleMode for variant::Down {
    const VALUE: WarpShuffleMode = WarpShuffleMode::Down;
}
impl ShuffleMode for variant::Butterfly {
    const VALUE: WarpShuffleMode = WarpShuffleMode::Xor;
}

fn source_mask(
    context: ExecCtx,
    member_mask: R<u32>,
    selector: R<u32>,
    control: R<u32>,
    mode: WarpShuffleMode,
) -> Result<LaneMask, EngineError> {
    let mask = warp_shuffle_source_mask(
        context.active_mask().into_inner(),
        member_mask.inner(),
        selector.inner(),
        control.inner(),
        mode,
    )?;
    Ok(LaneMask::from_bits(mask.bits()))
}

macro_rules! shuffle_source_mask {
    ($name:ident, $mode:expr) => {
        #[inline(never)]
        pub fn $name(
            context: ExecCtx,
            member_mask: R<u32>,
            selector: R<u32>,
            control: R<u32>,
        ) -> Result<LaneMask, EngineError> {
            source_mask(context, member_mask, selector, control, $mode)
        }
    };
}

shuffle_source_mask!(shfl_source_mask_idx, WarpShuffleMode::Index);
shuffle_source_mask!(shfl_source_mask_up, WarpShuffleMode::Up);
shuffle_source_mask!(shfl_source_mask_down, WarpShuffleMode::Down);
shuffle_source_mask!(shfl_source_mask_bfly, WarpShuffleMode::Xor);

#[inline(never)]
fn execute_shuffle<T, Mode>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    (member_mask, values, selector, control): (R<u32>, R<T::Scalar>, R<u32>, R<u32>),
) -> Result<(R<T::Scalar>, R<bool>), EngineError>
where
    T: ShuffleType,
    Mode: ShuffleMode,
{
    let operation = engine(warp).begin_optional_operation(
        context.into_inner(),
        site.get(),
        OperationKind::Collective,
        false,
    )?;
    let (values, predicates) =
        engine(warp).collective_numeric_operation(operation.as_ref(), || {
            warp_shuffle_ptx(
                context.active_mask().into_inner(),
                member_mask.inner(),
                values.inner(),
                selector.inner(),
                control.inner(),
                Mode::VALUE,
            )
        })?;
    engine(warp).finish_optional_operation(&operation)?;
    Ok((R::from_inner(values), R::from_inner(predicates)))
}

macro_rules! shuffle_execute {
    ($ty:ty, $mode:ty) => {
        impl ShuffleExecute<$ty> for $mode {
            #[inline(never)]
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: (R<u32>, R<<$ty as MemoryType>::Scalar>, R<u32>, R<u32>),
            ) -> Result<(R<<$ty as MemoryType>::Scalar>, R<bool>), EngineError> {
                execute_shuffle::<$ty, $mode>(warp, context, site, args)
            }
        }
    };
}

macro_rules! shuffle_execute_all_modes {
    ($ty:ty) => {
        shuffle_execute!($ty, variant::Index);
        shuffle_execute!($ty, variant::Up);
        shuffle_execute!($ty, variant::Down);
        shuffle_execute!($ty, variant::Butterfly);
    };
}

shuffle_execute_all_modes!(super::reg::variant::I32);
shuffle_execute_all_modes!(super::reg::variant::U32);
shuffle_execute_all_modes!(super::reg::variant::B32);
shuffle_execute_all_modes!(super::reg::variant::F32);

macro_rules! shuffle_variant {
    ($destination:ty, $output:ty, $finish:expr) => {
        instruction_variant! {
            [impl<T, Mode>] shfl_sync_spec, variant::Shfl<T, Mode, $destination>
            where [
                T: ShuffleType,
                Mode: ShuffleMode + ShuffleExecute<T>,
            ],
            (R<u32>, R<T::Scalar>, R<u32>, R<u32>) => $output;
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                let (values, predicates) = Mode::execute(warp, context, site, args)?;
                Ok(($finish)(values, predicates))
            }
        }
    };
}

shuffle_variant!(variant::DataOnly, R<T::Scalar>, |values, _predicates| {
    values
});
shuffle_variant!(
    variant::DataAndPredicate,
    (R<T::Scalar>, R<bool>),
    |values, predicates| (values, predicates)
);

fn vote_predicate(
    active: WarpMask,
    member_masks: &WarpValue<u32>,
    predicates: &WarpValue<bool>,
    decide: impl Fn(&[bool]) -> bool,
) -> Result<R<bool>, crate::EngineError> {
    let members = validate_warp_collective_participants(
        active,
        member_masks,
        &crate::DiagnosticLabel::new("vote.sync"),
    )?;
    let values = (0..WARP_SIZE)
        .filter(|lane| members & (1_u32 << lane) != 0)
        .map(|lane| predicates[lane])
        .collect::<Vec<_>>();
    Ok(R::splat(decide(&values)))
}

fn execute_vote<Output, F>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    member_mask: R<u32>,
    predicates: R<bool>,
    numeric: F,
) -> Result<Output, EngineError>
where
    F: FnOnce(WarpMask, &WarpValue<u32>, &WarpValue<bool>) -> Result<Output, crate::EngineError>,
{
    let operation = engine(warp).begin_optional_operation(
        context.into_inner(),
        site.get(),
        OperationKind::Collective,
        false,
    )?;
    let result = engine(warp).collective_numeric_operation(operation.as_ref(), || {
        numeric(
            context.active_mask().into_inner(),
            member_mask.inner(),
            predicates.inner(),
        )
    })?;
    engine(warp).finish_optional_operation(&operation)?;
    Ok(result)
}

macro_rules! vote_variant {
    ($marker:ty, $output:ty, $numeric:expr) => {
        instruction_variant! {
            [impl] vote_sync_spec, $marker,
            (R<u32>, R<bool>) => $output;
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                (member_mask, predicates): Self::Args,
            ) -> Result<Self::Output, EngineError> {
                execute_vote(warp, context, site, member_mask, predicates, $numeric)
            }
        }
    };
}

vote_variant!(variant::Any, R<bool>, |active, masks, predicates| {
    vote_predicate(active, masks, predicates, |values| {
        values.iter().copied().any(|value| value)
    })
});
vote_variant!(variant::All, R<bool>, |active, masks, predicates| {
    vote_predicate(active, masks, predicates, |values| {
        values.iter().copied().all(|value| value)
    })
});
vote_variant!(variant::Uniform, R<bool>, |active, masks, predicates| {
    vote_predicate(active, masks, predicates, |values| {
        values
            .first()
            .is_none_or(|first| values.iter().all(|value| value == first))
    })
});
vote_variant!(variant::Ballot, R<u32>, |active, masks, predicates| {
    crate::runtime::warp_ops::warp_ballot_sync(active, masks, predicates).map(R::from_inner)
});

trait MatchType: MemoryType<Scalar: Eq> {}

impl MatchType for super::reg::variant::B32 {}
impl MatchType for super::reg::variant::B64 {}

trait MatchMode {
    const ALL: bool;
}

impl MatchMode for variant::Any {
    const ALL: bool = false;
}

impl MatchMode for variant::All {
    const ALL: bool = true;
}

fn match_values<T, Mode>(
    active: WarpMask,
    member_masks: &WarpValue<u32>,
    values: &WarpValue<T::Scalar>,
) -> Result<(R<u32>, R<bool>), crate::EngineError>
where
    T: MatchType,
    Mode: MatchMode,
{
    let members = validate_warp_collective_participants(
        active,
        member_masks,
        &crate::DiagnosticLabel::new("match.sync"),
    )?;
    let first = (0..WARP_SIZE)
        .find(|lane| members & (1_u32 << lane) != 0)
        .expect("validated member mask is nonempty");

    if Mode::ALL {
        let all_equal = (first + 1..WARP_SIZE)
            .filter(|lane| members & (1_u32 << lane) != 0)
            .all(|lane| values[first] == values[lane]);
        return Ok((
            R::splat(if all_equal { members } else { 0 }),
            R::splat(all_equal),
        ));
    }

    let mut matching = R::splat(0_u32);
    for destination_lane in active {
        let destination_value = values[destination_lane];
        matching[destination_lane] = (0..WARP_SIZE)
            .filter(|source_lane| members & (1_u32 << source_lane) != 0)
            .filter(|source_lane| destination_value == values[*source_lane])
            .fold(0_u32, |mask, source_lane| mask | (1_u32 << source_lane));
    }
    Ok((matching, R::splat(false)))
}

fn execute_match<T, Mode>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    (member_masks, values): (R<u32>, R<T::Scalar>),
) -> Result<(R<u32>, R<bool>), EngineError>
where
    T: MatchType,
    Mode: MatchMode,
{
    let operation = engine(warp).begin_optional_operation(
        context.into_inner(),
        site.get(),
        OperationKind::Collective,
        false,
    )?;
    let result = engine(warp).collective_numeric_operation(operation.as_ref(), || {
        match_values::<T, Mode>(
            context.active_mask().into_inner(),
            member_masks.inner(),
            values.inner(),
        )
    })?;
    engine(warp).finish_optional_operation(&operation)?;
    Ok(result)
}

instruction_variant! {
    [impl<T, Mode>] match_sync_spec, variant::Match<T, Mode, variant::DataOnly>
    where [
        T: MatchType,
        Mode: MatchMode,
    ],
    (R<u32>, R<T::Scalar>) => R<u32>;
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        execute_match::<T, Mode>(warp, context, site, args).map(|(masks, _predicate)| masks)
    }
}

instruction_variant! {
    [impl<T>] match_sync_spec, variant::Match<T, variant::All, variant::DataAndPredicate>
    where [
        T: MatchType,
    ],
    (R<u32>, R<T::Scalar>) => (R<u32>, R<bool>);
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        execute_match::<T, variant::All>(warp, context, site, args)
    }
}

trait ReduxOperation<T> {
    fn combine(lhs: T, rhs: T) -> T;
}

macro_rules! redux_operation {
    ($marker:ty, $scalar:ty, $body:expr) => {
        impl ReduxOperation<$scalar> for $marker {
            fn combine(lhs: $scalar, rhs: $scalar) -> $scalar {
                ($body)(lhs, rhs)
            }
        }
    };
}

redux_operation!(variant::Add, u32, u32::wrapping_add);
redux_operation!(variant::Add, i32, i32::wrapping_add);
redux_operation!(variant::Min, u32, u32::min);
redux_operation!(variant::Min, i32, i32::min);
redux_operation!(variant::Max, u32, u32::max);
redux_operation!(variant::Max, i32, i32::max);
redux_operation!(variant::Min, f32, |lhs, rhs| {
    crate::scalar::ptx_min_f32(lhs, rhs, false, false)
});
redux_operation!(variant::Max, f32, |lhs, rhs| {
    crate::scalar::ptx_max_f32(lhs, rhs, false, false)
});
redux_operation!(variant::MinNan, f32, |lhs, rhs| {
    crate::scalar::ptx_min_f32(lhs, rhs, false, true)
});
redux_operation!(variant::MaxNan, f32, |lhs, rhs| {
    crate::scalar::ptx_max_f32(lhs, rhs, false, true)
});
redux_operation!(variant::And, u32, |lhs, rhs| lhs & rhs);
redux_operation!(variant::Or, u32, |lhs, rhs| lhs | rhs);
redux_operation!(variant::Xor, u32, |lhs, rhs| lhs ^ rhs);

trait ReduxType: MemoryType {
    fn finish(value: Self::Scalar) -> Self::Scalar {
        value
    }
}
impl ReduxType for super::reg::variant::I32 {}
impl ReduxType for super::reg::variant::U32 {}
impl ReduxType for super::reg::variant::B32 {}
impl ReduxType for super::reg::variant::F32 {
    fn finish(value: f32) -> f32 {
        // A singleton reduction never calls combine, but still canonicalizes NaN.
        crate::scalar::cuda_canonicalize_nan_f32(value)
    }
}

trait ReduxExecute<T: ReduxType>: ReduxOperation<T::Scalar> {
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        member_masks: R<u32>,
        values: R<T::Scalar>,
    ) -> Result<R<T::Scalar>, EngineError>;
}

#[inline(never)]
fn execute_redux<T, Op>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    member_masks: R<u32>,
    values: R<T::Scalar>,
) -> Result<R<T::Scalar>, EngineError>
where
    T: ReduxType,
    Op: ReduxOperation<T::Scalar>,
{
    let operation = engine(warp).begin_optional_operation(
        context.into_inner(),
        site.get(),
        OperationKind::Collective,
        false,
    )?;
    let result = engine(warp).collective_numeric_operation(operation.as_ref(), || {
        let members = validate_warp_collective_participants(
            context.active_mask().into_inner(),
            member_masks.inner(),
            &crate::DiagnosticLabel::new("redux.sync"),
        )?;
        let first = (0..WARP_SIZE)
            .find(|lane| members & (1_u32 << lane) != 0)
            .expect("validated member mask is nonempty");
        let reduced = (first + 1..WARP_SIZE)
            .filter(|lane| members & (1_u32 << lane) != 0)
            .fold(values[first], |accumulator, lane| {
                Op::combine(accumulator, values[lane])
            });
        Ok(R::splat(T::finish(reduced)))
    })?;
    engine(warp).finish_optional_operation(&operation)?;
    Ok(result)
}

macro_rules! redux_execute {
    ($ty:ty, $op:ty) => {
        impl ReduxExecute<$ty> for $op {
            #[inline(never)]
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                member_masks: R<u32>,
                values: R<<$ty as MemoryType>::Scalar>,
            ) -> Result<R<<$ty as MemoryType>::Scalar>, EngineError> {
                execute_redux::<$ty, $op>(warp, context, site, member_masks, values)
            }
        }
    };
}

redux_execute!(super::reg::variant::I32, variant::Add);
redux_execute!(super::reg::variant::I32, variant::Min);
redux_execute!(super::reg::variant::I32, variant::Max);
redux_execute!(super::reg::variant::U32, variant::Add);
redux_execute!(super::reg::variant::U32, variant::Min);
redux_execute!(super::reg::variant::U32, variant::Max);
redux_execute!(super::reg::variant::U32, variant::And);
redux_execute!(super::reg::variant::U32, variant::Or);
redux_execute!(super::reg::variant::U32, variant::Xor);
redux_execute!(super::reg::variant::B32, variant::Add);
redux_execute!(super::reg::variant::B32, variant::Min);
redux_execute!(super::reg::variant::B32, variant::Max);
redux_execute!(super::reg::variant::B32, variant::And);
redux_execute!(super::reg::variant::B32, variant::Or);
redux_execute!(super::reg::variant::B32, variant::Xor);
redux_execute!(super::reg::variant::F32, variant::Min);
redux_execute!(super::reg::variant::F32, variant::Max);
redux_execute!(super::reg::variant::F32, variant::MinNan);
redux_execute!(super::reg::variant::F32, variant::MaxNan);

instruction_variant! {
    [impl<T, Op>] redux_sync_spec, variant::Redux<T, Op>
    where [
        T: ReduxType,
        Op: ReduxExecute<T>,
    ],
    (R<u32>, R<T::Scalar>) => R<T::Scalar>;
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (member_masks, values): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        Op::execute(warp, context, site, member_masks, values)
    }
}

/// Return the current active mask for one `activemask` instruction.
#[inline(never)]
pub fn activemask(
    _warp: &mut super::Engine,
    context: ExecCtx,
    _site: SiteId,
) -> Result<R<u32>, EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(context, _site, std::any::type_name_of_val(&activemask));
    Ok(R::splat(context.active_mask().bits()))
}

/// Deterministic leader selected by one `elect.sync` instruction.
#[inline(never)]
pub fn elect_sync(
    _warp: &mut super::Engine,
    context: ExecCtx,
    _site: SiteId,
    member_masks: R<u32>,
) -> Result<(R<u32>, R<bool>), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(context, _site, std::any::type_name_of_val(&elect_sync));
    let members = validate_warp_collective_participants(
        context.active_mask().into_inner(),
        member_masks.inner(),
        &crate::DiagnosticLabel::new("elect.sync"),
    )?;
    let elected = WarpMask::from_bits(members)
        .first_active()
        .ok_or_else(|| EngineError::message("elect.sync has no participating lane"))?;
    Ok((R::splat(elected as u32), R::from_fn(|lane| elected == lane)))
}

/// Execute one `bar.warp.sync` instruction. `member_mask` is a runtime PTX
/// operand; agreement and active-participant constraints are checked before
/// the engine-owned rendezvous is registered.
#[inline(never)]
pub async fn bar_warp_sync(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    member_mask: R<u32>,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(context, site, std::any::type_name_of_val(&bar_warp_sync));
    execute_bar_warp_sync(warp, context, site, member_mask).await
}

pub(crate) async fn execute_bar_warp_sync<W: WarpHandle + Send>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    member_mask: R<u32>,
) -> Result<(), EngineError> {
    validate_warp_collective_participants(
        context.active_mask().into_inner(),
        member_mask.inner(),
        &crate::DiagnosticLabel::new("bar.warp.sync"),
    )?;
    let operation = engine(warp).begin_optional_operation(
        context.into_inner(),
        site.get(),
        OperationKind::Collective,
        true,
    )?;
    engine(warp)
        .rendezvous_sync(operation.as_ref(), "warp", "bar.warp.sync")
        .await?;
    engine(warp).finish_optional_operation(&operation)?;
    Ok(())
}

#[cfg(all(test, not(feature = "analysis-core")))]
mod tests {
    use super::*;
    use crate::runtime::{run_kernel_engine_launch, ExecutionPolicy, LaunchSelection};
    use crate::{LaunchTopology, NumSimMode, PhysicalMemory};
    use std::sync::Arc;

    #[test]
    fn raw_shuffle_consumes_packed_ptx_control_and_returns_optional_predicate() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        run_kernel_engine_launch::<NumSimMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| async move {
                let context = ExecCtx::from_inner(warp.context());
                let (values, predicates) = shfl_sync::<
                    variant::Shfl<
                        super::super::reg::variant::U32,
                        variant::Down,
                        variant::DataAndPredicate,
                    >,
                >(
                    &mut warp,
                    context,
                    SiteId::new(1),
                    (
                        R::splat(u32::MAX),
                        R::from_fn(|lane| lane as u32),
                        R::splat(1),
                        R::splat(31),
                    ),
                )?;
                assert_eq!(values[0], 1);
                assert!(predicates[0]);
                assert_eq!(values[31], 31);
                assert!(!predicates[31]);
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn vote_match_and_redux_are_one_instruction_over_the_runtime_member_mask() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        run_kernel_engine_launch::<NumSimMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| async move {
                let context = ExecCtx::from_inner(warp.context());
                let members = R::splat(u32::MAX);
                let ballot = vote_sync::<variant::Ballot>(
                    &mut warp,
                    context,
                    SiteId::new(2),
                    (members.clone(), R::from_fn(|lane| lane % 2 == 0)),
                )?;
                assert_eq!(ballot[0], 0x5555_5555);
                let sum =
                    redux_sync::<variant::Redux<super::super::reg::variant::U32, variant::Add>>(
                        &mut warp,
                        context,
                        SiteId::new(3),
                        (members.clone(), R::splat(1)),
                    )?;
                assert_eq!(sum[0], 32);
                let matches = match_sync::<
                    variant::Match<
                        super::super::reg::variant::B32,
                        variant::Any,
                        variant::DataOnly,
                    >,
                >(
                    &mut warp,
                    context,
                    SiteId::new(4),
                    (members, R::from_fn(|lane| (lane % 4) as u32)),
                )?;
                assert_eq!(matches[0], 0x1111_1111);
                assert_eq!(matches[1], 0x2222_2222);
                let members = R::splat(u32::MAX);
                let sums =
                    redux_sync::<variant::Redux<super::super::reg::variant::U32, variant::Add>>(
                        &mut warp,
                        context,
                        SiteId::new(5),
                        (members.clone(), R::from_fn(|lane| lane as u32)),
                    )?;
                let minima =
                    redux_sync::<variant::Redux<super::super::reg::variant::U32, variant::Min>>(
                        &mut warp,
                        context,
                        SiteId::new(6),
                        (members.clone(), R::from_fn(|lane| lane as u32)),
                    )?;
                let any = vote_sync::<variant::Any>(
                    &mut warp,
                    context,
                    SiteId::new(7),
                    (members, R::from_fn(|lane| lane == 31)),
                )?;
                for lane in 0..32 {
                    assert_eq!(sums[lane], 496);
                    assert_eq!(minima[lane], 0);
                    assert!(any[lane]);
                    assert_eq!(ballot[lane], 0x5555_5555);
                }
                let active = super::super::LaneMask::from_bits((1 << 3) | (1 << 7) | (1 << 12));
                let elected_context = crate::abi::v2::control::branch_context::<
                    crate::abi::v2::control::Ordinary,
                >(context, active)?;
                let (leader, elected) = elect_sync(
                    &mut warp,
                    elected_context,
                    SiteId::new(8),
                    R::splat(active.bits()),
                )?;
                for lane in 0..32 {
                    assert_eq!(leader[lane], 3);
                    assert_eq!(elected[lane], lane == 3);
                }
                Ok(())
            },
        )
        .unwrap();
    }
}
