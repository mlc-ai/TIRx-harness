//! Engine implementation of v2 barrier and fence instruction specializations.
//!
//! A public function selects the mnemonic.  Marker types in [`variant`]
//! select only compile-time PTX qualifiers; runtime operands stay in
//! `Variant::Args`.

use std::marker::PhantomData;

use super::instruction::{async_instruction, instruction_variant, sync_instruction};
use super::transport::engine;
use super::{Address, EngineError, ExecCtx, Shared, SiteId, WarpHandle, R};
use crate::{MemoryOrder, MemoryProxy, MemoryScope, OperationKind};

sync_instruction!(
    barrier_cluster_arrive_spec,
    BarrierClusterArriveVariant,
    barrier_cluster_arrive
);
async_instruction!(
    barrier_cluster_wait_spec,
    BarrierClusterWaitVariant,
    barrier_cluster_wait
);
sync_instruction!(mbarrier_init_spec, MbarrierInitVariant, mbarrier_init);
sync_instruction!(mbarrier_arrive_spec, MbarrierArriveVariant, mbarrier_arrive);
sync_instruction!(
    mbarrier_expect_tx_spec,
    MbarrierExpectTxVariant,
    mbarrier_expect_tx
);
sync_instruction!(
    mbarrier_arrive_expect_tx_spec,
    MbarrierArriveExpectTxVariant,
    mbarrier_arrive_expect_tx
);
sync_instruction!(
    mbarrier_complete_tx_spec,
    MbarrierCompleteTxVariant,
    mbarrier_complete_tx
);
sync_instruction!(
    mbarrier_test_wait_spec,
    MbarrierTestWaitVariant,
    mbarrier_test_wait
);
async_instruction!(
    mbarrier_try_wait_spec,
    MbarrierTryWaitVariant,
    mbarrier_try_wait
);
async_instruction!(
    mbarrier_wait_until_spec,
    MbarrierWaitUntilVariant,
    mbarrier_wait_until
);
sync_instruction!(fence_spec, FenceVariant, fence);
sync_instruction!(
    tensor_map_replace_spec,
    TensorMapReplaceVariant,
    tensormap_replace
);
sync_instruction!(
    tensor_map_copy_spec,
    TensorMapCopyVariant,
    tensormap_cp_fenceproxy
);
sync_instruction!(
    fence_proxy_async_spec,
    FenceProxyAsyncVariant,
    fence_proxy_async
);

/// Compile-time qualifiers for synchronization instructions.
pub mod variant {
    use super::PhantomData;

    pub struct TensorMapCopy<Scope>(PhantomData<fn() -> Scope>);
    pub struct TensorMapField<Space>(PhantomData<fn() -> Space>);
    pub struct TensorMapAddress<Space>(PhantomData<fn() -> Space>);

    pub struct Report<V>(PhantomData<fn(V)>);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DefaultRelease;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Release;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Acquire;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct AcqRel;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Sc;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Relaxed;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Cta;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Cluster;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Gpu;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Sys;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct All;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Global;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SharedCta;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SharedCluster;

    /// `barrier.cluster.arrive` spelling. `ALIGNED` is a PTX qualifier, not a
    /// runtime lane predicate.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ClusterArrive<Semantics, const ALIGNED: bool>(PhantomData<fn() -> Semantics>);

    /// `barrier.cluster.wait` spelling.  The wait side carries no semantics
    /// axis: PTX defines the unqualified form as acquire and the engine
    /// hardcodes an acquiring wait, so default and explicit acquire were the
    /// same single modeled behaviour.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ClusterWait<const ALIGNED: bool>;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MbarrierInit<const V1: bool = false>;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ArriveLocal<const DROP: bool = false, const RELEASE: bool = true>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ArriveLocalPredicated<const DROP: bool = false, const RELEASE: bool = true>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ArriveLocalCount<const DROP: bool = false, const RELEASE: bool = true>;
    /// CTA-local counted arrival returning the opaque pre-arrival phase state.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ArriveLocalCountState<
        const NO_COMPLETE: bool = false,
        const DROP: bool = false,
        const RELEASE: bool = true,
    >;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ArriveLocalCountPredicated<const DROP: bool = false, const RELEASE: bool = true>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ArriveRemote<const DROP: bool = false, const RELEASE: bool = true>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ArriveRemotePredicated<const DROP: bool = false, const RELEASE: bool = true>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ArriveRemoteCount<const DROP: bool = false, const RELEASE: bool = true>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ArriveRemoteCountPredicated<const DROP: bool = false, const RELEASE: bool = true>;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ArriveExpectTxLocal<const DROP: bool = false, const RELEASE: bool = true>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ArriveExpectTxLocalState<const DROP: bool = false, const RELEASE: bool = true>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ArriveExpectTxLocalPredicated<const DROP: bool = false, const RELEASE: bool = true>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ArriveExpectTxRemote<const DROP: bool = false, const RELEASE: bool = true>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ArriveExpectTxRemotePredicated<const DROP: bool = false, const RELEASE: bool = true>;

    /// Apply an existing barrier transition to every lane-selected cluster CTA.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Multicast<V>(pub PhantomData<V>);

    /// Exact standalone `mbarrier.expect_tx` spelling.  The operation changes
    /// no arrival count; predication is represented by `ExecCtx`, not a fake
    /// instruction operand.  The PTX semantics/scope/address-space qualifiers
    /// select no modeled behaviour and do not type the operand carrier, so the
    /// mnemonic has exactly one specialization.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ExpectTx;

    /// `mbarrier.complete_tx.relaxed.<scope>`; address-space and operand
    /// presence are closed by the concrete marker aliases below.  The scope
    /// qualifier selects no modeled behaviour and is not part of the marker.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CompleteTxLocal;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CompleteTxLocalPredicated;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CompleteTxRemote;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CompleteTxRemotePredicated;

    /// Exact parity and opaque state-token query forms.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TestWaitParity<const CONDITIONAL: bool = false>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TestWaitParityRelaxed<const CONDITIONAL: bool = false>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TryWaitParity<const CONDITIONAL: bool = false>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TryWaitParityRelaxed<const CONDITIONAL: bool = false>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TryWaitState;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TryWaitStateRelaxed;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TestWaitState;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TestWaitStateRelaxed;

    /// Engine scheduling protocol for a frontend blocking wait, implemented
    /// as repeated `mbarrier.try_wait` attempts. This is not another PTX
    /// mnemonic: it exists because an external artifact cannot drive the
    /// engine's blocked-warp scheduler itself.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct WaitUntilParity;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Fence<Scope, Order = AcqRel>(PhantomData<fn() -> (Scope, Order)>);
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ProxyAsync<Space>(PhantomData<fn() -> Space>);
}

fn begin(
    warp: &mut impl WarpHandle,
    context: ExecCtx,
    site: SiteId,
    kind: OperationKind,
    required: bool,
) -> Result<Option<crate::OperationContext>, EngineError> {
    engine(warp)
        .begin_optional_operation(context.into_inner(), site.get(), kind, required)
        .map_err(Into::into)
}

fn finish(
    warp: &mut impl WarpHandle,
    operation: &Option<crate::OperationContext>,
) -> Result<(), EngineError> {
    engine(warp)
        .finish_optional_operation(operation)
        .map_err(Into::into)
}

fn selected_context(context: ExecCtx, predicate: Option<&R<bool>>) -> ExecCtx {
    let Some(predicate) = predicate else {
        return context;
    };
    let mask = context.active_mask() & predicate.to_mask(|_, value| *value);
    ExecCtx::from_inner(context.into_inner().with_active_mask(mask.into_inner()))
}

fn multicast_mbarrier(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    barrier: Address<Shared>,
    amount: R<i64>,
    masks: R<u32>,
    mut execute: impl FnMut(
        &mut super::Engine,
        ExecCtx,
        SiteId,
        (Address<Shared>, R<i64>),
    ) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    let ctx = context.into_inner();
    let ctas = ctx.topology().ctas_per_cluster();
    for lane in ctx.active_mask() {
        if ctas < 32 && (masks[lane] >> ctas) != 0 {
            return Err(EngineError::message(
                "mbarrier multicast mask targets a CTA outside the cluster",
            ));
        }
    }
    for target in 0..ctas.min(32) {
        let mask = ctx.active_mask()
            & masks
                .to_mask(|_, bits| bits & (1 << target) != 0)
                .into_inner();
        if mask.is_empty() {
            continue;
        }
        let pointer =
            barrier
                .inner()
                .map_shared_rank(&ctx, &crate::WarpValue::splat(target as i64), mask)?;
        execute(
            warp,
            ExecCtx::from_inner(ctx.with_active_mask(mask)),
            site,
            (Address::from_inner(pointer), amount.clone()),
        )?;
    }
    Ok(())
}

macro_rules! multicast_mbarrier_variant {
    ($spec:ident) => {
        impl<V: $spec::Variant<Args = (Address<Shared>, R<i64>), Output = ()>> $spec::sealed::Sealed
            for variant::Multicast<V>
        {
        }
        impl<V: $spec::Variant<Args = (Address<Shared>, R<i64>), Output = ()>> $spec::Variant
            for variant::Multicast<V>
        {
            type Args = (Address<Shared>, R<i64>, R<u32>);
            type Output = ();
        }
        impl<
                V: $spec::Variant<Args = (Address<Shared>, R<i64>), Output = ()>
                    + $spec::sealed::Execute,
            > $spec::sealed::Execute for variant::Multicast<V>
        {
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
            ) -> Result<(), EngineError> {
                multicast_mbarrier(
                    warp,
                    context,
                    site,
                    args.0,
                    args.1,
                    args.2,
                    <V as $spec::sealed::Execute>::execute,
                )
            }
        }
    };
}

multicast_mbarrier_variant!(mbarrier_arrive_spec);
multicast_mbarrier_variant!(mbarrier_arrive_expect_tx_spec);
multicast_mbarrier_variant!(mbarrier_expect_tx_spec);
multicast_mbarrier_variant!(mbarrier_complete_tx_spec);

/// Execute one `bar.arrive` instruction.
#[inline(never)]
pub fn bar_arrive(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    barrier_id: i64,
    expected_count: i64,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(context, site, std::any::type_name_of_val(&bar_arrive));
    let operation = begin(warp, context, site, OperationKind::Collective, false)?;
    engine(warp).named_barrier_arrive(
        operation.as_ref(),
        barrier_id,
        expected_count,
        context.active_mask().into_inner(),
    )?;
    finish(warp, &operation)
}

async fn named_barrier_sync(
    warp: &mut (impl WarpHandle + Send),
    context: ExecCtx,
    site: SiteId,
    barrier_id: i64,
    expected_count: i64,
    aligned: bool,
) -> Result<(), EngineError> {
    let operation = begin(warp, context, site, OperationKind::Collective, false)?;
    engine(warp)
        .named_barrier_sync_with_alignment(
            operation.as_ref(),
            barrier_id,
            expected_count,
            context.active_mask().into_inner(),
            aligned,
        )
        .await?;
    finish(warp, &operation)
}

/// Execute one aligned `bar.sync` instruction.
#[inline(never)]
pub async fn bar_sync(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    barrier_id: i64,
    expected_count: i64,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(context, site, std::any::type_name_of_val(&bar_sync));
    execute_bar_sync(warp, context, site, barrier_id, expected_count).await
}

pub(crate) async fn execute_bar_sync<W: WarpHandle + Send>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    barrier_id: i64,
    expected_count: i64,
) -> Result<(), EngineError> {
    named_barrier_sync(warp, context, site, barrier_id, expected_count, true).await
}

/// Execute one unaligned `barrier.sync` instruction.  It shares the private
/// named-barrier protocol with [`bar_sync`] but remains a distinct ABI call
/// because it is a distinct PTX mnemonic.
#[inline(never)]
pub async fn barrier_sync(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    barrier_id: i64,
    expected_count: i64,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(context, site, std::any::type_name_of_val(&barrier_sync));
    named_barrier_sync(warp, context, site, barrier_id, expected_count, false).await
}

mod static_sealed {
    pub trait ArrivalSemantics {}
    pub trait FenceScope {}
    pub trait FenceOrder {}
    pub trait ProxySpace {}
}

trait ArrivalSemantics: static_sealed::ArrivalSemantics {
    const PUBLISHES_MEMORY: bool;
}

macro_rules! arrival_semantics {
    ($marker:ty, $publishes:expr) => {
        impl static_sealed::ArrivalSemantics for $marker {}
        impl ArrivalSemantics for $marker {
            const PUBLISHES_MEMORY: bool = $publishes;
        }
    };
}

arrival_semantics!(variant::DefaultRelease, true);
arrival_semantics!(variant::Release, true);
arrival_semantics!(variant::Relaxed, false);

instruction_variant! {
    [impl<S: ArrivalSemantics, const ALIGNED: bool>]
    barrier_cluster_arrive_spec, variant::ClusterArrive<S, ALIGNED>,
    () => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        cluster_barrier_arrive_entry(warp, context, site, S::PUBLISHES_MEMORY, ALIGNED)
    }
}

#[inline(never)]
fn cluster_barrier_arrive_entry(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    publishes_memory: bool,
    aligned: bool,
) -> Result<(), EngineError> {
    let operation = begin(warp, context, site, OperationKind::Collective, false)?;
    engine(warp).cluster_barrier_arrive(
        operation.as_ref(),
        context.active_mask().into_inner(),
        publishes_memory,
        aligned,
    )?;
    finish(warp, &operation)
}

instruction_variant! {
    [impl<const ALIGNED: bool>] barrier_cluster_wait_spec, variant::ClusterWait<ALIGNED>,
    () => ();
    async fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        cluster_barrier_wait_entry(warp, context, site, ALIGNED).await
    }
}

#[inline(never)]
async fn cluster_barrier_wait_entry(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    aligned: bool,
) -> Result<(), EngineError> {
    let operation = begin(warp, context, site, OperationKind::Collective, false)?;
    engine(warp)
        .cluster_barrier_wait(
            operation.as_ref(),
            context.active_mask().into_inner(),
            aligned,
        )
        .await?;
    finish(warp, &operation)
}

impl<const V1: bool> mbarrier_init_spec::sealed::Sealed for variant::MbarrierInit<V1> {}
pub fn mbarrier_inval(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    barrier: Address<Shared>,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(context, site, std::any::type_name_of_val(&mbarrier_inval));
    let operation = begin(warp, context, site, OperationKind::Barrier, false)?;
    engine(warp).mbarrier_invalidate(
        operation.as_ref(),
        barrier.inner(),
        context.active_mask().into_inner(),
    )?;
    finish(warp, &operation)
}
pub fn mbarrier_check_layout<const LAYOUT: u8>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    barrier: Address<Shared>,
) -> Result<R<bool>, EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(
        context,
        site,
        std::any::type_name_of_val(&mbarrier_check_layout::<LAYOUT>),
    );
    if LAYOUT > 1 {
        return Err(EngineError::message(
            "mbarrier.check_layout requires v0 or v1",
        ));
    }
    let operation = begin(warp, context, site, OperationKind::Control, false)?;
    let result = engine(warp)
        .mbarrier_check_layout(barrier.inner(), context.active_mask().into_inner(), LAYOUT)
        .map_err(|error| match operation.as_ref() {
            Some(operation) => error.with_operation_context(operation),
            None => error,
        })?;
    finish(warp, &operation)?;
    Ok(R::from_inner(result))
}
impl<const V1: bool> mbarrier_init_spec::Variant for variant::MbarrierInit<V1> {
    type Args = (Address<Shared>, R<i64>);
    type Output = ();
}

impl<const V1: bool> mbarrier_init_spec::sealed::Execute for variant::MbarrierInit<V1> {
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (barrier, expected): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        let operation = begin(warp, context, site, OperationKind::Barrier, false)?;
        engine(warp).mbarrier_init_layout(
            operation.as_ref(),
            barrier.inner(),
            context.active_mask().into_inner(),
            expected.inner(),
            V1,
        )?;
        finish(warp, &operation)
    }
}

trait ArriveOperands {
    type Args;
    const REMOTE: bool;

    fn split(args: Self::Args) -> (Address<Shared>, Option<R<i64>>, Option<R<bool>>);
}

macro_rules! arrive_operands {
    ($marker:ty, $args:ty, $remote:expr, |$value:ident| $split:expr) => {
        impl<const DROP: bool, const RELEASE: bool> ArriveOperands for $marker {
            type Args = $args;
            const REMOTE: bool = $remote;

            fn split($value: Self::Args) -> (Address<Shared>, Option<R<i64>>, Option<R<bool>>) {
                $split
            }
        }
    };
}

arrive_operands!(
    variant::ArriveLocal<DROP, RELEASE>,
    Address<Shared>,
    false,
    |barrier| (barrier, None, None)
);
arrive_operands!(
    variant::ArriveLocalPredicated<DROP, RELEASE>,
    (Address<Shared>, R<bool>),
    false,
    |args| (args.0, None, Some(args.1))
);
arrive_operands!(
    variant::ArriveLocalCount<DROP, RELEASE>,
    (Address<Shared>, R<i64>),
    false,
    |args| (args.0, Some(args.1), None)
);
arrive_operands!(
    variant::ArriveLocalCountPredicated<DROP, RELEASE>,
    (Address<Shared>, R<i64>, R<bool>),
    false,
    |args| (args.0, Some(args.1), Some(args.2))
);
arrive_operands!(
    variant::ArriveRemote<DROP, RELEASE>,
    Address<Shared>,
    true,
    |barrier| (barrier, None, None)
);
arrive_operands!(
    variant::ArriveRemotePredicated<DROP, RELEASE>,
    (Address<Shared>, R<bool>),
    true,
    |args| (args.0, None, Some(args.1))
);
arrive_operands!(
    variant::ArriveRemoteCount<DROP, RELEASE>,
    (Address<Shared>, R<i64>),
    true,
    |args| (args.0, Some(args.1), None)
);
arrive_operands!(
    variant::ArriveRemoteCountPredicated<DROP, RELEASE>,
    (Address<Shared>, R<i64>, R<bool>),
    true,
    |args| (args.0, Some(args.1), Some(args.2))
);

macro_rules! mbarrier_arrive_variant {
    ($marker:ty, $args:ty) => {
        instruction_variant! {
            [impl<const DROP: bool, const RELEASE: bool>] mbarrier_arrive_spec, $marker,
            $args => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                let (barrier, count, predicate) = <$marker as ArriveOperands>::split(args);
                let context = selected_context(context, predicate.as_ref());
                let target = if <$marker as ArriveOperands>::REMOTE {
                    Some(barrier.inner().shared_target_cta_ranks(
                        &context.into_inner(),
                        context.active_mask().into_inner(),
                    )?)
                } else {
                    None
                };
                let operation = begin(warp, context, site, OperationKind::Barrier, false)?;
                engine(warp).mbarrier_arrive::<false, DROP>(
                    operation.as_ref(),
                    barrier.inner(),
                    context.active_mask().into_inner(),
                    target.as_ref(),
                    count.as_ref().map(R::inner),
                    None,
                    RELEASE,
                )?;
                finish(warp, &operation)
            }
        }
    };
}

mbarrier_arrive_variant!(variant::ArriveLocal<DROP, RELEASE>, Address<Shared>);
mbarrier_arrive_variant!(
    variant::ArriveLocalPredicated<DROP, RELEASE>,
    (Address<Shared>, R<bool>)
);
mbarrier_arrive_variant!(variant::ArriveLocalCount<DROP, RELEASE>, (Address<Shared>, R<i64>));
mbarrier_arrive_variant!(
    variant::ArriveLocalCountPredicated<DROP, RELEASE>,
    (Address<Shared>, R<i64>, R<bool>)
);

instruction_variant! {
    [impl<const NO_COMPLETE: bool, const DROP: bool, const RELEASE: bool>]
    mbarrier_arrive_spec, variant::ArriveLocalCountState<NO_COMPLETE, DROP, RELEASE>,
    (Address<Shared>, R<i64>) => R<u64>;
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (barrier, count): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        let operation = begin(warp, context, site, OperationKind::Barrier, false)?;
        let states = engine(warp).mbarrier_arrive::<NO_COMPLETE, DROP>(
            operation.as_ref(),
            barrier.inner(),
            context.active_mask().into_inner(),
            None,
            Some(count.inner()),
            None,
            RELEASE,
        )?;
        finish(warp, &operation)?;
        Ok(R::from_inner(states))
    }
}
mbarrier_arrive_variant!(variant::ArriveRemote<DROP, RELEASE>, Address<Shared>);
mbarrier_arrive_variant!(
    variant::ArriveRemotePredicated<DROP, RELEASE>,
    (Address<Shared>, R<bool>)
);
mbarrier_arrive_variant!(variant::ArriveRemoteCount<DROP, RELEASE>, (Address<Shared>, R<i64>));
mbarrier_arrive_variant!(
    variant::ArriveRemoteCountPredicated<DROP, RELEASE>,
    (Address<Shared>, R<i64>, R<bool>)
);

trait ArriveExpectTxOperands {
    type Args;
    const REMOTE: bool;
    fn split(args: Self::Args) -> (Address<Shared>, R<i64>, Option<R<bool>>);
}

macro_rules! arrive_expect_tx_operands {
    ($marker:ty, $args:ty, $remote:expr, |$value:ident| $split:expr) => {
        impl<const DROP: bool, const RELEASE: bool> ArriveExpectTxOperands for $marker {
            type Args = $args;
            const REMOTE: bool = $remote;
            fn split($value: Self::Args) -> (Address<Shared>, R<i64>, Option<R<bool>>) {
                $split
            }
        }
    };
}

arrive_expect_tx_operands!(
    variant::ArriveExpectTxLocal<DROP, RELEASE>,
    (Address<Shared>, R<i64>),
    false,
    |args| (args.0, args.1, None)
);
arrive_expect_tx_operands!(
    variant::ArriveExpectTxLocalState<DROP, RELEASE>,
    (Address<Shared>, R<i64>),
    false,
    |args| (args.0, args.1, None)
);
arrive_expect_tx_operands!(
    variant::ArriveExpectTxLocalPredicated<DROP, RELEASE>,
    (Address<Shared>, R<i64>, R<bool>),
    false,
    |args| (args.0, args.1, Some(args.2))
);
arrive_expect_tx_operands!(
    variant::ArriveExpectTxRemote<DROP, RELEASE>,
    (Address<Shared>, R<i64>),
    true,
    |args| (args.0, args.1, None)
);
arrive_expect_tx_operands!(
    variant::ArriveExpectTxRemotePredicated<DROP, RELEASE>,
    (Address<Shared>, R<i64>, R<bool>),
    true,
    |args| (args.0, args.1, Some(args.2))
);

macro_rules! mbarrier_arrive_expect_tx_variant {
    ($marker:ty, $args:ty) => {
        mbarrier_arrive_expect_tx_variant!($marker, $args, (), |_| ());
    };
    ($marker:ty, $args:ty, $output:ty, $result:expr) => {
        instruction_variant! {
            [impl<const DROP: bool, const RELEASE: bool>] mbarrier_arrive_expect_tx_spec, $marker,
            $args => $output;
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                let (barrier, transactions, predicate) =
                    <$marker as ArriveExpectTxOperands>::split(args);
                let context = selected_context(context, predicate.as_ref());
                let target = if <$marker as ArriveExpectTxOperands>::REMOTE {
                    Some(barrier.inner().shared_target_cta_ranks(
                        &context.into_inner(),
                        context.active_mask().into_inner(),
                    )?)
                } else {
                    None
                };
                let operation = begin(warp, context, site, OperationKind::Barrier, false)?;
                let states = engine(warp).mbarrier_arrive::<false, DROP>(
                    operation.as_ref(),
                    barrier.inner(),
                    context.active_mask().into_inner(),
                    target.as_ref(),
                    None,
                    Some(transactions.inner()),
                    RELEASE,
                )?;
                finish(warp, &operation)?;
                Ok(($result)(states))
            }
        }
    };
}

mbarrier_arrive_expect_tx_variant!(
    variant::ArriveExpectTxLocal<DROP, RELEASE>,
    (Address<Shared>, R<i64>)
);
mbarrier_arrive_expect_tx_variant!(
    variant::ArriveExpectTxLocalState<DROP, RELEASE>,
    (Address<Shared>, R<i64>),
    R<u64>,
    R::from_inner
);
mbarrier_arrive_expect_tx_variant!(
    variant::ArriveExpectTxLocalPredicated<DROP, RELEASE>,
    (Address<Shared>, R<i64>, R<bool>)
);
mbarrier_arrive_expect_tx_variant!(
    variant::ArriveExpectTxRemote<DROP, RELEASE>,
    (Address<Shared>, R<i64>)
);
mbarrier_arrive_expect_tx_variant!(
    variant::ArriveExpectTxRemotePredicated<DROP, RELEASE>,
    (Address<Shared>, R<i64>, R<bool>)
);

instruction_variant! {
    [impl] mbarrier_expect_tx_spec, variant::ExpectTx,
    (Address<Shared>, R<i64>) => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (barrier, transactions): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        mbarrier_expect_tx_entry(warp, context, site, barrier, transactions)
    }
}

#[inline(never)]
fn mbarrier_expect_tx_entry(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    barrier: Address<Shared>,
    transactions: R<i64>,
) -> Result<(), EngineError> {
    let operation = begin(warp, context, site, OperationKind::Barrier, false)?;
    engine(warp).mbarrier_expect_tx(
        operation.as_ref(),
        barrier.inner(),
        context.active_mask().into_inner(),
        transactions.inner(),
    )?;
    finish(warp, &operation)
}

trait CompleteTxOperands {
    type Args;
    fn split(args: Self::Args) -> (Address<Shared>, R<i64>, Option<R<i64>>, Option<R<bool>>);
}

macro_rules! complete_tx_operands {
    ($marker:ty, $args:ty, |$value:ident| $split:expr) => {
        impl CompleteTxOperands for $marker {
            type Args = $args;
            fn split(
                $value: Self::Args,
            ) -> (Address<Shared>, R<i64>, Option<R<i64>>, Option<R<bool>>) {
                $split
            }
        }
    };
}

complete_tx_operands!(
    variant::CompleteTxLocal,
    (Address<Shared>, R<i64>),
    |args| (args.0, args.1, None, None)
);
complete_tx_operands!(
    variant::CompleteTxLocalPredicated,
    (Address<Shared>, R<i64>, R<bool>),
    |args| (args.0, args.1, None, Some(args.2))
);
complete_tx_operands!(
    variant::CompleteTxRemote,
    (Address<Shared>, R<i64>, R<i64>),
    |args| (args.0, args.1, Some(args.2), None)
);
complete_tx_operands!(
    variant::CompleteTxRemotePredicated,
    (Address<Shared>, R<i64>, R<i64>, R<bool>),
    |args| (args.0, args.1, Some(args.2), Some(args.3))
);

fn transaction_bytes(values: &R<i64>, mask: super::LaneMask) -> Result<R<u64>, EngineError> {
    let mut failure = None;
    let converted = R::from_fn(|lane| match u64::try_from(values[lane]) {
        Ok(value) => value,
        Err(_) if mask.contains(lane) => {
            failure.get_or_insert(lane);
            0
        }
        Err(_) => 0,
    });
    if let Some(lane) = failure {
        return Err(EngineError::message(format!(
            "negative mbarrier transaction count in active lane {lane}"
        )));
    }
    Ok(converted)
}

macro_rules! mbarrier_complete_tx_variant {
    ($marker:ty, $args:ty) => {
        instruction_variant! {
            [impl] mbarrier_complete_tx_spec, $marker,
            $args => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                let (barrier, transactions, target, predicate) =
                    <$marker as CompleteTxOperands>::split(args);
                let context = selected_context(context, predicate.as_ref());
                let mask = context.active_mask();
                let transactions = transaction_bytes(&transactions, mask)?;
                let multicast_masks = target.map(|targets| {
                    R::from_fn(|lane| {
                        u32::try_from(targets[lane])
                            .ok()
                            .and_then(|target| 1_i64.checked_shl(target))
                            .unwrap_or(0)
                    })
                });
                let operation = begin(warp, context, site, OperationKind::AsyncIssue, true)?;
                engine(warp).mbarrier_complete_tx(
                    operation.as_ref(),
                    barrier.inner(),
                    mask.into_inner(),
                    multicast_masks.as_ref().map(R::inner),
                    transactions.inner(),
                )?;
                finish(warp, &operation)
            }
        }
    };
}

mbarrier_complete_tx_variant!(variant::CompleteTxLocal, (Address<Shared>, R<i64>));
mbarrier_complete_tx_variant!(
    variant::CompleteTxLocalPredicated,
    (Address<Shared>, R<i64>, R<bool>)
);
mbarrier_complete_tx_variant!(variant::CompleteTxRemote, (Address<Shared>, R<i64>, R<i64>));
mbarrier_complete_tx_variant!(
    variant::CompleteTxRemotePredicated,
    (Address<Shared>, R<i64>, R<i64>, R<bool>)
);

instruction_variant! {
    [impl<const CONDITIONAL: bool>] mbarrier_test_wait_spec, variant::TestWaitParity<CONDITIONAL>,
    (Address<Shared>, R<i64>) => R<bool>;
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (barrier, parity): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        mbarrier_test_wait_entry::<CONDITIONAL>(warp, context, site, barrier, parity, true)
    }
}

instruction_variant! {
    [impl<const CONDITIONAL: bool>]
    mbarrier_test_wait_spec, variant::TestWaitParityRelaxed<CONDITIONAL>,
    (Address<Shared>, R<i64>) => R<bool>;
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (barrier, parity): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        mbarrier_test_wait_entry::<CONDITIONAL>(warp, context, site, barrier, parity, false)
    }
}

/// Shared numeric entry for nonblocking parity queries. Default/acquire forms
/// publish an acquire edge only when the returned predicate is true; explicit
/// relaxed forms preserve the same readiness without that edge.
#[inline(never)]
fn mbarrier_test_wait_entry<const CONDITIONAL: bool>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    barrier: Address<Shared>,
    parity: R<i64>,
    acquire_on_true: bool,
) -> Result<R<bool>, EngineError> {
    let operation = begin(warp, context, site, OperationKind::Barrier, false)?;
    let (ready, _) = engine(warp).mbarrier_test_wait_instruction::<CONDITIONAL>(
        operation.as_ref(),
        barrier.inner(),
        context.active_mask().into_inner(),
        parity.inner(),
        acquire_on_true,
    )?;
    finish(warp, &operation)?;
    Ok(R::from_inner(ready.map(|_, value| value != 0)))
}

/// Shared numeric entry for opaque state-token queries.  The token stays a
/// full 64-bit value until the barrier hub validates that it names the current
/// or immediately preceding generation.
#[inline(never)]
fn mbarrier_test_wait_state_entry(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    barrier: Address<Shared>,
    state: R<u64>,
    acquire_on_true: bool,
) -> Result<R<bool>, EngineError> {
    let operation = begin(warp, context, site, OperationKind::Barrier, false)?;
    let (ready, _) = engine(warp).mbarrier_test_wait_state_instruction(
        operation.as_ref(),
        barrier.inner(),
        context.active_mask().into_inner(),
        state.inner(),
        acquire_on_true,
    )?;
    finish(warp, &operation)?;
    Ok(R::from_inner(ready.map(|_, value| value != 0)))
}

/// Query only the arrival's register snapshot; this is not a barrier acquire.
pub fn mbarrier_pending_count(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    state: R<u64>,
) -> Result<R<u32>, EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(
        context,
        site,
        std::any::type_name_of_val(&mbarrier_pending_count),
    );
    let operation = begin(warp, context, site, OperationKind::Control, false)?;
    let mut counts = crate::WarpValue::splat(0_u32);
    for lane in context.active_mask().into_inner() {
        counts[lane] = crate::hardware_barriers::mbarrier_state_pending_count(state[lane])?;
    }
    finish(warp, &operation)?;
    Ok(R::from_inner(counts))
}

impl<const CONDITIONAL: bool> mbarrier_try_wait_spec::sealed::Sealed
    for variant::TryWaitParity<CONDITIONAL>
{
}

impl<const CONDITIONAL: bool> mbarrier_try_wait_spec::Variant
    for variant::TryWaitParity<CONDITIONAL>
{
    // NumSim deterministically chooses the permitted zero-suspension
    // execution, so the optional PTX suspend-time hint cannot affect logical
    // readiness and is not an operand of this instruction.
    type Args = (Address<Shared>, R<i64>);
    type Output = R<bool>;
}

impl<const CONDITIONAL: bool> mbarrier_try_wait_spec::sealed::Execute
    for variant::TryWaitParity<CONDITIONAL>
{
    async fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (barrier, parity): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        mbarrier_test_wait_entry::<CONDITIONAL>(warp, context, site, barrier, parity, true)
    }
}

instruction_variant! {
    [impl<const CONDITIONAL: bool>]
    mbarrier_try_wait_spec, variant::TryWaitParityRelaxed<CONDITIONAL>,
    (Address<Shared>, R<i64>) => R<bool>;
    async fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (barrier, parity): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        mbarrier_test_wait_entry::<CONDITIONAL>(warp, context, site, barrier, parity, false)
    }
}

macro_rules! mbarrier_state_query_variant {
    ($spec:ident, $marker:ty, $acquire:expr $(, $async:ident)?) => {
        impl $spec::sealed::Sealed for $marker {}
        impl $spec::Variant for $marker {
            type Args = (Address<Shared>, R<u64>);
            type Output = R<bool>;
        }
        impl $spec::sealed::Execute for $marker {
            $($async)? fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                (barrier, state): Self::Args,
            ) -> Result<Self::Output, EngineError> {
                mbarrier_test_wait_state_entry(warp, context, site, barrier, state, $acquire)
            }
        }
    };
}

mbarrier_state_query_variant!(mbarrier_try_wait_spec, variant::TryWaitState, true, async);
mbarrier_state_query_variant!(
    mbarrier_try_wait_spec,
    variant::TryWaitStateRelaxed,
    false,
    async
);
mbarrier_state_query_variant!(mbarrier_test_wait_spec, variant::TestWaitState, true);
mbarrier_state_query_variant!(
    mbarrier_test_wait_spec,
    variant::TestWaitStateRelaxed,
    false
);

// Report variants select additional outputs of the same physical query,
// not a second read of mutable barrier state after the readiness test.
macro_rules! report_query_variant {
    ($spec:ident, $marker:ident, $ty:ty, $query:ident $(<$conditional:literal>)?, $acquire:expr $(, $async:ident)?) => {
        impl $spec::sealed::Sealed for variant::Report<variant::$marker> {}
        impl $spec::Variant for variant::Report<variant::$marker> {
            type Args = (Address<Shared>, R<$ty>);
            type Output = (R<bool>, R<bool>, R<u8>);
        }
        impl $spec::sealed::Execute for variant::Report<variant::$marker> {
            $($async)? fn execute(
                warp: &mut super::Engine, context: ExecCtx, site: SiteId,
                (barrier, phase): Self::Args,
            ) -> Result<Self::Output, EngineError> {
                let operation = begin(warp, context, site, OperationKind::Barrier, false)?;
                let (ready, report) = engine(warp).$query $(::<$conditional>)? (
                    operation.as_ref(), barrier.inner(), context.active_mask().into_inner(),
                    phase.inner(), $acquire,
                )?;
                finish(warp, &operation)?;
                // Copy validity reports only OR zero into the value register.
                Ok((
                    R::from_inner(ready.map(|_, value| value != 0)),
                    R::from_inner(report),
                    R::from_inner(crate::WarpValue::splat(0u8)),
                ))
            }
        }
    };
}
report_query_variant!(
    mbarrier_test_wait_spec,
    TestWaitParity,
    i64,
    mbarrier_test_wait_instruction<false>,
    true
);
report_query_variant!(
    mbarrier_test_wait_spec,
    TestWaitParityRelaxed,
    i64,
    mbarrier_test_wait_instruction<false>,
    false
);
report_query_variant!(
    mbarrier_test_wait_spec,
    TestWaitState,
    u64,
    mbarrier_test_wait_state_instruction,
    true
);
report_query_variant!(
    mbarrier_test_wait_spec,
    TestWaitStateRelaxed,
    u64,
    mbarrier_test_wait_state_instruction,
    false
);
report_query_variant!(
    mbarrier_try_wait_spec,
    TryWaitParity,
    i64,
    mbarrier_test_wait_instruction<false>,
    true,
    async
);
report_query_variant!(
    mbarrier_try_wait_spec,
    TryWaitParityRelaxed,
    i64,
    mbarrier_test_wait_instruction<false>,
    false,
    async
);
report_query_variant!(
    mbarrier_try_wait_spec,
    TryWaitState,
    u64,
    mbarrier_test_wait_state_instruction,
    true,
    async
);
report_query_variant!(
    mbarrier_try_wait_spec,
    TryWaitStateRelaxed,
    u64,
    mbarrier_test_wait_state_instruction,
    false,
    async
);

instruction_variant! {
    [impl] mbarrier_wait_until_spec, variant::WaitUntilParity,
    (Address<Shared>, R<i64>) => ();
    async fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (barrier, parity): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        mbarrier_wait_until_entry(warp, context, site, barrier, parity).await
    }
}

#[inline(never)]
async fn mbarrier_wait_until_entry(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    barrier: Address<Shared>,
    parity: R<i64>,
) -> Result<(), EngineError> {
    let operation = begin(warp, context, site, OperationKind::Barrier, false)?;
    engine(warp)
        .mbarrier_wait(
            operation.as_ref(),
            barrier.inner(),
            context.active_mask().into_inner(),
            parity.inner(),
        )
        .await?;
    finish(warp, &operation)
}

trait FenceScope: static_sealed::FenceScope {
    const VALUE: MemoryScope;
}
macro_rules! fence_scope {
    ($marker:ty, $value:ident) => {
        impl static_sealed::FenceScope for $marker {}
        impl FenceScope for $marker {
            const VALUE: MemoryScope = MemoryScope::$value;
        }
    };
}
fence_scope!(variant::Cta, Cta);
fence_scope!(variant::Cluster, Cluster);
fence_scope!(variant::Gpu, Gpu);
fence_scope!(variant::Sys, Sys);

trait FenceOrder: static_sealed::FenceOrder {
    const VALUE: MemoryOrder;
}
macro_rules! fence_order {
    ($marker:ident) => {
        impl static_sealed::FenceOrder for variant::$marker {}
        impl FenceOrder for variant::$marker {
            const VALUE: MemoryOrder = MemoryOrder::$marker;
        }
    };
}
fence_order!(Acquire);
fence_order!(Release);
fence_order!(AcqRel);
fence_order!(Sc);

fn execute_tensor_map_replace<S: super::mem::SpaceVariant>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    address: Address<S>,
    mut update: impl FnMut(
        &crate::PhysicalMemory,
        &crate::WarpContext,
        &crate::runtime::PhysicalPtr,
    ) -> Result<(), crate::EngineError>,
) -> Result<(), EngineError> {
    let ctx = context.into_inner();
    let mask = ctx.active_mask();
    if mask.is_empty() {
        return Ok(());
    }
    let pointer = address.inner().with_byte_storage_access_width(128)?;
    pointer.require_ptx_space_for_mask(S::PTX_SPACE, mask)?;
    let warp = engine(warp);
    // PTX treats replacement as a weak read/write of the entire opaque
    // 1024-bit object, even when only one logical field changes.
    for kind in [OperationKind::Load, OperationKind::Store] {
        let operation = warp.begin_optional_pointer_operation(ctx, site.get(), kind, &pointer)?;
        warp.physical_pointer_access(
            operation.as_ref(),
            kind,
            &pointer,
            None,
            mask,
            128,
            false,
            false,
            crate::MemoryAccessSemantics::plain(),
            || {
                if kind == OperationKind::Store {
                    for lane in mask {
                        let issuer = ctx.with_active_mask(crate::WarpMask::from_bits(1 << lane));
                        update(warp.kernel().physical(), &issuer, &pointer)?;
                    }
                }
                Ok(())
            },
        )?;
        warp.finish_optional_operation(&operation)?;
    }
    Ok(())
}

instruction_variant! {
    [impl<S: super::mem::SpaceVariant>] tensor_map_replace_spec, variant::TensorMapField<S>,
    (
        Address<S>,
        crate::runtime::RuntimeTensorMapRegistry,
        &'static str,
        Option<usize>,
        R<u64>,
    ) => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (address, registry, field, index, value): Self::Args,
    ) -> Result<(), EngineError> {
        execute_tensor_map_replace(warp, context, site, address, |physical, ctx, descriptor| {
            let lane = ctx
                .active_mask()
                .first_active()
                .expect("one replacement lane");
            let value = usize::try_from(value.inner().lanes()[lane]).map_err(|_| {
                crate::EngineError::message("TensorMap replacement value does not fit usize")
            })?;
            registry.replace_field(
                physical,
                ctx,
                ctx.active_mask(),
                descriptor,
                field,
                index,
                value,
            )
        })
    }
}
instruction_variant! {
    [impl<S: super::mem::SpaceVariant>] tensor_map_replace_spec, variant::TensorMapAddress<S>,
    (
        Address<S>,
        crate::runtime::RuntimeTensorMapRegistry,
        Address<super::Global>,
    ) => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (address, registry, replacement): Self::Args,
    ) -> Result<(), EngineError> {
        execute_tensor_map_replace(warp, context, site, address, |physical, ctx, descriptor| {
            registry.replace_global_address(
                physical,
                ctx,
                ctx.active_mask(),
                descriptor,
                replacement.inner(),
            )
        })
    }
}

instruction_variant! {
    [impl<S: FenceScope>] tensor_map_copy_spec, variant::TensorMapCopy<S>,
    (
        Address<super::Global>,
        Address<super::SharedCta>,
        crate::runtime::RuntimeTensorMapRegistry,
    ) => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (destination, source, registry): Self::Args,
    ) -> Result<(), EngineError> {
        let ctx = context.into_inner();
        let mask = ctx.active_mask();
        crate::runtime::require_full_warp_sync(mask, "tensormap.cp_fenceproxy")?;
        for pointer in [source.inner(), destination.inner()] {
            let address = pointer.resolve_uniform(&ctx, mask)?;
            if address.byte_offset() % 128 != 0 {
                return Err(EngineError::message(
                    "tensormap.cp_fenceproxy requires 128-byte aligned addresses",
                ));
            }
        }
        // Partition the collective 128-byte generic-proxy copy into disjoint
        // lane words, reusing ordinary memory permissions and exact effects.
        let offsets = crate::WarpValue::from_fn(|lane| (lane * 4) as i64);
        let values =
            super::mem::ld::<super::mem::variant::Ld<super::reg::variant::U32, super::SharedCta>>(
                warp,
                context,
                site,
                Address::from_inner(source.inner().with_byte_offset(&offsets, 4, mask)?),
            )?;
        super::mem::st::<super::mem::variant::St<super::reg::variant::U32, super::Global>>(
            warp,
            context,
            site,
            (
                Address::from_inner(destination.inner().with_byte_offset(&offsets, 4, mask)?),
                values,
            ),
        )?;
        registry
            .publish_copy(warp, context, site, destination.inner(), S::VALUE)
            .map_err(Into::into)
    }
}

instruction_variant! {
    [impl<S: FenceScope, O: FenceOrder>] fence_spec, variant::Fence<S, O>,
    () => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        fence_entry(warp, context, site, S::VALUE, O::VALUE)
    }
}

/// `fence.<order>.<scope>` for every admitted order spelling.
///
/// Acquire/release halves must stay distinct: strengthening either half can
/// manufacture a publication edge and hide a race. SC also participates in
/// the launch's scoped fence order, owned by the native memory model.
#[inline(never)]
fn fence_entry(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    scope: MemoryScope,
    order: MemoryOrder,
) -> Result<(), EngineError> {
    let operation = begin(warp, context, site, OperationKind::Fence, false)?;
    engine(warp).memory_fence(operation.as_ref(), order, scope, MemoryProxy::Generic)?;
    finish(warp, &operation)
}

trait ProxySpace: static_sealed::ProxySpace {
    const VALUE: &'static str;
}
macro_rules! proxy_space {
    ($marker:ty, $value:literal) => {
        impl static_sealed::ProxySpace for $marker {}
        impl ProxySpace for $marker {
            const VALUE: &'static str = $value;
        }
    };
}
proxy_space!(variant::All, "");
proxy_space!(variant::Global, "global");
proxy_space!(variant::SharedCta, "shared::cta");
proxy_space!(variant::SharedCluster, "shared::cluster");

instruction_variant! {
    [impl<S: ProxySpace>] fence_proxy_async_spec, variant::ProxyAsync<S>,
    () => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        fence_proxy_async_entry(warp, context, site, S::VALUE)
    }
}

#[inline(never)]
fn fence_proxy_async_entry(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    space: &'static str,
) -> Result<(), EngineError> {
    let operation = begin(warp, context, site, OperationKind::Fence, false)?;
    engine(warp).proxy_async_fence(operation.as_ref(), space)?;
    finish(warp, &operation)
}

/// Execute one `fence.mbarrier_init` instruction.  This has a distinct
/// checker-visible operation kind and cannot be merged with generic `fence`.
#[inline(never)]
pub fn fence_mbarrier_init(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(
        context,
        site,
        std::any::type_name_of_val(&fence_mbarrier_init),
    );
    let operation = begin(warp, context, site, OperationKind::MbarrierInitFence, false)?;
    engine(warp).mbarrier_init_fence(operation.as_ref())?;
    finish(warp, &operation)
}

// State-token forms remain absent until the engine has a 64-bit state-token
// transport. Acquire-qualified nonblocking waits execute exact numeric
// readiness; checker modes additionally honor acquire-on-true semantics.

#[cfg(all(test, not(feature = "analysis-core")))]
mod tests {
    use super::*;
    use crate::runtime::{
        run_kernel_engine_launch, ExecutionPolicy, LaunchSelection, PhysicalPtr, RuntimeBuffer,
    };
    use crate::{CtaId, LaunchTopology, NumSimMode, PhysicalMemory, WarpValue};
    use std::sync::Arc;

    fn shared_pointer(physical: &PhysicalMemory) -> PhysicalPtr {
        let topology = physical.topology();
        let owner = CtaId::new(topology, 0, 0).unwrap();
        let allocation = physical.shared().allocate_cta_zeroed(owner, 16).unwrap();
        PhysicalPtr::new(
            RuntimeBuffer::Shared {
                allocations: Arc::new(vec![allocation]),
                virtual_base: 0,
                byte_offset: 0,
                byte_len: 16,
                backing_byte_len: 16,
            },
            WarpValue::splat(0),
            8,
        )
    }

    #[test]
    fn mbarrier_variants_share_the_engine_protocol_without_runtime_options() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let pointer = shared_pointer(&physical);
        run_kernel_engine_launch::<NumSimMode, _, _>(
            physical,
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let pointer = pointer.clone();
                async move {
                    let context = ExecCtx::from_inner(
                        warp.context()
                            .with_active_mask(crate::WarpMask::from_bits(1)),
                    );
                    let barrier = Address::<Shared>::from_inner(pointer.clone());
                    mbarrier_init::<variant::MbarrierInit>(
                        &mut warp,
                        context,
                        SiteId::new(1),
                        (barrier.clone(), R::splat(1)),
                    )?;
                    mbarrier_arrive_expect_tx::<variant::ArriveExpectTxLocal>(
                        &mut warp,
                        context,
                        SiteId::new(2),
                        (barrier.clone(), R::splat(16)),
                    )?;
                    mbarrier_complete_tx::<variant::CompleteTxLocal>(
                        &mut warp,
                        context,
                        SiteId::new(3),
                        (barrier, R::splat(16)),
                    )?;
                    let ready = mbarrier_try_wait::<variant::TryWaitParity>(
                        &mut warp,
                        context,
                        SiteId::new(4),
                        (Address::<Shared>::from_inner(pointer.clone()), R::splat(0)),
                    )
                    .await?;
                    assert!(ready[0]);
                    Ok(())
                }
            },
        )
        .unwrap();
    }

    #[test]
    fn standalone_expect_tx_does_not_consume_an_arrival() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let pointer = shared_pointer(&physical);
        run_kernel_engine_launch::<NumSimMode, _, _>(
            physical,
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let pointer = pointer.clone();
                async move {
                    let context = ExecCtx::from_inner(
                        warp.context()
                            .with_active_mask(crate::WarpMask::from_bits(1)),
                    );
                    let barrier = Address::<Shared>::from_inner(pointer.clone());
                    mbarrier_init::<variant::MbarrierInit>(
                        &mut warp,
                        context,
                        SiteId::new(10),
                        (barrier.clone(), R::splat(1)),
                    )?;
                    mbarrier_expect_tx::<variant::ExpectTx>(
                        &mut warp,
                        context,
                        SiteId::new(11),
                        (barrier.clone(), R::splat(16)),
                    )?;
                    // This ordinary arrival must still be required.  Reusing
                    // arrive.expect_tx for the previous call would overflow.
                    mbarrier_arrive::<variant::ArriveLocal>(
                        &mut warp,
                        context,
                        SiteId::new(12),
                        barrier.clone(),
                    )?;
                    mbarrier_complete_tx::<variant::CompleteTxLocal>(
                        &mut warp,
                        context,
                        SiteId::new(13),
                        (barrier, R::splat(16)),
                    )?;
                    let ready = mbarrier_try_wait::<variant::TryWaitParity>(
                        &mut warp,
                        context,
                        SiteId::new(14),
                        (Address::<Shared>::from_inner(pointer), R::splat(0)),
                    )
                    .await?;
                    assert!(ready[0]);
                    Ok(())
                }
            },
        )
        .unwrap();
    }

    #[test]
    fn raw_try_wait_is_one_nonblocking_instruction_not_a_frontend_wait_loop() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let pointer = shared_pointer(&physical);
        run_kernel_engine_launch::<NumSimMode, _, _>(
            physical,
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let pointer = pointer.clone();
                async move {
                    let context = ExecCtx::from_inner(
                        warp.context()
                            .with_active_mask(crate::WarpMask::from_bits(1)),
                    );
                    let barrier = Address::<Shared>::from_inner(pointer.clone());
                    mbarrier_init::<variant::MbarrierInit>(
                        &mut warp,
                        context,
                        SiteId::new(20),
                        (barrier.clone(), R::splat(1)),
                    )?;
                    mbarrier_arrive_expect_tx::<variant::ArriveExpectTxLocal>(
                        &mut warp,
                        context,
                        SiteId::new(21),
                        (barrier.clone(), R::splat(16)),
                    )?;
                    let before = mbarrier_try_wait::<variant::TryWaitParity>(
                        &mut warp,
                        context,
                        SiteId::new(22),
                        (barrier.clone(), R::splat(0)),
                    )
                    .await?;
                    assert!(!before[0]);

                    mbarrier_complete_tx::<variant::CompleteTxLocal>(
                        &mut warp,
                        context,
                        SiteId::new(23),
                        (barrier.clone(), R::splat(16)),
                    )?;
                    let after = mbarrier_test_wait::<variant::TestWaitParity>(
                        &mut warp,
                        context,
                        SiteId::new(24),
                        (barrier, R::splat(0)),
                    )?;
                    assert!(after[0]);
                    Ok(())
                }
            },
        )
        .unwrap();
    }

    #[test]
    fn negative_complete_tx_count_fails_before_protocol_mutation() {
        let values = R::from_fn(|lane| if lane == 3 { -1 } else { 4 });
        let error =
            transaction_bytes(&values, super::super::LaneMask::from_bits(1 << 3)).unwrap_err();
        assert!(error.to_string().contains("active lane 3"));
    }
}
