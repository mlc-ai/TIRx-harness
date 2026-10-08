use std::error::Error;
use std::fmt;

// The payload vocabulary of the effect types below.
//
// These types are declared where the engine builds them, but they are effect
// *payloads*: an observer only ever meets them as a field of an
// `OperationEffect` / `CompletionEffect` variant. Re-exporting them here makes
// this module the single import surface a checker needs, so `native_analysis/`
// never has to name `crate::runtime` or another engine-internal path to spell
// the payload it was handed.
//
// Adding a payload type to an effect variant therefore means adding it here;
// the checker-import gate (`tests/checker_import_gate.rs`) enforces that
// checkers import payloads through this surface and nowhere else.
pub(crate) use crate::runtime::{
    ClusterBarrierArrivePlan, ClusterBarrierRegistrationOutcome, ClusterBarrierWaitPlan,
    ClusterBarrierWaitResumePlan, CpAsyncMbarrierArrivePlan, NamedBarrierArrivePlan,
    NamedBarrierSyncPlan, NamedBarrierSyncRegistrationOutcome, NamedBarrierSyncResumePlan,
    PhysicalMbarrierArrivalBatchOutcome, PhysicalMbarrierArriveBatchPlan,
    PhysicalMbarrierArrivePlan, PhysicalMbarrierCompletionIssuePlan,
    PhysicalMbarrierExpectTxOutcome, PhysicalMbarrierExpectTxPlan, PhysicalMbarrierInitPlan,
    DeclaredWordWaitPlan, PhysicalMbarrierWaitOutcome, PhysicalMbarrierWaitPlan,
    TcgenCommitIssuePlan,
    TcgenLifecyclePlan, TcgenLifecycleResumePlan, TcgenMmaPipelineClass, TcgenPipelineOperation,
    TcgenWorkIssue, TcgenWorkKind, TcgenWorkSet,
};
pub(crate) use crate::{
    AsyncGroupCommitOutcome, AsyncGroupCommitPlan, AsyncGroupCompletionAction,
    AsyncGroupCompletionOutcome, AsyncGroupIssueBatchEffect, AsyncGroupIssueEffect,
    AsyncGroupWaitOutcome, AsyncGroupWaitPlan, AsyncTokenId, ClusterBarrierArrivalOutcome,
    DeferredPayloadCompletionAction, DeferredPayloadCompletionOutcome, DynamicOpId, MemoryOrder,
    MemoryProxy, MemoryScope, NamedBarrierArrivalOutcome, OperationContext, PhysicalAccessBatch,
    PhysicalAccessKind, PhysicalBarrierId, PhysicalByteSpan, PhysicalCompletionAction,
    PhysicalCompletionActionId, PhysicalCompletionOutcome, PhysicalMbarrierArrivalOutcome,
    SetmaxnregCompletionAction, SetmaxnregCompletionOutcome, SetmaxnregPlan, SetmaxnregResumePlan,
    WarpMask,
};

/// One fully resolved asynchronous payload transaction.
///
/// The access set and completion plan are owned so every analysis mode observes
/// one semantic effect for one [`DynamicOpId`]. Completion action IDs are bound
/// only after the numerical payload succeeds and the runtime enqueues the whole
/// completion batch transactionally.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsyncPayloadEffect {
    token: AsyncTokenId,
    operation: OperationContext,
    accesses: Box<[PhysicalAccessBatch]>,
    issue_accesses: Box<[PhysicalAccessBatch]>,
    completion_accesses: Box<[PhysicalAccessBatch]>,
    completion_plan: PhysicalMbarrierCompletionIssuePlan,
    completion_action_ids: Option<Box<[PhysicalCompletionActionId]>>,
}

impl AsyncPayloadEffect {
    pub fn new(
        operation: OperationContext,
        accesses: impl IntoIterator<Item = PhysicalAccessBatch>,
        completion_plan: PhysicalMbarrierCompletionIssuePlan,
    ) -> Result<Self, AsyncPayloadEffectError> {
        Self::new_with_access_semantics(
            operation,
            accesses,
            completion_plan,
            crate::MemoryAccessSemantics::async_proxy(),
        )
    }

    pub(crate) fn new_with_access_semantics(
        operation: OperationContext,
        accesses: impl IntoIterator<Item = PhysicalAccessBatch>,
        completion_plan: PhysicalMbarrierCompletionIssuePlan,
        access_semantics: crate::MemoryAccessSemantics,
    ) -> Result<Self, AsyncPayloadEffectError> {
        let accesses = accesses.into_iter().collect::<Vec<_>>();
        let is_reduction = accesses
            .iter()
            .any(|batch| batch.descriptor().kind() == PhysicalAccessKind::AtomicReadModifyWrite);
        let accesses = accesses
            .into_iter()
            .map(|batch| {
                if matches!(
                    batch.descriptor().space(),
                    crate::PhysicalAccessSpace::Global | crate::PhysicalAccessSpace::Shared
                ) {
                    // Bulk reductions apply .sem.scope only to the destination
                    // element-wise RMW. Their source reads remain weak/async.
                    let semantics = if is_reduction
                        && batch.descriptor().kind() == PhysicalAccessKind::Read
                        && access_semantics.proxy() == crate::MemoryProxy::Async
                    {
                        crate::MemoryAccessSemantics::async_proxy()
                    } else {
                        access_semantics
                    };
                    batch.with_memory_semantics(semantics)
                } else {
                    batch
                }
            })
            .collect::<Vec<_>>();
        for (index, access) in accesses.iter().enumerate() {
            if access.operation() != &operation {
                return Err(AsyncPayloadEffectError::AccessOperationMismatch {
                    index,
                    expected: operation.id().clone(),
                    actual: access.operation().id().clone(),
                });
            }
        }
        let mut issue_accesses = Vec::new();
        let mut completion_accesses = Vec::new();
        for access in &accesses {
            match access.descriptor().kind() {
                PhysicalAccessKind::Read => issue_accesses.push(access.clone()),
                // An asynchronous reduction is one indivisible RMW at completion,
                // not a source read at issue followed by a plain destination write.
                PhysicalAccessKind::Write | PhysicalAccessKind::AtomicReadModifyWrite => {
                    completion_accesses.push(access.clone())
                }
            }
        }
        Ok(Self {
            token: AsyncTokenId::new(operation.id().clone(), 0),
            operation,
            accesses: accesses.into_boxed_slice(),
            issue_accesses: issue_accesses.into_boxed_slice(),
            completion_accesses: completion_accesses.into_boxed_slice(),
            completion_plan,
            completion_action_ids: None,
        })
    }

    pub const fn operation(&self) -> &OperationContext {
        &self.operation
    }

    pub const fn token(&self) -> &AsyncTokenId {
        &self.token
    }

    pub fn accesses(&self) -> &[PhysicalAccessBatch] {
        &self.accesses
    }

    pub fn issue_accesses(&self) -> &[PhysicalAccessBatch] {
        &self.issue_accesses
    }

    pub fn completion_accesses(&self) -> &[PhysicalAccessBatch] {
        &self.completion_accesses
    }

    pub const fn completion_plan(&self) -> &PhysicalMbarrierCompletionIssuePlan {
        &self.completion_plan
    }

    pub fn completion_action_ids(&self) -> Option<&[PhysicalCompletionActionId]> {
        self.completion_action_ids.as_deref()
    }

    pub(crate) fn bind_completion_action_ids(
        &mut self,
        action_ids: Box<[PhysicalCompletionActionId]>,
    ) -> Result<(), AsyncPayloadEffectError> {
        if self.completion_action_ids.is_some() {
            return Err(AsyncPayloadEffectError::CompletionActionsAlreadyBound);
        }
        if action_ids.len() != self.completion_plan.completions().len() {
            return Err(AsyncPayloadEffectError::CompletionActionCountMismatch {
                expected: self.completion_plan.completions().len(),
                actual: action_ids.len(),
            });
        }
        self.completion_action_ids = Some(action_ids);
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AsyncPayloadEffectError {
    AccessOperationMismatch {
        index: usize,
        expected: DynamicOpId,
        actual: DynamicOpId,
    },
    CompletionActionCountMismatch {
        expected: usize,
        actual: usize,
    },
    CompletionActionsAlreadyBound,
}

impl fmt::Display for AsyncPayloadEffectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AccessOperationMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "async payload access batch {index} belongs to {actual}, expected {expected}"
            ),
            Self::CompletionActionCountMismatch { expected, actual } => write!(
                f,
                "async payload bound {actual} completion action IDs for {expected} completion targets"
            ),
            Self::CompletionActionsAlreadyBound => {
                f.write_str("async payload completion action IDs are already bound")
            }
        }
    }
}

impl Error for AsyncPayloadEffectError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProxyAsyncFenceScope {
    All,
    SharedCta,
    SharedCluster,
    Global,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProxyAsyncFenceEffect {
    scope: ProxyAsyncFenceScope,
    kernel_index: usize,
    global_cta_id: usize,
    cluster_id: usize,
}

impl ProxyAsyncFenceEffect {
    pub(crate) const fn new(
        scope: ProxyAsyncFenceScope,
        kernel_index: usize,
        global_cta_id: usize,
        cluster_id: usize,
    ) -> Self {
        Self {
            scope,
            kernel_index,
            global_cta_id,
            cluster_id,
        }
    }

    pub const fn scope(self) -> ProxyAsyncFenceScope {
        self.scope
    }

    pub const fn kernel_index(self) -> usize {
        self.kernel_index
    }

    pub const fn global_cta_id(self) -> usize {
        self.global_cta_id
    }

    pub const fn cluster_id(self) -> usize {
        self.cluster_id
    }
}

/// A checker-visible PTX/CUDA memory fence.
///
/// SC order creates causality between mutually scoped fences, not completion
/// of asynchronous work. Shared-memory lane-frontier transport must preserve
/// that distinction before generated PTX can opt into the SC marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryFenceEffect {
    order: MemoryOrder,
    scope: MemoryScope,
    proxy: MemoryProxy,
}

impl MemoryFenceEffect {
    pub const fn new(order: MemoryOrder, scope: MemoryScope, proxy: MemoryProxy) -> Self {
        Self {
            order,
            scope,
            proxy,
        }
    }

    pub const fn order(self) -> MemoryOrder {
        self.order
    }

    pub const fn scope(self) -> MemoryScope {
        self.scope
    }

    pub const fn proxy(self) -> MemoryProxy {
        self.proxy
    }
}

/// One descriptor-proxy observation. The registry owns descriptor generations;
/// analysis carries an acquisition only along the executing lanes' causal edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorMapObservation {
    Release {
        scope: MemoryScope,
    },
    CopyRelease {
        descriptor: PhysicalByteSpan,
        scope: MemoryScope,
    },
    Acquire {
        descriptor: PhysicalByteSpan,
        generation: u64,
        lane: usize,
        cta: usize,
        scope: MemoryScope,
    },
    Consume {
        descriptor: PhysicalByteSpan,
        generation: u64,
        cta: usize,
    },
}

/// A completed same-warp lane rendezvous.
///
/// Published only after the numeric wait succeeds, so there is no pre-numeric
/// decision point and no matching `before_effect`. The mask is the set of lanes
/// that met; it is the whole payload, which is why this used to be delivered as
/// a bare `WarpMask` through a dedicated hook.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WarpSyncEffect {
    mask: WarpMask,
}

impl WarpSyncEffect {
    pub(crate) const fn new(mask: WarpMask) -> Self {
        Self { mask }
    }

    pub const fn mask(self) -> WarpMask {
        self.mask
    }
}

/// Direction of the specialized TCGEN/thread execution-ordering bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TcgenFenceKind {
    BeforeThreadSync,
    AfterThreadSync,
}

/// Analysis domain whose exact semantics are not yet modeled by the native
/// checker modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AnalysisGapDomain {
    Tcgen,
    ClusterBarrier,
    Atomic,
}

impl AnalysisGapDomain {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Tcgen => "tcgen",
            Self::ClusterBarrier => "cluster_barrier",
            Self::Atomic => "atomic",
        }
    }
}

/// Concrete operation kind with mode-specific missing analysis evidence.
/// These variants are coverage markers, not approximate semantics; a mode may
/// filter one after independently modeling the part of the operation it owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum AnalysisGapKind {
    TcgenMma,
    TcgenShift,
    ClusterBarrierUnaligned,
    AtomicLaneSerialization,
}

impl AnalysisGapKind {
    pub(crate) const fn domain(self) -> AnalysisGapDomain {
        match self {
            Self::TcgenMma | Self::TcgenShift => AnalysisGapDomain::Tcgen,
            Self::ClusterBarrierUnaligned => AnalysisGapDomain::ClusterBarrier,
            Self::AtomicLaneSerialization => AnalysisGapDomain::Atomic,
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::TcgenMma => "tcgen05.mma",
            Self::TcgenShift => "tcgen05.shift",
            Self::ClusterBarrierUnaligned => "cluster_barrier_unaligned_unmodeled",
            Self::AtomicLaneSerialization => "atomic_lane_serialization_unmodeled",
        }
    }
}

/// Path-sensitive marker proving that one concrete mode-specific analysis gap
/// executed. Numerical execution continues unchanged; observing checker modes
/// use this marker to fail closed instead of claiming a clean result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AnalysisGapEffect {
    kind: AnalysisGapKind,
    kernel_index: usize,
    global_cta_id: usize,
    cta_group: Option<u32>,
}

impl AnalysisGapEffect {
    pub(crate) const fn new(
        kind: AnalysisGapKind,
        kernel_index: usize,
        global_cta_id: usize,
        cta_group: Option<u32>,
    ) -> Self {
        Self {
            kind,
            kernel_index,
            global_cta_id,
            cta_group,
        }
    }

    pub(crate) const fn kind(self) -> AnalysisGapKind {
        self.kind
    }

    pub const fn domain(self) -> AnalysisGapDomain {
        self.kind.domain()
    }

    pub const fn kernel_index(self) -> usize {
        self.kernel_index
    }

    pub const fn global_cta_id(self) -> usize {
        self.global_cta_id
    }

    pub const fn cta_group(self) -> Option<u32> {
        self.cta_group
    }
}

// ---------------------------------------------------------------------------
// One effect vocabulary, two ownership forms.
//
// `OperationEffect<'a>` (delivery form, borrowed, `Copy`) and
// `OwnedOperationEffect` (storage form) are generated from the single variant
// declaration below, together with `OperationEffect::to_owned_effect`.
//
// Delivery is unchanged: observers still receive the borrowed form and pay
// nothing. An observer that needs history calls `to_owned_effect()` on the
// subset it keeps; every other mode never materializes anything.
//
// Because both forms come from one declaration, a variant or a field cannot be
// added to one and forgotten in the other, and storage cannot silently
// drop a payload field.
//
// Field ownership kinds:
//   `copy T`             delivered `T` (Copy)          stored `T`
//   `borrow T`           delivered `&'a T`             stored `T`
//   `opt_copy T`         delivered `Option<T>`         stored `Option<T>`
//   `opt_borrow T`       delivered `Option<&'a T>`     stored `Option<T>`
//   `borrow_slice T`     delivered `&'a [T]`           stored `Box<[T]>`
//   `opt_borrow_slice T` delivered `Option<&'a [T]>`   stored `Option<Box<[T]>>`
// ---------------------------------------------------------------------------

macro_rules! effect_field_borrowed_ty {
    ($lt:lifetime, copy $ty:ty) => { $ty };
    ($lt:lifetime, borrow $ty:ty) => { &$lt $ty };
    ($lt:lifetime, opt_copy $ty:ty) => { Option<$ty> };
    ($lt:lifetime, opt_borrow $ty:ty) => { Option<&$lt $ty> };
    ($lt:lifetime, borrow_slice $ty:ty) => { &$lt [$ty] };
    ($lt:lifetime, opt_borrow_slice $ty:ty) => { Option<&$lt [$ty]> };
}

macro_rules! effect_field_owned_ty {
    (copy $ty:ty) => { $ty };
    (borrow $ty:ty) => { $ty };
    (opt_copy $ty:ty) => { Option<$ty> };
    (opt_borrow $ty:ty) => { Option<$ty> };
    (borrow_slice $ty:ty) => { Box<[$ty]> };
    (opt_borrow_slice $ty:ty) => { Option<Box<[$ty]>> };
}

macro_rules! effect_field_to_owned {
    (copy $value:expr) => {
        $value
    };
    (borrow $value:expr) => {
        ::std::clone::Clone::clone($value)
    };
    (opt_copy $value:expr) => {
        $value
    };
    (opt_borrow $value:expr) => {
        $value.cloned()
    };
    (borrow_slice $value:expr) => {
        $value.to_vec().into_boxed_slice()
    };
    (opt_borrow_slice $value:expr) => {
        $value.map(|slice| slice.to_vec().into_boxed_slice())
    };
}

macro_rules! declare_effect_vocabulary {
    // Tuple variant with exactly one payload field.
    (@munch
        rest { $variant:ident ( $kind:ident $ty:ty ) , $($rest:tt)* }
        borrowed { $($borrowed:tt)* }
        owned { $($owned:tt)* }
        to_owned { $($to_owned:tt)* }
    ) => {
        declare_effect_vocabulary! { @munch
            rest { $($rest)* }
            borrowed {
                $($borrowed)*
                $variant(effect_field_borrowed_ty!('a, $kind $ty)),
            }
            owned {
                $($owned)*
                $variant(effect_field_owned_ty!($kind $ty)),
            }
            to_owned {
                $($to_owned)*
                OperationEffect::$variant(field) => {
                    OwnedOperationEffect::$variant(effect_field_to_owned!($kind field))
                }
            }
        }
    };

    // Struct variant with one or more named payload fields.
    (@munch
        rest { $variant:ident { $( $field:ident : $kind:ident $ty:ty ),* $(,)? } , $($rest:tt)* }
        borrowed { $($borrowed:tt)* }
        owned { $($owned:tt)* }
        to_owned { $($to_owned:tt)* }
    ) => {
        declare_effect_vocabulary! { @munch
            rest { $($rest)* }
            borrowed {
                $($borrowed)*
                $variant { $( $field: effect_field_borrowed_ty!('a, $kind $ty), )* },
            }
            owned {
                $($owned)*
                $variant { $( $field: effect_field_owned_ty!($kind $ty), )* },
            }
            to_owned {
                $($to_owned)*
                OperationEffect::$variant { $( $field, )* } => {
                    OwnedOperationEffect::$variant {
                        $( $field: effect_field_to_owned!($kind $field), )*
                    }
                }
            }
        }
    };

    (@munch
        rest {}
        borrowed { $($borrowed:tt)* }
        owned { $($owned:tt)* }
        to_owned { $($to_owned:tt)* }
    ) => {
        /// Complete concrete descriptor for one checker-visible engine effect.
        ///
        /// The descriptor is produced after pointer/count/phase resolution and before
        /// the numerical runtime mutates state. Analysis modes can therefore reject a
        /// protocol operation without observing a partially applied effect.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[non_exhaustive]
        pub enum OperationEffect<'a> {
            $($borrowed)*
        }

        /// Storage form of [`OperationEffect`], in the same vocabulary.
        ///
        /// Produced by [`OperationEffect::to_owned_effect`] for observers that
        /// retain history. Stored records are inspected directly, not replayed.
        #[derive(Clone, Debug, PartialEq, Eq)]
        #[non_exhaustive]
        pub enum OwnedOperationEffect {
            $($owned)*
        }

        impl OperationEffect<'_> {
            /// Materialize this effect for storage. Never called on a path that
            /// only inspects an effect.
            pub fn to_owned_effect(self) -> OwnedOperationEffect {
                match self {
                    $($to_owned)*
                }
            }
        }

    };

    // Entry point. Declared last: the `@munch` arms above are literal-prefixed,
    // so this catch-all cannot shadow them.
    ($($variants:tt)*) => {
        declare_effect_vocabulary! { @munch
            rest { $($variants)* }
            borrowed {}
            owned {}
            to_owned {}
        }
    };
}

declare_effect_vocabulary! {
    PhysicalAccess(borrow PhysicalAccessBatch),
    AsyncPayload(borrow AsyncPayloadEffect),
    AsyncGroupIssue(borrow AsyncGroupIssueEffect),
    AsyncGroupIssueBatch(borrow AsyncGroupIssueBatchEffect),
    AsyncGroupCommit {
        plan: borrow AsyncGroupCommitPlan,
        outcome: opt_borrow AsyncGroupCommitOutcome,
    },
    CpAsyncMbarrierArrive {
        plan: borrow CpAsyncMbarrierArrivePlan,
        commit_plan: borrow AsyncGroupCommitPlan,
        outcome: opt_borrow AsyncGroupCommitOutcome,
        actions: opt_borrow_slice PhysicalCompletionAction,
    },
    AsyncGroupWait {
        plan: borrow AsyncGroupWaitPlan,
        outcome: opt_borrow AsyncGroupWaitOutcome,
    },
    MemoryFence(copy MemoryFenceEffect),
    ProxyAsyncFence(copy ProxyAsyncFenceEffect),
    TensorMap(copy TensorMapObservation),
    WarpSync(copy WarpSyncEffect),
    AnalysisGap(copy AnalysisGapEffect),
    MbarrierInit(borrow PhysicalMbarrierInitPlan),
    MbarrierInvalidate {
        barrier_ids: borrow_slice PhysicalBarrierId,
    },
    MbarrierInitFence {
        barrier_ids: borrow_slice PhysicalBarrierId,
    },
    MbarrierExpectTx {
        plan: borrow PhysicalMbarrierExpectTxPlan,
        outcome: opt_borrow PhysicalMbarrierExpectTxOutcome,
    },
    TcgenFence(copy TcgenFenceKind),
    MbarrierArrive {
        plan: copy PhysicalMbarrierArrivePlan,
        outcome: opt_copy PhysicalMbarrierArrivalOutcome,
    },
    MbarrierArriveBatch {
        plan: borrow PhysicalMbarrierArriveBatchPlan,
        outcome: opt_borrow PhysicalMbarrierArrivalBatchOutcome,
    },
    MbarrierWait {
        plan: copy PhysicalMbarrierWaitPlan,
        outcome: opt_copy PhysicalMbarrierWaitOutcome,
    },
    DeclaredWordWait {
        plan: copy DeclaredWordWaitPlan,
    },
    MbarrierCompletionIssue {
        plan: borrow PhysicalMbarrierCompletionIssuePlan,
        action_ids: opt_borrow_slice PhysicalCompletionActionId,
    },
    TcgenWorkIssue(borrow TcgenWorkIssue),
    TcgenCommitIssue {
        plan: borrow TcgenCommitIssuePlan,
        work: borrow TcgenWorkSet,
        actions: opt_borrow_slice PhysicalCompletionAction,
    },
    TcgenWait {
        work: borrow TcgenWorkSet,
    },
    NamedBarrierArrive {
        plan: copy NamedBarrierArrivePlan,
        outcome: opt_copy NamedBarrierArrivalOutcome,
    },
    NamedBarrierSyncRegister {
        plan: copy NamedBarrierSyncPlan,
        outcome: opt_copy NamedBarrierSyncRegistrationOutcome,
    },
    NamedBarrierSyncResume(copy NamedBarrierSyncResumePlan),
    ClusterBarrierArrive {
        plan: borrow ClusterBarrierArrivePlan,
        outcome: opt_copy ClusterBarrierArrivalOutcome,
    },
    ClusterBarrierWaitRegister {
        plan: borrow ClusterBarrierWaitPlan,
        outcome: opt_copy ClusterBarrierRegistrationOutcome,
    },
    ClusterBarrierWaitResume(borrow ClusterBarrierWaitResumePlan),
    TcgenLifecycleRegister(borrow TcgenLifecyclePlan),
    TcgenLifecycleResume(borrow TcgenLifecycleResumePlan),
    SetmaxnregRegister(borrow SetmaxnregPlan),
    SetmaxnregResume(borrow SetmaxnregResumePlan),
}

impl OperationEffect<'_> {
    pub const fn name(self) -> &'static str {
        match self {
            Self::PhysicalAccess(_) => "physical_access",
            Self::AsyncPayload(_) => "async_payload",
            Self::AsyncGroupIssue(_) => "async_group.issue",
            Self::AsyncGroupIssueBatch(_) => "async_group.issue_batch",
            Self::AsyncGroupCommit { .. } => "async_group.commit",
            Self::CpAsyncMbarrierArrive { plan, .. } => {
                if plan.increments_pending() {
                    "cp.async.mbarrier.arrive"
                } else {
                    "cp.async.mbarrier.arrive.noinc"
                }
            }
            Self::AsyncGroupWait { .. } => "async_group.wait",
            Self::MemoryFence(_) => "memory_fence",
            Self::ProxyAsyncFence(_) => "fence.proxy_async",
            Self::TensorMap(TensorMapObservation::Release { .. }) => "tensormap.release",
            Self::TensorMap(TensorMapObservation::CopyRelease { .. }) => "tensormap.copy_release",
            Self::TensorMap(TensorMapObservation::Acquire { .. }) => "tensormap.acquire",
            Self::TensorMap(TensorMapObservation::Consume { .. }) => "tensormap.consume",
            Self::WarpSync(_) => "warp.sync",
            Self::TcgenFence(TcgenFenceKind::BeforeThreadSync) => {
                "tcgen05.fence.before_thread_sync"
            }
            Self::TcgenFence(TcgenFenceKind::AfterThreadSync) => "tcgen05.fence.after_thread_sync",
            Self::AnalysisGap(_) => "analysis_gap",
            Self::MbarrierInit(_) => "mbarrier.init",
            Self::MbarrierInvalidate { .. } => "mbarrier.inval",
            Self::MbarrierInitFence { .. } => "fence.mbarrier_init",
            Self::MbarrierExpectTx { .. } => "mbarrier.expect_tx",
            Self::MbarrierArrive { .. } => "mbarrier.arrive",
            Self::MbarrierArriveBatch { .. } => "mbarrier.arrive",
            Self::MbarrierWait { .. } => "mbarrier.wait",
            Self::DeclaredWordWait { .. } => "wait_until",
            Self::MbarrierCompletionIssue { .. } => "mbarrier.completion_issue",
            Self::TcgenWorkIssue(issue) => issue.kind().name(),
            Self::TcgenCommitIssue { .. } => "tcgen05.commit.issue",
            Self::TcgenWait { work } => match work.kind() {
                crate::runtime::TcgenWorkKind::Load => "tcgen05.wait.ld",
                crate::runtime::TcgenWorkKind::Store => "tcgen05.wait.st",
                _ => "tcgen05.wait",
            },
            Self::NamedBarrierArrive { .. } => "bar.arrive.register",
            Self::NamedBarrierSyncRegister { .. } => "bar.sync.register",
            Self::NamedBarrierSyncResume(_) => "bar.sync.resume",
            Self::ClusterBarrierArrive { .. } => "barrier.cluster.arrive",
            Self::ClusterBarrierWaitRegister { .. } => "barrier.cluster.wait.register",
            Self::ClusterBarrierWaitResume(_) => "barrier.cluster.wait.resume",
            Self::TcgenLifecycleRegister(plan) => plan.action().label(),
            Self::TcgenLifecycleResume(plan) => plan.plan().action().label(),
            Self::SetmaxnregRegister(plan) => plan.action().label(),
            Self::SetmaxnregResume(plan) => plan.plan().action().label(),
        }
    }
}

/// Exact asynchronous action offered to a mode before runtime mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompletionActionEffect<'a> {
    PhysicalMbarrier(&'a PhysicalCompletionAction),
    DeferredPayload(&'a DeferredPayloadCompletionAction),
    AsyncGroup(&'a AsyncGroupCompletionAction),
    Setmaxnreg(&'a SetmaxnregCompletionAction),
}

/// Exact asynchronous completion selected by the controlled scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompletionEffect<'a> {
    PhysicalMbarrier(&'a PhysicalCompletionOutcome),
    DeferredPayload(&'a DeferredPayloadCompletionOutcome),
    AsyncGroup(&'a AsyncGroupCompletionOutcome),
    Setmaxnreg(&'a SetmaxnregCompletionOutcome),
}

#[cfg(test)]
mod owned_vocabulary_tests {
    use std::sync::Arc;

    use crate::runtime::{
        plan_cp_async_mbarrier_arrive, plan_physical_mbarrier_arrive,
        plan_physical_mbarrier_arrive_lanes, plan_physical_mbarrier_init,
        plan_physical_mbarrier_wait, plan_tcgen_allocate, PhysicalMbarrierCompletionIssuePlan,
        PhysicalPtr, RuntimeBuffer,
    };
    use crate::{
        AsyncGroupDomain, AsyncGroupHub, CtaId, DynamicOpId, LaunchTopology, OperationKind,
        PhysicalBarrierHub, PhysicalBarrierId, PhysicalMemory, StaticOpId, WarpContext, WarpMask,
        WarpValue,
    };

    use super::*;

    fn shared_pointer() -> (PhysicalMemory, WarpContext, PhysicalPtr) {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let physical = PhysicalMemory::new(topology);
        let owner = CtaId::new(topology, 0, 0).unwrap();
        let allocation = physical.shared().allocate_cta_zeroed(owner, 256).unwrap();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Shared {
                allocations: Arc::new(vec![allocation]),
                byte_offset: 0,
                byte_len: 256,
                backing_byte_len: 256,
                virtual_base: 0,
            },
            WarpValue::splat(2),
            8,
        );
        (physical, context, pointer)
    }

    #[test]
    fn copy_payload_variants_are_stored_exactly() {
        let fence = MemoryFenceEffect::new(
            crate::MemoryOrder::Release,
            crate::MemoryScope::Cta,
            crate::MemoryProxy::Generic,
        );
        assert_eq!(
            OperationEffect::MemoryFence(fence).to_owned_effect(),
            OwnedOperationEffect::MemoryFence(fence)
        );
        let proxy = ProxyAsyncFenceEffect::new(ProxyAsyncFenceScope::SharedCluster, 1, 2, 3);
        assert_eq!(
            OperationEffect::ProxyAsyncFence(proxy).to_owned_effect(),
            OwnedOperationEffect::ProxyAsyncFence(proxy)
        );
        let gap = AnalysisGapEffect::new(AnalysisGapKind::TcgenMma, 1, 2, Some(2));
        assert_eq!(
            OperationEffect::AnalysisGap(gap).to_owned_effect(),
            OwnedOperationEffect::AnalysisGap(gap)
        );
        let sync = WarpSyncEffect::new(WarpMask::from_lanes([0, 3, 17]).unwrap());
        assert_eq!(
            OperationEffect::WarpSync(sync).to_owned_effect(),
            OwnedOperationEffect::WarpSync(sync)
        );
        for kind in [TcgenFenceKind::BeforeThreadSync, TcgenFenceKind::AfterThreadSync] {
            assert_eq!(
                OperationEffect::TcgenFence(kind).to_owned_effect(),
                OwnedOperationEffect::TcgenFence(kind)
            );
        }
    }

    #[test]
    fn borrowed_plan_variants_are_stored_exactly() {
        let (_physical, context, pointer) = shared_pointer();
        let mask = WarpMask::from_lanes([0, 1]).unwrap();

        // `borrow` field whose plan owns a boxed slice.
        let init = plan_physical_mbarrier_init(&context, &pointer, mask, 2).unwrap();
        assert_eq!(
            (OperationEffect::MbarrierInit(&init)).to_owned_effect(),
            OwnedOperationEffect::MbarrierInit(init.clone())
        );

        let lifecycle = plan_tcgen_allocate(0, 7, [0], context, 64, 1).unwrap();
        assert_eq!(
            (OperationEffect::TcgenLifecycleRegister(&lifecycle)).to_owned_effect(),
            OwnedOperationEffect::TcgenLifecycleRegister(lifecycle.clone())
        );
    }

    #[test]
    fn optional_copy_payloads_are_stored_exactly_present_and_absent() {
        let (_physical, context, pointer) = shared_pointer();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let hub = PhysicalBarrierHub::new();
        let init = plan_physical_mbarrier_init(&context, &pointer, mask, 1).unwrap();
        init.apply(&hub).unwrap();

        let arrive = plan_physical_mbarrier_arrive(&context, &pointer, mask, None, Some(16))
            .unwrap()
            .unwrap();
        let outcome = arrive.apply(&hub).unwrap();
        for outcome in [None, Some(outcome)] {
            assert_eq!(
                OperationEffect::MbarrierArrive {
                    plan: arrive,
                    outcome,
                }
                .to_owned_effect(),
                OwnedOperationEffect::MbarrierArrive {
                    plan: arrive,
                    outcome,
                }
            );
        }

        let wait = plan_physical_mbarrier_wait(&context, &pointer, mask, 0)
            .unwrap()
            .unwrap();
        assert_eq!(
            (OperationEffect::MbarrierWait {
                plan: wait,
                outcome: None,
            })
            .to_owned_effect(),
            OwnedOperationEffect::MbarrierWait {
                plan: wait,
                outcome: None,
            }
        );
    }

    #[test]
    fn optional_borrowed_payloads_are_stored_exactly_present_and_absent() {
        let (_physical, context, pointer) = shared_pointer();
        let hub = PhysicalBarrierHub::new();
        // Every lane of `shared_pointer` addresses the same barrier, so exactly
        // one lane may initialize it.
        let init =
            plan_physical_mbarrier_init(&context, &pointer, WarpMask::from_lanes([0]).unwrap(), 4)
                .unwrap();
        init.apply(&hub).unwrap();

        let mask = WarpMask::from_lanes([0, 1]).unwrap();
        let batch = plan_physical_mbarrier_arrive_lanes(&context, &pointer, mask, None, None, None)
            .unwrap();
        let outcome = batch.apply(&hub).unwrap();
        for outcome in [None, Some(&outcome)] {
            assert_eq!(
                OperationEffect::MbarrierArriveBatch {
                    plan: &batch,
                    outcome,
                }
                .to_owned_effect(),
                OwnedOperationEffect::MbarrierArriveBatch {
                    plan: batch.clone(),
                    outcome: outcome.cloned(),
                }
            );
        }
    }

    #[test]
    fn optional_borrowed_slice_payloads_are_stored_exactly_present_and_absent() {
        let barrier = crate::PhysicalBarrierId::new(22, 8, 0);
        let hub = PhysicalBarrierHub::new();
        hub.init(barrier, 1).unwrap();
        let plan = PhysicalMbarrierCompletionIssuePlan::single(barrier, 32);

        let action_ids = plan.apply(&hub).unwrap();
        for action_ids in [None, Some(action_ids.as_ref())] {
            assert_eq!(
                OperationEffect::MbarrierCompletionIssue {
                    plan: &plan,
                    action_ids,
                }
                .to_owned_effect(),
                OwnedOperationEffect::MbarrierCompletionIssue {
                    plan: plan.clone(),
                    action_ids: action_ids.map(Box::from),
                }
            );
        }
    }

    #[test]
    fn classic_cp_async_batch_and_arrive_on_payloads_are_stored_exactly() {
        check_cp_async_arrive_payloads(false, 41);
    }

    #[test]
    fn cp_async_mbarrier_arrive_pending_increment_payload_is_stored_exactly() {
        check_cp_async_arrive_payloads(true, 43);
    }

    fn check_cp_async_arrive_payloads(increments_pending: bool, op_id: u64) {
        let (_physical, context, pointer) = shared_pointer();
        let mask = WarpMask::from_lanes([0]).unwrap();
        let operation = OperationContext::new(
            DynamicOpId::new(0, context.global_warp_id(), 0, StaticOpId::new(op_id), []),
            OperationKind::AsyncIssue,
            mask,
        );
        let batch = AsyncGroupIssueBatchEffect::new(
            operation,
            AsyncGroupDomain::CpAsync,
            [(0, Vec::new(), Vec::new())],
        )
        .unwrap();
        assert_eq!(
            (OperationEffect::AsyncGroupIssueBatch(&batch)).to_owned_effect(),
            OwnedOperationEffect::AsyncGroupIssueBatch(batch.clone())
        );

        let barriers = PhysicalBarrierHub::new();
        let init = plan_physical_mbarrier_init(&context, &pointer, mask, 1).unwrap();
        init.apply(&barriers).unwrap();
        let arrive =
            plan_cp_async_mbarrier_arrive(&context, &pointer, mask, increments_pending).unwrap();
        assert_eq!(arrive.increments_pending(), increments_pending);
        assert_eq!(
            arrive.pending_increases().len(),
            usize::from(increments_pending)
        );
        let groups = AsyncGroupHub::new(context.topology());
        groups
            .issue_exact_batch(&context, &batch, Vec::new())
            .unwrap();
        let commit_plan = groups
            .commit_plan(&context, AsyncGroupDomain::CpAsync, mask)
            .unwrap();

        let effect = OperationEffect::CpAsyncMbarrierArrive {
            plan: &arrive,
            commit_plan: &commit_plan,
            outcome: None,
            actions: None,
        };
        assert_eq!(
            effect.name(),
            if increments_pending {
                "cp.async.mbarrier.arrive"
            } else {
                "cp.async.mbarrier.arrive.noinc"
            }
        );
        assert_eq!(
            effect.to_owned_effect(),
            OwnedOperationEffect::CpAsyncMbarrierArrive {
                plan: arrive.clone(),
                commit_plan: commit_plan.clone(),
                outcome: None,
                actions: None,
            }
        );

        let actions = arrive.apply(&barriers).unwrap();
        let outcome = groups
            .commit_detailed_with_physical_arrivals(commit_plan.clone(), &actions, |_| Ok(()))
            .unwrap();
        assert_eq!(
            (OperationEffect::CpAsyncMbarrierArrive {
                plan: &arrive,
                commit_plan: &commit_plan,
                outcome: Some(&outcome),
                actions: Some(&actions),
            })
            .to_owned_effect(),
            OwnedOperationEffect::CpAsyncMbarrierArrive {
                plan: arrive.clone(),
                commit_plan: commit_plan.clone(),
                outcome: Some(outcome.clone()),
                actions: Some(actions.clone()),
            }
        );
    }

    #[test]
    fn slice_payloads_survive_source_reuse() {
        let barrier = PhysicalBarrierId::new(22, 8, 0);
        let mut barriers = vec![barrier, PhysicalBarrierId::new(22, 16, 0)];
        let stored = OperationEffect::MbarrierInvalidate {
            barrier_ids: &barriers,
        }
        .to_owned_effect();
        barriers.clear();
        assert_eq!(
            stored,
            OwnedOperationEffect::MbarrierInvalidate {
                barrier_ids: Box::new([barrier, PhysicalBarrierId::new(22, 16, 0)]),
            },
        );

        let hub = PhysicalBarrierHub::new();
        hub.init(barrier, 1).unwrap();
        let plan = PhysicalMbarrierCompletionIssuePlan::single(barrier, 32);
        let mut action_ids = plan.apply(&hub).unwrap().into_vec();
        let stored = OperationEffect::MbarrierCompletionIssue {
            plan: &plan,
            action_ids: Some(&action_ids),
        }
        .to_owned_effect();
        let expected_ids = action_ids.clone().into_boxed_slice();
        action_ids.clear();
        assert_eq!(
            stored,
            OwnedOperationEffect::MbarrierCompletionIssue {
                plan,
                action_ids: Some(expected_ids)
            },
        );
    }
}
