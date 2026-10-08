use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use crate::{
    bf16_bits_to_f32, cuda_f32_add, cuda_f32_max, cuda_f32_min, cuda_f64_add, cuda_f64_max,
    cuda_f64_min, cuda_reduce_bf16_add, cuda_reduce_bf16_max, cuda_reduce_bf16_min,
    cuda_reduce_fp16_add, cuda_reduce_fp16_max, cuda_reduce_fp16_min, f32_to_bf16_bits,
    f32_to_fp16_bits, fp16_bits_to_f32, BlockedOperation, CompletionProgress, CompletionSource,
    DiagnosticLabel, EngineError, LaunchTopology, OccurrenceKey, ParticipantContract,
    ParticipantSet, ParticipantState, SynchronizationError, WarpContext, WarpMask, WARP_SIZE,
};

type Publisher<Input, Output> =
    dyn Fn(BTreeMap<usize, Input>) -> Result<Output, SynchronizationError> + Send + Sync;

/// Typed, dynamically keyed collective state owned by the engine.
///
/// One contribution is accepted from each contracted warp. The publisher runs
/// exactly once after the final contribution, and every waiter observes the
/// same `Arc<Output>`.
pub struct CollectiveHub<Input, Output> {
    publisher: Arc<Publisher<Input, Output>>,
    state: Mutex<BTreeMap<OccurrenceKey, CollectiveEntry<Input, Output>>>,
}

struct CollectiveEntry<Input, Output> {
    contract: ParticipantContract,
    contributions: BTreeMap<usize, Input>,
    submitted_participants: BTreeSet<usize>,
    publication: Publication<Output>,
    waiters: BTreeMap<usize, Waker>,
}

enum Publication<Output> {
    Collecting,
    Publishing,
    Published(Arc<Output>),
    Failed(SynchronizationError),
}

impl<Input, Output> CollectiveHub<Input, Output>
where
    Input: Send + Unpin + 'static,
    Output: Send + Sync + 'static,
{
    pub fn new(
        publisher: impl Fn(BTreeMap<usize, Input>) -> Result<Output, SynchronizationError>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            publisher: Arc::new(publisher),
            state: Mutex::new(BTreeMap::new()),
        }
    }

    /// Create a typed Future that owns this warp's contribution and contract.
    pub fn collect(
        self: &Arc<Self>,
        key: OccurrenceKey,
        contract: ParticipantContract,
        warp_id: usize,
        contribution: Input,
    ) -> Result<CollectiveWait<Input, Output>, SynchronizationError> {
        contract.validate_key_and_participant(&key, warp_id)?;
        Ok(CollectiveWait {
            hub: Arc::clone(self),
            key,
            contract,
            warp_id,
            contribution: Some(contribution),
            immediate_result: None,
            submitted: false,
            registered: false,
            finished: false,
        })
    }

    pub fn occurrence_count(&self) -> usize {
        self.state
            .lock()
            .expect("collective hub mutex poisoned")
            .len()
    }

    pub(crate) fn participant_contract(&self, key: &OccurrenceKey) -> Option<ParticipantContract> {
        self.state
            .lock()
            .expect("collective hub mutex poisoned")
            .get(key)
            .map(|entry| entry.contract.clone())
    }
}

/// Rendezvous collectives scoped to the warps that actually participate in one launch.
pub struct RendezvousHub {
    collective: Arc<CollectiveHub<(), ()>>,
    participation: Mutex<BTreeMap<OccurrenceKey, ParticipationEntry>>,
    participation_ordinals: Mutex<BTreeMap<(usize, String), u64>>,
    selected_warps: Option<Arc<BTreeSet<usize>>>,
}

struct ParticipationEntry {
    contract: ParticipantContract,
    arrived: BTreeSet<usize>,
}

impl Default for RendezvousHub {
    fn default() -> Self {
        Self::new()
    }
}

impl RendezvousHub {
    pub(crate) fn new() -> Self {
        Self {
            collective: Arc::new(CollectiveHub::new(|_| Ok(()))),
            participation: Mutex::new(BTreeMap::new()),
            participation_ordinals: Mutex::new(BTreeMap::new()),
            selected_warps: None,
        }
    }

    pub(crate) fn for_launch(selected_warps: Arc<BTreeSet<usize>>) -> Self {
        Self {
            collective: Arc::new(CollectiveHub::new(|_| Ok(()))),
            participation: Mutex::new(BTreeMap::new()),
            participation_ordinals: Mutex::new(BTreeMap::new()),
            selected_warps: Some(selected_warps),
        }
    }

    pub(crate) fn occurrence_count(&self) -> usize {
        self.collective.occurrence_count()
            + self
                .participation
                .lock()
                .expect("rendezvous participation mutex poisoned")
                .len()
    }

    pub(crate) fn participation_deadlock(&self) -> Option<(Vec<usize>, Vec<BlockedOperation>)> {
        let state = self
            .participation
            .lock()
            .expect("rendezvous participation mutex poisoned");
        let mut blocked_warps = BTreeSet::new();
        let mut blocked_operations = Vec::new();
        for (key, entry) in &*state {
            if entry.arrived.len() == entry.contract.participants().len() {
                continue;
            }
            let participants = ParticipantState::new(&entry.contract, &entry.arrived, None, None);
            for warp_id in entry.arrived.iter().copied() {
                blocked_warps.insert(warp_id);
                blocked_operations.push(BlockedOperation::new(
                    warp_id,
                    crate::AwaitedOperation::CollectivePublish,
                    key.clone(),
                    None,
                    participants.clone(),
                ));
            }
        }
        if blocked_operations.is_empty() {
            None
        } else {
            blocked_operations.sort_by(BlockedOperation::diagnostic_cmp);
            Some((blocked_warps.into_iter().collect(), blocked_operations))
        }
    }

    fn participate(
        &self,
        key: OccurrenceKey,
        contract: ParticipantContract,
        warp_id: usize,
    ) -> Result<(), SynchronizationError> {
        contract.validate_key_and_participant(&key, warp_id)?;
        let mut state = self
            .participation
            .lock()
            .expect("rendezvous participation mutex poisoned");
        let entry = state
            .entry(key.clone())
            .or_insert_with(|| ParticipationEntry {
                contract: contract.clone(),
                arrived: BTreeSet::new(),
            });
        if entry.contract != contract {
            return Err(SynchronizationError::ContractMismatch { key });
        }
        if !entry.arrived.insert(warp_id) {
            return Err(SynchronizationError::DuplicateContribution { key, warp_id });
        }
        Ok(())
    }

    fn launch_contract(
        &self,
        contract: ParticipantContract,
    ) -> Result<ParticipantContract, SynchronizationError> {
        let Some(selected_warps) = &self.selected_warps else {
            return Ok(contract);
        };
        let participants = ParticipantSet::new(
            contract
                .participants()
                .iter()
                .filter(|warp_id| selected_warps.contains(warp_id)),
        )?;
        Ok(ParticipantContract::explicit(
            contract.scope().clone(),
            participants,
        ))
    }

    pub(crate) fn rendezvous(
        self: &Arc<Self>,
        key: OccurrenceKey,
        contract: ParticipantContract,
        warp_id: usize,
    ) -> Result<CollectiveWait<(), ()>, SynchronizationError> {
        self.collective.collect(key, contract, warp_id, ())
    }

    pub(crate) fn warp(
        self: &Arc<Self>,
        static_op_id: u64,
        operation: impl Into<String>,
        loop_iteration_path: impl IntoIterator<Item = i64>,
        context: WarpContext,
    ) -> Result<CollectiveWait<(), ()>, SynchronizationError> {
        let contract = self.launch_contract(ParticipantContract::warp(context))?;
        let key = OccurrenceKey::from_label(
            static_op_id,
            crate::DiagnosticLabel::new(operation),
            loop_iteration_path,
            contract.scope().clone(),
        );
        contract.validate_key_and_participant(&key, context.global_warp_id())?;
        Ok(CollectiveWait {
            hub: Arc::clone(&self.collective),
            key,
            contract,
            warp_id: context.global_warp_id(),
            contribution: None,
            immediate_result: Some(Arc::new(())),
            submitted: true,
            registered: false,
            finished: false,
        })
    }

    pub(crate) fn warpgroup(
        self: &Arc<Self>,
        static_op_id: u64,
        operation: impl Into<String>,
        loop_iteration_path: impl IntoIterator<Item = i64>,
        context: WarpContext,
        warps_per_group: usize,
    ) -> Result<CollectiveWait<(), ()>, SynchronizationError> {
        let contract =
            self.launch_contract(ParticipantContract::warpgroup(context, warps_per_group)?)?;
        self.rendezvous(
            OccurrenceKey::from_label(
                static_op_id,
                crate::DiagnosticLabel::new(operation),
                loop_iteration_path,
                contract.scope().clone(),
            ),
            contract,
            context.global_warp_id(),
        )
    }

    pub(crate) fn cta(
        self: &Arc<Self>,
        static_op_id: u64,
        operation: impl Into<String>,
        loop_iteration_path: impl IntoIterator<Item = i64>,
        context: WarpContext,
    ) -> Result<CollectiveWait<(), ()>, SynchronizationError> {
        let contract = self.launch_contract(ParticipantContract::cta(context))?;
        self.rendezvous(
            OccurrenceKey::from_label(
                static_op_id,
                crate::DiagnosticLabel::new(operation),
                loop_iteration_path,
                contract.scope().clone(),
            ),
            contract,
            context.global_warp_id(),
        )
    }

    /// Register a generated tile participation contract without constructing a
    /// waiter.  Keep scope dispatch in this non-generic service so generated
    /// artifact crates do not monomorphize the participation state machine as
    /// part of every `WarpEngine<M>` rendezvous future.
    pub(crate) fn participate_generated_scope(
        &self,
        scope: &str,
        _static_op_id: u64,
        operation: &str,
        _loop_iteration_path: &[i64],
        context: WarpContext,
        active_mask: WarpMask,
        warps_per_group: usize,
    ) -> Result<bool, EngineError> {
        let contract = match scope {
            "participation:warpgroup" => ParticipantContract::warpgroup(context, warps_per_group)?,
            "participation:cta" => ParticipantContract::cta(context),
            _ => return Ok(false),
        };
        if active_mask != WarpMask::FULL {
            return Err(DiagnosticLabel::new(operation).warp_collective_divergence(active_mask));
        }
        let contract = self.launch_contract(contract)?;
        // Inline expansion and warp-role branches can give one convergent
        // collective a different static source op in each warp.  Hardware
        // matches the dynamic instruction sequence, so use the same per-warp
        // ordinal scheme as setmaxnreg while retaining the normalized tile
        // contract in `operation` to reject mismatched instructions.
        let ordinal = {
            let mut ordinals = self
                .participation_ordinals
                .lock()
                .expect("rendezvous participation ordinal mutex poisoned");
            let next = ordinals
                .entry((context.global_warp_id(), scope.to_string()))
                .or_insert(0);
            let ordinal = *next;
            *next = next.checked_add(1).ok_or_else(|| {
                EngineError::message("tile participation dynamic sequence ordinal overflow")
            })?;
            ordinal
        };
        self.participate(
            OccurrenceKey::new(
                ordinal,
                operation,
                std::iter::empty::<i64>(),
                contract.scope().clone(),
            ),
            contract,
            context.global_warp_id(),
        )?;
        Ok(true)
    }

    pub(crate) fn cluster(
        self: &Arc<Self>,
        static_op_id: u64,
        operation: impl Into<String>,
        loop_iteration_path: impl IntoIterator<Item = i64>,
        context: WarpContext,
    ) -> Result<CollectiveWait<(), ()>, SynchronizationError> {
        let contract = self.launch_contract(ParticipantContract::cluster(context))?;
        self.rendezvous(
            OccurrenceKey::from_label(
                static_op_id,
                crate::DiagnosticLabel::new(operation),
                loop_iteration_path,
                contract.scope().clone(),
            ),
            contract,
            context.global_warp_id(),
        )
    }

    pub(crate) fn grid(
        self: &Arc<Self>,
        static_op_id: u64,
        operation: impl Into<String>,
        loop_iteration_path: impl IntoIterator<Item = i64>,
        context: WarpContext,
    ) -> Result<CollectiveWait<(), ()>, SynchronizationError> {
        let contract = self.launch_contract(ParticipantContract::grid(context))?;
        self.rendezvous(
            OccurrenceKey::from_label(
                static_op_id,
                crate::DiagnosticLabel::new(operation),
                loop_iteration_path,
                contract.scope().clone(),
            ),
            contract,
            context.global_warp_id(),
        )
    }
}

impl CompletionSource for RendezvousHub {
    fn source_name(&self) -> &'static str {
        self.collective.source_name()
    }

    fn pump(&self) -> Result<CompletionProgress, SynchronizationError> {
        self.collective.pump()
    }

    fn blocked_operations(&self) -> Vec<BlockedOperation> {
        self.collective.blocked_operations()
    }

    fn validate_quiescent(&self) -> Result<(), SynchronizationError> {
        self.collective.validate_quiescent()?;
        let state = self
            .participation
            .lock()
            .expect("rendezvous participation mutex poisoned");
        for (key, entry) in &*state {
            if entry.arrived.len() != entry.contract.participants().len() {
                let participants =
                    ParticipantState::new(&entry.contract, &entry.arrived, None, None);
                return Err(key.completion_not_quiescent_error(
                    "rendezvous participation",
                    format_args!(" is incomplete: {participants}"),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CtaReduceOp {
    Sum,
    Max,
    Min,
}

impl CtaReduceOp {
    const fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Max => "max",
            Self::Min => "min",
        }
    }
}

fn cta_reduce_failure(details: impl Into<String>) -> SynchronizationError {
    SynchronizationError::CompletionSourceOperationFailed {
        source_name: "cta_reduce",
        details: details.into(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CtaReduceValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F16(Fp16Reduce),
    Bf16(Bf16Reduce),
    F32(f32),
    F64(f64),
}

pub trait CtaReduceElement: Copy {
    fn identity(operation: CtaReduceOp) -> Self;
    fn combine(operation: CtaReduceOp, lhs: Self, rhs: Self) -> Self;
    fn into_cta_reduce_value(self) -> CtaReduceValue;
    fn from_cta_reduce_value(value: CtaReduceValue) -> Result<Self, SynchronizationError>;
}

macro_rules! impl_integer_cta_reduce_element {
    ($rust_type:ty, $variant:ident) => {
        impl CtaReduceElement for $rust_type {
            fn identity(operation: CtaReduceOp) -> Self {
                match operation {
                    CtaReduceOp::Sum => 0,
                    CtaReduceOp::Max => <$rust_type>::MIN,
                    CtaReduceOp::Min => <$rust_type>::MAX,
                }
            }

            fn combine(operation: CtaReduceOp, lhs: Self, rhs: Self) -> Self {
                match operation {
                    CtaReduceOp::Sum => lhs.wrapping_add(rhs),
                    CtaReduceOp::Max => lhs.max(rhs),
                    CtaReduceOp::Min => lhs.min(rhs),
                }
            }

            fn into_cta_reduce_value(self) -> CtaReduceValue {
                CtaReduceValue::$variant(self)
            }

            fn from_cta_reduce_value(value: CtaReduceValue) -> Result<Self, SynchronizationError> {
                match value {
                    CtaReduceValue::$variant(value) => Ok(value),
                    other => Err(cta_reduce_failure(format!(
                        "CTA reduction result type mismatch: expected {}, got {}",
                        stringify!($rust_type),
                        other.type_name(),
                    ))),
                }
            }
        }
    };
}

impl_integer_cta_reduce_element!(i8, I8);
impl_integer_cta_reduce_element!(i16, I16);
impl_integer_cta_reduce_element!(i32, I32);
impl_integer_cta_reduce_element!(i64, I64);
impl_integer_cta_reduce_element!(u8, U8);
impl_integer_cta_reduce_element!(u16, U16);
impl_integer_cta_reduce_element!(u32, U32);
impl_integer_cta_reduce_element!(u64, U64);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fp16Reduce(u16);

impl Fp16Reduce {
    pub fn from_f32(value: f32) -> Self {
        Self(f32_to_fp16_bits(value))
    }

    pub fn to_f32(self) -> f32 {
        fp16_bits_to_f32(self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16Reduce(u16);

impl Bf16Reduce {
    pub fn from_f32(value: f32) -> Self {
        Self(f32_to_bf16_bits(value))
    }

    pub fn to_f32(self) -> f32 {
        bf16_bits_to_f32(self.0)
    }
}

macro_rules! impl_low_precision_cta_reduce_element {
    ($wrapper:ty, $variant:ident, $sum:path, $max:path, $min:path) => {
        impl CtaReduceElement for $wrapper {
            fn identity(operation: CtaReduceOp) -> Self {
                Self::from_f32(match operation {
                    CtaReduceOp::Sum => 0.0,
                    CtaReduceOp::Max => f32::NEG_INFINITY,
                    CtaReduceOp::Min => f32::INFINITY,
                })
            }

            fn combine(operation: CtaReduceOp, lhs: Self, rhs: Self) -> Self {
                let lhs = lhs.to_f32();
                let rhs = rhs.to_f32();
                Self::from_f32(match operation {
                    CtaReduceOp::Sum => $sum(lhs, rhs),
                    CtaReduceOp::Max => $max(lhs, rhs),
                    CtaReduceOp::Min => $min(lhs, rhs),
                })
            }

            fn into_cta_reduce_value(self) -> CtaReduceValue {
                CtaReduceValue::$variant(self)
            }

            fn from_cta_reduce_value(value: CtaReduceValue) -> Result<Self, SynchronizationError> {
                match value {
                    CtaReduceValue::$variant(value) => Ok(value),
                    other => Err(cta_reduce_failure(format!(
                        "CTA reduction result type mismatch: expected {}, got {}",
                        stringify!($wrapper),
                        other.type_name(),
                    ))),
                }
            }
        }
    };
}

impl_low_precision_cta_reduce_element!(
    Fp16Reduce,
    F16,
    cuda_reduce_fp16_add,
    cuda_reduce_fp16_max,
    cuda_reduce_fp16_min
);
impl_low_precision_cta_reduce_element!(
    Bf16Reduce,
    Bf16,
    cuda_reduce_bf16_add,
    cuda_reduce_bf16_max,
    cuda_reduce_bf16_min
);

impl CtaReduceElement for f32 {
    fn identity(operation: CtaReduceOp) -> Self {
        match operation {
            CtaReduceOp::Sum => 0.0,
            CtaReduceOp::Max => f32::NEG_INFINITY,
            CtaReduceOp::Min => f32::INFINITY,
        }
    }

    fn combine(operation: CtaReduceOp, lhs: Self, rhs: Self) -> Self {
        match operation {
            CtaReduceOp::Sum => cuda_f32_add(lhs, rhs),
            CtaReduceOp::Max => cuda_f32_max(lhs, rhs),
            CtaReduceOp::Min => cuda_f32_min(lhs, rhs),
        }
    }

    fn into_cta_reduce_value(self) -> CtaReduceValue {
        CtaReduceValue::F32(self)
    }

    fn from_cta_reduce_value(value: CtaReduceValue) -> Result<Self, SynchronizationError> {
        match value {
            CtaReduceValue::F32(value) => Ok(value),
            other => Err(cta_reduce_failure(format!(
                "CTA reduction result type mismatch: expected f32, got {}",
                other.type_name(),
            ))),
        }
    }
}

impl CtaReduceElement for f64 {
    fn identity(operation: CtaReduceOp) -> Self {
        match operation {
            CtaReduceOp::Sum => 0.0,
            CtaReduceOp::Max => f64::NEG_INFINITY,
            CtaReduceOp::Min => f64::INFINITY,
        }
    }

    fn combine(operation: CtaReduceOp, lhs: Self, rhs: Self) -> Self {
        match operation {
            CtaReduceOp::Sum => cuda_f64_add(lhs, rhs),
            CtaReduceOp::Max => cuda_f64_max(lhs, rhs),
            CtaReduceOp::Min => cuda_f64_min(lhs, rhs),
        }
    }

    fn into_cta_reduce_value(self) -> CtaReduceValue {
        CtaReduceValue::F64(self)
    }

    fn from_cta_reduce_value(value: CtaReduceValue) -> Result<Self, SynchronizationError> {
        match value {
            CtaReduceValue::F64(value) => Ok(value),
            other => Err(cta_reduce_failure(format!(
                "CTA reduction result type mismatch: expected f64, got {}",
                other.type_name(),
            ))),
        }
    }
}

impl CtaReduceValue {
    fn type_name(self) -> &'static str {
        match self {
            Self::I8(_) => "i8",
            Self::I16(_) => "i16",
            Self::I32(_) => "i32",
            Self::I64(_) => "i64",
            Self::U8(_) => "u8",
            Self::U16(_) => "u16",
            Self::U32(_) => "u32",
            Self::U64(_) => "u64",
            Self::F16(_) => "float16",
            Self::Bf16(_) => "bfloat16",
            Self::F32(_) => "f32",
            Self::F64(_) => "f64",
        }
    }

    fn identity(self, operation: CtaReduceOp) -> Self {
        match self {
            Self::I8(_) => Self::I8(i8::identity(operation)),
            Self::I16(_) => Self::I16(i16::identity(operation)),
            Self::I32(_) => Self::I32(i32::identity(operation)),
            Self::I64(_) => Self::I64(i64::identity(operation)),
            Self::U8(_) => Self::U8(u8::identity(operation)),
            Self::U16(_) => Self::U16(u16::identity(operation)),
            Self::U32(_) => Self::U32(u32::identity(operation)),
            Self::U64(_) => Self::U64(u64::identity(operation)),
            Self::F16(_) => Self::F16(Fp16Reduce::identity(operation)),
            Self::Bf16(_) => Self::Bf16(Bf16Reduce::identity(operation)),
            Self::F32(_) => Self::F32(f32::identity(operation)),
            Self::F64(_) => Self::F64(f64::identity(operation)),
        }
    }

    fn combine(self, operation: CtaReduceOp, rhs: Self) -> Result<Self, SynchronizationError> {
        let result = match (self, rhs) {
            (Self::I8(lhs), Self::I8(rhs)) => Self::I8(i8::combine(operation, lhs, rhs)),
            (Self::I16(lhs), Self::I16(rhs)) => Self::I16(i16::combine(operation, lhs, rhs)),
            (Self::I32(lhs), Self::I32(rhs)) => Self::I32(i32::combine(operation, lhs, rhs)),
            (Self::I64(lhs), Self::I64(rhs)) => Self::I64(i64::combine(operation, lhs, rhs)),
            (Self::U8(lhs), Self::U8(rhs)) => Self::U8(u8::combine(operation, lhs, rhs)),
            (Self::U16(lhs), Self::U16(rhs)) => Self::U16(u16::combine(operation, lhs, rhs)),
            (Self::U32(lhs), Self::U32(rhs)) => Self::U32(u32::combine(operation, lhs, rhs)),
            (Self::U64(lhs), Self::U64(rhs)) => Self::U64(u64::combine(operation, lhs, rhs)),
            (Self::F16(lhs), Self::F16(rhs)) => Self::F16(Fp16Reduce::combine(operation, lhs, rhs)),
            (Self::Bf16(lhs), Self::Bf16(rhs)) => {
                Self::Bf16(Bf16Reduce::combine(operation, lhs, rhs))
            }
            (Self::F32(lhs), Self::F32(rhs)) => Self::F32(f32::combine(operation, lhs, rhs)),
            (Self::F64(lhs), Self::F64(rhs)) => Self::F64(f64::combine(operation, lhs, rhs)),
            (lhs, rhs) => {
                return Err(cta_reduce_failure(format!(
                    "CTA reduction participants disagree on scalar type: {} versus {}",
                    lhs.type_name(),
                    rhs.type_name(),
                )))
            }
        };
        Ok(result)
    }

    fn same_type(self, other: Self) -> bool {
        self.type_name() == other.type_name()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CtaReduceContribution {
    operation: CtaReduceOp,
    warp_partial: CtaReduceValue,
}

impl CtaReduceContribution {
    pub fn new<T: CtaReduceElement>(operation: CtaReduceOp, warp_partial: T) -> Self {
        Self {
            operation,
            warp_partial: warp_partial.into_cta_reduce_value(),
        }
    }
}

/// CTA-wide typed scalar reduction collective used by generated kernels.
pub type CtaReduceHub = CollectiveHub<CtaReduceContribution, CtaReduceValue>;

impl CollectiveHub<CtaReduceContribution, CtaReduceValue> {
    /// Build the cross-warp stage of the two-stage XOR reduction used by
    /// `tirx.cuda.cta_reduce`. Generated code owns the first-stage warp
    /// reduction so it can reproduce the helper's scratch writes exactly.
    pub fn cta_reduce_hub(topology: LaunchTopology) -> Self {
        Self::new(move |inputs: BTreeMap<usize, CtaReduceContribution>| {
            let first = inputs
                .values()
                .next()
                .ok_or_else(|| cta_reduce_failure("CTA reduction has no inputs"))?;
            let operation = first.operation;
            let exemplar = first.warp_partial;
            let mut partials = [exemplar.identity(operation); WARP_SIZE];
            for (global_warp_id, contribution) in inputs {
                if contribution.operation != operation {
                    return Err(cta_reduce_failure(format!(
                        "CTA reduction participants disagree on operation: {} versus {}",
                        operation.name(),
                        contribution.operation.name(),
                    )));
                }
                if !contribution.warp_partial.same_type(exemplar) {
                    return Err(cta_reduce_failure(format!(
                        "CTA reduction participants disagree on scalar type: {} versus {}",
                        exemplar.type_name(),
                        contribution.warp_partial.type_name(),
                    )));
                }
                partials[global_warp_id % topology.warps_per_cta()] = contribution.warp_partial;
            }
            for delta in [16_usize, 8, 4, 2, 1] {
                let previous = partials;
                for lane in 0..WARP_SIZE {
                    partials[lane] = previous[lane].combine(operation, previous[lane ^ delta])?;
                }
            }
            Ok(partials[0])
        })
    }
}

impl<Input, Output> CompletionSource for CollectiveHub<Input, Output>
where
    Input: Send + Unpin + 'static,
    Output: Send + Sync + 'static,
{
    fn source_name(&self) -> &'static str {
        "collective"
    }

    fn pump(&self) -> Result<CompletionProgress, SynchronizationError> {
        Ok(CompletionProgress::default())
    }

    fn blocked_operations(&self) -> Vec<BlockedOperation> {
        let state = self.state.lock().expect("collective hub mutex poisoned");
        let mut blocked = Vec::new();
        for (key, entry) in state.iter() {
            let participant_state =
                ParticipantState::new(&entry.contract, &entry.submitted_participants, None, None);
            for warp_id in entry.waiters.keys().copied() {
                blocked.push(BlockedOperation::new(
                    warp_id,
                    crate::AwaitedOperation::CollectivePublish,
                    key.clone(),
                    None,
                    participant_state.clone(),
                ));
            }
        }
        blocked
    }

    fn validate_quiescent(&self) -> Result<(), SynchronizationError> {
        let state = self.state.lock().expect("collective hub mutex poisoned");
        for (key, entry) in &*state {
            match &entry.publication {
                Publication::Published(_) if entry.waiters.is_empty() => {}
                Publication::Failed(error) => return Err(error.clone()),
                Publication::Collecting | Publication::Publishing | Publication::Published(_) => {
                    let publication = match &entry.publication {
                        Publication::Collecting => "collecting",
                        Publication::Publishing => "publishing",
                        Publication::Published(_) => "published",
                        Publication::Failed(_) => unreachable!(),
                    };
                    return Err(key.completion_not_quiescent_error(
                        self.source_name(),
                        format_args!(
                            " is {publication}: contributions={}/{}, waiting_warps={:?}",
                            entry.submitted_participants.len(),
                            entry.contract.participants().len(),
                            entry.waiters.keys().copied().collect::<Vec<_>>()
                        ),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Future that contributes once and resolves to the uniquely published result.
pub struct CollectiveWait<Input, Output>
where
    Input: Send + Unpin + 'static,
    Output: Send + Sync + 'static,
{
    hub: Arc<CollectiveHub<Input, Output>>,
    key: OccurrenceKey,
    contract: ParticipantContract,
    warp_id: usize,
    contribution: Option<Input>,
    immediate_result: Option<Arc<Output>>,
    submitted: bool,
    registered: bool,
    finished: bool,
}

impl<Input, Output> Future for CollectiveWait<Input, Output>
where
    Input: Send + Unpin + 'static,
    Output: Send + Sync + 'static,
{
    type Output = Result<Arc<Output>, SynchronizationError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if let Some(result) = &this.immediate_result {
            this.finished = true;
            return Poll::Ready(Ok(Arc::clone(result)));
        }
        let publish_inputs = {
            let mut state = this
                .hub
                .state
                .lock()
                .expect("collective hub mutex poisoned");
            let entry = state
                .entry(this.key.clone())
                .or_insert_with(|| CollectiveEntry {
                    contract: this.contract.clone(),
                    contributions: BTreeMap::new(),
                    submitted_participants: BTreeSet::new(),
                    publication: Publication::Collecting,
                    waiters: BTreeMap::new(),
                });
            if entry.contract != this.contract {
                return Poll::Ready(Err(SynchronizationError::ContractMismatch {
                    key: this.key.clone(),
                }));
            }

            if !this.submitted {
                if !entry.submitted_participants.insert(this.warp_id) {
                    return Poll::Ready(Err(SynchronizationError::DuplicateContribution {
                        key: this.key.clone(),
                        warp_id: this.warp_id,
                    }));
                }
                let contribution = this
                    .contribution
                    .take()
                    .expect("an unsubmitted collective Future owns its contribution");
                entry.contributions.insert(this.warp_id, contribution);
                this.submitted = true;
            }

            match &entry.publication {
                Publication::Published(result) => {
                    this.registered = false;
                    this.finished = true;
                    return Poll::Ready(Ok(Arc::clone(result)));
                }
                Publication::Failed(error) => {
                    this.registered = false;
                    this.finished = true;
                    return Poll::Ready(Err(error.clone()));
                }
                Publication::Collecting
                    if entry.submitted_participants.len()
                        == entry.contract.participants().len() =>
                {
                    entry.publication = Publication::Publishing;
                    std::mem::take(&mut entry.contributions)
                }
                Publication::Collecting | Publication::Publishing => {
                    register_collective_waiter(
                        entry,
                        &this.key,
                        this.warp_id,
                        this.registered,
                        context.waker(),
                    )?;
                    this.registered = true;
                    return Poll::Pending;
                }
            }
        };

        let publication = (this.hub.publisher)(publish_inputs);
        let (result, wakers) = {
            let mut state = this
                .hub
                .state
                .lock()
                .expect("collective hub mutex poisoned");
            let entry = state
                .get_mut(&this.key)
                .expect("collective entry cannot disappear during publication");
            let result = match publication {
                Ok(output) => {
                    let output = Arc::new(output);
                    entry.publication = Publication::Published(Arc::clone(&output));
                    Ok(output)
                }
                Err(error) => {
                    entry.publication = Publication::Failed(error.clone());
                    Err(error)
                }
            };
            let wakers: Vec<Waker> = std::mem::take(&mut entry.waiters).into_values().collect();
            (result, wakers)
        };
        for waker in wakers {
            waker.wake();
        }
        this.registered = false;
        this.finished = true;
        Poll::Ready(result)
    }
}

impl<Input, Output> Drop for CollectiveWait<Input, Output>
where
    Input: Send + Unpin + 'static,
    Output: Send + Sync + 'static,
{
    fn drop(&mut self) {
        if !self.registered || self.finished {
            return;
        }
        let mut state = self
            .hub
            .state
            .lock()
            .expect("collective hub mutex poisoned");
        if let Some(entry) = state.get_mut(&self.key) {
            entry.waiters.remove(&self.warp_id);
        }
    }
}

fn register_collective_waiter<Input, Output>(
    entry: &mut CollectiveEntry<Input, Output>,
    key: &OccurrenceKey,
    warp_id: usize,
    already_registered: bool,
    new_waker: &Waker,
) -> Result<(), SynchronizationError> {
    match entry.waiters.get_mut(&warp_id) {
        Some(waker) if already_registered => {
            if !waker.will_wake(new_waker) {
                *waker = new_waker.clone();
            }
        }
        Some(_) => {
            return Err(SynchronizationError::DuplicateWaiter {
                key: key.clone(),
                phase: None,
                warp_id,
            });
        }
        None => {
            entry.waiters.insert(warp_id, new_waker.clone());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompletionRegistry, Executor, LaunchTopology, ScopeInstance, WarpTask};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn collective_publisher_runs_exactly_once() {
        let publication_count = Arc::new(AtomicUsize::new(0));
        let collective = Arc::new(CollectiveHub::new({
            let publication_count = Arc::clone(&publication_count);
            move |inputs: BTreeMap<usize, i32>| {
                publication_count.fetch_add(1, Ordering::SeqCst);
                Ok(inputs.into_values().sum::<i32>())
            }
        }));
        let scope = ScopeInstance::Custom { domain_id: 17 };
        let contract = ParticipantContract::explicit(
            scope.clone(),
            crate::ParticipantSet::new([0, 1, 2]).unwrap(),
        );
        let outputs = Arc::new(Mutex::new(BTreeMap::new()));
        let tasks = [0, 1, 2].map(|warp_id| {
            let collective = Arc::clone(&collective);
            let contract = contract.clone();
            let outputs = Arc::clone(&outputs);
            let key = OccurrenceKey::single(101, "reduce", 5, scope.clone());
            WarpTask::new(warp_id, async move {
                let result = collective
                    .collect(key, contract, warp_id, (warp_id as i32 + 1) * 10)?
                    .await?;
                outputs.lock().unwrap().insert(warp_id, *result);
                Ok(())
            })
        });

        let stats = Executor::default().run(tasks).unwrap();
        assert_eq!(stats.completed_task_count, 3);
        assert_eq!(publication_count.load(Ordering::SeqCst), 1);
        assert_eq!(
            *outputs.lock().unwrap(),
            BTreeMap::from([(0, 60), (1, 60), (2, 60)])
        );
    }

    #[test]
    fn cta_reduce_hub_owns_the_cross_warp_stage() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let collective = Arc::new(CtaReduceHub::cta_reduce_hub(topology));
        let tasks = topology.warp_contexts().map(|context| {
            let collective = Arc::clone(&collective);
            WarpTask::new(context.global_warp_id(), async move {
                let contract = ParticipantContract::cta(context);
                let key = OccurrenceKey::new(
                    102,
                    "cta-sum",
                    std::iter::empty::<i64>(),
                    contract.scope().clone(),
                );
                let contribution = CtaReduceContribution::new(
                    CtaReduceOp::Sum,
                    32.0 * (context.warp_id_in_cta() + 1) as f32,
                );
                let result = collective
                    .collect(key, contract, context.global_warp_id(), contribution)?
                    .await?;
                assert_eq!(
                    <f32 as CtaReduceElement>::from_cta_reduce_value(*result)?,
                    96.0
                );
                Ok(())
            })
        });

        let stats = Executor::default().run(tasks).unwrap();
        assert_eq!(stats.completed_task_count, 2);
    }

    #[test]
    fn cta_reduce_hub_supports_max_and_min_identities() {
        for (operation, expected) in [(CtaReduceOp::Max, 7.0), (CtaReduceOp::Min, -3.0)] {
            let topology = LaunchTopology::new(1, 1, 2).unwrap();
            let collective = Arc::new(CtaReduceHub::cta_reduce_hub(topology));
            let tasks = topology.warp_contexts().map(|context| {
                let collective = Arc::clone(&collective);
                WarpTask::new(context.global_warp_id(), async move {
                    let contract = ParticipantContract::cta(context);
                    let key = OccurrenceKey::new(
                        103,
                        "cta-reduce",
                        std::iter::empty::<i64>(),
                        contract.scope().clone(),
                    );
                    let partial: f32 = if context.warp_id_in_cta() == 0 {
                        -3.0
                    } else {
                        7.0
                    };
                    let result = collective
                        .collect(
                            key,
                            contract,
                            context.global_warp_id(),
                            CtaReduceContribution::new(operation, partial),
                        )?
                        .await?;
                    assert_eq!(
                        <f32 as CtaReduceElement>::from_cta_reduce_value(*result)?,
                        expected
                    );
                    Ok(())
                })
            });

            Executor::default().run(tasks).unwrap();
        }
    }

    #[test]
    fn cta_reduce_hub_supports_integer_and_float64_scalars() {
        for expected in [CtaReduceValue::I32(30), CtaReduceValue::F64(3.75)] {
            let topology = LaunchTopology::new(1, 1, 2).unwrap();
            let collective = Arc::new(CtaReduceHub::cta_reduce_hub(topology));
            let tasks = topology.warp_contexts().map(|context| {
                let collective = Arc::clone(&collective);
                WarpTask::new(context.global_warp_id(), async move {
                    let contract = ParticipantContract::cta(context);
                    let key = OccurrenceKey::new(
                        104,
                        "cta-typed-sum",
                        std::iter::empty::<i64>(),
                        contract.scope().clone(),
                    );
                    let contribution = match expected {
                        CtaReduceValue::I32(_) => CtaReduceContribution::new(
                            CtaReduceOp::Sum,
                            if context.warp_id_in_cta() == 0 {
                                10_i32
                            } else {
                                20_i32
                            },
                        ),
                        CtaReduceValue::F64(_) => CtaReduceContribution::new(
                            CtaReduceOp::Sum,
                            if context.warp_id_in_cta() == 0 {
                                1.25_f64
                            } else {
                                2.5_f64
                            },
                        ),
                        _ => unreachable!(),
                    };
                    let result = collective
                        .collect(key, contract, context.global_warp_id(), contribution)?
                        .await?;
                    assert_eq!(*result, expected);
                    Ok(())
                })
            });

            Executor::default().run(tasks).unwrap();
        }
    }

    #[test]
    fn float64_cta_reduce_combine_preserves_cuda_zero_and_nan_selection() {
        let nan_a = f64::from_bits(0x7ff8_0000_0000_1234);
        let nan_b = f64::from_bits(0xfff8_0000_0000_5678);

        assert_eq!(
            <f64 as CtaReduceElement>::combine(CtaReduceOp::Max, -0.0, -0.0).to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(
            <f64 as CtaReduceElement>::combine(CtaReduceOp::Min, 0.0, 0.0).to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            <f64 as CtaReduceElement>::combine(CtaReduceOp::Max, nan_a, nan_b).to_bits(),
            nan_b.to_bits()
        );
        assert_eq!(
            <f64 as CtaReduceElement>::combine(CtaReduceOp::Min, nan_a, nan_b).to_bits(),
            nan_b.to_bits()
        );
    }

    #[test]
    fn warp_local_rendezvous_bypasses_waiting() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let rendezvous = Arc::new(RendezvousHub::new());
        let task = {
            let rendezvous = Arc::clone(&rendezvous);
            WarpTask::new(0, async move {
                rendezvous.warp(201, "warp-sync", [11], context)?.await?;
                Ok(())
            })
        };

        let stats = Executor::default().run([task]).unwrap();
        assert_eq!(stats.poll_count, 1);
        assert_eq!(stats.poll_order, vec![0]);
        assert_eq!(rendezvous.occurrence_count(), 0);
    }

    #[test]
    fn cta_warpgroup_and_cluster_rendezvous_compose() {
        let topology = LaunchTopology::new(1, 2, 4).unwrap();
        let rendezvous = Arc::new(RendezvousHub::new());
        let tasks = topology.warp_contexts().map(|context| {
            let rendezvous = Arc::clone(&rendezvous);
            WarpTask::new(context.global_warp_id(), async move {
                rendezvous.cta(301, "cta-sync", [0], context)?.await?;
                rendezvous
                    .warpgroup(302, "wg-sync", [0], context, 2)?
                    .await?;
                rendezvous
                    .cluster(303, "cluster-sync", [0], context)?
                    .await?;
                Ok(())
            })
        });
        let mut completions = CompletionRegistry::new();
        completions.register(Arc::clone(&rendezvous));

        let stats = Executor::default()
            .run_with_completions(tasks, &completions)
            .unwrap();
        assert_eq!(stats.completed_task_count, topology.warp_count());
        assert!(stats.poll_count > topology.warp_count());
    }

    #[test]
    fn launch_rendezvous_contracts_include_only_selected_warps() {
        let topology = LaunchTopology::new(1, 2, 2).unwrap();
        let selected_warps = Arc::new(BTreeSet::from([0, 1]));
        let rendezvous = Arc::new(RendezvousHub::for_launch(selected_warps));
        let tasks = topology.warp_contexts().take(2).map(|context| {
            let rendezvous = Arc::clone(&rendezvous);
            WarpTask::new(context.global_warp_id(), async move {
                rendezvous
                    .cluster(401, "selected-cluster-sync", [0], context)?
                    .await?;
                Ok(())
            })
        });
        let mut completions = CompletionRegistry::new();
        completions.register(Arc::clone(&rendezvous));

        let stats = Executor::default()
            .run_with_topology_and_completions(tasks, topology, &completions)
            .unwrap();
        assert_eq!(stats.completed_task_count, 2);
    }
}
