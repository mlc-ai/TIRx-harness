use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::memory::{SemanticProgress, SemanticProgressSnapshot, SemanticProgressWatch};
use crate::physical_access::CompactPhysicalAccessBatch;
use crate::runtime::RuntimeBuffer;
use crate::{
    AnalysisGapKind, CompletionActionEffect, CompletionEffect, EngineError, OperationContext,
    OperationEffect, OperationKind, PhysicalAccessDescriptor, PhysicalAccessKind,
    PhysicalAccessSpace, PhysicalAllocationId, PhysicalByteSpan, WarpMask,
};

/// Minimal identity needed to reuse one single-lane global-read analysis event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CachedGlobalReadAccess {
    global_warp_id: usize,
    per_warp_sequence: u64,
    source_op_id: u64,
    descriptor: PhysicalAccessDescriptor,
    lane: usize,
    span: PhysicalByteSpan,
}

impl CachedGlobalReadAccess {
    pub(crate) fn new(
        global_warp_id: usize,
        per_warp_sequence: u64,
        source_op_id: u64,
        descriptor: PhysicalAccessDescriptor,
        lane: usize,
        span: PhysicalByteSpan,
    ) -> Self {
        debug_assert_eq!(descriptor.space(), PhysicalAccessSpace::Global);
        debug_assert_eq!(descriptor.kind(), PhysicalAccessKind::Read);
        Self {
            global_warp_id,
            per_warp_sequence,
            source_op_id,
            descriptor,
            lane,
            span,
        }
    }

    pub(crate) const fn global_warp_id(self) -> usize {
        self.global_warp_id
    }

    pub(crate) const fn per_warp_sequence(self) -> u64 {
        self.per_warp_sequence
    }

    pub(crate) const fn source_op_id(self) -> u64 {
        self.source_op_id
    }

    pub(crate) const fn descriptor(self) -> PhysicalAccessDescriptor {
        self.descriptor
    }

    pub(crate) const fn lane(self) -> usize {
        self.lane
    }

    pub(crate) const fn span(self) -> PhysicalByteSpan {
        self.span
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CachedGlobalReadFinish {
    Stable,
    Reprocess,
}

/// Exact-range progress owned by a mode and consumed by the engine scheduler.
///
/// The wrapper keeps the checker/engine contract independent of the physical
/// memory implementation's launch-wide progress hub while reusing its
/// lost-wake-safe generation primitive internally.
#[derive(Clone, Default)]
pub(crate) struct GlobalMemoryProgress(SemanticProgress);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlobalMemoryProgressEpoch(SemanticProgressSnapshot);

impl GlobalMemoryProgress {
    pub(crate) fn snapshot(&self) -> GlobalMemoryProgressEpoch {
        GlobalMemoryProgressEpoch(self.0.snapshot())
    }

    pub(crate) fn record_change(&self) {
        self.0.record_change();
    }
}

#[derive(Clone, Default)]
pub(crate) struct GlobalMemoryProgressSnapshot {
    entries: Vec<(GlobalMemoryProgress, GlobalMemoryProgressEpoch)>,
}

impl GlobalMemoryProgressSnapshot {
    pub(crate) fn capture(progresses: impl IntoIterator<Item = GlobalMemoryProgress>) -> Self {
        Self {
            entries: progresses
                .into_iter()
                .map(|progress| {
                    let snapshot = progress.snapshot();
                    (progress, snapshot)
                })
                .collect(),
        }
    }

    pub(crate) fn merge(&mut self, other: Self) {
        self.entries.extend(other.entries);
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn changed(&self) -> bool {
        self.entries
            .iter()
            .any(|(progress, observed)| progress.snapshot() != *observed)
    }

    pub(crate) fn watch(&self) -> GlobalMemoryProgressWatch {
        GlobalMemoryProgressWatch {
            watches: self
                .entries
                .iter()
                .map(|(progress, observed)| progress.0.watch(observed.0))
                .collect(),
        }
    }
}

pub(crate) struct GlobalMemoryProgressWatch {
    watches: Vec<SemanticProgressWatch>,
}

impl Future for GlobalMemoryProgressWatch {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self
            .watches
            .iter_mut()
            .any(|watch| Pin::new(watch).poll(context).is_ready())
        {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

/// Compile-time policy selecting the behavior layered on one transpiled kernel.
///
/// The generated control-flow body is generic over this trait. Mode-specific
/// launch state is shared by every concrete warp without introducing a runtime
/// semantic-op stream.
#[allow(private_bounds)] // Private supertrait seals implementation hooks inside the engine.
pub trait EngineMode: EngineModeImpl {}

impl<T: EngineModeImpl> EngineMode for T {}

/// Engine-private implementation hooks behind the generated-code marker.
///
/// Keeping these hooks off [`EngineMode`] prevents checker internals from
/// becoming generated-artifact ABI merely because a generated helper needs a
/// generic mode bound.
pub(crate) trait EngineModeImpl: Send + Sync + 'static {
    type LaunchState: Send + Sync + 'static;
    type GlobalMemoryTransactionGuard<'a>: Default + 'a
    where
        Self: 'a;

    const NAME: &'static str;
    const OBSERVES_OPERATIONS: bool;
    /// Whether physical accesses must preserve the PTX proxy-fence domain.
    ///
    /// Only Racecheck consumes this metadata. Keeping the default false lets
    /// NumSim and Synccheck erase proxy-domain resolution from their generated
    /// `WarpEngine<M>` instantiations.
    const OBSERVES_PROXY_MEMORY_DOMAINS: bool = false;
    const USES_GLOBAL_MEMORY_TRANSACTION: bool = false;
    const USES_CACHED_GLOBAL_READ_FAST_PATH: bool = false;

    fn observes_analysis_gap(_state: &Self::LaunchState, _kind: AnalysisGapKind) -> bool {
        Self::OBSERVES_OPERATIONS
    }

    fn controls_physical_access(
        _state: &Self::LaunchState,
        _kind: OperationKind,
        _space: PhysicalAccessSpace,
    ) -> bool {
        Self::OBSERVES_OPERATIONS
    }

    /// Whether a mode can skip resolving the physical space and allocation for
    /// one ordinary operation kind.
    ///
    /// Opting in promises that `controls_physical_access_allocation` is false
    /// for every possible runtime buffer and that
    /// `after_unobserved_physical_access` has no mode-visible effect. The
    /// numerical operation and its lazy error context still execute normally.
    fn elides_physical_access_resolution(_state: &Self::LaunchState, _kind: OperationKind) -> bool {
        false
    }

    fn elides_physical_access_resolution_for_buffer(
        state: &Self::LaunchState,
        kind: OperationKind,
        _buffer: &RuntimeBuffer,
    ) -> bool {
        Self::elides_physical_access_resolution(state, kind)
    }

    fn controls_physical_access_allocation(
        state: &Self::LaunchState,
        kind: OperationKind,
        space: PhysicalAccessSpace,
        _allocation: Option<PhysicalAllocationId>,
    ) -> bool {
        Self::controls_physical_access(state, kind, space)
    }

    fn observes_physical_access_batch(
        _state: &Self::LaunchState,
        _descriptor: PhysicalAccessDescriptor,
        _mask: WarpMask,
    ) -> bool {
        Self::OBSERVES_OPERATIONS
    }

    fn observes_atomic_physical_access_batch(
        state: &Self::LaunchState,
        descriptor: PhysicalAccessDescriptor,
        mask: WarpMask,
        _return_sync_relevant: bool,
    ) -> bool {
        Self::observes_physical_access_batch(state, descriptor, mask)
    }

    /// Whether async payload/group issue must resolve its complete physical footprint.
    ///
    /// Eligibility probes may skip this expensive work when any executed
    /// unsupported effect already makes the result incomplete. A probe that
    /// observes no blocker must be discarded and followed by a full run.
    fn resolves_async_accesses(_state: &Self::LaunchState) -> bool {
        Self::OBSERVES_OPERATIONS
    }

    /// Whether generated element callbacks may accumulate one compact exact
    /// footprint instead of materializing an owned batch per copied element.
    /// Modes that retain element-level access records keep the default.
    fn compacts_async_accesses(_state: &Self::LaunchState) -> bool {
        false
    }

    /// Whether the mode can transactionally consume an allocation-free
    /// synchronous one-span-per-lane access.
    ///
    /// This is an engine-internal fast path, not generated-artifact ABI. A
    /// mode opting in validates the complete compact batch before numeric
    /// execution and commits that validation only after numeric success. It
    /// must not retain the owned lane records exposed by
    /// `PhysicalAccessBatch`.
    fn compacts_direct_physical_access(
        _state: &Self::LaunchState,
        _descriptor: PhysicalAccessDescriptor,
        _mask: WarpMask,
        _atomic_return_sync_relevant: bool,
    ) -> bool {
        false
    }

    /// Whether compact validation is analysis-only and may run after the
    /// numeric effect succeeds.
    ///
    /// Opting in means a numeric failure records no analysis state and an
    /// analysis rejection terminates the run, so the rejected run's numeric
    /// mutation is never observable as a result. This permits a checker to
    /// fuse validation and commit into one in-place traversal.
    fn applies_compact_physical_access_after_numeric(
        _state: &Self::LaunchState,
        _descriptor: PhysicalAccessDescriptor,
    ) -> bool {
        false
    }

    fn before_compact_physical_access(
        _state: &Self::LaunchState,
        _batch: &CompactPhysicalAccessBatch<'_>,
    ) -> Result<(), EngineError> {
        Err(EngineError::message(
            "engine mode selected compact physical access without a validation implementation",
        ))
    }

    fn after_compact_physical_access(
        _state: &Self::LaunchState,
        _batch: &CompactPhysicalAccessBatch<'_>,
    ) -> Result<(), EngineError> {
        Err(EngineError::message(
            "engine mode selected compact physical access without a commit implementation",
        ))
    }

    fn begin_cached_global_read(
        _state: &Self::LaunchState,
        _access: CachedGlobalReadAccess,
    ) -> Result<bool, EngineError> {
        Ok(false)
    }

    fn finish_cached_global_read(
        _state: &Self::LaunchState,
        _access: CachedGlobalReadAccess,
    ) -> Result<CachedGlobalReadFinish, EngineError> {
        Err(EngineError::message(
            "engine mode finished a cached global read without an implementation",
        ))
    }

    fn global_memory_progress_snapshot(
        _state: &Self::LaunchState,
        _spans: impl IntoIterator<Item = PhysicalByteSpan>,
    ) -> Option<GlobalMemoryProgressSnapshot> {
        None
    }

    /// The writes a wait on one declared word could still be released by,
    /// and how many of that word's writes precede them.
    ///
    /// A wait needs the values to decide which write its predicate first
    /// accepts, and the predicate reads thread-local scalars that only the
    /// instruction can see -- so the values come out to the instruction rather
    /// than the predicate going in. Empty for a mode that keeps no history and
    /// for an address no protocol declared.
    fn declared_word_candidates(
        _state: &Self::LaunchState,
        _span: PhysicalByteSpan,
        _warp_id: usize,
        _lane: usize,
    ) -> (usize, Vec<u64>) {
        (0, Vec::new())
    }

    /// Whether high-level TCGEN operations must resolve exact SMEM/TMEM
    /// footprints. Queue/token protocol state is modeled independently.
    fn observes_tcgen_accesses(_state: &Self::LaunchState) -> bool {
        false
    }

    /// Notify the mode that one physical access executed without it observing.
    ///
    /// Covers both reasons an access goes unobserved: the mode declined control
    /// of it, or it elided the resolution entirely. The two used to be separate
    /// hooks with byte-identical implementations.
    ///
    /// This stays a hook rather than becoming an effect kind because its call
    /// sites do not share one effect-delivery point. Two of them run with no
    /// `OperationContext` at all (guarded by `debug_assert!(operation.is_none())`),
    /// and `after_effect` only invokes the mode when a context is present, so an
    /// effect kind could not cover them. The remaining sites do hold a live
    /// context, but they are reached precisely because a mode gate --
    /// `controls_physical_access` or `observes_physical_access_batch` -- declared
    /// this access unobserved; publishing it on the effect channel would
    /// contradict the gate that just opted out.
    fn after_unobserved_physical_access(
        _state: &Self::LaunchState,
        _global_warp_id: usize,
    ) -> Result<(), EngineError> {
        Ok(())
    }

    /// Begin one checker-visible global-memory transaction over `spans` (the
    /// global bytes the operation touches; an empty list joins no
    /// transaction).
    ///
    /// Exclusive transactions prevent any modeled global access to the same
    /// allocations from interleaving between validation, numerical execution,
    /// and effect commit. Shared transactions may overlap each other, but remain excluded
    /// by exact-read-from accesses and ordering effects. The compact repeated-
    /// poll path instead uses allocation epochs to detect intervening writes
    /// and reprocess stale reads. Lock acquisition is only an implementation
    /// guard and must not become a happens-before edge or diagnostic identity.
    fn begin_global_memory_transaction<'a>(
        _state: &'a Self::LaunchState,
        _exclusive: bool,
        _spans: &[crate::PhysicalByteSpan],
    ) -> Result<Self::GlobalMemoryTransactionGuard<'a>, EngineError> {
        Ok(Self::GlobalMemoryTransactionGuard::default())
    }

    fn before_operation(
        state: &Self::LaunchState,
        operation: &OperationContext,
    ) -> Result<(), EngineError>;

    fn after_operation(
        state: &Self::LaunchState,
        operation: &OperationContext,
    ) -> Result<(), EngineError>;

    /// Inspect a fully resolved effect before numerical/runtime state changes.
    fn before_effect(
        _state: &Self::LaunchState,
        _operation: &OperationContext,
        _effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        Ok(())
    }

    /// Observe a successfully applied effect before the semantic scope closes.
    fn after_effect(
        _state: &Self::LaunchState,
        _operation: &OperationContext,
        _effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        Ok(())
    }

    /// Validate one selected completion before numerical/runtime state changes.
    fn before_completion(
        _state: &Self::LaunchState,
        _effect: CompletionActionEffect<'_>,
    ) -> Result<(), EngineError> {
        Ok(())
    }

    fn after_completion(
        _state: &Self::LaunchState,
        _effect: CompletionEffect<'_>,
    ) -> Result<(), EngineError> {
        Ok(())
    }
    /// Observe one boundary of the mandatory `.sync` rendezvous built into a
    /// collective warp memory instruction. Callers invoke this immediately
    /// before the access and again after the access has completed so prior
    /// lane effects reach the collective and its effects reach later lanes.
    fn warp_collective_rendezvous(
        _state: &Self::LaunchState,
        _operation: &OperationContext,
        _mask: WarpMask,
    ) -> Result<(), EngineError> {
        Ok(())
    }
}

/// Keep global-transaction locking out of NumSim and checkers that do not
/// model global read-from. Their associated guard is zero-sized, so generated
/// specializations carry neither a lock field nor a conditional drop. Expensive
/// participation checks at call sites must still be guarded by
/// [`EngineModeImpl::USES_GLOBAL_MEMORY_TRANSACTION`].
#[inline(always)]
pub(crate) fn begin_global_memory_transaction<'a, M: EngineMode>(
    state: &'a <M as EngineModeImpl>::LaunchState,
    participates: bool,
    exclusive: bool,
    spans: &[crate::PhysicalByteSpan],
) -> Result<<M as EngineModeImpl>::GlobalMemoryTransactionGuard<'a>, EngineError> {
    if M::USES_GLOBAL_MEMORY_TRANSACTION && participates {
        M::begin_global_memory_transaction(state, exclusive, spans)
    } else {
        Ok(<M as EngineModeImpl>::GlobalMemoryTransactionGuard::default())
    }
}

/// Numerical execution with no analysis-only state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NumSimMode;

impl EngineModeImpl for NumSimMode {
    type LaunchState = ();
    type GlobalMemoryTransactionGuard<'a> = ();

    const NAME: &'static str = "numsim";
    const OBSERVES_OPERATIONS: bool = false;

    fn before_operation(
        _state: &Self::LaunchState,
        _operation: &OperationContext,
    ) -> Result<(), EngineError> {
        Ok(())
    }

    fn after_operation(
        _state: &Self::LaunchState,
        _operation: &OperationContext,
    ) -> Result<(), EngineError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{begin_global_memory_transaction, EngineModeImpl, NumSimMode};

    #[test]
    fn numeric_mode_has_no_analysis_launch_state() {
        fn assert_unit(_: <NumSimMode as EngineModeImpl>::LaunchState) {}

        assert_unit(());
        assert_eq!(NumSimMode::NAME, "numsim");
        assert!(!NumSimMode::OBSERVES_OPERATIONS);
        assert_eq!(
            std::mem::size_of::<<NumSimMode as EngineModeImpl>::LaunchState>(),
            0
        );
    }

    #[test]
    fn numeric_mode_does_not_acquire_global_transactions() {
        let guard = begin_global_memory_transaction::<NumSimMode>(&(), true, true, &[])
            .expect("the inactive NumSim transaction should succeed");

        assert_eq!(guard, ());
    }
}
