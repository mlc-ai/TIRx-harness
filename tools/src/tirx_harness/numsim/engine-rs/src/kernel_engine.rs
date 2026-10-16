use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::runtime::tcgen_work::{TcgenLaneSpans, tcgen_batch_from_lane_spans};

use crate::engine_mode::{
    CachedGlobalReadAccess, CachedGlobalReadFinish, GlobalMemoryProgressSnapshot,
    begin_global_memory_transaction,
};
use crate::memory::ReadSource;
use crate::physical_access::ProxyMemoryDomain;
use crate::runtime::sync::plan_internal_warpgroup_sync;
use crate::runtime::tensor_map::{RawTmaReductionOp, RawTmaS2gTransferPlan};
use crate::runtime::DeclaredWordWaitPlan;
use crate::runtime::{
    copy_physical_ptr_bytes, element_byte_offset, execute_async_copy_element_at_lane,
    execute_async_copy_element_batch_in_write_session, execute_raw_bulk_copy_g2s,
    execute_raw_bulk_copy_s2s, initialize_physical_mbarriers, load_scalar_warp,
    load_scalar_warp_at_byte_offsets, load_scalar_warp_with_source,
    plan_async_copy_element_at_lane_accesses,
    plan_cluster_barrier_arrive, plan_cluster_barrier_wait, plan_cp_async_mbarrier_arrive,
    plan_cp_async_physical_ptr_lane_accesses, plan_named_barrier_arrive,
    plan_named_barrier_sync_with_alignment, plan_physical_mbarrier_arrive_lanes,
    plan_physical_mbarrier_expect_tx, plan_physical_mbarrier_init,
    plan_physical_mbarrier_state_wait_lanes, plan_physical_mbarrier_wait_lanes,
    plan_raw_bulk_copy_g2s_accesses, plan_raw_bulk_copy_g2s_ignore_oob_accesses,
    plan_raw_bulk_copy_lane_varying_accesses, plan_raw_bulk_copy_s2g_accesses,
    plan_raw_bulk_copy_s2s_accesses, plan_raw_bulk_reduce_s2g_accesses, plan_raw_ldmatrix_access,
    plan_raw_st_bulk_zero_access, plan_raw_stmatrix_access, plan_tcgen_allocate,
    plan_tcgen_commit_issue, plan_tcgen_deallocate, plan_tcgen_relinquish,
    raw_bulk_copy_g2s_cta_ignore_oob, raw_bulk_copy_g2s_multicast, raw_bulk_copy_s2g,
    raw_bulk_copy_s2g_masked, raw_bulk_reduce_s2g, require_full_warp_sync,
    resolve_mbarrier_completion_targets, resolve_runtime_physical_access,
    resolve_runtime_physical_access_at_cta, resolve_shared_runtime_physical_access_to_cta,
    resolve_tmem_physical_access, resolve_tmem_physical_span, runtime_scalar_byte_offsets,
    store_physical_ptr_u32, store_scalar_warp, supports_shared_runtime_write_session,
    validate_runtime_scalar_warp_access, validate_runtime_scalar_warp_access_at_byte_offsets,
    validate_tmem_scalar_warp_access, wait_physical_mbarrier, with_shared_runtime_write_session,
    write_runtime_bytes, write_shared_runtime_bytes_to_cta, AsyncSourceFill,
    ClusterBarrierArrivalSemantics, ClusterBarrierWaitSemantics, CompactAsyncCopyAccessPlan,
    KernelRuntimeServices, NamedBarrierSyncResumePlan, PhysicalMbarrierCompletionIssuePlan,
    PhysicalMbarrierCompletionTargets, PhysicalPtr, PointerSpace, PtxStateSpace,
    RawBulkCopyFootprintShape, RuntimeBuffer, RuntimeTensorMap, StmatrixDescriptor,
    TcgenAccessFootprintBuilder, TcgenLifecyclePlan, TcgenLifecycleResumePlan,
    TcgenMmaPipelineClass, TcgenPipelineOperation, TcgenWorkIssue, TcgenWorkKind,
};
use crate::spaces::PhysicalSemanticProgressSnapshot;
use crate::{
    AnalysisGapEffect, AnalysisGapKind, AsyncGroupDomain, AsyncGroupIssueBatchEffect,
    AsyncGroupIssueEffect, AsyncPayloadEffect, CompletionActionEffect, CompletionEffect,
    DeferredGlobalReduction, DeferredGlobalWrite, DiagnosticLabel, DynamicOpId, EngineError,
    EngineMode, LoopFrame, MemoryAccessSemantics, MemoryFenceEffect, MemoryOrder, MemoryProxy,
    MemoryScope, OperationContext, OperationEffect, OperationKind, PhysicalAccessBatch,
    PhysicalAccessBatchError, PhysicalAccessDescriptor, PhysicalAccessKind, PhysicalAccessSpace,
    PhysicalAllocationId, PhysicalBarrierId, PhysicalMemory, ProfileKind, ProfileTimer,
    ProxyAsyncFenceEffect, ProxyAsyncFenceScope, RuntimeScalar, SETMAXNREG_WARPS_PER_GROUP,
    SetmaxnregAction, StaticOpId, TcgenAllocation, TcgenFenceKind, TcgenLifecycleHub,
    TcgenTransferKind, TmemAccessMode, WarpContext, WarpMask, WarpSyncEffect, WarpValue,
};

/// Launch-wide engine shared by every warp of one concrete kernel execution.
pub(crate) struct KernelEngine<M: EngineMode> {
    kernel_index: usize,
    physical: PhysicalMemory,
    services: KernelRuntimeServices,
    mode_state: Arc<M::LaunchState>,
}

impl<M: EngineMode> KernelEngine<M> {
    pub(crate) fn new(
        kernel_index: usize,
        physical: PhysicalMemory,
        services: KernelRuntimeServices,
        mode_state: Arc<M::LaunchState>,
    ) -> Self {
        Self {
            kernel_index,
            physical,
            services,
            mode_state,
        }
    }

    pub(crate) const fn mode_name(&self) -> &'static str {
        M::NAME
    }

    pub(crate) const fn kernel_index(&self) -> usize {
        self.kernel_index
    }

    pub(crate) const fn physical(&self) -> &PhysicalMemory {
        &self.physical
    }

    pub(crate) const fn services(&self) -> &KernelRuntimeServices {
        &self.services
    }

    pub(crate) fn mode_state(&self) -> &M::LaunchState {
        self.mode_state.as_ref()
    }
}

impl<M: EngineMode> Clone for KernelEngine<M> {
    fn clone(&self) -> Self {
        Self {
            kernel_index: self.kernel_index,
            physical: self.physical.clone(),
            services: self.services.clone(),
            mode_state: Arc::clone(&self.mode_state),
        }
    }
}

/// Stable identity of the `RuntimeBuffer` views a TCGEN footprint resolves
/// through. The underlying allocation lists are pinned by
/// [`RuntimeBufferPin`] for as long as a memo entry refers to them, so a
/// list address cannot be recycled for a different buffer meanwhile.
#[derive(Clone, Hash, PartialEq, Eq)]
enum RuntimeBufferIdentity {
    Tmem {
        allocations: usize,
        lane_span: usize,
        tcol_span_elements: usize,
        elem_offset: usize,
        itemsize: usize,
    },
    Shared {
        allocations: usize,
        byte_offset: usize,
        byte_len: usize,
        backing_byte_len: usize,
        virtual_base: usize,
    },
    View {
        buffer: Box<RuntimeBufferIdentity>,
        readable_lanes: u32,
        writable_lanes: u32,
    },
}

enum RuntimeBufferPin {
    Tmem(Arc<Vec<crate::TmemAllocation>>),
    Shared(Arc<Vec<crate::SharedAllocation>>),
}

fn runtime_buffer_memo_identity(
    buffer: &RuntimeBuffer,
) -> Option<(RuntimeBufferIdentity, RuntimeBufferPin)> {
    match buffer {
        RuntimeBuffer::Tmem {
            allocations,
            lane_span,
            tcol_span_elements,
            elem_offset,
            itemsize,
        } => Some((
            RuntimeBufferIdentity::Tmem {
                allocations: Arc::as_ptr(allocations) as usize,
                lane_span: *lane_span,
                tcol_span_elements: *tcol_span_elements,
                elem_offset: *elem_offset,
                itemsize: *itemsize,
            },
            RuntimeBufferPin::Tmem(Arc::clone(allocations)),
        )),
        RuntimeBuffer::Shared {
            allocations,
            byte_offset,
            byte_len,
            backing_byte_len,
            virtual_base,
        } => Some((
            RuntimeBufferIdentity::Shared {
                allocations: Arc::as_ptr(allocations) as usize,
                byte_offset: *byte_offset,
                byte_len: *byte_len,
                backing_byte_len: *backing_byte_len,
                virtual_base: *virtual_base,
            },
            RuntimeBufferPin::Shared(Arc::clone(allocations)),
        )),
        RuntimeBuffer::AccessView {
            buffer,
            readable_lanes,
            writable_lanes,
        } => runtime_buffer_memo_identity(buffer).map(|(identity, pin)| {
            (
                RuntimeBufferIdentity::View {
                    buffer: Box::new(identity),
                    readable_lanes: readable_lanes.bits(),
                    writable_lanes: writable_lanes.bits(),
                },
                pin,
            )
        }),
        _ => None,
    }
}

/// What one memoized TCGEN issue resolution depends on: the instruction
/// site and form, the register operands the site decodes its footprints
/// from, the buffers it resolves through, the active mask, and the TMEM
/// lifecycle generation its lifecycle validation ran under.
#[derive(Hash, PartialEq, Eq)]
pub(crate) struct TcgenIssueMemoKey {
    site: u64,
    form: u8,
    operands: [u64; 8],
    buffers: Vec<RuntimeBufferIdentity>,
    mask: u32,
    generation: u64,
}

/// The caller-known half of a [`TcgenIssueMemoKey`], built before the
/// operation context exists.
pub(crate) struct TcgenIssueMemoSeed {
    site: u64,
    form: u8,
    operands: [u64; 8],
    buffers: Vec<RuntimeBufferIdentity>,
    pins: Vec<RuntimeBufferPin>,
}

impl TcgenIssueMemoSeed {
    fn key(&self, mask: u32, generation: u64) -> TcgenIssueMemoKey {
        TcgenIssueMemoKey {
            site: self.site,
            form: self.form,
            operands: self.operands,
            buffers: self.buffers.clone(),
            mask,
            generation,
        }
    }
}

/// Build the memo seed for a TCGEN issue whose footprints are a pure
/// function of `operands` and `buffers`; `None` when a buffer has no stable
/// identity, in which case the issue resolves its footprints every time.
pub(crate) fn tcgen_issue_memo_seed<'a>(
    site: crate::runtime::instructions::SiteId,
    form: u8,
    operands: [u64; 8],
    buffers: impl IntoIterator<Item = &'a RuntimeBuffer>,
) -> Option<TcgenIssueMemoSeed> {
    let mut identities = Vec::new();
    let mut pins = Vec::new();
    for buffer in buffers {
        let (identity, pin) = runtime_buffer_memo_identity(buffer)?;
        identities.push(identity);
        pins.push(pin);
    }
    Some(TcgenIssueMemoSeed {
        site: site.get(),
        form,
        operands,
        buffers: identities,
        pins,
    })
}

/// One resolved access batch of a memoized issue, minus the operation
/// context that changes per issue.
struct TcgenBatchTemplate {
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    logical_buffer: Option<Arc<str>>,
    proxy_domain: ProxyMemoryDomain,
    lane_spans: TcgenLaneSpans,
}

impl TcgenBatchTemplate {
    fn from_batch(batch: &PhysicalAccessBatch) -> Self {
        let descriptor = batch.descriptor();
        let mut lane_spans: TcgenLaneSpans = std::array::from_fn(|_| Vec::new());
        for lane in batch.lanes() {
            lane_spans[lane.provenance().lane()] = lane.footprint().spans().to_vec();
        }
        Self {
            kind: descriptor.kind(),
            space: descriptor.space(),
            logical_buffer: batch.logical_buffer().map(Arc::from),
            proxy_domain: descriptor.proxy_memory_domain(),
            lane_spans,
        }
    }

    fn instantiate(
        &self,
        operation: &OperationContext,
    ) -> Result<PhysicalAccessBatch, EngineError> {
        tcgen_batch_from_lane_spans(
            operation.clone(),
            self.kind,
            self.space,
            self.logical_buffer.as_deref(),
            &self.lane_spans,
        )
        .map(|batch| batch.with_proxy_memory_domain(self.proxy_domain))
    }
}

struct TcgenIssueMemoEntry {
    batches: Vec<TcgenBatchTemplate>,
    shared_a_reads: Vec<TcgenBatchTemplate>,
    _pins: Vec<RuntimeBufferPin>,
}

/// Multiplicative mixing hasher for footprint keys: the accumulator key of
/// one MMA carries ~128 lane tuples, and SipHash over that would cost a
/// noticeable fraction of the resolution it replaces.
#[derive(Default)]
struct TcgenFootprintHasher(u64);

impl std::hash::Hasher for TcgenFootprintHasher {
    fn finish(&self) -> u64 {
        let mut hash = self.0;
        hash ^= hash >> 32;
        hash = hash.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        hash ^ (hash >> 29)
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0_u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(word));
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    fn write_isize(&mut self, value: isize) {
        self.write_u64(value as u64);
    }

    fn write_i64(&mut self, value: i64) {
        self.write_u64(value as u64);
    }

    fn write_u32(&mut self, value: u32) {
        self.write_u64(u64::from(value));
    }

    fn write_u8(&mut self, value: u8) {
        self.write_u64(u64::from(value));
    }
}

/// Per-warp memo of resolved TCGEN issue footprints.
///
/// A K-loop re-issues the same MMA site with the same accumulator window,
/// scale-factor window, and stage descriptors every iteration; decoding and
/// resolving the footprints is a pure function of the memo key, so repeats
/// rebuild the batches from the memoized templates instead.
#[derive(Default)]
struct TcgenIssueMemo {
    entries: HashMap<
        TcgenIssueMemoKey,
        Arc<TcgenIssueMemoEntry>,
        BuildHasherDefault<TcgenFootprintHasher>,
    >,
}

impl TcgenIssueMemo {
    const CAPACITY: usize = 4096;

    fn get(&self, key: &TcgenIssueMemoKey) -> Option<Arc<TcgenIssueMemoEntry>> {
        self.entries.get(key).cloned()
    }

    fn insert(&mut self, key: TcgenIssueMemoKey, entry: TcgenIssueMemoEntry) {
        if self.entries.len() >= Self::CAPACITY {
            self.entries.clear();
        }
        self.entries.insert(key, Arc::new(entry));
    }
}

/// Per-warp entry object pairing native coordinates with the launch engine.
pub struct WarpEngine<M: EngineMode> {
    context: WarpContext,
    kernel: KernelEngine<M>,
    next_operation_sequence: u64,
    loop_frames: Vec<LoopFrame>,
    loop_frames_snapshot: Option<Arc<[LoopFrame]>>,
    native_while_loops: Vec<NativeWhileLoopState>,
    pending_unobserved_physical_operation: Mutex<Option<PendingPhysicalOperation>>,
    local_engine_progress: AtomicU64,
    while_body_observations: Mutex<Vec<WhileBodyObservation>>,
    tcgen_issue_memo: Mutex<TcgenIssueMemo>,
    // Logical collector validity belongs to each MMA-issuing thread, not
    // the allocation or the memoized footprint. A, B0, B1, B2, B3 bits.
    pub(crate) tcgen_collectors: [u8; crate::WARP_SIZE],
}

#[derive(Clone, Copy)]
struct PendingPhysicalOperation {
    context: WarpContext,
    source_op_id: StaticOpId,
    sequence: u64,
    kind: OperationKind,
}

/// Engine-level progress baseline for one generated native while loop.
///
/// The transpiler owns only the control-flow checkpoint placement.  Concrete
/// memory and protocol state decide whether a quantum made progress.
pub struct WhileLoopProgress {
    snapshot: PhysicalSemanticProgressSnapshot,
    live_mask: WarpMask,
    local_engine_progress: u64,
    parked_once: bool,
    memory_only: bool,
    global_memory_progress: GlobalMemoryProgressSnapshot,
}

struct WhileBodyObservation {
    atomic_poll: bool,
    requires_engine_progress: bool,
    exact_global_progress_supported: bool,
    global_memory_progress: GlobalMemoryProgressSnapshot,
}

impl Default for WhileBodyObservation {
    fn default() -> Self {
        Self {
            atomic_poll: false,
            requires_engine_progress: false,
            exact_global_progress_supported: true,
            global_memory_progress: GlobalMemoryProgressSnapshot::default(),
        }
    }
}

struct NativeWhileLoopState {
    site: StaticOpId,
    next_ordinal: u64,
    in_body: bool,
    while_progress: Option<WhileLoopProgress>,
}

impl<M: EngineMode> WarpEngine<M> {
    pub(crate) fn new(context: WarpContext, kernel: KernelEngine<M>) -> Self {
        Self {
            context,
            kernel,
            next_operation_sequence: 0,
            loop_frames: Vec::new(),
            loop_frames_snapshot: None,
            native_while_loops: Vec::new(),
            pending_unobserved_physical_operation: Mutex::new(None),
            local_engine_progress: AtomicU64::new(0),
            while_body_observations: Mutex::new(Vec::new()),
            tcgen_issue_memo: Mutex::new(TcgenIssueMemo::default()),
            tcgen_collectors: [0; crate::WARP_SIZE],
        }
    }

    pub(crate) const fn context(&self) -> WarpContext {
        self.context
    }

    pub(crate) fn discard_mma_collectors(&mut self, lane: usize, weight_stationary: bool) {
        self.tcgen_collectors[lane] &= if weight_stationary { !2_u8 } else { !3_u8 };
    }

    pub(crate) const fn kernel(&self) -> &KernelEngine<M> {
        &self.kernel
    }

    pub(crate) const fn observes_operations(&self) -> bool {
        M::OBSERVES_OPERATIONS
    }

    fn record_engine_progress(&self) {
        if M::OBSERVES_OPERATIONS {
            if let Some(observation) = self
                .while_body_observations
                .lock()
                .expect("native while-loop observations poisoned")
                .last_mut()
            {
                observation.requires_engine_progress = true;
            }
        }
        self.kernel.physical().record_semantic_progress();
    }

    fn record_local_engine_progress(&self) {
        if M::OBSERVES_OPERATIONS {
            if let Some(observation) = self
                .while_body_observations
                .lock()
                .expect("native while-loop observations poisoned")
                .last_mut()
            {
                observation.requires_engine_progress = true;
            }
        }
        self.local_engine_progress
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |generation| {
                generation.checked_add(1)
            })
            .expect("NumSim per-warp semantic-progress generation exhausted");
    }

    pub(crate) fn begin_while_loop_progress(&self, live_mask: WarpMask) -> WhileLoopProgress {
        self.kernel.physical().enable_semantic_progress();
        WhileLoopProgress {
            snapshot: self
                .kernel
                .physical()
                .semantic_progress_snapshot_for_warp(self.context.global_warp_id()),
            live_mask,
            local_engine_progress: self.local_engine_progress.load(Ordering::Relaxed),
            parked_once: false,
            memory_only: false,
            global_memory_progress: GlobalMemoryProgressSnapshot::default(),
        }
    }

    /// Yield one native while quantum according to concrete engine progress.
    ///
    /// Every continuing while quantum re-enters at poll-recheck priority so
    /// ordinary runnable work can advance the condition first. A stuttering
    /// quantum subscribes to launch-wide semantic progress. Executor-wide
    /// quiescence reports deadlock only when every warp is blocked or parked
    /// and completion sources cannot advance.
    pub(crate) async fn while_loop_checkpoint(
        &self,
        progress: &mut WhileLoopProgress,
        live_mask: WarpMask,
    ) -> Result<(), EngineError> {
        let current = self
            .kernel
            .physical()
            .semantic_progress_snapshot_for_warp(self.context.global_warp_id());
        let local_engine_progress = self.local_engine_progress.load(Ordering::Relaxed);
        let exact_global_progress = !progress.global_memory_progress.is_empty();
        let semantic_progress_changed = if exact_global_progress {
            progress.global_memory_progress.changed()
        } else {
            current.changed_since(progress.snapshot, progress.memory_only)
        };
        if semantic_progress_changed
            || live_mask != progress.live_mask
            || local_engine_progress != progress.local_engine_progress
        {
            progress.snapshot = current;
            progress.live_mask = live_mask;
            progress.local_engine_progress = local_engine_progress;
            crate::scheduling::poll_recheck_reschedule().await;
            return Ok(());
        }

        // The watch's waiter table is keyed by an anonymous counter carrying no
        // warp or operation, so this park cannot be reconstructed at report
        // time. Record it here, where both are still in hand.
        // No `DynamicOpId` is in scope here — the checkpoint is driven by the
        // loop scaffold, not by an operation — so the record names the warp and
        // the reason but not a source site. Synccheck's `stalled_operations`
        // still supplies the last reported operation for these warps.
        {
            progress.parked_once = true;
            let ordering = self.kernel.services().ordering();
            let _park = ordering.park(
                self.context.global_warp_id(),
                crate::ordering::ParkReason::SemanticProgress,
                None,
            );
            if exact_global_progress {
                progress.global_memory_progress.watch().await;
            } else {
                self.kernel
                    .physical()
                    .watch_semantic_progress(current, progress.memory_only)
                    .await;
            }
        }
        progress.snapshot = self
            .kernel
            .physical()
            .semantic_progress_snapshot_for_warp(self.context.global_warp_id());
        progress.live_mask = live_mask;
        progress.local_engine_progress = self.local_engine_progress.load(Ordering::Relaxed);
        Ok(())
    }

    fn native_while_state_error(
        action: &str,
        expected_site: StaticOpId,
        actual: Option<&NativeWhileLoopState>,
    ) -> EngineError {
        match actual {
            Some(actual) => EngineError::message(format!(
                "native while-loop {action} mismatch: expected site {expected_site}, found site {} ({})",
                actual.site,
                if actual.in_body {
                    "inside body"
                } else {
                    "between iterations"
                },
            )),
            None => EngineError::message(format!(
                "native while-loop {action} without matching site {expected_site}"
            )),
        }
    }

    fn native_while_body_enter(
        &mut self,
        site: u64,
        live_mask: WarpMask,
    ) -> Result<(), EngineError> {
        let site_id = StaticOpId::new(site);
        if live_mask.is_empty() {
            if let Some(state) = self.native_while_loops.last() {
                if !state.in_body {
                    if state.site != site_id {
                        return Err(Self::native_while_state_error(
                            "close",
                            site_id,
                            Some(state),
                        ));
                    }
                    self.native_while_loops.pop();
                }
            }
            return Ok(());
        }

        let continuing = self
            .native_while_loops
            .last()
            .is_some_and(|state| !state.in_body);
        if continuing {
            let state = self.native_while_loops.last().expect("checked above");
            if state.site != site_id {
                return Err(Self::native_while_state_error(
                    "enter",
                    site_id,
                    Some(state),
                ));
            }
        } else {
            let while_progress = self.begin_while_loop_progress(live_mask);
            self.native_while_loops.push(NativeWhileLoopState {
                site: site_id,
                next_ordinal: 0,
                in_body: false,
                while_progress: Some(while_progress),
            });
        }

        let budget = self
            .kernel
            .services()
            .execution_policy()
            .native_loop_iteration_budget();
        let state = self
            .native_while_loops
            .last_mut()
            .expect("while-loop state was created");
        let ordinal = usize::try_from(state.next_ordinal).ok();
        if ordinal.is_none_or(|ordinal| ordinal >= budget) {
            return Err(EngineError::native_loop_iteration_limit(
                format!("site {site}"),
                budget,
            ));
        }
        state.in_body = true;
        let ordinal = state.next_ordinal;
        self.loop_frames.push(LoopFrame::new(site_id, ordinal));
        if M::OBSERVES_OPERATIONS {
            self.while_body_observations
                .lock()
                .expect("native while-loop observations poisoned")
                .push(WhileBodyObservation::default());
        }
        self.loop_frames_snapshot = None;
        Ok(())
    }

    fn native_while_body_exit(
        &mut self,
        site: u64,
        next_live: WarpMask,
    ) -> Result<(bool, bool), EngineError> {
        let site_id = StaticOpId::new(site);
        let Some(state) = self.native_while_loops.last() else {
            return Err(Self::native_while_state_error("exit", site_id, None));
        };
        if state.site != site_id || !state.in_body {
            return Err(Self::native_while_state_error("exit", site_id, Some(state)));
        }

        let Some(frame) = self.loop_frames.last() else {
            return Err(EngineError::message(
                "native while-loop body has no dynamic operation frame",
            ));
        };
        if frame.loop_site_id() != site_id {
            return Err(EngineError::message(format!(
                "native while-loop frame mismatch: expected {site_id}, found {}",
                frame.loop_site_id(),
            )));
        }
        self.loop_frames.pop();
        self.loop_frames_snapshot = None;

        let state = self.native_while_loops.last_mut().expect("validated above");
        state.in_body = false;
        if M::OBSERVES_OPERATIONS {
            let observation = self
                .while_body_observations
                .lock()
                .expect("native while-loop observations poisoned")
                .pop()
                .expect("native while-loop body observation matches its frame");
            if let Some(parent) = self
                .while_body_observations
                .lock()
                .expect("native while-loop observations poisoned")
                .last_mut()
            {
                // A nested loop can depend on protocol state that the outer body
                // does not observe directly, so the outer wait stays conservative.
                parent.requires_engine_progress = true;
            }
            let progress = state
                .while_progress
                .as_mut()
                .expect("native while loop retains progress between body visits");
            progress.memory_only = observation.atomic_poll && !observation.requires_engine_progress;
            progress.global_memory_progress = if progress.memory_only
                && observation.exact_global_progress_supported
                && !observation.global_memory_progress.is_empty()
            {
                observation.global_memory_progress
            } else {
                GlobalMemoryProgressSnapshot::default()
            };
        }
        state.next_ordinal = state
            .next_ordinal
            .checked_add(1)
            .ok_or_else(|| EngineError::message("native while-loop iteration ordinal overflow"))?;

        let quantum = self
            .kernel
            .services()
            .execution_policy()
            .native_loop_reschedule_quantum();
        let should_yield = usize::try_from(state.next_ordinal)
            .map(|ordinal| ordinal % quantum == 0)
            .unwrap_or(true);

        if next_live.is_empty() {
            self.native_while_loops.pop();
            return Ok((false, should_yield));
        }
        Ok((true, should_yield))
    }

    pub(crate) fn native_for_enter(
        &mut self,
        site: u64,
        iteration_ordinal: i64,
    ) -> Result<(), EngineError> {
        let ordinal = usize::try_from(iteration_ordinal)
            .map_err(|_| EngineError::message("negative for-loop iteration ordinal"))?;
        let budget = self
            .kernel
            .services()
            .execution_policy()
            .native_loop_iteration_budget();
        if ordinal >= budget {
            return Err(EngineError::native_loop_iteration_limit(
                format!("site {site}"),
                budget,
            ));
        }
        self.push_loop_frame(site, iteration_ordinal)
    }

    pub(crate) async fn native_for_exit(
        &mut self,
        site: u64,
        next_iteration_ordinal: i64,
        next_live: WarpMask,
    ) -> Result<(), EngineError> {
        let expected_site = StaticOpId::new(site);
        let frame = self.loop_frames.last().copied().ok_or_else(|| {
            EngineError::message("native for-loop body has no dynamic operation frame")
        })?;
        if frame.loop_site_id() != expected_site {
            return Err(EngineError::message(format!(
                "native for-loop frame mismatch: expected {expected_site}, found {}",
                frame.loop_site_id(),
            )));
        }
        let expected_next = frame
            .iteration_ordinal()
            .checked_add(1)
            .ok_or_else(|| EngineError::message("for-loop iteration ordinal overflow"))?;
        let next = u64::try_from(next_iteration_ordinal)
            .map_err(|_| EngineError::message("negative next for-loop iteration ordinal"))?;
        if next != expected_next {
            return Err(EngineError::message(format!(
                "native for-loop next ordinal mismatch: expected {expected_next}, found {next}"
            )));
        }
        self.pop_loop_frame(site)?;
        let quantum = self
            .kernel
            .services()
            .execution_policy()
            .native_loop_reschedule_quantum();
        let should_yield = usize::try_from(next)
            .map(|ordinal| ordinal % quantum == 0)
            .unwrap_or(true);
        if !next_live.is_empty() && should_yield {
            crate::scheduling::reschedule().await;
        }
        Ok(())
    }

    pub(crate) async fn native_while_enter(
        &mut self,
        site: u64,
        live_mask: WarpMask,
    ) -> Result<(), EngineError> {
        let site_id = StaticOpId::new(site);
        let should_checkpoint = if live_mask.is_empty() || !M::OBSERVES_OPERATIONS {
            false
        } else {
            match self.native_while_loops.last() {
                Some(state) if !state.in_body => {
                    if state.site != site_id {
                        return Err(Self::native_while_state_error(
                            "enter",
                            site_id,
                            Some(state),
                        ));
                    }
                    if state
                        .while_progress
                        .as_ref()
                        .is_some_and(|progress| progress.parked_once)
                    {
                        true
                    } else {
                        let quantum = self
                            .kernel
                            .services()
                            .execution_policy()
                            .native_loop_reschedule_quantum();
                        usize::try_from(state.next_ordinal)
                            .map(|ordinal| ordinal % quantum == 0)
                            .unwrap_or(true)
                    }
                }
                _ => false,
            }
        };

        if should_checkpoint {
            let mut progress = self
                .native_while_loops
                .last_mut()
                .and_then(|state| state.while_progress.take())
                .ok_or_else(|| {
                    EngineError::message(format!(
                        "native While loop site {site} lost semantic-progress state"
                    ))
                })?;
            self.while_loop_checkpoint(&mut progress, live_mask).await?;
            let state = self.native_while_loops.last_mut();
            match state {
                Some(state) if state.site == site_id && !state.in_body => {
                    state.while_progress = Some(progress);
                }
                state => {
                    return Err(Self::native_while_state_error(
                        "resume",
                        site_id,
                        state.as_deref(),
                    ));
                }
            }
        }
        self.native_while_body_enter(site, live_mask)
    }

    pub(crate) async fn native_while_exit(
        &mut self,
        site: u64,
        next_live: WarpMask,
    ) -> Result<(), EngineError> {
        let (continuing, should_yield) = self.native_while_body_exit(site, next_live)?;
        // Every continuing while quantum re-enters at poll-recheck priority.
        // Numeric execution yields before the next condition recheck. Checker
        // execution checkpoints after reloading the condition, so `while_enter`
        // owns that checkpoint and the condition remains outside the body
        // occurrence frame.
        if continuing && should_yield && !M::OBSERVES_OPERATIONS {
            crate::scheduling::poll_recheck_reschedule().await;
        }
        Ok(())
    }

    pub(crate) fn push_loop_frame(
        &mut self,
        loop_site_id: u64,
        iteration_ordinal: i64,
    ) -> Result<(), EngineError> {
        let loop_site_id = StaticOpId::new(loop_site_id);
        let iteration_ordinal = u64::try_from(iteration_ordinal)
            .map_err(|_| EngineError::message("negative loop iteration ordinal"))?;
        self.loop_frames
            .push(LoopFrame::new(loop_site_id, iteration_ordinal));
        self.loop_frames_snapshot = None;
        Ok(())
    }

    pub(crate) fn pop_loop_frame(&mut self, expected_loop_site_id: u64) -> Result<(), EngineError> {
        let expected_loop_site_id = StaticOpId::new(expected_loop_site_id);
        let Some(frame) = self.loop_frames.pop() else {
            return Err(EngineError::message("generated loop frame stack underflow"));
        };
        if frame.loop_site_id() != expected_loop_site_id {
            return Err(EngineError::message(format!(
                "generated loop frame stack mismatch: expected {}, found {}",
                expected_loop_site_id,
                frame.loop_site_id(),
            )));
        }
        self.loop_frames_snapshot = None;
        Ok(())
    }

    fn next_operation_context_shared(
        &mut self,
        context: WarpContext,
        source_op_id: StaticOpId,
        kind: OperationKind,
        loop_frames: Arc<[LoopFrame]>,
    ) -> Result<OperationContext, EngineError> {
        if context.global_warp_id() != self.context.global_warp_id() {
            return Err(EngineError::message(format!(
                "operation context warp {} does not match engine warp {}",
                context.global_warp_id(),
                self.context.global_warp_id(),
            )));
        }
        let sequence = self.take_operation_sequence()?;
        Ok(OperationContext::new(
            DynamicOpId::new_shared(
                self.kernel.kernel_index(),
                context.global_warp_id(),
                sequence,
                source_op_id,
                loop_frames,
            ),
            kind,
            context.active_mask(),
        )
        .with_control_provenance(context.control_provenance()))
    }

    pub(crate) fn next_operation_context(
        &mut self,
        context: WarpContext,
        source_op_id: StaticOpId,
        kind: OperationKind,
        loop_frames: impl Into<Box<[LoopFrame]>>,
    ) -> Result<OperationContext, EngineError> {
        self.next_operation_context_shared(
            context,
            source_op_id,
            kind,
            Arc::from(loop_frames.into()),
        )
    }

    fn take_operation_sequence(&mut self) -> Result<u64, EngineError> {
        let sequence = self.next_operation_sequence;
        self.next_operation_sequence = self
            .next_operation_sequence
            .checked_add(1)
            .ok_or_else(|| EngineError::message("per-warp operation sequence overflow"))?;
        Ok(sequence)
    }

    fn operation_context_for_consumed_sequence(
        &mut self,
        context: WarpContext,
        source_op_id: StaticOpId,
        kind: OperationKind,
        sequence: u64,
    ) -> Result<OperationContext, EngineError> {
        let _profile_timer = ProfileTimer::new(ProfileKind::OperationBegin);
        if context.global_warp_id() != self.context.global_warp_id() {
            return Err(EngineError::message(format!(
                "operation context warp {} does not match engine warp {}",
                context.global_warp_id(),
                self.context.global_warp_id(),
            )));
        }
        let loop_frames = match self.loop_frames_snapshot.as_ref() {
            Some(snapshot) => Arc::clone(snapshot),
            None => {
                let snapshot = Arc::<[LoopFrame]>::from(self.loop_frames.as_slice());
                self.loop_frames_snapshot = Some(Arc::clone(&snapshot));
                snapshot
            }
        };
        Ok(OperationContext::new(
            DynamicOpId::new_shared(
                self.kernel.kernel_index(),
                context.global_warp_id(),
                sequence,
                source_op_id,
                loop_frames,
            ),
            kind,
            context.active_mask(),
        )
        .with_control_provenance(context.control_provenance()))
    }

    pub(crate) fn begin_operation(
        &mut self,
        context: WarpContext,
        source_op_id: StaticOpId,
        kind: OperationKind,
        loop_frames: impl Into<Box<[LoopFrame]>>,
    ) -> Result<OperationContext, EngineError> {
        let operation = self.next_operation_context(context, source_op_id, kind, loop_frames)?;
        if M::OBSERVES_OPERATIONS {
            M::before_operation(self.kernel.mode_state(), &operation)?;
        }
        Ok(operation)
    }

    pub(crate) fn begin_current_operation(
        &mut self,
        context: WarpContext,
        source_op_id: StaticOpId,
        kind: OperationKind,
    ) -> Result<OperationContext, EngineError> {
        let _profile_timer = ProfileTimer::new(ProfileKind::OperationBegin);
        let loop_frames = match self.loop_frames_snapshot.as_ref() {
            Some(snapshot) => Arc::clone(snapshot),
            None => {
                let snapshot = Arc::<[LoopFrame]>::from(self.loop_frames.as_slice());
                self.loop_frames_snapshot = Some(Arc::clone(&snapshot));
                snapshot
            }
        };
        let operation =
            self.next_operation_context_shared(context, source_op_id, kind, loop_frames)?;
        if M::OBSERVES_OPERATIONS {
            M::before_operation(self.kernel.mode_state(), &operation)?;
        }
        Ok(operation)
    }

    pub(crate) fn begin_optional_operation(
        &mut self,
        context: WarpContext,
        source_op_id: u64,
        kind: OperationKind,
        required: bool,
    ) -> Result<Option<OperationContext>, EngineError> {
        if !required && !M::OBSERVES_OPERATIONS {
            return Ok(None);
        }
        self.begin_current_operation(context, StaticOpId::new(source_op_id), kind)
            .map(Some)
    }

    /// Reuse an epoch-stable single-lane global-read event without allocating
    /// another dynamic operation identity for every spin-loop poll.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_run_cached_single_lane_global_load<T: RuntimeScalar>(
        &mut self,
        numeric_context: &WarpContext,
        context: WarpContext,
        source_op_id: u64,
        byte_width: usize,
        logical_buffer: &str,
        buffer: &RuntimeBuffer,
        indices: &WarpValue<i64>,
        mask: WarpMask,
    ) -> Option<Result<WarpValue<T>, EngineError>> {
        if !M::USES_CACHED_GLOBAL_READ_FAST_PATH || byte_width != T::BYTE_LEN || mask.len() != 1 {
            return None;
        }
        let allocation = runtime_global_allocation(buffer)?;
        if !M::controls_physical_access_allocation(
            self.kernel.mode_state(),
            OperationKind::Load,
            PhysicalAccessSpace::Global,
            Some(allocation),
        ) {
            return None;
        }
        let descriptor = PhysicalAccessDescriptor::new(
            PhysicalAccessKind::Read,
            PhysicalAccessSpace::Global,
            byte_width,
        )
        .expect("runtime scalar byte width is nonzero");
        if !M::compacts_direct_physical_access(self.kernel.mode_state(), descriptor, mask, false) {
            return None;
        }
        let byte_offsets = match runtime_scalar_byte_offsets(buffer, indices, byte_width, mask) {
            Ok(byte_offsets) => byte_offsets,
            Err(_) => return None,
        };
        let lane = mask
            .first_active()
            .expect("single-lane cached global read has an active lane");
        let resolved = match resolve_runtime_physical_access(
            numeric_context,
            buffer,
            lane,
            byte_offsets[lane],
            byte_width,
            PhysicalAccessKind::Read,
        ) {
            Ok(resolved) if resolved.space() == PhysicalAccessSpace::Global => resolved,
            Ok(_) | Err(_) => return None,
        };
        let sequence = match self.take_operation_sequence() {
            Ok(sequence) => sequence,
            Err(error) => return Some(Err(error)),
        };
        let operation_context = context.with_active_mask(mask);
        let access = CachedGlobalReadAccess::new(
            operation_context.global_warp_id(),
            sequence,
            source_op_id,
            descriptor,
            lane,
            resolved.span(),
        );
        let cache_hit = match M::begin_cached_global_read(self.kernel.mode_state(), access) {
            Ok(cache_hit) => cache_hit,
            Err(error) => return Some(Err(error)),
        };
        if !cache_hit {
            let operation = match self.operation_context_for_consumed_sequence(
                operation_context,
                StaticOpId::new(source_op_id),
                OperationKind::Load,
                sequence,
            ) {
                Ok(operation) => operation,
                Err(error) => return Some(Err(error)),
            };
            if let Err(error) = M::before_operation(self.kernel.mode_state(), &operation) {
                return Some(Err(error));
            }
            let values = match self.analysis_named_scalar_load::<T>(
                Some(&operation),
                self.kernel.physical(),
                numeric_context,
                byte_width,
                logical_buffer,
                buffer,
                indices,
                mask,
                false,
            ) {
                Ok(values) => values,
                Err(error) => return Some(Err(error)),
            };
            return Some(M::after_operation(self.kernel.mode_state(), &operation).map(|()| values));
        }

        let values = match load_scalar_warp_at_byte_offsets::<T>(
            self.kernel.physical(),
            numeric_context,
            buffer,
            &byte_offsets,
            mask,
            None,
        ) {
            Ok(values) => values,
            Err(error) => {
                let contextual = self
                    .operation_context_for_consumed_sequence(
                        operation_context,
                        StaticOpId::new(source_op_id),
                        OperationKind::Load,
                        sequence,
                    )
                    .map(|operation| {
                        self.physical_access_error_context(
                            error,
                            Some(&operation),
                            Some(logical_buffer),
                        )
                    })
                    .unwrap_or_else(|context_error| context_error);
                return Some(Err(contextual));
            }
        };
        match M::finish_cached_global_read(self.kernel.mode_state(), access) {
            Ok(CachedGlobalReadFinish::Stable) => Some(Ok(values)),
            Ok(CachedGlobalReadFinish::Reprocess) => {
                let operation = match self.operation_context_for_consumed_sequence(
                    operation_context,
                    StaticOpId::new(source_op_id),
                    OperationKind::Load,
                    sequence,
                ) {
                    Ok(operation) => operation,
                    Err(error) => return Some(Err(error)),
                };
                if let Err(error) = M::before_operation(self.kernel.mode_state(), &operation) {
                    return Some(Err(error));
                }
                let mut lane_spans = [None; crate::WARP_SIZE];
                lane_spans[lane] = Some(access.span());
                let batch = crate::physical_access::CompactPhysicalAccessBatch::new(
                    &operation,
                    descriptor,
                    Some(logical_buffer),
                    false,
                    &lane_spans,
                );
                let result = (|| {
                    let _profile =
                        ProfileTimer::new(ProfileKind::RaceGlobalTransactionReadReprocess);
                    let _transaction = begin_global_memory_transaction::<M>(
                        self.kernel.mode_state(),
                        true,
                        true,
                        &[access.span()],
                    )?;
                    M::before_compact_physical_access(self.kernel.mode_state(), &batch)?;
                    M::after_compact_physical_access(self.kernel.mode_state(), &batch)?;
                    M::after_operation(self.kernel.mode_state(), &operation)
                })();
                Some(result.map(|()| values))
            }
            Err(error) => Some(Err(error)),
        }
    }

    /// Open an operation only when some mode needs it, and record the gap when
    /// the mode observes that gap kind.
    ///
    /// A mode that observes neither operations nor this gap gets no operation
    /// at all, which is the contract `record_analysis_gap`'s debug assert
    /// relies on.
    pub(crate) fn begin_optional_analysis_gap_internal(
        &mut self,
        context: WarpContext,
        source_op_id: u64,
        operation_kind: OperationKind,
        gap_kind: AnalysisGapKind,
        cta_group: Option<u32>,
    ) -> Result<Option<OperationContext>, EngineError> {
        let observes_gap = M::observes_analysis_gap(self.kernel.mode_state(), gap_kind);
        if !observes_gap && !M::OBSERVES_OPERATIONS {
            self.take_operation_sequence()?;
            return Ok(None);
        }
        let operation =
            self.begin_current_operation(context, StaticOpId::new(source_op_id), operation_kind)?;
        if observes_gap {
            self.record_analysis_gap(Some(&operation), gap_kind, cta_group)?;
        }
        Ok(Some(operation))
    }

    pub(crate) fn begin_optional_pointer_operation(
        &mut self,
        context: WarpContext,
        source_op_id: u64,
        kind: OperationKind,
        pointer: &PhysicalPtr,
    ) -> Result<Option<OperationContext>, EngineError> {
        // Integer/null lanes have no allocation. Their placeholder buffer
        // cannot decide physical-space elision; retain an observed operation
        // so dereference validation can report the actual address error.
        if !(pointer.integer_lanes() & context.active_mask()).is_empty() {
            self.begin_optional_operation(context, source_op_id, kind, false)
        } else {
            self.begin_optional_physical_operation(context, source_op_id, kind, pointer.buffer())
        }
    }

    pub(crate) fn begin_optional_physical_operation(
        &mut self,
        context: WarpContext,
        source_op_id: u64,
        kind: OperationKind,
        buffer: &RuntimeBuffer,
    ) -> Result<Option<OperationContext>, EngineError> {
        if M::elides_physical_access_resolution_for_buffer(self.kernel.mode_state(), kind, buffer) {
            let pending = PendingPhysicalOperation {
                context,
                source_op_id: StaticOpId::new(source_op_id),
                sequence: self.take_operation_sequence()?,
                kind,
            };
            if self
                .pending_unobserved_physical_operation
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .replace(pending)
                .is_some()
            {
                return Err(EngineError::message(
                    "generated physical operations overlap before completion",
                ));
            }
            return Ok(None);
        }
        let Some(space) = runtime_buffer_physical_space_for_mask(buffer, context.active_mask())?
        else {
            self.take_operation_sequence()?;
            return Ok(None);
        };
        if !M::controls_physical_access_allocation(
            self.kernel.mode_state(),
            kind,
            space,
            runtime_global_allocation(buffer),
        ) {
            let pending = PendingPhysicalOperation {
                context,
                source_op_id: StaticOpId::new(source_op_id),
                sequence: self.take_operation_sequence()?,
                kind,
            };
            if self
                .pending_unobserved_physical_operation
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .replace(pending)
                .is_some()
            {
                return Err(EngineError::message(
                    "generated physical operations overlap before completion",
                ));
            }
            return Ok(None);
        }
        self.begin_current_operation(context, StaticOpId::new(source_op_id), kind)
            .map(Some)
    }

    /// Execute an ordinary load/store whose analysis mode intentionally
    /// elides physical-access resolution. This combines the generated
    /// begin/access/finish sequence so the hot compact-analysis path does not
    /// materialize a pending operation solely for lazy numeric-error context.
    #[inline(always)]
    pub(crate) fn try_run_elided_named_physical_operation<R>(
        &mut self,
        context: WarpContext,
        source_op_id: u64,
        kind: OperationKind,
        logical_buffer: &str,
        buffer: &RuntimeBuffer,
        numeric_effect: impl FnOnce(&PhysicalMemory) -> Result<R, EngineError>,
    ) -> Option<Result<R, EngineError>> {
        if !M::OBSERVES_OPERATIONS {
            return Some(numeric_effect(self.kernel.physical()).map_err(|error| {
                physical_access_error_context(error, None, Some(logical_buffer))
            }));
        }
        if !M::elides_physical_access_resolution_for_buffer(self.kernel.mode_state(), kind, buffer)
        {
            return None;
        }
        let sequence = match self.take_operation_sequence() {
            Ok(sequence) => sequence,
            Err(error) => return Some(Err(error)),
        };
        let result = numeric_effect(self.kernel.physical()).map_err(|error| {
            let operation = OperationContext::new(
                DynamicOpId::new(
                    self.kernel.kernel_index(),
                    context.global_warp_id(),
                    sequence,
                    StaticOpId::new(source_op_id),
                    self.loop_frames.clone().into_boxed_slice(),
                ),
                kind,
                context.active_mask(),
            )
            .with_control_provenance(context.control_provenance());
            physical_access_error_context(error, Some(&operation), Some(logical_buffer))
        });
        Some(result.and_then(|value| {
            M::after_unobserved_physical_access(
                self.kernel.mode_state(),
                context.global_warp_id(),
            )?;
            Ok(value)
        }))
    }

    pub(crate) fn finish_operation(&self, operation: &OperationContext) -> Result<(), EngineError> {
        if M::OBSERVES_OPERATIONS {
            M::after_operation(self.kernel.mode_state(), operation)?;
        }
        Ok(())
    }

    pub(crate) fn finish_optional_operation(
        &mut self,
        operation: &Option<OperationContext>,
    ) -> Result<(), EngineError> {
        let pending = self
            .pending_unobserved_physical_operation
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if pending.is_some() && operation.is_some() {
            return Err(EngineError::message(
                "observed and unobserved physical operations overlap",
            ));
        }
        if let Some(operation) = operation {
            self.finish_operation(operation)?;
        }
        Ok(())
    }

    /// Execute one scalar load from a statically bound frontend buffer.
    ///
    /// The ABI supplies the exact PTX specialization plus runtime indices;
    /// mode-specific access observation and the numeric effect stay together
    /// in the engine. NumSim therefore takes the same compact path as the
    /// pre-v2 generated code, while checkers retain exact operation identity.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(crate) fn execute_named_scalar_load<T: RuntimeScalar>(
        &mut self,
        context: WarpContext,
        source_op_id: u64,
        logical_buffer: &str,
        buffer: &RuntimeBuffer,
        indices: &WarpValue<i64>,
        mask: WarpMask,
    ) -> Result<WarpValue<T>, EngineError> {
        let read_source = ReadSource {
            kernel_index: self.kernel.kernel_index(),
            source_op_id,
            global_warp_id: context.global_warp_id(),
        };
        if !M::OBSERVES_OPERATIONS {
            return load_scalar_warp_with_source::<T>(
                self.kernel.physical(),
                &context,
                buffer,
                indices,
                mask,
                read_source,
            );
        }
        if let Some(result) = self.try_run_cached_single_lane_global_load::<T>(
            &context,
            context.with_active_mask(mask),
            source_op_id,
            T::BYTE_LEN,
            logical_buffer,
            buffer,
            indices,
            mask,
        ) {
            return result;
        }
        if let Some(result) = self.try_run_elided_named_physical_operation(
            context,
            source_op_id,
            OperationKind::Load,
            logical_buffer,
            buffer,
            |physical| {
                load_scalar_warp_with_source::<T>(
                    physical,
                    &context,
                    buffer,
                    indices,
                    mask,
                    read_source,
                )
            },
        ) {
            return result;
        }
        let operation = self.begin_optional_physical_operation(
            context,
            source_op_id,
            OperationKind::Load,
            buffer,
        )?;
        let result = self.load_named_scalar::<T>(
            operation.as_ref(),
            context,
            T::BYTE_LEN,
            logical_buffer,
            buffer,
            indices,
            mask,
            Some(read_source),
        )?;
        self.finish_optional_operation(&operation)?;
        Ok(result)
    }

    /// Store one scalar register value to a statically bound frontend buffer.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(crate) fn execute_named_scalar_store<T: RuntimeScalar>(
        &mut self,
        context: WarpContext,
        source_op_id: u64,
        logical_buffer: &str,
        buffer: &RuntimeBuffer,
        indices: &WarpValue<i64>,
        values: &WarpValue<T>,
        mask: WarpMask,
    ) -> Result<(), EngineError> {
        if !M::OBSERVES_OPERATIONS {
            return store_scalar_warp::<T>(
                self.kernel.physical(),
                &context,
                buffer,
                indices,
                values,
                mask,
            );
        }
        if let Some(result) = self.try_run_elided_named_physical_operation(
            context,
            source_op_id,
            OperationKind::Store,
            logical_buffer,
            buffer,
            |physical| store_scalar_warp::<T>(physical, &context, buffer, indices, values, mask),
        ) {
            return result;
        }
        let operation = self.begin_optional_physical_operation(
            context,
            source_op_id,
            OperationKind::Store,
            buffer,
        )?;
        self.store_named_scalar::<T>(
            operation.as_ref(),
            context,
            T::BYTE_LEN,
            logical_buffer,
            buffer,
            indices,
            values,
            mask,
        )?;
        self.finish_optional_operation(&operation)
    }

    fn physical_access_error_context(
        &self,
        error: EngineError,
        operation: Option<&OperationContext>,
        logical_buffer: Option<&str>,
    ) -> EngineError {
        if operation.is_some() {
            return physical_access_error_context(error, operation, logical_buffer);
        }
        let pending = *self
            .pending_unobserved_physical_operation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let fallback = pending.map(|pending| {
            OperationContext::new(
                DynamicOpId::new(
                    self.kernel.kernel_index(),
                    pending.context.global_warp_id(),
                    pending.sequence,
                    pending.source_op_id,
                    self.loop_frames.clone().into_boxed_slice(),
                ),
                pending.kind,
                pending.context.active_mask(),
            )
            .with_control_provenance(pending.context.control_provenance())
        });
        physical_access_error_context(error, fallback.as_ref(), logical_buffer)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_named_scalar<T: RuntimeScalar>(
        &self,
        operation: Option<&OperationContext>,
        context: WarpContext,
        byte_width: usize,
        logical_buffer: &str,
        buffer: &RuntimeBuffer,
        indices: &WarpValue<i64>,
        mask: WarpMask,
        source: Option<ReadSource>,
    ) -> Result<WarpValue<T>, EngineError> {
        let physical = self.kernel.physical();
        self.runtime_named_physical_access(
            operation,
            OperationKind::Load,
            byte_width,
            byte_width,
            logical_buffer,
            buffer,
            indices,
            mask,
            false,
            &mut || match source {
                Some(source) => load_scalar_warp_with_source::<T>(
                    physical, &context, buffer, indices, mask, source,
                ),
                None => load_scalar_warp::<T>(physical, &context, buffer, indices, mask),
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn store_named_scalar<T: RuntimeScalar>(
        &self,
        operation: Option<&OperationContext>,
        numeric_context: WarpContext,
        byte_width: usize,
        logical_buffer: &str,
        buffer: &RuntimeBuffer,
        indices: &WarpValue<i64>,
        values: &WarpValue<T>,
        mask: WarpMask,
    ) -> Result<(), EngineError> {
        let physical = self.kernel.physical();
        self.runtime_named_physical_access(
            operation,
            OperationKind::Store,
            byte_width,
            byte_width,
            logical_buffer,
            buffer,
            indices,
            mask,
            false,
            &mut || {
                store_scalar_warp::<T>(physical, &numeric_context, buffer, indices, values, mask)
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn analysis_named_scalar_load<T: RuntimeScalar>(
        &self,
        operation: Option<&OperationContext>,
        physical: &PhysicalMemory,
        context: &WarpContext,
        byte_width: usize,
        logical_buffer: &str,
        buffer: &RuntimeBuffer,
        indices: &WarpValue<i64>,
        mask: WarpMask,
        validation_only: bool,
    ) -> Result<WarpValue<T>, EngineError> {
        if byte_width != T::BYTE_LEN {
            return Err(EngineError::message(format!(
                "analysis scalar load width {byte_width} does not match runtime scalar width {}",
                T::BYTE_LEN,
            )));
        }
        let byte_offsets =
            runtime_scalar_byte_offsets(buffer, indices, byte_width, mask).map_err(|error| {
                self.physical_access_error_context(error, operation, Some(logical_buffer))
            })?;
        self.runtime_named_physical_access_with_offsets(
            operation,
            OperationKind::Load,
            byte_width,
            byte_width,
            logical_buffer,
            buffer,
            indices,
            Some(&byte_offsets),
            mask,
            validation_only,
            &mut || {
                load_scalar_warp_at_byte_offsets::<T>(
                    physical,
                    context,
                    buffer,
                    &byte_offsets,
                    mask,
                    None,
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Record a path-sensitive analysis coverage gap without changing the
    /// numerical effect implemented by the caller.
    fn record_analysis_gap(
        &self,
        operation: Option<&OperationContext>,
        kind: AnalysisGapKind,
        cta_group: Option<u32>,
    ) -> Result<(), EngineError> {
        if !M::observes_analysis_gap(self.kernel.mode_state(), kind) {
            debug_assert!(operation.is_none());
            return Ok(());
        }
        let operation = operation.ok_or_else(|| {
            EngineError::message(format!(
                "{} analysis gap is missing its operation context",
                kind.name()
            ))
        })?;
        let effect = AnalysisGapEffect::new(
            kind,
            self.kernel.kernel_index(),
            self.context.global_cta_id(),
            cta_group,
        );
        self.before_effect(Some(operation), OperationEffect::AnalysisGap(effect))?;
        self.after_effect(Some(operation), OperationEffect::AnalysisGap(effect))
    }

    pub(crate) fn proxy_async_fence(
        &self,
        operation: Option<&OperationContext>,
        scope: &str,
    ) -> Result<(), EngineError> {
        let operation = self.checked_effect_operation(operation, OperationKind::Fence)?;
        let Some(operation) = operation else {
            return Ok(());
        };
        let scope = match scope {
            "" => ProxyAsyncFenceScope::All,
            "shared::cta" => ProxyAsyncFenceScope::SharedCta,
            "shared::cluster" => ProxyAsyncFenceScope::SharedCluster,
            "global" => ProxyAsyncFenceScope::Global,
            "alias" => ProxyAsyncFenceScope::Alias,
            other => {
                return Err(EngineError::message(format!(
                    "unsupported fence.proxy_async scope {other:?}"
                )));
            }
        };
        let effect = ProxyAsyncFenceEffect::new(
            scope,
            self.kernel.kernel_index(),
            self.context.global_cta_id(),
            self.context.cluster_id(),
        );
        // A proxy fence advances only the issuing warp's actor clocks. Program
        // order already prevents that warp's adjacent accesses from
        // overlapping the fence, while the racecheck state mutex chooses an
        // arbitrary valid order against unrelated warps. Taking the launch-
        // wide global transaction here would add cross-warp ordering that the
        // fence does not provide and serialize every independent proxy fence.
        self.before_effect(Some(operation), OperationEffect::ProxyAsyncFence(effect))?;
        self.after_effect(Some(operation), OperationEffect::ProxyAsyncFence(effect))?;
        self.record_local_engine_progress();
        Ok(())
    }

    pub(crate) fn observe_tensor_map(
        &self,
        operation: Option<&OperationContext>,
        observation: crate::effect::TensorMapObservation,
    ) -> Result<(), EngineError> {
        let effect = OperationEffect::TensorMap(observation);
        self.before_effect(operation, effect)
            .and_then(|()| self.after_effect(operation, effect))
            .map_err(|error| match operation {
                Some(operation) => error.with_operation_context(operation),
                None => error,
            })
    }

    pub(crate) fn memory_fence(
        &self,
        operation: Option<&OperationContext>,
        order: MemoryOrder,
        scope: MemoryScope,
        proxy: MemoryProxy,
    ) -> Result<(), EngineError> {
        let operation = self.checked_effect_operation(operation, OperationKind::Fence)?;
        let Some(operation) = operation else {
            return Ok(());
        };
        if !matches!(
            order,
            MemoryOrder::Acquire | MemoryOrder::Release | MemoryOrder::AcqRel | MemoryOrder::Sc
        ) {
            return Err(EngineError::message(format!(
                "memory fence requires acquire, release, acq_rel, or sc ordering, got {order}"
            )));
        }
        let effect = MemoryFenceEffect::new(order, scope, proxy);
        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalTransactionFence);
        let _transaction =
            begin_global_memory_transaction::<M>(self.kernel.mode_state(), true, true, &[])?;
        self.before_effect(Some(operation), OperationEffect::MemoryFence(effect))?;
        self.after_effect(Some(operation), OperationEffect::MemoryFence(effect))?;
        self.record_local_engine_progress();
        Ok(())
    }

    pub(crate) fn tcgen_fence(
        &self,
        operation: Option<&OperationContext>,
        kind: TcgenFenceKind,
    ) -> Result<(), EngineError> {
        let operation = self.checked_effect_operation(operation, OperationKind::Fence)?;
        let Some(operation) = operation else {
            return Ok(());
        };
        self.before_effect(Some(operation), OperationEffect::TcgenFence(kind))?;
        self.after_effect(Some(operation), OperationEffect::TcgenFence(kind))?;
        self.record_local_engine_progress();
        Ok(())
    }

    /// Execute one generated rendezvous.
    ///
    /// Synchronizing scopes publish their completed effect.  Participation-only
    /// scopes validate a tile execution contract without inventing a
    /// memory-ordering edge.
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    pub(crate) async fn rendezvous_sync(
        &self,
        operation: Option<&OperationContext>,
        scope: &str,
        operation_name: &str,
    ) -> Result<(), EngineError> {
        let (operation, static_op_id, loop_iteration_path) =
            self.required_operation_identity(operation, OperationKind::Collective)?;
        let context = self.operation_warp_context(operation);
        let mask = operation.active_mask();
        let rendezvous = self.kernel.services().rendezvous();
        if rendezvous.participate_generated_scope(
            scope,
            static_op_id,
            operation_name,
            &loop_iteration_path,
            context,
            mask,
            SETMAXNREG_WARPS_PER_GROUP,
        )? {
            self.record_local_engine_progress();
            return Ok(());
        }
        match scope {
            "warp" => {
                rendezvous
                    .warp(static_op_id, operation_name, loop_iteration_path, context)?
                    .await?;
                if M::OBSERVES_OPERATIONS {
                    // Published after the numeric wait succeeds, so this
                    // rendezvous has no pre-numeric decision point and no
                    // paired `before_effect`.
                    self.after_effect(
                        Some(operation),
                        OperationEffect::WarpSync(WarpSyncEffect::new(mask)),
                    )?;
                }
            }
            "grid" => {
                rendezvous
                    .grid(static_op_id, operation_name, loop_iteration_path, context)?
                    .await?;
            }
            "warpgroup" => {
                // Generated tile helpers may revisit the same static site from
                // an internal Rust loop that is intentionally absent from the
                // TIR loop path.  The private named-barrier namespace supplies
                // real generations, so both engine modes can distinguish those
                // visits without exposing a synthetic loop coordinate in the ABI.
                let Some(plan) = plan_internal_warpgroup_sync(
                    &context,
                    static_op_id,
                    SETMAXNREG_WARPS_PER_GROUP,
                    mask,
                )?
                else {
                    return Ok(());
                };
                self.named_barrier_sync_plan_effect(Some(operation), plan)
                    .await?;
            }
            other => {
                return Err(EngineError::message(format!(
                    "unsupported generated rendezvous scope {other:?}"
                )));
            }
        }

        self.record_local_engine_progress();
        Ok(())
    }

    /// Run a numerically evaluated warp collective while retaining its exact
    /// dynamic operation as error evidence for analysis-capable artifacts.
    pub(crate) fn collective_numeric_operation<R>(
        &self,
        operation: Option<&OperationContext>,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        if !M::OBSERVES_OPERATIONS {
            debug_assert!(operation.is_none());
            return numeric_effect();
        }
        let operation = self
            .checked_effect_operation(operation, OperationKind::Collective)?
            .expect("observed mode requires an operation context");
        numeric_effect().map_err(|error| error.with_operation_context(operation))
    }

    /// Execute one runtime memory operation behind an exact active-lane
    /// physical access batch.
    pub(crate) fn runtime_physical_access<R>(
        &self,
        operation: Option<&OperationContext>,
        descriptor: PhysicalAccessDescriptor,
        buffer: &RuntimeBuffer,
        indices: &WarpValue<i64>,
        mask: WarpMask,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        let element_stride = descriptor.width().bytes();
        self.runtime_physical_access_impl(
            operation,
            descriptor,
            element_stride,
            None,
            buffer,
            indices,
            None,
            mask,
            numeric_effect,
        )
    }

    /// Execute a runtime memory operation while retaining the logical buffer
    /// name that selected the resolved physical bytes.
    pub(crate) fn runtime_named_physical_access<R>(
        &self,
        operation: Option<&OperationContext>,
        operation_kind: OperationKind,
        byte_width: usize,
        element_stride: usize,
        logical_buffer: &str,
        buffer: &RuntimeBuffer,
        indices: &WarpValue<i64>,
        mask: WarpMask,
        validation_only: bool,
        numeric_effect: &mut dyn FnMut() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        self.runtime_named_physical_access_with_offsets(
            operation,
            operation_kind,
            byte_width,
            element_stride,
            logical_buffer,
            buffer,
            indices,
            None,
            mask,
            validation_only,
            numeric_effect,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn runtime_named_physical_access_with_offsets<R>(
        &self,
        operation: Option<&OperationContext>,
        operation_kind: OperationKind,
        byte_width: usize,
        element_stride: usize,
        logical_buffer: &str,
        buffer: &RuntimeBuffer,
        indices: &WarpValue<i64>,
        byte_offsets: Option<&WarpValue<usize>>,
        mask: WarpMask,
        validation_only: bool,
        numeric_effect: &mut dyn FnMut() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        if validation_only && operation.is_none() {
            let kind = physical_access_kind(operation_kind)?;
            if let Some(byte_offsets) = byte_offsets {
                validate_runtime_scalar_warp_access_at_byte_offsets(
                    &self.context,
                    buffer,
                    byte_offsets,
                    mask,
                    byte_width,
                    kind,
                )?;
            } else {
                validate_runtime_scalar_warp_access(
                    &self.context,
                    buffer,
                    indices,
                    mask,
                    byte_width,
                    kind,
                )?;
            }
            let result = numeric_effect()?;
            if M::OBSERVES_OPERATIONS {
                if let Some(space) = runtime_buffer_physical_space_for_mask(buffer, mask)? {
                    // Resolved for its validation error path only; the
                    // notification itself needs just the warp.
                    let _descriptor = physical_access_descriptor_for_mode::<M>(
                        kind, space, byte_width, buffer, mask,
                    )?;
                    M::after_unobserved_physical_access(
                        self.kernel.mode_state(),
                        self.context.global_warp_id(),
                    )?;
                }
            }
            return Ok(result);
        }
        if !M::OBSERVES_OPERATIONS {
            debug_assert!(operation.is_none());
            return numeric_effect();
        }
        if M::elides_physical_access_resolution(self.kernel.mode_state(), operation_kind) {
            debug_assert!(operation.is_none());
            return numeric_effect().map_err(|error| {
                self.physical_access_error_context(error, operation, Some(logical_buffer))
            });
        }
        let kind = physical_access_kind(operation_kind)?;
        let Some(space) = runtime_buffer_physical_space_for_mask(buffer, mask)? else {
            debug_assert!(operation.is_none());
            return numeric_effect();
        };
        let descriptor =
            physical_access_descriptor_for_mode::<M>(kind, space, byte_width, buffer, mask)?;
        self.runtime_physical_access_impl(
            operation,
            descriptor,
            element_stride,
            Some(logical_buffer),
            buffer,
            indices,
            byte_offsets,
            mask,
            || numeric_effect(),
        )
    }

    /// Execute the summary-only native-analysis path without materializing an
    /// owned `LanePhysicalAccess` for every active lane.
    #[allow(clippy::too_many_arguments)]
    fn compact_single_span_physical_access<R>(
        &self,
        operation: &OperationContext,
        descriptor: PhysicalAccessDescriptor,
        logical_buffer: Option<&str>,
        atomic_return_sync_relevant: bool,
        global_access_prelinearized: bool,
        mask: WarpMask,
        mut resolve_lane: impl FnMut(usize) -> Result<crate::PhysicalByteSpan, EngineError>,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        let mut lane_spans = [None; crate::WARP_SIZE];
        for lane in mask {
            let span = resolve_lane(lane).map_err(|error| {
                self.physical_access_error_context(error, Some(operation), logical_buffer)
            })?;
            if span.byte_len() != descriptor.width().bytes() {
                return Err(EngineError::message(format!(
                    "physical access at {}/lane:{lane} resolved {} bytes, expected {}",
                    operation.id(),
                    span.byte_len(),
                    descriptor.width().bytes(),
                )));
            }
            lane_spans[lane] = Some(span);
        }
        let batch = crate::physical_access::CompactPhysicalAccessBatch::new(
            operation,
            descriptor,
            logical_buffer,
            atomic_return_sync_relevant,
            &lane_spans,
        );
        let participates_in_global_memory = descriptor.space().has_read_from_versions()
            && (descriptor.kind().writes()
                || (descriptor.space() == PhysicalAccessSpace::Shared
                    && descriptor.memory_semantics().order().is_strong()))
            && !global_access_prelinearized;
        let _profile = participates_in_global_memory
            .then(|| ProfileTimer::new(ProfileKind::RaceGlobalTransactionPhysical));
        let transaction_spans = if M::USES_GLOBAL_MEMORY_TRANSACTION
            && (descriptor.space() == PhysicalAccessSpace::Global || participates_in_global_memory)
        {
            let mut spans = lane_spans.iter().flatten().copied().collect::<Vec<_>>();
            spans.sort_unstable();
            spans.dedup();
            spans
        } else {
            Vec::new()
        };
        let _transaction = begin_global_memory_transaction::<M>(
            self.kernel.mode_state(),
            participates_in_global_memory,
            descriptor.memory_semantics() != MemoryAccessSemantics::plain()
                || (descriptor.space() == PhysicalAccessSpace::Shared
                    && descriptor.kind().writes()),
            &transaction_spans,
        )?;
        let apply_after_numeric =
            M::applies_compact_physical_access_after_numeric(self.kernel.mode_state(), descriptor);
        if !apply_after_numeric {
            M::before_compact_physical_access(self.kernel.mode_state(), &batch)?;
        }
        let result = numeric_effect().map_err(|error| {
            if !apply_after_numeric {
                M::abort_compact_physical_access(self.kernel.mode_state(), &batch);
            }
            self.physical_access_error_context(error, Some(operation), logical_buffer)
        })?;
        M::after_compact_physical_access(self.kernel.mode_state(), &batch)?;
        Ok(result)
    }

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn runtime_physical_access_impl<R>(
        &self,
        operation: Option<&OperationContext>,
        descriptor: PhysicalAccessDescriptor,
        element_stride: usize,
        logical_buffer: Option<&str>,
        buffer: &RuntimeBuffer,
        indices: &WarpValue<i64>,
        byte_offsets: Option<&WarpValue<usize>>,
        mask: WarpMask,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        let expected_kind = match descriptor.kind() {
            PhysicalAccessKind::Read => OperationKind::Load,
            PhysicalAccessKind::Write => OperationKind::Store,
            PhysicalAccessKind::AtomicReadModifyWrite => OperationKind::Atomic,
        };
        if !M::controls_physical_access_allocation(
            self.kernel.mode_state(),
            expected_kind,
            descriptor.space(),
            runtime_global_allocation(buffer),
        ) {
            if operation.is_some() {
                let operation = self
                    .checked_effect_operation(operation, expected_kind)?
                    .expect("supplied operation context must remain available");
                if operation.active_mask() != mask {
                    return Err(EngineError::message(format!(
                        "physical access operation mask {:#010x} does not match resolved mask {:#010x}",
                        operation.active_mask().bits(),
                        mask.bits(),
                    )));
                }
            }
            let result = numeric_effect().map_err(|error| {
                self.physical_access_error_context(error, operation, logical_buffer)
            })?;
            M::after_unobserved_physical_access(
                self.kernel.mode_state(),
                self.context.global_warp_id(),
            )?;
            return Ok(result);
        }
        if !M::controls_physical_access(self.kernel.mode_state(), expected_kind, descriptor.space())
        {
            debug_assert!(operation.is_none());
            let result = numeric_effect().map_err(|error| {
                self.physical_access_error_context(error, operation, logical_buffer)
            })?;
            M::after_unobserved_physical_access(
                self.kernel.mode_state(),
                self.context.global_warp_id(),
            )?;
            return Ok(result);
        }
        let operation = self
            .checked_effect_operation(operation, expected_kind)?
            .expect("observed mode requires an operation context");
        if operation.active_mask() != mask {
            return Err(EngineError::message(format!(
                "physical access operation mask {:#010x} does not match resolved mask {:#010x}",
                operation.active_mask().bits(),
                mask.bits(),
            )));
        }
        if !M::observes_physical_access_batch(self.kernel.mode_state(), descriptor, mask) {
            let result =
                numeric_effect().map_err(|error| error.with_operation_context(operation))?;
            M::after_unobserved_physical_access(
                self.kernel.mode_state(),
                self.context.global_warp_id(),
            )
            .map_err(|error| error.with_operation_context(operation))?;
            return Ok(result);
        }
        if M::compacts_direct_physical_access(self.kernel.mode_state(), descriptor, mask, false) {
            return self.compact_single_span_physical_access(
                operation,
                descriptor,
                logical_buffer,
                false,
                false,
                mask,
                |lane| {
                    let byte_offset = match byte_offsets {
                        Some(byte_offsets) => byte_offsets[lane],
                        None => {
                            let index = usize::try_from(indices[lane]).map_err(|_| {
                                EngineError::out_of_bounds(format!(
                                    "negative buffer index {} on lane {lane}",
                                    indices[lane]
                                ))
                            })?;
                            element_byte_offset(buffer, index, element_stride, lane)?
                        }
                    };
                    let resolved = resolve_runtime_physical_access(
                        &self.context,
                        buffer,
                        lane,
                        byte_offset,
                        descriptor.width().bytes(),
                        descriptor.kind(),
                    )?;
                    if resolved.space() != descriptor.space() {
                        return Err(EngineError::message(format!(
                            "resolved physical access space {} does not match descriptor {} on lane {lane}",
                            resolved.space(),
                            descriptor.space(),
                        )));
                    }
                    Ok(resolved.span())
                },
                numeric_effect,
            );
        }
        let mut batch =
            PhysicalAccessBatch::resolve_single_span(operation.clone(), descriptor, |provenance| {
                let lane = provenance.lane();
                let byte_offset = match byte_offsets {
                    Some(byte_offsets) => byte_offsets[lane],
                    None => {
                        let index = usize::try_from(indices[lane]).map_err(|_| {
                            EngineError::out_of_bounds(format!(
                                "negative buffer index {} on lane {lane}",
                                indices[lane]
                            ))
                        })?;
                        element_byte_offset(buffer, index, element_stride, lane)?
                    }
                };
                let resolved = resolve_runtime_physical_access(
                    &self.context,
                    buffer,
                    lane,
                    byte_offset,
                    descriptor.width().bytes(),
                    descriptor.kind(),
                )?;
                if resolved.space() != descriptor.space() {
                    return Err(EngineError::message(format!(
                    "resolved physical access space {} does not match descriptor {} on lane {lane}",
                    resolved.space(),
                    descriptor.space(),
                )));
                }
                Ok(resolved.span())
            })
            .map_err(|error| match error {
                PhysicalAccessBatchError::LaneResolution { source, .. } => source,
                error => EngineError::message(error.to_string()),
            })
            .map_err(|error| error.with_operation_context(operation))?;
        if let Some(logical_buffer) = logical_buffer {
            batch = batch.with_logical_buffer(logical_buffer);
        }
        let effect = OperationEffect::PhysicalAccess(&batch);
        let participates_in_global_memory = descriptor.space().has_read_from_versions();
        let _profile = participates_in_global_memory
            .then(|| ProfileTimer::new(ProfileKind::RaceGlobalTransactionPhysical));
        let transaction_spans =
            if M::USES_GLOBAL_MEMORY_TRANSACTION && descriptor.space().has_read_from_versions() {
                crate::physical_access::global_batch_spans(std::iter::once(&batch))
            } else {
                Vec::new()
            };
        let _transaction = begin_global_memory_transaction::<M>(
            self.kernel.mode_state(),
            participates_in_global_memory,
            descriptor.memory_semantics() != MemoryAccessSemantics::plain()
                || (descriptor.space() == PhysicalAccessSpace::Shared
                    && descriptor.kind().writes()),
            &transaction_spans,
        )?;
        M::before_effect(self.kernel.mode_state(), operation, effect)?;
        let result = numeric_effect().map_err(|error| {
            M::abort_effect(self.kernel.mode_state(), operation, effect);
            self.physical_access_error_context(error, Some(operation), logical_buffer)
        })?;
        M::after_effect(self.kernel.mode_state(), operation, effect)?;
        Ok(result)
    }

    /// Execute a direct TMEM access while retaining the logical buffer name
    /// used to select the physical TMEM coordinates.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn runtime_named_tmem_physical_access<R>(
        &self,
        operation: Option<&OperationContext>,
        operation_kind: OperationKind,
        byte_width: usize,
        logical_buffer: &str,
        access_mode: TmemAccessMode,
        buffer: &RuntimeBuffer,
        mapped_lanes: &WarpValue<i64>,
        tcol_elements: &WarpValue<i64>,
        allocated_addrs: &WarpValue<i64>,
        mask: WarpMask,
        validation_only: bool,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        let kind = physical_access_kind(operation_kind)?;
        if validation_only && operation.is_none() {
            validate_tmem_scalar_warp_access(
                &self.context,
                &self.kernel.services().tcgen(),
                access_mode,
                buffer,
                mapped_lanes,
                tcol_elements,
                allocated_addrs,
                byte_width,
                mask,
            )?;
            let result = numeric_effect()?;
            if M::OBSERVES_OPERATIONS {
                // Resolved for its validation error path only; the
                // notification itself needs just the warp.
                let _descriptor =
                    PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Tmem, byte_width)
                        .map_err(|error| EngineError::message(error.to_string()))?;
                M::after_unobserved_physical_access(
                    self.kernel.mode_state(),
                    self.context.global_warp_id(),
                )?;
            }
            return Ok(result);
        }
        let descriptor = PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Tmem, byte_width)
            .map_err(|error| EngineError::message(error.to_string()))?;
        let lifecycle = self.kernel.services().tcgen();
        self.runtime_tmem_physical_access_impl(
            operation,
            descriptor,
            Some(logical_buffer),
            &lifecycle,
            access_mode,
            buffer,
            mapped_lanes,
            tcol_elements,
            allocated_addrs,
            mask,
            numeric_effect,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn runtime_tmem_physical_access_impl<R>(
        &self,
        operation: Option<&OperationContext>,
        descriptor: PhysicalAccessDescriptor,
        logical_buffer: Option<&str>,
        lifecycle: &TcgenLifecycleHub,
        access_mode: TmemAccessMode,
        buffer: &RuntimeBuffer,
        mapped_lanes: &WarpValue<i64>,
        tcol_elements: &WarpValue<i64>,
        allocated_addrs: &WarpValue<i64>,
        mask: WarpMask,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        if !M::OBSERVES_OPERATIONS {
            debug_assert!(operation.is_none());
            // The byte arena is deliberately silent, so the instruction
            // gateway publishes once after a successful typed TMEM write.
            // Reads only change the issuing warp's private registers.
            return self.complete_tmem_physical_access(descriptor, numeric_effect());
        }
        if descriptor.space() != PhysicalAccessSpace::Tmem {
            return Err(EngineError::message(format!(
                "direct TMEM access used {} descriptor",
                descriptor.space()
            )));
        }
        let expected_kind = match descriptor.kind() {
            PhysicalAccessKind::Read => OperationKind::Load,
            PhysicalAccessKind::Write => OperationKind::Store,
            PhysicalAccessKind::AtomicReadModifyWrite => OperationKind::Atomic,
        };
        if operation.is_none()
            && !M::controls_physical_access(
                self.kernel.mode_state(),
                expected_kind,
                descriptor.space(),
            )
        {
            let result = numeric_effect().map_err(|error| {
                self.physical_access_error_context(error, operation, logical_buffer)
            })?;
            M::after_unobserved_physical_access(
                self.kernel.mode_state(),
                self.context.global_warp_id(),
            )?;
            return self.complete_tmem_physical_access(descriptor, Ok(result));
        }
        let operation = self
            .checked_effect_operation(operation, expected_kind)?
            .expect("observed mode requires an operation context");
        if operation.active_mask() != mask {
            return Err(EngineError::message(format!(
                "TMEM access operation mask {:#010x} does not match resolved mask {:#010x}",
                operation.active_mask().bits(),
                mask.bits(),
            )));
        }
        if !M::observes_physical_access_batch(self.kernel.mode_state(), descriptor, mask) {
            let result = numeric_effect().map_err(|error| {
                self.physical_access_error_context(error, Some(operation), logical_buffer)
            })?;
            M::after_unobserved_physical_access(
                self.kernel.mode_state(),
                self.context.global_warp_id(),
            )?;
            return self.complete_tmem_physical_access(descriptor, Ok(result));
        }
        if M::compacts_direct_physical_access(self.kernel.mode_state(), descriptor, mask, false) {
            let result = self.compact_single_span_physical_access(
                operation,
                descriptor,
                logical_buffer,
                false,
                false,
                mask,
                |lane| {
                    resolve_tmem_physical_access(
                        &self.context,
                        lifecycle,
                        access_mode,
                        buffer,
                        mapped_lanes[lane],
                        tcol_elements[lane],
                        allocated_addrs[lane],
                        descriptor.width().bytes(),
                        lane,
                    )
                },
                numeric_effect,
            )?;
            return self.complete_tmem_physical_access(descriptor, Ok(result));
        }
        let mut batch =
            PhysicalAccessBatch::resolve_single_span(operation.clone(), descriptor, |provenance| {
                let lane = provenance.lane();
                resolve_tmem_physical_access(
                    &self.context,
                    lifecycle,
                    access_mode,
                    buffer,
                    mapped_lanes[lane],
                    tcol_elements[lane],
                    allocated_addrs[lane],
                    descriptor.width().bytes(),
                    lane,
                )
            })
            .map_err(|error| match error {
                PhysicalAccessBatchError::LaneResolution { source, .. } => source,
                other => EngineError::message(other.to_string()),
            })?;
        if let Some(logical_buffer) = logical_buffer {
            batch = batch.with_logical_buffer(logical_buffer);
        }
        let effect = OperationEffect::PhysicalAccess(&batch);
        M::before_effect(self.kernel.mode_state(), operation, effect)?;
        let result = numeric_effect().map_err(|error| {
            M::abort_effect(self.kernel.mode_state(), operation, effect);
            self.physical_access_error_context(error, Some(operation), logical_buffer)
        })?;
        M::after_effect(self.kernel.mode_state(), operation, effect)?;
        self.complete_tmem_physical_access(descriptor, Ok(result))
    }

    fn complete_tmem_physical_access<R>(
        &self,
        descriptor: PhysicalAccessDescriptor,
        completed: Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        let result = completed?;
        if descriptor.kind().writes() {
            self.record_engine_progress();
        }
        Ok(result)
    }

    /// Execute a physical-pointer access with exact per-lane byte identities
    /// and PTX memory-order, scope, proxy, and atomicity metadata.
    /// Returning atomics additionally declare whether their value can affect
    /// synchronization control flow; the flag must be false for other kinds.
    /// Observe one physical access of `access_width` bytes per lane.
    ///
    /// `access_width` is how many bytes this access moves, which is not the
    /// same as the pointer's element width: a packed `f32x2` store spans two
    /// `float32` elements, and a vector store spans several. Analysis records
    /// the footprint from this width, so passing the element width instead
    /// would under-record the access and hide real races.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn physical_pointer_access<R>(
        &self,
        operation: Option<&OperationContext>,
        operation_kind: OperationKind,
        pointer: &PhysicalPtr,
        logical_buffer: Option<&str>,
        mask: WarpMask,
        access_width: usize,
        atomic_return_sync_relevant: bool,
        atomic_coherent_load: bool,
        semantics: MemoryAccessSemantics,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        let observes_while_body = M::OBSERVES_OPERATIONS
            && self
                .while_body_observations
                .lock()
                .expect("native while-loop observations poisoned")
                .last()
                .is_some();
        if observes_while_body {
            let exact_progress = if atomic_coherent_load
                && pointer.pointer_space_for_mask(mask)? == PointerSpace::Global
            {
                let spans = self
                    .pointer_atomic_ordering_addresses(pointer, mask, access_width)?
                    .into_iter()
                    .map(|(address, byte_len)| {
                        crate::PhysicalByteSpan::new(
                            crate::PhysicalAllocationId::new(address.allocation_id()),
                            address.byte_offset(),
                            byte_len,
                        )
                        .map_err(|error| EngineError::message(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                M::global_memory_progress_snapshot(self.kernel.mode_state(), spans)
            } else {
                None
            };
            let mut observations = self
                .while_body_observations
                .lock()
                .expect("native while-loop observations poisoned");
            let observation = observations
                .last_mut()
                .expect("checked while-loop observation remains active");
            if atomic_coherent_load {
                observation.atomic_poll = true;
                if let Some(exact_progress) = exact_progress {
                    observation.global_memory_progress.merge(exact_progress);
                } else {
                    observation.exact_global_progress_supported = false;
                }
            } else {
                observation.requires_engine_progress = true;
            }
        }
        let kind = self.prepare_physical_pointer_access(
            operation_kind,
            pointer,
            mask,
            access_width,
            atomic_return_sync_relevant,
            atomic_coherent_load,
        )?;
        let result = self.physical_pointer_access_impl(
            operation,
            kind,
            pointer,
            logical_buffer,
            mask,
            access_width,
            atomic_return_sync_relevant,
            false,
            semantics,
            numeric_effect,
        )?;
        if kind == PhysicalAccessKind::AtomicReadModifyWrite {
            self.complete_pointer_atomic_ordering(pointer, mask, access_width)?;
        }
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn physical_pointer_atomic_access<R>(
        &self,
        operation: Option<&OperationContext>,
        pointer: &PhysicalPtr,
        logical_buffer: Option<&str>,
        mask: WarpMask,
        access_width: usize,
        atomic_return_sync_relevant: bool,
        semantics: MemoryAccessSemantics,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        let kind = self.prepare_physical_pointer_access(
            OperationKind::Atomic,
            pointer,
            mask,
            access_width,
            atomic_return_sync_relevant,
            false,
        )?;
        debug_assert_eq!(kind, PhysicalAccessKind::AtomicReadModifyWrite);
        let ordering = self.kernel.services().ordering();
        let _reservation = ordering
            .reserve_prepared_atomic_access_async(self.context, true)
            .await?;
        let result = self.physical_pointer_access_impl(
            operation,
            kind,
            pointer,
            logical_buffer,
            mask,
            access_width,
            atomic_return_sync_relevant,
            _reservation.is_some(),
            semantics,
            numeric_effect,
        )?;
        self.complete_pointer_atomic_ordering(pointer, mask, access_width)?;
        Ok(result)
    }

    fn prepare_physical_pointer_access(
        &self,
        operation_kind: OperationKind,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        access_width: usize,
        atomic_return_sync_relevant: bool,
        atomic_coherent_load: bool,
    ) -> Result<PhysicalAccessKind, EngineError> {
        if access_width == 0 {
            return Err(EngineError::message(
                "physical access width must be nonzero",
            ));
        }
        // The observed width comes from the call site, so a wrong constant would
        // silently mis-record the footprint. Reject a width that cannot be a
        // whole number of the pointer's elements, which catches the mistake at
        // the point the access is registered.
        if access_width % pointer.pointee_itemsize() != 0 {
            return Err(EngineError::message(format!(
                "physical access width {access_width} does not cover whole {}-byte elements",
                pointer.pointee_itemsize()
            )));
        }
        let kind = physical_access_kind(operation_kind)?;
        if atomic_return_sync_relevant && kind != PhysicalAccessKind::AtomicReadModifyWrite {
            return Err(EngineError::message(
                "only an atomic physical access may have a synchronization-relevant return",
            ));
        }
        if atomic_coherent_load && kind != PhysicalAccessKind::Read {
            return Err(EngineError::message(
                "atomic-coherent ordering applies only to physical loads",
            ));
        }
        if kind == PhysicalAccessKind::AtomicReadModifyWrite {
            self.prepare_pointer_atomic_ordering(pointer, mask, access_width)?;
        }
        Ok(kind)
    }

    fn pointer_atomic_ordering_addresses(
        &self,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        byte_len: usize,
    ) -> Result<Vec<(crate::PhysicalAddress, usize)>, EngineError> {
        mask.into_iter()
            .map(|lane| {
                let lane_mask = WarpMask::from_bits(1_u32 << lane);
                Ok((pointer.resolve_uniform(&self.context, lane_mask)?, byte_len))
            })
            .collect()
    }

    fn prepare_pointer_atomic_ordering(
        &self,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        access_width: usize,
    ) -> Result<(), EngineError> {
        let addresses = self.pointer_atomic_ordering_addresses(pointer, mask, access_width)?;
        if addresses.is_empty() {
            return Ok(());
        }
        match pointer.pointer_space_for_mask(mask)? {
            PointerSpace::Global => self
                .kernel
                .services()
                .ordering()
                .prepare_atomic_access(self.context, addresses),
            PointerSpace::Shared => Ok(()),
            PointerSpace::Local | PointerSpace::Register => Err(EngineError::message(
                "atomic access requires global or shared memory",
            )),
        }
    }

    fn complete_pointer_atomic_ordering(
        &self,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        access_width: usize,
    ) -> Result<(), EngineError> {
        let addresses = self.pointer_atomic_ordering_addresses(pointer, mask, access_width)?;
        if addresses.is_empty() {
            return Ok(());
        }
        match pointer.pointer_space_for_mask(mask)? {
            PointerSpace::Global => self
                .kernel
                .services()
                .ordering()
                .complete_atomic_rmw(self.context, addresses),
            PointerSpace::Shared => Ok(()),
            PointerSpace::Local | PointerSpace::Register => Err(EngineError::message(
                "atomic access requires global or shared memory",
            )),
        }
    }

    fn physical_pointer_access_impl<R>(
        &self,
        operation: Option<&OperationContext>,
        kind: PhysicalAccessKind,
        pointer: &PhysicalPtr,
        logical_buffer: Option<&str>,
        mask: WarpMask,
        access_width: usize,
        atomic_return_sync_relevant: bool,
        global_access_prelinearized: bool,
        semantics: MemoryAccessSemantics,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        debug_assert!(
            !atomic_return_sync_relevant || kind == PhysicalAccessKind::AtomicReadModifyWrite
        );
        if !M::OBSERVES_OPERATIONS {
            debug_assert!(operation.is_none());
            return numeric_effect();
        }
        if mask.is_empty() {
            // A fully predicated-off pointer instruction has no physical
            // address, access, or proxy domain.  Runtime-buffer accesses use
            // the same no-effect path when their resolved mask is empty.
            return numeric_effect();
        }
        let expected_kind = match kind {
            PhysicalAccessKind::Read => OperationKind::Load,
            PhysicalAccessKind::Write => OperationKind::Store,
            PhysicalAccessKind::AtomicReadModifyWrite => OperationKind::Atomic,
        };
        let space = match pointer
            .pointer_space_for_mask(mask)
            .map_err(|error| self.physical_access_error_context(error, operation, logical_buffer))?
        {
            PointerSpace::Global => PhysicalAccessSpace::Global,
            PointerSpace::Shared => PhysicalAccessSpace::Shared,
            PointerSpace::Local => PhysicalAccessSpace::Local,
            PointerSpace::Register => PhysicalAccessSpace::Register,
        };
        let byte_width = access_width;
        let descriptor = physical_access_descriptor_for_mode::<M>(
            kind,
            space,
            byte_width,
            pointer.buffer(),
            mask,
        )?
        .with_memory_semantics(semantics);
        if !M::controls_physical_access_allocation(
            self.kernel.mode_state(),
            expected_kind,
            space,
            runtime_global_allocation(pointer.buffer()),
        ) {
            if operation.is_some() {
                let operation = self
                    .checked_effect_operation(operation, expected_kind)?
                    .expect("supplied operation context must remain available");
                if operation.active_mask() != mask {
                    return Err(EngineError::message(format!(
                        "physical pointer operation mask {:#010x} does not match resolved mask {:#010x}",
                        operation.active_mask().bits(),
                        mask.bits(),
                    )));
                }
            }
            let result = numeric_effect().map_err(|error| {
                self.physical_access_error_context(error, operation, logical_buffer)
            })?;
            M::after_unobserved_physical_access(
                self.kernel.mode_state(),
                self.context.global_warp_id(),
            )?;
            return Ok(result);
        }
        if !M::controls_physical_access(self.kernel.mode_state(), expected_kind, space) {
            debug_assert!(operation.is_none());
            let result = numeric_effect().map_err(|error| {
                self.physical_access_error_context(error, operation, logical_buffer)
            })?;
            M::after_unobserved_physical_access(
                self.kernel.mode_state(),
                self.context.global_warp_id(),
            )?;
            return Ok(result);
        }
        let operation = self
            .checked_effect_operation(operation, expected_kind)?
            .expect("observed mode requires an operation context");
        if operation.active_mask() != mask {
            return Err(EngineError::message(format!(
                "physical pointer operation mask {:#010x} does not match resolved mask {:#010x}",
                operation.active_mask().bits(),
                mask.bits(),
            )));
        }
        let observes_batch = if kind == PhysicalAccessKind::AtomicReadModifyWrite {
            M::observes_atomic_physical_access_batch(
                self.kernel.mode_state(),
                descriptor,
                mask,
                atomic_return_sync_relevant,
            )
        } else {
            M::observes_physical_access_batch(self.kernel.mode_state(), descriptor, mask)
        };
        if !observes_batch {
            let result =
                numeric_effect().map_err(|error| error.with_operation_context(operation))?;
            M::after_unobserved_physical_access(
                self.kernel.mode_state(),
                self.context.global_warp_id(),
            )
            .map_err(|error| error.with_operation_context(operation))?;
            return Ok(result);
        }
        if M::compacts_direct_physical_access(
                self.kernel.mode_state(),
                descriptor,
                mask,
                atomic_return_sync_relevant,
            )
        {
            return self.compact_single_span_physical_access(
                operation,
                descriptor,
                logical_buffer,
                atomic_return_sync_relevant,
                global_access_prelinearized,
                mask,
                |lane| {
                    let byte_offset = match kind {
                        PhysicalAccessKind::Read => {
                            pointer.lane_read_byte_offset(lane, byte_width)?
                        }
                        PhysicalAccessKind::Write => {
                            pointer.lane_write_byte_offset(lane, byte_width)?
                        }
                        PhysicalAccessKind::AtomicReadModifyWrite => {
                            pointer.lane_read_write_byte_offset(lane, byte_width)?
                        }
                    };
                    let resolved = resolve_runtime_physical_access(
                        &self.context,
                        pointer.buffer(),
                        lane,
                        byte_offset,
                        byte_width,
                        kind,
                    )?;
                    if resolved.space() != space {
                        return Err(EngineError::message(format!(
                            "resolved pointer access space {} does not match pointer space {space} on lane {lane}",
                            resolved.space(),
                        )));
                    }
                    Ok(resolved.span())
                },
                numeric_effect,
            );
        }
        let mut batch = PhysicalAccessBatch::resolve_single_span(
            operation.clone(),
            descriptor,
            |provenance| {
            let lane = provenance.lane();
            let byte_offset = match kind {
                PhysicalAccessKind::Read => pointer.lane_read_byte_offset(lane, byte_width)?,
                PhysicalAccessKind::Write => pointer.lane_write_byte_offset(lane, byte_width)?,
                PhysicalAccessKind::AtomicReadModifyWrite => {
                    pointer.lane_read_write_byte_offset(lane, byte_width)?
                }
            };
            let resolved = resolve_runtime_physical_access(
                &self.context,
                pointer.buffer(),
                lane,
                byte_offset,
                byte_width,
                kind,
            )?;
            if resolved.space() != space {
                return Err(EngineError::message(format!(
                    "resolved pointer access space {} does not match pointer space {space} on lane {lane}",
                    resolved.space(),
                )));
            }
                Ok(resolved.span())
            },
        )
        .map_err(|error| match error {
            PhysicalAccessBatchError::LaneResolution { source, .. } => source,
            error => EngineError::message(error.to_string()),
        })
        .map_err(|error| error.with_operation_context(operation))?
        .with_atomic_return_sync_relevant(atomic_return_sync_relevant);
        if let Some(logical_buffer) = logical_buffer {
            batch = batch.with_logical_buffer(logical_buffer);
        }
        let effect = OperationEffect::PhysicalAccess(&batch);
        let participates_in_global_memory =
            descriptor.space().has_read_from_versions() && !global_access_prelinearized;
        let _profile = participates_in_global_memory
            .then(|| ProfileTimer::new(ProfileKind::RaceGlobalTransactionPhysical));
        let transaction_spans =
            if M::USES_GLOBAL_MEMORY_TRANSACTION && descriptor.space().has_read_from_versions() {
                crate::physical_access::global_batch_spans(std::iter::once(&batch))
            } else {
                Vec::new()
            };
        let _transaction = begin_global_memory_transaction::<M>(
            self.kernel.mode_state(),
            participates_in_global_memory,
            descriptor.memory_semantics() != MemoryAccessSemantics::plain()
                || (descriptor.space() == PhysicalAccessSpace::Shared
                    && descriptor.kind().writes()),
            &transaction_spans,
        )?;
        M::before_effect(self.kernel.mode_state(), operation, effect)
            .map_err(|error| error.with_operation_context(operation))?;
        let result = numeric_effect().map_err(|error| {
            M::abort_effect(self.kernel.mode_state(), operation, effect);
            error.with_operation_context(operation)
        })?;
        // A synchronization word's content is read back here, between the
        // write and the commit. Nothing else runs in this span -- no await
        // separates the two -- so what the word holds now is exactly what this
        // access left behind.
        //
        // Every strong *global* write takes this path, not only a primitive's:
        // a declared word's publisher is spelled in raw PTX, so gating on the
        // operation's identity would leave the waiter with no history to name
        // the write it accepted. A plain write cannot publish anything and
        // stays content-blind, and so does a shared-memory one -- a declared
        // word is global (`_ATOMIC_REF_SPACE`), and the read-back resolves the
        // address in the global space.
        //
        // The width bound is the word's own: `wait_until` polls 4 or 8
        // bytes, so nothing wider can *be* a declared word, and reading one
        // back would only mean inventing a `u64` for bytes the wait cannot
        // name. A wider strong write that overlaps a declared word is not
        // silently dropped -- the checker turns that into an incomplete
        // reason, because a history missing a publication would otherwise make
        // a correct protocol look unpublished.
        if descriptor.space() == PhysicalAccessSpace::Global
            && semantics.class().is_atomic_class()
            && kind.writes()
            && matches!(byte_width, 4 | 8)
        {
            let values = self
                .declared_word_post_images(&batch, pointer, mask, byte_width)
                .inspect_err(|_| {
                    M::abort_effect(
                        self.kernel.mode_state(),
                        operation,
                        OperationEffect::PhysicalAccess(&batch),
                    )
                })?;
            batch = batch.with_declared_values(values);
        }
        M::after_effect(
            self.kernel.mode_state(),
            operation,
            OperationEffect::PhysicalAccess(&batch),
        )
        .map_err(|error| error.with_operation_context(operation))?;
        Ok(result)
    }

    /// Wait on a declared synchronization word until its predicate holds.
    ///
    /// One operation, the way an mbarrier wait is one operation. A kernel
    /// spells the wait as a loop and so does the CUDA it lowers to, but the
    /// polling is the mechanism, not a stream of reads for the memory model to
    /// adjudicate: every one of them races the publisher's write by
    /// construction, so judging them reports every correct protocol and the
    /// answer moves with the schedule. The engine does the waiting instead,
    /// and what the checker sees is the word's write history and one verdict.
    ///
    /// The word's own bytes are what wakes it, so an unsatisfiable wait parks
    /// rather than spins and the executor's deadlock detection can name it --
    /// the same trade an mbarrier wait makes, and the reason neither needs a
    /// loop budget.
    pub(crate) async fn declared_word_wait_until(
        &self,
        operation: Option<&OperationContext>,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        byte_width: usize,
        semantics: MemoryAccessSemantics,
        mut accepts: impl FnMut(u64, usize) -> Result<bool, EngineError>,
    ) -> Result<WarpValue<u64>, EngineError> {
        let spans = self.declared_word_spans(pointer, mask, byte_width)?;
        loop {
            // The snapshot is taken *before* the word is read, and that order
            // is the whole of the wait's liveness. A publication landing
            // between the two must either be visible to the read below -- in
            // which case the wait exits -- or leave this snapshot stale, in
            // which case the watch returns at once and the loop reads again.
            // Taken afterwards it would already carry that write, and the
            // watch would be waiting for the *next* one: a publisher that has
            // finished never sends it, and the warp sleeps forever.
            let watch = M::global_memory_progress_snapshot(
                self.kernel.mode_state(),
                spans.iter().copied(),
            );
            let semantic = watch.is_none().then(|| {
                self.kernel
                    .physical()
                    .semantic_progress_snapshot_for_warp(self.context.global_warp_id())
            });
            let mut observed = self.declared_word_read(pointer, mask, byte_width)?;
            let mut accepted_by_all = true;
            for lane in mask {
                if !accepts(observed.get(lane).copied().unwrap_or(0), lane)? {
                    accepted_by_all = false;
                    break;
                }
            }
            if accepted_by_all {
                // Polling needs no publication evidence. Only a possible
                // exit joins the publishers' existing serialization: scoped
                // stores use the global transaction, and atomics retain an
                // exact-address reservation through their history commit.
                // Readers may coexist; neither guard adds a modeled HB edge.
                let _reservation = if M::OBSERVES_OPERATIONS {
                    self.prepare_pointer_atomic_ordering(pointer, mask, byte_width)?;
                    self.kernel
                        .services()
                        .ordering()
                        .reserve_prepared_atomic_access_async(self.context, false)
                        .await?
                } else {
                    None
                };
                let _transaction = begin_global_memory_transaction::<M>(
                    self.kernel.mode_state(), true, false, &spans,
                )?;
                if M::OBSERVES_OPERATIONS {
                    // The value may have changed while acquiring the guards.
                    // Revalidate under them before naming an exit's history.
                    observed = self.declared_word_read(pointer, mask, byte_width)?;
                    for lane in mask {
                        if !accepts(observed.get(lane).copied().unwrap_or(0), lane)? {
                            accepted_by_all = false;
                            break;
                        }
                    }
                    if !accepted_by_all {
                        // Scope exit drops both guards before polling again.
                        continue;
                    }
                }
                self.record_engine_progress();
                self.declared_word_wait(operation, pointer, mask, byte_width, semantics, accepts)?;
                return Ok(WarpValue::<u64>::from_fn(|lane| {
                    observed.get(lane).copied().unwrap_or(0)
                }));
            }
            // Nothing this actor can see satisfies the predicate yet, so it is
            // the word's bytes it is waiting on. Parking on exactly those makes
            // the wait a pending effect a deadlock report can name, instead of
            // a warp that merely keeps being runnable.
            let ordering = self.kernel.services().ordering();
            let _park = ordering.park(
                self.context.global_warp_id(),
                crate::ordering::ParkReason::DeclaredWordWait,
                operation.map(|operation| operation.id().clone()),
            );
            match (watch, semantic) {
                (Some(progress), _) => progress.watch().await,
                (None, Some(current)) => {
                    self.kernel
                        .physical()
                        .watch_semantic_progress(current, true)
                        .await;
                }
                (None, None) => unreachable!("a semantic snapshot is taken whenever the memory one is absent"),
            }
        }
    }

    /// The byte span each active lane's wait is on.
    fn declared_word_spans(
        &self,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        byte_width: usize,
    ) -> Result<Vec<crate::PhysicalByteSpan>, EngineError> {
        let mut spans = Vec::with_capacity(mask.len());
        for lane in mask {
            let byte_offset = pointer.lane_read_byte_offset(lane, byte_width)?;
            spans.push(
                resolve_runtime_physical_access(
                    &self.context,
                    pointer.buffer(),
                    lane,
                    byte_offset,
                    byte_width,
                    PhysicalAccessKind::Read,
                )?
                .span(),
            );
        }
        Ok(spans)
    }

    /// What the word holds right now, per lane, without recording an access.
    ///
    /// The read is the wait's own mechanism; making it an observed access is
    /// exactly what this operation exists to avoid.
    fn declared_word_read(
        &self,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        byte_width: usize,
    ) -> Result<Vec<u64>, EngineError> {
        use crate::runtime::io::raw_load_physical_ptr_warp;

        let physical = self.kernel.physical();
        let mut values = vec![0_u64; crate::WARP_SIZE];
        match byte_width {
            4 => {
                let read = raw_load_physical_ptr_warp::<u32>(
                    physical,
                    &self.context,
                    pointer,
                    mask,
                    PtxStateSpace::Global,
                )?;
                for lane in mask {
                    values[lane] = u64::from(read[lane]);
                }
            }
            8 => {
                let read = raw_load_physical_ptr_warp::<u64>(
                    physical,
                    &self.context,
                    pointer,
                    mask,
                    PtxStateSpace::Global,
                )?;
                for lane in mask {
                    values[lane] = read[lane];
                }
            }
            _ => {
                return Err(EngineError::message(format!(
                    "a declared synchronization word is 4 or 8 bytes wide, got {byte_width}"
                )))
            }
        }
        Ok(values)
    }

    /// Resolve one lane-wise wait on a declared synchronization word.
    ///
    /// The predicate stays with the generated code, which is the only place it
    /// can run -- it reads thread-local scalars such as a barrier's phase. What
    /// crosses into the checker is a position in the word's own write history,
    /// so the checker is handed a decision about its own data rather than a
    /// callback (API §3 steps 1 and 4).
    pub(crate) fn declared_word_wait(
        &self,
        operation: Option<&OperationContext>,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        byte_width: usize,
        semantics: MemoryAccessSemantics,
        mut accepts: impl FnMut(u64, usize) -> Result<bool, EngineError>,
    ) -> Result<(), EngineError> {
        let Some(operation) = operation else {
            return Ok(());
        };
        if !M::OBSERVES_OPERATIONS || mask.is_empty() {
            return Ok(());
        }
        for lane in mask {
            let byte_offset = pointer.lane_read_byte_offset(lane, byte_width)?;
            let resolved = resolve_runtime_physical_access(
                &self.context,
                pointer.buffer(),
                lane,
                byte_offset,
                byte_width,
                PhysicalAccessKind::Read,
            )?;
            let span = resolved.span();
            // `skip` is how many of the word's writes this actor is already
            // causally after; the predicate only sees what could still have
            // released it, and the position comes back absolute.
            let (skip, values) = M::declared_word_candidates(
                self.kernel.mode_state(),
                span,
                self.context.global_warp_id(),
                lane,
            );
            let mut accepted = None;
            for (index, value) in values.iter().enumerate().skip(skip) {
                if accepts(*value, lane)? {
                    accepted = Some(index);
                    break;
                }
            }
            let mut satisfied_on_entry = false;
            if accepted.is_none() {
                for value in &values[..skip.min(values.len())] {
                    if accepts(*value, lane)? {
                        satisfied_on_entry = true;
                        break;
                    }
                }
            }
            // A wait whose predicate already holds on the launch value needed
            // nobody to publish: the host's writes precede every actor, so the
            // earliest exit the loop could have taken is the one that was
            // there before the kernel started. Judging by that earliest exit
            // is the same rule the accepted position follows, so the answer is
            // schedule-independent. It is only worth asking when the history
            // named nothing -- otherwise a recorded write already explains the
            // exit.
            let satisfied_by_launch_value = if accepted.is_none() && !satisfied_on_entry {
                match self.kernel.physical().global().launch_word(
                    crate::memory::AllocationId::from_u64(span.allocation().get()),
                    span.byte_offset(),
                    span.byte_len(),
                ) {
                    Some(launch) => Some(accepts(launch, lane)?),
                    None => None,
                }
            } else {
                None
            };
            let plan = DeclaredWordWaitPlan::new(
                span,
                self.context.global_warp_id(),
                lane,
                accepted,
                satisfied_on_entry,
                satisfied_by_launch_value,
                semantics,
            );
            let effect = OperationEffect::DeclaredWordWait { plan };
            self.before_effect(Some(operation), effect)?;
            self.after_effect(Some(operation), effect)?;
        }
        Ok(())
    }

    /// What each active lane left in a declared word, in `batch.lanes()` order.
    fn declared_word_post_images(
        &self,
        batch: &PhysicalAccessBatch,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        byte_width: usize,
    ) -> Result<Arc<[u64]>, EngineError> {
        use crate::runtime::io::raw_load_physical_ptr_warp;

        let physical = self.kernel.physical();
        let lanes = batch.lanes();
        let values: Vec<u64> = match byte_width {
            4 => {
                let read = raw_load_physical_ptr_warp::<u32>(
                    physical,
                    &self.context,
                    pointer,
                    mask,
                    PtxStateSpace::Global,
                )?;
                lanes
                    .iter()
                    .map(|lane| u64::from(read[lane.provenance().lane()]))
                    .collect()
            }
            8 => {
                let read = raw_load_physical_ptr_warp::<u64>(
                    physical,
                    &self.context,
                    pointer,
                    mask,
                    PtxStateSpace::Global,
                )?;
                lanes
                    .iter()
                    .map(|lane| read[lane.provenance().lane()])
                    .collect()
            }
            _ => {
                return Err(EngineError::message(format!(
                    "a declared synchronization word is 4 or 8 bytes wide, got {byte_width}"
                )));
            }
        };
        Ok(values.into())
    }

    /// Execute a primitive whose exact per-lane footprint needs a custom
    /// resolver (for example ldmatrix/stmatrix lane swizzles).
    pub(crate) fn resolved_physical_access<R>(
        &self,
        operation: Option<&OperationContext>,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        byte_width: usize,
        mask: WarpMask,
        resolve: impl FnOnce(
            &OperationContext,
            PhysicalAccessDescriptor,
        ) -> Result<PhysicalAccessBatch, EngineError>,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        let descriptor = PhysicalAccessDescriptor::new(kind, space, byte_width)
            .map_err(|error| EngineError::message(error.to_string()))?;
        if !M::OBSERVES_OPERATIONS {
            debug_assert!(operation.is_none());
            return numeric_effect();
        }
        let expected_kind = match descriptor.kind() {
            PhysicalAccessKind::Read => OperationKind::Load,
            PhysicalAccessKind::Write => OperationKind::Store,
            PhysicalAccessKind::AtomicReadModifyWrite => OperationKind::Atomic,
        };
        let operation = self
            .checked_effect_operation(operation, expected_kind)?
            .expect("observed mode requires an operation context");
        if operation.active_mask() != mask {
            return Err(EngineError::message(format!(
                "resolved physical operation mask {:#010x} does not match {:#010x}",
                operation.active_mask().bits(),
                mask.bits(),
            )));
        }
        if !M::observes_physical_access_batch(self.kernel.mode_state(), descriptor, mask) {
            let result =
                numeric_effect().map_err(|error| error.with_operation_context(operation))?;
            M::after_unobserved_physical_access(
                self.kernel.mode_state(),
                self.context.global_warp_id(),
            )
            .map_err(|error| error.with_operation_context(operation))?;
            return Ok(result);
        }
        let batch = resolve(operation, descriptor)
            .map_err(|error| error.with_operation_context(operation))?;
        if batch.operation() != operation || batch.descriptor() != descriptor {
            return Err(EngineError::message(
                "custom physical footprint does not match its operation descriptor",
            ));
        }
        let effect = OperationEffect::PhysicalAccess(&batch);
        let participates_in_global_memory = descriptor.space().has_read_from_versions();
        let _profile = participates_in_global_memory
            .then(|| ProfileTimer::new(ProfileKind::RaceGlobalTransactionPhysical));
        let transaction_spans =
            if M::USES_GLOBAL_MEMORY_TRANSACTION && descriptor.space().has_read_from_versions() {
                crate::physical_access::global_batch_spans(std::iter::once(&batch))
            } else {
                Vec::new()
            };
        let _transaction = begin_global_memory_transaction::<M>(
            self.kernel.mode_state(),
            participates_in_global_memory,
            descriptor.memory_semantics() != MemoryAccessSemantics::plain()
                || (descriptor.space() == PhysicalAccessSpace::Shared
                    && descriptor.kind().writes()),
            &transaction_spans,
        )?;
        M::before_effect(self.kernel.mode_state(), operation, effect)
            .map_err(|error| error.with_operation_context(operation))?;
        let result = numeric_effect().map_err(|error| {
            M::abort_effect(self.kernel.mode_state(), operation, effect);
            error.with_operation_context(operation)
        })?;
        M::after_effect(self.kernel.mode_state(), operation, effect)
            .map_err(|error| error.with_operation_context(operation))?;
        Ok(result)
    }

    /// Execute one instruction with a pre-resolved batch whose lane widths may vary.
    pub(crate) fn resolved_physical_access_batch<R>(
        &self,
        operation: Option<&OperationContext>,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        mask: WarpMask,
        resolve: impl FnOnce(&OperationContext) -> Result<PhysicalAccessBatch, EngineError>,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        let expected_kind = match kind {
            PhysicalAccessKind::Read => OperationKind::Load,
            PhysicalAccessKind::Write => OperationKind::Store,
            PhysicalAccessKind::AtomicReadModifyWrite => OperationKind::Atomic,
        };
        if !M::controls_physical_access(self.kernel.mode_state(), expected_kind, space) {
            return numeric_effect();
        }
        let operation = self
            .checked_effect_operation(operation, expected_kind)?
            .expect("observed mode requires an operation context");
        if operation.active_mask() != mask {
            return Err(EngineError::message(format!(
                "resolved physical operation mask {:#010x} does not match {:#010x}",
                operation.active_mask().bits(),
                mask.bits(),
            )));
        }
        let batch = resolve(operation).map_err(|error| error.with_operation_context(operation))?;
        if batch.operation() != operation
            || batch.descriptor().kind() != kind
            || batch.descriptor().space() != space
        {
            return Err(EngineError::message(
                "resolved physical batch does not match its operation contract",
            ));
        }
        let effect = OperationEffect::PhysicalAccess(&batch);
        let participates_in_global_memory = space.has_read_from_versions();
        let _profile = participates_in_global_memory
            .then(|| ProfileTimer::new(ProfileKind::RaceGlobalTransactionPhysical));
        let transaction_spans =
            if M::USES_GLOBAL_MEMORY_TRANSACTION && space.has_read_from_versions() {
                crate::physical_access::global_batch_spans(std::iter::once(&batch))
            } else {
                Vec::new()
            };
        let _transaction = begin_global_memory_transaction::<M>(
            self.kernel.mode_state(),
            participates_in_global_memory,
            batch.descriptor().memory_semantics() != MemoryAccessSemantics::plain()
                || (space == PhysicalAccessSpace::Shared && kind.writes()),
            &transaction_spans,
        )?;
        M::before_effect(self.kernel.mode_state(), operation, effect)
            .map_err(|error| error.with_operation_context(operation))?;
        let result = numeric_effect().map_err(|error| {
            M::abort_effect(self.kernel.mode_state(), operation, effect);
            error.with_operation_context(operation)
        })?;
        M::after_effect(self.kernel.mode_state(), operation, effect)
            .map_err(|error| error.with_operation_context(operation))?;
        Ok(result)
    }

    /// Execute one exact `ldmatrix` form behind its engine-owned footprint.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ldmatrix_access<R>(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        source: &PhysicalPtr,
        register_count: usize,
        transpose: bool,
        source_bits: usize,
        mask: WarpMask,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        if !matches!(register_count, 1 | 2 | 4) {
            return Err(EngineError::message(format!(
                "ldmatrix fragment count must be 1, 2, or 4, got {register_count}"
            )));
        }
        if !matches!(source_bits, 4 | 6 | 8 | 16)
            || (source_bits == 8 && !transpose)
            || (source_bits != 16 && transpose && register_count == 1)
        {
            return Err(EngineError::message("unsupported ldmatrix layout"));
        }
        require_full_warp_sync(mask, "ldmatrix.sync.aligned")?;
        let collective_operation = if M::OBSERVES_OPERATIONS {
            let operation = self.required_operation(operation, OperationKind::Load)?;
            M::warp_collective_rendezvous(self.kernel.mode_state(), operation, mask)?;
            Some(operation)
        } else {
            None
        };
        let result = self.resolved_physical_access_batch(
            operation,
            PhysicalAccessKind::Read,
            PhysicalAccessSpace::Shared,
            mask,
            |operation| {
                plan_raw_ldmatrix_access(
                    operation,
                    context,
                    source,
                    register_count,
                    transpose,
                    source_bits,
                    mask,
                )
            },
            numeric_effect,
        )?;
        if let Some(operation) = collective_operation {
            M::warp_collective_rendezvous(self.kernel.mode_state(), operation, mask)?;
        }
        Ok(result)
    }

    /// Execute one exact `stmatrix` form behind its engine-owned footprint.
    pub(crate) fn stmatrix_access<R>(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        source_count: usize,
        stmatrix: StmatrixDescriptor,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        let mask = context.active_mask();
        require_full_warp_sync(mask, "stmatrix.sync.aligned")?;
        let collective_operation = if M::OBSERVES_OPERATIONS {
            let operation = self.required_operation(operation, OperationKind::Store)?;
            M::warp_collective_rendezvous(self.kernel.mode_state(), operation, mask)?;
            Some(operation)
        } else {
            None
        };
        let byte_width = source_count * 4;
        let result = self.resolved_physical_access(
            operation,
            PhysicalAccessKind::Write,
            PhysicalAccessSpace::Shared,
            byte_width,
            mask,
            |operation, _descriptor| {
                plan_raw_stmatrix_access(operation, context, destination, source_count, stmatrix)
            },
            numeric_effect,
        )?;
        if let Some(operation) = collective_operation {
            M::warp_collective_rendezvous(self.kernel.mode_state(), operation, mask)?;
        }
        Ok(result)
    }

    /// Execute one variable-width `st.bulk` behind its engine-owned footprint.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn st_bulk_zero_access<R>(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        num_bytes: &WarpValue<i64>,
        mask: WarpMask,
        ptx_space: PtxStateSpace,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        self.resolved_physical_access_batch(
            operation,
            PhysicalAccessKind::Write,
            PhysicalAccessSpace::Shared,
            mask,
            |operation| {
                plan_raw_st_bulk_zero_access(
                    operation,
                    context,
                    destination,
                    num_bytes,
                    mask,
                    ptx_space,
                )
            },
            numeric_effect,
        )
    }

    pub(crate) fn mbarrier_invalidate(
        &self,
        operation: Option<&OperationContext>,
        pointer: &PhysicalPtr,
        mask: WarpMask,
    ) -> Result<(), EngineError> {
        let mut ids = Vec::new();
        for lane in mask {
            ids.push(pointer.resolve_shared_barrier_write(
                &self.context,
                WarpMask::from_bits(1 << lane),
                None,
            )?);
        }
        let operation = self.checked_effect_operation(operation, OperationKind::Barrier)?;
        let effect = OperationEffect::MbarrierInvalidate { barrier_ids: &ids };
        self.before_effect(operation, effect)?;
        self.kernel.services().mbarriers().invalidate_many(&ids)?;
        self.kernel
            .services()
            .mbarrier_init_fences()
            .invalidate_many(&ids);
        self.after_effect(operation, effect)?;
        self.record_engine_progress();
        Ok(())
    }

    pub(crate) fn mbarrier_check_layout(
        &self,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        layout: u8,
    ) -> Result<WarpValue<bool>, EngineError> {
        let mut result = WarpValue::splat(false);
        for lane in mask {
            let id = pointer.resolve_shared_barrier(
                &self.context,
                WarpMask::from_bits(1 << lane),
                None,
            )?;
            result[lane] = self
                .kernel
                .services()
                .mbarriers()
                .check_layout(id, layout)?;
        }
        Ok(result)
    }

    /// Resolve and apply one `mbarrier.init` through the mode-generic engine.
    pub(crate) fn mbarrier_init(
        &self,
        operation: Option<&OperationContext>,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        expected_arrivals: &WarpValue<i64>,
    ) -> Result<(), EngineError> {
        self.mbarrier_init_layout(operation, pointer, mask, expected_arrivals, false)
    }

    pub(crate) fn mbarrier_init_layout(
        &self,
        operation: Option<&OperationContext>,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        expected_arrivals: &WarpValue<i64>,
        layout_v1: bool,
    ) -> Result<(), EngineError> {
        if !M::OBSERVES_OPERATIONS {
            debug_assert!(operation.is_none());
            initialize_physical_mbarriers(
                &self.context,
                pointer,
                mask,
                &self.kernel.services().mbarriers(),
                expected_arrivals,
                layout_v1,
            )?;
            self.record_engine_progress();
            return Ok(());
        }
        let Some(first_lane) = mask.first_active() else {
            return Ok(());
        };
        let expected_value = expected_arrivals[first_lane];
        if mask
            .into_iter()
            .any(|lane| expected_value != expected_arrivals[lane])
        {
            return Err(EngineError::message(
                "mbarrier.init expected arrivals must be warp-uniform",
            ));
        }
        let plan = plan_physical_mbarrier_init(&self.context, pointer, mask, expected_value)?
            .with_layout(layout_v1);
        let operation = self.checked_effect_operation(operation, OperationKind::Barrier)?;
        let effect = OperationEffect::MbarrierInit(&plan);
        self.before_effect(operation, effect)?;
        plan.apply(&self.kernel.services().mbarriers())?;
        if let Some(operation) = operation {
            // Own the init-fence coverage set engine-side, off the same
            // (barrier, lane) pairing every observer sees in this effect.
            let fences = self.kernel.services().mbarrier_init_fences();
            let warp_id = operation.id().global_warp_id();
            for (&barrier_id, lane) in plan.barrier_ids().iter().zip(operation.active_mask()) {
                fences.record_init(barrier_id, warp_id, lane);
            }
        }
        self.after_effect(operation, effect)?;
        self.record_engine_progress();
        Ok(())
    }

    /// Execute one `fence.mbarrier_init`, publishing the barrier set it covers.
    ///
    /// The covered set is the engine's own: every barrier this warp initialized
    /// from a lane in the fence's active mask and has not already fenced.
    pub(crate) fn mbarrier_init_fence(
        &self,
        operation: Option<&OperationContext>,
    ) -> Result<(), EngineError> {
        let Some(operation) =
            self.checked_effect_operation(operation, OperationKind::MbarrierInitFence)?
        else {
            return Ok(());
        };
        let barrier_ids = self
            .kernel
            .services()
            .mbarrier_init_fences()
            .take_fenced(operation.id().global_warp_id(), operation.active_mask());
        self.after_effect(
            Some(operation),
            OperationEffect::MbarrierInitFence {
                barrier_ids: &barrier_ids,
            },
        )
    }

    /// Resolve and apply one `mbarrier.arrive[.expect_tx]` with exact lane operands.
    pub(crate) fn mbarrier_arrive<const NO_COMPLETE: bool, const DROP: bool>(
        &self,
        operation: Option<&OperationContext>,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        target_ctas: Option<&WarpValue<i64>>,
        arrival_counts: Option<&WarpValue<i64>>,
        expected_transactions: Option<&WarpValue<i64>>,
        release: bool,
    ) -> Result<WarpValue<u64>, EngineError> {
        if NO_COMPLETE && !release {
            return Err(EngineError::message(
                "mbarrier.noComplete requires release semantics",
            ));
        }
        let plan = plan_physical_mbarrier_arrive_lanes(
            &self.context,
            pointer,
            mask,
            target_ctas,
            arrival_counts,
            expected_transactions,
        )?
        .with_semantics(DROP, release);
        let mut states = WarpValue::splat(0_u64);
        if plan.is_empty() {
            return Ok(states);
        }
        if NO_COMPLETE {
            self.kernel
                .services()
                .mbarriers()
                .validate_no_complete(plan.entries().iter().map(|entry| {
                    let plan = entry.plan();
                    (plan.barrier_id(), plan.arrival_count())
                }))?;
        }
        if !M::OBSERVES_OPERATIONS {
            debug_assert!(operation.is_none());
            let outcome = plan.apply(&self.kernel.services().mbarriers())?;
            for (entry, outcome) in plan.entries().iter().zip(outcome.outcomes()) {
                outcome.write_states::<NO_COMPLETE>(
                    &mut states,
                    entry.arrival_mask(),
                    arrival_counts,
                )?;
            }
            self.record_engine_progress();
            return Ok(states);
        }
        if plan.entries().len() == 1 {
            let entry = plan.entries()[0];
            let operation = self.checked_effect_operation(operation, OperationKind::Barrier)?;
            let single = entry.plan();
            self.before_effect(
                operation,
                OperationEffect::MbarrierArrive {
                    plan: single,
                    outcome: None,
                },
            )?;
            let outcome =
                single.apply_with_outcome(&self.kernel.services().mbarriers(), |outcome| {
                    self.after_effect(
                        operation,
                        OperationEffect::MbarrierArrive {
                            plan: single,
                            outcome: Some(*outcome),
                        },
                    )
                })?;
            outcome.write_states::<NO_COMPLETE>(
                &mut states,
                entry.arrival_mask(),
                arrival_counts,
            )?;
            self.record_engine_progress();
            return Ok(states);
        }
        let operation = self.checked_effect_operation(operation, OperationKind::Barrier)?;
        self.before_effect(
            operation,
            OperationEffect::MbarrierArriveBatch {
                plan: &plan,
                outcome: None,
            },
        )?;
        let outcome = plan.apply_with_outcome(&self.kernel.services().mbarriers(), |outcome| {
            self.after_effect(
                operation,
                OperationEffect::MbarrierArriveBatch {
                    plan: &plan,
                    outcome: Some(outcome),
                },
            )
        })?;
        for (entry, outcome) in plan.entries().iter().zip(outcome.outcomes()) {
            outcome.write_states::<NO_COMPLETE>(
                &mut states,
                entry.arrival_mask(),
                arrival_counts,
            )?;
        }
        self.record_engine_progress();
        Ok(states)
    }

    /// Apply one standalone `mbarrier.expect_tx` without consuming an arrival.
    pub(crate) fn mbarrier_expect_tx(
        &self,
        operation: Option<&OperationContext>,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        expected_transactions: &WarpValue<i64>,
    ) -> Result<(), EngineError> {
        let plan =
            plan_physical_mbarrier_expect_tx(&self.context, pointer, mask, expected_transactions)?;
        if plan.is_empty() {
            return Ok(());
        }
        if !M::OBSERVES_OPERATIONS {
            debug_assert!(operation.is_none());
            let _ = plan.apply(&self.kernel.services().mbarriers())?;
            self.record_engine_progress();
            return Ok(());
        }
        let operation = self.checked_effect_operation(operation, OperationKind::Barrier)?;
        self.before_effect(
            operation,
            OperationEffect::MbarrierExpectTx {
                plan: &plan,
                outcome: None,
            },
        )?;
        let outcome = plan.apply(&self.kernel.services().mbarriers())?;
        self.after_effect(
            operation,
            OperationEffect::MbarrierExpectTx {
                plan: &plan,
                outcome: Some(&outcome),
            },
        )?;
        self.record_engine_progress();
        Ok(())
    }

    /// Execute one nonblocking readiness test. Every successful query observes
    /// the completed generation; only acquire/default forms acquire its memory.
    pub(crate) fn mbarrier_test_wait_instruction<const CONDITIONAL: bool>(
        &self,
        operation: Option<&OperationContext>,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        phases: &WarpValue<i64>,
        acquire_on_true: bool,
    ) -> Result<(WarpValue<u32>, WarpValue<bool>), EngineError> {
        let plans = plan_physical_mbarrier_wait_lanes(&self.context, pointer, mask, phases)?;
        let requests = plans
            .iter()
            .map(|entry| {
                (
                    entry.plan().with_conditional_phase(CONDITIONAL),
                    entry.wait_mask(),
                    entry.plan().requested_phase(),
                )
            })
            .collect::<Vec<_>>();
        let kind = if CONDITIONAL {
            crate::hardware_barriers::MbarrierQuery::ConditionalParity
        } else {
            crate::hardware_barriers::MbarrierQuery::PrimaryParity
        };
        self.mbarrier_query_instruction(operation, &requests, kind, acquire_on_true)
    }

    pub(crate) fn mbarrier_test_wait_state_instruction(
        &self,
        operation: Option<&OperationContext>,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        states: &WarpValue<u64>,
        acquire_on_true: bool,
    ) -> Result<(WarpValue<u32>, WarpValue<bool>), EngineError> {
        let mut requests = BTreeMap::new();
        for lane in mask {
            let lane_mask = WarpMask::from_bits(1 << lane);
            let Some((plan, generation)) =
                plan_physical_mbarrier_state_wait_lanes(&self.context, pointer, lane_mask, states)?
            else {
                continue;
            };
            let entry = requests.entry((plan.barrier_id(), generation)).or_insert((
                plan,
                WarpMask::EMPTY,
                generation,
            ));
            entry.1 = entry.1.union(lane_mask);
        }
        self.mbarrier_query_instruction(
            operation,
            &requests.into_values().collect::<Vec<_>>(),
            crate::hardware_barriers::MbarrierQuery::PrimaryState,
            acquire_on_true,
        )
    }

    /// One physical snapshot owns all query outputs and the completion witness.
    fn mbarrier_query_instruction(
        &self,
        operation: Option<&OperationContext>,
        requests: &[(crate::runtime::PhysicalMbarrierWaitPlan, WarpMask, u64)],
        kind: crate::hardware_barriers::MbarrierQuery,
        acquire_on_true: bool,
    ) -> Result<(WarpValue<u32>, WarpValue<bool>), EngineError> {
        let operation = self.checked_effect_operation(operation, OperationKind::Barrier)?;
        let physical_requests = requests
            .iter()
            .map(|(plan, _, phase)| (plan.barrier_id(), *phase))
            .collect::<Vec<_>>();
        self.kernel
            .services()
            .mbarriers()
            .query_many_with(&physical_requests, kind, |outcomes| {
                let mut ready_values = WarpValue::splat(0_u32);
                let mut reports = WarpValue::splat(false);
                for (&(plan, mask, _), (ready, generation, report)) in requests.iter().zip(outcomes)
                {
                    for lane in mask {
                        ready_values[lane] = u32::from(ready);
                        reports[lane] = report;
                    }
                    if ready {
                        if let Some(operation) = operation {
                            let plan = plan.with_acquire(acquire_on_true);
                            let lane_operation = operation.clone().with_active_mask(mask);
                            let outcome =
                                Some(crate::runtime::PhysicalMbarrierWaitOutcome::new(generation));
                            self.before_effect(
                                Some(&lane_operation),
                                OperationEffect::MbarrierWait { plan, outcome },
                            )?;
                            self.after_effect(
                                Some(&lane_operation),
                                OperationEffect::MbarrierWait { plan, outcome },
                            )?;
                        }
                    }
                }
                Ok((ready_values, reports))
            })
    }

    /// Resolve, await, and publish one `mbarrier.try_wait` with exact lane phases.
    pub(crate) async fn mbarrier_wait(
        &self,
        operation: Option<&OperationContext>,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        requested_phases: &WarpValue<i64>,
    ) -> Result<(), EngineError> {
        if !M::OBSERVES_OPERATIONS {
            debug_assert!(operation.is_none());
            wait_physical_mbarrier(
                &self.context,
                pointer,
                mask,
                requested_phases,
                &self.kernel.services().mbarriers(),
            )
            .await?;
            self.record_engine_progress();
            return Ok(());
        }
        let plans =
            plan_physical_mbarrier_wait_lanes(&self.context, pointer, mask, requested_phases)?;
        if plans.is_empty() {
            return Ok(());
        }
        let operation = self
            .checked_effect_operation(operation, OperationKind::Barrier)?
            .ok_or_else(|| EngineError::message("mbarrier wait requires an operation context"))?;
        for entry in plans.iter().copied() {
            let plan = entry.plan();
            let lane_operation = operation.clone().with_active_mask(entry.wait_mask());
            self.before_effect(
                Some(&lane_operation),
                OperationEffect::MbarrierWait {
                    plan,
                    outcome: None,
                },
            )?;
            let outcome = plan
                .apply_with_operation(&self.kernel.services().mbarriers(), Some(operation.id()))
                .await?;
            self.after_effect(
                Some(&lane_operation),
                OperationEffect::MbarrierWait {
                    plan,
                    outcome: Some(outcome),
                },
            )?;
        }
        self.record_engine_progress();
        Ok(())
    }

    /// Resolve, register, block, and resume one exact CTA-local named barrier.
    pub(crate) async fn named_barrier_sync_with_alignment(
        &self,
        operation: Option<&OperationContext>,
        barrier_id: i64,
        expected_arrivals: i64,
        mask: WarpMask,
        aligned: bool,
    ) -> Result<(), EngineError> {
        self.named_barrier_sync_effect(operation, barrier_id, expected_arrivals, mask, aligned)
            .await?;
        Ok(())
    }

    pub(crate) async fn named_barrier_sync_effect(
        &self,
        operation: Option<&OperationContext>,
        barrier_id: i64,
        expected_arrivals: i64,
        mask: WarpMask,
        aligned: bool,
    ) -> Result<Option<NamedBarrierSyncResumePlan>, EngineError> {
        let Some(plan) = plan_named_barrier_sync_with_alignment(
            &self.context,
            barrier_id,
            expected_arrivals,
            mask,
            aligned,
        )?
        else {
            return Ok(None);
        };
        self.named_barrier_sync_plan_effect(operation, plan).await
    }

    async fn named_barrier_sync_plan_effect(
        &self,
        operation: Option<&OperationContext>,
        plan: crate::runtime::NamedBarrierSyncPlan,
    ) -> Result<Option<NamedBarrierSyncResumePlan>, EngineError> {
        let operation = self.checked_effect_operation(operation, OperationKind::Collective)?;
        if let Some(operation) = operation {
            if operation.active_mask() != plan.arrival_mask() {
                return Err(EngineError::message(format!(
                    "named barrier operation mask {:#010x} does not match resolved mask {:#010x}",
                    operation.active_mask().bits(),
                    plan.arrival_mask().bits(),
                )));
            }
        }
        if plan.aligned() {
            validate_aligned_named_barrier_mask(operation, plan.arrival_mask(), "bar.sync")?;
        }

        let staged = OperationEffect::NamedBarrierSyncRegister {
            plan,
            outcome: None,
        };
        self.before_effect(operation, staged)?;
        let registration = plan.register(&self.kernel.services().named_barriers())?;
        let accumulated_arrival_mask = registration.accumulated_arrival_mask()?;
        // `bar.sync` has no frontend-only "warpgroup" bit. Infer the
        // setmaxnreg synchronization while every participant registers: the
        // hub credits a group only after its exact four full warps report the
        // same barrier generation. Doing this before suspension makes the
        // completed generation visible before any released warp continues.
        if plan.aligned() {
            let completed = self.kernel.services().setmaxnreg().record_warpgroup_sync(
                self.context,
                plan.barrier_id(),
                registration.outcome().generation(),
                plan.arrival_mask(),
            )?;
            if !M::OBSERVES_OPERATIONS && completed {
                self.kernel
                    .services()
                    .ordering()
                    .setmaxnreg_warpgroup_sync(self.context, 4);
            }
        }
        let committed = OperationEffect::NamedBarrierSyncRegister {
            plan,
            outcome: Some(registration.outcome()),
        };
        self.after_effect(operation, committed)?;
        self.record_engine_progress();

        // `barrier.sync` permits one warp to reach the same full-CTA barrier
        // through disjoint divergent paths.  A warp-unit executor must let an
        // early partial path return so the remaining lanes can register.  The
        // path that completes this warp's full mask performs the actual wait
        // and one recombined acquire for all of its lanes.
        let full_cta_arrivals = u64::try_from(self.context.topology().warps_per_cta())
            .ok()
            .and_then(|warps| warps.checked_mul(crate::WARP_SIZE as u64))
            .ok_or_else(|| EngineError::message("CTA named-barrier size overflow"))?;
        let recombine_unaligned_full_cta =
            !plan.aligned() && plan.expected_arrivals() == full_cta_arrivals;
        if recombine_unaligned_full_cta && accumulated_arrival_mask != WarpMask::FULL {
            return Ok(None);
        }

        let resume = if recombine_unaligned_full_cta {
            registration
                .resume_recombined(accumulated_arrival_mask)
                .await?
        } else {
            registration.resume().await?
        };
        let resumed = OperationEffect::NamedBarrierSyncResume(resume);
        self.before_effect(operation, resumed)?;
        self.after_effect(operation, resumed)?;
        self.record_engine_progress();
        Ok(Some(resume))
    }

    /// Resolve and apply one exact nonblocking CTA-local `bar.arrive`.
    pub(crate) fn named_barrier_arrive(
        &self,
        operation: Option<&OperationContext>,
        barrier_id: i64,
        expected_arrivals: i64,
        mask: WarpMask,
    ) -> Result<(), EngineError> {
        let Some(plan) =
            plan_named_barrier_arrive(&self.context, barrier_id, expected_arrivals, mask)?
        else {
            return Ok(());
        };
        let operation = self.checked_effect_operation(operation, OperationKind::Collective)?;
        if let Some(operation) = operation {
            if operation.active_mask() != plan.arrival_mask() {
                return Err(EngineError::message(format!(
                    "named barrier operation mask {:#010x} does not match resolved mask {:#010x}",
                    operation.active_mask().bits(),
                    plan.arrival_mask().bits(),
                )));
            }
        }
        validate_aligned_named_barrier_mask(operation, plan.arrival_mask(), "bar.arrive")?;

        let staged = OperationEffect::NamedBarrierArrive {
            plan,
            outcome: None,
        };
        self.before_effect(operation, staged)?;
        let outcome = plan.apply(&self.kernel.services().named_barriers())?;
        let committed = OperationEffect::NamedBarrierArrive {
            plan,
            outcome: Some(outcome),
        };
        self.after_effect(operation, committed)?;
        self.record_engine_progress();
        Ok(())
    }

    /// Resolve and apply one exact nonblocking cluster-barrier arrival.
    pub(crate) fn cluster_barrier_arrive(
        &self,
        operation: Option<&OperationContext>,
        mask: WarpMask,
        publishes_memory: bool,
        aligned: bool,
    ) -> Result<(), EngineError> {
        if !aligned
            && !mask.is_full()
            && M::observes_analysis_gap(
                self.kernel.mode_state(),
                AnalysisGapKind::ClusterBarrierUnaligned,
            )
        {
            self.record_analysis_gap(operation, AnalysisGapKind::ClusterBarrierUnaligned, None)?;
            return Err(EngineError::analysis_incomplete(
                AnalysisGapKind::ClusterBarrierUnaligned.name(),
            ));
        }
        let arrival_semantics = if publishes_memory {
            ClusterBarrierArrivalSemantics::Release
        } else {
            ClusterBarrierArrivalSemantics::Relaxed
        };
        let plan = plan_cluster_barrier_arrive(
            self.kernel.kernel_index(),
            &self.context.with_active_mask(mask),
            arrival_semantics,
            aligned,
        )?;
        let operation = self.checked_effect_operation(operation, OperationKind::Collective)?;
        let staged = OperationEffect::ClusterBarrierArrive {
            plan: &plan,
            outcome: None,
        };
        self.before_effect(operation, staged)?;
        let outcome = plan.apply(&self.kernel.services().cluster_barriers())?;
        let committed = OperationEffect::ClusterBarrierArrive {
            plan: &plan,
            outcome: Some(outcome),
        };
        self.after_effect(operation, committed)?;
        self.record_engine_progress();
        Ok(())
    }

    /// Register, publish, block, and resume one split cluster-barrier wait.
    pub(crate) async fn cluster_barrier_wait(
        &self,
        operation: Option<&OperationContext>,
        mask: WarpMask,
        aligned: bool,
    ) -> Result<(), EngineError> {
        if !aligned
            && !mask.is_full()
            && M::observes_analysis_gap(
                self.kernel.mode_state(),
                AnalysisGapKind::ClusterBarrierUnaligned,
            )
        {
            self.record_analysis_gap(operation, AnalysisGapKind::ClusterBarrierUnaligned, None)?;
            return Err(EngineError::analysis_incomplete(
                AnalysisGapKind::ClusterBarrierUnaligned.name(),
            ));
        }
        let plan = plan_cluster_barrier_wait(
            self.kernel.kernel_index(),
            &self.context.with_active_mask(mask),
            ClusterBarrierWaitSemantics::Acquire,
            aligned,
        )?;
        let operation = self.checked_effect_operation(operation, OperationKind::Collective)?;
        let staged = OperationEffect::ClusterBarrierWaitRegister {
            plan: &plan,
            outcome: None,
        };
        self.before_effect(operation, staged)?;
        let registration = plan.register(&self.kernel.services().cluster_barriers())?;
        let committed = OperationEffect::ClusterBarrierWaitRegister {
            plan: &plan,
            outcome: Some(registration.outcome()),
        };
        self.after_effect(operation, committed)?;
        self.record_engine_progress();

        let resume = registration.resume().await?;
        let resumed = OperationEffect::ClusterBarrierWaitResume(&resume);
        self.before_effect(operation, resumed)?;
        self.after_effect(operation, resumed)?;
        self.record_engine_progress();
        Ok(())
    }

    /// Allocate TMEM through the shared lifecycle engine and publish exact
    /// register/resume effects around the potentially blocking runtime action.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn tcgen_allocate(
        &self,
        operation: Option<&OperationContext>,
        columns: usize,
        cta_group: usize,
        exclusive: bool,
    ) -> Result<TcgenAllocation, EngineError> {
        let (operation, static_op_id, loop_iteration_path) =
            self.required_operation_identity(operation, OperationKind::Lifecycle)?;
        let context = self.operation_warp_context(operation);
        let plan = plan_tcgen_allocate(
            self.kernel.kernel_index(),
            static_op_id,
            loop_iteration_path,
            context,
            columns,
            cta_group,
        )?
        .with_column_capacity(self.kernel.services().tcgen().column_capacity(), exclusive)?;
        let resume = self.tcgen_lifecycle(Some(operation), plan).await?;
        resume
            .allocation()
            .ok_or_else(|| EngineError::message("tcgen05.alloc completed without an allocation"))
    }

    /// Execute the complete `tcgen05.alloc` instruction, including its
    /// architected shared-memory destination write. The two existing checker
    /// transactions are created here rather than by generated code.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn tcgen_alloc_instruction(
        &mut self,
        context: WarpContext,
        site: u64,
        destination: &PhysicalPtr,
        columns: usize,
        cta_group: usize,
        exclusive: bool,
    ) -> Result<(), EngineError> {
        require_full_warp_sync(context.active_mask(), "tcgen05.alloc.sync.aligned")?;
        let lifecycle =
            self.begin_optional_operation(context, site, OperationKind::Lifecycle, true)?;
        let allocation = self
            .tcgen_allocate(lifecycle.as_ref(), columns, cta_group, exclusive)
            .await?;
        self.finish_optional_operation(&lifecycle)?;

        let lane = context
            .active_mask()
            .first_active()
            .ok_or_else(|| EngineError::message("tcgen05.alloc has no destination lane"))?;
        let store_mask = WarpMask::from_bits(1_u32 << lane);
        let store_context = context.with_active_mask(store_mask);
        let store =
            self.begin_optional_operation(store_context, site, OperationKind::Store, false)?;
        let destination = destination.with_byte_storage_access_width(4)?;
        let physical = self.kernel.physical().clone();
        self.physical_pointer_access(
            store.as_ref(),
            OperationKind::Store,
            &destination,
            None,
            store_mask,
            4,
            false,
            false,
            MemoryAccessSemantics::plain(),
            || {
                store_physical_ptr_u32(
                    &physical,
                    &context,
                    &destination,
                    store_mask,
                    allocation.base_column,
                )
            },
        )?;
        self.finish_optional_operation(&store)?;

        // `.sync.aligned` is mandatory on tcgen05.alloc.  Its shared-memory
        // destination is written by one lane, but every lane in the issuing
        // warp may consume the allocated address after the instruction
        // completes.  Publish that lane frontier through a distinct
        // collective operation so Synccheck keeps the lifecycle transition
        // and Racecheck observes the architected warp rendezvous.
        let sync = self.begin_optional_operation(context, site, OperationKind::Collective, true)?;
        self.rendezvous_sync(sync.as_ref(), "warp", "tcgen05.alloc.sync.aligned")
            .await?;
        self.finish_optional_operation(&sync)
    }

    /// Deallocate one exact live TMEM interval through the shared lifecycle engine.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn tcgen_deallocate(
        &self,
        operation: Option<&OperationContext>,
        address: u32,
        columns: usize,
        cta_group: usize,
        exclusive: bool,
    ) -> Result<(), EngineError> {
        let (operation, static_op_id, loop_iteration_path) =
            self.required_operation_identity(operation, OperationKind::Lifecycle)?;
        let context = self.operation_warp_context(operation);
        let plan = plan_tcgen_deallocate(
            self.kernel.kernel_index(),
            static_op_id,
            loop_iteration_path,
            context,
            address,
            columns,
            cta_group,
        )?
        .with_column_capacity(self.kernel.services().tcgen().column_capacity(), exclusive)?;
        self.tcgen_lifecycle(Some(operation), plan).await?;
        Ok(())
    }

    /// Relinquish the CTA allocation permit through the shared lifecycle engine.
    pub(crate) async fn tcgen_relinquish(
        &self,
        operation: Option<&OperationContext>,
        cta_group: usize,
    ) -> Result<(), EngineError> {
        let (operation, static_op_id, loop_iteration_path) =
            self.required_operation_identity(operation, OperationKind::Lifecycle)?;
        let context = self.operation_warp_context(operation);
        let plan = plan_tcgen_relinquish(
            self.kernel.kernel_index(),
            static_op_id,
            loop_iteration_path,
            context,
            cta_group,
        )?;
        self.tcgen_lifecycle(Some(operation), plan).await?;
        Ok(())
    }

    async fn tcgen_lifecycle(
        &self,
        operation: Option<&OperationContext>,
        plan: TcgenLifecyclePlan,
    ) -> Result<TcgenLifecycleResumePlan, EngineError> {
        let operation = self.checked_effect_operation(operation, OperationKind::Lifecycle)?;
        if let Some(operation) = operation {
            if operation.active_mask() != plan.context().active_mask() {
                return Err(EngineError::message(format!(
                    "tcgen lifecycle operation mask {:#010x} does not match resolved mask {:#010x}",
                    operation.active_mask().bits(),
                    plan.context().active_mask().bits(),
                )));
            }
        }
        self.before_effect(operation, OperationEffect::TcgenLifecycleRegister(&plan))?;
        let registration = plan.register(&self.kernel.services().tcgen())?;
        self.after_effect(operation, OperationEffect::TcgenLifecycleRegister(&plan))?;
        self.record_engine_progress();

        let resume = registration.resume().await?;
        self.before_effect(operation, OperationEffect::TcgenLifecycleResume(&resume))?;
        self.after_effect(operation, OperationEffect::TcgenLifecycleResume(&resume))?;
        self.record_engine_progress();
        Ok(resume)
    }

    /// Execute one warpgroup-wide dynamic register-limit request through the
    /// shared ordinal-keyed protocol hub.
    pub(crate) async fn setmaxnreg<const CALLING_INITIAL_COUNT: u32>(
        &self,
        operation: Option<&OperationContext>,
        increase: bool,
        count: u32,
    ) -> Result<(), EngineError> {
        if M::OBSERVES_OPERATIONS {
            self.kernel
                .services()
                .setmaxnreg()
                .configure_calling_initial_count(i64::from(CALLING_INITIAL_COUNT))?;
        }
        self.setmaxnreg_instruction(operation, increase, count)
            .await
    }

    /// Execute one `setmaxnreg` after launch metadata has configured the
    /// caller's initial register allocation. The v2 instruction path is the
    /// only generated-artifact caller.
    pub(crate) async fn setmaxnreg_instruction(
        &self,
        operation: Option<&OperationContext>,
        increase: bool,
        count: u32,
    ) -> Result<(), EngineError> {
        let (operation, static_op_id, loop_iteration_path) =
            self.required_operation_identity(operation, OperationKind::Collective)?;
        let context = self.operation_warp_context(operation);
        require_full_warp_sync(context.active_mask(), "ptx.setmaxnreg")?;
        if !M::OBSERVES_OPERATIONS {
            let occurrence = self.kernel.services().ordering().setmaxnreg_arrive(
                static_op_id,
                loop_iteration_path,
                context,
                SETMAXNREG_WARPS_PER_GROUP,
                increase,
                count,
            )?;
            self.kernel
                .services()
                .rendezvous()
                .warpgroup(
                    0,
                    "ptx.setmaxnreg",
                    [i64::try_from(occurrence)
                        .map_err(|_| EngineError::message("setmaxnreg occurrence exceeds i64"))?],
                    context,
                    SETMAXNREG_WARPS_PER_GROUP,
                )?
                .await?;
            self.kernel.services().ordering().setmaxnreg_complete(
                context,
                SETMAXNREG_WARPS_PER_GROUP,
                occurrence,
            )?;
            self.record_engine_progress();
            return Ok(());
        }
        let action = if increase {
            SetmaxnregAction::Increase
        } else {
            SetmaxnregAction::Decrease
        };
        let operation = Some(operation);
        let contextualize =
            |error: EngineError| error.with_operation_context(operation.expect("required above"));
        let hub = self.kernel.services().setmaxnreg();
        let plan = hub
            .plan(
                self.kernel.kernel_index(),
                operation,
                context,
                action,
                i64::from(count),
            )
            .map_err(EngineError::from)
            .map_err(&contextualize)?;
        self.before_effect(operation, OperationEffect::SetmaxnregRegister(&plan))
            .map_err(&contextualize)?;
        let registration = plan
            .register(&hub)
            .map_err(EngineError::from)
            .map_err(&contextualize)?;
        self.after_effect(operation, OperationEffect::SetmaxnregRegister(&plan))
            .map_err(&contextualize)?;
        self.record_engine_progress();

        let resume = registration
            .resume()
            .await
            .map_err(EngineError::from)
            .map_err(&contextualize)?;
        self.before_effect(operation, OperationEffect::SetmaxnregResume(&resume))
            .map_err(&contextualize)?;
        self.after_effect(operation, OperationEffect::SetmaxnregResume(&resume))
            .map_err(&contextualize)?;
        self.record_engine_progress();
        Ok(())
    }

    /// Resolve one completion target per issuing lane, expanding each lane's
    /// multicast CTA mask into one target barrier per selected CTA.
    ///
    /// `.multicast` may issue from several lanes at once with per-lane byte
    /// counts, which no single-target `issue_plan` can express.
    fn lane_resolved_mbarrier_completion_plan(
        &self,
        barrier_pointer: &PhysicalPtr,
        issue_mask: WarpMask,
        multicast_cta_masks: Option<&WarpValue<i64>>,
        transactions_per_lane: &WarpValue<u64>,
    ) -> Result<PhysicalMbarrierCompletionIssuePlan, EngineError> {
        let mut completions = Vec::new();
        for lane in issue_mask {
            let lane_mask = WarpMask::from_bits(1_u32 << lane);
            if let Some(masks) = multicast_cta_masks {
                let cta_mask = u64::try_from(masks[lane])
                    .map_err(|_| EngineError::message("negative mbarrier completion CTA mask"))?;
                crate::runtime::io::validate_tma_multicast_mask(
                    cta_mask,
                    self.context.topology().ctas_per_cluster(),
                )?;
                for target_cta in 0..self.context.topology().ctas_per_cluster() {
                    if cta_mask & (1_u64 << target_cta) == 0 {
                        continue;
                    }
                    completions.push((
                        barrier_pointer.resolve_shared_barrier_multicast(
                            &self.context,
                            lane_mask,
                            target_cta,
                        )?,
                        transactions_per_lane[lane],
                    ));
                }
            } else {
                completions.push((
                    barrier_pointer.resolve_shared_barrier(&self.context, lane_mask, None)?,
                    transactions_per_lane[lane],
                ));
            }
        }
        Ok(PhysicalMbarrierCompletionIssuePlan::from_completions(
            completions,
        ))
    }

    /// Publish one fully resolved asynchronous payload completion issue.
    pub(crate) fn mbarrier_completion_issue(
        &self,
        operation: Option<&OperationContext>,
        barrier_pointer: &PhysicalPtr,
        issue_mask: WarpMask,
        source_barrier: Option<PhysicalBarrierId>,
        cta_group: i64,
        multicast_cta_masks: Option<&WarpValue<i64>>,
        transactions_per_lane: &WarpValue<u64>,
        payload: Option<(&PhysicalPtr, [u8; 16])>,
    ) -> Result<(), EngineError> {
        let plan = if let Some(source_barrier) = source_barrier {
            if issue_mask.len() != 1 {
                return Err(EngineError::message(
                    "TMA mbarrier completion requires exactly one issuing lane",
                ));
            }
            let lane = issue_mask
                .first_active()
                .ok_or_else(|| EngineError::message("mbarrier completion mask is empty"))?;
            let cta_mask = match multicast_cta_masks {
                Some(masks) => u64::try_from(masks[lane])
                    .map_err(|_| EngineError::message("negative mbarrier completion CTA mask"))?,
                None => 0,
            };
            let targets = resolve_mbarrier_completion_targets(
                &self.context,
                barrier_pointer,
                issue_mask,
                source_barrier,
                cta_group,
                cta_mask,
                multicast_cta_masks.is_some(),
            )?;
            targets.issue_plan(transactions_per_lane[lane])
        } else {
            if cta_group != 1 {
                return Err(EngineError::message(format!(
                    "lane-resolved mbarrier completion requires cta_group 1, got {cta_group}"
                )));
            }
            let mut completions = Vec::new();
            for lane in issue_mask {
                let lane_mask = WarpMask::from_bits(1_u32 << lane);
                if let Some(masks) = multicast_cta_masks {
                    let cta_mask = u64::try_from(masks[lane]).map_err(|_| {
                        EngineError::message("negative mbarrier completion CTA mask")
                    })?;
                    crate::runtime::io::validate_tma_multicast_mask(
                        cta_mask,
                        self.context.topology().ctas_per_cluster(),
                    )?;
                    for target_cta in 0..self.context.topology().ctas_per_cluster() {
                        if cta_mask & (1_u64 << target_cta) == 0 {
                            continue;
                        }
                        completions.push((
                            barrier_pointer.resolve_shared_barrier_multicast(
                                &self.context,
                                lane_mask,
                                target_cta,
                            )?,
                            transactions_per_lane[lane],
                        ));
                    }
                } else {
                    completions.push((
                        barrier_pointer.resolve_shared_barrier(&self.context, lane_mask, None)?,
                        transactions_per_lane[lane],
                    ));
                }
            }
            PhysicalMbarrierCompletionIssuePlan::from_completions(completions)
        };
        let operation = self.checked_effect_operation(operation, OperationKind::AsyncIssue)?;
        if let Some((payload_destination, payload_bytes)) = payload {
            let operation = operation.ok_or_else(|| {
                EngineError::message(
                    "mbarrier completion payload requires a stable operation context",
                )
            })?;
            return self.mbarrier_completion_payload_issue(
                operation,
                issue_mask,
                multicast_cta_masks,
                payload_destination,
                payload_bytes,
                plan,
            );
        }
        if !M::OBSERVES_OPERATIONS {
            plan.complete_numeric(&self.kernel.services().mbarriers(), 0)?;
            self.record_engine_progress();
            return Ok(());
        }
        let staged = OperationEffect::MbarrierCompletionIssue {
            plan: &plan,
            action_ids: None,
        };
        self.before_effect(operation, staged)?;
        let action_ids = plan.apply(&self.kernel.services().mbarriers())?;
        let committed = OperationEffect::MbarrierCompletionIssue {
            plan: &plan,
            action_ids: Some(&action_ids),
        };
        self.after_effect(operation, committed)?;
        self.record_engine_progress();
        Ok(())
    }

    /// Apply one explicit `mbarrier.complete_tx` instruction immediately.
    ///
    /// This is deliberately separate from `mbarrier_completion_issue`: TMA,
    /// CLC, and TCGEN enqueue a completion that a later scheduler actor
    /// performs, whereas the explicit PTX instruction performs the transaction
    /// credit before it retires.  Both paths reuse the same physical plan and
    /// checker completion callbacks.
    pub(crate) fn mbarrier_complete_tx(
        &self,
        operation: Option<&OperationContext>,
        barrier_pointer: &PhysicalPtr,
        issue_mask: WarpMask,
        multicast_cta_masks: Option<&WarpValue<i64>>,
        transactions_per_lane: &WarpValue<u64>,
    ) -> Result<(), EngineError> {
        let mut completions = Vec::new();
        for lane in issue_mask {
            let transactions = transactions_per_lane[lane];
            if transactions == 0 {
                continue;
            }
            let lane_mask = WarpMask::from_bits(1_u32 << lane);
            if let Some(masks) = multicast_cta_masks {
                let cta_mask = u64::try_from(masks[lane])
                    .map_err(|_| EngineError::message("negative mbarrier completion CTA mask"))?;
                crate::runtime::io::validate_tma_multicast_mask(
                    cta_mask,
                    self.context.topology().ctas_per_cluster(),
                )?;
                for target_cta in 0..self.context.topology().ctas_per_cluster() {
                    if cta_mask & (1_u64 << target_cta) != 0 {
                        completions.push((
                            barrier_pointer.resolve_shared_barrier_multicast(
                                &self.context,
                                lane_mask,
                                target_cta,
                            )?,
                            transactions,
                        ));
                    }
                }
            } else {
                completions.push((
                    barrier_pointer.resolve_shared_barrier(&self.context, lane_mask, None)?,
                    transactions,
                ));
            }
        }
        let plan =
            PhysicalMbarrierCompletionIssuePlan::from_completions(completions).counter_only();
        let operation = self.required_operation(operation, OperationKind::AsyncIssue)?;
        let staged = OperationEffect::MbarrierCompletionIssue {
            plan: &plan,
            action_ids: None,
        };
        self.before_effect(Some(operation), staged)?;
        let mbarriers = self.kernel.services().mbarriers();
        let action_ids = plan.apply(&mbarriers)?;
        let committed = OperationEffect::MbarrierCompletionIssue {
            plan: &plan,
            action_ids: Some(&action_ids),
        };

        // Call the mode hook directly instead of `self.after_effect`: the
        // latter publishes IDs to the asynchronous completion pump. These
        // actions belong to the executing instruction and must remain private
        // until they are committed below.
        if let Err(error) = M::after_effect(self.kernel.mode_state(), operation, committed) {
            mbarriers.discard_unpublished_transaction_completions(&action_ids);
            return Err(error);
        }

        let actions = mbarriers.queued_completion_actions(&action_ids);
        for action in &actions {
            if let Err(error) = M::before_completion(
                self.kernel.mode_state(),
                CompletionActionEffect::PhysicalMbarrier(action),
            ) {
                mbarriers.discard_unpublished_transaction_completions(&action_ids);
                return Err(error);
            }
        }
        let result = mbarriers.apply_completion_batch_detailed_with_outcomes(
            &action_ids,
            || Ok(()),
            |outcomes| {
                for outcome in outcomes {
                    M::after_completion(
                        self.kernel.mode_state(),
                        CompletionEffect::PhysicalMbarrier(outcome),
                    )?;
                }
                Ok(())
            },
        );
        match result {
            Ok(_) => {
                self.record_engine_progress();
                Ok(())
            }
            Err(error) => {
                mbarriers.discard_unpublished_transaction_completions(&action_ids);
                Err(EngineError::message(error.to_string()))
            }
        }
    }

    fn mbarrier_completion_payload_issue(
        &self,
        operation: &OperationContext,
        issue_mask: WarpMask,
        multicast_cta_masks: Option<&WarpValue<i64>>,
        destination: &PhysicalPtr,
        completion_bytes: [u8; 16],
        completion_plan: PhysicalMbarrierCompletionIssuePlan,
    ) -> Result<(), EngineError> {
        if issue_mask.len() != 1 {
            return Err(EngineError::message(
                "mbarrier completion payload requires exactly one issuing lane",
            ));
        }
        destination.require_ptx_space_for_mask(PtxStateSpace::Shared, issue_mask)?;
        let lane = issue_mask
            .first_active()
            .expect("one-lane payload issue has an active lane");
        let byte_offset = destination.lane_write_byte_offset(lane, 16)?;
        let target_ctas = if let Some(masks) = multicast_cta_masks {
            let mask = u64::try_from(masks[lane])
                .map_err(|_| EngineError::message("negative completion payload CTA mask"))?;
            crate::runtime::io::validate_tma_multicast_mask(
                mask,
                self.context.topology().ctas_per_cluster(),
            )?;
            (0..self.context.topology().ctas_per_cluster())
                .filter(|&target| mask & (1_u64 << target) != 0)
                .collect::<Vec<_>>()
        } else {
            vec![self.context.cta_id_in_cluster()]
        };

        if !M::OBSERVES_OPERATIONS {
            for &target_cta in &target_ctas {
                write_shared_runtime_bytes_to_cta(
                    self.kernel.physical(),
                    &self.context,
                    destination.buffer(),
                    lane,
                    target_cta,
                    byte_offset,
                    &completion_bytes,
                )
                .map_err(|error| error.with_operation_context(operation))?;
            }
            let delivered_bytes = u64::try_from(target_ctas.len())
                .ok()
                .and_then(|count| count.checked_mul(completion_bytes.len() as u64))
                .ok_or_else(|| EngineError::message("completion payload byte count overflow"))?;
            completion_plan
                .complete_numeric(&self.kernel.services().mbarriers(), delivered_bytes)?;
            self.record_engine_progress();
            return Ok(());
        }

        let payload_operation =
            OperationContext::new(operation.id().clone(), operation.kind(), issue_mask)
                .with_control_provenance(operation.control_provenance());
        let mut accesses = Vec::new();
        for &target_cta in &target_ctas {
            if M::OBSERVES_OPERATIONS && M::resolves_async_accesses(self.kernel.mode_state()) {
                let resolved = resolve_shared_runtime_physical_access_to_cta(
                    &self.context,
                    destination.buffer(),
                    lane,
                    target_cta,
                    byte_offset,
                    16,
                    PhysicalAccessKind::Write,
                )?;
                let descriptor = PhysicalAccessDescriptor::new(
                    PhysicalAccessKind::Write,
                    PhysicalAccessSpace::Shared,
                    16,
                )
                .map_err(|error| EngineError::message(error.to_string()))?
                .with_proxy_memory_domain(async_completion_payload_proxy_domain(
                    self.context.cta_id_in_cluster(),
                    target_cta,
                ));
                let span = resolved.span();
                accesses.push(
                    PhysicalAccessBatch::resolve_single_span(
                        payload_operation.clone(),
                        descriptor,
                        |_| Ok::<_, EngineError>(span),
                    )
                    .map_err(|error| EngineError::message(error.to_string()))?,
                );
            }
        }

        let mut payload =
            AsyncPayloadEffect::new(payload_operation.clone(), accesses, completion_plan)
                .map_err(|error| EngineError::message(error.to_string()))?;
        self.before_effect(
            Some(&payload_operation),
            OperationEffect::AsyncPayload(&payload),
        )?;
        for &target_cta in &target_ctas {
            write_shared_runtime_bytes_to_cta(
                self.kernel.physical(),
                &self.context,
                destination.buffer(),
                lane,
                target_cta,
                byte_offset,
                &completion_bytes,
            )
            .map_err(|error| error.with_operation_context(&payload_operation))?;
        }
        let payload_byte_len = u64::try_from(target_ctas.len())
            .ok()
            .and_then(|count| count.checked_mul(completion_bytes.len() as u64))
            .ok_or_else(|| EngineError::message("completion payload byte count overflow"))?;
        let action_ids = self
            .kernel
            .services()
            .deferred_payloads()
            .enqueue_payload(&payload, payload_byte_len)?;
        payload
            .bind_completion_action_ids(action_ids)
            .map_err(|error| EngineError::message(error.to_string()))?;
        self.after_effect(
            Some(&payload_operation),
            OperationEffect::AsyncPayload(&payload),
        )?;
        self.record_engine_progress();
        Ok(())
    }

    fn tcgen_runtime_access_footprint(
        &self,
        operation: &OperationContext,
        access_operation_kind: OperationKind,
        buffer: &RuntimeBuffer,
        logical_buffer: Option<&str>,
        lane_accesses: &[(usize, Option<usize>, usize, usize)],
    ) -> Result<PhysicalAccessBatch, EngineError> {
        let kind = physical_access_kind(access_operation_kind)?;
        let space = runtime_buffer_physical_space_for_mask(buffer, operation.active_mask())?
            .ok_or_else(|| EngineError::message("TCGEN footprint has no active lanes"))?;
        let mut builder =
            TcgenAccessFootprintBuilder::new(operation.clone(), kind, space, logical_buffer)?;
        for &(execution_lane, target_cta, byte_offset, byte_len) in lane_accesses {
            let resolved = match target_cta {
                Some(target_cta) => resolve_runtime_physical_access_at_cta(
                    &self.context,
                    buffer,
                    target_cta,
                    execution_lane,
                    byte_offset,
                    byte_len,
                    kind,
                )?,
                None => resolve_runtime_physical_access(
                    &self.context,
                    buffer,
                    execution_lane,
                    byte_offset,
                    byte_len,
                    kind,
                )?,
            };
            builder.record_span(execution_lane, resolved.span())?;
        }
        let batch = builder.finish()?;
        if M::OBSERVES_PROXY_MEMORY_DOMAINS {
            let proxy_domain = runtime_buffer_proxy_memory_domain_for_space_and_mask(
                buffer,
                space,
                operation.active_mask(),
            )?;
            Ok(batch.with_proxy_memory_domain(proxy_domain))
        } else {
            Ok(batch)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn tcgen_tmem_access_footprint(
        &self,
        operation: &OperationContext,
        access_operation_kind: OperationKind,
        buffer: &RuntimeBuffer,
        logical_buffer: &str,
        access_mode: TmemAccessMode,
        lane_accesses: &[(usize, usize, Option<usize>, i64, i64, i64, usize)],
    ) -> Result<PhysicalAccessBatch, EngineError> {
        let kind = physical_access_kind(access_operation_kind)?;
        let space = runtime_buffer_physical_space_for_mask(buffer, operation.active_mask())?
            .ok_or_else(|| EngineError::message("TCGEN footprint has no active lanes"))?;
        let mut builder =
            TcgenAccessFootprintBuilder::new(operation.clone(), kind, space, Some(logical_buffer))?;
        let lifecycle = self.kernel.services().tcgen();
        let mut validator =
            crate::runtime::tmem::TmemLifecycleValidator::new(&lifecycle, access_mode);
        for &(
            provenance_lane,
            execution_lane,
            target_cta,
            mapped_lane,
            tcol_element,
            allocated_addr,
            access_bytes,
        ) in lane_accesses
        {
            let span = match target_cta {
                Some(target_cta) => crate::runtime::tmem::resolve_tmem_physical_span_at_cta_cached(
                    &self.context,
                    &mut validator,
                    buffer,
                    target_cta,
                    mapped_lane,
                    tcol_element,
                    allocated_addr,
                    access_bytes,
                    execution_lane,
                )?,
                None => resolve_tmem_physical_span(
                    &self.context,
                    &lifecycle,
                    access_mode,
                    buffer,
                    mapped_lane,
                    tcol_element,
                    allocated_addr,
                    access_bytes,
                    execution_lane,
                )?,
            };
            builder.record_span(provenance_lane, span)?;
        }
        builder.finish()
    }

    /// Execute one TCGEN instruction in every engine mode. Numeric mode runs
    /// only the numeric effect; checker modes additionally resolve and record
    /// the instruction's physical footprint.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tcgen_instruction_issue<R>(
        &self,
        operation: Option<&OperationContext>,
        cta_group: u32,
        pipeline_operation: TcgenPipelineOperation,
        mma_pipeline_class: Option<TcgenMmaPipelineClass>,
        resolve_accesses: impl FnOnce(
            &mut dyn FnMut(
                bool,
                OperationKind,
                &RuntimeBuffer,
                Option<&str>,
                &[(usize, Option<usize>, usize, usize)],
            ) -> Result<(), EngineError>,
            &mut dyn FnMut(
                OperationKind,
                &RuntimeBuffer,
                &str,
                TmemAccessMode,
                &[(usize, usize, Option<usize>, i64, i64, i64, usize)],
            ) -> Result<(), EngineError>,
        ) -> Result<(), EngineError>,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        if !M::OBSERVES_OPERATIONS {
            let result = numeric_effect()?;
            // NumSim skips the ordering/checker path below, so publish a
            // completed TMEM write here. A TCGEN load changes only the
            // issuing warp's private registers and cannot release another
            // warp's poll.
            if matches!(
                pipeline_operation.work_kind(),
                TcgenWorkKind::Commit | TcgenWorkKind::Store
            ) {
                self.record_engine_progress();
            }
            return Ok(result);
        }
        self.tcgen_work_issue_internal(
            operation,
            cta_group,
            pipeline_operation,
            mma_pipeline_class,
            None,
            resolve_accesses,
            numeric_effect,
        )
    }

    /// [`Self::tcgen_instruction_issue`] for an instruction whose footprints
    /// are a pure function of `issue_memo`: repeats reuse the resolved
    /// footprints instead of decoding and resolving them again.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tcgen_instruction_issue_memoized<R>(
        &self,
        issue_memo: Option<TcgenIssueMemoSeed>,
        operation: Option<&OperationContext>,
        cta_group: u32,
        pipeline_operation: TcgenPipelineOperation,
        mma_pipeline_class: Option<TcgenMmaPipelineClass>,
        resolve_accesses: impl FnOnce(
            &mut dyn FnMut(
                bool,
                OperationKind,
                &RuntimeBuffer,
                Option<&str>,
                &[(usize, Option<usize>, usize, usize)],
            ) -> Result<(), EngineError>,
            &mut dyn FnMut(
                OperationKind,
                &RuntimeBuffer,
                &str,
                TmemAccessMode,
                &[(usize, usize, Option<usize>, i64, i64, i64, usize)],
            ) -> Result<(), EngineError>,
        ) -> Result<(), EngineError>,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        if !M::OBSERVES_OPERATIONS {
            let result = numeric_effect()?;
            if matches!(
                pipeline_operation.work_kind(),
                TcgenWorkKind::Commit | TcgenWorkKind::Store
            ) {
                self.record_engine_progress();
            }
            return Ok(result);
        }
        self.tcgen_work_issue_internal(
            operation,
            cta_group,
            pipeline_operation,
            mma_pipeline_class,
            issue_memo,
            resolve_accesses,
            numeric_effect,
        )
    }

    #[inline(never)]
    pub(crate) fn tcgen_work_issue_internal<R>(
        &self,
        operation: Option<&OperationContext>,
        cta_group: u32,
        pipeline_operation: TcgenPipelineOperation,
        mma_pipeline_class: Option<TcgenMmaPipelineClass>,
        issue_memo: Option<TcgenIssueMemoSeed>,
        resolve_accesses: impl FnOnce(
            &mut dyn FnMut(
                bool,
                OperationKind,
                &RuntimeBuffer,
                Option<&str>,
                &[(usize, Option<usize>, usize, usize)],
            ) -> Result<(), EngineError>,
            &mut dyn FnMut(
                OperationKind,
                &RuntimeBuffer,
                &str,
                TmemAccessMode,
                &[(usize, usize, Option<usize>, i64, i64, i64, usize)],
            ) -> Result<(), EngineError>,
        ) -> Result<(), EngineError>,
        numeric_effect: impl FnOnce() -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        let operation = self
            .checked_effect_operation(operation, OperationKind::TcgenWork)?
            .ok_or_else(|| {
                EngineError::message("TCGEN work issue requires a stable operation context")
            })?;
        let (accesses, shared_a_reads) = if M::observes_tcgen_accesses(self.kernel.mode_state()) {
            let _profile_timer = ProfileTimer::new(ProfileKind::TcgenResolveAccesses);
            let memo_key = issue_memo.as_ref().map(|seed| {
                seed.key(
                    operation.active_mask().bits(),
                    self.kernel.services().tcgen().generation(),
                )
            });
            let memoized = memo_key.as_ref().and_then(|key| {
                self.tcgen_issue_memo
                    .lock()
                    .expect("tcgen issue memo poisoned")
                    .get(key)
            });
            if let Some(entry) = memoized {
                (
                    entry
                        .batches
                        .iter()
                        .map(|template| template.instantiate(operation))
                        .collect::<Result<Vec<_>, _>>()?,
                    entry
                        .shared_a_reads
                        .iter()
                        .map(|template| template.instantiate(operation))
                        .collect::<Result<Vec<_>, _>>()?,
                )
            } else {
                let accesses = RefCell::new(Vec::new());
                let shared_a_reads = RefCell::new(Vec::new());
                let mut record_runtime =
                    |mma_shared_a: bool,
                     access_kind: OperationKind,
                     buffer: &RuntimeBuffer,
                     logical_buffer: Option<&str>,
                     lane_accesses: &[(usize, Option<usize>, usize, usize)]|
                     -> Result<(), EngineError> {
                        if mma_shared_a && pipeline_operation != TcgenPipelineOperation::Mma {
                            return Err(EngineError::message("only MMA can have a shared-A read"));
                        }
                        (if mma_shared_a {
                            &shared_a_reads
                        } else {
                            &accesses
                        })
                        .borrow_mut()
                        .push(self.tcgen_runtime_access_footprint(
                            operation,
                            access_kind,
                            buffer,
                            logical_buffer,
                            lane_accesses,
                        )?);
                        Ok(())
                    };
                let mut record_tmem =
                    |access_kind: OperationKind,
                     buffer: &RuntimeBuffer,
                     logical_buffer: &str,
                     access_mode: TmemAccessMode,
                     lane_accesses: &[(usize, usize, Option<usize>, i64, i64, i64, usize)]|
                     -> Result<(), EngineError> {
                        accesses.borrow_mut().push(self.tcgen_tmem_access_footprint(
                            operation,
                            access_kind,
                            buffer,
                            logical_buffer,
                            access_mode,
                            lane_accesses,
                        )?);
                        Ok(())
                    };
                resolve_accesses(&mut record_runtime, &mut record_tmem)
                    .map_err(|error| error.with_operation_context(operation))?;
                let accesses = accesses.into_inner();
                let shared_a_reads = shared_a_reads.into_inner();
                if let (Some(key), Some(seed)) = (memo_key, issue_memo) {
                    let entry = TcgenIssueMemoEntry {
                        batches: accesses
                            .iter()
                            .map(TcgenBatchTemplate::from_batch)
                            .collect(),
                        shared_a_reads: shared_a_reads
                            .iter()
                            .map(TcgenBatchTemplate::from_batch)
                            .collect(),
                        _pins: seed.pins,
                    };
                    self.tcgen_issue_memo
                        .lock()
                        .expect("tcgen issue memo poisoned")
                        .insert(key, entry);
                }
                (accesses, shared_a_reads)
            }
        } else {
            (Vec::new(), Vec::new())
        };
        let shared_a_issue = if shared_a_reads.is_empty() {
            None
        } else {
            Some(TcgenWorkIssue::new(
                operation.clone(),
                cta_group,
                TcgenPipelineOperation::MmaSharedARead,
                None,
                shared_a_reads,
            )?)
        };
        let issue = TcgenWorkIssue::new(
            operation.clone(),
            cta_group,
            pipeline_operation,
            mma_pipeline_class,
            accesses,
        )?;
        {
            let _profile_timer = ProfileTimer::new(ProfileKind::TcgenBeforeEffect);
            self.before_effect(Some(operation), OperationEffect::TcgenWorkIssue(&issue))?;
            if let Some(issue) = &shared_a_issue {
                self.before_effect(Some(operation), OperationEffect::TcgenWorkIssue(issue))?;
            }
        }
        let result = {
            let _profile_timer = ProfileTimer::new(ProfileKind::TcgenNumericEffect);
            numeric_effect().map_err(|error| error.with_operation_context(operation))?
        };
        self.kernel
            .services()
            .ordering()
            .commit_tcgen_work_issue(&issue)?;
        {
            let _profile_timer = ProfileTimer::new(ProfileKind::TcgenAfterEffect);
            self.after_effect(Some(operation), OperationEffect::TcgenWorkIssue(&issue))?;
            if let Some(issue) = &shared_a_issue {
                self.kernel
                    .services()
                    .ordering()
                    .commit_tcgen_work_issue(issue)?;
                self.after_effect(Some(operation), OperationEffect::TcgenWorkIssue(issue))?;
            }
        }
        self.record_engine_progress();
        Ok(result)
    }

    /// Issue one deferred `tcgen05.commit.mbarrier::arrive::one` action and
    /// atomically drain the exact issuer-thread TCGEN work set it tracks.
    pub(crate) fn tcgen_commit_issue_with_work(
        &self,
        operation: Option<&OperationContext>,
        pointer: &PhysicalPtr,
        cta_group: u32,
        multicast_cta_masks: Option<&WarpValue<i64>>,
        shared_a_only: bool,
    ) -> Result<(), EngineError> {
        let operation = self.required_operation(operation, OperationKind::AsyncIssue)?;
        let context = self.operation_warp_context(operation);
        let issue_mask = operation.active_mask();
        let work = self.kernel.services().ordering().plan_tcgen_commit(
            context,
            issue_mask,
            cta_group,
            shared_a_only,
        )?;
        let Some(plan) =
            plan_tcgen_commit_issue(&context, pointer, issue_mask, multicast_cta_masks)?
        else {
            return Ok(());
        };
        if M::OBSERVES_OPERATIONS {
            self.before_effect(
                Some(operation),
                OperationEffect::TcgenCommitIssue {
                    plan: &plan,
                    work: &work,
                    actions: None,
                },
            )?;
        }
        if !M::OBSERVES_OPERATIONS {
            self.kernel
                .services()
                .ordering()
                .commit_tcgen_work_set(&work)?;
            plan.complete_numeric(&self.kernel.services().mbarriers())?;
            self.record_engine_progress();
            return Ok(());
        }
        let actions = plan.apply(&self.kernel.services().mbarriers())?;
        self.kernel
            .services()
            .ordering()
            .commit_tcgen_work_set(&work)?;
        if M::OBSERVES_OPERATIONS {
            self.after_effect(
                Some(operation),
                OperationEffect::TcgenCommitIssue {
                    plan: &plan,
                    work: &work,
                    actions: Some(&actions),
                },
            )?;
        }
        self.record_engine_progress();
        Ok(())
    }

    /// Complete all prior TCGEN LD/ST work issued by this warp and publish the
    /// exact drained token set to analysis modes.
    pub(crate) fn tcgen_wait_work_internal(
        &self,
        operation: Option<&OperationContext>,
        kind: TcgenTransferKind,
    ) -> Result<(), EngineError> {
        let operation = self.checked_effect_operation(operation, OperationKind::Control)?;
        let work = self
            .kernel
            .services()
            .ordering()
            .plan_tcgen_wait(self.context, kind)?;
        if M::OBSERVES_OPERATIONS {
            self.before_effect(operation, OperationEffect::TcgenWait { work: &work })?;
        }
        self.kernel
            .services()
            .ordering()
            .commit_tcgen_work_set(&work)?;
        if M::OBSERVES_OPERATIONS {
            self.after_effect(operation, OperationEffect::TcgenWait { work: &work })?;
        }
        self.record_engine_progress();
        Ok(())
    }

    /// Apply one exact asynchronous payload and publish one composite effect.
    ///
    /// In observed modes every payload address is resolved first, then the mode
    /// validates the complete access/completion transaction before numerical
    /// memory changes. The numerical payload runs once, the mbarrier completion
    /// batch is enqueued transactionally, and only then does analysis commit the
    /// same effect with its exact action IDs bound.
    ///
    /// This does not roll back numerical memory if the runtime completion enqueue
    /// itself fails after the payload. Callers must surface that execution error;
    /// no analysis commit is published for the failed issue.
    ///
    /// Issue one classic global-to-shared copy without exposing its separate
    /// footprint-planning and numerical execution phases.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn raw_cp_async_issue(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        source: &PhysicalPtr,
        source_sizes: &WarpValue<u32>,
        issue_mask: WarpMask,
        byte_count: usize,
    ) -> Result<(), EngineError> {
        let operation = self.required_operation(operation, OperationKind::AsyncIssue)?;
        if !M::OBSERVES_OPERATIONS {
            copy_physical_ptr_bytes(
                self.kernel.physical(),
                context,
                destination,
                source,
                source_sizes,
                issue_mask,
                byte_count,
            )?;
            self.kernel.services().async_groups().issue(
                context,
                AsyncGroupDomain::CpAsync,
                issue_mask,
            )?;
            self.record_engine_progress();
            return Ok(());
        }

        let mut lane_accesses = Vec::with_capacity(issue_mask.len());
        for lane in issue_mask {
            let lane_mask = WarpMask::from_lanes([lane])
                .map_err(|error| EngineError::message(error.to_string()))?;
            let lane_operation = operation.clone().with_active_mask(lane_mask);
            let (source_accesses, destination_accesses) =
                if M::resolves_async_accesses(self.kernel.mode_state()) {
                    plan_cp_async_physical_ptr_lane_accesses(
                        &lane_operation,
                        context,
                        destination,
                        source,
                        lane,
                        source_sizes[lane] as usize,
                        byte_count,
                    )?
                } else {
                    (Vec::new(), Vec::new())
                };
            lane_accesses.push((lane, source_accesses, destination_accesses));
        }
        let effect = AsyncGroupIssueBatchEffect::new(
            operation.clone(),
            AsyncGroupDomain::CpAsync,
            lane_accesses,
        )?;
        let participates_in_global_memory = M::USES_GLOBAL_MEMORY_TRANSACTION
            && effect.members().iter().any(|member| {
                member
                    .source_accesses()
                    .iter()
                    .chain(member.destination_accesses())
                    .any(|batch| batch.descriptor().space().has_read_from_versions())
            });
        let _profile = participates_in_global_memory
            .then(|| ProfileTimer::new(ProfileKind::RaceGlobalTransactionAsyncIssue));
        let member_batches = || {
            effect.members().iter().flat_map(|member| {
                member
                    .source_accesses()
                    .iter()
                    .chain(member.destination_accesses())
            })
        };
        let transaction_spans = if participates_in_global_memory {
            crate::physical_access::global_batch_spans(member_batches())
        } else {
            Vec::new()
        };
        let _transaction = begin_global_memory_transaction::<M>(
            self.kernel.mode_state(),
            participates_in_global_memory,
            crate::physical_access::global_transaction_exclusive(member_batches()),
            &transaction_spans,
        )?;
        self.before_effect(
            Some(operation),
            OperationEffect::AsyncGroupIssueBatch(&effect),
        )?;
        copy_physical_ptr_bytes(
            self.kernel.physical(),
            context,
            destination,
            source,
            source_sizes,
            issue_mask,
            byte_count,
        )?;
        self.kernel
            .services()
            .async_groups()
            .issue_exact_batch(context, &effect, Vec::new())?;
        self.after_effect(
            Some(operation),
            OperationEffect::AsyncGroupIssueBatch(&effect),
        )?;
        self.record_engine_progress();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn raw_bulk_copy_g2s_issue(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        source: &PhysicalPtr,
        num_bytes: i64,
        mask: WarpMask,
        destination_space: PtxStateSpace,
        barrier_pointer: &PhysicalPtr,
        source_barrier: PhysicalBarrierId,
        report_pattern: u32,
        scope: Option<crate::MemoryScope>,
    ) -> Result<(), EngineError> {
        if report_pattern != 0
            && !self
                .kernel
                .services()
                .mbarriers()
                .check_layout(source_barrier, 1)?
        {
            return Err(EngineError::message(
                "copy reporting requires mbarrier layout::v1",
            ));
        }
        let completion_targets = resolve_mbarrier_completion_targets(
            context,
            barrier_pointer,
            mask,
            source_barrier,
            1,
            0,
            false,
        )?;
        self.async_payload_issue_with_resolved_targets_and_semantics(
            operation,
            &completion_targets,
            crate::runtime::bulk_copy_memory_semantics(scope),
            |operation| {
                let (accesses, delivered) = plan_raw_bulk_copy_g2s_accesses(
                    operation,
                    context,
                    destination,
                    source,
                    num_bytes,
                    mask,
                    destination_space,
                    scope,
                )?;
                Ok((accesses, delivered))
            },
            || {
                let (delivered, reported) = execute_raw_bulk_copy_g2s(
                    self.kernel.physical(),
                    context,
                    destination,
                    source,
                    num_bytes,
                    mask,
                    destination_space,
                    report_pattern,
                )?;
                if report_pattern != 0 {
                    self.kernel
                        .services()
                        .mbarriers()
                        .report_on(source_barrier, reported)?;
                }
                Ok(((), delivered))
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn raw_bulk_copy_s2s_issue(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        source: &PhysicalPtr,
        num_bytes: i64,
        mask: WarpMask,
        barrier_pointer: &PhysicalPtr,
        source_barrier: PhysicalBarrierId,
        scope: Option<crate::MemoryScope>,
    ) -> Result<(), EngineError> {
        self.validate_remote_shared_completion(context, destination, barrier_pointer, mask)?;
        let targets = resolve_mbarrier_completion_targets(
            context,
            barrier_pointer,
            mask,
            source_barrier,
            1,
            0,
            false,
        )?;
        self.async_payload_issue_with_resolved_targets_and_semantics(
            operation,
            &targets,
            crate::runtime::bulk_copy_memory_semantics(scope),
            |operation| {
                let (mut accesses, delivered) = plan_raw_bulk_copy_s2s_accesses(
                    operation,
                    context,
                    destination,
                    source,
                    num_bytes,
                    mask,
                    None,
                )?;
                crate::runtime::scope_bulk_copy_accesses(accesses.iter_mut(), scope)?;
                Ok((accesses, delivered))
            },
            || {
                let delivered = execute_raw_bulk_copy_s2s(
                    self.kernel.physical(),
                    context,
                    destination,
                    source,
                    num_bytes,
                    mask,
                )?;
                Ok(((), delivered))
            },
        )
    }

    pub(crate) fn raw_bulk_reduce_s2c_issue<T: crate::runtime::RawAtomicScalar>(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        source: &PhysicalPtr,
        num_bytes: i64,
        mask: WarpMask,
        barrier: &PhysicalPtr,
        reduction: crate::runtime::RawAtomicOperation,
        scope: crate::MemoryScope,
    ) -> Result<(), EngineError> {
        self.validate_remote_shared_completion(context, destination, barrier, mask)?;
        let id = barrier.resolve_shared_barrier(context, mask, None)?;
        let targets = resolve_mbarrier_completion_targets(context, barrier, mask, id, 1, 0, false)?;
        self.async_payload_issue_with_resolved_targets_and_semantics(
            operation,
            &targets,
            MemoryAccessSemantics::async_reduction_at(scope),
            |operation| {
                plan_raw_bulk_copy_s2s_accesses(
                    operation,
                    context,
                    destination,
                    source,
                    num_bytes,
                    mask,
                    Some(T::BYTE_LEN),
                )
            },
            || {
                crate::runtime::execute_raw_bulk_reduce_s2c::<T>(
                    self.kernel.physical(),
                    context,
                    destination,
                    source,
                    num_bytes,
                    mask,
                    reduction,
                )
                .map(|bytes| ((), bytes))
            },
        )
    }

    /// Issue one `.multicast` global-to-shared bulk copy and publish its exact
    /// per-CTA footprint.
    ///
    /// The runtime issues each thread separately. The multicast planner keeps
    /// that thread's CTA mask and replicates its exact footprint and completion
    /// byte count to each selected target, without joining other issuers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn raw_bulk_copy_g2s_multicast_issue(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        source: &PhysicalPtr,
        num_bytes: &WarpValue<i64>,
        issue_mask: WarpMask,
        barrier_pointer: &PhysicalPtr,
        cta_masks: &WarpValue<i64>,
        report_pattern: u32,
        scope: Option<crate::MemoryScope>,
    ) -> Result<(), EngineError> {
        let label = DiagnosticLabel::new("cp.async.bulk.g2s.cluster.multicast");
        let plan_completions = |delivered: &WarpValue<u64>| {
            self.lane_resolved_mbarrier_completion_plan(
                barrier_pointer,
                issue_mask,
                Some(cta_masks),
                delivered,
            )
        };
        if report_pattern != 0 {
            for &(id, _) in plan_completions(&WarpValue::splat(0))?.completions() {
                if !self.kernel.services().mbarriers().check_layout(id, 1)? {
                    return Err(EngineError::message(
                        "copy reporting requires mbarrier layout::v1",
                    ));
                }
            }
        }
        self.async_payload_issue_with_resolved_plan(
            operation,
            crate::runtime::bulk_copy_memory_semantics(scope),
            |operation| {
                let (reads, writes, delivered) = plan_raw_bulk_copy_lane_varying_accesses(
                    operation,
                    context,
                    destination,
                    source,
                    num_bytes,
                    issue_mask,
                    PtxStateSpace::SharedCluster,
                    PtxStateSpace::Global,
                    &label,
                    RawBulkCopyFootprintShape {
                        multicast_cta_masks: Some(cta_masks),
                        ..RawBulkCopyFootprintShape::default()
                    },
                )?;
                let plan = plan_completions(&delivered)?;
                let payload_byte_len = delivered.lanes().iter().copied().max().unwrap_or(0);
                let mut accesses = reads.into_iter().chain(writes).collect::<Vec<_>>();
                crate::runtime::scope_bulk_copy_accesses(accesses.iter_mut(), scope)?;
                Ok((accesses, plan, payload_byte_len))
            },
            || {
                let (delivered, reported) = raw_bulk_copy_g2s_multicast(
                    self.kernel.physical(),
                    context,
                    destination,
                    source,
                    num_bytes,
                    issue_mask,
                    cta_masks,
                    report_pattern,
                )?;
                if report_pattern != 0 {
                    for lane in issue_mask {
                        let lane_targets = self.lane_resolved_mbarrier_completion_plan(
                            barrier_pointer,
                            WarpMask::from_bits(1 << lane),
                            Some(cta_masks),
                            &delivered,
                        )?;
                        for &(id, _) in lane_targets.completions() {
                            self.kernel
                                .services()
                                .mbarriers()
                                .report_on(id, reported[lane])?;
                        }
                    }
                }
                let plan = plan_completions(&delivered)?;
                let payload_byte_len = delivered.lanes().iter().copied().max().unwrap_or(0);
                Ok(((), plan, payload_byte_len))
            },
        )
    }

    /// Issue one `.ignore_oob` global-to-shared bulk copy and publish its exact
    /// footprint: the in-bounds source slice and the whole destination window.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn raw_bulk_copy_g2s_ignore_oob_issue(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        source: &PhysicalPtr,
        num_bytes: &WarpValue<i64>,
        ignore_left: &WarpValue<i64>,
        ignore_right: &WarpValue<i64>,
        mask: WarpMask,
        barrier_pointer: &PhysicalPtr,
        source_barrier: PhysicalBarrierId,
        scope: Option<crate::MemoryScope>,
    ) -> Result<(), EngineError> {
        let targets = resolve_mbarrier_completion_targets(
            context,
            barrier_pointer,
            mask,
            source_barrier,
            1,
            0,
            false,
        )?;
        self.async_payload_issue_with_resolved_targets_and_semantics(
            operation,
            &targets,
            crate::runtime::bulk_copy_memory_semantics(scope),
            |operation| {
                let (mut accesses, delivered) = plan_raw_bulk_copy_g2s_ignore_oob_accesses(
                    operation,
                    context,
                    destination,
                    source,
                    num_bytes,
                    ignore_left,
                    ignore_right,
                    mask,
                )?;
                crate::runtime::scope_bulk_copy_accesses(accesses.iter_mut(), scope)?;
                Ok((accesses, delivered))
            },
            || {
                let delivered = raw_bulk_copy_g2s_cta_ignore_oob(
                    self.kernel.physical(),
                    context,
                    destination,
                    source,
                    num_bytes,
                    ignore_left,
                    ignore_right,
                    mask,
                )?;
                let lane = mask.first_active().ok_or_else(|| {
                    EngineError::message("ignore_oob bulk copy has no issuing lane")
                })?;
                Ok(((), delivered[lane]))
            },
        )
    }

    /// Masked S2G form.
    ///
    /// `.cp_mask` selects which bytes of each 16-byte group reach global
    /// memory, so the published destination write covers exactly the selected
    /// byte runs while the shared source read covers the whole window.
    ///
    /// The instruction executor supplies one issuing thread per transaction,
    /// preserving that thread's byte mask and bulk-group completion.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn raw_bulk_copy_s2g_masked_issue(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        source: &PhysicalPtr,
        num_bytes: &WarpValue<i64>,
        issue_mask: WarpMask,
        byte_masks: &WarpValue<i64>,
        scope: Option<crate::MemoryScope>,
    ) -> Result<(), EngineError> {
        self.async_group_issue_with_resolved_accesses(
            operation,
            AsyncGroupDomain::Bulk,
            true,
            issue_mask,
            |operation| {
                let (mut reads, mut writes, _delivered) = plan_raw_bulk_copy_lane_varying_accesses(
                    operation,
                    context,
                    destination,
                    source,
                    num_bytes,
                    issue_mask,
                    PtxStateSpace::Global,
                    PtxStateSpace::SharedCta,
                    &DiagnosticLabel::new("cp.async.bulk.s2g.cp_mask"),
                    RawBulkCopyFootprintShape {
                        byte_masks: Some(byte_masks),
                        ..RawBulkCopyFootprintShape::default()
                    },
                )?;
                crate::runtime::scope_bulk_copy_accesses(
                    reads.iter_mut().chain(writes.iter_mut()),
                    scope,
                )?;
                Ok((reads, writes))
            },
            || {
                let writes = raw_bulk_copy_s2g_masked(
                    self.kernel.physical(),
                    context,
                    destination,
                    source,
                    num_bytes,
                    issue_mask,
                    byte_masks,
                )?;
                Ok(((), writes))
            },
        )
    }

    /// Issue one 16-byte register-to-remote-shared asynchronous store.
    ///
    /// Each active lane selects one CTA rank and contributes four packed u32
    /// values. The destination write uses PTX generic-async proxy semantics,
    /// and the mapped mbarrier receives exactly 16 transaction bytes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn st_async_cluster_u32x4_issue(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        barrier_pointer: &PhysicalPtr,
        target_ranks: &WarpValue<i64>,
        values: &[&WarpValue<u32>; 4],
        issue_mask: WarpMask,
    ) -> Result<(), EngineError> {
        if issue_mask.is_empty() {
            return Err(EngineError::message(
                "remote asynchronous store requires at least one issuing lane",
            ));
        }
        let mapped_destination = destination.map_shared_rank(context, target_ranks, issue_mask)?;
        let mapped_barrier = barrier_pointer.map_shared_rank(context, target_ranks, issue_mask)?;
        self.st_async_cluster_mapped_words_issue(
            operation,
            context,
            &mapped_destination,
            &mapped_barrier,
            values,
            issue_mask,
        )
    }

    /// Issue `st.async` from addresses that have already passed through `mapa`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn st_async_cluster_mapped_words_issue<const WORDS: usize>(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        mapped_destination: &PhysicalPtr,
        mapped_barrier: &PhysicalPtr,
        values: &[&WarpValue<u32>; WORDS],
        issue_mask: WarpMask,
    ) -> Result<(), EngineError> {
        if !matches!(WORDS, 1 | 2 | 4) {
            return Err(EngineError::message("st.async supports 4, 8, or 16 bytes"));
        }
        let byte_len = WORDS * 4;
        self.shared_async_write_issue(
            operation,
            context,
            mapped_destination,
            mapped_barrier,
            byte_len,
            PhysicalAccessKind::Write,
            issue_mask,
            || {
                for lane in issue_mask {
                    let offset = mapped_destination.lane_write_byte_offset(lane, byte_len)?;
                    let mut bytes = [0_u8; 16];
                    for (index, value) in values.iter().enumerate() {
                        let start = index * 4;
                        bytes[start..start + 4].copy_from_slice(&value[lane].to_le_bytes());
                    }
                    write_runtime_bytes(
                        self.kernel.physical(),
                        context,
                        mapped_destination.buffer(),
                        lane,
                        offset,
                        &bytes[..byte_len],
                    )?;
                }
                Ok(())
            },
        )
    }

    fn validate_remote_shared_completion(
        &self,
        context: &WarpContext,
        destination: &PhysicalPtr,
        barrier: &PhysicalPtr,
        mask: WarpMask,
    ) -> Result<(), EngineError> {
        let target = destination.shared_target_cta_ranks(context, mask)?;
        let barrier_target = barrier.shared_target_cta_ranks(context, mask)?;
        for lane in mask {
            if target[lane] != barrier_target[lane]
                || target[lane] == context.cta_id_in_cluster() as i64
            {
                return Err(EngineError::message(
                    "async shared destination and barrier must belong to the same remote CTA",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn red_async_cluster_issue<T: crate::runtime::RawAtomicScalar>(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        barrier: &PhysicalPtr,
        values: &WarpValue<T>,
        mask: WarpMask,
        reduction: crate::runtime::RawAtomicOperation,
    ) -> Result<(), EngineError> {
        self.validate_remote_shared_completion(context, destination, barrier, mask)?;
        let destination = destination.with_pointee_itemsize(T::BYTE_LEN);
        self.shared_async_write_issue(
            operation,
            context,
            &destination,
            barrier,
            T::BYTE_LEN,
            PhysicalAccessKind::AtomicReadModifyWrite,
            mask,
            || {
                crate::runtime::raw_atomic_scalar_physical_ptr_warp(
                    self.kernel.physical(),
                    context,
                    &destination,
                    values,
                    mask,
                    PtxStateSpace::SharedCluster,
                    reduction,
                )?;
                Ok(())
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn shared_async_write_issue(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        mapped_destination: &PhysicalPtr,
        mapped_barrier: &PhysicalPtr,
        byte_len: usize,
        kind: PhysicalAccessKind,
        issue_mask: WarpMask,
        numeric: impl FnOnce() -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        if issue_mask.is_empty() {
            return Err(EngineError::message(
                "remote asynchronous store requires at least one issuing lane",
            ));
        }
        mapped_destination.require_ptx_space_for_mask(PtxStateSpace::SharedCluster, issue_mask)?;

        let mut barrier_ids = Vec::with_capacity(issue_mask.len());
        for lane in issue_mask {
            let lane_mask = WarpMask::from_bits(1_u32 << lane);
            barrier_ids.push(mapped_barrier.resolve_shared_barrier(context, lane_mask, None)?);
        }
        let completion_targets = PhysicalMbarrierCompletionTargets::from_barrier_ids(barrier_ids);

        self.async_payload_issue_with_resolved_targets_and_semantics(
            operation,
            &completion_targets,
            if kind == PhysicalAccessKind::AtomicReadModifyWrite {
                MemoryAccessSemantics::scoped(
                    crate::MemoryOrder::Relaxed,
                    crate::MemoryScope::Cluster,
                    crate::MemoryProxy::Generic,
                    crate::MemoryAccessClass::Reduction,
                )
            } else {
                MemoryAccessSemantics::generic_async()
            },
            |operation| {
                let descriptor =
                    PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Shared, byte_len)
                        .map_err(|error| EngineError::message(error.to_string()))?
                        .with_proxy_memory_domain(ProxyMemoryDomain::SharedCluster);
                let batch = PhysicalAccessBatch::resolve_single_span(
                    operation.clone(),
                    descriptor,
                    |provenance| {
                        let lane = provenance.lane();
                        let offset = mapped_destination.lane_write_byte_offset(lane, byte_len)?;
                        resolve_runtime_physical_access(
                            context,
                            mapped_destination.buffer(),
                            lane,
                            offset,
                            byte_len,
                            kind,
                        )
                        .map(|resolved| resolved.span())
                    },
                )
                .map_err(|error| EngineError::message(error.to_string()))?;
                Ok((vec![batch], byte_len as u64))
            },
            || {
                numeric()?;
                Ok(((), byte_len as u64))
            },
        )
    }

    /// Issue one TensorMap global-to-shared copy without exposing its separate
    /// footprint-planning and numerical execution phases.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn raw_tma_g2c_issue(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        tensor_map: &RuntimeTensorMap,
        origin: &[i64],
        cta_mask: u64,
        multicast: bool,
        barrier_pointer: &PhysicalPtr,
        source_barrier: PhysicalBarrierId,
        cta_group: i64,
        report_pattern: u32,
        im2col: Option<(crate::runtime::tensor_map::Im2colMode, &[i64])>,
    ) -> Result<(), EngineError> {
        self.validate_raw_tma_shared_pointer_alignment(operation, context, destination)?;
        let operation_context = self.required_operation(operation, OperationKind::AsyncIssue)?;
        let targets = resolve_mbarrier_completion_targets(
            context,
            barrier_pointer,
            context.active_mask(),
            source_barrier,
            cta_group,
            cta_mask,
            multicast,
        )?;
        let transfer = if let Some((mode, info)) = im2col {
            crate::runtime::tensor_map::RawTmaG2cTransferPlan::im2col(
                context,
                destination,
                tensor_map,
                origin,
                mode,
                info,
                cta_mask,
                multicast,
            )
        } else {
            crate::runtime::tensor_map::RawTmaG2cTransferPlan::new(
                context,
                destination,
                tensor_map,
                origin,
                cta_mask,
                multicast,
            )
        }
        .map_err(|error| error.with_operation_context(operation_context))?;
        self.issue_tma_g2s_transfer(
            operation,
            context,
            tensor_map,
            &targets,
            &transfer,
            report_pattern,
        )
    }

    fn issue_tma_g2s_transfer(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        tensor_map: &RuntimeTensorMap,
        targets: &PhysicalMbarrierCompletionTargets,
        transfer: &crate::runtime::tensor_map::RawTmaG2cTransferPlan,
        report_pattern: u32,
    ) -> Result<(), EngineError> {
        if report_pattern != 0 {
            for &id in targets.barrier_ids() {
                if !self.kernel.services().mbarriers().check_layout(id, 1)? {
                    return Err(EngineError::message(
                        "copy reporting requires mbarrier layout::v1",
                    ));
                }
            }
        }
        self.async_payload_issue_with_resolved_targets_and_semantics(
            operation,
            targets,
            MemoryAccessSemantics::async_proxy(),
            |operation| {
                let source = if M::compacts_async_accesses(self.kernel.mode_state()) {
                    crate::runtime::TmaSourceAccessPlan::Runs
                } else {
                    crate::runtime::TmaSourceAccessPlan::Units
                };
                transfer.accesses_with(operation, context, tensor_map, source)
            },
            || {
                let (delivery, reported) = transfer.execute_with_report(
                    self.kernel.physical(),
                    context,
                    tensor_map,
                    report_pattern,
                )?;
                if report_pattern != 0 {
                    for &id in targets.barrier_ids() {
                        self.kernel.services().mbarriers().report_on(id, reported)?;
                    }
                }
                Ok(((), delivery.bytes_per_target()))
            },
        )
    }

    /// Reject a raw TMA issue whose concrete shared pointer is not 128B aligned.
    fn validate_raw_tma_shared_pointer_alignment(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        pointer: &PhysicalPtr,
    ) -> Result<(), EngineError> {
        let operation = self.required_operation(operation, OperationKind::AsyncIssue)?;
        let byte_offset =
            pointer.resolve_uniform_shared_address_u32(context, operation.active_mask())?;
        if !byte_offset.is_multiple_of(128) {
            return Err(
                EngineError::tma_shared_address_misaligned(0, i64::from(byte_offset))
                    .with_operation_context(operation),
            );
        }
        Ok(())
    }

    /// Resolve typed-TMA element addresses inside the engine, merge adjacent
    /// shared bytes into payload components, and validate every component
    /// start. Padded and sliced layouts are checked from their concrete
    /// runtime addresses; no compiler alignment plan crosses the ABI.
    pub(crate) fn validate_typed_tma_shared_alignment(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        itemsize: usize,
        shared: &RuntimeBuffer,
        access_kind: PhysicalAccessKind,
        elements: impl FnOnce(
            &mut dyn FnMut(i64, bool, usize) -> Result<(), EngineError>,
        ) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        let operation = self.required_operation(operation, OperationKind::AsyncIssue)?;
        let mut spans = Vec::new();
        elements(&mut |element_index, in_bounds, lane| {
            if !in_bounds {
                return Ok(());
            }
            let element_index = usize::try_from(element_index).map_err(|_| {
                EngineError::out_of_bounds(format!(
                    "TMA shared element index {element_index} is negative on lane {lane}"
                ))
            })?;
            let byte_offset = element_byte_offset(shared, element_index, itemsize, lane)?;
            let access = resolve_runtime_physical_access(
                context,
                shared,
                lane,
                byte_offset,
                itemsize,
                access_kind,
            )?;
            if access.space() != PhysicalAccessSpace::Shared {
                return Err(EngineError::message(
                    "typed TMA alignment validation requires shared memory",
                ));
            }
            spans.push(access.span());
            Ok(())
        })?;

        spans.sort_unstable();
        let mut component_index = 0_usize;
        let mut component: Option<(PhysicalAllocationId, usize)> = None;
        for span in spans {
            let starts_new_component = component.is_none_or(|(allocation, byte_end)| {
                allocation != span.allocation() || span.byte_offset() > byte_end
            });
            if starts_new_component {
                if !span.byte_offset().is_multiple_of(128) {
                    let byte_offset = i64::try_from(span.byte_offset())
                        .map_err(|_| EngineError::message("TMA shared byte offset exceeds i64"))?;
                    return Err(EngineError::tma_shared_address_misaligned(
                        component_index,
                        byte_offset,
                    )
                    .with_operation_context(operation));
                }
                component = Some((span.allocation(), span.byte_end()));
                component_index = component_index
                    .checked_add(1)
                    .ok_or_else(|| EngineError::message("TMA shared component count overflow"))?;
                continue;
            }
            if let Some((_, byte_end)) = &mut component {
                *byte_end = (*byte_end).max(span.byte_end());
            }
        }
        Ok(())
    }

    fn raw_tma_source_barrier(
        context: &WarpContext,
        barrier_pointer: &PhysicalPtr,
        issue_mask: WarpMask,
        cta_group: i64,
    ) -> Result<PhysicalBarrierId, EngineError> {
        let target = match cta_group {
            1 => None,
            2 => Some(context.cta_id_in_cluster() & !1_usize),
            other => {
                return Err(EngineError::message(format!(
                    "TMA cta_group must be 1 or 2, got {other}"
                )));
            }
        };
        barrier_pointer.resolve_shared_barrier(context, issue_mask, target)
    }

    /// Raw TensorMap G2S instruction boundary. Descriptor decoding, barrier
    /// identity, footprint planning, numerical transfer, and completion issue
    /// all remain behind this private core.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn raw_tma_g2c_instruction(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        tensor_map: &RuntimeTensorMap,
        origin: &[i64],
        cta_mask: u64,
        multicast: bool,
        barrier_pointer: &PhysicalPtr,
        cta_group: i64,
        report_pattern: u32,
        im2col: Option<(crate::runtime::tensor_map::Im2colMode, &[i64])>,
    ) -> Result<(), EngineError> {
        let source_barrier = Self::raw_tma_source_barrier(
            context,
            barrier_pointer,
            context.active_mask(),
            cta_group,
        )?;
        self.raw_tma_g2c_issue(
            operation,
            context,
            destination,
            tensor_map,
            origin,
            cta_mask,
            multicast,
            barrier_pointer,
            source_barrier,
            cta_group,
            report_pattern,
            im2col,
        )
    }

    /// Raw TensorMap gather4 instruction boundary.
    ///
    /// Descriptor decoding, footprint planning, the numerical transfer, and the
    /// completion issue all stay behind this core exactly as they do for the
    /// non-gather G2C form; only the row walk differs, and both phases take it
    /// from the shared `raw_tma_gather4_layout`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn raw_tma_gather4_instruction(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        tensor_map: &RuntimeTensorMap,
        column: i64,
        rows: &[i64],
        cta_mask: u64,
        multicast: bool,
        barrier_pointer: &PhysicalPtr,
        cta_group: i64,
        report_pattern: u32,
    ) -> Result<(), EngineError> {
        self.validate_raw_tma_shared_pointer_alignment(operation, context, destination)?;
        let operation_context = self.required_operation(operation, OperationKind::AsyncIssue)?;
        let issue_mask = context.active_mask();
        let source_barrier =
            Self::raw_tma_source_barrier(context, barrier_pointer, issue_mask, cta_group)?;
        let targets = resolve_mbarrier_completion_targets(
            context,
            barrier_pointer,
            issue_mask,
            source_barrier,
            cta_group,
            cta_mask,
            multicast,
        )?;
        let transfer = crate::runtime::tensor_map::RawTmaG2cTransferPlan::gather4(
            context,
            destination,
            tensor_map,
            column,
            rows,
            cta_mask,
            multicast,
        )
        .map_err(|error| error.with_operation_context(operation_context))?;
        self.issue_tma_g2s_transfer(
            operation,
            context,
            tensor_map,
            &targets,
            &transfer,
            report_pattern,
        )
    }

    /// Issue one shared-to-global bulk copy through the existing async-group
    /// transaction without exposing its footprint planner.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn raw_bulk_copy_s2g_issue(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        source: &PhysicalPtr,
        num_bytes: i64,
        mask: WarpMask,
        scope: Option<crate::MemoryScope>,
    ) -> Result<(), EngineError> {
        let num_bytes_warp = WarpValue::splat(num_bytes);
        self.async_group_issue_with_resolved_accesses(
            operation,
            AsyncGroupDomain::Bulk,
            true,
            mask,
            |operation| {
                plan_raw_bulk_copy_s2g_accesses(
                    operation,
                    context,
                    destination,
                    source,
                    num_bytes,
                    mask,
                    scope,
                )
            },
            || {
                let writes = raw_bulk_copy_s2g(
                    self.kernel.physical(),
                    context,
                    destination,
                    source,
                    &num_bytes_warp,
                    mask,
                )?;
                Ok(((), writes))
            },
        )
    }

    /// Capture one shared-to-global reduction at issue time and defer
    /// its atomic global RMWs until the bulk group's full-completion milestone.
    pub(crate) fn raw_bulk_reduce_s2g_issue(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        source: &PhysicalPtr,
        num_bytes: i64,
        mask: WarpMask,
        scope: MemoryScope,
        reduction: crate::DeferredGlobalReduction,
    ) -> Result<(), EngineError> {
        self.async_group_issue_with_resolved_accesses(
            operation,
            AsyncGroupDomain::Bulk,
            true,
            mask,
            |operation| {
                plan_raw_bulk_reduce_s2g_accesses(
                    operation,
                    context,
                    destination,
                    source,
                    num_bytes,
                    mask,
                    scope,
                    reduction,
                )
            },
            || {
                let writes = raw_bulk_reduce_s2g(
                    self.kernel.physical(),
                    context,
                    destination,
                    source,
                    num_bytes,
                    mask,
                    reduction,
                )?;
                Ok(((), writes))
            },
        )
    }

    /// Issue one TensorMap shared-to-global operation through the existing
    /// async-group transaction without exposing its footprint planner.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn raw_tma_s2g_issue(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        source: &PhysicalPtr,
        tensor_map: &RuntimeTensorMap,
        origin: &[i64],
        reduction: Option<RawTmaReductionOp>,
        im2col: Option<crate::runtime::tensor_map::Im2colMode>,
    ) -> Result<(), EngineError> {
        self.validate_raw_tma_shared_pointer_alignment(operation, context, source)?;
        let operation_context = self.required_operation(operation, OperationKind::AsyncIssue)?;
        let plan = if let Some(mode) = im2col {
            RawTmaS2gTransferPlan::im2col(context, source, tensor_map, origin, mode)
        } else {
            RawTmaS2gTransferPlan::new(context, source, tensor_map, origin)
        }
        .map_err(|error| error.with_operation_context(operation_context))?;
        self.async_group_issue_with_resolved_accesses(
            operation,
            AsyncGroupDomain::Bulk,
            true,
            operation.map_or(self.context.active_mask(), OperationContext::active_mask),
            |operation| {
                plan.accesses(
                    operation,
                    context,
                    tensor_map,
                    if reduction.is_some() {
                        PhysicalAccessKind::AtomicReadModifyWrite
                    } else {
                        PhysicalAccessKind::Write
                    },
                )
            },
            || {
                let writes =
                    plan.execute(self.kernel.physical(), context, tensor_map, reduction)?;
                Ok(((), writes))
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_async_payload_batches(
        &self,
        context: &WarpContext,
        issuer_lane: usize,
        itemsize: usize,
        source: &RuntimeBuffer,
        destination: &RuntimeBuffer,
        multicast_cta_mask: Option<u64>,
        remote_cta_id: Option<usize>,
        source_fill: AsyncSourceFill,
        round_to_tf32: bool,
        reduction: Option<DeferredGlobalReduction>,
        element_batches: &mut dyn FnMut(
            &mut dyn FnMut(
                &[(i64, i64, bool, bool); crate::WARP_SIZE],
                usize,
            ) -> Result<(), EngineError>,
        ) -> Result<(), EngineError>,
    ) -> Result<u64, EngineError> {
        let mut transactions_per_target = 0_u64;
        let use_element_batch = multicast_cta_mask.is_none()
            && reduction.is_none()
            && itemsize != 0
            && itemsize <= crate::runtime::MAX_RUNTIME_SCALAR_BYTES
            && matches!(source, RuntimeBuffer::Global(_))
            && supports_shared_runtime_write_session(destination, issuer_lane, remote_cta_id);
        if use_element_batch {
            if !context.active_mask().contains(issuer_lane) {
                return Err(EngineError::message(format!(
                    "copy_async issuing lane {issuer_lane} is not active"
                )));
            }
            let RuntimeBuffer::Global(source_view) = source else {
                unreachable!("element batch requires a global source")
            };
            self.kernel
                .physical()
                .global()
                .with_shared_read_session(source_view, |reader| {
                    with_shared_runtime_write_session(
                        self.kernel.physical(),
                        context,
                        destination,
                        issuer_lane,
                        remote_cta_id,
                        |writer| {
                            let source_element_capacity = reader.byte_len() / itemsize;
                            let destination_element_capacity = writer.byte_len() / itemsize;
                            let mut execute_batch =
                                |elements: &[(i64, i64, bool, bool); crate::WARP_SIZE],
                                 batch_count: usize| {
                                    let copied = execute_async_copy_element_batch_in_write_session(
                                        itemsize,
                                        reader,
                                        source_element_capacity,
                                        elements,
                                        destination_element_capacity,
                                        source_fill,
                                        round_to_tf32,
                                        issuer_lane,
                                        writer,
                                        batch_count,
                                    )?;
                                    transactions_per_target = transactions_per_target
                                        .checked_add(copied)
                                        .ok_or_else(|| {
                                            EngineError::message(
                                                "copy_async delivered-byte count overflow",
                                            )
                                        })?;
                                    Ok(())
                                };
                            element_batches(&mut execute_batch)?;
                            Ok(())
                        },
                    )
                })??;
        } else {
            let mut execute_batch = |elements: &[(i64, i64, bool, bool); crate::WARP_SIZE],
                                     count: usize| {
                for &(source_index, destination_index, source_in_bounds, destination_in_bounds) in
                    &elements[..count]
                {
                    let copied = execute_async_copy_element_at_lane(
                        self.kernel.physical(),
                        context,
                        itemsize,
                        source,
                        source_index,
                        source_in_bounds,
                        destination,
                        destination_index,
                        destination_in_bounds,
                        multicast_cta_mask,
                        remote_cta_id,
                        source_fill,
                        round_to_tf32,
                        reduction,
                        issuer_lane,
                    )?;
                    transactions_per_target =
                        transactions_per_target.checked_add(copied).ok_or_else(|| {
                            EngineError::message("copy_async delivered-byte count overflow")
                        })?;
                }
                Ok(())
            };
            element_batches(&mut execute_batch)?;
        }
        Ok(transactions_per_target)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    pub(crate) fn async_payload_issue_batches(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        barrier_pointer: &PhysicalPtr,
        issue_mask: WarpMask,
        source_barrier: PhysicalBarrierId,
        cta_group: i64,
        cta_mask: u64,
        multicast: bool,
        itemsize: usize,
        source: &RuntimeBuffer,
        destination: &RuntimeBuffer,
        multicast_cta_mask: Option<u64>,
        remote_cta_id: Option<usize>,
        source_fill: AsyncSourceFill,
        round_to_tf32: bool,
        reduction: Option<DeferredGlobalReduction>,
        full_destination_semantics: Option<(Option<i64>, usize, usize)>,
        element_batches: &mut dyn FnMut(
            &mut dyn FnMut(
                &[(i64, i64, bool, bool); crate::WARP_SIZE],
                usize,
            ) -> Result<(), EngineError>,
        ) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        let completion_targets = resolve_mbarrier_completion_targets(
            context,
            barrier_pointer,
            issue_mask,
            source_barrier,
            cta_group,
            cta_mask,
            multicast,
        )?;
        let operation = self.checked_effect_operation(operation, OperationKind::AsyncIssue)?;
        let operation = operation.ok_or_else(|| {
            EngineError::message(
                "mbarrier-backed async payload requires a stable operation context",
            )
        })?;
        if issue_mask.len() != 1 {
            return Err(EngineError::message(
                "mbarrier-backed async payload requires exactly one active issuing lane",
            ));
        }
        let issuer_lane = issue_mask
            .first_active()
            .expect("singleton async payload issue mask has an active lane");
        let allow_source_fill = source_fill != AsyncSourceFill::None;
        if source.uniform_physical_space() == Some(PhysicalAccessSpace::Global)
            && destination.uniform_physical_space() == Some(PhysicalAccessSpace::Shared)
        {
            self.validate_typed_tma_shared_alignment(
                Some(operation),
                context,
                itemsize,
                destination,
                PhysicalAccessKind::Write,
                |element| {
                    element_batches(&mut |elements, count| {
                        if count > crate::WARP_SIZE {
                            return Err(EngineError::message(format!(
                                "copy_async element batch count {count} exceeds warp size"
                            )));
                        }
                        for &(_, destination_index, _, destination_in_bounds) in &elements[..count]
                        {
                            element(destination_index, destination_in_bounds, issuer_lane)?;
                        }
                        Ok(())
                    })
                },
            )?;
        }

        // Genuinely batch-only: NumSim executes the payload numerically at issue
        // time, so it registers just the scheduler-visible completion and never
        // materializes an `AsyncPayloadEffect`. The observing modes below share
        // the issue sequence with `async_payload_issue_with_resolved_targets`.
        if !M::OBSERVES_OPERATIONS {
            let transactions_per_target = self.execute_async_payload_batches(
                context,
                issuer_lane,
                itemsize,
                source,
                destination,
                multicast_cta_mask,
                remote_cta_id,
                source_fill,
                round_to_tf32,
                reduction,
                element_batches,
            )?;
            let completion_plan = completion_targets.issue_plan(transactions_per_target);
            completion_plan
                .complete_numeric(&self.kernel.services().mbarriers(), transactions_per_target)?;
            self.record_engine_progress();
            return Ok(());
        }

        // `resolve` records the batches it consumed so `numeric` can replay them;
        // both closures are handed to the shared issue sequence at once, so the
        // state they share lives in cells rather than in plain locals.
        let element_batches = RefCell::new(element_batches);
        let replay_batches = RefCell::new(Vec::new());
        let planned_from_element_batches = Cell::new(false);

        let resolve_effect = |operation: &OperationContext| {
            let can_use_full_destination_plan =
                full_destination_semantics
                    .as_ref()
                    .is_some_and(|(source_element_base, _, _)| {
                        source_element_base.is_some()
                            || !M::controls_physical_access_allocation(
                                self.kernel.mode_state(),
                                OperationKind::Load,
                                PhysicalAccessSpace::Global,
                                runtime_global_allocation(source),
                            )
                    });
            if can_use_full_destination_plan
                && M::compacts_async_accesses(self.kernel.mode_state())
                && multicast_cta_mask.is_none()
                && remote_cta_id.is_none()
                && reduction.is_none()
            {
                let (source_element_base, source_count, destination_count) =
                    full_destination_semantics.expect("checked above");
                return CompactAsyncCopyAccessPlan::plan_full_local_destination(
                    operation,
                    context,
                    itemsize,
                    source,
                    source_element_base,
                    source_count,
                    destination,
                    destination_count,
                    issuer_lane,
                );
            }
            if M::compacts_async_accesses(self.kernel.mode_state()) {
                let mut plan = CompactAsyncCopyAccessPlan::default();
                let mut plan_batch = |elements: &[(i64, i64, bool, bool); crate::WARP_SIZE],
                                      count: usize| {
                    replay_batches.borrow_mut().push((*elements, count));
                    for &(
                        source_index,
                        destination_index,
                        source_in_bounds,
                        destination_in_bounds,
                    ) in &elements[..count]
                    {
                        plan.plan_element(
                            operation,
                            context,
                            itemsize,
                            source,
                            source_index,
                            source_in_bounds,
                            destination,
                            destination_index,
                            destination_in_bounds,
                            multicast_cta_mask,
                            remote_cta_id,
                            allow_source_fill,
                            issuer_lane,
                        )?;
                    }
                    Ok(())
                };
                (*element_batches.borrow_mut())(&mut plan_batch)?;
                planned_from_element_batches.set(true);
                return plan.finish(operation, issuer_lane);
            }
            let mut accesses = Vec::new();
            let mut transactions_per_target = 0_u64;
            let mut plan_batch = |elements: &[(i64, i64, bool, bool); crate::WARP_SIZE],
                                  count: usize| {
                replay_batches.borrow_mut().push((*elements, count));
                for &(source_index, destination_index, source_in_bounds, destination_in_bounds) in
                    &elements[..count]
                {
                    let (source_accesses, destination_accesses, copied) =
                        plan_async_copy_element_at_lane_accesses(
                            operation,
                            context,
                            itemsize,
                            source,
                            source_index,
                            source_in_bounds,
                            destination,
                            destination_index,
                            destination_in_bounds,
                            multicast_cta_mask,
                            remote_cta_id,
                            allow_source_fill,
                            issuer_lane,
                        )?;
                    accesses.extend(source_accesses);
                    accesses.extend(destination_accesses);
                    transactions_per_target =
                        transactions_per_target.checked_add(copied).ok_or_else(|| {
                            EngineError::message("copy_async delivered-byte count overflow")
                        })?;
                }
                Ok(())
            };
            (*element_batches.borrow_mut())(&mut plan_batch)?;
            planned_from_element_batches.set(true);
            Ok((accesses, transactions_per_target))
        };

        let numeric_effect = || {
            let transactions_per_target = if planned_from_element_batches.get() {
                let replayed = replay_batches.borrow();
                let mut replay = |execute_batch: &mut dyn FnMut(
                    &[(i64, i64, bool, bool); crate::WARP_SIZE],
                    usize,
                )
                    -> Result<(), EngineError>| {
                    for (elements, count) in replayed.iter() {
                        execute_batch(elements, *count)?;
                    }
                    Ok(())
                };
                self.execute_async_payload_batches(
                    context,
                    issuer_lane,
                    itemsize,
                    source,
                    destination,
                    multicast_cta_mask,
                    remote_cta_id,
                    source_fill,
                    round_to_tf32,
                    reduction,
                    &mut replay,
                )?
            } else {
                self.execute_async_payload_batches(
                    context,
                    issuer_lane,
                    itemsize,
                    source,
                    destination,
                    multicast_cta_mask,
                    remote_cta_id,
                    source_fill,
                    round_to_tf32,
                    reduction,
                    &mut *element_batches.borrow_mut(),
                )?
            };
            Ok(((), transactions_per_target))
        };

        self.async_payload_issue_with_resolved_targets(
            Some(operation),
            &completion_targets,
            resolve_effect,
            numeric_effect,
        )
    }

    fn async_payload_issue_with_resolved_targets<R>(
        &self,
        operation: Option<&OperationContext>,
        completion_targets: &PhysicalMbarrierCompletionTargets,
        resolve_effect: impl FnOnce(
            &OperationContext,
        ) -> Result<(Vec<PhysicalAccessBatch>, u64), EngineError>,
        numeric_effect: impl FnOnce() -> Result<(R, u64), EngineError>,
    ) -> Result<R, EngineError> {
        self.async_payload_issue_with_resolved_targets_and_semantics(
            operation,
            completion_targets,
            MemoryAccessSemantics::async_proxy(),
            resolve_effect,
            numeric_effect,
        )
    }

    fn async_payload_issue_with_resolved_targets_and_semantics<R>(
        &self,
        operation: Option<&OperationContext>,
        completion_targets: &PhysicalMbarrierCompletionTargets,
        access_semantics: MemoryAccessSemantics,
        resolve_effect: impl FnOnce(
            &OperationContext,
        ) -> Result<(Vec<PhysicalAccessBatch>, u64), EngineError>,
        numeric_effect: impl FnOnce() -> Result<(R, u64), EngineError>,
    ) -> Result<R, EngineError> {
        self.async_payload_issue_with_resolved_plan(
            operation,
            access_semantics,
            |operation| {
                let (accesses, transactions_per_target) = resolve_effect(operation)?;
                Ok((
                    accesses,
                    completion_targets.issue_plan(transactions_per_target),
                    transactions_per_target,
                ))
            },
            || {
                let (result, transactions_per_target) = numeric_effect()?;
                Ok((
                    result,
                    completion_targets.issue_plan(transactions_per_target),
                    transactions_per_target,
                ))
            },
        )
    }

    /// Publish one asynchronous payload whose completion plan is already
    /// resolved per target rather than derived from one uniform byte count.
    ///
    /// The plan is produced twice --- once beside the footprint and once beside
    /// the numeric transfer --- and the two must agree, which is what keeps a
    /// planner that disagrees with the executed copy from being credited.
    fn async_payload_issue_with_resolved_plan<R>(
        &self,
        operation: Option<&OperationContext>,
        access_semantics: MemoryAccessSemantics,
        resolve_effect: impl FnOnce(
            &OperationContext,
        ) -> Result<
            (
                Vec<PhysicalAccessBatch>,
                PhysicalMbarrierCompletionIssuePlan,
                u64,
            ),
            EngineError,
        >,
        numeric_effect: impl FnOnce() -> Result<
            (R, PhysicalMbarrierCompletionIssuePlan, u64),
            EngineError,
        >,
    ) -> Result<R, EngineError> {
        let operation = self.checked_effect_operation(operation, OperationKind::AsyncIssue)?;
        let operation = operation.ok_or_else(|| {
            EngineError::message(
                "mbarrier-backed async payload requires a stable operation context",
            )
        })?;
        if !M::OBSERVES_OPERATIONS {
            let (result, completion_plan, payload_byte_len) = numeric_effect()?;
            completion_plan
                .complete_numeric(&self.kernel.services().mbarriers(), payload_byte_len)?;
            self.record_engine_progress();
            return Ok(result);
        }
        if !M::resolves_async_accesses(self.kernel.mode_state()) {
            let (result, completion_plan, payload_byte_len) = numeric_effect()?;
            let mut payload = AsyncPayloadEffect::new_with_access_semantics(
                operation.clone(),
                Vec::new(),
                completion_plan,
                access_semantics,
            )
            .map_err(|error| EngineError::message(error.to_string()))?;
            self.before_effect(Some(operation), OperationEffect::AsyncPayload(&payload))?;
            let action_ids = self
                .kernel
                .services()
                .deferred_payloads()
                .enqueue_payload(&payload, payload_byte_len)?;
            payload
                .bind_completion_action_ids(action_ids)
                .map_err(|error| EngineError::message(error.to_string()))?;
            self.after_effect(Some(operation), OperationEffect::AsyncPayload(&payload))?;
            self.record_engine_progress();
            return Ok(result);
        }
        let (accesses, completion_plan, _planned_byte_len) = resolve_effect(operation)?;
        let mut payload = AsyncPayloadEffect::new_with_access_semantics(
            operation.clone(),
            accesses,
            completion_plan,
            access_semantics,
        )
        .map_err(|error| EngineError::message(error.to_string()))?;
        let participates_in_global_memory = M::USES_GLOBAL_MEMORY_TRANSACTION
            && payload
                .accesses()
                .iter()
                .any(|batch| batch.descriptor().space().has_read_from_versions());
        let _profile = participates_in_global_memory
            .then(|| ProfileTimer::new(ProfileKind::RaceGlobalTransactionAsyncIssue));
        let transaction_spans = if participates_in_global_memory {
            crate::physical_access::global_batch_spans(payload.accesses())
        } else {
            Vec::new()
        };
        let _transaction = begin_global_memory_transaction::<M>(
            self.kernel.mode_state(),
            participates_in_global_memory,
            crate::physical_access::global_transaction_exclusive(payload.accesses()),
            &transaction_spans,
        )?;
        {
            self.before_effect(Some(operation), OperationEffect::AsyncPayload(&payload))?;
        }
        let (result, numeric_completion_plan, numeric_payload_byte_len) = {
            numeric_effect()?
        };
        if &numeric_completion_plan != payload.completion_plan() {
            return Err(EngineError::message(format!(
                "async payload completion plan changed after numeric execution at {}",
                operation.id()
            )));
        }
        let action_ids = self
            .kernel
            .services()
            .deferred_payloads()
            .enqueue_payload(&payload, numeric_payload_byte_len)?;
        payload
            .bind_completion_action_ids(action_ids)
            .map_err(|error| EngineError::message(error.to_string()))?;
        {
            self.after_effect(Some(operation), OperationEffect::AsyncPayload(&payload))?;
        }
        self.record_engine_progress();
        Ok(result)
    }

    pub(crate) fn async_group_issue_with_destination_kind<R>(
        &self,
        operation: Option<&OperationContext>,
        domain: AsyncGroupDomain,
        resolve_exact_accesses: bool,
        mask: WarpMask,
        destination_kind: PhysicalAccessKind,
        resolve_effect: impl FnOnce(
            &mut dyn FnMut(
                usize,
                &RuntimeBuffer,
                i64,
                bool,
                &RuntimeBuffer,
                i64,
                bool,
                Option<u64>,
                Option<usize>,
                bool,
                usize,
            ) -> Result<(), EngineError>,
        ) -> Result<(), EngineError>,
        numeric_effect: impl FnOnce() -> Result<(R, Vec<DeferredGlobalWrite>), EngineError>,
    ) -> Result<R, EngineError> {
        self.async_group_issue_with_resolved_accesses(
            operation,
            domain,
            resolve_exact_accesses,
            mask,
            |operation| {
                let with_destination_kind = |batch: PhysicalAccessBatch| {
                    let batch = batch.with_kind(destination_kind);
                    if destination_kind == PhysicalAccessKind::AtomicReadModifyWrite {
                        batch.with_memory_semantics(MemoryAccessSemantics::async_reduction())
                    } else {
                        batch
                    }
                };
                if M::compacts_async_accesses(self.kernel.mode_state()) {
                    let lane = operation.active_mask().first_active().ok_or_else(|| {
                        EngineError::message("copy_async compact plan has no issuing lane")
                    })?;
                    let mut plan = CompactAsyncCopyAccessPlan::default();
                    let mut plan_element =
                        |itemsize: usize,
                         source: &RuntimeBuffer,
                         source_index: i64,
                         source_in_bounds: bool,
                         destination: &RuntimeBuffer,
                         destination_index: i64,
                         destination_in_bounds: bool,
                         multicast_cta_mask: Option<u64>,
                         remote_cta_id: Option<usize>,
                         allow_source_zero_fill: bool,
                         element_lane: usize| {
                            plan.plan_element(
                                operation,
                                &self.context,
                                itemsize,
                                source,
                                source_index,
                                source_in_bounds,
                                destination,
                                destination_index,
                                destination_in_bounds,
                                multicast_cta_mask,
                                remote_cta_id,
                                allow_source_zero_fill,
                                element_lane,
                            )
                        };
                    resolve_effect(&mut plan_element)?;
                    let (batches, _delivered) = plan.finish(operation, lane)?;
                    let mut source_accesses = Vec::new();
                    let mut destination_accesses = Vec::new();
                    for batch in batches {
                        match batch.descriptor().kind() {
                            PhysicalAccessKind::Read => source_accesses.push(batch),
                            PhysicalAccessKind::Write => {
                                destination_accesses.push(with_destination_kind(batch))
                            }
                            PhysicalAccessKind::AtomicReadModifyWrite => unreachable!(
                                "typed async copy planning never emits an atomic access"
                            ),
                        }
                    }
                    return Ok((source_accesses, destination_accesses));
                }
                let source_accesses = RefCell::new(Vec::new());
                let destination_accesses = RefCell::new(Vec::new());
                let mut plan_element = |itemsize: usize,
                                        source: &RuntimeBuffer,
                                        source_index: i64,
                                        source_in_bounds: bool,
                                        destination: &RuntimeBuffer,
                                        destination_index: i64,
                                        destination_in_bounds: bool,
                                        multicast_cta_mask: Option<u64>,
                                        remote_cta_id: Option<usize>,
                                        allow_source_zero_fill: bool,
                                        lane: usize| {
                    let (source, destination, _copied) = plan_async_copy_element_at_lane_accesses(
                        operation,
                        &self.context,
                        itemsize,
                        source,
                        source_index,
                        source_in_bounds,
                        destination,
                        destination_index,
                        destination_in_bounds,
                        multicast_cta_mask,
                        remote_cta_id,
                        allow_source_zero_fill,
                        lane,
                    )?;
                    source_accesses.borrow_mut().extend(source);
                    destination_accesses.borrow_mut().extend(destination);
                    Ok(())
                };
                resolve_effect(&mut plan_element)?;
                Ok((
                    source_accesses.into_inner(),
                    destination_accesses
                        .into_inner()
                        .into_iter()
                        .map(with_destination_kind)
                        .collect(),
                ))
            },
            numeric_effect,
        )
    }

    /// Generic-proxy release writes reuse deferred completion and per-lane
    /// async actors, but have no user-visible commit/wait group. Committing
    /// their private domain at issue enables progress and launch-exit draining
    /// without allowing cp.async/bulk waits to acquire these writes.
    pub(crate) fn async_release_issue(
        &self,
        operation: Option<&OperationContext>,
        context: &WarpContext,
        destination: &PhysicalPtr,
        values: &WarpValue<u64>,
        byte_len: usize,
        scope: MemoryScope,
        mmio: bool,
        reduction: bool,
    ) -> Result<(), EngineError> {
        let operation = self.required_operation(operation, OperationKind::AsyncIssue)?;
        let mask = context.active_mask();
        if !matches!(byte_len, 1 | 2 | 4 | 8)
            || (reduction && byte_len < 4)
            || !matches!(scope, MemoryScope::Gpu | MemoryScope::Sys)
            || (mmio && scope != MemoryScope::Sys)
        {
            return Err(EngineError::message("invalid async release width/scope"));
        }
        destination.require_ptx_space_for_mask(PtxStateSpace::Global, mask)?;
        let kind = if reduction {
            PhysicalAccessKind::AtomicReadModifyWrite
        } else {
            PhysicalAccessKind::Write
        };
        let semantics = MemoryAccessSemantics::scoped(
            MemoryOrder::Release,
            scope,
            if mmio {
                MemoryProxy::Mmio
            } else {
                MemoryProxy::Generic
            },
            if reduction {
                crate::MemoryAccessClass::Reduction
            } else {
                crate::MemoryAccessClass::Atomic
            },
        );
        let mut lane_accesses = Vec::with_capacity(mask.len());
        let mut writes = Vec::with_capacity(mask.len());
        let memory = self.kernel.physical().global();
        for lane in mask {
            let offset = if reduction {
                destination.lane_read_write_byte_offset(lane, byte_len)?
            } else {
                destination.lane_write_byte_offset(lane, byte_len)?
            };
            let access = resolve_runtime_physical_access(
                context,
                destination.buffer(),
                lane,
                offset,
                byte_len,
                kind,
            )?;
            if access.space() != PhysicalAccessSpace::Global {
                return Err(EngineError::message(
                    "async release destination must be global",
                ));
            }
            let span = access.span();
            let view = memory.full_view(crate::AllocationId::from_u64(span.allocation().get()))?;
            if (view.observed_allocation_address().unwrap_or(0) % byte_len as u64
                + (span.byte_offset() % byte_len) as u64)
                % byte_len as u64
                != 0
            {
                return Err(EngineError::message(
                    "async release address is not naturally aligned",
                ));
            }
            let bytes = values[lane].to_le_bytes()[..byte_len].to_vec();
            writes.push(if reduction {
                memory.defer_reduction_write_bytes(
                    &view,
                    span.byte_offset(),
                    bytes,
                    if byte_len == 4 {
                        DeferredGlobalReduction::AddU32
                    } else {
                        DeferredGlobalReduction::AddU64
                    },
                )?
            } else {
                memory.defer_write_bytes(&view, span.byte_offset(), bytes)?
            });
            let accesses = if M::resolves_async_accesses(self.kernel.mode_state()) {
                let lane_operation = operation.clone().with_active_mask(
                    WarpMask::from_lanes([lane])
                        .map_err(|error| EngineError::message(error.to_string()))?,
                );
                vec![
                    crate::runtime::single_lane_physical_access_batch(
                        &lane_operation,
                        lane,
                        kind,
                        PhysicalAccessSpace::Global,
                        vec![span],
                    )?
                    .with_memory_semantics(semantics),
                ]
            } else {
                Vec::new()
            };
            lane_accesses.push((lane, Vec::new(), accesses));
        }
        let hub = self.kernel.services().async_groups();
        if M::OBSERVES_OPERATIONS {
            let effect = AsyncGroupIssueBatchEffect::new(
                operation.clone(),
                AsyncGroupDomain::Release,
                lane_accesses,
            )?;
            let transaction_spans = if M::USES_GLOBAL_MEMORY_TRANSACTION {
                crate::physical_access::global_batch_spans(
                    effect
                        .members()
                        .iter()
                        .flat_map(|member| member.destination_accesses()),
                )
            } else {
                Vec::new()
            };
            let _transaction = begin_global_memory_transaction::<M>(
                self.kernel.mode_state(),
                M::USES_GLOBAL_MEMORY_TRANSACTION,
                true,
                &transaction_spans,
            )?;
            self.before_effect(
                Some(operation),
                OperationEffect::AsyncGroupIssueBatch(&effect),
            )?;
            hub.issue_exact_batch(context, &effect, writes)?;
            self.after_effect(
                Some(operation),
                OperationEffect::AsyncGroupIssueBatch(&effect),
            )?;
        } else {
            hub.issue_unmodeled(context, AsyncGroupDomain::Release, mask, writes)?;
        }
        // This is an internal scheduling transition, not a PTX commit_group.
        let committed =
            hub.commit_detailed(hub.commit_plan(context, AsyncGroupDomain::Release, mask)?)?;
        self.kernel
            .services()
            .completion_publications()
            .publish_async_groups(
                committed
                    .groups()
                    .iter()
                    .flat_map(|group| [group.source_read_action_id(), group.full_action_id()]),
            );
        self.record_engine_progress();
        Ok(())
    }

    /// Issue one no-payload Bulk async-group token for every participating lane.
    ///
    /// Cache-priority operations use the ordinary per-thread bulk-group
    /// lifecycle even though they carry no memory footprint for NumSim or the
    /// native analyses to observe.  Reusing the batch effect keeps their token,
    /// commit, completion, and wait semantics identical to other lane-local
    /// async-group issues without inventing a second queue or effect kind.
    pub(crate) fn bulk_async_group_issue_without_accesses(
        &self,
        operation: Option<&OperationContext>,
        mask: WarpMask,
    ) -> Result<(), EngineError> {
        let operation = self.checked_effect_operation(operation, OperationKind::AsyncIssue)?;
        if operation.is_some_and(|operation| operation.active_mask() != mask) {
            return Err(EngineError::message(format!(
                "async-group issue operation mask {:#010x} does not match resolved mask {:#010x}",
                operation.expect("checked above").active_mask().bits(),
                mask.bits(),
            )));
        }
        let async_groups = self.kernel.services().async_groups();
        if !M::OBSERVES_OPERATIONS {
            async_groups.issue(&self.context, AsyncGroupDomain::Bulk, mask)?;
            self.record_engine_progress();
            return Ok(());
        }
        let operation = operation.ok_or_else(|| {
            EngineError::message("exact async-group issue requires a stable operation context")
        })?;
        let effect = AsyncGroupIssueBatchEffect::new(
            operation.clone(),
            AsyncGroupDomain::Bulk,
            mask.into_iter().map(|lane| (lane, Vec::new(), Vec::new())),
        )?;
        self.before_effect(
            Some(operation),
            OperationEffect::AsyncGroupIssueBatch(&effect),
        )?;
        async_groups.issue_exact_batch(&self.context, &effect, Vec::new())?;
        self.after_effect(
            Some(operation),
            OperationEffect::AsyncGroupIssueBatch(&effect),
        )?;
        self.record_engine_progress();
        Ok(())
    }

    fn async_group_issue_with_resolved_accesses<R>(
        &self,
        operation: Option<&OperationContext>,
        domain: AsyncGroupDomain,
        resolve_exact_accesses: bool,
        mask: WarpMask,
        resolve_effect: impl FnOnce(
            &OperationContext,
        ) -> Result<
            (Vec<PhysicalAccessBatch>, Vec<PhysicalAccessBatch>),
            EngineError,
        >,
        numeric_effect: impl FnOnce() -> Result<(R, Vec<DeferredGlobalWrite>), EngineError>,
    ) -> Result<R, EngineError> {
        let operation = self.checked_effect_operation(operation, OperationKind::AsyncIssue)?;
        if operation.is_some_and(|operation| operation.active_mask() != mask) {
            return Err(EngineError::message(format!(
                "async-group issue operation mask {:#010x} does not match resolved mask {:#010x}",
                operation.expect("checked above").active_mask().bits(),
                mask.bits(),
            )));
        }
        let async_groups = self.kernel.services().async_groups();
        if !resolve_exact_accesses {
            let (result, writes) = numeric_effect()?;
            async_groups.issue_unmodeled(&self.context, domain, mask, writes)?;
            self.record_engine_progress();
            return Ok(result);
        }
        let operation = operation.ok_or_else(|| {
            EngineError::message("exact async-group issue requires a stable operation context")
        })?;
        if !M::OBSERVES_OPERATIONS {
            let (result, writes) = numeric_effect()?;
            async_groups.issue_unmodeled(&self.context, domain, mask, writes)?;
            self.record_engine_progress();
            return Ok(result);
        }
        async_groups.validate_exact_issue_participation(&self.context, operation, domain)?;
        let (source_accesses, destination_accesses) =
            if M::resolves_async_accesses(self.kernel.mode_state()) {
                resolve_effect(operation)?
            } else {
                (Vec::new(), Vec::new())
            };
        let effect = AsyncGroupIssueEffect::new(
            operation.clone(),
            domain,
            source_accesses,
            destination_accesses,
        )?;
        async_groups.validate_exact_issue(&self.context, &effect)?;
        let participates_in_global_memory = M::USES_GLOBAL_MEMORY_TRANSACTION
            && effect
                .source_accesses()
                .iter()
                .chain(effect.destination_accesses())
                .any(|batch| batch.descriptor().space().has_read_from_versions());
        let _profile = participates_in_global_memory
            .then(|| ProfileTimer::new(ProfileKind::RaceGlobalTransactionAsyncIssue));
        let transaction_spans = if participates_in_global_memory {
            crate::physical_access::global_batch_spans(
                effect
                    .source_accesses()
                    .iter()
                    .chain(effect.destination_accesses()),
            )
        } else {
            Vec::new()
        };
        let _transaction = begin_global_memory_transaction::<M>(
            self.kernel.mode_state(),
            participates_in_global_memory,
            crate::physical_access::global_transaction_exclusive(
                effect
                    .source_accesses()
                    .iter()
                    .chain(effect.destination_accesses()),
            ),
            &transaction_spans,
        )?;
        self.before_effect(Some(operation), OperationEffect::AsyncGroupIssue(&effect))?;
        let (result, writes) = numeric_effect()?;
        async_groups.issue_exact(&self.context, effect.clone(), writes)?;
        self.after_effect(Some(operation), OperationEffect::AsyncGroupIssue(&effect))?;
        self.record_engine_progress();
        Ok(result)
    }

    /// Commit the current per-lane async-group FIFO with exact token membership.
    pub(crate) fn async_group_commit(
        &self,
        operation: Option<&OperationContext>,
        domain: AsyncGroupDomain,
        mask: WarpMask,
    ) -> Result<(), EngineError> {
        let operation = self.checked_effect_operation(operation, OperationKind::Barrier)?;
        let hub = self.kernel.services().async_groups();
        let plan = hub.commit_plan(&self.context, domain, mask)?;
        self.before_effect(
            operation,
            OperationEffect::AsyncGroupCommit {
                plan: &plan,
                outcome: None,
            },
        )?;
        let outcome = hub.commit_detailed(plan.clone())?;
        self.after_effect(
            operation,
            OperationEffect::AsyncGroupCommit {
                plan: &plan,
                outcome: Some(&outcome),
            },
        )?;
        self.record_engine_progress();
        Ok(())
    }

    /// Schedule prior classic `cp.async` work and bind one deferred mbarrier
    /// arrival to each issuing lane's full-completion milestone, without
    /// closing an explicit cp.async group.
    ///
    /// `increments_pending` selects the PTX spelling. Without `.noinc` the
    /// instruction first raises the current phase's pending arrival count by one
    /// per issuing lane, which is what makes the deferred arrive-on a net-zero
    /// transition instead of an unbudgeted extra arrival; `.noinc` skips that
    /// raise. Both halves are published inside one operation, so no observer
    /// sees the barrier between the raise and the enqueued arrival.
    pub(crate) fn cp_async_mbarrier_arrive_completion(
        &self,
        operation: Option<&OperationContext>,
        pointer: &PhysicalPtr,
        mask: WarpMask,
        increments_pending: bool,
    ) -> Result<(), EngineError> {
        let operation = self.checked_effect_operation(operation, OperationKind::Barrier)?;
        let async_groups = self.kernel.services().async_groups();
        let commit_plan =
            async_groups.commit_plan(&self.context, AsyncGroupDomain::CpAsync, mask)?;
        let arrival_plan =
            plan_cp_async_mbarrier_arrive(&self.context, pointer, mask, increments_pending)?;
        self.before_effect(
            operation,
            OperationEffect::CpAsyncMbarrierArrive {
                plan: &arrival_plan,
                commit_plan: &commit_plan,
                outcome: None,
                actions: None,
            },
        )?;
        let actions = arrival_plan.apply(&self.kernel.services().mbarriers())?;
        async_groups.commit_detailed_with_physical_arrivals(
            commit_plan.clone(),
            &actions,
            |outcome| {
                self.after_effect(
                    operation,
                    OperationEffect::CpAsyncMbarrierArrive {
                        plan: &arrival_plan,
                        commit_plan: &commit_plan,
                        outcome: Some(outcome),
                        actions: Some(&actions),
                    },
                )
            },
        )?;
        self.record_engine_progress();
        Ok(())
    }

    /// Complete and retire the exact FIFO prefix selected by one wait-group.
    pub(crate) async fn async_group_wait(
        &self,
        operation: Option<&OperationContext>,
        domain: AsyncGroupDomain,
        mask: WarpMask,
        pending_groups: i64,
        read_only: bool,
    ) -> Result<(), EngineError> {
        let operation = self
            .checked_effect_operation(operation, OperationKind::Barrier)?
            .ok_or_else(|| {
                EngineError::message("async-group wait requires a stable operation context")
            })?;
        let hub = self.kernel.services().async_groups();
        let plan = hub.wait_plan(&self.context, domain, mask, pending_groups, read_only)?;
        self.before_effect(
            Some(operation),
            OperationEffect::AsyncGroupWait {
                plan: &plan,
                outcome: None,
            },
        )?;
        let outcome = crate::AsyncGroupHub::wait(&hub, plan.clone(), operation.clone())?.await?;
        self.after_effect(
            Some(operation),
            OperationEffect::AsyncGroupWait {
                plan: &plan,
                outcome: Some(&outcome),
            },
        )?;
        self.record_engine_progress();
        Ok(())
    }

    fn checked_effect_operation<'a>(
        &self,
        operation: Option<&'a OperationContext>,
        expected_kind: OperationKind,
    ) -> Result<Option<&'a OperationContext>, EngineError> {
        if !M::OBSERVES_OPERATIONS && operation.is_none() {
            return Ok(None);
        }
        let operation = operation.ok_or_else(|| {
            EngineError::message(format!(
                "{} mode requires an operation context for {expected_kind}",
                M::NAME
            ))
        })?;
        if operation.id().global_warp_id() != self.context.global_warp_id() {
            return Err(EngineError::message(format!(
                "effect operation warp {} does not match engine warp {}",
                operation.id().global_warp_id(),
                self.context.global_warp_id(),
            )));
        }
        if operation.kind() != expected_kind {
            return Err(EngineError::message(format!(
                "effect operation kind {} does not match expected {expected_kind}",
                operation.kind(),
            )));
        }
        Ok(Some(operation))
    }

    fn required_operation_identity<'a>(
        &self,
        operation: Option<&'a OperationContext>,
        expected_kind: OperationKind,
    ) -> Result<(&'a OperationContext, u64, Vec<i64>), EngineError> {
        let operation = self.required_operation(operation, expected_kind)?;
        let loop_iteration_path = operation
            .id()
            .loop_frames()
            .iter()
            .map(|frame| {
                i64::try_from(frame.iteration_ordinal())
                    .map_err(|_| EngineError::message("operation loop iteration exceeds i64"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((
            operation,
            operation.id().source_op_id().get(),
            loop_iteration_path,
        ))
    }

    fn required_operation<'a>(
        &self,
        operation: Option<&'a OperationContext>,
        expected_kind: OperationKind,
    ) -> Result<&'a OperationContext, EngineError> {
        self.checked_effect_operation(operation, expected_kind)?
            .ok_or_else(|| {
                EngineError::message(format!(
                    "{expected_kind} operation requires a stable generated identity"
                ))
            })
    }

    fn operation_warp_context(&self, operation: &OperationContext) -> WarpContext {
        self.context
            .with_active_mask(operation.active_mask())
            .with_control_provenance(operation.control_provenance())
    }

    fn before_effect(
        &self,
        operation: Option<&OperationContext>,
        effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        if let Some(operation) = operation {
            M::before_effect(self.kernel.mode_state(), operation, effect)?;
        }
        Ok(())
    }

    fn after_effect(
        &self,
        operation: Option<&OperationContext>,
        effect: OperationEffect<'_>,
    ) -> Result<(), EngineError> {
        if let Some(operation) = operation {
            M::after_effect(self.kernel.mode_state(), operation, effect)?;
        }
        let publications = self.kernel.services().completion_publications();
        match effect {
            OperationEffect::AsyncPayload(payload) => {
                if let Some(action_ids) = payload.completion_action_ids() {
                    publications.publish_physical(action_ids.iter().copied());
                }
            }
            OperationEffect::MbarrierCompletionIssue {
                action_ids: Some(action_ids),
                ..
            } => publications.publish_physical(action_ids.iter().copied()),
            OperationEffect::TcgenCommitIssue {
                actions: Some(actions),
                ..
            } => publications.publish_physical(actions.iter().map(|action| action.id())),
            OperationEffect::AsyncGroupCommit {
                outcome: Some(outcome),
                ..
            } => publications.publish_async_groups(
                outcome
                    .groups()
                    .iter()
                    .flat_map(|group| [group.source_read_action_id(), group.full_action_id()]),
            ),
            OperationEffect::CpAsyncMbarrierArrive {
                outcome: Some(outcome),
                ..
            } => {
                publications.publish_async_groups(
                    outcome
                        .groups()
                        .iter()
                        .flat_map(|group| [group.source_read_action_id(), group.full_action_id()]),
                );
                publications.publish_physical(
                    outcome
                        .immediately_ready_physical_actions()
                        .iter()
                        .map(|action| action.id()),
                );
            }
            OperationEffect::PhysicalAccess(_)
            | OperationEffect::AsyncGroupIssue(_)
            | OperationEffect::AsyncGroupIssueBatch(_)
            | OperationEffect::AsyncGroupCommit { outcome: None, .. }
            | OperationEffect::CpAsyncMbarrierArrive { outcome: None, .. }
            | OperationEffect::AsyncGroupWait { .. }
            | OperationEffect::MemoryFence(_)
            | OperationEffect::ProxyAsyncFence(_)
            | OperationEffect::TensorMap(_)
            | OperationEffect::WarpSync(_)
            | OperationEffect::TcgenFence(_)
            | OperationEffect::AnalysisGap(_)
            | OperationEffect::MbarrierInit(_)
            | OperationEffect::MbarrierInvalidate { .. }
            | OperationEffect::MbarrierInitFence { .. }
            | OperationEffect::MbarrierExpectTx { .. }
            | OperationEffect::MbarrierArrive { .. }
            | OperationEffect::MbarrierArriveBatch { .. }
            | OperationEffect::MbarrierWait { .. }
            | OperationEffect::DeclaredWordWait { .. }
            | OperationEffect::MbarrierCompletionIssue {
                action_ids: None, ..
            }
            | OperationEffect::TcgenWorkIssue(_)
            | OperationEffect::TcgenCommitIssue { actions: None, .. }
            | OperationEffect::TcgenWait { .. }
            | OperationEffect::NamedBarrierArrive { .. }
            | OperationEffect::NamedBarrierSyncRegister { .. }
            | OperationEffect::NamedBarrierSyncResume(_)
            | OperationEffect::ClusterBarrierArrive { .. }
            | OperationEffect::ClusterBarrierWaitRegister { .. }
            | OperationEffect::ClusterBarrierWaitResume(_)
            | OperationEffect::TcgenLifecycleRegister(_)
            | OperationEffect::TcgenLifecycleResume(_)
            | OperationEffect::SetmaxnregRegister(_)
            | OperationEffect::SetmaxnregResume(_) => {}
        }
        Ok(())
    }
}

fn validate_aligned_named_barrier_mask(
    operation: Option<&OperationContext>,
    mask: WarpMask,
    label: &str,
) -> Result<(), EngineError> {
    if operation.is_some_and(|operation| {
        operation
            .control_provenance()
            .elect_sync_entry_mask()
            .is_some()
    }) {
        return Ok(());
    }
    require_full_warp_sync(mask, label)
}

fn physical_access_error_context(
    error: EngineError,
    operation: Option<&OperationContext>,
    logical_buffer: Option<&str>,
) -> EngineError {
    if let Some(operation) = operation {
        return error.with_operation_context(operation);
    }
    match logical_buffer {
        Some(buffer) => error.with_context(format_args!("buffer {buffer}: ")),
        None => error,
    }
}

fn physical_access_kind(kind: OperationKind) -> Result<PhysicalAccessKind, EngineError> {
    match kind {
        OperationKind::Load => Ok(PhysicalAccessKind::Read),
        OperationKind::Store => Ok(PhysicalAccessKind::Write),
        OperationKind::Atomic => Ok(PhysicalAccessKind::AtomicReadModifyWrite),
        other => Err(EngineError::message(format!(
            "operation kind {other} does not describe a physical memory access"
        ))),
    }
}

fn runtime_buffer_physical_space_at(buffer: &RuntimeBuffer, lane: usize) -> PhysicalAccessSpace {
    match buffer {
        RuntimeBuffer::AccessView { buffer, .. } => runtime_buffer_physical_space_at(buffer, lane),
        RuntimeBuffer::LaneSelected { buffers } => {
            runtime_buffer_physical_space_at(&buffers[lane], lane)
        }
        RuntimeBuffer::Global(_) => PhysicalAccessSpace::Global,
        RuntimeBuffer::Shared { .. } | RuntimeBuffer::RemoteShared { .. } => {
            PhysicalAccessSpace::Shared
        }
        RuntimeBuffer::Local { .. } => PhysicalAccessSpace::Local,
        RuntimeBuffer::Register { .. } => PhysicalAccessSpace::Register,
        RuntimeBuffer::Tmem { .. } => PhysicalAccessSpace::Tmem,
    }
}

fn runtime_buffer_physical_space_for_mask(
    buffer: &RuntimeBuffer,
    mask: WarpMask,
) -> Result<Option<PhysicalAccessSpace>, EngineError> {
    let Some(first_lane) = mask.first_active() else {
        return Ok(None);
    };
    let first = runtime_buffer_physical_space_at(buffer, first_lane);
    for lane in mask {
        let candidate = runtime_buffer_physical_space_at(buffer, lane);
        if candidate != first {
            return Err(EngineError::message(format!(
                "active lane-selected buffers use different physical spaces: lane {first_lane} is {first}, lane {lane} is {candidate}"
            )));
        }
    }
    Ok(Some(first))
}

fn runtime_buffer_proxy_memory_domain_at(buffer: &RuntimeBuffer, lane: usize) -> ProxyMemoryDomain {
    match buffer {
        RuntimeBuffer::AccessView { buffer, .. } => {
            runtime_buffer_proxy_memory_domain_at(buffer, lane)
        }
        RuntimeBuffer::LaneSelected { buffers } => {
            runtime_buffer_proxy_memory_domain_at(&buffers[lane], lane)
        }
        RuntimeBuffer::Global(_) => ProxyMemoryDomain::Global,
        RuntimeBuffer::Shared { .. } => ProxyMemoryDomain::SharedCta,
        RuntimeBuffer::RemoteShared { .. } => ProxyMemoryDomain::SharedCluster,
        RuntimeBuffer::Local { .. }
        | RuntimeBuffer::Register { .. }
        | RuntimeBuffer::Tmem { .. } => ProxyMemoryDomain::Other,
    }
}

fn uniform_runtime_buffer_proxy_memory_domain(buffer: &RuntimeBuffer) -> Option<ProxyMemoryDomain> {
    match buffer {
        RuntimeBuffer::AccessView { buffer, .. } => {
            uniform_runtime_buffer_proxy_memory_domain(buffer)
        }
        RuntimeBuffer::LaneSelected { .. } => None,
        RuntimeBuffer::Global(_) => Some(ProxyMemoryDomain::Global),
        RuntimeBuffer::Shared { .. } => Some(ProxyMemoryDomain::SharedCta),
        RuntimeBuffer::RemoteShared { .. } => Some(ProxyMemoryDomain::SharedCluster),
        RuntimeBuffer::Local { .. }
        | RuntimeBuffer::Register { .. }
        | RuntimeBuffer::Tmem { .. } => Some(ProxyMemoryDomain::Other),
    }
}

fn runtime_buffer_proxy_memory_domain_for_mask(
    buffer: &RuntimeBuffer,
    mask: WarpMask,
) -> Result<Option<ProxyMemoryDomain>, EngineError> {
    let Some(first_lane) = mask.first_active() else {
        return Ok(None);
    };
    if let Some(domain) = uniform_runtime_buffer_proxy_memory_domain(buffer) {
        return Ok(Some(domain));
    }
    let first = runtime_buffer_proxy_memory_domain_at(buffer, first_lane);
    for lane in mask {
        let candidate = runtime_buffer_proxy_memory_domain_at(buffer, lane);
        if candidate != first {
            return Err(EngineError::message(format!(
                "active lane-selected buffers use different proxy memory domains: \
                 lane {first_lane} is {first:?}, lane {lane} is {candidate:?}"
            )));
        }
    }
    Ok(Some(first))
}

fn runtime_buffer_proxy_memory_domain_for_space_and_mask(
    buffer: &RuntimeBuffer,
    space: PhysicalAccessSpace,
    mask: WarpMask,
) -> Result<ProxyMemoryDomain, EngineError> {
    match space {
        PhysicalAccessSpace::Global => Ok(ProxyMemoryDomain::Global),
        PhysicalAccessSpace::Shared => runtime_buffer_proxy_memory_domain_for_mask(buffer, mask)
            .map(|domain| domain.expect("a resolved shared space has an active proxy domain")),
        PhysicalAccessSpace::Local | PhysicalAccessSpace::Register | PhysicalAccessSpace::Tmem => {
            Ok(ProxyMemoryDomain::Other)
        }
    }
}

fn physical_access_descriptor_for_mode<M: EngineMode>(
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    byte_width: usize,
    buffer: &RuntimeBuffer,
    mask: WarpMask,
) -> Result<PhysicalAccessDescriptor, EngineError> {
    let descriptor = PhysicalAccessDescriptor::new(kind, space, byte_width)
        .map_err(|error| EngineError::message(error.to_string()))?;
    if M::OBSERVES_PROXY_MEMORY_DOMAINS {
        Ok(descriptor.with_proxy_memory_domain(
            runtime_buffer_proxy_memory_domain_for_space_and_mask(buffer, space, mask)?,
        ))
    } else {
        Ok(descriptor)
    }
}

fn runtime_global_allocation(buffer: &RuntimeBuffer) -> Option<PhysicalAllocationId> {
    match buffer {
        RuntimeBuffer::AccessView { buffer, .. } => runtime_global_allocation(buffer),
        RuntimeBuffer::LaneSelected { buffers } => {
            let mut allocations = buffers
                .lanes()
                .iter()
                .map(|buffer| runtime_global_allocation(buffer.as_ref()));
            let allocation = allocations.next().flatten()?;
            allocations
                .all(|candidate| candidate == Some(allocation))
                .then_some(allocation)
        }
        RuntimeBuffer::Global(view) => Some(view.allocation().into()),
        RuntimeBuffer::Shared { .. }
        | RuntimeBuffer::RemoteShared { .. }
        | RuntimeBuffer::Local { .. }
        | RuntimeBuffer::Register { .. }
        | RuntimeBuffer::Tmem { .. } => None,
    }
}

const fn async_completion_payload_proxy_domain(
    issuing_cta: usize,
    target_cta: usize,
) -> ProxyMemoryDomain {
    if issuing_cta == target_cta {
        ProxyMemoryDomain::SharedCta
    } else {
        ProxyMemoryDomain::SharedCluster
    }
}

#[cfg(test)]
mod tests {
    use std::future::poll_fn;
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::Poll;

    use crate::runtime::{
        ExecutionPolicy, LaunchSelection, PhysicalMbarrierCompletionTargets, RuntimeBuffer,
        TcgenPipelineOperation, allocate_cta_tmem, load_tmem_scalar_warp, run_kernel_engine_launch,
        store_scalar_warp, store_tmem_scalar_warp,
    };
    use crate::{
        LaunchTopology, OperationContext, OperationEffect, OperationKind, PhysicalAccessBatch,
        PhysicalAccessDescriptor, PhysicalAccessKind, PhysicalAccessSpace, PhysicalAllocationId,
        PhysicalBarrierId, PhysicalByteSpan, PhysicalMemory, StaticOpId, TmemAccessMode, WarpMask,
        WarpValue,
    };

    #[test]
    fn completion_payload_uses_cta_domain_locally_and_cluster_domain_remotely() {
        assert_eq!(
            super::async_completion_payload_proxy_domain(2, 2),
            super::ProxyMemoryDomain::SharedCta,
        );
        assert_eq!(
            super::async_completion_payload_proxy_domain(2, 3),
            super::ProxyMemoryDomain::SharedCluster,
        );
    }

    struct AsyncPayloadProbeMode;

    struct PollingProbeMode;

    struct DeclaredWaitTransactionProbe;

    #[derive(Default)]
    struct DeclaredWaitGuard<'a>(Option<&'a AtomicBool>);

    impl Drop for DeclaredWaitGuard<'_> {
        fn drop(&mut self) {
            if let Some(held) = self.0 {
                assert!(held.swap(false, Ordering::SeqCst));
            }
        }
    }

    impl crate::engine_mode::EngineModeImpl for DeclaredWaitTransactionProbe {
        type LaunchState = AtomicBool;
        type GlobalMemoryTransactionGuard<'a> = DeclaredWaitGuard<'a>;
        const NAME: &'static str = "declared-wait-transaction-probe";
        const OBSERVES_OPERATIONS: bool = true;
        const USES_GLOBAL_MEMORY_TRANSACTION: bool = true;

        fn begin_global_memory_transaction<'a>(
            state: &'a AtomicBool,
            exclusive: bool,
            spans: &[PhysicalByteSpan],
        ) -> Result<DeclaredWaitGuard<'a>, crate::EngineError> {
            assert!(!exclusive);
            assert_eq!(spans.len(), 1);
            assert!(!state.swap(true, Ordering::SeqCst));
            Ok(DeclaredWaitGuard(Some(state)))
        }

        fn before_operation(_: &AtomicBool, _: &OperationContext) -> Result<(), crate::EngineError> {
            Ok(())
        }

        fn after_operation(_: &AtomicBool, _: &OperationContext) -> Result<(), crate::EngineError> {
            Ok(())
        }

        fn after_effect(
            state: &AtomicBool,
            _: &OperationContext,
            effect: OperationEffect<'_>,
        ) -> Result<(), crate::EngineError> {
            if matches!(effect, OperationEffect::DeclaredWordWait { .. } | OperationEffect::PhysicalAccess(_)) {
                assert!(state.load(Ordering::SeqCst), "exit evidence must commit under the transaction");
            }
            Ok(())
        }
    }

    #[test]
    fn declared_wait_holds_transaction_through_exit_evidence() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocation = physical.global().allocate_from_bytes(7_u32.to_le_bytes()).unwrap();
        let pointer = crate::runtime::PhysicalPtr::new(
            RuntimeBuffer::Global(physical.global().full_view(allocation).unwrap()),
            WarpValue::splat(0_i64),
            4,
        );
        let held = Arc::new(AtomicBool::new(false));
        let probe = Arc::clone(&held);
        run_kernel_engine_launch::<DeclaredWaitTransactionProbe, _, _>(
            physical, 0, Arc::clone(&held), LaunchSelection::default(), 1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let pointer = pointer.clone();
                let probe = Arc::clone(&probe);
                async move {
                    let mask = WarpMask::from_lanes([0]).unwrap();
                    let operation = warp.begin_current_operation(
                        warp.context().with_active_mask(mask),
                        StaticOpId::new(1), OperationKind::Load,
                    )?;
                    let observed = warp.declared_word_wait_until(
                        Some(&operation), &pointer, mask, 4,
                        super::MemoryAccessSemantics::plain(),
                        |value, _| {
                            Ok(value == 7)
                        },
                    ).await?;
                    assert_eq!(observed[0], 7);
                    assert!(!probe.load(Ordering::SeqCst));
                    Ok(())
                }
            },
        ).unwrap();
        assert!(!held.load(Ordering::SeqCst));
    }

    #[test]
    fn declared_wait_revalidates_after_pending_publication() {
        use std::future::Future;
        use std::task::{Context, Waker};

        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let physical = PhysicalMemory::new(topology);
        physical.enable_semantic_progress();
        let allocation = physical.global().allocate_from_bytes(7_u32.to_le_bytes()).unwrap();
        let pointer = crate::runtime::PhysicalPtr::new(
            RuntimeBuffer::Global(physical.global().full_view(allocation).unwrap()),
            WarpValue::splat(0_i64), 4,
        );
        let held = Arc::new(AtomicBool::new(false));
        let probe = Arc::clone(&held);
        run_kernel_engine_launch::<DeclaredWaitTransactionProbe, _, _>(
            physical, 0, Arc::clone(&held), LaunchSelection::default(), 1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let pointer = pointer.clone();
                let probe = Arc::clone(&probe);
                async move {
                    if warp.context().global_warp_id() != 0 {
                        return Ok(());
                    }
                    let mask = WarpMask::from_lanes([0]).unwrap();
                    let publisher = crate::WarpContext::from_topology(topology, 1)
                        .with_active_mask(mask);
                    let ordering = warp.kernel().services().ordering();
                    let address = pointer.resolve_uniform(&publisher, mask)?;
                    ordering.prepare_atomic_access(publisher, [(address, 4)])?;
                    let publication = ordering
                        .reserve_prepared_atomic_access_async(publisher, true)
                        .await?
                        .unwrap();
                    let operation = warp.begin_current_operation(
                        warp.context().with_active_mask(mask),
                        StaticOpId::new(1), OperationKind::Load,
                    )?;
                    let wait = warp.declared_word_wait_until(
                        Some(&operation), &pointer, mask, 4,
                        super::MemoryAccessSemantics::plain(),
                        |value, _| Ok(value == 7),
                    );
                    let mut wait = std::pin::pin!(wait);
                    let mut cx = Context::from_waker(Waker::noop());
                    // Bytes already satisfy the predicate, but the publisher
                    // has not finished its metadata: the waiter must park.
                    assert!(matches!(wait.as_mut().poll(&mut cx), Poll::Pending));
                    assert!(!probe.load(Ordering::SeqCst));
                    store_scalar_warp(
                        warp.kernel().physical(), &publisher, pointer.buffer(),
                        &WarpValue::splat(0_i64), &WarpValue::splat(0_u32), mask,
                    )?;
                    drop(publication);
                    // The accepted value changed while acquiring protection.
                    // It must be reread, and both guards released before sleep.
                    assert!(matches!(wait.as_mut().poll(&mut cx), Poll::Pending));
                    assert!(!probe.load(Ordering::SeqCst));
                    ordering.prepare_atomic_access(publisher, [(address, 4)])?;
                    let next = ordering.reserve_prepared_atomic_access_async(publisher, true);
                    let mut next = std::pin::pin!(next);
                    let publication = match next.as_mut().poll(&mut cx) {
                        Poll::Ready(Ok(Some(guard))) => guard,
                        _ => panic!("a sleeping waiter must not retain its reservation"),
                    };
                    store_scalar_warp(
                        warp.kernel().physical(), &publisher, pointer.buffer(),
                        &WarpValue::splat(0_i64), &WarpValue::splat(7_u32), mask,
                    )?;
                    drop(publication);
                    match wait.as_mut().poll(&mut cx) {
                        Poll::Ready(Ok(value)) => assert_eq!(value[0], 7),
                        _ => panic!("committed publication must wake and complete the wait"),
                    }
                    assert!(!probe.load(Ordering::SeqCst));
                    Ok(())
                }
            },
        ).unwrap();
        assert!(!held.load(Ordering::SeqCst));
    }

    struct ObservingTmemProgressProbeMode;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum ObservingTmemProgressPath {
        Uncontrolled,
        UnobservedBatch,
        CompactBatch,
        FullBatch,
    }

    struct ObservingTmemProgressProbeState {
        path: ObservingTmemProgressPath,
        physical: PhysicalMemory,
        initial_progress: super::PhysicalSemanticProgressSnapshot,
        mode_hook_count: AtomicUsize,
    }

    impl ObservingTmemProgressProbeState {
        fn record_mode_hook_before_publication(&self) {
            assert_eq!(
                self.physical.semantic_progress_snapshot(),
                self.initial_progress,
                "semantic progress must publish after the observing-mode hook",
            );
            self.mode_hook_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl crate::engine_mode::EngineModeImpl for PollingProbeMode {
        type LaunchState = ();
        type GlobalMemoryTransactionGuard<'a> = ();

        const NAME: &'static str = "polling-probe";
        const OBSERVES_OPERATIONS: bool = false;

        fn before_operation(
            _state: &Self::LaunchState,
            _operation: &OperationContext,
        ) -> Result<(), crate::EngineError> {
            Ok(())
        }

        fn after_operation(
            _state: &Self::LaunchState,
            _operation: &OperationContext,
        ) -> Result<(), crate::EngineError> {
            Ok(())
        }
    }

    impl crate::engine_mode::EngineModeImpl for ObservingTmemProgressProbeMode {
        type LaunchState = ObservingTmemProgressProbeState;
        type GlobalMemoryTransactionGuard<'a> = ();

        const NAME: &'static str = "observing-tmem-progress-probe";
        const OBSERVES_OPERATIONS: bool = true;

        fn controls_physical_access(
            state: &Self::LaunchState,
            _kind: OperationKind,
            _space: PhysicalAccessSpace,
        ) -> bool {
            state.path != ObservingTmemProgressPath::Uncontrolled
        }

        fn observes_physical_access_batch(
            state: &Self::LaunchState,
            _descriptor: PhysicalAccessDescriptor,
            _mask: WarpMask,
        ) -> bool {
            state.path != ObservingTmemProgressPath::UnobservedBatch
        }

        fn compacts_direct_physical_access(
            state: &Self::LaunchState,
            _descriptor: PhysicalAccessDescriptor,
            _mask: WarpMask,
            _atomic_return_sync_relevant: bool,
        ) -> bool {
            state.path == ObservingTmemProgressPath::CompactBatch
        }

        fn before_compact_physical_access(
            state: &Self::LaunchState,
            _batch: &crate::physical_access::CompactPhysicalAccessBatch<'_>,
        ) -> Result<(), crate::EngineError> {
            state.record_mode_hook_before_publication();
            Ok(())
        }

        fn after_compact_physical_access(
            state: &Self::LaunchState,
            _batch: &crate::physical_access::CompactPhysicalAccessBatch<'_>,
        ) -> Result<(), crate::EngineError> {
            state.record_mode_hook_before_publication();
            Ok(())
        }

        fn before_operation(
            _state: &Self::LaunchState,
            _operation: &OperationContext,
        ) -> Result<(), crate::EngineError> {
            Ok(())
        }

        fn after_operation(
            _state: &Self::LaunchState,
            _operation: &OperationContext,
        ) -> Result<(), crate::EngineError> {
            Ok(())
        }

        fn before_effect(
            state: &Self::LaunchState,
            _operation: &OperationContext,
            effect: OperationEffect<'_>,
        ) -> Result<(), crate::EngineError> {
            if matches!(effect, OperationEffect::PhysicalAccess(_)) {
                state.record_mode_hook_before_publication();
            }
            Ok(())
        }

        fn after_effect(
            state: &Self::LaunchState,
            _operation: &OperationContext,
            effect: OperationEffect<'_>,
        ) -> Result<(), crate::EngineError> {
            if matches!(effect, OperationEffect::PhysicalAccess(_)) {
                state.record_mode_hook_before_publication();
            }
            Ok(())
        }

        fn after_unobserved_physical_access(
            state: &Self::LaunchState,
            _global_warp_id: usize,
        ) -> Result<(), crate::EngineError> {
            state.record_mode_hook_before_publication();
            Ok(())
        }
    }

    struct AsyncPayloadProbeState {
        reject: bool,
        before_count: AtomicUsize,
        after_count: AtomicUsize,
    }

    impl crate::engine_mode::EngineModeImpl for AsyncPayloadProbeMode {
        type LaunchState = AsyncPayloadProbeState;
        type GlobalMemoryTransactionGuard<'a> = ();

        const NAME: &'static str = "async-payload-probe";
        const OBSERVES_OPERATIONS: bool = true;

        fn before_operation(
            _state: &Self::LaunchState,
            _operation: &OperationContext,
        ) -> Result<(), crate::EngineError> {
            Ok(())
        }

        fn after_operation(
            _state: &Self::LaunchState,
            _operation: &OperationContext,
        ) -> Result<(), crate::EngineError> {
            Ok(())
        }

        fn before_effect(
            state: &Self::LaunchState,
            _operation: &OperationContext,
            effect: OperationEffect<'_>,
        ) -> Result<(), crate::EngineError> {
            if matches!(effect, OperationEffect::AsyncPayload(_)) {
                state.before_count.fetch_add(1, Ordering::SeqCst);
                if state.reject {
                    return Err(crate::EngineError::message(
                        "async payload rejected before numeric execution",
                    ));
                }
            }
            Ok(())
        }

        fn after_effect(
            state: &Self::LaunchState,
            _operation: &OperationContext,
            effect: OperationEffect<'_>,
        ) -> Result<(), crate::EngineError> {
            if matches!(effect, OperationEffect::AsyncPayload(_)) {
                state.after_count.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }
    }

    fn one_lane_write(operation: &OperationContext) -> PhysicalAccessBatch {
        let descriptor = PhysicalAccessDescriptor::new(
            PhysicalAccessKind::Write,
            PhysicalAccessSpace::Shared,
            4,
        )
        .unwrap();
        PhysicalAccessBatch::resolve(operation.clone(), descriptor, |_| {
            Ok::<_, std::convert::Infallible>(vec![
                PhysicalByteSpan::new(PhysicalAllocationId::new(17), 0, 4).unwrap(),
            ])
        })
        .unwrap()
    }

    #[cfg(feature = "analysis-core")]
    #[test]
    fn stuttering_while_checkpoint_resumes_after_semantic_memory_progress() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocation = physical
            .global()
            .allocate_from_bytes(0_i32.to_le_bytes())
            .unwrap();
        let signal = RuntimeBuffer::Global(physical.global().full_view(allocation).unwrap());
        let load_completed = Arc::new(AtomicBool::new(false));
        let load_completed_by_warp = Arc::clone(&load_completed);
        let observed_suspension = Arc::new(AtomicBool::new(false));
        let observed_suspension_by_warp = Arc::clone(&observed_suspension);

        run_kernel_engine_launch::<PollingProbeMode, _, _>(
            physical,
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |warp| {
                let signal = signal.clone();
                let load_completed = Arc::clone(&load_completed_by_warp);
                let observed_suspension = Arc::clone(&observed_suspension_by_warp);
                async move {
                    let lane_zero = WarpMask::from_lanes([0]).unwrap();
                    let context = warp.context().with_active_mask(lane_zero);
                    let indices = WarpValue::splat(0_i64);
                    if context.warp_id_in_cta() == 0 {
                        let mut progress = warp.begin_while_loop_progress(lane_zero);
                        let value = warp.load_named_scalar::<i32>(
                            None, context, 4, "signal", &signal, &indices, lane_zero, None,
                        )?;
                        assert_eq!(value[0], 0);
                        load_completed.store(true, Ordering::SeqCst);
                        warp.while_loop_checkpoint(&mut progress, lane_zero).await?;
                        observed_suspension.store(true, Ordering::SeqCst);
                    } else {
                        poll_fn(|cx| {
                            if load_completed.load(Ordering::SeqCst) {
                                Poll::Ready(())
                            } else {
                                cx.waker().wake_by_ref();
                                Poll::Pending
                            }
                        })
                        .await;
                        store_scalar_warp(
                            warp.kernel().physical(),
                            &context,
                            &signal,
                            &indices,
                            &WarpValue::splat(1_i32),
                            lane_zero,
                        )?;
                    }
                    Ok(())
                }
            },
        )
        .unwrap();

        assert!(observed_suspension.load(Ordering::SeqCst));
    }

    #[test]
    fn a_stalled_while_checkpoint_is_a_reportable_blocked_operation() {
        // Before park reasons were recorded, a warp suspended here was invisible
        // to `blocked_operations`: the semantic-progress watch keys its waiter
        // table by an anonymous counter with no warp or operation, so nothing
        // could reconstruct it. Warp 1 observes the live park, then releases
        // warp 0 — proving the record exists exactly while the warp is parked.
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocation = physical
            .global()
            .allocate_from_bytes(0_i32.to_le_bytes())
            .unwrap();
        let signal = RuntimeBuffer::Global(physical.global().full_view(allocation).unwrap());
        let observed = Arc::new(Mutex::new(Vec::<String>::new()));
        let observed_by_warp = Arc::clone(&observed);

        run_kernel_engine_launch::<PollingProbeMode, _, _>(
            physical,
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |warp| {
                let signal = signal.clone();
                let observed = Arc::clone(&observed_by_warp);
                async move {
                    let lane_zero = WarpMask::from_lanes([0]).unwrap();
                    let context = warp.context().with_active_mask(lane_zero);
                    let indices = WarpValue::splat(0_i64);
                    if context.warp_id_in_cta() == 0 {
                        let mut progress = warp.begin_while_loop_progress(lane_zero);
                        warp.while_loop_checkpoint(&mut progress, lane_zero).await?;
                    } else {
                        let ordering = warp.kernel().services().ordering();
                        // Bounded on purpose: if the park is ever *not*
                        // recorded, this must fail the assertion below rather
                        // than spin forever, because a self-waking poll keeps
                        // the worker Active and so suppresses deadlock
                        // detection — the very blind spot under test.
                        let mut polls_left = 1_000_usize;
                        poll_fn(|cx| {
                            let parked = ordering.parked_operations();
                            if parked.is_empty() && polls_left > 0 {
                                polls_left -= 1;
                                cx.waker().wake_by_ref();
                                return Poll::Pending;
                            }
                            *observed.lock().unwrap() =
                                parked.iter().map(ToString::to_string).collect();
                            Poll::Ready(())
                        })
                        .await;
                        store_scalar_warp(
                            warp.kernel().physical(),
                            &context,
                            &signal,
                            &indices,
                            &WarpValue::splat(1_i32),
                            lane_zero,
                        )?;
                    }
                    Ok(())
                }
            },
        )
        .unwrap();

        let observed = observed.lock().unwrap().clone();
        assert_eq!(observed.len(), 1, "exactly warp 0 should be parked");
        assert!(
            observed[0].starts_with("warp 0 awaits engine.semantic_progress"),
            "the report must name the warp and why it parked, got {:?}",
            observed[0]
        );
    }

    #[test]
    fn numeric_tcgen_issue_wakes_a_stalled_while_checkpoint() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let checkpoint_started = Arc::new(AtomicBool::new(false));
        let checkpoint_started_by_warp = Arc::clone(&checkpoint_started);
        let resumed = Arc::new(AtomicBool::new(false));
        let resumed_by_warp = Arc::clone(&resumed);

        run_kernel_engine_launch::<PollingProbeMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |warp| {
                let checkpoint_started = Arc::clone(&checkpoint_started_by_warp);
                let resumed = Arc::clone(&resumed_by_warp);
                async move {
                    let live = WarpMask::from_lanes([0]).unwrap();
                    if warp.context().warp_id_in_cta() == 0 {
                        let mut progress = warp.begin_while_loop_progress(live);
                        checkpoint_started.store(true, Ordering::SeqCst);
                        warp.while_loop_checkpoint(&mut progress, live).await?;
                        resumed.store(true, Ordering::SeqCst);
                    } else {
                        poll_fn(|cx| {
                            if checkpoint_started.load(Ordering::SeqCst) {
                                Poll::Ready(())
                            } else {
                                cx.waker().wake_by_ref();
                                Poll::Pending
                            }
                        })
                        .await;
                        warp.tcgen_instruction_issue(
                            None,
                            1,
                            TcgenPipelineOperation::Store,
                            None,
                            |_, _| Ok(()),
                            || Ok(()),
                        )?;
                    }
                    Ok(())
                }
            },
        )
        .unwrap();

        assert!(resumed.load(Ordering::SeqCst));
    }

    #[test]
    fn numeric_typed_tmem_store_wakes_a_stalled_while_checkpoint() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocations = Arc::new(allocate_cta_tmem(&physical, topology, 1, 1).unwrap());
        let buffer = RuntimeBuffer::Tmem {
            allocations,
            lane_span: 1,
            tcol_span_elements: 1,
            elem_offset: 0,
            itemsize: std::mem::size_of::<u32>(),
        };
        let checkpoint_started = Arc::new(AtomicBool::new(false));
        let checkpoint_started_by_warp = Arc::clone(&checkpoint_started);
        let resumed = Arc::new(AtomicBool::new(false));
        let resumed_by_warp = Arc::clone(&resumed);
        let stored_value = Arc::new(AtomicU32::new(0));
        let stored_value_by_warp = Arc::clone(&stored_value);

        run_kernel_engine_launch::<PollingProbeMode, _, _>(
            physical.clone(),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |warp| {
                let buffer = buffer.clone();
                let checkpoint_started = Arc::clone(&checkpoint_started_by_warp);
                let resumed = Arc::clone(&resumed_by_warp);
                let stored_value = Arc::clone(&stored_value_by_warp);
                async move {
                    let live = WarpMask::from_lanes([0]).unwrap();
                    let context = warp.context().with_active_mask(live);
                    if context.warp_id_in_cta() == 0 {
                        let mut progress = warp.begin_while_loop_progress(live);
                        checkpoint_started.store(true, Ordering::SeqCst);
                        warp.while_loop_checkpoint(&mut progress, live).await?;
                        resumed.store(true, Ordering::SeqCst);
                    } else {
                        poll_fn(|cx| {
                            if checkpoint_started.load(Ordering::SeqCst) {
                                Poll::Ready(())
                            } else {
                                cx.waker().wake_by_ref();
                                Poll::Pending
                            }
                        })
                        .await;
                        let physical = warp.kernel().physical().clone();
                        let lifecycle = warp.kernel().services().tcgen();
                        let coordinates = WarpValue::splat(0_i64);
                        let values = WarpValue::splat(0x1234_5678_u32);
                        warp.runtime_named_tmem_physical_access(
                            None,
                            OperationKind::Store,
                            std::mem::size_of::<u32>(),
                            "typed_tmem",
                            TmemAccessMode::Static,
                            &buffer,
                            &coordinates,
                            &coordinates,
                            &coordinates,
                            live,
                            false,
                            || {
                                store_tmem_scalar_warp(
                                    &physical,
                                    &context,
                                    &lifecycle,
                                    TmemAccessMode::Static,
                                    &buffer,
                                    &coordinates,
                                    &coordinates,
                                    &coordinates,
                                    &values,
                                    live,
                                )
                            },
                        )?;
                        let observed = load_tmem_scalar_warp::<u32>(
                            &physical,
                            &context,
                            &lifecycle,
                            TmemAccessMode::Static,
                            &buffer,
                            &coordinates,
                            &coordinates,
                            &coordinates,
                            live,
                        )?;
                        stored_value.store(observed[0], Ordering::SeqCst);
                    }
                    Ok(())
                }
            },
        )
        .unwrap();

        assert!(resumed.load(Ordering::SeqCst));
        assert_eq!(stored_value.load(Ordering::SeqCst), 0x1234_5678);
    }

    #[test]
    fn observing_typed_tmem_store_paths_publish_after_mode_effects() {
        for path in [
            ObservingTmemProgressPath::Uncontrolled,
            ObservingTmemProgressPath::UnobservedBatch,
            ObservingTmemProgressPath::CompactBatch,
            ObservingTmemProgressPath::FullBatch,
        ] {
            let topology = LaunchTopology::new(1, 1, 1).unwrap();
            let physical = PhysicalMemory::new(topology);
            physical.enable_semantic_progress();
            let initial_progress = physical.semantic_progress_snapshot();
            let allocations = Arc::new(allocate_cta_tmem(&physical, topology, 1, 1).unwrap());
            let buffer = RuntimeBuffer::Tmem {
                allocations,
                lane_span: 1,
                tcol_span_elements: 1,
                elem_offset: 0,
                itemsize: std::mem::size_of::<u32>(),
            };
            let state = Arc::new(ObservingTmemProgressProbeState {
                path,
                physical: physical.clone(),
                initial_progress,
                mode_hook_count: AtomicUsize::new(0),
            });

            run_kernel_engine_launch::<ObservingTmemProgressProbeMode, _, _>(
                physical.clone(),
                0,
                Arc::clone(&state),
                LaunchSelection::default(),
                1,
                ExecutionPolicy::default(),
                move |mut warp| {
                    let buffer = buffer.clone();
                    async move {
                        let live = WarpMask::from_lanes([0]).unwrap();
                        let context = warp.context().with_active_mask(live);
                        let operation = if path == ObservingTmemProgressPath::Uncontrolled {
                            None
                        } else {
                            Some(warp.begin_current_operation(
                                context,
                                StaticOpId::new(1),
                                OperationKind::Store,
                            )?)
                        };
                        let physical = warp.kernel().physical().clone();
                        let lifecycle = warp.kernel().services().tcgen();
                        let coordinates = WarpValue::splat(0_i64);
                        warp.runtime_named_tmem_physical_access(
                            operation.as_ref(),
                            OperationKind::Store,
                            std::mem::size_of::<u32>(),
                            "typed_tmem",
                            TmemAccessMode::Static,
                            &buffer,
                            &coordinates,
                            &coordinates,
                            &coordinates,
                            live,
                            false,
                            || {
                                store_tmem_scalar_warp(
                                    &physical,
                                    &context,
                                    &lifecycle,
                                    TmemAccessMode::Static,
                                    &buffer,
                                    &coordinates,
                                    &coordinates,
                                    &coordinates,
                                    &WarpValue::splat(0x1234_5678_u32),
                                    live,
                                )
                            },
                        )?;
                        warp.finish_optional_operation(&operation)?;
                        Ok(())
                    }
                },
            )
            .unwrap_or_else(|error| panic!("{path:?} path failed: {error}"));

            let expected_hook_count = match path {
                ObservingTmemProgressPath::Uncontrolled
                | ObservingTmemProgressPath::UnobservedBatch => 1,
                ObservingTmemProgressPath::CompactBatch | ObservingTmemProgressPath::FullBatch => 2,
            };
            assert_eq!(
                state.mode_hook_count.load(Ordering::SeqCst),
                expected_hook_count,
                "{path:?} must execute its observing-mode hooks",
            );
            assert_ne!(
                physical.semantic_progress_snapshot(),
                initial_progress,
                "{path:?} must publish semantic progress after a successful write",
            );
        }
    }

    #[test]
    fn progressing_while_checkpoint_defers_to_ordinary_ready_work() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let stats = run_kernel_engine_launch::<PollingProbeMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |warp| async move {
                if warp.context().warp_id_in_cta() == 0 {
                    let live = WarpMask::from_lanes([0]).unwrap();
                    let mut progress = warp.begin_while_loop_progress(live);
                    warp.record_local_engine_progress();
                    warp.while_loop_checkpoint(&mut progress, live).await?;
                } else {
                    crate::reschedule().await;
                }
                Ok(())
            },
        )
        .unwrap();

        // Even a progressing while quantum stays behind ordinary work. Warp 1
        // therefore completes its normal reschedule before warp 0 rechecks.
        assert_eq!(stats.poll_order, vec![0, 1, 1, 0]);
    }

    #[test]
    fn checker_rejection_prevents_numeric_async_payload_mutation() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let state = Arc::new(AsyncPayloadProbeState {
            reject: true,
            before_count: AtomicUsize::new(0),
            after_count: AtomicUsize::new(0),
        });
        let numeric_mutations = Arc::new(AtomicUsize::new(0));
        let numeric_by_warp = Arc::clone(&numeric_mutations);

        let error = run_kernel_engine_launch::<AsyncPayloadProbeMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let numeric_mutations = Arc::clone(&numeric_by_warp);
                async move {
                    let context = warp
                        .context()
                        .with_active_mask(WarpMask::from_lanes([0]).unwrap());
                    let operation = warp.begin_operation(
                        context,
                        StaticOpId::new(501),
                        OperationKind::AsyncIssue,
                        [],
                    )?;
                    let completion_targets =
                        PhysicalMbarrierCompletionTargets::single(PhysicalBarrierId::new(0, 0, 0));
                    warp.async_payload_issue_with_resolved_targets(
                        Some(&operation),
                        &completion_targets,
                        |operation| Ok((vec![one_lane_write(operation)], 0)),
                        || {
                            numeric_mutations.fetch_add(1, Ordering::SeqCst);
                            Ok(((), 0))
                        },
                    )?;
                    Ok(())
                }
            },
        )
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("async payload rejected before numeric execution"));
        assert_eq!(numeric_mutations.load(Ordering::SeqCst), 0);
        assert_eq!(state.before_count.load(Ordering::SeqCst), 1);
        assert_eq!(state.after_count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn runtime_enqueue_failure_is_after_numeric_payload_and_before_analysis_commit() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let state = Arc::new(AsyncPayloadProbeState {
            reject: false,
            before_count: AtomicUsize::new(0),
            after_count: AtomicUsize::new(0),
        });
        let numeric_mutations = Arc::new(AtomicUsize::new(0));
        let numeric_by_warp = Arc::clone(&numeric_mutations);

        let error = run_kernel_engine_launch::<AsyncPayloadProbeMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::clone(&state),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let numeric_mutations = Arc::clone(&numeric_by_warp);
                async move {
                    let context = warp
                        .context()
                        .with_active_mask(WarpMask::from_lanes([0]).unwrap());
                    let operation = warp.begin_operation(
                        context,
                        StaticOpId::new(502),
                        OperationKind::AsyncIssue,
                        [],
                    )?;
                    let completion_targets =
                        PhysicalMbarrierCompletionTargets::single(PhysicalBarrierId::new(0, 0, 0));
                    warp.async_payload_issue_with_resolved_targets(
                        Some(&operation),
                        &completion_targets,
                        |operation| Ok((vec![one_lane_write(operation)], 4)),
                        || {
                            numeric_mutations.fetch_add(1, Ordering::SeqCst);
                            Ok(((), 4))
                        },
                    )?;
                    Ok(())
                }
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("before mbarrier.init"));
        assert_eq!(numeric_mutations.load(Ordering::SeqCst), 1);
        assert_eq!(state.before_count.load(Ordering::SeqCst), 1);
        assert_eq!(state.after_count.load(Ordering::SeqCst), 0);
    }
}
