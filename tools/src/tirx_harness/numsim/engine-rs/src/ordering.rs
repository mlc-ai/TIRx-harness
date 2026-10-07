use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

use crate::{
    runtime::{TcgenWorkIssue, TcgenWorkKind, TcgenWorkSet},
    AsyncTokenId, DiagnosticLabel, EngineError, LaunchTopology, PhysicalAddress, ProfileKind,
    ProfileTimer, WarpContext, WarpMask, WARP_SIZE,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TcgenTransferKind {
    Load,
    Store,
}

#[derive(Clone, Debug)]
struct SetmaxnregOccurrence {
    increase: bool,
    register_count: u32,
    arrived_warps: BTreeSet<usize>,
    completed_warps: BTreeSet<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct MemoryRange {
    allocation_id: u64,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
struct PreparedAtomicAccess {
    ranges: Vec<MemoryRange>,
}

#[derive(Clone, Debug)]
struct PendingAtomicAccess {
    operation_id: u64,
    ranges: Vec<MemoryRange>,
    value_started: bool,
    value_complete: bool,
}

#[derive(Clone, Debug)]
struct AtomicReservation {
    warp_id: usize,
    ranges: Vec<MemoryRange>,
    exclusive: bool,
}

#[derive(Default)]
struct AtomicSerialState {
    next_operation_id: u64,
    reservations: BTreeMap<u64, AtomicReservation>,
    waiters: BTreeMap<usize, AtomicReservationWaiter>,
}

struct AtomicReservationWaiter {
    ranges: Vec<MemoryRange>,
    exclusive: bool,
    waker: Waker,
}

#[derive(Clone, Debug, Default)]
struct ThreadTcgenState {
    commit_by_cta_group: BTreeMap<u32, Vec<AsyncTokenId>>,
    mma_shared_a_by_cta_group: BTreeMap<u32, Vec<AsyncTokenId>>,
    loads: Vec<AsyncTokenId>,
    stores: Vec<AsyncTokenId>,
}

impl ThreadTcgenState {
    fn pending_work(&self, kind: TcgenWorkKind, cta_group: Option<u32>) -> Vec<AsyncTokenId> {
        match kind {
            TcgenWorkKind::Commit | TcgenWorkKind::MmaSharedARead => {
                let group = cta_group.expect("commit work retains CTA group");
                let mut tokens = self
                    .mma_shared_a_by_cta_group
                    .get(&group)
                    .cloned()
                    .unwrap_or_default();
                if kind == TcgenWorkKind::Commit {
                    tokens.extend(
                        self.commit_by_cta_group
                            .get(&group)
                            .into_iter()
                            .flatten()
                            .cloned(),
                    );
                }
                tokens
            }
            TcgenWorkKind::Load => self.loads.clone(),
            TcgenWorkKind::Store => self.stores.clone(),
        }
    }
}

struct WarpOrderingState {
    prepared_atomic_access: Option<PreparedAtomicAccess>,
    pending_atomic_access: Option<PendingAtomicAccess>,
    setmaxnreg_next_occurrence: u64,
    tcgen_threads: [ThreadTcgenState; WARP_SIZE],
}

impl Default for WarpOrderingState {
    fn default() -> Self {
        Self {
            prepared_atomic_access: None,
            pending_atomic_access: None,
            setmaxnreg_next_occurrence: 0,
            tcgen_threads: std::array::from_fn(|_| ThreadTcgenState::default()),
        }
    }
}

#[derive(Default)]
struct CoordinationState {
    setmaxnreg_occurrences: BTreeMap<(usize, u64), SetmaxnregOccurrence>,
    setmaxnreg_requires_sync: BTreeSet<usize>,
}

/// Why a warp is not runnable, recorded by the warp itself as it parks.
///
/// Every other blocked-operation source in the engine reconstructs its answer
/// at report time by walking a hub's live waiter table. That works only for
/// waits a hub owns. These two are owned by nobody:
///
/// - `AtomicLinearization` blocks the OS worker on a `Condvar`, not the warp
///   future, so the scheduler never even learns the warp stopped.
/// - `SemanticProgress` parks on a `memory.rs` watch whose waiter table is
///   keyed by an anonymous counter carrying no warp or operation, so it cannot
///   be reconstructed after the fact even in principle.
///
/// Recording the reason *with* the park is the only way either can appear in a
/// deadlock report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ParkReason {
    /// Global atomic RMW linearization: this warp's byte ranges overlap a
    /// reservation held by a peer.
    AtomicLinearization { blocking_warps: Vec<usize> },
    /// A native `while` quantum observed no semantic progress and is waiting
    /// for any warp to publish some.
    SemanticProgress,
    /// A wait on a declared synchronization word whose predicate does not hold
    /// yet. Unlike the `while` quantum this one names its operation, because a
    /// real instruction is in scope: the report can point at the wait itself.
    DeclaredWordWait,
}

impl ParkReason {
    const fn awaited_operation(&self) -> crate::AwaitedOperation {
        match self {
            Self::AtomicLinearization { .. } => crate::AwaitedOperation::AtomicLinearization,
            Self::SemanticProgress => crate::AwaitedOperation::SemanticProgress,
            Self::DeclaredWordWait => crate::AwaitedOperation::DeclaredWordWait,
        }
    }

    fn static_op_id(&self) -> u64 {
        match self {
            Self::AtomicLinearization { .. } => 0,
            Self::SemanticProgress => 1,
            Self::DeclaredWordWait => 2,
        }
    }
}

#[derive(Clone, Debug)]
struct ParkRecord {
    reason: ParkReason,
    operation: Option<crate::DynamicOpId>,
}

/// Launch-wide record of warps parked outside any completion source.
#[derive(Default)]
pub(crate) struct ParkRegistry {
    parked: Mutex<BTreeMap<usize, ParkRecord>>,
}

impl ParkRegistry {
    fn enter(&self, warp_id: usize, reason: ParkReason, operation: Option<crate::DynamicOpId>) {
        self.parked
            .lock()
            .expect("park registry mutex poisoned")
            .insert(warp_id, ParkRecord { reason, operation });
    }

    fn leave(&self, warp_id: usize) {
        self.parked
            .lock()
            .expect("park registry mutex poisoned")
            .remove(&warp_id);
    }

    fn blocked_operations(&self) -> Vec<crate::BlockedOperation> {
        let parked = self.parked.lock().expect("park registry mutex poisoned");
        parked
            .iter()
            .map(|(&warp_id, record)| {
                let missing = match &record.reason {
                    ParkReason::AtomicLinearization { blocking_warps } => blocking_warps.clone(),
                    ParkReason::SemanticProgress | ParkReason::DeclaredWordWait => Vec::new(),
                };
                crate::BlockedOperation::new(
                    warp_id,
                    record.reason.awaited_operation(),
                    crate::OccurrenceKey::new(
                        record.reason.static_op_id(),
                        "engine.park",
                        std::iter::empty::<i64>(),
                        crate::ScopeInstance::Warp {
                            global_warp_id: warp_id,
                        },
                    ),
                    None,
                    crate::ParticipantState {
                        expected: Vec::new(),
                        arrived: Vec::new(),
                        missing,
                        expected_arrival_count: None,
                        completed_arrival_count: None,
                        expected_transactions: None,
                        completed_transactions: None,
                    },
                )
                .with_operation(record.operation.clone())
            })
            .collect()
    }
}

/// RAII park record: the reason is removed however the warp resumes.
pub(crate) struct ParkGuard<'a> {
    registry: &'a ParkRegistry,
    warp_id: usize,
}

impl Drop for ParkGuard<'_> {
    fn drop(&mut self) {
        self.registry.leave(self.warp_id);
    }
}

pub(crate) struct OrderingHub {
    topology: LaunchTopology,
    warps: Box<[Mutex<WarpOrderingState>]>,
    coordination: Mutex<CoordinationState>,
    atomic_serial: Mutex<AtomicSerialState>,
    atomic_serial_changed: Condvar,
    parks: ParkRegistry,
}

pub(crate) struct AtomicAccessGuard<'a> {
    hub: &'a OrderingHub,
    warp_id: usize,
    operation_id: u64,
    completed: bool,
}

/// Keeps an already-linearized physical access reserved across mode callbacks.
///
/// The raw numeric primitive borrows the reservation through
/// `begin_atomic_access`; dropping this outer guard aborts any reservation that
/// did not reach its normal metadata completion path.
pub(crate) struct AtomicReservationGuard {
    hub: Arc<OrderingHub>,
    warp_id: usize,
    operation_id: u64,
}

struct AtomicReservationWait {
    hub: Arc<OrderingHub>,
    warp_id: usize,
    ranges: Vec<MemoryRange>,
    exclusive: bool,
    parked: bool,
}

impl Future for AtomicReservationWait {
    type Output = Result<AtomicReservationGuard, EngineError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let mut serial = self
            .hub
            .atomic_serial
            .lock()
            .expect("atomic serial mutex poisoned");
        if let Some(blocking_warps) =
            overlapping_reservation_warps(&serial, &self.ranges, self.exclusive)
        {
            serial.waiters.insert(
                self.warp_id,
                AtomicReservationWaiter {
                    ranges: self.ranges.clone(),
                    exclusive: self.exclusive,
                    waker: context.waker().clone(),
                },
            );
            drop(serial);
            self.hub.parks.enter(
                self.warp_id,
                ParkReason::AtomicLinearization { blocking_warps },
                None,
            );
            self.parked = true;
            return Poll::Pending;
        }
        serial.waiters.remove(&self.warp_id);
        let operation_id = match serial.next_operation_id.checked_add(1) {
            Some(next) => {
                let operation_id = serial.next_operation_id;
                serial.next_operation_id = next;
                serial.reservations.insert(
                    operation_id,
                    AtomicReservation {
                        warp_id: self.warp_id,
                        ranges: self.ranges.clone(),
                        exclusive: self.exclusive,
                    },
                );
                operation_id
            }
            None => {
                drop(serial);
                return Poll::Ready(Err(EngineError::message(
                    "NumSim atomic linearization ID overflow",
                )));
            }
        };
        drop(serial);
        if self.parked {
            self.hub.parks.leave(self.warp_id);
            self.parked = false;
        }
        if let Err(error) = self.hub.install_pending_atomic_access(
            self.warp_id,
            operation_id,
            self.ranges.clone(),
            false,
        ) {
            return Poll::Ready(Err(error));
        }
        Poll::Ready(Ok(AtomicReservationGuard {
            hub: Arc::clone(&self.hub),
            warp_id: self.warp_id,
            operation_id,
        }))
    }
}

impl Drop for AtomicReservationWait {
    fn drop(&mut self) {
        if self.parked {
            self.hub
                .atomic_serial
                .lock()
                .expect("atomic serial mutex poisoned")
                .waiters
                .remove(&self.warp_id);
            self.hub.parks.leave(self.warp_id);
        }
    }
}

impl Drop for AtomicReservationGuard {
    fn drop(&mut self) {
        self.hub
            .abort_atomic_access(self.warp_id, self.operation_id);
    }
}

impl AtomicAccessGuard<'_> {
    pub(crate) fn mark_value_complete(mut self) -> Result<(), EngineError> {
        self.hub
            .mark_atomic_value_complete(self.warp_id, self.operation_id)?;
        self.completed = true;
        Ok(())
    }

    pub(crate) fn finish_load(mut self) -> Result<(), EngineError> {
        let result = self
            .hub
            .complete_atomic_load(self.warp_id, self.operation_id);
        self.completed = true;
        result
    }
}

impl Drop for AtomicAccessGuard<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.hub
                .abort_atomic_access(self.warp_id, self.operation_id);
        }
    }
}

impl OrderingHub {
    pub(crate) fn new(topology: LaunchTopology) -> Self {
        Self {
            topology,
            warps: (0..topology.warp_count())
                .map(|_| Mutex::new(WarpOrderingState::default()))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            coordination: Mutex::new(CoordinationState::default()),
            atomic_serial: Mutex::new(AtomicSerialState::default()),
            atomic_serial_changed: Condvar::new(),
            parks: ParkRegistry::default(),
        }
    }

    /// Record why `warp_id` is parking. The returned guard clears the record
    /// however the warp resumes, including on unwind.
    pub(crate) fn park(
        &self,
        warp_id: usize,
        reason: ParkReason,
        operation: Option<crate::DynamicOpId>,
    ) -> ParkGuard<'_> {
        self.parks.enter(warp_id, reason, operation);
        ParkGuard {
            registry: &self.parks,
            warp_id,
        }
    }

    /// Warps parked outside every completion source, with the reason they
    /// recorded when they parked.
    pub(crate) fn parked_operations(&self) -> Vec<crate::BlockedOperation> {
        self.parks.blocked_operations()
    }

    pub(crate) fn prepare_atomic_access(
        &self,
        context: WarpContext,
        addresses: impl IntoIterator<Item = (PhysicalAddress, usize)>,
    ) -> Result<(), EngineError> {
        let ranges = physical_memory_ranges(addresses, &DiagnosticLabel::new("atomic access"))?;
        if ranges.is_empty() {
            return Ok(());
        }
        let warp_id = context.global_warp_id();
        let mut warp = self.warps[warp_id]
            .lock()
            .expect("ordering warp mutex poisoned");
        if warp.prepared_atomic_access.is_some() || warp.pending_atomic_access.is_some() {
            return Err(EngineError::message(format!(
                "warp {warp_id} started an atomic access before completing its prior atomic access"
            )));
        }
        warp.prepared_atomic_access = Some(PreparedAtomicAccess { ranges });
        Ok(())
    }

    pub(crate) fn begin_atomic_access(
        &self,
        context: WarpContext,
        addresses: impl IntoIterator<Item = (PhysicalAddress, usize)>,
    ) -> Result<Option<AtomicAccessGuard<'_>>, EngineError> {
        let ranges =
            physical_memory_ranges(addresses, &DiagnosticLabel::new("atomic linearization"))?;
        let warp_id = context.global_warp_id();
        let prepared = {
            let mut warp = self.warps[warp_id]
                .lock()
                .expect("ordering warp mutex poisoned");
            if let Some(pending) = warp.pending_atomic_access.as_mut() {
                if pending.ranges != ranges {
                    return Err(EngineError::message(format!(
                        "warp {warp_id} atomic runtime range {:?} does not match reserved range {:?}",
                        ranges, pending.ranges,
                    )));
                }
                if pending.value_started {
                    return Err(EngineError::message(format!(
                        "warp {warp_id} began the physical value for atomic operation {} twice",
                        pending.operation_id,
                    )));
                }
                pending.value_started = true;
                return Ok(Some(AtomicAccessGuard {
                    hub: self,
                    warp_id,
                    operation_id: pending.operation_id,
                    completed: false,
                }));
            }
            warp.prepared_atomic_access.take()
        };
        let Some(prepared) = prepared else {
            return Ok(None);
        };
        if prepared.ranges != ranges {
            return Err(EngineError::message(format!(
                "warp {warp_id} atomic runtime range {:?} does not match prepared range {:?}",
                ranges, prepared.ranges,
            )));
        }

        let operation_id = self.reserve_atomic_ranges(warp_id, &ranges)?;
        self.install_pending_atomic_access(warp_id, operation_id, ranges, true)?;
        Ok(Some(AtomicAccessGuard {
            hub: self,
            warp_id,
            operation_id,
            completed: false,
        }))
    }

    pub(crate) fn reserve_prepared_atomic_access(
        self: &Arc<Self>,
        context: WarpContext,
    ) -> Result<Option<AtomicReservationGuard>, EngineError> {
        let warp_id = context.global_warp_id();
        let prepared = self.warps[warp_id]
            .lock()
            .expect("ordering warp mutex poisoned")
            .prepared_atomic_access
            .take();
        let Some(prepared) = prepared else {
            return Ok(None);
        };
        let operation_id = self.reserve_atomic_ranges(warp_id, &prepared.ranges)?;
        self.install_pending_atomic_access(warp_id, operation_id, prepared.ranges, false)?;
        Ok(Some(AtomicReservationGuard {
            hub: Arc::clone(self),
            warp_id,
            operation_id,
        }))
    }

    /// Observe a complete publication with a shared reservation, or retain
    /// an exclusive reservation through an atomic's numerical/metadata commit.
    /// These guards serialize implementation state; they add no modeled HB.
    pub(crate) async fn reserve_prepared_atomic_access_async(
        self: &Arc<Self>,
        context: WarpContext,
        exclusive: bool,
    ) -> Result<Option<AtomicReservationGuard>, EngineError> {
        let warp_id = context.global_warp_id();
        let prepared = self.warps[warp_id]
            .lock()
            .expect("ordering warp mutex poisoned")
            .prepared_atomic_access
            .take();
        let Some(prepared) = prepared else {
            return Ok(None);
        };
        AtomicReservationWait {
            hub: Arc::clone(self),
            warp_id,
            ranges: prepared.ranges,
            exclusive,
            parked: false,
        }
        .await
        .map(Some)
    }

    fn reserve_atomic_ranges(
        &self,
        warp_id: usize,
        ranges: &[MemoryRange],
    ) -> Result<u64, EngineError> {
        let _profile = ProfileTimer::new(ProfileKind::AtomicReservationSync);
        let mut serial = self
            .atomic_serial
            .lock()
            .expect("atomic serial mutex poisoned");
        // This blocks the OS worker rather than the warp future, so the
        // scheduler cannot see it. Record the reason before waiting so a
        // deadlock report can name the warp and its blockers.
        let mut park = None;
        while let Some(blocking_warps) = overlapping_reservation_warps(&serial, ranges, true) {
            if park.is_none() {
                park = Some(self.park(
                    warp_id,
                    ParkReason::AtomicLinearization { blocking_warps },
                    None,
                ));
            }
            serial = self
                .atomic_serial_changed
                .wait(serial)
                .expect("atomic serial mutex poisoned while waiting");
        }
        drop(park);
        let operation_id = serial.next_operation_id;
        serial.next_operation_id = operation_id
            .checked_add(1)
            .ok_or_else(|| EngineError::message("NumSim atomic linearization ID overflow"))?;
        serial.reservations.insert(
            operation_id,
            AtomicReservation {
                warp_id,
                ranges: ranges.to_vec(),
                exclusive: true,
            },
        );
        Ok(operation_id)
    }

    fn install_pending_atomic_access(
        &self,
        warp_id: usize,
        operation_id: u64,
        ranges: Vec<MemoryRange>,
        value_started: bool,
    ) -> Result<(), EngineError> {
        let mut warp = self.warps[warp_id]
            .lock()
            .expect("ordering warp mutex poisoned");
        if warp.pending_atomic_access.is_some() {
            drop(warp);
            self.release_atomic_reservation(operation_id, warp_id);
            return Err(EngineError::message(format!(
                "warp {warp_id} has two simultaneous atomic linearization reservations"
            )));
        }
        warp.pending_atomic_access = Some(PendingAtomicAccess {
            operation_id,
            ranges,
            value_started,
            value_complete: false,
        });
        Ok(())
    }

    pub(crate) fn complete_atomic_rmw(
        &self,
        context: WarpContext,
        addresses: impl IntoIterator<Item = (PhysicalAddress, usize)>,
    ) -> Result<(), EngineError> {
        let ranges = physical_memory_ranges(addresses, &DiagnosticLabel::new("atomic completion"))?;
        if ranges.is_empty() {
            return Ok(());
        }
        let warp_id = context.global_warp_id();
        let pending = self.warps[warp_id]
            .lock()
            .expect("ordering warp mutex poisoned")
            .pending_atomic_access
            .take()
            .ok_or_else(|| {
                EngineError::message(format!(
                    "warp {warp_id} completed a global atomic without a linearized RMW"
                ))
            })?;
        let operation_id = pending.operation_id;
        let result = if pending.ranges != ranges {
            Err(EngineError::message(format!(
                "warp {warp_id} completed atomic range {:?}, expected {:?}",
                ranges, pending.ranges,
            )))
        } else if !pending.value_complete {
            Err(EngineError::message(format!(
                "warp {warp_id} completed atomic metadata before its physical RMW"
            )))
        } else {
            Ok(())
        };
        self.release_atomic_reservation(operation_id, warp_id);
        result
    }

    fn mark_atomic_value_complete(
        &self,
        warp_id: usize,
        operation_id: u64,
    ) -> Result<(), EngineError> {
        let mut warp = self.warps[warp_id]
            .lock()
            .expect("ordering warp mutex poisoned");
        let pending = warp.pending_atomic_access.as_mut().ok_or_else(|| {
            EngineError::message(format!(
                "warp {warp_id} completed an atomic value without a pending access"
            ))
        })?;
        if pending.operation_id != operation_id {
            return Err(EngineError::message(format!(
                "warp {warp_id} completed atomic operation {operation_id}, expected {}",
                pending.operation_id,
            )));
        }
        pending.value_complete = true;
        // A pre-reserved access keeps its physical order through the mode's
        // after-callback, where Racecheck commits the matching read-from
        // version. The outer reservation guard aborts this pending record if a
        // callback exits early.
        Ok(())
    }

    fn complete_atomic_load(&self, warp_id: usize, operation_id: u64) -> Result<(), EngineError> {
        let pending = self.warps[warp_id]
            .lock()
            .expect("ordering warp mutex poisoned")
            .pending_atomic_access
            .take()
            .ok_or_else(|| {
                EngineError::message(format!(
                    "warp {warp_id} completed an atomic load without a pending access"
                ))
            })?;
        let result = if pending.operation_id != operation_id {
            Err(EngineError::message(format!(
                "warp {warp_id} atomic-load completion does not match its prepared access"
            )))
        } else {
            Ok(())
        };
        self.release_atomic_reservation(operation_id, warp_id);
        result
    }

    fn abort_atomic_access(&self, warp_id: usize, operation_id: u64) {
        let mut warp = self.warps[warp_id]
            .lock()
            .expect("ordering warp mutex poisoned");
        if warp
            .pending_atomic_access
            .as_ref()
            .is_some_and(|pending| pending.operation_id == operation_id)
        {
            warp.pending_atomic_access = None;
        }
        drop(warp);
        self.release_atomic_reservation(operation_id, warp_id);
    }

    fn release_atomic_reservation(&self, operation_id: u64, warp_id: usize) {
        let mut serial = self
            .atomic_serial
            .lock()
            .expect("atomic serial mutex poisoned");
        if let Some(reservation) = serial.reservations.remove(&operation_id) {
            debug_assert_eq!(reservation.warp_id, warp_id);
            self.atomic_serial_changed.notify_all();
        }
        let ready_warps = serial
            .waiters
            .iter()
            .filter_map(|(&waiting_warp, waiter)| {
                overlapping_reservation_warps(&serial, &waiter.ranges, waiter.exclusive)
                    .is_none()
                    .then_some(waiting_warp)
            })
            .collect::<Vec<_>>();
        let ready = ready_warps
            .into_iter()
            .filter_map(|waiting_warp| {
                serial
                    .waiters
                    .remove(&waiting_warp)
                    .map(|waiter| waiter.waker)
            })
            .collect::<Vec<_>>();
        drop(serial);
        for waker in ready {
            waker.wake();
        }
    }

    pub(crate) fn tcgen_issue(
        &self,
        _context: WarpContext,
        kind: TcgenTransferKind,
        mask: WarpMask,
    ) -> Result<(), EngineError> {
        let operation = DiagnosticLabel::new(match kind {
            TcgenTransferKind::Load => "tcgen05.ld",
            TcgenTransferKind::Store => "tcgen05.st",
        });
        require_full_warp(mask, &operation)
    }

    pub(crate) fn commit_tcgen_work_issue(
        &self,
        issue: &TcgenWorkIssue,
    ) -> Result<(), EngineError> {
        let warp_id = issue.operation().id().global_warp_id();
        let mut warp = self.warps[warp_id]
            .lock()
            .expect("ordering warp mutex poisoned");
        for lane in issue.operation().active_mask() {
            let thread = &mut warp.tcgen_threads[lane];
            let queue = match issue.kind() {
                TcgenWorkKind::Commit => thread
                    .commit_by_cta_group
                    .entry(issue.cta_group())
                    .or_default(),
                TcgenWorkKind::MmaSharedARead => thread
                    .mma_shared_a_by_cta_group
                    .entry(issue.cta_group())
                    .or_default(),
                TcgenWorkKind::Load => &mut thread.loads,
                TcgenWorkKind::Store => &mut thread.stores,
            };
            if queue.contains(issue.token()) {
                return Err(EngineError::message(format!(
                    "{} token {:?} is already pending for warp {warp_id} lane {lane}",
                    issue.kind().name(),
                    issue.token()
                )));
            }
            queue.push(issue.token().clone());
        }
        Ok(())
    }

    pub(crate) fn plan_tcgen_commit(
        &self,
        context: WarpContext,
        issue_mask: WarpMask,
        cta_group: u32,
        shared_a_only: bool,
    ) -> Result<TcgenWorkSet, EngineError> {
        if !matches!(cta_group, 1 | 2) {
            return Err(EngineError::message(format!(
                "tcgen05.commit cta_group must be 1 or 2, got {cta_group}"
            )));
        }
        let warp_id = context.global_warp_id();
        let warp = self.warps[warp_id]
            .lock()
            .expect("ordering warp mutex poisoned");
        let kind = if shared_a_only {
            TcgenWorkKind::MmaSharedARead
        } else {
            TcgenWorkKind::Commit
        };
        let lane_tokens = issue_mask.into_iter().map(|lane| {
            let tokens = warp.tcgen_threads[lane]
                .pending_work(kind, Some(cta_group))
                .into_boxed_slice();
            (lane, tokens)
        });
        Ok(TcgenWorkSet::new(
            kind,
            Some(cta_group),
            warp_id,
            lane_tokens,
        ))
    }

    pub(crate) fn plan_tcgen_wait(
        &self,
        context: WarpContext,
        kind: TcgenTransferKind,
    ) -> Result<TcgenWorkSet, EngineError> {
        self.tcgen_wait(context, kind)?;
        let warp_id = context.global_warp_id();
        let warp = self.warps[warp_id]
            .lock()
            .expect("ordering warp mutex poisoned");
        let lane_tokens = context.active_mask().into_iter().map(|lane| {
            let tokens = match kind {
                TcgenTransferKind::Load => &warp.tcgen_threads[lane].loads,
                TcgenTransferKind::Store => &warp.tcgen_threads[lane].stores,
            };
            (lane, tokens.clone().into_boxed_slice())
        });
        Ok(TcgenWorkSet::new(
            match kind {
                TcgenTransferKind::Load => TcgenWorkKind::Load,
                TcgenTransferKind::Store => TcgenWorkKind::Store,
            },
            None,
            warp_id,
            lane_tokens,
        ))
    }

    pub(crate) fn commit_tcgen_work_set(&self, work: &TcgenWorkSet) -> Result<(), EngineError> {
        let warp_id = work.global_warp_id();
        let mut warp = self.warps[warp_id]
            .lock()
            .expect("ordering warp mutex poisoned");
        for (lane, expected) in work.lane_tokens() {
            let thread = &warp.tcgen_threads[*lane];
            let queue = thread.pending_work(work.kind(), work.cta_group());
            if queue.as_slice() != expected.as_ref() {
                return Err(EngineError::message(format!(
                    "{} pending work changed between plan and commit for warp {warp_id} lane {lane}",
                    work.kind().name()
                )));
            }
        }
        for (lane, _) in work.lane_tokens() {
            let thread = &mut warp.tcgen_threads[*lane];
            match work.kind() {
                TcgenWorkKind::Commit | TcgenWorkKind::MmaSharedARead => {
                    let group = work.cta_group().expect("commit work retains cta_group");
                    thread.mma_shared_a_by_cta_group.remove(&group);
                    if work.kind() == TcgenWorkKind::Commit {
                        thread.commit_by_cta_group.remove(&group);
                    }
                }
                TcgenWorkKind::Load => thread.loads.clear(),
                TcgenWorkKind::Store => thread.stores.clear(),
            }
        }
        Ok(())
    }

    pub(crate) fn tcgen_wait(
        &self,
        context: WarpContext,
        kind: TcgenTransferKind,
    ) -> Result<(), EngineError> {
        let operation = DiagnosticLabel::new(match kind {
            TcgenTransferKind::Load => "tcgen05.wait::ld",
            TcgenTransferKind::Store => "tcgen05.wait::st",
        });
        require_full_warp(context.active_mask(), &operation)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn setmaxnreg_arrive(
        &self,
        _static_op_id: u64,
        _path: impl IntoIterator<Item = i64>,
        context: WarpContext,
        warps_per_group: usize,
        increase: bool,
        register_count: u32,
    ) -> Result<u64, EngineError> {
        if warps_per_group == 0 {
            return Err(EngineError::message(
                "setmaxnreg warpgroup width must be positive",
            ));
        }
        let group_start = (context.warp_id_in_cta() / warps_per_group) * warps_per_group;
        if warps_per_group != 4
            || group_start.saturating_add(warps_per_group) > self.topology.warps_per_cta()
        {
            return Err(EngineError::message(format!(
                "setmaxnreg requires a complete 4-warp warpgroup, but CTA {} has {} warp(s) and this group starts at warp {}",
                context.global_cta_id(),
                self.topology.warps_per_cta(),
                group_start,
            )));
        }
        let warp_id = context.global_warp_id();
        let group_id = warpgroup_id(context, warps_per_group);
        let mut warp = self.warps[warp_id]
            .lock()
            .expect("ordering warp mutex poisoned");
        let mut coordination = self
            .coordination
            .lock()
            .expect("ordering coordination mutex poisoned");
        if coordination.setmaxnreg_requires_sync.contains(&group_id) {
            return Err(EngineError::message(format!(
                "warpgroup {group_id} executed a subsequent setmaxnreg without an explicit warpgroup synchronization"
            )));
        }
        let occurrence = warp.setmaxnreg_next_occurrence;
        warp.setmaxnreg_next_occurrence = occurrence
            .checked_add(1)
            .ok_or_else(|| EngineError::message("setmaxnreg occurrence overflow"))?;
        let entry = coordination
            .setmaxnreg_occurrences
            .entry((group_id, occurrence))
            .or_insert_with(|| SetmaxnregOccurrence {
                increase,
                register_count,
                arrived_warps: BTreeSet::new(),
                completed_warps: BTreeSet::new(),
            });
        if entry.increase != increase || entry.register_count != register_count {
            return Err(EngineError::message(format!(
                "warpgroup {group_id} setmaxnreg occurrence {occurrence} disagreed across warps: expected action={} count={}, got action={} count={register_count}",
                if entry.increase { "inc" } else { "dec" },
                entry.register_count,
                if increase { "inc" } else { "dec" },
            )));
        }
        if !entry.arrived_warps.insert(warp_id) {
            return Err(EngineError::message(format!(
                "warp {warp_id} repeated setmaxnreg occurrence {occurrence}"
            )));
        }
        Ok(occurrence)
    }

    pub(crate) fn setmaxnreg_complete(
        &self,
        context: WarpContext,
        warps_per_group: usize,
        occurrence: u64,
    ) -> Result<(), EngineError> {
        let warp_id = context.global_warp_id();
        let group_id = warpgroup_id(context, warps_per_group);
        let expected = expected_warpgroup_warps(warps_per_group);
        let mut coordination = self
            .coordination
            .lock()
            .expect("ordering coordination mutex poisoned");
        let complete = {
            let entry = coordination
                .setmaxnreg_occurrences
                .get_mut(&(group_id, occurrence))
                .ok_or_else(|| EngineError::message("setmaxnreg completed without an arrival"))?;
            if !entry.completed_warps.insert(warp_id) {
                return Err(EngineError::message(format!(
                    "warp {warp_id} completed setmaxnreg occurrence {occurrence} twice"
                )));
            }
            entry.completed_warps.len() == expected
        };
        coordination.setmaxnreg_requires_sync.insert(group_id);
        if complete {
            coordination
                .setmaxnreg_occurrences
                .remove(&(group_id, occurrence));
        }
        Ok(())
    }

    pub(crate) fn setmaxnreg_warpgroup_sync(&self, context: WarpContext, warps_per_group: usize) {
        let group_id = warpgroup_id(context, warps_per_group);
        self.coordination
            .lock()
            .expect("ordering coordination mutex poisoned")
            .setmaxnreg_requires_sync
            .remove(&group_id);
    }

    pub(crate) fn validate_quiescent(&self) -> Result<(), EngineError> {
        let mut pending = Vec::new();
        for (warp_id, warp) in self.warps.iter().enumerate() {
            let warp = warp.lock().expect("ordering warp mutex poisoned");
            if warp.prepared_atomic_access.is_some() || warp.pending_atomic_access.is_some() {
                pending.push(format!(
                    "warp {warp_id} has an unfinished global atomic access"
                ));
            }
        }
        let coordination = self
            .coordination
            .lock()
            .expect("ordering coordination mutex poisoned");
        for ((group_id, occurrence), entry) in &coordination.setmaxnreg_occurrences {
            pending.push(format!(
                "warpgroup {group_id} setmaxnreg occurrence {occurrence} has arrivals {:?} and completions {:?}",
                entry.arrived_warps, entry.completed_warps
            ));
        }
        drop(coordination);
        let serial = self
            .atomic_serial
            .lock()
            .expect("atomic serial mutex poisoned");
        if !serial.reservations.is_empty() {
            pending.push(format!(
                "{} global atomic linearization reservation(s) remain active",
                serial.reservations.len(),
            ));
        }
        if pending.is_empty() {
            Ok(())
        } else {
            Err(EngineError::message(pending.join("; ")))
        }
    }
}

/// `OrderingHub` joins the registry for its *blocked operations only*.
///
/// The design notes it is the one lifecycle hub that is not a
/// `CompletionSource`; that is why its parked warps could never reach a
/// deadlock report. It has no completions to advance, so `pump` is a no-op, and
/// `validate_quiescent` stays a no-op here because `PreparedLaunch::run_report`
/// already calls the hub's own richer `validate_quiescent` directly — routing
/// it through the registry as well would double-report the same error.
impl crate::CompletionSource for OrderingHub {
    fn source_name(&self) -> &'static str {
        "ordering-parks"
    }

    fn pump(&self) -> Result<crate::CompletionProgress, crate::SynchronizationError> {
        Ok(crate::CompletionProgress::default())
    }

    fn blocked_operations(&self) -> Vec<crate::BlockedOperation> {
        self.parked_operations()
    }

    fn validate_quiescent(&self) -> Result<(), crate::SynchronizationError> {
        Ok(())
    }
}

fn physical_memory_ranges(
    addresses: impl IntoIterator<Item = (PhysicalAddress, usize)>,
    operation: &DiagnosticLabel,
) -> Result<Vec<MemoryRange>, EngineError> {
    let mut ranges = Vec::new();
    for (address, byte_len) in addresses {
        if byte_len == 0 {
            continue;
        }
        let end = address
            .byte_offset()
            .checked_add(byte_len)
            .ok_or_else(|| operation.engine_error(format_args!(" range overflow")))?;
        ranges.push(MemoryRange {
            allocation_id: address.allocation_id(),
            start: address.byte_offset(),
            end,
        });
    }
    ranges.sort_unstable();
    let mut normalized: Vec<MemoryRange> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(last) = normalized.last_mut() {
            if last.allocation_id == range.allocation_id && last.end >= range.start {
                last.end = last.end.max(range.end);
                continue;
            }
        }
        normalized.push(range);
    }
    Ok(normalized)
}

/// The warps whose live reservations overlap `ranges`, or `None` when the
/// caller may proceed.
///
/// Same predicate as the loop condition it replaces; it additionally reports
/// *which* warps are in the way so a park record can name them.
fn overlapping_reservation_warps(
    serial: &AtomicSerialState,
    ranges: &[MemoryRange],
    exclusive: bool,
) -> Option<Vec<usize>> {
    let blocking = serial
        .reservations
        .values()
        .filter(|reservation| exclusive || reservation.exclusive)
        .filter(|reservation| memory_range_sets_overlap(&reservation.ranges, ranges))
        .map(|reservation| reservation.warp_id)
        .collect::<BTreeSet<_>>();
    (!blocking.is_empty()).then(|| blocking.into_iter().collect())
}

fn memory_range_sets_overlap(left: &[MemoryRange], right: &[MemoryRange]) -> bool {
    let mut left_index = 0;
    let mut right_index = 0;
    while left_index < left.len() && right_index < right.len() {
        let lhs = left[left_index];
        let rhs = right[right_index];
        if lhs.allocation_id < rhs.allocation_id
            || (lhs.allocation_id == rhs.allocation_id && lhs.end <= rhs.start)
        {
            left_index += 1;
        } else if rhs.allocation_id < lhs.allocation_id
            || (lhs.allocation_id == rhs.allocation_id && rhs.end <= lhs.start)
        {
            right_index += 1;
        } else {
            return true;
        }
    }
    false
}

fn require_full_warp(mask: WarpMask, operation: &DiagnosticLabel) -> Result<(), EngineError> {
    if mask != WarpMask::FULL {
        return Err(operation.warp_collective_divergence(mask));
    }
    Ok(())
}

const fn expected_warpgroup_warps(warps_per_group: usize) -> usize {
    warps_per_group
}

fn warpgroup_id(context: WarpContext, warps_per_group: usize) -> usize {
    context.global_cta_id() * context.topology().warps_per_cta()
        + (context.warp_id_in_cta() / warps_per_group) * warps_per_group
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::pin::pin;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    };
    use std::task::{Context, Poll, Wake, Waker};
    use std::time::Duration;

    use super::{OrderingHub, ParkReason, TcgenTransferKind};
    use crate::{
        runtime::{
            TcgenAccumulatorDtype, TcgenMmaPipelineClass, TcgenPipelineOperation, TcgenWorkIssue,
            TcgenWorkKind,
        },
        CompletionSource, DynamicOpId, LaunchTopology, OperationContext, OperationKind,
        PhysicalAddress, StaticOpId, WarpMask,
    };

    #[test]
    fn an_unparked_launch_reports_no_blocked_operations() {
        let hub = OrderingHub::new(LaunchTopology::new(1, 1, 2).unwrap());
        assert!(CompletionSource::blocked_operations(&hub).is_empty());
    }

    #[test]
    fn a_park_is_visible_while_held_and_gone_once_the_guard_drops() {
        // This is the property the whole step exists for: a warp parked outside
        // every completion source used to be invisible to deadlock reporting.
        let hub = OrderingHub::new(LaunchTopology::new(1, 1, 2).unwrap());
        {
            let _park = hub.park(1, ParkReason::SemanticProgress, None);
            let blocked = CompletionSource::blocked_operations(&hub);
            assert_eq!(blocked.len(), 1, "the parked warp must be reported");
            assert_eq!(blocked[0].warp_id, 1);
            assert_eq!(
                blocked[0].to_string(),
                "warp 1 awaits engine.semantic_progress key \
                 engine.park[op=1]@warp[1]#path=[]; arrived=[], missing=[], expected=[]"
            );
        }
        assert!(
            CompletionSource::blocked_operations(&hub).is_empty(),
            "resuming must clear the record"
        );
    }

    #[test]
    fn an_atomic_linearization_park_names_the_warps_in_the_way() {
        let hub = OrderingHub::new(LaunchTopology::new(1, 1, 4).unwrap());
        let _park = hub.park(
            3,
            ParkReason::AtomicLinearization {
                blocking_warps: vec![0, 2],
            },
            None,
        );
        let blocked = CompletionSource::blocked_operations(&hub);
        assert_eq!(blocked.len(), 1);
        assert_eq!(blocked[0].participant_state.missing, vec![0, 2]);
        assert!(
            blocked[0].to_string().contains("atom.linearize"),
            "the reason must name the operation: {}",
            blocked[0]
        );
    }

    #[test]
    fn a_park_record_is_cleared_even_when_the_parked_scope_unwinds() {
        // The guard is RAII precisely so a failing warp cannot leave a phantom
        // blocked operation behind for every later deadlock report.
        let hub = Arc::new(OrderingHub::new(LaunchTopology::new(1, 1, 2).unwrap()));
        let unwind_hub = Arc::clone(&hub);
        let result = std::panic::catch_unwind(move || {
            let _park = unwind_hub.park(0, ParkReason::SemanticProgress, None);
            panic!("warp failed while parked");
        });
        assert!(result.is_err(), "the test body must actually unwind");
        assert!(
            CompletionSource::blocked_operations(hub.as_ref()).is_empty(),
            "an unwound park must not leak a blocked operation"
        );
    }

    fn tcgen_issue(
        context: crate::WarpContext,
        sequence: u64,
        source_op_id: u64,
        mask: WarpMask,
        kind: TcgenWorkKind,
        cta_group: u32,
    ) -> TcgenWorkIssue {
        TcgenWorkIssue::new(
            OperationContext::new(
                DynamicOpId::new(
                    0,
                    context.global_warp_id(),
                    sequence,
                    StaticOpId::new(source_op_id),
                    [],
                ),
                OperationKind::AsyncIssue,
                mask,
            ),
            cta_group,
            match kind {
                TcgenWorkKind::Commit => TcgenPipelineOperation::Mma,
                TcgenWorkKind::MmaSharedARead => TcgenPipelineOperation::MmaSharedARead,
                TcgenWorkKind::Load => TcgenPipelineOperation::Load,
                TcgenWorkKind::Store => TcgenPipelineOperation::Store,
            },
            (kind == TcgenWorkKind::Commit).then_some(TcgenMmaPipelineClass::new(
                128,
                128,
                16,
                TcgenAccumulatorDtype::F32,
            )),
            [],
        )
        .unwrap()
    }

    #[test]
    fn pre_reserved_global_atomic_rmw_stays_linearized_through_metadata() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let contexts = topology.warp_contexts().collect::<Vec<_>>();
        let first_context = contexts[0];
        let second_context = contexts[1];
        let hub = Arc::new(OrderingHub::new(topology));
        let signal = PhysicalAddress::new(106, 0);
        let (first_reserved_tx, first_reserved_rx) = mpsc::channel();
        let (finish_first_value_tx, finish_first_value_rx) = mpsc::channel();
        let (first_value_complete_tx, first_value_complete_rx) = mpsc::channel();
        let (finish_first_metadata_tx, finish_first_metadata_rx) = mpsc::channel();
        let (second_started_tx, second_started_rx) = mpsc::channel();
        let (second_reserved_tx, second_reserved_rx) = mpsc::channel();

        let first_hub = hub.clone();
        let first = std::thread::spawn(move || {
            first_hub
                .prepare_atomic_access(first_context, [(signal, 4)])
                .unwrap();
            let reservation = first_hub
                .reserve_prepared_atomic_access(first_context)
                .unwrap()
                .unwrap();
            let access = first_hub
                .begin_atomic_access(first_context, [(signal, 4)])
                .unwrap()
                .unwrap();
            first_reserved_tx.send(()).unwrap();
            finish_first_value_rx.recv().unwrap();
            access.mark_value_complete().unwrap();
            first_value_complete_tx.send(()).unwrap();
            finish_first_metadata_rx.recv().unwrap();
            first_hub
                .complete_atomic_rmw(first_context, [(signal, 4)])
                .unwrap();
            drop(reservation);
        });
        first_reserved_rx.recv().unwrap();

        let second_hub = hub.clone();
        let second = std::thread::spawn(move || {
            second_hub
                .prepare_atomic_access(second_context, [(signal, 4)])
                .unwrap();
            second_started_tx.send(()).unwrap();
            let reservation = second_hub
                .reserve_prepared_atomic_access(second_context)
                .unwrap()
                .unwrap();
            let access = second_hub
                .begin_atomic_access(second_context, [(signal, 4)])
                .unwrap()
                .unwrap();
            second_reserved_tx.send(()).unwrap();
            access.mark_value_complete().unwrap();
            second_hub
                .complete_atomic_rmw(second_context, [(signal, 4)])
                .unwrap();
            drop(reservation);
        });
        second_started_rx.recv().unwrap();
        assert!(matches!(
            second_reserved_rx.recv_timeout(Duration::from_millis(20)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        finish_first_value_tx.send(()).unwrap();
        first_value_complete_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        assert!(matches!(
            second_reserved_rx.recv_timeout(Duration::from_millis(20)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        finish_first_metadata_tx.send(()).unwrap();
        second_reserved_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        first.join().unwrap();
        second.join().unwrap();
    }

    #[derive(Default)]
    struct WakeCount(AtomicUsize);

    impl Wake for WakeCount {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn async_atomic_reservation_parks_the_warp_and_wakes_after_metadata() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let contexts = topology.warp_contexts().collect::<Vec<_>>();
        let first_context = contexts[0];
        let second_context = contexts[1];
        let hub = Arc::new(OrderingHub::new(topology));
        let signal = PhysicalAddress::new(107, 0);
        let wake_count = Arc::new(WakeCount::default());
        let waker = Waker::from(wake_count.clone());
        let mut task_context = Context::from_waker(&waker);

        hub.prepare_atomic_access(first_context, [(signal, 4)])
            .unwrap();
        let first_wait = hub.reserve_prepared_atomic_access_async(first_context, true);
        let mut first_wait = pin!(first_wait);
        let first_reservation = match first_wait.as_mut().poll(&mut task_context) {
            Poll::Ready(Ok(Some(reservation))) => reservation,
            _ => panic!("an uncontended reservation must be ready immediately"),
        };
        let first_access = hub
            .begin_atomic_access(first_context, [(signal, 4)])
            .unwrap()
            .unwrap();
        first_access.mark_value_complete().unwrap();

        hub.prepare_atomic_access(second_context, [(signal, 4)])
            .unwrap();
        let second_wait = hub.reserve_prepared_atomic_access_async(second_context, true);
        let mut second_wait = pin!(second_wait);
        assert!(matches!(
            second_wait.as_mut().poll(&mut task_context),
            Poll::Pending
        ));
        let blocked = CompletionSource::blocked_operations(hub.as_ref());
        assert_eq!(blocked.len(), 1);
        assert_eq!(blocked[0].warp_id, second_context.global_warp_id());
        assert_eq!(
            blocked[0].participant_state.missing,
            vec![first_context.global_warp_id()]
        );
        assert_eq!(wake_count.0.load(Ordering::Relaxed), 0);

        hub.complete_atomic_rmw(first_context, [(signal, 4)])
            .unwrap();
        assert_eq!(wake_count.0.load(Ordering::Relaxed), 1);
        drop(first_reservation);

        let second_reservation = match second_wait.as_mut().poll(&mut task_context) {
            Poll::Ready(Ok(Some(reservation))) => reservation,
            _ => panic!("releasing the conflicting reservation must wake its waiter"),
        };
        assert!(CompletionSource::blocked_operations(hub.as_ref()).is_empty());
        let second_access = hub
            .begin_atomic_access(second_context, [(signal, 4)])
            .unwrap()
            .unwrap();
        second_access.mark_value_complete().unwrap();
        hub.complete_atomic_rmw(second_context, [(signal, 4)])
            .unwrap();
        drop(second_reservation);
    }

    #[test]
    fn shared_publication_observers_coexist_but_exclude_writers() {
        let topology = LaunchTopology::new(4, 1, 1).unwrap();
        let contexts = topology.warp_contexts().collect::<Vec<_>>();
        let hub = Arc::new(OrderingHub::new(topology));
        let signal = PhysicalAddress::new(107, 0);
        let wake_count = Arc::new(WakeCount::default());
        let waker = Waker::from(wake_count.clone());
        let mut cx = Context::from_waker(&waker);
        let mut readers = Vec::new();
        for context in &contexts[..2] {
            hub.prepare_atomic_access(*context, [(signal, 4)]).unwrap();
            let read = hub.reserve_prepared_atomic_access_async(*context, false);
            let mut read = pin!(read);
            match read.as_mut().poll(&mut cx) {
                Poll::Ready(Ok(Some(guard))) => readers.push(guard),
                _ => panic!("overlapping read reservations must coexist"),
            }
        }
        hub.prepare_atomic_access(contexts[2], [(signal, 4)]).unwrap();
        let write = hub.reserve_prepared_atomic_access_async(contexts[2], true);
        let mut write = pin!(write);
        assert!(matches!(write.as_mut().poll(&mut cx), Poll::Pending));
        drop(readers.pop());
        assert_eq!(wake_count.0.load(Ordering::Relaxed), 0);
        assert!(matches!(write.as_mut().poll(&mut cx), Poll::Pending));
        drop(readers.pop());
        assert_eq!(wake_count.0.load(Ordering::Relaxed), 1);
        let writer = match write.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(Some(guard))) => guard,
            _ => panic!("writer must wake after the final reader leaves"),
        };
        hub.prepare_atomic_access(contexts[3], [(signal, 4)]).unwrap();
        let read = hub.reserve_prepared_atomic_access_async(contexts[3], false);
        let mut read = pin!(read);
        assert!(matches!(read.as_mut().poll(&mut cx), Poll::Pending));
        drop(writer);
        assert_eq!(wake_count.0.load(Ordering::Relaxed), 2);
        let reader = match read.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(Some(guard))) => guard,
            _ => panic!("reader must wake after writer metadata is committed"),
        };
        drop(reader);
        hub.validate_quiescent().unwrap();
    }

    #[test]
    fn setmaxnreg_rejects_a_partial_warpgroup_topology() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let hub = OrderingHub::new(topology);
        let error = hub
            .setmaxnreg_arrive(0, [], context, 4, true, 24)
            .unwrap_err();
        assert!(error.to_string().contains("complete 4-warp warpgroup"));
    }

    #[test]
    fn tcgen_commit_drains_only_the_issuing_thread_and_matching_cta_group() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let hub = OrderingHub::new(topology);
        let lane_zero = WarpMask::from_lanes([0]).unwrap();
        let lane_one = WarpMask::from_lanes([1]).unwrap();
        let first = tcgen_issue(context, 0, 10, lane_zero, TcgenWorkKind::Commit, 1);
        let other_lane = tcgen_issue(context, 1, 11, lane_one, TcgenWorkKind::Commit, 1);
        let other_group = tcgen_issue(context, 2, 12, lane_zero, TcgenWorkKind::Commit, 2);
        hub.commit_tcgen_work_issue(&first).unwrap();
        hub.commit_tcgen_work_issue(&other_lane).unwrap();
        hub.commit_tcgen_work_issue(&other_group).unwrap();

        let lane_zero_group_one = hub.plan_tcgen_commit(context, lane_zero, 1, false).unwrap();
        assert_eq!(lane_zero_group_one.tokens(), [first.token().clone()]);
        hub.commit_tcgen_work_set(&lane_zero_group_one).unwrap();

        assert!(hub
            .plan_tcgen_commit(context, lane_zero, 1, false)
            .unwrap()
            .is_empty());
        assert_eq!(
            hub.plan_tcgen_commit(context, lane_one, 1, false)
                .unwrap()
                .tokens(),
            [other_lane.token().clone()]
        );
        assert_eq!(
            hub.plan_tcgen_commit(context, lane_zero, 2, false)
                .unwrap()
                .tokens(),
            [other_group.token().clone()]
        );
    }

    #[test]
    fn tcgen_wait_deduplicates_one_warp_wide_source_token() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let hub = OrderingHub::new(topology);
        let load = tcgen_issue(context, 0, 20, WarpMask::FULL, TcgenWorkKind::Load, 1);
        hub.commit_tcgen_work_issue(&load).unwrap();

        let wait = hub
            .plan_tcgen_wait(context, TcgenTransferKind::Load)
            .unwrap();
        assert_eq!(wait.tokens(), [load.token().clone()]);
        assert_eq!(wait.lane_tokens().len(), 32);
        hub.commit_tcgen_work_set(&wait).unwrap();
        assert!(hub
            .plan_tcgen_wait(context, TcgenTransferKind::Load)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn tcgen_wait_drains_one_representative_lane_token_from_a_full_warp_wait() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let hub = OrderingHub::new(topology);
        let issuer = WarpMask::from_lanes([7]).unwrap();
        let load = tcgen_issue(context, 0, 21, issuer, TcgenWorkKind::Load, 1);
        hub.commit_tcgen_work_issue(&load).unwrap();

        let wait = hub
            .plan_tcgen_wait(context, TcgenTransferKind::Load)
            .unwrap();
        assert_eq!(wait.tokens(), [load.token().clone()]);
        assert_eq!(wait.lane_tokens().len(), 32);
        assert_eq!(wait.lane_tokens()[7].1.as_ref(), [load.token().clone()]);
        assert!(wait
            .lane_tokens()
            .iter()
            .enumerate()
            .all(|(lane, (_, tokens))| lane == 7 || tokens.is_empty()));
        hub.commit_tcgen_work_set(&wait).unwrap();
        assert!(hub
            .plan_tcgen_wait(context, TcgenTransferKind::Load)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn elected_tcgen_commit_leaves_no_nonissuer_lane_residue() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let hub = OrderingHub::new(topology);
        let issuer = WarpMask::from_lanes([7]).unwrap();
        let mma = tcgen_issue(context, 0, 30, issuer, TcgenWorkKind::Commit, 1);
        hub.commit_tcgen_work_issue(&mma).unwrap();

        let commit = hub.plan_tcgen_commit(context, issuer, 1, false).unwrap();
        assert_eq!(commit.tokens(), [mma.token().clone()]);
        hub.commit_tcgen_work_set(&commit).unwrap();

        let all_lanes = hub
            .plan_tcgen_commit(context, WarpMask::FULL, 1, false)
            .unwrap();
        assert!(all_lanes.is_empty());
        assert!(all_lanes
            .lane_tokens()
            .iter()
            .all(|(_, tokens)| tokens.is_empty()));
    }
}
