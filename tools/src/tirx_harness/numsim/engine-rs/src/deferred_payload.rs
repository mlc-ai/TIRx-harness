use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use crate::runtime::PhysicalMbarrierCompletionIssuePlan;
use crate::{
    AsyncPayloadEffect, AsyncTokenId, BlockedOperation, CompletionProgress, CompletionSource,
    EngineError, OperationContext, PhysicalAccessBatch, PhysicalBarrierHub, PhysicalBarrierId,
    PhysicalCompletionAction, PhysicalCompletionActionId, PhysicalCompletionOutcome,
    SynchronizationError,
};

/// One scheduler-visible logical completion. Multi-target physical actions are
/// grouped behind the first action ID and complete as one transition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeferredPayloadCompletionAction {
    token: AsyncTokenId,
    operation: OperationContext,
    physical_actions: Box<[PhysicalCompletionAction]>,
    // Completion candidates are snapshotted several times while the executor
    // pumps enabled actions.  Their resolved footprints are immutable, so
    // share them instead of cloning every lane and provenance frame per poll.
    completion_accesses: Arc<[PhysicalAccessBatch]>,
}

impl DeferredPayloadCompletionAction {
    pub const fn token(&self) -> &AsyncTokenId {
        &self.token
    }

    pub const fn operation(&self) -> &OperationContext {
        &self.operation
    }

    pub fn physical_actions(&self) -> &[PhysicalCompletionAction] {
        &self.physical_actions
    }

    pub fn completion_accesses(&self) -> &[PhysicalAccessBatch] {
        &self.completion_accesses
    }

    pub fn action_ids(&self) -> impl ExactSizeIterator<Item = PhysicalCompletionActionId> + '_ {
        self.physical_actions.iter().map(|action| action.id())
    }

    pub fn scheduler_action_id(&self) -> PhysicalCompletionActionId {
        self.physical_actions
            .first()
            .expect("deferred payload action has a physical completion")
            .id()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeferredPayloadCompletionOutcome {
    action: DeferredPayloadCompletionAction,
    physical_outcomes: Box<[PhysicalCompletionOutcome]>,
}

impl DeferredPayloadCompletionOutcome {
    pub const fn action(&self) -> &DeferredPayloadCompletionAction {
        &self.action
    }

    pub fn physical_outcomes(&self) -> &[PhysicalCompletionOutcome] {
        &self.physical_outcomes
    }

    pub fn progress(&self) -> CompletionProgress {
        self.physical_outcomes
            .iter()
            .fold(CompletionProgress::default(), |mut total, outcome| {
                total.completed_operations += outcome.progress().completed_operations;
                total.woken_warps += outcome.progress().woken_warps;
                total
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MbarrierCompletionAction {
    Physical(PhysicalCompletionAction),
    DeferredPayload(DeferredPayloadCompletionAction),
}

impl MbarrierCompletionAction {
    pub fn scheduler_action_id(&self) -> PhysicalCompletionActionId {
        match self {
            Self::Physical(action) => action.id(),
            Self::DeferredPayload(action) => action.scheduler_action_id(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MbarrierCompletionOutcome {
    Physical(PhysicalCompletionOutcome),
    DeferredPayload(DeferredPayloadCompletionOutcome),
}

impl MbarrierCompletionOutcome {
    pub fn progress(&self) -> CompletionProgress {
        match self {
            Self::Physical(outcome) => outcome.progress(),
            Self::DeferredPayload(outcome) => outcome.progress(),
        }
    }
}

struct DeferredPayloadRecord {
    action: Option<DeferredPayloadCompletionAction>,
}

/// Barrier generations whose completed tokens stay listed per barrier (the
/// current one and the one before): a launch issues millions of payload
/// tokens, and their full identities (an operation with its loop frames)
/// kept per completed generation were the largest thing the hub retained.
const RETAINED_COMPLETED_TOKEN_GENERATIONS: u64 = 2;

#[derive(Default)]
struct DeferredPayloadState {
    /// The pending payloads; a completed payload leaves this map and is
    /// remembered by the hash of its token only, for the duplicate check.
    records: BTreeMap<AsyncTokenId, DeferredPayloadRecord>,
    completed: HashSet<u64>,
    action_tokens: BTreeMap<PhysicalCompletionActionId, AsyncTokenId>,
    completed_tokens: BTreeMap<(PhysicalBarrierId, u64), BTreeSet<AsyncTokenId>>,
}

fn token_digest(token: &AsyncTokenId) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    token.hash(&mut hasher);
    hasher.finish()
}

impl DeferredPayloadState {
    fn knows_token(&self, token: &AsyncTokenId) -> bool {
        self.records.contains_key(token) || self.completed.contains(&token_digest(token))
    }

    /// Mark `token` complete: drop its record, keep its digest.
    fn complete_token(&mut self, token: &AsyncTokenId) {
        self.records.remove(token);
        self.completed.insert(token_digest(token));
    }

    /// List `token` under the barrier generation it completed, dropping the
    /// generations that fell out of the retained window.
    fn list_completed_token(
        &mut self,
        barrier_id: PhysicalBarrierId,
        generation: u64,
        token: &AsyncTokenId,
    ) {
        self.completed_tokens
            .entry((barrier_id, generation))
            .or_default()
            .insert(token.clone());
        let Some(oldest_retained) = generation.checked_sub(RETAINED_COMPLETED_TOKEN_GENERATIONS - 1) else {
            return;
        };
        let stale: Vec<(PhysicalBarrierId, u64)> = self
            .completed_tokens
            .range((barrier_id, 0)..(barrier_id, oldest_retained))
            .map(|(key, _)| *key)
            .collect();
        for key in stale {
            self.completed_tokens.remove(&key);
        }
    }
}

/// Launch-wide owner for mbarrier-backed payload tokens and plain physical
/// completion actions.
pub struct DeferredPayloadHub {
    mbarriers: Arc<PhysicalBarrierHub>,
    state: Mutex<DeferredPayloadState>,
}

impl DeferredPayloadHub {
    pub fn new(mbarriers: Arc<PhysicalBarrierHub>) -> Self {
        Self {
            mbarriers,
            state: Mutex::new(DeferredPayloadState::default()),
        }
    }

    /// Retain only the scheduler-visible completion for a payload whose
    /// numerical bytes have already executed at issue time.
    pub fn enqueue_payload(
        &self,
        effect: &AsyncPayloadEffect,
        payload_byte_len: u64,
    ) -> Result<Box<[PhysicalCompletionActionId]>, EngineError> {
        self.enqueue_payload_parts(
            effect.token(),
            effect.operation(),
            effect.completion_plan(),
            effect.completion_accesses(),
            payload_byte_len,
        )
    }

    /// Register only the scheduler-visible completion after NumSim has already
    /// executed the numerical payload at issue time.
    pub fn enqueue_eager_payload_completion(
        &self,
        operation: &OperationContext,
        completion_plan: &PhysicalMbarrierCompletionIssuePlan,
        payload_byte_len: u64,
    ) -> Result<(), EngineError> {
        let token = AsyncTokenId::new(operation.id().clone(), 0);
        self.enqueue_payload_parts(&token, operation, completion_plan, &[], payload_byte_len)
            .map(drop)
    }

    fn enqueue_payload_parts(
        &self,
        token: &AsyncTokenId,
        operation: &OperationContext,
        completion_plan: &PhysicalMbarrierCompletionIssuePlan,
        completion_accesses: &[PhysicalAccessBatch],
        payload_byte_len: u64,
    ) -> Result<Box<[PhysicalCompletionActionId]>, EngineError> {
        // The launch-wide state lock is held only for the map updates. It
        // used to be held across the mbarrier arrivals and the clone of the
        // completion accesses too, which under racecheck (large access
        // batches, completions applied inside the barrier publication gate)
        // made every issuing warp and every completion in the launch queue
        // behind one payload.
        {
            let state = self.state.lock().expect("deferred payload mutex poisoned");
            if state.knows_token(token) {
                return Err(duplicate_token_error(token));
            }
        }
        let completion_accesses: Arc<[PhysicalAccessBatch]> = Arc::from(completion_accesses);
        let operation = operation.clone();
        // `register` runs under the physical barrier lock, before the queued
        // actions become visible to the completion pump, so the pump can
        // never see one of them without its token.
        let (action_ids, ()) =
            completion_plan.apply_with(&self.mbarriers, |_, physical_actions| {
                let completes_immediately = physical_actions.is_empty();
                if completes_immediately && payload_byte_len != 0 {
                    return Err(EngineError::message(format!(
                        "async payload token {:?} executed {} destination bytes without a schedulable mbarrier completion",
                        token,
                        payload_byte_len
                    )));
                }
                let action = (!completes_immediately).then(|| DeferredPayloadCompletionAction {
                    token: token.clone(),
                    operation,
                    physical_actions: physical_actions.into(),
                    completion_accesses,
                });
                let mut state = {
                    self.state.lock().expect("deferred payload mutex poisoned")
                };
                if state.knows_token(token) {
                    return Err(duplicate_token_error(token));
                }
                if completes_immediately {
                    state.complete_token(token);
                    return Ok(());
                }
                if let Some(action) = &action {
                    for action_id in action.action_ids() {
                        state.action_tokens.insert(action_id, token.clone());
                    }
                }
                state
                    .records
                    .insert(token.clone(), DeferredPayloadRecord { action });
                Ok(())
            })?;
        Ok(action_ids)
    }

    pub fn pending_completion_actions(&self) -> Vec<MbarrierCompletionAction> {
        let physical = self.mbarriers.pending_completion_actions();
        let enabled = physical
            .iter()
            .map(|action| action.id())
            .collect::<BTreeSet<_>>();
        let state = self.state.lock().expect("deferred payload mutex poisoned");
        let mut actions = Vec::new();
        for physical_action in physical {
            let Some(token) = state.action_tokens.get(&physical_action.id()) else {
                actions.push(MbarrierCompletionAction::Physical(physical_action));
                continue;
            };
            let record = state
                .records
                .get(token)
                .expect("payload action owner retains its record");
            let action = record
                .action
                .as_ref()
                .expect("queued payload action retains completion metadata");
            if physical_action.id() != action.scheduler_action_id() {
                continue;
            }
            if action
                .action_ids()
                .all(|action_id| enabled.contains(&action_id))
            {
                actions.push(MbarrierCompletionAction::DeferredPayload(action.clone()));
            }
        }
        actions
    }

    pub fn apply_completion_detailed(
        &self,
        action_id: PhysicalCompletionActionId,
    ) -> Result<MbarrierCompletionOutcome, EngineError> {
        self.apply_completion_detailed_with_outcome(action_id, |_| Ok(()))
    }

    /// Apply one logical completion and publish its mode-visible outcome
    /// before any physical-barrier waiter is woken.
    pub fn apply_completion_detailed_with_outcome(
        &self,
        action_id: PhysicalCompletionActionId,
        publish_before_wake: impl FnOnce(&MbarrierCompletionOutcome) -> Result<(), EngineError>,
    ) -> Result<MbarrierCompletionOutcome, EngineError> {
        let pending = self
            .pending_completion_actions()
            .into_iter()
            .find(|action| action.scheduler_action_id() == action_id)
            .ok_or_else(|| {
                EngineError::message(format!(
                    "mbarrier completion action {action_id} is not enabled"
                ))
            })?;
        self.apply_completion_action_detailed_with_outcome(&pending, publish_before_wake)
    }

    /// Apply an enabled completion snapshot without rebuilding the launch-wide
    /// pending-action list.
    ///
    /// The mode-aware completion pump already obtained this exact action from
    /// `pending_completion_actions`. The physical barrier hub still validates
    /// live enablement while applying each action, so retaining the snapshot
    /// only removes redundant discovery and metadata cloning.
    pub(crate) fn apply_completion_action_detailed_with_outcome(
        &self,
        pending: &MbarrierCompletionAction,
        publish_before_wake: impl FnOnce(&MbarrierCompletionOutcome) -> Result<(), EngineError>,
    ) -> Result<MbarrierCompletionOutcome, EngineError> {
        match pending {
            MbarrierCompletionAction::Physical(action) => {
                let outcome = self
                    .mbarriers
                    .apply_completion_detailed_with_outcome(action.id(), |outcome| {
                        publish_before_wake(&MbarrierCompletionOutcome::Physical(outcome.clone()))
                    })
                    .map_err(|error| EngineError::message(error.to_string()))?;
                Ok(MbarrierCompletionOutcome::Physical(outcome))
            }
            MbarrierCompletionAction::DeferredPayload(action) => {
                let action_ids = action.action_ids().collect::<Vec<_>>();
                let published_action = action.clone();
                let physical_outcomes = self
                    .mbarriers
                    .apply_completion_batch_detailed_with_outcomes(
                        &action_ids,
                        || Ok(()),
                        |outcomes| {
                            let mut state =
                                self.state.lock().expect("deferred payload mutex poisoned");
                            debug_assert!(
                                state.records.contains_key(action.token()),
                                "completed payload retains its record"
                            );
                            // The record (its token, the completion metadata
                            // and the cloned completion accesses) is only
                            // needed while the payload is pending; the
                            // outcome handed to the caller carries its own
                            // copy, and the token's digest keeps the
                            // duplicate check.
                            state.complete_token(action.token());
                            for action_id in action.action_ids() {
                                state.action_tokens.remove(&action_id);
                            }
                            for physical_action in action.physical_actions() {
                                state.list_completed_token(
                                    physical_action.barrier_id(),
                                    physical_action.generation(),
                                    action.token(),
                                );
                            }
                            drop(state);
                            publish_before_wake(&MbarrierCompletionOutcome::DeferredPayload(
                                DeferredPayloadCompletionOutcome {
                                    action: published_action.clone(),
                                    physical_outcomes: outcomes.to_vec().into_boxed_slice(),
                                },
                            ))
                        },
                    )
                    .map_err(|error| EngineError::message(error.to_string()))?;
                Ok(MbarrierCompletionOutcome::DeferredPayload(
                    DeferredPayloadCompletionOutcome {
                        action: action.clone(),
                        physical_outcomes,
                    },
                ))
            }
        }
    }

    pub fn completed_tokens(
        &self,
        barrier_id: PhysicalBarrierId,
        generation: u64,
    ) -> Vec<AsyncTokenId> {
        self.state
            .lock()
            .expect("deferred payload mutex poisoned")
            .completed_tokens
            .get(&(barrier_id, generation))
            .map(|tokens| tokens.iter().cloned().collect())
            .unwrap_or_default()
    }
}

fn duplicate_token_error(token: &AsyncTokenId) -> EngineError {
    EngineError::message(format!(
        "async payload token {:?} was issued more than once",
        token
    ))
}

impl CompletionSource for DeferredPayloadHub {
    fn source_name(&self) -> &'static str {
        "mbarrier-payload"
    }

    fn pump(&self) -> Result<CompletionProgress, SynchronizationError> {
        let actions = self.pending_completion_actions();
        let mut progress = CompletionProgress::default();
        for action in actions {
            let outcome = self
                .apply_completion_action_detailed_with_outcome(&action, |_| Ok(()))
                .map_err(
                    |error| SynchronizationError::CompletionSourceOperationFailed {
                        source_name: self.source_name(),
                        details: error.to_string(),
                    },
                )?;
            let action_progress = outcome.progress();
            progress.completed_operations += action_progress.completed_operations;
            progress.woken_warps += action_progress.woken_warps;
        }
        Ok(progress)
    }

    fn blocked_operations(&self) -> Vec<BlockedOperation> {
        self.mbarriers.blocked_operations()
    }

    fn validate_quiescent(&self) -> Result<(), SynchronizationError> {
        self.mbarriers.validate_quiescent()?;
        let state = self.state.lock().expect("deferred payload mutex poisoned");
        // Only pending payloads keep a record.
        let pending = state.records.len();
        if pending != 0 {
            return Err(SynchronizationError::CompletionSourceNotQuiescent {
                source_name: self.source_name(),
                details: format!("{pending} deferred payload tokens remain incomplete"),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "deferred_payload_tests.rs"]
mod tests;
