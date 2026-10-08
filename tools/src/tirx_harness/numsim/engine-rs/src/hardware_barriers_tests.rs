use super::*;
use crate::{CompletionRegistry, EngineError, Executor, StaticOpId, WarpTask};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::Wake;

struct WakeCounter(AtomicUsize);

impl Wake for WakeCounter {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct PublicationCheckingWake {
    published: Arc<AtomicBool>,
    woke_before_publication: Arc<AtomicBool>,
}

impl Wake for PublicationCheckingWake {
    fn wake(self: Arc<Self>) {
        if !self.published.load(Ordering::Acquire) {
            self.woke_before_publication.store(true, Ordering::Release);
        }
    }
}

fn physical_id() -> PhysicalBarrierId {
    PhysicalBarrierId::new(11, 16, 0)
}


#[test]
fn query_publication_holds_the_phase_gate_but_not_numeric_state_locks() {
    let hub = PhysicalBarrierHub::new();
    let id = physical_id();
    hub.init(id, 1).unwrap();
    hub.arrive(id, 0, 1).unwrap();
    let stripe = PhysicalBarrierHub::publication_stripe(id);
    for fail in [false, true] {
        let result =
            hub.query_many_with(&[(id, 0)], MbarrierQuery::ConditionalParity, |outcomes| {
                assert_eq!(outcomes, vec![(true, Some(0), false)]);
                assert!(matches!(
                    hub.publication_gates[stripe].try_lock(),
                    Err(std::sync::TryLockError::WouldBlock)
                ));
                assert!(hub.shards[stripe].try_lock().is_ok());
                if fail {
                    Err(EngineError::message("query publication failed"))
                } else {
                    Ok(())
                }
            });
        assert_eq!(result.is_err(), fail);
        assert!(hub.publication_gates[stripe].try_lock().is_ok());
    }
    assert!(hub.arrive(id, 0, 1).unwrap().completed_now());
}

#[test]
fn conditional_phase_retains_its_actual_primary_completion() {
    let id = physical_id();
    let hub = PhysicalBarrierHub::new();
    hub.init_many_layout(&[id], 1, true).unwrap();
    let query = |parity| {
        hub.query_many(&[(id, parity)], MbarrierQuery::ConditionalParity)
            .unwrap()[0]
    };
    assert_eq!(query(0), (false, None, false));
    assert!(hub
        .query_many(&[(id, 2)], MbarrierQuery::ConditionalParity)
        .is_err());
    for generation in 0..22 {
        let failed = generation != 1 && generation != 21;
        hub.report_on(id, failed).unwrap();
        let expected_pin = match generation {
            0 => None,
            21 => Some(21),
            _ => Some(1),
        };
        if generation % 2 == 0 {
            let outcome = hub.arrive(id, 0, 1).unwrap();
            assert_eq!(outcome.conditional_completed_generation(), expected_pin);
        } else {
            hub.arrive_expect_tx(id, 0, 1, 16).unwrap();
            let action = hub.enqueue_transaction_completion(id, 16).unwrap();
            let outcome = hub.apply_completion_detailed(action).unwrap();
            assert_eq!(outcome.conditional_completed_generation(), expected_pin);
        }
        assert_eq!(
            hub.query_many(&[(id, generation & 1)], MbarrierQuery::PrimaryParity)
                .unwrap()[0],
            (true, Some(generation), failed)
        );
        if generation == 0 {
            assert_eq!(query(0), (false, None, false));
        } else if generation < 21 {
            // More than the checkers' ordinary history window can pass while
            // the conditional phase still refers to primary generation 1.
            assert_eq!(query(0), (true, Some(1), false));
            assert_eq!(query(1), (false, None, false));
        } else {
            assert_eq!(query(0), (false, None, false));
            assert_eq!(query(1), (true, Some(21), false));
        }
    }
    hub.invalidate_many(&[id]).unwrap();
    hub.init(id, 1).unwrap();
    assert_eq!(query(0), (false, None, false));
    // Layout v0 cannot report failure; both phase counters advance together.
    for generation in 0..4 {
        hub.arrive(id, 0, 1).unwrap();
        assert_eq!(query(generation & 1), (true, Some(generation), false));
    }
}



#[test]
fn dropping_last_arrivals_completes_current_phase_with_zero_future_expectation() {
    let id = physical_id();
    let hub = PhysicalBarrierHub::new();
    hub.init(id, 2).unwrap();
    hub.drop_expected_arrivals([(id, 2)]).unwrap();
    hub.arrive(id, 0, 2).unwrap();
    assert!(hub.test_wait(id, 0).unwrap());
    assert!(hub.arrive(id, 0, 1).is_err());

    let strict = crate::StrictMbarrierProtocol::new();
    strict.init(id, 2, None).unwrap();
    strict.mark_init_fenced_many(&[id], None);
    let effect = strict.arrive_with_drop(id, 0, 2, None, true, None).unwrap();
    assert_eq!(effect.completed_generation(), Some(0));
}

fn second_physical_id() -> PhysicalBarrierId {
    PhysicalBarrierId::new(11, 32, 0)
}

#[test]
fn no_complete_preserves_pending_arrivals_across_phases_and_transactions() {
    let hub = PhysicalBarrierHub::new();
    assert!(hub.validate_no_complete([(physical_id(), 1)]).is_err());
    hub.init(physical_id(), 3).unwrap();
    for _ in 0..2 {
        // Validation does not mutate the barrier; counts are aggregated per target.
        assert!(hub.validate_no_complete([(physical_id(), 3)]).is_err());
        hub.validate_no_complete([(physical_id(), 2)]).unwrap();
        hub.arrive_expect_tx(physical_id(), 0, 2, 16).unwrap();
        // Outstanding bytes do not make exhausting arrivals legal for noComplete.
        assert!(hub.validate_no_complete([(physical_id(), 1)]).is_err());
        hub.arrive(physical_id(), 0, 1).unwrap();
        hub.complete_transactions_immediately(&[(physical_id(), 16)])
            .unwrap();
    }
}

#[test]
fn physical_arrival_outcome_is_generation_exact_at_commit() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 2).unwrap();

    let first = hub.arrive(physical_id(), 0, 1).unwrap();
    let second = hub.arrive(physical_id(), 1, 1).unwrap();
    let third = hub.arrive(physical_id(), 0, 1).unwrap();
    let fourth = hub.arrive(physical_id(), 1, 1).unwrap();

    assert_eq!(
        first,
        PhysicalMbarrierArrivalOutcome::new(0, false).with_pending_arrivals_before(2)
    );
    assert_eq!(
        second,
        PhysicalMbarrierArrivalOutcome::new(0, true).with_pending_arrivals_before(1)
    );
    assert_eq!(
        third,
        PhysicalMbarrierArrivalOutcome::new(1, false)
            .with_pending_arrivals_before(2)
            .with_conditional_generation(Some(0))
    );
    assert_eq!(
        fourth,
        PhysicalMbarrierArrivalOutcome::new(1, true).with_pending_arrivals_before(1)
    );
}

#[test]
fn physical_barrier_reuses_parity_without_leaking_lane_arrivals() {
    let hub = Arc::new(PhysicalBarrierHub::new());
    hub.init(physical_id(), 64).unwrap();
    let mut completions = CompletionRegistry::new();
    completions.register(Arc::clone(&hub));

    for phase in 0..2 {
        let tasks = [0, 1].map(|warp_id| {
            let hub = Arc::clone(&hub);
            WarpTask::new(warp_id, async move {
                hub.arrive(physical_id(), warp_id, 32)?;
                hub.wait(physical_id(), phase, warp_id)?.await?;
                Ok(())
            })
        });
        Executor::default()
            .run_with_completions(tasks, &completions)
            .unwrap();
    }
}

#[test]
fn freshly_initialized_phase_one_is_immediately_available() {
    let hub = Arc::new(PhysicalBarrierHub::new());
    hub.init(physical_id(), 1).unwrap();
    let task = {
        let hub = Arc::clone(&hub);
        WarpTask::new(0, async move {
            hub.wait(physical_id(), 1, 0)?.await?;
            Ok(())
        })
    };
    let stats = Executor::default().run([task]).unwrap();
    assert_eq!(stats.poll_count, 1);
}

#[test]
fn physical_barrier_multi_test_preserves_request_order() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.init(second_physical_id(), 1).unwrap();
    hub.arrive(second_physical_id(), 1, 1).unwrap();

    let ready = hub
        .test_wait_many(&[(physical_id(), 1), (second_physical_id(), 0)])
        .unwrap();

    assert_eq!(ready, vec![true, true]);
}

#[test]
fn multi_target_init_is_transactional() {
    let hub = PhysicalBarrierHub::new();
    hub.init(second_physical_id(), 2).unwrap();
    hub.arrive(second_physical_id(), 0, 1).unwrap();

    let error = hub
        .init_many(&[physical_id(), second_physical_id()], 1)
        .unwrap_err();
    assert!(matches!(
        error,
        SynchronizationError::BarrierReinitializedWhileActive { .. }
    ));

    let error = hub.arrive(physical_id(), 1, 1).unwrap_err();
    assert!(matches!(
        error,
        SynchronizationError::BarrierUninitialized { .. }
    ));
}

#[test]
fn same_lane_target_init_is_idempotent() {
    let hub = PhysicalBarrierHub::new();

    hub.init_many(&[physical_id(), second_physical_id(), physical_id()], 1)
        .unwrap();

    hub.arrive(physical_id(), 0, 1).unwrap();
    hub.arrive(second_physical_id(), 0, 1).unwrap();
}

#[test]
fn multi_target_arrive_is_transactional_and_rejects_duplicate_targets() {
    let hub = PhysicalBarrierHub::new();
    hub.init_many(&[physical_id(), second_physical_id()], 2)
        .unwrap();

    assert!(matches!(
        hub.arrive_many(&[(physical_id(), 0, 1, 0), (second_physical_id(), 0, 3, 0),]),
        Err(SynchronizationError::BarrierArrivalOverflow { .. })
    ));
    assert_eq!(
        hub.arrive(physical_id(), 1, 2).unwrap(),
        PhysicalMbarrierArrivalOutcome::new(0, true).with_pending_arrivals_before(2)
    );
    assert_eq!(
        hub.arrive(second_physical_id(), 1, 2).unwrap(),
        PhysicalMbarrierArrivalOutcome::new(0, true).with_pending_arrivals_before(2)
    );

    let duplicate_hub = PhysicalBarrierHub::new();
    duplicate_hub.init(physical_id(), 2).unwrap();
    assert!(matches!(
        duplicate_hub.arrive_many(&[(physical_id(), 0, 1, 0), (physical_id(), 0, 1, 0),]),
        Err(SynchronizationError::DuplicateMbarrierArrivalTarget { .. })
    ));
    assert_eq!(
        duplicate_hub.arrive(physical_id(), 1, 2).unwrap(),
        PhysicalMbarrierArrivalOutcome::new(0, true).with_pending_arrivals_before(2)
    );
}

#[test]
fn physical_barrier_init_rejects_invalid_count_and_valid_object_reuse() {
    let hub = PhysicalBarrierHub::new();

    for count in [0, MAX_MBARRIER_EXPECTED_ARRIVALS + 1] {
        assert!(matches!(
            hub.init(physical_id(), count),
            Err(SynchronizationError::InvalidBarrierArrivalCount {
                count: actual,
                ..
            }) if actual == count
        ));
    }

    hub.init(physical_id(), MAX_MBARRIER_EXPECTED_ARRIVALS)
        .unwrap();
    assert!(matches!(
        hub.init(physical_id(), 1),
        Err(SynchronizationError::BarrierReinitializedWithoutInvalidation { .. })
    ));
}

#[test]
fn physical_barrier_deadlock_reports_lane_and_transaction_counts() {
    let hub = Arc::new(PhysicalBarrierHub::new());
    hub.init(physical_id(), 64).unwrap();
    hub.arrive(physical_id(), 0, 32).unwrap();
    let task = {
        let hub = Arc::clone(&hub);
        WarpTask::new(0, async move {
            hub.wait(physical_id(), 0, 0)?.await?;
            Ok(())
        })
    };
    let mut completions = CompletionRegistry::new();
    completions.register(Arc::clone(&hub));
    let error = Executor::default()
        .run_with_completions([task], &completions)
        .unwrap_err();
    let crate::EngineErrorKind::Deadlock {
        blocked_operations, ..
    } = error.kind()
    else {
        panic!("expected deadlock");
    };
    assert_eq!(
        blocked_operations[0]
            .participant_state
            .completed_arrival_count,
        Some(32)
    );
    assert_eq!(
        blocked_operations[0]
            .participant_state
            .expected_arrival_count,
        Some(64)
    );
    assert!(blocked_operations[0]
        .to_string()
        .contains("arrival_count=32/64"));
}

#[test]
fn physical_barrier_blocked_operation_preserves_exact_wait_identity() {
    let hub = Arc::new(PhysicalBarrierHub::new());
    hub.init(physical_id(), 2).unwrap();
    hub.arrive(physical_id(), 0, 1).unwrap();
    let operation = DynamicOpId::new(0, 7, 19, StaticOpId::new(313), []);
    let expected = operation.clone();
    let task = {
        let hub = Arc::clone(&hub);
        WarpTask::new(7, async move {
            hub.wait_with_operation(physical_id(), 0, 7, Some(operation))?
                .await?;
            Ok(())
        })
    };
    let mut completions = CompletionRegistry::new();
    completions.register(Arc::clone(&hub));

    let error = Executor::default()
        .run_with_completions([task], &completions)
        .unwrap_err();
    let crate::EngineErrorKind::Deadlock {
        blocked_operations, ..
    } = error.kind()
    else {
        panic!("expected deadlock");
    };
    assert_eq!(blocked_operations.len(), 1);
    assert_eq!(blocked_operations[0].operation(), Some(&expected));
    assert_eq!(blocked_operations[0].key, physical_id().occurrence_key());
}

#[test]
fn physical_barrier_uses_copy_delivered_transaction_bytes() {
    let hub = Arc::new(PhysicalBarrierHub::new());
    hub.init(physical_id(), 1).unwrap();
    // Payload issue precedes expect_tx in the canonical TMA source order.
    hub.enqueue_transaction_completion(physical_id(), 64)
        .unwrap();
    let task = {
        let hub = Arc::clone(&hub);
        WarpTask::new(0, async move {
            hub.arrive_expect_tx(physical_id(), 0, 1, 64)?;
            hub.wait(physical_id(), 0, 0)?.await?;
            Ok(())
        })
    };
    let mut completions = CompletionRegistry::new();
    completions.register(Arc::clone(&hub));

    let stats = Executor::default()
        .run_with_completions([task], &completions)
        .unwrap();

    assert_eq!(stats.completed_task_count, 1);
    assert_eq!(stats.completion_operation_count, 1);
}

#[test]
fn physical_barrier_reinit_rejects_queued_old_completion() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.enqueue_transaction_completion(physical_id(), 64)
        .unwrap();

    let error = hub.init(physical_id(), 1).unwrap_err();

    assert!(matches!(
        error,
        SynchronizationError::BarrierReinitializedWhileActive { generation: 0, .. }
    ));
    assert_eq!(hub.pending_completion_count(), 1);

    assert_eq!(hub.pump().unwrap().completed_operations, 1);
    let error = hub.init(physical_id(), 1).unwrap_err();
    assert!(matches!(
        error,
        SynchronizationError::BarrierReinitializedWhileActive { generation: 0, .. }
    ));
}

#[test]
fn completed_physical_barrier_is_quiescent_without_an_explicit_wait() {
    let hub = Arc::new(PhysicalBarrierHub::new());
    hub.init(physical_id(), 1).unwrap();
    hub.enqueue_transaction_completion(physical_id(), 64)
        .unwrap();
    hub.enqueue_transaction_completion(physical_id(), 64)
        .unwrap();
    hub.arrive_expect_tx(physical_id(), 0, 1, 128).unwrap();
    let mut completions = CompletionRegistry::new();
    completions.register(Arc::clone(&hub));

    let stats = completions.drain_to_stable_and_validate().unwrap();

    assert_eq!(stats.completed_operation_count, 2);
    assert_eq!(hub.pending_completion_count(), 0);
}

#[test]
fn deferred_arrival_completes_only_when_its_action_is_applied() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();

    let action = hub.enqueue_arrival_completion(physical_id(), 3, 1).unwrap();
    assert_eq!(action.arrival(), Some((3, 1)));
    assert_eq!(hub.pending_completion_actions(), vec![action]);

    let outcome = hub.apply_completion_detailed(action.id()).unwrap();

    assert_eq!(
        outcome.arrival_outcome(),
        Some(PhysicalMbarrierArrivalOutcome::new(0, true))
    );
    assert_eq!(outcome.completed_generation(), Some(0));
    assert_eq!(outcome.progress().completed_operations, 1);
    assert!(hub.pending_completion_actions().is_empty());
}

#[test]
fn deferred_arrival_batch_resolves_all_targets_before_reserving_actions() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();

    let error = hub
        .enqueue_arrival_completions(&[(physical_id(), 3, 1), (second_physical_id(), 3, 1)])
        .unwrap_err();

    assert!(matches!(
        error,
        SynchronizationError::BarrierUninitialized { .. }
    ));
    assert!(hub.pending_completion_actions().is_empty());

    hub.init(second_physical_id(), 1).unwrap();
    let actions = hub
        .enqueue_arrival_completions(&[(physical_id(), 3, 1), (second_physical_id(), 3, 1)])
        .unwrap();
    assert_eq!(
        actions
            .iter()
            .map(|action| action.id().get())
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        actions
            .iter()
            .map(|action| action.barrier_id())
            .collect::<Vec<_>>(),
        vec![physical_id(), second_physical_id()]
    );
    assert!(actions
        .iter()
        .all(|action| action.arrival() == Some((3, 1))));
    assert_eq!(hub.pending_completion_actions(), actions.as_ref());
}

#[test]
fn deferred_arrival_reports_completion_after_advancing_generation() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.arrive(physical_id(), 1, 1).unwrap();

    let action = hub.enqueue_arrival_completion(physical_id(), 2, 1).unwrap();
    assert_eq!(action.generation(), 1);

    let outcome = hub.apply_completion_detailed(action.id()).unwrap();

    assert_eq!(outcome.completed_generation(), Some(1));
    assert_eq!(
        outcome.arrival_outcome(),
        Some(PhysicalMbarrierArrivalOutcome::new(1, true))
    );
}

#[test]
fn terminal_arrival_only_partial_phase_is_quiescent() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 64).unwrap();
    hub.arrive(physical_id(), 0, 32).unwrap();

    hub.validate_quiescent().unwrap();
}

#[test]
fn terminal_arrived_expect_tx_without_payload_is_quiescent() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.arrive_expect_tx(physical_id(), 0, 1, 64).unwrap();

    hub.validate_quiescent().unwrap();
}

#[test]
fn terminal_partial_phase_with_unresolved_transactions_is_not_quiescent() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 64).unwrap();
    hub.arrive_expect_tx(physical_id(), 0, 32, 64).unwrap();

    assert!(matches!(
        hub.validate_quiescent(),
        Err(SynchronizationError::CompletionSourceNotQuiescent { details, .. })
            if details.contains("arrivals=32/64")
                && details.contains("transactions=0/64")
    ));
}

#[test]
fn physical_completions_can_be_selected_individually_in_stable_order() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.init(second_physical_id(), 1).unwrap();
    hub.arrive_expect_tx(physical_id(), 0, 1, 32).unwrap();
    hub.arrive_expect_tx(second_physical_id(), 1, 1, 64)
        .unwrap();
    let first_id = hub
        .enqueue_transaction_completion(physical_id(), 32)
        .unwrap();
    let second_id = hub
        .enqueue_transaction_completion(second_physical_id(), 64)
        .unwrap();

    assert_eq!(second_id.get(), first_id.get() + 1);
    let actions = hub.pending_completion_actions();
    assert_eq!(
        actions.iter().map(|action| action.id()).collect::<Vec<_>>(),
        vec![first_id, second_id]
    );
    assert_eq!(actions[0].barrier_id(), physical_id());
    assert_eq!(actions[0].generation(), 0);
    assert_eq!(actions[0].transactions(), 32);

    assert_eq!(
        hub.apply_completion(second_id).unwrap(),
        CompletionProgress {
            completed_operations: 1,
            woken_warps: 0,
        }
    );
    assert_eq!(
        hub.pending_completion_actions()
            .iter()
            .map(|action| action.id())
            .collect::<Vec<_>>(),
        vec![first_id]
    );
    assert_eq!(
        hub.apply_completion(first_id).unwrap().completed_operations,
        1
    );
    hub.validate_quiescent().unwrap();
}

#[test]
fn multi_target_completion_enqueue_is_transactional_and_stably_numbered() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.init(second_physical_id(), 1).unwrap();
    let missing = PhysicalBarrierId::new(99, 24, 0);

    let error = hub
        .enqueue_transaction_completions(&[
            (physical_id(), 32),
            (missing, 64),
            (second_physical_id(), 64),
        ])
        .unwrap_err();
    assert!(matches!(
        error,
        SynchronizationError::BarrierUninitialized { .. }
    ));
    assert_eq!(hub.pending_completion_count(), 0);

    let ids = hub
        .enqueue_transaction_completions(&[
            (physical_id(), 32),
            (second_physical_id(), 64),
            (physical_id(), 0),
        ])
        .unwrap();
    assert_eq!(
        ids.iter().map(|id| id.get()).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    let actions = hub.pending_completion_actions();
    assert_eq!(actions.len(), 2);
    assert_eq!(actions[0].id(), ids[0]);
    assert_eq!(actions[0].barrier_id(), physical_id());
    assert_eq!(actions[1].id(), ids[1]);
    assert_eq!(actions[1].barrier_id(), second_physical_id());
}

#[test]
fn selected_physical_completion_wakes_its_waiters() {
    let hub = Arc::new(PhysicalBarrierHub::new());
    hub.init(physical_id(), 1).unwrap();
    hub.arrive_expect_tx(physical_id(), 0, 1, 64).unwrap();
    let action_id = hub
        .enqueue_transaction_completion(physical_id(), 64)
        .unwrap();
    let mut wait = Box::pin(hub.wait(physical_id(), 0, 0).unwrap());
    let wake_count = Arc::new(WakeCounter(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&wake_count));
    let mut context = Context::from_waker(&waker);
    assert_eq!(wait.as_mut().poll(&mut context), Poll::Pending);

    let outcome = hub.apply_completion_detailed(action_id).unwrap();
    assert_eq!(
        outcome.progress(),
        CompletionProgress {
            completed_operations: 1,
            woken_warps: 1,
        }
    );
    assert_eq!(outcome.action().id(), action_id);
    assert_eq!(outcome.completed_generation(), Some(0));
    assert_eq!(outcome.woken_warp_ids(), [0]);
    assert_eq!(wake_count.0.load(Ordering::SeqCst), 1);
    assert_eq!(wait.as_mut().poll(&mut context), Poll::Ready(Ok(Some(0))));
}

#[test]
fn future_generation_completion_is_enabled_and_buffered_before_generation_begins() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.arrive(physical_id(), 0, 1).unwrap();
    let action_id = hub
        .enqueue_transaction_completion(physical_id(), 64)
        .unwrap();

    assert_eq!(hub.pending_completion_actions()[0].id(), action_id);
    let outcome = hub.apply_completion_detailed(action_id).unwrap();
    assert_eq!(outcome.completed_generation(), None);
    assert!(outcome.woken_warp_ids().is_empty());
    assert_eq!(hub.pending_completion_count(), 0);
    assert!(matches!(
        hub.validate_quiescent(),
        Err(SynchronizationError::CompletionSourceNotQuiescent { details, .. })
            if details.contains("buffered_transactions={1: 64}")
    ));

    hub.arrive_expect_tx(physical_id(), 0, 1, 64).unwrap();
    assert!(hub.pending_completion_actions().is_empty());
    hub.validate_quiescent().unwrap();
}

#[test]
fn invalid_physical_completion_id_fails_without_consuming_an_action() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.arrive_expect_tx(physical_id(), 0, 1, 64).unwrap();
    let action_id = hub
        .enqueue_transaction_completion(physical_id(), 64)
        .unwrap();
    let invalid_id = PhysicalCompletionActionId(action_id.get() + 100);

    assert_eq!(
        hub.apply_completion(invalid_id).unwrap_err(),
        PhysicalCompletionActionError::Missing {
            action_id: invalid_id,
        }
    );
    assert_eq!(hub.pending_completion_count(), 1);
    assert_eq!(hub.pending_completion_actions()[0].id(), action_id);
    hub.apply_completion(action_id).unwrap();
    hub.validate_quiescent().unwrap();
}

#[test]
fn failed_physical_completion_does_not_partially_update_or_consume_action() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.arrive_expect_tx(physical_id(), 0, 1, 64).unwrap();
    let action_id = hub
        .enqueue_transaction_completion(physical_id(), 80)
        .unwrap();

    let error = hub.apply_completion(action_id).unwrap_err();
    assert!(matches!(
        error,
        PhysicalCompletionActionError::ApplyFailed {
            action,
            source: SynchronizationError::TransactionOverflow {
                expected: 64,
                completed: 80,
                ..
            },
        } if action.id() == action_id
    ));
    assert_eq!(hub.pending_completion_count(), 1);

    assert!(matches!(
        hub.apply_completion(action_id).unwrap_err(),
        PhysicalCompletionActionError::ApplyFailed {
            source: SynchronizationError::TransactionOverflow {
                expected: 64,
                completed: 80,
                ..
            },
            ..
        }
    ));
    assert_eq!(hub.pending_completion_count(), 1);
}

#[test]
fn failed_single_target_mutation_does_not_commit_barrier_completion() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.arrive_expect_tx(physical_id(), 0, 1, 64).unwrap();
    let action_id = hub
        .enqueue_transaction_completion(physical_id(), 64)
        .unwrap();

    let error = hub
        .apply_completion_batch_detailed_with_outcomes(
            &[action_id],
            || Err(EngineError::message("injected mutation failure")),
            |_| Ok(()),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        PhysicalCompletionBatchCommitError::Mutation(error)
            if error.to_string().contains("injected mutation failure")
    ));
    assert_eq!(hub.pending_completion_count(), 1);
    assert_eq!(hub.pending_completion_actions()[0].id(), action_id);

    hub.apply_completion(action_id).unwrap();
    hub.validate_quiescent().unwrap();
}

#[test]
fn one_pump_batches_independent_physical_barrier_completions() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.init(second_physical_id(), 1).unwrap();
    hub.arrive_expect_tx(physical_id(), 0, 1, 32).unwrap();
    hub.arrive_expect_tx(second_physical_id(), 1, 1, 64)
        .unwrap();
    let first_id = hub
        .enqueue_transaction_completion(physical_id(), 32)
        .unwrap();
    let second_id = hub
        .enqueue_transaction_completion(second_physical_id(), 64)
        .unwrap();

    assert_eq!(
        hub.pending_completion_actions()
            .iter()
            .map(|action| action.id())
            .collect::<Vec<_>>(),
        vec![first_id, second_id]
    );

    let progress = hub.pump().unwrap();

    assert_eq!(progress.completed_operations, 2);
    assert_eq!(hub.pending_completion_count(), 0);
    hub.validate_quiescent().unwrap();
}

#[test]
fn transaction_completion_can_precede_peer_expect_tx() {
    let hub = Arc::new(PhysicalBarrierHub::new());
    hub.init(physical_id(), 1).unwrap();

    hub.enqueue_transaction_completion(physical_id(), 16)
        .unwrap();
    assert_eq!(hub.pump().unwrap().completed_operations, 1);
    hub.arrive_expect_tx(physical_id(), 0, 1, 32).unwrap();
    hub.enqueue_transaction_completion(physical_id(), 16)
        .unwrap();

    let task = {
        let hub = Arc::clone(&hub);
        WarpTask::new(0, async move {
            hub.wait(physical_id(), 0, 0)?.await?;
            Ok(())
        })
    };
    let mut completions = CompletionRegistry::new();
    completions.register(Arc::clone(&hub));
    Executor::default()
        .run_with_completions([task], &completions)
        .unwrap();
}

#[test]
fn early_over_delivery_is_reported_when_expectation_is_fixed() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.enqueue_transaction_completion(physical_id(), 80)
        .unwrap();
    assert_eq!(hub.pump().unwrap().completed_operations, 1);

    let error = hub.arrive_expect_tx(physical_id(), 0, 1, 64).unwrap_err();
    assert!(matches!(
        error,
        SynchronizationError::TransactionOverflow {
            expected: 64,
            completed: 80,
            ..
        }
    ));
}

#[test]
fn payload_issued_after_completion_targets_the_next_generation() {
    let hub = Arc::new(PhysicalBarrierHub::new());
    hub.init(physical_id(), 1).unwrap();
    let first = {
        let hub = Arc::clone(&hub);
        WarpTask::new(0, async move {
            hub.arrive(physical_id(), 0, 1)?;
            hub.wait(physical_id(), 0, 0)?.await?;
            Ok(())
        })
    };
    Executor::default().run([first]).unwrap();

    // The previous phase is complete, but expect_tx has not started the
    // next generation yet. This payload must be tagged for generation 1.
    hub.enqueue_transaction_completion(physical_id(), 64)
        .unwrap();
    let second = {
        let hub = Arc::clone(&hub);
        WarpTask::new(0, async move {
            hub.arrive_expect_tx(physical_id(), 0, 1, 64)?;
            hub.wait(physical_id(), 1, 0)?.await?;
            Ok(())
        })
    };
    let mut completions = CompletionRegistry::new();
    completions.register(Arc::clone(&hub));

    let stats = Executor::default()
        .run_with_completions([second], &completions)
        .unwrap();

    assert_eq!(stats.completed_task_count, 1);
    assert_eq!(stats.completion_operation_count, 1);
}

#[test]
fn next_generation_payload_completion_buffers_until_expect_tx_arms_it() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.arrive(physical_id(), 0, 1).unwrap();

    hub.enqueue_transaction_completion(physical_id(), 64)
        .unwrap();
    assert_eq!(hub.pump().unwrap().completed_operations, 1);
    assert_eq!(hub.pending_completion_count(), 0);

    hub.arrive_expect_tx(physical_id(), 0, 1, 64).unwrap();
    assert_eq!(hub.pump().unwrap(), CompletionProgress::default());
    assert_eq!(hub.pending_completion_count(), 0);
    hub.validate_quiescent().unwrap();
}

#[test]
fn immediate_numeric_completion_finishes_an_armed_generation_without_an_action() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.arrive_expect_tx(physical_id(), 0, 1, 64).unwrap();

    hub.complete_transactions_immediately(&[(physical_id(), 64)])
        .unwrap();

    assert_eq!(hub.pending_completion_count(), 0);
    assert!(hub.test_wait(physical_id(), 0).unwrap());
    hub.validate_quiescent().unwrap();
}

#[test]
fn immediate_numeric_completion_buffers_for_the_next_generation() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.arrive(physical_id(), 0, 1).unwrap();

    hub.complete_transactions_immediately(&[(physical_id(), 64)])
        .unwrap();
    assert_eq!(hub.pending_completion_count(), 0);

    hub.arrive_expect_tx(physical_id(), 0, 1, 64).unwrap();
    assert!(hub.test_wait(physical_id(), 1).unwrap());
    hub.validate_quiescent().unwrap();
}

#[test]
fn rejected_expect_tx_preserves_the_completed_generation_and_future_credit() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.arrive(physical_id(), 0, 1).unwrap();
    let action_id = hub
        .enqueue_transaction_completion(physical_id(), 64)
        .unwrap();
    hub.apply_completion(action_id).unwrap();

    assert!(matches!(
        hub.arrive_expect_tx(physical_id(), 0, 1, 32),
        Err(SynchronizationError::TransactionOverflow {
            phase: 1,
            expected: 32,
            completed: 64,
            ..
        })
    ));
    assert!(matches!(
        hub.validate_quiescent(),
        Err(SynchronizationError::CompletionSourceNotQuiescent { details, .. })
            if details.contains("generation 0")
                && details.contains("arrivals=1/1")
                && details.contains("transactions=0/0")
                && details.contains("buffered_transactions={1: 64}")
    ));

    hub.arrive_expect_tx(physical_id(), 0, 1, 64).unwrap();
    hub.validate_quiescent().unwrap();
}

#[test]
fn buffered_next_generation_payload_does_not_block_an_armed_barrier() {
    let hub = PhysicalBarrierHub::new();
    hub.init(physical_id(), 1).unwrap();
    hub.init(second_physical_id(), 1).unwrap();
    hub.arrive(physical_id(), 0, 1).unwrap();
    hub.arrive(second_physical_id(), 1, 1).unwrap();

    hub.enqueue_transaction_completion(second_physical_id(), 64)
        .unwrap();
    hub.enqueue_transaction_completion(physical_id(), 32)
        .unwrap();
    hub.arrive_expect_tx(physical_id(), 0, 1, 32).unwrap();

    assert_eq!(hub.pump().unwrap().completed_operations, 2);
    assert_eq!(hub.pending_completion_count(), 0);

    hub.arrive_expect_tx(second_physical_id(), 1, 1, 64)
        .unwrap();
    assert_eq!(hub.pump().unwrap(), CompletionProgress::default());
    assert_eq!(hub.pending_completion_count(), 0);
    hub.validate_quiescent().unwrap();
}

#[test]
fn physical_barrier_under_delivery_deadlocks_with_exact_byte_counts() {
    let hub = Arc::new(PhysicalBarrierHub::new());
    hub.init(physical_id(), 1).unwrap();
    hub.enqueue_transaction_completion(physical_id(), 48)
        .unwrap();
    let task = {
        let hub = Arc::clone(&hub);
        WarpTask::new(0, async move {
            hub.arrive_expect_tx(physical_id(), 0, 1, 64)?;
            hub.wait(physical_id(), 0, 0)?.await?;
            Ok(())
        })
    };
    let mut completions = CompletionRegistry::new();
    completions.register(Arc::clone(&hub));

    let error = Executor::default()
        .run_with_completions([task], &completions)
        .unwrap_err();
    let crate::EngineErrorKind::Deadlock {
        blocked_operations, ..
    } = error.kind()
    else {
        panic!("expected deadlock");
    };
    assert_eq!(
        blocked_operations[0]
            .participant_state
            .completed_transactions,
        Some(48)
    );
    assert_eq!(
        blocked_operations[0]
            .participant_state
            .expected_transactions,
        Some(64)
    );
    assert!(blocked_operations[0]
        .to_string()
        .contains("transactions=48/64"));
}

#[test]
fn physical_barrier_over_delivery_is_a_completion_error() {
    let hub = Arc::new(PhysicalBarrierHub::new());
    hub.init(physical_id(), 1).unwrap();
    hub.enqueue_transaction_completion(physical_id(), 80)
        .unwrap();
    let task = {
        let hub = Arc::clone(&hub);
        WarpTask::new(0, async move {
            hub.arrive_expect_tx(physical_id(), 0, 1, 64)?;
            hub.wait(physical_id(), 0, 0)?.await?;
            Ok(())
        })
    };
    let mut completions = CompletionRegistry::new();
    completions.register(Arc::clone(&hub));

    let error = Executor::default()
        .run_with_completions([task], &completions)
        .unwrap_err();

    assert!(error
        .to_string()
        .contains("completed 80 transactions, expected 64"));
}

#[test]
fn named_barrier_counts_active_lanes_across_warps() {
    let hub = Arc::new(NamedBarrierHub::new());
    let id = NamedBarrierId::new(0, 7);
    let tasks = (0..4).map(|warp_id| {
        let hub = Arc::clone(&hub);
        WarpTask::new(warp_id, async move {
            hub.sync(id, 128, warp_id, 32)?.await?;
            Ok(())
        })
    });
    let mut completions = CompletionRegistry::new();
    completions.register(Arc::clone(&hub));
    let stats = Executor::default()
        .run_with_completions(tasks, &completions)
        .unwrap();
    assert_eq!(stats.completed_task_count, 4);
    hub.validate_quiescent().unwrap();
}

#[test]
fn named_barrier_publishes_last_arrival_before_waking_waiters() {
    let hub = Arc::new(NamedBarrierHub::new());
    let id = NamedBarrierId::new(0, 8);
    let published = Arc::new(AtomicBool::new(false));
    let woke_before_publication = Arc::new(AtomicBool::new(false));
    let waker = Waker::from(Arc::new(PublicationCheckingWake {
        published: Arc::clone(&published),
        woke_before_publication: Arc::clone(&woke_before_publication),
    }));
    let mut context = Context::from_waker(&waker);

    let mut waiter = Box::pin(hub.register_sync(id, 64, 0, WarpMask::FULL).unwrap());
    assert_eq!(waiter.as_mut().poll(&mut context), Poll::Pending);
    let mut last_arriver = Box::pin(
        hub.register_sync_with_outcome(id, 64, 1, WarpMask::FULL, |_| {
            published.store(true, Ordering::Release);
            Ok(())
        })
        .unwrap(),
    );

    assert!(last_arriver.completed_now());
    assert!(!woke_before_publication.load(Ordering::Acquire));
    assert_eq!(waiter.as_mut().poll(&mut context), Poll::Ready(Ok(())));
    assert_eq!(
        last_arriver.as_mut().poll(&mut context),
        Poll::Ready(Ok(()))
    );
    hub.validate_quiescent().unwrap();
}

#[test]
fn named_barrier_preserves_disjoint_lane_masks_from_one_warp() {
    let hub = Arc::new(NamedBarrierHub::new());
    let id = NamedBarrierId::new(0, 9);
    let low = WarpMask::from_bits(0x0000_000f);
    let high = WarpMask::from_bits(0x0000_00f0);

    let mut first = Box::pin(hub.register_sync(id, 8, 0, low).unwrap());
    assert_eq!(first.generation(), 0);
    assert!(!first.completed_now());
    let mut second = Box::pin(hub.register_sync(id, 8, 0, high).unwrap());
    assert_eq!(second.generation(), 0);
    assert!(second.completed_now());

    let waker = Waker::from(Arc::new(WakeCounter(AtomicUsize::new(0))));
    let mut context = Context::from_waker(&waker);
    assert_eq!(first.as_mut().poll(&mut context), Poll::Ready(Ok(())));
    assert_eq!(second.as_mut().poll(&mut context), Poll::Ready(Ok(())));
    hub.validate_quiescent().unwrap();
}

#[test]
fn named_barrier_arrive_and_sync_share_one_generation() {
    let hub = Arc::new(NamedBarrierHub::new());
    let id = NamedBarrierId::new(0, 12);

    let arrival = hub.arrive(id, 64, 0, WarpMask::FULL).unwrap();
    assert_eq!(arrival.generation(), 0);
    assert!(!arrival.completed_now());

    let mut waiter = Box::pin(hub.register_sync(id, 64, 1, WarpMask::FULL).unwrap());
    assert_eq!(waiter.generation(), 0);
    assert!(waiter.completed_now());
    let waker = Waker::from(Arc::new(WakeCounter(AtomicUsize::new(0))));
    let mut context = Context::from_waker(&waker);
    assert_eq!(waiter.as_mut().poll(&mut context), Poll::Ready(Ok(())));
    hub.validate_quiescent().unwrap();
}

#[test]
fn named_barrier_rejects_overlapping_lane_masks_transactionally() {
    let hub = Arc::new(NamedBarrierHub::new());
    let id = NamedBarrierId::new(0, 10);
    let low = WarpMask::from_bits(0x0000_000f);
    let overlap = WarpMask::from_bits(0x0000_000c);
    let high = WarpMask::from_bits(0x0000_00f0);

    let mut first = Box::pin(hub.register_sync(id, 8, 0, low).unwrap());
    let error = match hub.register_sync(id, 8, 0, overlap) {
        Ok(_) => panic!("overlapping named-barrier contribution must fail"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        SynchronizationError::DuplicateArrival { .. }
    ));
    let blocked = hub.blocked_operations();
    assert_eq!(blocked.len(), 1);
    assert_eq!(
        blocked[0].participant_state.completed_arrival_count,
        Some(4)
    );

    let mut second = Box::pin(hub.register_sync(id, 8, 0, high).unwrap());
    let waker = Waker::from(Arc::new(WakeCounter(AtomicUsize::new(0))));
    let mut context = Context::from_waker(&waker);
    assert_eq!(first.as_mut().poll(&mut context), Poll::Ready(Ok(())));
    assert_eq!(second.as_mut().poll(&mut context), Poll::Ready(Ok(())));
    hub.validate_quiescent().unwrap();
}

#[test]
fn named_barrier_contract_mismatch_does_not_mutate_registration() {
    let hub = Arc::new(NamedBarrierHub::new());
    let id = NamedBarrierId::new(0, 11);
    let low = WarpMask::from_bits(0x0000_000f);
    let high = WarpMask::from_bits(0x0000_00f0);

    let mut first = Box::pin(hub.register_sync(id, 8, 0, low).unwrap());
    let error = match hub.register_sync(id, 7, 1, high) {
        Ok(_) => panic!("named-barrier contract mismatch must fail"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        SynchronizationError::ContractMismatch { .. }
    ));
    let blocked = hub.blocked_operations();
    assert_eq!(blocked.len(), 1);
    assert_eq!(blocked[0].participant_state.expected_arrival_count, Some(8));
    assert_eq!(
        blocked[0].participant_state.completed_arrival_count,
        Some(4)
    );

    let mut second = Box::pin(hub.register_sync(id, 8, 1, high).unwrap());
    let waker = Waker::from(Arc::new(WakeCounter(AtomicUsize::new(0))));
    let mut context = Context::from_waker(&waker);
    assert_eq!(first.as_mut().poll(&mut context), Poll::Ready(Ok(())));
    assert_eq!(second.as_mut().poll(&mut context), Poll::Ready(Ok(())));
    hub.validate_quiescent().unwrap();
}

#[test]
fn named_barrier_counts_arrive_then_sync_from_the_same_lanes() {
    let hub = Arc::new(NamedBarrierHub::new());
    let id = NamedBarrierId::new(0, 14);

    let arrival = hub.arrive(id, 64, 0, WarpMask::FULL).unwrap();
    assert_eq!(arrival.generation(), 0);
    assert!(!arrival.completed_now());

    let mut waiter = Box::pin(hub.register_sync(id, 64, 0, WarpMask::FULL).unwrap());
    assert_eq!(waiter.generation(), 0);
    assert!(waiter.completed_now());
    let waker = Waker::from(Arc::new(WakeCounter(AtomicUsize::new(0))));
    let mut context = Context::from_waker(&waker);
    assert_eq!(waiter.as_mut().poll(&mut context), Poll::Ready(Ok(())));
    hub.validate_quiescent().unwrap();
}

#[test]
fn named_barrier_arrive_overflow_is_transactional() {
    let hub = Arc::new(NamedBarrierHub::new());
    let id = NamedBarrierId::new(0, 13);

    let first = hub.arrive(id, 48, 0, WarpMask::FULL).unwrap();
    assert!(!first.completed_now());
    assert!(matches!(
        hub.arrive(id, 48, 1, WarpMask::FULL),
        Err(SynchronizationError::BarrierArrivalOverflow { .. })
    ));

    let low_half = WarpMask::from_bits(0x0000_ffff);
    let completion = hub.arrive(id, 48, 1, low_half).unwrap();
    assert_eq!(completion.generation(), 0);
    assert!(completion.completed_now());
    hub.validate_quiescent().unwrap();
}

/// The init-fence tracker must reproduce `SyncCausality`'s fence rule: cover
/// only barriers this warp initialized from a lane in the mask, cover each at
/// most once per initialization, and re-arm on re-initialization.
#[test]
fn init_fence_tracker_covers_each_initialization_once_per_owning_lane() {
    let tracker = MbarrierInitFenceTracker::new();
    let first = PhysicalBarrierId::new(0, 0, 0);
    let second = PhysicalBarrierId::new(0, 8, 0);
    let other_warp = PhysicalBarrierId::new(0, 16, 0);

    tracker.record_init(first, 0, 0);
    tracker.record_init(second, 0, 3);
    tracker.record_init(other_warp, 1, 0);

    // A mask that excludes lane 3 leaves `second` uncovered, and warp 1's
    // barrier is never this warp's to fence.
    assert_eq!(
        tracker.take_fenced(0, WarpMask::from_lanes([0]).unwrap()),
        vec![first]
    );
    // Already fenced: a second fence covers nothing new.
    assert!(tracker
        .take_fenced(0, WarpMask::from_lanes([0]).unwrap())
        .is_empty());
    // Widening the mask now picks up the still-unfenced lane-3 barrier only.
    assert_eq!(tracker.take_fenced(0, WarpMask::FULL), vec![second]);

    // Re-initialization re-arms eligibility and may hand the barrier to a new
    // owning lane, exactly as `mbarrier_reinitialize` resets `init_fence_clock`.
    tracker.record_init(first, 0, 5);
    assert!(tracker
        .take_fenced(0, WarpMask::from_lanes([0]).unwrap())
        .is_empty());
    assert_eq!(tracker.take_fenced(0, WarpMask::FULL), vec![first]);
}

/// The covered set is ordered by `PhysicalBarrierId`, which is the order the
/// causality map's `BTreeMap` yields, regardless of initialization order.
#[test]
fn init_fence_tracker_covers_in_barrier_id_order() {
    let tracker = MbarrierInitFenceTracker::new();
    let low = PhysicalBarrierId::new(0, 0, 0);
    let high = PhysicalBarrierId::new(0, 64, 0);

    tracker.record_init(high, 0, 1);
    tracker.record_init(low, 0, 0);

    assert_eq!(tracker.take_fenced(0, WarpMask::FULL), vec![low, high]);
}

/// The engine-internal warpgroup rendezvous keys one static site to one
/// barrier identity per warpgroup. Both identities are derived through the
/// production planner (`plan_internal_warpgroup_sync`) from real
/// `WarpContext` values, so its context-to-warpgroup/expected-arrivals
/// derivation is pinned together with the hub semantics. Two warpgroups on
/// the same static site and hardware barrier id must count arrivals and
/// advance generations independently: a completion in one group never
/// releases the other group's waiters and never advances the other group's
/// generation.
#[test]
fn internal_warpgroup_rendezvous_generations_advance_without_cross_talk() {
    let hub = Arc::new(NamedBarrierHub::new());
    let static_op_id = 4242;
    // One CTA holding exactly two production warpgroups.
    let warps_per_group = crate::SETMAXNREG_WARPS_PER_GROUP;
    let topology = crate::LaunchTopology::new(1, 1, 2 * warps_per_group).unwrap();
    // Plan through the production planner from the real context of one warp.
    let plan = |warp_in_cta: usize| {
        let context = topology
            .warp_contexts()
            .find(|context| context.warp_id_in_cta() == warp_in_cta)
            .unwrap();
        crate::runtime::sync::plan_internal_warpgroup_sync(
            &context,
            static_op_id,
            warps_per_group,
            WarpMask::FULL,
        )
        .unwrap()
        .unwrap()
    };
    let waker = Waker::from(Arc::new(WakeCounter(AtomicUsize::new(0))));
    let mut context = Context::from_waker(&waker);

    // Planner derivation: warps 0..4 share warpgroup 0's identity, warps
    // 4..8 share warpgroup 1's distinct identity on the same static site,
    // and each group expects its four full warps.
    assert_eq!(plan(0).barrier_id(), plan(warps_per_group - 1).barrier_id());
    assert_eq!(
        plan(warps_per_group).barrier_id(),
        plan(2 * warps_per_group - 1).barrier_id()
    );
    assert_ne!(plan(0).barrier_id(), plan(warps_per_group).barrier_id());
    assert_eq!(
        plan(0).expected_arrivals(),
        (warps_per_group * WARP_SIZE) as u64
    );

    // Group 1's generation-0 waiter registers first and must stay pending
    // across both group-0 completions.
    let group_one_waiter = plan(warps_per_group).register(&hub).unwrap();
    assert_eq!(group_one_waiter.outcome().generation(), 0);
    assert!(!group_one_waiter.outcome().completed_now());
    let mut group_one_wait = Box::pin(group_one_waiter.resume());
    assert!(matches!(
        group_one_wait.as_mut().poll(&mut context),
        Poll::Pending
    ));

    // While that waiter is pending, the hub reports it under the
    // warpgroup-scoped occurrence identity the planner derived.
    let blocked = hub.blocked_operations();
    assert_eq!(blocked.len(), 1);
    assert_eq!(blocked[0].warp_id, warps_per_group);
    assert_eq!(blocked[0].key.static_op_id(), static_op_id);
    assert_eq!(
        blocked[0].key.scope(),
        &ScopeInstance::WarpGroup {
            global_cta_id: 0,
            warpgroup_id: 1,
        }
    );
    assert_eq!(blocked[0].key.loop_iteration_path(), &[0]);

    // Group 0 completes two consecutive generations, each from all four of
    // its planned warps.
    for generation in 0..2u64 {
        let mut waits = Vec::new();
        for warp_in_cta in 0..warps_per_group - 1 {
            let registered = plan(warp_in_cta).register(&hub).unwrap();
            assert_eq!(registered.outcome().generation(), generation);
            assert!(!registered.outcome().completed_now());
            let mut wait = Box::pin(registered.resume());
            assert!(matches!(wait.as_mut().poll(&mut context), Poll::Pending));
            waits.push(wait);
        }
        let completer = plan(warps_per_group - 1).register(&hub).unwrap();
        assert_eq!(completer.outcome().generation(), generation);
        assert!(completer.outcome().completed_now());
        waits.push(Box::pin(completer.resume()));
        for wait in &mut waits {
            assert!(matches!(
                wait.as_mut().poll(&mut context),
                Poll::Ready(Ok(_))
            ));
        }

        // No cross-talk: group 0 completing this generation neither released
        // group 1's waiter nor advanced group 1's generation.
        assert!(matches!(
            group_one_wait.as_mut().poll(&mut context),
            Poll::Pending
        ));
    }

    // Group 1 still completes from its own generation 0 ...
    let mut group_one_waits = Vec::new();
    for warp_in_cta in warps_per_group + 1..2 * warps_per_group - 1 {
        let registered = plan(warp_in_cta).register(&hub).unwrap();
        assert_eq!(registered.outcome().generation(), 0);
        assert!(!registered.outcome().completed_now());
        let mut wait = Box::pin(registered.resume());
        assert!(matches!(wait.as_mut().poll(&mut context), Poll::Pending));
        group_one_waits.push(wait);
    }
    let group_one_completer = plan(2 * warps_per_group - 1).register(&hub).unwrap();
    assert_eq!(group_one_completer.outcome().generation(), 0);
    assert!(group_one_completer.outcome().completed_now());
    group_one_waits.push(group_one_wait);
    group_one_waits.push(Box::pin(group_one_completer.resume()));
    for wait in &mut group_one_waits {
        assert!(matches!(
            wait.as_mut().poll(&mut context),
            Poll::Ready(Ok(_))
        ));
    }

    // ... and then advances to its own generation 1, independent of group 0
    // already sitting at completed generation 1.
    let mut second_waits = Vec::new();
    for warp_in_cta in warps_per_group..2 * warps_per_group - 1 {
        let registered = plan(warp_in_cta).register(&hub).unwrap();
        assert_eq!(registered.outcome().generation(), 1);
        assert!(!registered.outcome().completed_now());
        let mut wait = Box::pin(registered.resume());
        assert!(matches!(wait.as_mut().poll(&mut context), Poll::Pending));
        second_waits.push(wait);
    }
    let second_completer = plan(2 * warps_per_group - 1).register(&hub).unwrap();
    assert_eq!(second_completer.outcome().generation(), 1);
    assert!(second_completer.outcome().completed_now());
    second_waits.push(Box::pin(second_completer.resume()));
    for wait in &mut second_waits {
        assert!(matches!(
            wait.as_mut().poll(&mut context),
            Poll::Ready(Ok(_))
        ));
    }
    hub.validate_quiescent().unwrap();
}
