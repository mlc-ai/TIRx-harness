use super::*;
use crate::{
    CompletionRegistry, EngineError, Executor, OperationEffect, ResolvedSyncResourceKey,
    ResolvedTransitionSummary, StaticOpId, WarpTask,
};

fn operation(context: WarpContext, sequence: u64, source: u64) -> OperationContext {
    OperationContext::new(
        DynamicOpId::new(
            7,
            context.global_warp_id(),
            sequence,
            StaticOpId::new(source),
            Vec::new().into_boxed_slice(),
        ),
        OperationKind::Collective,
        context.active_mask(),
    )
}

fn run_plans(
    topology: LaunchTopology,
    hub: Arc<SetmaxnregHub>,
    plans: Vec<SetmaxnregPlan>,
) -> Result<(), EngineError> {
    let tasks = plans.into_iter().map(|plan| {
        let warp_id = plan.context().global_warp_id();
        let hub = Arc::clone(&hub);
        WarpTask::new(warp_id, async move {
            plan.register(&hub)?.resume().await?;
            Ok(())
        })
    });
    let mut completions = CompletionRegistry::new();
    completions.register(Arc::clone(&hub));
    Executor::default()
        .run_with_topology_and_completions(tasks, topology, &completions)
        .map(|_| ())
}

fn error_kind(error: &EngineError) -> Option<SetmaxnregErrorKind> {
    match error.kind() {
        crate::EngineErrorKind::Synchronization(error) => match error.as_ref() {
            SynchronizationError::Setmaxnreg(error) => Some(error.kind()),
            _ => None,
        },
        crate::EngineErrorKind::Context { source, .. }
        | crate::EngineErrorKind::WarpFailed { source, .. } => error_kind(source),
        crate::EngineErrorKind::CompletionFailed { source, .. } => match source.as_ref() {
            SynchronizationError::Setmaxnreg(error) => Some(error.kind()),
            _ => None,
        },
        _ => None,
    }
}

fn plans(
    hub: &SetmaxnregHub,
    topology: LaunchTopology,
    source: impl Fn(usize) -> u64,
    action: impl Fn(usize) -> SetmaxnregAction,
    count: impl Fn(usize) -> i64,
) -> Vec<SetmaxnregPlan> {
    topology
        .warp_contexts()
        .map(|context| {
            hub.plan(
                7,
                Some(&operation(context, 0, source(context.global_warp_id()))),
                context,
                action(context.global_warp_id()),
                count(context.global_warp_id()),
            )
            .unwrap()
        })
        .collect()
}

fn warpgroup_plans(
    hub: &SetmaxnregHub,
    topology: LaunchTopology,
    global_cta_id: usize,
    warpgroup_id: usize,
    source: u64,
    action: SetmaxnregAction,
    count: i64,
) -> Vec<SetmaxnregPlan> {
    topology
        .warp_contexts()
        .filter(|context| {
            context.global_cta_id() == global_cta_id
                && context.warp_id_in_cta() / SETMAXNREG_WARPS_PER_GROUP == warpgroup_id
        })
        .map(|context| {
            hub.plan(
                7,
                Some(&operation(
                    context,
                    0,
                    source + context.global_warp_id() as u64,
                )),
                context,
                action,
                count,
            )
            .unwrap()
        })
        .collect()
}

fn publish_warpgroup_plans(hub: &SetmaxnregHub, plans: &[SetmaxnregPlan]) -> SetmaxnregOutcome {
    let contributions = plans
        .iter()
        .map(|plan| {
            (
                plan.context().global_warp_id(),
                SetmaxnregContribution {
                    resource: plan.resource(),
                    context: plan.context(),
                    action: plan.action(),
                    count: plan.count(),
                    witness: plan.witness.clone(),
                },
            )
        })
        .collect();
    publish_setmaxnreg(&hub.state, contributions).unwrap()
}

fn current_count(hub: &SetmaxnregHub, global_cta_id: usize, warpgroup_id: usize) -> i64 {
    hub.state
        .lock()
        .unwrap()
        .warpgroups
        .get(&(global_cta_id, warpgroup_id))
        .unwrap()
        .current_count
}

fn available_count(hub: &SetmaxnregHub, global_cta_id: usize) -> i64 {
    hub.state
        .lock()
        .unwrap()
        .ctas
        .get(&global_cta_id)
        .unwrap()
        .available_count
}

#[test]
fn count_must_match_the_ptx_range_and_granularity() {
    let topology = LaunchTopology::new(1, 1, 4).unwrap();
    for count in [23, 25, 257] {
        let hub = SetmaxnregHub::new(topology);
        let context = topology.warp_contexts().next().unwrap();
        let error = hub
            .plan(
                7,
                Some(&operation(context, 0, 10)),
                context,
                SetmaxnregAction::Decrease,
                count,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            SynchronizationError::Setmaxnreg(ref error)
                if error.kind() == SetmaxnregErrorKind::InvalidCount
        ));
        assert!(error.to_string().contains("multiple of 8"));
    }
    for count in [24, 32, 256] {
        let hub = SetmaxnregHub::new(topology);
        let context = topology.warp_contexts().next().unwrap();
        assert!(hub
            .plan(
                7,
                Some(&operation(context, 0, 11)),
                context,
                SetmaxnregAction::Increase,
                count,
            )
            .is_ok());
    }
}

#[test]
fn dynamic_ordinal_ignores_source_site_and_uniform_collective_completes() {
    let topology = LaunchTopology::new(1, 1, 4).unwrap();
    let hub = Arc::new(SetmaxnregHub::new(topology));
    let plans = plans(
        &hub,
        topology,
        |warp| 100 + warp as u64,
        |_| SetmaxnregAction::Decrease,
        |_| 88,
    );
    assert!(plans.iter().all(|plan| plan.resource().ordinal() == 0));
    assert!(
        plans
            .iter()
            .map(|plan| plan.witness().unwrap().source_op_id())
            .collect::<BTreeSet<_>>()
            .len()
            == 4
    );
    run_plans(topology, hub, plans).unwrap();
}

#[test]
fn different_source_sites_with_different_arguments_report_divergence() {
    let topology = LaunchTopology::new(1, 1, 4).unwrap();
    let hub = Arc::new(SetmaxnregHub::new(topology));
    let plans = plans(
        &hub,
        topology,
        |warp| if warp == 0 { 200 } else { 201 },
        |_| SetmaxnregAction::Decrease,
        |warp| if warp == 0 { 96 } else { 88 },
    );
    let error = run_plans(topology, hub, plans).unwrap_err();
    assert_eq!(error_kind(&error), Some(SetmaxnregErrorKind::Divergence));
}

#[test]
fn subsequent_setmaxnreg_requires_a_completed_explicit_warpgroup_sync() {
    let topology = LaunchTopology::new(1, 1, 4).unwrap();
    let hub = Arc::new(SetmaxnregHub::new(topology));
    let first = plans(
        &hub,
        topology,
        |_| 300,
        |_| SetmaxnregAction::Decrease,
        |_| 88,
    );
    run_plans(topology, Arc::clone(&hub), first).unwrap();
    let second = plans(
        &hub,
        topology,
        |_| 301,
        |_| SetmaxnregAction::Increase,
        |_| 96,
    );
    let error = run_plans(topology, hub, second).unwrap_err();
    assert_eq!(
        error_kind(&error),
        Some(SetmaxnregErrorKind::MissingWarpgroupSync)
    );

    let hub = Arc::new(SetmaxnregHub::new(topology));
    let first = plans(
        &hub,
        topology,
        |_| 400,
        |_| SetmaxnregAction::Decrease,
        |_| 88,
    );
    run_plans(topology, Arc::clone(&hub), first).unwrap();
    let barrier = NamedBarrierId::new(0, 3);
    for context in topology.warp_contexts() {
        hub.record_warpgroup_sync(context, barrier, 0, WarpMask::FULL)
            .unwrap();
    }
    let second = plans(
        &hub,
        topology,
        |_| 401,
        |_| SetmaxnregAction::Increase,
        |_| 96,
    );
    run_plans(topology, hub, second).unwrap();
}

#[test]
fn mixed_named_barrier_participants_do_not_credit_either_warpgroup() {
    let topology = LaunchTopology::new(1, 1, 8).unwrap();
    let hub = Arc::new(SetmaxnregHub::new(topology));
    let first = warpgroup_plans(&hub, topology, 0, 0, 410, SetmaxnregAction::Decrease, 128);
    run_plans(topology, Arc::clone(&hub), first).unwrap();

    let barrier = NamedBarrierId::new(0, 3);
    for context in topology
        .warp_contexts()
        .filter(|context| context.warp_id_in_cta() % SETMAXNREG_WARPS_PER_GROUP < 2)
    {
        hub.record_warpgroup_sync(context, barrier, 0, WarpMask::FULL)
            .unwrap();
    }

    let second = warpgroup_plans(&hub, topology, 0, 0, 420, SetmaxnregAction::Increase, 256);
    let error = run_plans(topology, hub, second).unwrap_err();
    assert_eq!(
        error_kind(&error),
        Some(SetmaxnregErrorKind::MissingWarpgroupSync)
    );
}

#[test]
fn cta_default_register_counts_follow_warpgroup_count_and_round_down() {
    for (warps_per_cta, expected_count) in [(4, 512), (8, 256), (12, 168), (16, 128)] {
        let topology = LaunchTopology::new(1, 1, warps_per_cta).unwrap();
        let hub = SetmaxnregHub::new(topology);
        let warpgroup_count = warps_per_cta / SETMAXNREG_WARPS_PER_GROUP;
        assert_eq!(available_count(&hub, 0), 0);
        for warpgroup_id in 0..warpgroup_count {
            assert_eq!(current_count(&hub, 0, warpgroup_id), expected_count);
        }
    }
}

#[test]
fn equal_target_is_a_legal_noop_for_both_actions() {
    let topology = LaunchTopology::new(1, 1, 8).unwrap();
    for action in [SetmaxnregAction::Decrease, SetmaxnregAction::Increase] {
        let hub = Arc::new(SetmaxnregHub::new(topology));
        let plans = warpgroup_plans(&hub, topology, 0, 0, 500, action, 256);
        run_plans(topology, Arc::clone(&hub), plans).unwrap();
        assert_eq!(current_count(&hub, 0, 0), 256);
        assert_eq!(available_count(&hub, 0), 0);
    }
}

#[test]
fn invalid_action_direction_is_a_typed_error() {
    let topology = LaunchTopology::new(1, 1, 8).unwrap();
    let hub = Arc::new(SetmaxnregHub::new(topology));
    let plans = warpgroup_plans(&hub, topology, 0, 0, 520, SetmaxnregAction::Increase, 128);
    let error = run_plans(topology, hub, plans).unwrap_err();
    assert_eq!(
        error_kind(&error),
        Some(SetmaxnregErrorKind::InvalidDirection)
    );
    assert!(error.to_string().contains("current register count 256"));

    let topology = LaunchTopology::new(1, 1, 16).unwrap();
    let hub = Arc::new(SetmaxnregHub::new(topology));
    let plans = warpgroup_plans(&hub, topology, 0, 0, 540, SetmaxnregAction::Decrease, 256);
    let error = run_plans(topology, hub, plans).unwrap_err();
    assert_eq!(
        error_kind(&error),
        Some(SetmaxnregErrorKind::InvalidDirection)
    );
    assert!(error.to_string().contains("current register count 128"));
}

#[test]
fn caller_launch_count_starts_with_an_empty_redistribution_pool() {
    let topology = LaunchTopology::new(1, 1, 8).unwrap();
    let hub = Arc::new(SetmaxnregHub::new(topology));
    hub.configure_calling_initial_count(200).unwrap();
    assert_eq!(current_count(&hub, 0, 0), 200);
    assert_eq!(current_count(&hub, 0, 1), 200);
    assert_eq!(available_count(&hub, 0), 0);

    let plans = warpgroup_plans(&hub, topology, 0, 0, 560, SetmaxnregAction::Increase, 200);
    run_plans(topology, Arc::clone(&hub), plans).unwrap();
    hub.validate_quiescent().unwrap();
}

#[test]
#[should_panic(expected = "exceeds topology default")]
fn caller_launch_count_cannot_exceed_topology_default() {
    let topology = LaunchTopology::new(1, 1, 12).unwrap();
    SetmaxnregHub::new(topology)
        .configure_calling_initial_count(200)
        .unwrap();
}

#[test]
fn caller_launch_count_allows_redistribution_above_initial_count() {
    let topology = LaunchTopology::new(1, 1, 12).unwrap();
    let hub = Arc::new(SetmaxnregHub::new(topology));
    hub.configure_calling_initial_count(80).unwrap();
    let plans = plans(
        &hub,
        topology,
        |_| 570,
        |warp| {
            if warp / 4 == 2 {
                SetmaxnregAction::Decrease
            } else {
                SetmaxnregAction::Increase
            }
        },
        |warp| [104, 88, 48][warp / 4],
    );
    run_plans(topology, Arc::clone(&hub), plans).unwrap();
    assert_eq!(available_count(&hub, 0), 0);
    hub.validate_quiescent().unwrap();
}

#[test]
fn lower_caller_launch_count_still_blocks_without_a_donor() {
    let topology = LaunchTopology::new(1, 1, 12).unwrap();
    let hub = Arc::new(SetmaxnregHub::new(topology));
    hub.configure_calling_initial_count(128).unwrap();
    let plans = warpgroup_plans(&hub, topology, 0, 0, 571, SetmaxnregAction::Increase, 256);
    let error = run_plans(topology, Arc::clone(&hub), plans).unwrap_err();
    assert!(matches!(
        error.kind(),
        crate::EngineErrorKind::Deadlock { .. }
    ));
}

#[test]
fn decrease_frees_pool_for_a_later_increase() {
    let topology = LaunchTopology::new(1, 1, 12).unwrap();
    let hub = Arc::new(SetmaxnregHub::new(topology));
    let decrease = warpgroup_plans(&hub, topology, 0, 1, 560, SetmaxnregAction::Decrease, 80);
    run_plans(topology, Arc::clone(&hub), decrease).unwrap();
    assert_eq!(current_count(&hub, 0, 1), 80);
    assert_eq!(available_count(&hub, 0), 88);

    let increase = warpgroup_plans(&hub, topology, 0, 0, 580, SetmaxnregAction::Increase, 256);
    run_plans(topology, Arc::clone(&hub), increase).unwrap();
    assert_eq!(current_count(&hub, 0, 0), 256);
    assert_eq!(available_count(&hub, 0), 0);
}

#[test]
fn outcomes_and_grants_retain_exact_budget_snapshots() {
    let topology = LaunchTopology::new(1, 1, 12).unwrap();
    let hub = SetmaxnregHub::new(topology);
    let increase = warpgroup_plans(&hub, topology, 0, 0, 590, SetmaxnregAction::Increase, 256);
    let increase_outcome = publish_warpgroup_plans(&hub, &increase);
    let pending_action = match increase_outcome.budget_disposition() {
        SetmaxnregBudgetDisposition::IncreasePending { action_id } => action_id,
        disposition => panic!("expected a pending increase, got {disposition:?}"),
    };
    assert_eq!(increase_outcome.current_count_before(), 168);
    assert_eq!(increase_outcome.available_count_before(), 0);

    let decrease = warpgroup_plans(&hub, topology, 0, 1, 600, SetmaxnregAction::Decrease, 80);
    let decrease_outcome = publish_warpgroup_plans(&hub, &decrease);
    assert_eq!(decrease_outcome.current_count_before(), 168);
    assert_eq!(decrease_outcome.available_count_before(), 0);
    assert_eq!(
        decrease_outcome.budget_disposition(),
        SetmaxnregBudgetDisposition::DecreaseApplied
    );

    let actions = hub.pending_completion_actions();
    let [action] = actions.as_slice() else {
        panic!("the released budget must enable exactly one increase");
    };
    assert_eq!(action.id(), pending_action);
    assert_eq!(action.required_count(), 88);
    assert_eq!(action.available_count_before(), 88);
    let completion = hub.apply_completion_detailed(action.id()).unwrap();
    assert_eq!(completion.action().required_count(), 88);
    assert_eq!(completion.action().available_count_before(), 88);

    let immediate_hub = SetmaxnregHub::new(topology);
    let decrease = warpgroup_plans(
        &immediate_hub,
        topology,
        0,
        1,
        610,
        SetmaxnregAction::Decrease,
        80,
    );
    publish_warpgroup_plans(&immediate_hub, &decrease);
    let increase = warpgroup_plans(
        &immediate_hub,
        topology,
        0,
        0,
        620,
        SetmaxnregAction::Increase,
        256,
    );
    let immediate = publish_warpgroup_plans(&immediate_hub, &increase);
    assert_eq!(immediate.current_count_before(), 168);
    assert_eq!(immediate.available_count_before(), 88);
    assert_eq!(
        immediate.budget_disposition(),
        SetmaxnregBudgetDisposition::IncreaseImmediate
    );
}

#[test]
fn increase_blocks_until_a_later_decrease_and_carries_release_provenance() {
    let topology = LaunchTopology::new(1, 1, 12).unwrap();
    let hub = Arc::new(SetmaxnregHub::new(topology));
    let increase = warpgroup_plans(&hub, topology, 0, 0, 600, SetmaxnregAction::Increase, 256);
    let decrease = warpgroup_plans(&hub, topology, 0, 1, 620, SetmaxnregAction::Decrease, 80);
    let outcomes = Arc::new(Mutex::new(BTreeMap::new()));
    let tasks = increase.into_iter().chain(decrease).map(|plan| {
        let warp_id = plan.context().global_warp_id();
        let hub = Arc::clone(&hub);
        let outcomes = Arc::clone(&outcomes);
        WarpTask::new(warp_id, async move {
            let resume = plan.register(&hub)?.resume().await?;
            outcomes.lock().unwrap().insert(warp_id, resume);
            Ok(())
        })
    });
    let mut completions = CompletionRegistry::new();
    completions.register(Arc::clone(&hub));
    Executor::default()
        .run_with_topology_and_completions(tasks, topology, &completions)
        .unwrap();

    assert_eq!(current_count(&hub, 0, 0), 256);
    assert_eq!(current_count(&hub, 0, 1), 80);
    assert_eq!(available_count(&hub, 0), 0);
    let outcomes = outcomes.lock().unwrap();
    for warp_id in 0..4 {
        let provenance = outcomes[&warp_id]
            .outcome()
            .availability_provenance()
            .unwrap();
        assert_eq!(provenance.releases().len(), 1);
        let release = &provenance.releases()[0];
        assert_eq!(release.release_resource().warpgroup_id(), 1);
        assert_eq!(release.release_operations().len(), 4);
        assert_eq!(
            release
                .release_operations()
                .iter()
                .map(DynamicOpId::global_warp_id)
                .collect::<Vec<_>>(),
            vec![4, 5, 6, 7],
        );
    }
}

#[test]
fn availability_provenance_tracks_multiple_and_partially_consumed_release_lots() {
    let run = |first_target: i64, second_target: i64| {
        let topology = LaunchTopology::new(1, 1, 32).unwrap();
        let hub = SetmaxnregHub::new(topology);
        let release_a = warpgroup_plans(&hub, topology, 0, 2, 830, SetmaxnregAction::Decrease, 24);
        let release_b = warpgroup_plans(&hub, topology, 0, 3, 850, SetmaxnregAction::Decrease, 24);
        let release_a_resource = release_a[0].resource();
        let release_b_resource = release_b[0].resource();
        publish_warpgroup_plans(&hub, &release_a);
        publish_warpgroup_plans(&hub, &release_b);

        let first = warpgroup_plans(
            &hub,
            topology,
            0,
            0,
            870,
            SetmaxnregAction::Increase,
            first_target,
        );
        let first_outcome = publish_warpgroup_plans(&hub, &first);
        let second = warpgroup_plans(
            &hub,
            topology,
            0,
            1,
            890,
            SetmaxnregAction::Increase,
            second_target,
        );
        let second_outcome = publish_warpgroup_plans(&hub, &second);
        (
            release_a_resource,
            release_b_resource,
            first_outcome,
            second_outcome,
        )
    };

    let (release_a, release_b, first, second) = run(96, 112);
    assert_eq!(
        first
            .availability_provenance()
            .unwrap()
            .releases()
            .iter()
            .map(SetmaxnregReleaseProvenance::release_resource)
            .collect::<Vec<_>>(),
        vec![release_a],
    );
    assert_eq!(
        second
            .availability_provenance()
            .unwrap()
            .releases()
            .iter()
            .map(SetmaxnregReleaseProvenance::release_resource)
            .collect::<Vec<_>>(),
        vec![release_a, release_b],
    );

    let (release_a, release_b, first, second) = run(104, 104);
    assert_eq!(
        first
            .availability_provenance()
            .unwrap()
            .releases()
            .iter()
            .map(SetmaxnregReleaseProvenance::release_resource)
            .collect::<Vec<_>>(),
        vec![release_a],
    );
    assert_eq!(
        second
            .availability_provenance()
            .unwrap()
            .releases()
            .iter()
            .map(SetmaxnregReleaseProvenance::release_resource)
            .collect::<Vec<_>>(),
        vec![release_b],
    );
}

#[test]
fn permanent_oversubscription_deadlocks_with_stable_diagnostics() {
    let run = || {
        let topology = LaunchTopology::new(1, 1, 12).unwrap();
        let hub = Arc::new(SetmaxnregHub::new(topology));
        let increase = warpgroup_plans(&hub, topology, 0, 0, 640, SetmaxnregAction::Increase, 256);
        run_plans(topology, hub, increase).unwrap_err()
    };
    let first = run();
    let second = run();
    assert_eq!(first.to_string(), second.to_string());
    let crate::EngineErrorKind::Deadlock {
        blocked_warps,
        blocked_operations,
        ..
    } = first.kind()
    else {
        panic!("oversubscribed setmaxnreg.inc must deadlock");
    };
    assert_eq!(blocked_warps.as_slice(), [0, 1, 2, 3]);
    assert_eq!(blocked_operations.len(), 4);
    assert!(blocked_operations
        .iter()
        .all(|blocked| blocked.to_string().contains("setmaxnreg.pool")));
}

#[test]
fn ordinary_executor_grants_pending_increases_in_stable_resource_order() {
    let topology = LaunchTopology::new(1, 1, 16).unwrap();
    let hub = Arc::new(SetmaxnregHub::new(topology));
    let first = warpgroup_plans(&hub, topology, 0, 0, 660, SetmaxnregAction::Increase, 200);
    let second = warpgroup_plans(&hub, topology, 0, 1, 680, SetmaxnregAction::Increase, 200);
    let release = warpgroup_plans(&hub, topology, 0, 2, 700, SetmaxnregAction::Decrease, 56);
    let error = run_plans(
        topology,
        Arc::clone(&hub),
        first.into_iter().chain(second).chain(release).collect(),
    )
    .unwrap_err();
    assert!(matches!(
        error.kind(),
        crate::EngineErrorKind::Deadlock { .. }
    ));
    assert_eq!(current_count(&hub, 0, 0), 200);
    assert_eq!(current_count(&hub, 0, 1), 128);
    assert_eq!(available_count(&hub, 0), 0);
}

#[test]
fn cta_register_pools_are_independent() {
    let topology = LaunchTopology::new(1, 2, 12).unwrap();
    let hub = Arc::new(SetmaxnregHub::new(topology));
    let remote_release = warpgroup_plans(&hub, topology, 1, 1, 720, SetmaxnregAction::Decrease, 80);
    run_plans(topology, Arc::clone(&hub), remote_release).unwrap();
    assert_eq!(available_count(&hub, 0), 0);
    assert_eq!(available_count(&hub, 1), 88);

    let local_increase =
        warpgroup_plans(&hub, topology, 0, 0, 740, SetmaxnregAction::Increase, 256);
    let error = run_plans(topology, Arc::clone(&hub), local_increase).unwrap_err();
    assert!(matches!(
        error.kind(),
        crate::EngineErrorKind::Deadlock { .. }
    ));
    assert_eq!(current_count(&hub, 0, 0), 168);
    assert_eq!(available_count(&hub, 0), 0);
    assert_eq!(current_count(&hub, 1, 1), 80);
    assert_eq!(available_count(&hub, 1), 88);
}

#[test]
fn resolved_resource_is_stable_across_warps_and_source_sites() {
    let topology = LaunchTopology::new(1, 1, 4).unwrap();
    let hub = SetmaxnregHub::new(topology);
    let plans = plans(
        &hub,
        topology,
        |warp| 600 + warp as u64,
        |_| SetmaxnregAction::Decrease,
        |_| 88,
    );

    let keys = plans
        .iter()
        .map(|plan| {
            let ResolvedTransitionSummary::Synchronization(effect) =
                ResolvedTransitionSummary::from_operation_effect(
                    OperationEffect::SetmaxnregRegister(plan),
                )
            else {
                panic!("setmaxnreg must resolve as synchronization");
            };
            effect
                .resources()
                .iter()
                .map(|resource| resource.key())
                .collect::<BTreeSet<_>>()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        keys,
        BTreeSet::from([BTreeSet::from([
            ResolvedSyncResourceKey::Setmaxnreg {
                kernel_index: 7,
                global_cta_id: 0,
                warpgroup_id: 0,
                ordinal: 0,
            },
            ResolvedSyncResourceKey::SetmaxnregPool {
                kernel_index: 7,
                global_cta_id: 0,
            },
        ])])
    );
}
