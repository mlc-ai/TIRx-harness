use super::*;
use std::collections::{HashSet, VecDeque as TestQueue};

#[derive(Clone, Copy, Debug)]
struct VerifierRequest {
    resource: SetmaxnregResource,
    action: SetmaxnregAction,
    target_count: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ReferencePendingIncrease {
    target_count: i64,
    required_count: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ReferenceRegisterPool {
    available_count: i64,
    current_counts: Vec<i64>,
    pending_increases: BTreeMap<SetmaxnregResource, ReferencePendingIncrease>,
}

impl ReferenceRegisterPool {
    fn new(available_count: i64, current_counts: impl IntoIterator<Item = i64>) -> Self {
        let current_counts = current_counts.into_iter().collect::<Vec<_>>();
        assert_eq!(
            available_count + current_counts.iter().sum::<i64>(),
            SETMAXNREG_CTA_REGISTER_POOL,
        );
        Self {
            available_count,
            current_counts,
            pending_increases: BTreeMap::new(),
        }
    }

    fn warpgroup_has_pending_increase(&self, warpgroup_id: usize) -> bool {
        self.pending_increases
            .keys()
            .any(|resource| resource.warpgroup_id() == warpgroup_id)
    }

    fn apply_request(&mut self, request: VerifierRequest) {
        let warpgroup_id = request.resource.warpgroup_id();
        assert!(!self.warpgroup_has_pending_increase(warpgroup_id));
        let current_count = self.current_counts[warpgroup_id];
        match request.action {
            SetmaxnregAction::Decrease => {
                assert!(request.target_count <= current_count);
                self.current_counts[warpgroup_id] = request.target_count;
                self.available_count += current_count - request.target_count;
            }
            SetmaxnregAction::Increase => {
                assert!(request.target_count >= current_count);
                let required_count = request.target_count - current_count;
                if required_count <= self.available_count {
                    self.available_count -= required_count;
                    self.current_counts[warpgroup_id] = request.target_count;
                } else {
                    self.pending_increases.insert(
                        request.resource,
                        ReferencePendingIncrease {
                            target_count: request.target_count,
                            required_count,
                        },
                    );
                }
            }
        }
        assert_eq!(
            self.available_count + self.current_counts.iter().sum::<i64>(),
            SETMAXNREG_CTA_REGISTER_POOL,
        );
    }

    fn enabled_grants(&self) -> Vec<SetmaxnregResource> {
        self.pending_increases
            .iter()
            .filter_map(|(&resource, pending)| {
                (pending.required_count <= self.available_count).then_some(resource)
            })
            .collect()
    }

    fn apply_grant(&mut self, resource: SetmaxnregResource) {
        let pending = self.pending_increases[&resource];
        assert!(pending.required_count <= self.available_count);
        self.available_count -= pending.required_count;
        self.current_counts[resource.warpgroup_id()] = pending.target_count;
        self.pending_increases.remove(&resource);
        assert_eq!(
            self.available_count + self.current_counts.iter().sum::<i64>(),
            SETMAXNREG_CTA_REGISTER_POOL,
        );
    }

    fn grant_status(&self) -> SetmaxnregVerifierGrantStatus {
        if self.pending_increases.is_empty() {
            SetmaxnregVerifierGrantStatus::Quiescent
        } else if self
            .pending_increases
            .values()
            .any(|pending| pending.required_count <= self.available_count)
        {
            SetmaxnregVerifierGrantStatus::Grantable
        } else {
            SetmaxnregVerifierGrantStatus::Stalled
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct VerifierSnapshot {
    issued_requests: u64,
    available_count: i64,
    current_counts: Vec<i64>,
    pending_increases: Vec<(SetmaxnregResource, i64, i64)>,
    grant_status: SetmaxnregVerifierGrantStatus,
}

#[derive(Debug, PartialEq, Eq)]
struct VerifierExploration {
    reachable: HashSet<VerifierSnapshot>,
    terminal: HashSet<VerifierSnapshot>,
}

fn verifier_resource(warpgroup_id: usize, ordinal: u64) -> SetmaxnregResource {
    SetmaxnregResource::new(7, 0, warpgroup_id, ordinal)
}

fn core_snapshot(core: &SetmaxnregVerifierCore, issued_requests: u64) -> VerifierSnapshot {
    VerifierSnapshot {
        issued_requests,
        available_count: core.available_count(),
        current_counts: core.current_counts().to_vec(),
        pending_increases: core
            .pending_increases()
            .iter()
            .map(|(&resource, pending)| {
                (resource, pending.target_count(), pending.required_count())
            })
            .collect(),
        grant_status: core.grant_status(),
    }
}

fn reference_snapshot(pool: &ReferenceRegisterPool, issued_requests: u64) -> VerifierSnapshot {
    VerifierSnapshot {
        issued_requests,
        available_count: pool.available_count,
        current_counts: pool.current_counts.clone(),
        pending_increases: pool
            .pending_increases
            .iter()
            .map(|(&resource, pending)| (resource, pending.target_count, pending.required_count))
            .collect(),
        grant_status: pool.grant_status(),
    }
}

fn explore_core(
    initial: SetmaxnregVerifierCore,
    requests: &[VerifierRequest],
) -> VerifierExploration {
    assert!(requests.len() <= u64::BITS as usize);
    let mut queue = TestQueue::from([(initial, 0_u64)]);
    let mut seen = HashSet::new();
    let mut reachable = HashSet::new();
    let mut terminal = HashSet::new();
    while let Some((core, issued_requests)) = queue.pop_front() {
        if !seen.insert((core.clone(), issued_requests)) {
            continue;
        }
        let snapshot = core_snapshot(&core, issued_requests);
        reachable.insert(snapshot.clone());
        let mut has_successor = false;
        for (request_index, request) in requests.iter().copied().enumerate() {
            let request_bit = 1_u64 << request_index;
            if issued_requests & request_bit != 0
                || core.warpgroup_has_pending_increase(request.resource.warpgroup_id())
            {
                continue;
            }
            let mut next = core.clone();
            next.apply_request(request.resource, request.action, request.target_count)
                .unwrap();
            queue.push_back((next, issued_requests | request_bit));
            has_successor = true;
        }
        for resource in core.enabled_grants() {
            let mut next = core.clone();
            next.apply_grant(resource).unwrap();
            queue.push_back((next, issued_requests));
            has_successor = true;
        }
        assert_eq!(core.is_grant_terminal(), core.enabled_grants().is_empty());
        assert_eq!(core.is_quiescent(), core.pending_increases().is_empty());
        if !has_successor {
            terminal.insert(snapshot);
        }
    }
    VerifierExploration {
        reachable,
        terminal,
    }
}

fn explore_reference(
    initial: ReferenceRegisterPool,
    requests: &[VerifierRequest],
) -> VerifierExploration {
    assert!(requests.len() <= u64::BITS as usize);
    let mut queue = TestQueue::from([(initial, 0_u64)]);
    let mut seen = HashSet::new();
    let mut reachable = HashSet::new();
    let mut terminal = HashSet::new();
    while let Some((pool, issued_requests)) = queue.pop_front() {
        if !seen.insert((pool.clone(), issued_requests)) {
            continue;
        }
        let snapshot = reference_snapshot(&pool, issued_requests);
        reachable.insert(snapshot.clone());
        let mut has_successor = false;
        for (request_index, request) in requests.iter().copied().enumerate() {
            let request_bit = 1_u64 << request_index;
            if issued_requests & request_bit != 0
                || pool.warpgroup_has_pending_increase(request.resource.warpgroup_id())
            {
                continue;
            }
            let mut next = pool.clone();
            next.apply_request(request);
            queue.push_back((next, issued_requests | request_bit));
            has_successor = true;
        }
        for resource in pool.enabled_grants() {
            let mut next = pool.clone();
            next.apply_grant(resource);
            queue.push_back((next, issued_requests));
            has_successor = true;
        }
        if !has_successor {
            terminal.insert(snapshot);
        }
    }
    VerifierExploration {
        reachable,
        terminal,
    }
}

#[test]
fn setmaxnreg_verifier_preserves_intermediate_decrease_increase_states() {
    let increase = verifier_resource(1, 0);
    let decrease = verifier_resource(0, 0);
    let initial = SetmaxnregVerifierCore::from_parts(7, 0, 8, [168, 168, 168]).unwrap();

    let mut increase_first = initial.clone();
    let pending = increase_first
        .apply_request(increase, SetmaxnregAction::Increase, 256)
        .unwrap();
    assert_eq!(pending.resource(), increase);
    assert_eq!(pending.current_count_before(), 168);
    assert_eq!(pending.available_count_before(), 8);
    assert_eq!(
        pending.disposition(),
        SetmaxnregVerifierRequestDisposition::IncreasePending { required_count: 88 }
    );
    assert_eq!(increase_first.current_count(1), Some(168));
    assert_eq!(
        increase_first.grant_status(),
        SetmaxnregVerifierGrantStatus::Stalled
    );
    assert!(increase_first.is_grant_terminal());

    increase_first
        .apply_request(decrease, SetmaxnregAction::Decrease, 80)
        .unwrap();
    assert_eq!(increase_first.current_count(1), Some(168));
    assert_eq!(increase_first.available_count(), 96);
    assert_eq!(increase_first.enabled_grants(), vec![increase]);
    assert_eq!(
        increase_first.grant_status(),
        SetmaxnregVerifierGrantStatus::Grantable
    );

    let mut decrease_first = initial;
    let released = decrease_first
        .apply_request(decrease, SetmaxnregAction::Decrease, 80)
        .unwrap();
    assert_eq!(
        released.disposition(),
        SetmaxnregVerifierRequestDisposition::DecreaseApplied { released_count: 88 }
    );
    let immediate = decrease_first
        .apply_request(increase, SetmaxnregAction::Increase, 256)
        .unwrap();
    assert_eq!(
        immediate.disposition(),
        SetmaxnregVerifierRequestDisposition::IncreaseImmediate { required_count: 88 }
    );
    assert_ne!(increase_first, decrease_first);

    let grant = increase_first.apply_grant(increase).unwrap();
    assert_eq!(grant.resource(), increase);
    assert_eq!(grant.target_count(), 256);
    assert_eq!(grant.required_count(), 88);
    assert_eq!(grant.available_count_before(), 96);
    assert_eq!(increase_first, decrease_first);
    assert!(increase_first.is_quiescent());
    assert!(increase_first.is_grant_terminal());
}

#[test]
fn setmaxnreg_verifier_one_release_grants_only_one_pending_increase() {
    let first = verifier_resource(0, 0);
    let second = verifier_resource(1, 0);
    let release = verifier_resource(2, 0);
    let mut core = SetmaxnregVerifierCore::from_parts(7, 0, 8, [168, 168, 168]).unwrap();
    core.apply_request(first, SetmaxnregAction::Increase, 256)
        .unwrap();
    core.apply_request(second, SetmaxnregAction::Increase, 256)
        .unwrap();
    assert!(matches!(
        core.apply_grant(first),
        Err(SetmaxnregVerifierError::GrantNotEnabled { .. })
    ));
    core.apply_request(release, SetmaxnregAction::Decrease, 80)
        .unwrap();
    assert_eq!(core.enabled_grants(), vec![first, second]);

    let mut grant_first = core.clone();
    grant_first.apply_grant(first).unwrap();
    assert_eq!(grant_first.available_count(), 8);
    assert_eq!(grant_first.current_counts(), &[256, 168, 80]);
    assert_eq!(grant_first.enabled_grants(), Vec::new());
    assert_eq!(
        grant_first.grant_status(),
        SetmaxnregVerifierGrantStatus::Stalled
    );
    assert_eq!(
        grant_first
            .pending_increases()
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        vec![second]
    );

    let mut grant_second = core;
    grant_second.apply_grant(second).unwrap();
    assert_eq!(grant_second.available_count(), 8);
    assert_eq!(grant_second.current_counts(), &[168, 256, 80]);
    assert_eq!(
        grant_second
            .pending_increases()
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        vec![first]
    );
    assert_ne!(grant_first, grant_second);
}

#[test]
fn setmaxnreg_verifier_exhaustive_bfs_matches_independent_reference() {
    let intermediate_requests = [
        VerifierRequest {
            resource: verifier_resource(0, 0),
            action: SetmaxnregAction::Decrease,
            target_count: 80,
        },
        VerifierRequest {
            resource: verifier_resource(1, 0),
            action: SetmaxnregAction::Increase,
            target_count: 256,
        },
    ];
    let actual = explore_core(
        SetmaxnregVerifierCore::from_parts(7, 0, 8, [168, 168, 168]).unwrap(),
        &intermediate_requests,
    );
    let expected = explore_reference(
        ReferenceRegisterPool::new(8, [168, 168, 168]),
        &intermediate_requests,
    );
    assert_eq!(actual, expected);
    assert_eq!(actual.terminal.len(), 1);
    assert!(actual
        .terminal
        .iter()
        .all(|state| state.grant_status == SetmaxnregVerifierGrantStatus::Quiescent));

    let contending_requests = [
        VerifierRequest {
            resource: verifier_resource(0, 0),
            action: SetmaxnregAction::Increase,
            target_count: 256,
        },
        VerifierRequest {
            resource: verifier_resource(1, 0),
            action: SetmaxnregAction::Increase,
            target_count: 256,
        },
        VerifierRequest {
            resource: verifier_resource(2, 0),
            action: SetmaxnregAction::Decrease,
            target_count: 80,
        },
    ];
    let actual = explore_core(
        SetmaxnregVerifierCore::from_parts(7, 0, 8, [168, 168, 168]).unwrap(),
        &contending_requests,
    );
    let expected = explore_reference(
        ReferenceRegisterPool::new(8, [168, 168, 168]),
        &contending_requests,
    );
    assert_eq!(actual, expected);
    assert_eq!(actual.terminal.len(), 2);
    assert!(actual
        .terminal
        .iter()
        .all(|state| state.grant_status == SetmaxnregVerifierGrantStatus::Stalled));
}
