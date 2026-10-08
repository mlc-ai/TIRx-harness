//! Synccheck-owned partial-order state exploration primitives.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;

/// A finite synchronization transition system used by the fixed-trace verifier.
///
/// Implementations own the semantic state, including actor cursors, blocked
/// actors, protocol objects, and pending semantic completions. The explorer
/// never executes the numerical kernel body.
pub trait SyncTransitionSystem {
    type State: Clone + Eq + Hash;
    type Transition: Clone + Ord;
    type Error: Clone + Eq + Ord;
    type Deadlock: Clone + Eq + Ord;

    fn initial_state(&self) -> Self::State;

    /// Return every currently enabled fixed event or semantic completion.
    fn enabled_transitions(&self, state: &Self::State) -> Vec<Self::Transition>;

    /// Apply exactly one enabled transition.
    fn step(
        &self,
        state: &Self::State,
        transition: &Self::Transition,
    ) -> Result<Self::State, Self::Error>;

    /// True only when every fixed event has completed and every protocol is
    /// quiescent. An unfinished state with no enabled transition is deadlocked.
    fn is_complete(&self, state: &Self::State) -> bool;

    fn describe_deadlock(&self, state: &Self::State) -> Self::Deadlock;

    /// An enabled transition that may be moved to the front of every complete
    /// execution without losing errors, deadlocks or distinct terminal states.
    /// This requires a proof over future dependencies, not just a state-local
    /// commuting diamond. The default retains ordinary exhaustive exploration.
    fn persistent_transition(
        &self,
        _state: &Self::State,
        _enabled: &[Self::Transition],
    ) -> Option<Self::Transition> {
        None
    }

    /// A strong diamond guarantees that either transition can execute first,
    /// neither first step changes checker-visible successor exposure, and the
    /// two orders reach the same semantic state. Returning true is a proof
    /// obligation of the concrete protocol model.
    fn strong_diamond(
        &self,
        _state: &Self::State,
        _left: &Self::Transition,
        _right: &Self::Transition,
    ) -> bool {
        false
    }

    /// True when the two transitions form a state-local commuting diamond.
    ///
    /// Unlike [`Self::strong_diamond`], this proof may allow either first
    /// transition to expose additional successors.  A sleep-set reduction
    /// rechecks those newly exposed successors before keeping the other
    /// transition asleep, so exact successor-set equality is unnecessary.
    fn commutes(
        &self,
        state: &Self::State,
        left: &Self::Transition,
        right: &Self::Transition,
    ) -> bool {
        self.strong_diamond(state, left, right)
    }

    /// True when two already-computed one-step successors commute.
    ///
    /// The sleep-set explorer computes every enabled one-step successor once
    /// per state and passes those values here. Implementations that can reuse
    /// them should override this method. The default preserves the semantics
    /// of models that implement only [`Self::commutes`].
    fn successors_commute(
        &self,
        state: &Self::State,
        left: &Self::Transition,
        _after_left: &Self::State,
        right: &Self::Transition,
        _after_right: &Self::State,
    ) -> bool {
        self.commutes(state, left, right)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyncStateSearchLimits {
    pub max_states: usize,
    pub max_transitions: usize,
}

impl Default for SyncStateSearchLimits {
    fn default() -> Self {
        Self {
            max_states: 1_000_000,
            max_transitions: 10_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SyncStateSearchOptions {
    pub stop_on_first_failure: bool,
    pub reduce_all_strong_diamonds: bool,
    pub reduce_sleep_sets: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SyncStateSearchTermination {
    Exhausted,
    FirstFailure,
    StateLimit { limit: usize },
    TransitionLimit { limit: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SyncStateFailure<T, E, D> {
    Error {
        transition: T,
        error: E,
        witness: Box<[T]>,
    },
    Deadlock {
        deadlock: D,
        witness: Box<[T]>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyncStateSearchResult<T, E, D> {
    visited_states: usize,
    explored_transitions: usize,
    strong_diamond_pruned_transitions: usize,
    complete_states: usize,
    complete_witnesses: Box<[Box<[T]>]>,
    failures: Box<[SyncStateFailure<T, E, D>]>,
    termination: SyncStateSearchTermination,
}

impl<T, E, D> SyncStateSearchResult<T, E, D> {
    pub const fn visited_states(&self) -> usize {
        self.visited_states
    }

    pub const fn explored_transitions(&self) -> usize {
        self.explored_transitions
    }

    pub const fn strong_diamond_pruned_transitions(&self) -> usize {
        self.strong_diamond_pruned_transitions
    }

    pub const fn complete_states(&self) -> usize {
        self.complete_states
    }

    /// Up to two distinct complete-state schedules. Two are retained so a
    /// non-confluence report has a compact, replayable pair of witnesses.
    pub fn complete_witnesses(&self) -> &[Box<[T]>] {
        &self.complete_witnesses
    }

    /// Whether every explored complete execution normalized to one semantic state.
    ///
    /// Concrete models must exclude witness history and scheduler bookkeeping
    /// from `State` equality. Two distinct complete states therefore mean that
    /// an HB-compatible transition order changed checker-visible protocol state.
    pub const fn complete_states_are_confluent(&self) -> bool {
        self.complete_states == 1
    }

    pub fn failures(&self) -> &[SyncStateFailure<T, E, D>] {
        &self.failures
    }

    pub const fn termination(&self) -> &SyncStateSearchTermination {
        &self.termination
    }

    pub fn proves_clean(&self) -> bool {
        self.termination == SyncStateSearchTermination::Exhausted
            && self.failures.is_empty()
            && self.complete_states_are_confluent()
    }
}

type StateId = usize;

#[derive(Debug)]
struct ParentTransition<T> {
    state_id: StateId,
    transition: T,
}

#[derive(Debug)]
struct SearchNode<S, T> {
    state: Arc<S>,
    parent: Option<ParentTransition<T>>,
}

#[derive(Clone, Copy, Debug)]
struct SearchFrame {
    state_id: StateId,
}

fn reconstruct_witness<S, T: Clone>(
    nodes: &[SearchNode<S, T>],
    mut state_id: StateId,
    terminal_transition: Option<&T>,
) -> Box<[T]> {
    let mut witness = Vec::new();
    while let Some(parent) = &nodes[state_id].parent {
        witness.push(parent.transition.clone());
        state_id = parent.state_id;
    }
    witness.reverse();
    if let Some(transition) = terminal_transition {
        witness.push(transition.clone());
    }
    witness.into_boxed_slice()
}

/// Exhaustively explore the finite protocol state space with semantic-state
/// memoization. The numerical kernel is executed only once to obtain an
/// eligible fixed event program; this search contains no native replay loop.
pub fn explore_sync_states<M>(
    model: &M,
    limits: SyncStateSearchLimits,
    options: SyncStateSearchOptions,
) -> SyncStateSearchResult<M::Transition, M::Error, M::Deadlock>
where
    M: SyncTransitionSystem,
{
    if options.reduce_sleep_sets {
        return explore_sync_states_with_sleep_sets(model, limits, options);
    }

    // The visited index and node arena share each semantic-state allocation.
    let initial = Arc::new(model.initial_state());
    let mut visited = HashMap::from([(Arc::clone(&initial), 0)]);
    let mut nodes = vec![SearchNode {
        state: initial,
        parent: None,
    }];
    let mut stack = vec![SearchFrame { state_id: 0 }];
    let mut explored_transitions = 0_usize;
    let mut strong_diamond_pruned_transitions = 0_usize;
    let mut complete_states = 0_usize;
    let mut complete_witnesses = Vec::new();
    let mut failures = Vec::new();
    let mut failure_keys = BTreeSet::new();
    let mut termination = SyncStateSearchTermination::Exhausted;

    'search: while let Some(frame) = stack.pop() {
        let state_id = frame.state_id;
        if model.is_complete(&nodes[state_id].state) {
            complete_states = complete_states.saturating_add(1);
            if complete_witnesses.len() < 2 {
                complete_witnesses.push(reconstruct_witness(&nodes, state_id, None));
            }
            continue;
        }

        let mut enabled = model.enabled_transitions(&nodes[state_id].state);
        enabled.sort();
        enabled.dedup();
        if enabled.is_empty() {
            let deadlock = model.describe_deadlock(&nodes[state_id].state);
            if failure_keys.insert((None, Some(deadlock.clone()))) {
                failures.push(SyncStateFailure::Deadlock {
                    deadlock,
                    witness: reconstruct_witness(&nodes, state_id, None),
                });
            }
            if options.stop_on_first_failure {
                termination = SyncStateSearchTermination::FirstFailure;
                break;
            }
            continue;
        }

        if options.reduce_all_strong_diamonds
            && enabled.len() > 1
            && enabled.iter().enumerate().all(|(left_index, left)| {
                enabled
                    .iter()
                    .skip(left_index + 1)
                    .all(|right| model.strong_diamond(&nodes[state_id].state, left, right))
            })
        {
            strong_diamond_pruned_transitions =
                strong_diamond_pruned_transitions.saturating_add(enabled.len().saturating_sub(1));
            enabled.truncate(1);
        }

        for transition in enabled.into_iter().rev() {
            if explored_transitions >= limits.max_transitions {
                termination = SyncStateSearchTermination::TransitionLimit {
                    limit: limits.max_transitions,
                };
                break 'search;
            }
            explored_transitions = explored_transitions.saturating_add(1);
            match model.step(&nodes[state_id].state, &transition) {
                Ok(next) => {
                    if visited.contains_key(&next) {
                        continue;
                    }
                    if visited.len() >= limits.max_states {
                        termination = SyncStateSearchTermination::StateLimit {
                            limit: limits.max_states,
                        };
                        break 'search;
                    }
                    let next_state_id = nodes.len();
                    let next = Arc::new(next);
                    visited.insert(Arc::clone(&next), next_state_id);
                    nodes.push(SearchNode {
                        state: next,
                        parent: Some(ParentTransition {
                            state_id,
                            transition,
                        }),
                    });
                    stack.push(SearchFrame {
                        state_id: next_state_id,
                    });
                }
                Err(error) => {
                    if failure_keys.insert((Some(error.clone()), None)) {
                        let witness = reconstruct_witness(&nodes, state_id, Some(&transition));
                        failures.push(SyncStateFailure::Error {
                            transition,
                            error,
                            witness,
                        });
                    }
                    if options.stop_on_first_failure {
                        termination = SyncStateSearchTermination::FirstFailure;
                        break 'search;
                    }
                }
            }
        }
    }

    SyncStateSearchResult {
        visited_states: visited.len(),
        explored_transitions,
        strong_diamond_pruned_transitions,
        complete_states,
        complete_witnesses: complete_witnesses.into_boxed_slice(),
        failures: failures.into_boxed_slice(),
        termination,
    }
}

#[derive(Debug)]
struct SleepSearchNode<S, T> {
    state: Arc<S>,
    sleep: BTreeSet<T>,
    parent: Option<ParentTransition<T>>,
}

fn reconstruct_sleep_witness<S, T: Clone>(
    nodes: &[SleepSearchNode<S, T>],
    mut state_id: StateId,
    terminal_transition: Option<&T>,
) -> Box<[T]> {
    let mut witness = Vec::new();
    while let Some(parent) = &nodes[state_id].parent {
        witness.push(parent.transition.clone());
        state_id = parent.state_id;
    }
    witness.reverse();
    if let Some(transition) = terminal_transition {
        witness.push(transition.clone());
    }
    witness.into_boxed_slice()
}

/// Register one semantic-state/sleep-set context.
///
/// A smaller sleep set dominates a larger one at the same semantic state: it
/// permits every continuation available to the larger set and possibly more.
/// Keeping only the subset-minimal antichain avoids reintroducing the global
/// Cartesian product through equivalent commuting prefixes.
fn register_sleep_context<S, T>(
    visited: &mut HashMap<Arc<S>, Vec<BTreeSet<T>>>,
    state: Arc<S>,
    sleep: &BTreeSet<T>,
) -> bool
where
    S: Eq + Hash,
    T: Clone + Ord,
{
    match visited.get_mut(&state) {
        Some(contexts) => {
            if contexts.iter().any(|current| current.is_subset(sleep)) {
                return false;
            }
            contexts.retain(|current| !sleep.is_subset(current));
            contexts.push(sleep.clone());
        }
        None => {
            visited.insert(state, vec![sleep.clone()]);
        }
    }
    true
}

fn sleep_context_is_live<S, T>(
    visited: &HashMap<Arc<S>, Vec<BTreeSet<T>>>,
    state: &Arc<S>,
    sleep: &BTreeSet<T>,
) -> bool
where
    S: Eq + Hash,
    T: Ord,
{
    visited
        .get(state)
        .is_some_and(|contexts| contexts.iter().any(|current| current == sleep))
}

/// Explore one finite transition system with state-local sleep-set POR.
///
/// The numerical kernel is still executed only once.  This reduction removes
/// permutations of commuting protocol transitions while retaining a sleeping
/// transition as soon as a newly exposed dependent transition can distinguish
/// its order.  Semantic-state memoization is indexed by a subset-minimal
/// antichain of sleep sets so a restrictive visit cannot hide a later, less
/// restrictive continuation.
fn explore_sync_states_with_sleep_sets<M>(
    model: &M,
    limits: SyncStateSearchLimits,
    options: SyncStateSearchOptions,
) -> SyncStateSearchResult<M::Transition, M::Error, M::Deadlock>
where
    M: SyncTransitionSystem,
{
    let initial = Arc::new(model.initial_state());
    let initial_sleep = BTreeSet::new();
    let mut visited = HashMap::from([(Arc::clone(&initial), vec![initial_sleep.clone()])]);
    let mut nodes = vec![SleepSearchNode {
        state: initial,
        sleep: initial_sleep,
        parent: None,
    }];
    let mut stack = vec![SearchFrame { state_id: 0 }];
    let mut explored_transitions = 0_usize;
    let mut strong_diamond_pruned_transitions = 0_usize;
    let mut complete_states = HashSet::<Arc<M::State>>::new();
    let mut complete_witnesses = Vec::new();
    let mut failures = Vec::new();
    let mut failure_keys = BTreeSet::new();
    let mut termination = SyncStateSearchTermination::Exhausted;

    'search: while let Some(frame) = stack.pop() {
        let state_id = frame.state_id;
        if !sleep_context_is_live(&visited, &nodes[state_id].state, &nodes[state_id].sleep) {
            continue;
        }
        if model.is_complete(&nodes[state_id].state) {
            if complete_states.insert(Arc::clone(&nodes[state_id].state))
                && complete_witnesses.len() < 2
            {
                complete_witnesses.push(reconstruct_sleep_witness(&nodes, state_id, None));
            }
            continue;
        }

        let mut enabled = model.enabled_transitions(&nodes[state_id].state);
        enabled.sort();
        enabled.dedup();
        if enabled.is_empty() {
            let deadlock = model.describe_deadlock(&nodes[state_id].state);
            if failure_keys.insert((None, Some(deadlock.clone()))) {
                failures.push(SyncStateFailure::Deadlock {
                    deadlock,
                    witness: reconstruct_sleep_witness(&nodes, state_id, None),
                });
            }
            if options.stop_on_first_failure {
                termination = SyncStateSearchTermination::FirstFailure;
                break;
            }
            continue;
        }

        let mut branch_sleep = nodes[state_id].sleep.clone();
        branch_sleep.retain(|transition| enabled.binary_search(transition).is_ok());
        let active_transition_count = enabled
            .iter()
            .filter(|transition| !branch_sleep.contains(*transition))
            .count();
        // Do not combine a new persistent-set proof with inherited sleeping
        // prefixes. Their existing state-local commutation rules remain intact.
        let persistent = (options.reduce_all_strong_diamonds && branch_sleep.is_empty())
            .then(|| model.persistent_transition(&nodes[state_id].state, &enabled))
            .flatten()
            .map(|transition| {
                enabled
                    .binary_search(&transition)
                    .expect("persistent transition must be enabled")
            });
        let strong_diamond_canonical = (options.reduce_all_strong_diamonds
            && persistent.is_none()
            && active_transition_count > 1
            && enabled.iter().enumerate().all(|(left_index, left)| {
                enabled
                    .iter()
                    .skip(left_index + 1)
                    .all(|right| model.strong_diamond(&nodes[state_id].state, left, right))
            }))
        .then(|| {
            enabled
                .iter()
                .position(|transition| !branch_sleep.contains(transition))
                .expect("more than one active transition has a first transition")
        });
        if strong_diamond_canonical.is_some() {
            strong_diamond_pruned_transitions = strong_diamond_pruned_transitions
                .saturating_add(active_transition_count.saturating_sub(1));
        }
        // Cache each pure one-step successor once for this state. Pairwise
        // commutation checks can then reuse both first steps instead of
        // rebuilding them for every pair.
        let successors = enabled
            .iter()
            .map(|transition| model.step(&nodes[state_id].state, transition).map(Arc::new))
            .collect::<Vec<_>>();
        let mut children = Vec::new();
        for (transition_index, transition) in enabled.iter().enumerate() {
            if persistent.is_some_and(|index| index != transition_index) {
                continue;
            }
            if strong_diamond_canonical
                .is_some_and(|canonical_index| transition_index != canonical_index)
            {
                continue;
            }
            if branch_sleep.contains(transition) {
                continue;
            }
            if explored_transitions >= limits.max_transitions {
                termination = SyncStateSearchTermination::TransitionLimit {
                    limit: limits.max_transitions,
                };
                break 'search;
            }
            explored_transitions = explored_transitions.saturating_add(1);
            match &successors[transition_index] {
                Ok(next) => {
                    let next_enabled = model
                        .enabled_transitions(next)
                        .into_iter()
                        .collect::<BTreeSet<_>>();
                    let child_sleep = branch_sleep
                        .iter()
                        .filter(|sleeping| {
                            let Ok(sleeping_index) = enabled.binary_search(sleeping) else {
                                return false;
                            };
                            let Ok(after_sleeping) = &successors[sleeping_index] else {
                                return false;
                            };
                            next_enabled.contains(*sleeping)
                                && model.successors_commute(
                                    &nodes[state_id].state,
                                    transition,
                                    next,
                                    sleeping,
                                    after_sleeping,
                                )
                        })
                        .cloned()
                        .collect::<BTreeSet<_>>();
                    let is_new_semantic_state = !visited.contains_key(next);
                    if is_new_semantic_state && visited.len() >= limits.max_states {
                        termination = SyncStateSearchTermination::StateLimit {
                            limit: limits.max_states,
                        };
                        break 'search;
                    }
                    if register_sleep_context(&mut visited, Arc::clone(next), &child_sleep) {
                        let next_state_id = nodes.len();
                        nodes.push(SleepSearchNode {
                            state: Arc::clone(next),
                            sleep: child_sleep,
                            parent: Some(ParentTransition {
                                state_id,
                                transition: transition.clone(),
                            }),
                        });
                        children.push(next_state_id);
                    }
                }
                Err(error) => {
                    if failure_keys.insert((Some(error.clone()), None)) {
                        let witness = reconstruct_sleep_witness(&nodes, state_id, Some(transition));
                        failures.push(SyncStateFailure::Error {
                            transition: transition.clone(),
                            error: error.clone(),
                            witness,
                        });
                    }
                    if options.stop_on_first_failure {
                        termination = SyncStateSearchTermination::FirstFailure;
                        break 'search;
                    }
                }
            }
            if strong_diamond_canonical.is_none() {
                branch_sleep.insert(transition.clone());
            }
        }
        for child in children.into_iter().rev() {
            stack.push(SearchFrame { state_id: child });
        }
    }

    SyncStateSearchResult {
        visited_states: visited.len(),
        explored_transitions,
        strong_diamond_pruned_transitions,
        complete_states: complete_states.len(),
        complete_witnesses: complete_witnesses.into_boxed_slice(),
        failures: failures.into_boxed_slice(),
        termination,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct CounterState {
        cursors: [u8; 2],
        value: i8,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum CounterTransition {
        Increment,
        Decrement,
    }

    #[derive(Clone, Copy, Debug)]
    struct CounterModel;

    impl SyncTransitionSystem for CounterModel {
        type State = CounterState;
        type Transition = CounterTransition;
        type Error = &'static str;
        type Deadlock = &'static str;

        fn initial_state(&self) -> Self::State {
            CounterState {
                cursors: [0, 0],
                value: 0,
            }
        }

        fn enabled_transitions(&self, state: &Self::State) -> Vec<Self::Transition> {
            let mut enabled = Vec::new();
            if state.cursors[0] == 0 {
                enabled.push(CounterTransition::Increment);
            }
            if state.cursors[1] == 0 {
                enabled.push(CounterTransition::Decrement);
            }
            enabled
        }

        fn step(
            &self,
            state: &Self::State,
            transition: &Self::Transition,
        ) -> Result<Self::State, Self::Error> {
            let mut next = state.clone();
            match transition {
                CounterTransition::Increment if next.cursors[0] == 0 => {
                    next.cursors[0] = 1;
                    next.value += 1;
                }
                CounterTransition::Decrement if next.cursors[1] == 0 => {
                    next.cursors[1] = 1;
                    next.value -= 1;
                }
                _ => return Err("disabled transition"),
            }
            Ok(next)
        }

        fn is_complete(&self, state: &Self::State) -> bool {
            state.cursors == [1, 1]
        }

        fn describe_deadlock(&self, _state: &Self::State) -> Self::Deadlock {
            "counter deadlock"
        }

        fn strong_diamond(
            &self,
            _state: &Self::State,
            _left: &Self::Transition,
            _right: &Self::Transition,
        ) -> bool {
            true
        }
    }

    #[test]
    fn exhaustive_search_memoizes_a_commuting_diamond() {
        let result = explore_sync_states(
            &CounterModel,
            SyncStateSearchLimits::default(),
            SyncStateSearchOptions::default(),
        );
        assert!(result.proves_clean());
        assert_eq!(result.visited_states(), 4);
        assert_eq!(result.complete_states(), 1);
        assert_eq!(result.explored_transitions(), 4);
    }

    #[test]
    fn optional_strong_diamond_reduction_chooses_one_canonical_branch() {
        let result = explore_sync_states(
            &CounterModel,
            SyncStateSearchLimits::default(),
            SyncStateSearchOptions {
                reduce_all_strong_diamonds: true,
                ..SyncStateSearchOptions::default()
            },
        );
        assert!(result.proves_clean());
        assert_eq!(result.visited_states(), 3);
        assert_eq!(result.strong_diamond_pruned_transitions(), 1);
    }

    #[test]
    fn optional_strong_diamond_reduction_composes_with_sleep_sets() {
        let result = explore_sync_states(
            &CounterModel,
            SyncStateSearchLimits::default(),
            SyncStateSearchOptions {
                reduce_all_strong_diamonds: true,
                reduce_sleep_sets: true,
                ..SyncStateSearchOptions::default()
            },
        );
        assert!(result.proves_clean());
        assert_eq!(result.visited_states(), 3);
        assert_eq!(result.explored_transitions(), 2);
        assert_eq!(result.strong_diamond_pruned_transitions(), 1);
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    struct InheritedSleepState {
        independent: bool,
        revealed: bool,
        left: bool,
        right: bool,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum InheritedSleepTransition {
        Independent,
        RevealPair,
        Left,
        Right,
    }

    struct InheritedSleepModel;

    impl SyncTransitionSystem for InheritedSleepModel {
        type State = InheritedSleepState;
        type Transition = InheritedSleepTransition;
        type Error = &'static str;
        type Deadlock = &'static str;

        fn initial_state(&self) -> Self::State {
            InheritedSleepState::default()
        }

        fn enabled_transitions(&self, state: &Self::State) -> Vec<Self::Transition> {
            let mut enabled = Vec::new();
            if !state.independent {
                enabled.push(InheritedSleepTransition::Independent);
            }
            if !state.revealed {
                enabled.push(InheritedSleepTransition::RevealPair);
            }
            if state.revealed && !state.left {
                enabled.push(InheritedSleepTransition::Left);
            }
            if state.revealed && !state.right {
                enabled.push(InheritedSleepTransition::Right);
            }
            enabled
        }

        fn step(
            &self,
            state: &Self::State,
            transition: &Self::Transition,
        ) -> Result<Self::State, Self::Error> {
            if !self.enabled_transitions(state).contains(transition) {
                return Err("disabled transition");
            }
            let mut next = *state;
            match transition {
                InheritedSleepTransition::Independent => next.independent = true,
                InheritedSleepTransition::RevealPair => next.revealed = true,
                InheritedSleepTransition::Left => next.left = true,
                InheritedSleepTransition::Right => next.right = true,
            }
            Ok(next)
        }

        fn is_complete(&self, state: &Self::State) -> bool {
            state.independent && state.revealed && state.left && state.right
        }

        fn describe_deadlock(&self, _state: &Self::State) -> Self::Deadlock {
            "inherited-sleep deadlock"
        }

        fn strong_diamond(
            &self,
            state: &Self::State,
            left: &Self::Transition,
            right: &Self::Transition,
        ) -> bool {
            state.revealed && left != right
        }

        fn commutes(
            &self,
            _state: &Self::State,
            left: &Self::Transition,
            right: &Self::Transition,
        ) -> bool {
            left != right
        }
    }

    #[test]
    fn strong_diamond_counts_only_active_transitions_in_inherited_sleep_context() {
        let sleep_only = explore_sync_states(
            &InheritedSleepModel,
            SyncStateSearchLimits::default(),
            SyncStateSearchOptions {
                reduce_sleep_sets: true,
                ..SyncStateSearchOptions::default()
            },
        );
        assert!(sleep_only.proves_clean());
        assert_eq!(sleep_only.strong_diamond_pruned_transitions(), 0);

        let composed = explore_sync_states(
            &InheritedSleepModel,
            SyncStateSearchLimits::default(),
            SyncStateSearchOptions {
                reduce_all_strong_diamonds: true,
                reduce_sleep_sets: true,
                ..SyncStateSearchOptions::default()
            },
        );
        assert!(composed.proves_clean());
        assert_eq!(composed.strong_diamond_pruned_transitions(), 2);
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum TerminalState {
        Start,
        Complete,
        Deadlocked,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum TerminalTransition {
        Complete,
        Deadlock,
    }

    struct TerminalModel;

    impl SyncTransitionSystem for TerminalModel {
        type State = TerminalState;
        type Transition = TerminalTransition;
        type Error = &'static str;
        type Deadlock = &'static str;

        fn initial_state(&self) -> Self::State {
            TerminalState::Start
        }

        fn enabled_transitions(&self, state: &Self::State) -> Vec<Self::Transition> {
            match state {
                TerminalState::Start => {
                    vec![TerminalTransition::Complete, TerminalTransition::Deadlock]
                }
                TerminalState::Complete | TerminalState::Deadlocked => Vec::new(),
            }
        }

        fn step(
            &self,
            _state: &Self::State,
            transition: &Self::Transition,
        ) -> Result<Self::State, Self::Error> {
            Ok(match transition {
                TerminalTransition::Complete => TerminalState::Complete,
                TerminalTransition::Deadlock => TerminalState::Deadlocked,
            })
        }

        fn is_complete(&self, state: &Self::State) -> bool {
            *state == TerminalState::Complete
        }

        fn describe_deadlock(&self, _state: &Self::State) -> Self::Deadlock {
            "reachable deadlock"
        }
    }

    #[test]
    fn one_reachable_deadlock_prevents_a_clean_proof() {
        let result = explore_sync_states(
            &TerminalModel,
            SyncStateSearchLimits::default(),
            SyncStateSearchOptions::default(),
        );
        assert!(!result.proves_clean());
        assert_eq!(result.complete_states(), 1);
        assert_eq!(result.failures().len(), 1);
        assert_eq!(
            result.failures()[0],
            SyncStateFailure::Deadlock {
                deadlock: "reachable deadlock",
                witness: Box::new([TerminalTransition::Deadlock]),
            }
        );
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum LongDiamondTransitionKind {
        BranchLeft(usize),
        BranchRight(usize),
        MergeLeft(usize),
        MergeRight(usize),
        Fail,
        Deadlock,
    }

    #[derive(Debug)]
    struct CloneCountingTransition {
        kind: LongDiamondTransitionKind,
        clone_count: Arc<AtomicUsize>,
    }

    impl Clone for CloneCountingTransition {
        fn clone(&self) -> Self {
            self.clone_count.fetch_add(1, Ordering::Relaxed);
            Self {
                kind: self.kind,
                clone_count: Arc::clone(&self.clone_count),
            }
        }
    }

    impl PartialEq for CloneCountingTransition {
        fn eq(&self, other: &Self) -> bool {
            self.kind == other.kind
        }
    }

    impl Eq for CloneCountingTransition {}

    impl PartialOrd for CloneCountingTransition {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for CloneCountingTransition {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.kind.cmp(&other.kind)
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum LongDiamondState {
        Stage(usize),
        LeftArm(usize),
        RightArm(usize),
        Deadlocked,
    }

    struct LongDiamondModel {
        depth: usize,
        clone_count: Arc<AtomicUsize>,
    }

    impl LongDiamondModel {
        fn transition(&self, kind: LongDiamondTransitionKind) -> CloneCountingTransition {
            CloneCountingTransition {
                kind,
                clone_count: Arc::clone(&self.clone_count),
            }
        }
    }

    impl SyncTransitionSystem for LongDiamondModel {
        type State = LongDiamondState;
        type Transition = CloneCountingTransition;
        type Error = &'static str;
        type Deadlock = &'static str;

        fn initial_state(&self) -> Self::State {
            LongDiamondState::Stage(0)
        }

        fn enabled_transitions(&self, state: &Self::State) -> Vec<Self::Transition> {
            match *state {
                LongDiamondState::Stage(stage) if stage < self.depth => vec![
                    self.transition(LongDiamondTransitionKind::BranchLeft(stage)),
                    self.transition(LongDiamondTransitionKind::BranchRight(stage)),
                ],
                LongDiamondState::Stage(_) => vec![
                    self.transition(LongDiamondTransitionKind::Fail),
                    self.transition(LongDiamondTransitionKind::Deadlock),
                ],
                LongDiamondState::LeftArm(stage) => {
                    vec![self.transition(LongDiamondTransitionKind::MergeLeft(stage))]
                }
                LongDiamondState::RightArm(stage) => {
                    vec![self.transition(LongDiamondTransitionKind::MergeRight(stage))]
                }
                LongDiamondState::Deadlocked => Vec::new(),
            }
        }

        fn step(
            &self,
            state: &Self::State,
            transition: &Self::Transition,
        ) -> Result<Self::State, Self::Error> {
            match (*state, transition.kind) {
                (
                    LongDiamondState::Stage(stage),
                    LongDiamondTransitionKind::BranchLeft(transition_stage),
                ) if stage == transition_stage && stage < self.depth => {
                    Ok(LongDiamondState::LeftArm(stage))
                }
                (
                    LongDiamondState::Stage(stage),
                    LongDiamondTransitionKind::BranchRight(transition_stage),
                ) if stage == transition_stage && stage < self.depth => {
                    Ok(LongDiamondState::RightArm(stage))
                }
                (
                    LongDiamondState::LeftArm(stage),
                    LongDiamondTransitionKind::MergeLeft(transition_stage),
                ) if stage == transition_stage => Ok(LongDiamondState::Stage(stage + 1)),
                (
                    LongDiamondState::RightArm(stage),
                    LongDiamondTransitionKind::MergeRight(transition_stage),
                ) if stage == transition_stage => Ok(LongDiamondState::Stage(stage + 1)),
                (LongDiamondState::Stage(stage), LongDiamondTransitionKind::Fail)
                    if stage == self.depth =>
                {
                    Err("terminal error")
                }
                (LongDiamondState::Stage(stage), LongDiamondTransitionKind::Deadlock)
                    if stage == self.depth =>
                {
                    Ok(LongDiamondState::Deadlocked)
                }
                _ => Err("disabled transition"),
            }
        }

        fn is_complete(&self, _state: &Self::State) -> bool {
            false
        }

        fn describe_deadlock(&self, state: &Self::State) -> Self::Deadlock {
            assert_eq!(*state, LongDiamondState::Deadlocked);
            "terminal deadlock"
        }
    }

    fn assert_long_diamond_prefix(witness: &[CloneCountingTransition], depth: usize) {
        assert_eq!(witness.len(), depth * 2 + 1);
        for stage in 0..depth {
            assert_eq!(
                witness[stage * 2].kind,
                LongDiamondTransitionKind::BranchLeft(stage)
            );
            assert_eq!(
                witness[stage * 2 + 1].kind,
                LongDiamondTransitionKind::MergeLeft(stage)
            );
        }
    }

    #[test]
    fn long_diamond_reconstructs_failure_witnesses_without_successor_path_clones() {
        let depth = 128;
        let clone_count = Arc::new(AtomicUsize::new(0));
        let result = explore_sync_states(
            &LongDiamondModel {
                depth,
                clone_count: Arc::clone(&clone_count),
            },
            SyncStateSearchLimits::default(),
            SyncStateSearchOptions::default(),
        );

        assert_eq!(result.visited_states(), depth * 3 + 2);
        assert_eq!(result.explored_transitions(), depth * 4 + 2);
        assert_eq!(result.failures().len(), 2);

        let mut witnessed_transition_count = 0;
        for failure in result.failures() {
            match failure {
                SyncStateFailure::Error {
                    transition,
                    error,
                    witness,
                } => {
                    assert_eq!(transition.kind, LongDiamondTransitionKind::Fail);
                    assert_eq!(*error, "terminal error");
                    assert_long_diamond_prefix(witness, depth);
                    assert_eq!(
                        witness.last().map(|item| item.kind),
                        Some(LongDiamondTransitionKind::Fail)
                    );
                    witnessed_transition_count += witness.len();
                }
                SyncStateFailure::Deadlock { deadlock, witness } => {
                    assert_eq!(*deadlock, "terminal deadlock");
                    assert_long_diamond_prefix(witness, depth);
                    assert_eq!(
                        witness.last().map(|item| item.kind),
                        Some(LongDiamondTransitionKind::Deadlock)
                    );
                    witnessed_transition_count += witness.len();
                }
            }
        }

        assert_eq!(
            clone_count.load(Ordering::Relaxed),
            witnessed_transition_count,
            "transitions should be cloned only while materializing failure witnesses"
        );
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum NonconfluentState {
        Start,
        LeftComplete,
        RightComplete,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum NonconfluentTransition {
        Left,
        Right,
    }

    struct NonconfluentModel;

    impl SyncTransitionSystem for NonconfluentModel {
        type State = NonconfluentState;
        type Transition = NonconfluentTransition;
        type Error = &'static str;
        type Deadlock = &'static str;

        fn initial_state(&self) -> Self::State {
            NonconfluentState::Start
        }

        fn enabled_transitions(&self, state: &Self::State) -> Vec<Self::Transition> {
            match state {
                NonconfluentState::Start => {
                    vec![NonconfluentTransition::Left, NonconfluentTransition::Right]
                }
                NonconfluentState::LeftComplete | NonconfluentState::RightComplete => Vec::new(),
            }
        }

        fn step(
            &self,
            state: &Self::State,
            transition: &Self::Transition,
        ) -> Result<Self::State, Self::Error> {
            match (state, transition) {
                (NonconfluentState::Start, NonconfluentTransition::Left) => {
                    Ok(NonconfluentState::LeftComplete)
                }
                (NonconfluentState::Start, NonconfluentTransition::Right) => {
                    Ok(NonconfluentState::RightComplete)
                }
                _ => Err("disabled transition"),
            }
        }

        fn is_complete(&self, state: &Self::State) -> bool {
            matches!(
                state,
                NonconfluentState::LeftComplete | NonconfluentState::RightComplete
            )
        }

        fn describe_deadlock(&self, _state: &Self::State) -> Self::Deadlock {
            "nonconfluent model deadlock"
        }
    }

    #[test]
    fn distinct_complete_protocol_states_prevent_a_clean_proof() {
        let result = explore_sync_states(
            &NonconfluentModel,
            SyncStateSearchLimits::default(),
            SyncStateSearchOptions::default(),
        );
        assert!(result.failures().is_empty());
        assert_eq!(result.complete_states(), 2);
        assert_eq!(result.complete_witnesses().len(), 2);
        assert_eq!(
            result.complete_witnesses()[0].as_ref(),
            &[NonconfluentTransition::Left]
        );
        assert_eq!(
            result.complete_witnesses()[1].as_ref(),
            &[NonconfluentTransition::Right]
        );
        assert!(!result.complete_states_are_confluent());
        assert!(!result.proves_clean());
    }
}
