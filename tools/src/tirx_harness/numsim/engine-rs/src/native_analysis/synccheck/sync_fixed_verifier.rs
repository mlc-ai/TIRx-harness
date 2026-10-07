//! Synccheck-owned exhaustive fixed-program verifier.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::{
    explore_sync_states, DynamicOpId, FixedSyncDeadlock, FixedSyncProgram,
    FixedSyncProgramBuildError, FixedSyncProgramError, FixedSyncTransition, ResolvedTransitionLog,
    SyncStateFailure, SyncStateSearchLimits, SyncStateSearchOptions, SyncStateSearchTermination,
};

// Protocol projections and deduplicated state-search jobs are independent.
// Keep enough parallelism for large traces while bounding host fanout.
const MAX_FIXED_SYNC_VALIDATION_WORKERS: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FixedSyncVerificationStats {
    programs: usize,
    reused_clean_programs: usize,
    visited_states: usize,
    explored_transitions: usize,
    strong_diamond_pruned_transitions: usize,
}

impl FixedSyncVerificationStats {
    pub const fn programs(self) -> usize {
        self.programs
    }

    pub const fn reused_clean_programs(self) -> usize {
        self.reused_clean_programs
    }

    pub const fn visited_states(self) -> usize {
        self.visited_states
    }

    pub const fn explored_transitions(self) -> usize {
        self.explored_transitions
    }

    pub const fn strong_diamond_pruned_transitions(self) -> usize {
        self.strong_diamond_pruned_transitions
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedSyncTransitionEvidence {
    transition: FixedSyncTransition,
    operation: Option<DynamicOpId>,
    description: Box<str>,
}

impl FixedSyncTransitionEvidence {
    pub const fn transition(&self) -> &FixedSyncTransition {
        &self.transition
    }

    pub const fn operation(&self) -> Option<&DynamicOpId> {
        self.operation.as_ref()
    }

    pub fn description(&self) -> &str {
        &self.description
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedSyncVerificationError {
    Protocol {
        operation: DynamicOpId,
        transition: FixedSyncTransition,
        source: FixedSyncProgramError,
        witness: Box<[FixedSyncTransition]>,
        witness_evidence: Box<[FixedSyncTransitionEvidence]>,
    },
    Deadlock {
        operation: Option<DynamicOpId>,
        deadlock: FixedSyncDeadlock,
        witness: Box<[FixedSyncTransition]>,
        witness_evidence: Box<[FixedSyncTransitionEvidence]>,
    },
    NonConfluent {
        operation: Option<DynamicOpId>,
        complete_states: usize,
        witnesses: Box<[Box<[FixedSyncTransition]>]>,
        witnesses_evidence: Box<[Box<[FixedSyncTransitionEvidence]>]>,
    },
}

impl FixedSyncVerificationError {
    pub const fn operation(&self) -> Option<&DynamicOpId> {
        match self {
            Self::Protocol { operation, .. } => Some(operation),
            Self::Deadlock { operation, .. } | Self::NonConfluent { operation, .. } => {
                operation.as_ref()
            }
        }
    }
}

impl fmt::Display for FixedSyncVerificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Protocol {
                transition, source, ..
            } => write!(
                formatter,
                "fixed synchronization program rejected {transition:?}: {source}",
            ),
            Self::Deadlock { deadlock, .. } => write!(
                formatter,
                "fixed synchronization domain {} can deadlock in state {} with unfinished warps {:?}, blocked warps {:?}, pending setmaxnreg requests {:?}, and unready heads {:?}",
                deadlock.protocol_domain(),
                deadlock.protocol_state(),
                deadlock.unfinished_warps(),
                deadlock.blocked_warps(),
                deadlock.pending_setmaxnreg(),
                deadlock.unready_heads(),
            ),
            Self::NonConfluent {
                complete_states, ..
            } => write!(
                formatter,
                "fixed synchronization program has {complete_states} distinct terminal protocol states",
            ),
        }
    }
}

impl Error for FixedSyncVerificationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Protocol { source, .. } => Some(source),
            Self::Deadlock { .. } | Self::NonConfluent { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedSyncVerificationIncomplete {
    ProgramBuild {
        source: FixedSyncProgramBuildError,
    },
    ProgramModel {
        operation: Option<DynamicOpId>,
        transition: FixedSyncTransition,
        source: FixedSyncProgramError,
        witness: Box<[FixedSyncTransition]>,
    },
    StateLimit {
        operation: Option<DynamicOpId>,
        limit: usize,
    },
    TransitionLimit {
        operation: Option<DynamicOpId>,
        limit: usize,
    },
    FirstFailureStop {
        operation: Option<DynamicOpId>,
    },
}

impl fmt::Display for FixedSyncVerificationIncomplete {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProgramBuild { source } => {
                write!(
                    formatter,
                    "cannot build fixed synchronization program: {source}"
                )
            }
            Self::ProgramModel {
                transition, source, ..
            } => write!(
                formatter,
                "fixed synchronization verification is incomplete at {transition:?}: {source}",
            ),
            Self::StateLimit { limit, .. } => write!(
                formatter,
                "fixed synchronization program reached its {limit}-state limit",
            ),
            Self::TransitionLimit { limit, .. } => write!(
                formatter,
                "fixed synchronization program reached its {limit}-transition limit",
            ),
            Self::FirstFailureStop { .. } => write!(
                formatter,
                "fixed synchronization program stopped without retaining its first failure",
            ),
        }
    }
}

impl Error for FixedSyncVerificationIncomplete {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ProgramBuild { source } => Some(source),
            Self::ProgramModel { source, .. } => Some(source),
            Self::StateLimit { .. }
            | Self::TransitionLimit { .. }
            | Self::FirstFailureStop { .. } => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FixedSyncVerificationResult {
    stats: FixedSyncVerificationStats,
    errors: Box<[FixedSyncVerificationError]>,
    incomplete: Box<[FixedSyncVerificationIncomplete]>,
}

impl FixedSyncVerificationResult {
    pub const fn stats(&self) -> FixedSyncVerificationStats {
        self.stats
    }

    pub fn errors(&self) -> &[FixedSyncVerificationError] {
        &self.errors
    }

    pub fn incomplete(&self) -> &[FixedSyncVerificationIncomplete] {
        &self.incomplete
    }

    pub fn proves_clean(&self) -> bool {
        self.errors.is_empty() && self.incomplete.is_empty()
    }
}

/// Reconstruct and exhaustively verify one compact synchronization program.
///
/// The numerical kernel executes exactly once. This search contains only the
/// fixed per-warp synchronization commands, semantic completion tokens, and
/// protocol state reconstructed from that execution.
pub fn verify_fixed_sync_programs(
    transitions: &ResolvedTransitionLog,
    limits: SyncStateSearchLimits,
    max_workers: usize,
) -> FixedSyncVerificationResult {
    let profile_started = Instant::now();
    let projections = match FixedSyncProgram::protocol_projections_from_transition_log(
        transitions,
        max_workers,
    ) {
        Ok(projections) => projections,
        Err(source) => {
            return FixedSyncVerificationResult {
                stats: FixedSyncVerificationStats::default(),
                errors: Box::new([]),
                incomplete: vec![FixedSyncVerificationIncomplete::ProgramBuild { source }]
                    .into_boxed_slice(),
            };
        }
    };
    let projections_built = Instant::now();
    if projections.is_empty() {
        return FixedSyncVerificationResult::default();
    }
    let mut stats = FixedSyncVerificationStats::default();
    let mut errors = Vec::new();
    let mut incomplete = Vec::new();
    let mut state_search_projections = Vec::new();
    let causal_results = parallel_map_ordered(&projections, max_workers, verify_program_causally);
    let causal_verified = Instant::now();
    for (projection, causal_result) in projections.iter().zip(causal_results) {
        match causal_result {
            Some(result) => {
                let terminal = !result.errors.is_empty() || !result.incomplete.is_empty();
                merge_verification_result(&mut stats, &mut errors, &mut incomplete, result);
                if terminal {
                    return FixedSyncVerificationResult {
                        stats,
                        errors: errors.into_boxed_slice(),
                        incomplete: incomplete.into_boxed_slice(),
                    };
                }
            }
            None => state_search_projections.push(projection),
        }
    }
    let causal_merged = Instant::now();
    enum StateSearchPlan {
        Verify(usize),
        Reuse(usize),
    }
    let mut keyed_jobs = HashMap::<u64, Vec<usize>>::new();
    let mut jobs = Vec::<&FixedSyncProgram>::new();
    let mut plans = Vec::with_capacity(state_search_projections.len());
    for projection in state_search_projections {
        let fingerprint = projection.mbarrier_state_search_fingerprint();
        if let Some(job_index) = fingerprint.and_then(|fingerprint| {
            keyed_jobs.get(&fingerprint).and_then(|candidates| {
                candidates.iter().copied().find(|&job_index| {
                    projection.has_equivalent_mbarrier_state_search(jobs[job_index])
                })
            })
        }) {
            plans.push(StateSearchPlan::Reuse(job_index));
            continue;
        }
        let job_index = jobs.len();
        jobs.push(projection);
        plans.push(StateSearchPlan::Verify(job_index));
        if let Some(fingerprint) = fingerprint {
            keyed_jobs
                .entry(fingerprint)
                .or_insert_with(Vec::new)
                .push(job_index);
        }
    }
    let state_search_planned = Instant::now();
    let job_results = parallel_map_ordered(&jobs, max_workers, |program| {
        verify_program_by_state_search(program, limits)
    });
    let state_search_finished = Instant::now();
    let job_clean = job_results
        .iter()
        .map(FixedSyncVerificationResult::proves_clean)
        .collect::<Vec<_>>();
    let mut job_results = job_results.into_iter().map(Some).collect::<Vec<_>>();
    for plan in plans {
        let result = match plan {
            StateSearchPlan::Verify(job_index) => job_results[job_index]
                .take()
                .expect("each fixed-sync state-search job has one primary projection"),
            StateSearchPlan::Reuse(job_index) => {
                assert!(
                    job_clean[job_index],
                    "a duplicate fixed-sync projection follows its failing primary projection"
                );
                FixedSyncVerificationResult {
                    stats: FixedSyncVerificationStats {
                        programs: 1,
                        reused_clean_programs: 1,
                        ..FixedSyncVerificationStats::default()
                    },
                    errors: Box::new([]),
                    incomplete: Box::new([]),
                }
            }
        };
        let terminal = !result.errors.is_empty() || !result.incomplete.is_empty();
        merge_verification_result(&mut stats, &mut errors, &mut incomplete, result);
        if terminal {
            break;
        }
    }
    let results_merged = Instant::now();
    if std::env::var_os("NUMSIM_FIXED_SYNC_PROFILE").is_some() {
        eprintln!(
            "fixed-sync-profile: projection_build={:.6}s causal={:.6}s causal_merge={:.6}s plan={:.6}s search={:.6}s result_merge={:.6}s total={:.6}s projections={} searches={}",
            projections_built.duration_since(profile_started).as_secs_f64(),
            causal_verified.duration_since(projections_built).as_secs_f64(),
            causal_merged.duration_since(causal_verified).as_secs_f64(),
            state_search_planned.duration_since(causal_merged).as_secs_f64(),
            state_search_finished
                .duration_since(state_search_planned)
                .as_secs_f64(),
            results_merged
                .duration_since(state_search_finished)
                .as_secs_f64(),
            results_merged.duration_since(profile_started).as_secs_f64(),
            projections.len(),
            jobs.len(),
        );
    }
    FixedSyncVerificationResult {
        stats,
        errors: errors.into_boxed_slice(),
        incomplete: incomplete.into_boxed_slice(),
    }
}

fn parallel_map_ordered<T, R, F>(items: &[T], max_workers: usize, map: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync,
{
    let worker_count = max_workers
        .max(1)
        .min(MAX_FIXED_SYNC_VALIDATION_WORKERS)
        .min(items.len());
    if worker_count <= 1 {
        return items.iter().map(map).collect();
    }
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let map = &map;
        let handles = (0..worker_count)
            .map(|_| {
                let next = &next;
                scope.spawn(move || {
                    let mut results = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(index) else {
                            break;
                        };
                        results.push((index, map(item)));
                    }
                    results
                })
            })
            .collect::<Vec<_>>();
        let mut indexed_results = Vec::with_capacity(items.len());
        for handle in handles {
            let mut chunk_results = handle
                .join()
                .unwrap_or_else(|payload| std::panic::resume_unwind(payload));
            indexed_results.append(&mut chunk_results);
        }
        indexed_results.sort_unstable_by_key(|(index, _)| *index);
        indexed_results
            .into_iter()
            .map(|(_, result)| result)
            .collect()
    })
}

fn merge_verification_result(
    stats: &mut FixedSyncVerificationStats,
    errors: &mut Vec<FixedSyncVerificationError>,
    incomplete: &mut Vec<FixedSyncVerificationIncomplete>,
    result: FixedSyncVerificationResult,
) {
    stats.programs = stats.programs.saturating_add(result.stats.programs);
    stats.reused_clean_programs = stats
        .reused_clean_programs
        .saturating_add(result.stats.reused_clean_programs);
    stats.visited_states = stats
        .visited_states
        .saturating_add(result.stats.visited_states);
    stats.explored_transitions = stats
        .explored_transitions
        .saturating_add(result.stats.explored_transitions);
    stats.strong_diamond_pruned_transitions = stats
        .strong_diamond_pruned_transitions
        .saturating_add(result.stats.strong_diamond_pruned_transitions);
    errors.extend(result.errors);
    incomplete.extend(result.incomplete);
}

fn verify_program_causally(program: &FixedSyncProgram) -> Option<FixedSyncVerificationResult> {
    let causal = program
        .verify_mbarrier_causally()
        .or_else(|| program.verify_named_barrier_causally())
        .or_else(|| program.verify_cluster_barrier_causally())?;
    let stats = FixedSyncVerificationStats {
        programs: 1,
        reused_clean_programs: 0,
        visited_states: 1,
        explored_transitions: program.command_count(),
        strong_diamond_pruned_transitions: 0,
    };
    Some(match causal {
        Ok(()) => FixedSyncVerificationResult {
            stats,
            errors: Box::new([]),
            incomplete: Box::new([]),
        },
        Err(source) if source.is_incomplete() => FixedSyncVerificationResult {
            stats,
            errors: Box::new([]),
            incomplete: vec![FixedSyncVerificationIncomplete::ProgramModel {
                operation: source.operation().cloned(),
                transition: FixedSyncTransition::ValidateExit,
                source,
                witness: Box::new([]),
            }]
            .into_boxed_slice(),
        },
        Err(source) => {
            let operation = source
                .operation()
                .cloned()
                .or_else(|| program.first_operation().cloned())
                .expect("a causal protocol failure has an operation witness");
            FixedSyncVerificationResult {
                stats,
                errors: vec![FixedSyncVerificationError::Protocol {
                    operation,
                    transition: FixedSyncTransition::ValidateExit,
                    source,
                    witness: Box::new([]),
                    witness_evidence: Box::new([]),
                }]
                .into_boxed_slice(),
                incomplete: Box::new([]),
            }
        }
    })
}

fn verify_program_by_state_search(
    program: &FixedSyncProgram,
    limits: SyncStateSearchLimits,
) -> FixedSyncVerificationResult {
    let search = explore_sync_states(
        program,
        limits,
        SyncStateSearchOptions {
            stop_on_first_failure: true,
            reduce_all_strong_diamonds: true,
            reduce_sleep_sets: true,
        },
    );
    let stats = FixedSyncVerificationStats {
        programs: 1,
        reused_clean_programs: 0,
        visited_states: search.visited_states(),
        explored_transitions: search.explored_transitions(),
        strong_diamond_pruned_transitions: search.strong_diamond_pruned_transitions(),
    };
    let mut errors = Vec::new();
    let mut incomplete = Vec::new();

    if let Some(failure) = search.failures().first() {
        match failure {
            SyncStateFailure::Error {
                transition,
                error,
                witness,
            } if error.is_incomplete() => {
                incomplete.push(FixedSyncVerificationIncomplete::ProgramModel {
                    operation: error.operation().cloned(),
                    transition: transition.clone(),
                    source: error.clone(),
                    witness: witness.clone(),
                });
            }
            SyncStateFailure::Error {
                transition,
                error,
                witness,
            } => {
                let operation = error
                    .operation()
                    .cloned()
                    .or_else(|| transition_operation(program, transition))
                    .or_else(|| program.first_operation().cloned())
                    .expect("a protocol failure has a fixed operation witness");
                errors.push(FixedSyncVerificationError::Protocol {
                    operation,
                    transition: transition.clone(),
                    source: error.clone(),
                    witness: witness.clone(),
                    witness_evidence: build_witness_evidence(program, witness),
                });
            }
            SyncStateFailure::Deadlock { deadlock, witness } => {
                let operation = deadlock
                    .blocked_warps()
                    .first()
                    .and_then(|(_, command)| program.command_witness(*command))
                    .cloned()
                    .or_else(|| program.first_operation().cloned());
                errors.push(FixedSyncVerificationError::Deadlock {
                    operation,
                    deadlock: deadlock.clone(),
                    witness: witness.clone(),
                    witness_evidence: build_witness_evidence(program, witness),
                });
            }
        }
    } else {
        match search.termination() {
            SyncStateSearchTermination::Exhausted if !search.complete_states_are_confluent() => {
                let witnesses = search.complete_witnesses().to_vec().into_boxed_slice();
                let witnesses_evidence = witnesses
                    .iter()
                    .map(|witness| build_witness_evidence(program, witness))
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                errors.push(FixedSyncVerificationError::NonConfluent {
                    operation: program.first_operation().cloned(),
                    complete_states: search.complete_states(),
                    witnesses,
                    witnesses_evidence,
                });
            }
            SyncStateSearchTermination::StateLimit { limit } => {
                incomplete.push(FixedSyncVerificationIncomplete::StateLimit {
                    operation: program.first_operation().cloned(),
                    limit: *limit,
                });
            }
            SyncStateSearchTermination::TransitionLimit { limit } => {
                incomplete.push(FixedSyncVerificationIncomplete::TransitionLimit {
                    operation: program.first_operation().cloned(),
                    limit: *limit,
                });
            }
            SyncStateSearchTermination::FirstFailure => {
                incomplete.push(FixedSyncVerificationIncomplete::FirstFailureStop {
                    operation: program.first_operation().cloned(),
                });
            }
            SyncStateSearchTermination::Exhausted => {}
        }
    }

    FixedSyncVerificationResult {
        stats,
        errors: errors.into_boxed_slice(),
        incomplete: incomplete.into_boxed_slice(),
    }
}

fn build_witness_evidence(
    program: &FixedSyncProgram,
    witness: &[FixedSyncTransition],
) -> Box<[FixedSyncTransitionEvidence]> {
    witness
        .iter()
        .cloned()
        .map(|transition| FixedSyncTransitionEvidence {
            operation: program.transition_operation(&transition),
            description: program.transition_description(&transition),
            transition,
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn transition_operation(
    program: &FixedSyncProgram,
    transition: &FixedSyncTransition,
) -> Option<DynamicOpId> {
    program.transition_operation(transition)
}

#[cfg(test)]
mod tests {
    use super::parallel_map_ordered;

    #[test]
    fn parallel_map_preserves_input_order_and_empty_inputs() {
        let inputs = (0..64).collect::<Vec<_>>();
        let actual = parallel_map_ordered(&inputs, 4, |value| {
            for _ in 0..(value % 5) {
                std::thread::yield_now();
            }
            value * value
        });
        let expected = inputs.iter().map(|value| value * value).collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert!(parallel_map_ordered::<usize, usize, _>(&[], 4, |value| *value).is_empty());
    }
}
