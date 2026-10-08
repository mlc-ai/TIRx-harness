//! Checker-owned Python payload serialization for native Synccheck.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::runtime::{ExecutionPolicy, LaunchSelection};
use crate::{
    engine_error_kind, profile_reset, profile_snapshot, verify_fixed_sync_programs, AllocationId,
    AnalysisGapDomain, AnalysisGapKind, BlockedOperation, CoverageBounds, CoverageStatus,
    CoverageSummary, CoverageUsage, DynamicOpId, EngineError, EngineErrorKind, ExecutionReport,
    ExecutionStats, FixedSyncTransitionEvidence, FixedSyncVerificationError,
    FixedSyncVerificationIncomplete, FixedSyncVerificationResult, LaunchTopology,
    PhysicalBarrierId, PhysicalCompletionKind, PhysicalMemory, ResolvedTransitionLog,
    ResourceAmount, ResourceLimitHit, ResourceLimitKind, ResourceLimits, ResourceUsage,
    SearchTermination, StrictClusterBarrierError, StrictMbarrierError, StrictNamedBarrierError,
    SyncCheckEffectOutcome, SyncCheckIncompleteReason, SyncCheckLaunchState, SyncCheckMode,
    SyncCheckProtocolError, SyncCheckResult, SyncCheckStatus, SyncCheckWaitState,
    SyncStateSearchLimits, WarpEngine,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Engine-owned limits for one native Synccheck phase.
///
/// Generated artifacts construct this through the stable ABI wrapper; checker
/// state, verifier limits, and coverage accounting remain private to the
/// engine.
pub(crate) struct NativeSyncCheckOptions {
    coverage_bounds: CoverageBounds,
    resource_limits: ResourceLimits,
    fixed_sync_search_limits: SyncStateSearchLimits,
    max_polls: Option<usize>,
    max_transitions: Option<usize>,
}

impl NativeSyncCheckOptions {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        max_warp_preemptions: u64,
        max_completion_schedule_deviations: u64,
        max_schedules: u64,
        max_backtrack_nodes: u64,
        max_events_per_run: u64,
        max_total_events: u64,
        max_loop_steps: u64,
        max_wall_time_ms: u64,
        max_diagnostic_bytes: u64,
        max_polls: Option<usize>,
        max_transitions: Option<usize>,
    ) -> Result<Self, String> {
        if max_polls == Some(0) {
            return Err("native synccheck max_polls must be positive".to_string());
        }
        if max_transitions == Some(0) {
            return Err("native synccheck max_transitions must be positive".to_string());
        }
        if [
            max_schedules,
            max_backtrack_nodes,
            max_events_per_run,
            max_total_events,
            max_loop_steps,
            max_wall_time_ms,
            max_diagnostic_bytes,
        ]
        .contains(&0)
        {
            return Err("native synccheck resource limits must be positive".to_string());
        }
        let max_states = usize::try_from(max_backtrack_nodes)
            .map_err(|_| "native synccheck max_backtrack_nodes does not fit usize".to_string())?;
        let fixed_max_transitions = usize::try_from(max_loop_steps)
            .map_err(|_| "native synccheck max_loop_steps does not fit usize".to_string())?;
        Ok(Self {
            coverage_bounds: CoverageBounds::new(
                max_warp_preemptions,
                max_completion_schedule_deviations,
            ),
            resource_limits: ResourceLimits {
                max_schedules,
                max_backtrack_nodes,
                max_events_per_run,
                max_total_events,
                max_loop_steps,
                max_wall_time: Duration::from_millis(max_wall_time_ms),
                max_diagnostic_bytes,
            },
            fixed_sync_search_limits: SyncStateSearchLimits {
                max_states,
                max_transitions: fixed_max_transitions,
            },
            max_polls,
            max_transitions,
        })
    }
}

/// Execute, verify, and serialize one native Synccheck phase inside the engine.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_native_sync_check_phase<MakeWarp, WarpFuture>(
    py: Python<'_>,
    phase_index: usize,
    phase_name: &str,
    topology: LaunchTopology,
    physical: PhysicalMemory,
    inputs: &Bound<'_, PyDict>,
    allocation_ids: &[AllocationId],
    selection: LaunchSelection,
    max_workers: usize,
    execution_policy: ExecutionPolicy,
    fixed_trace_eligible: bool,
    options: NativeSyncCheckOptions,
    mut make_warp: MakeWarp,
) -> PyResult<Py<PyAny>>
where
    MakeWarp: FnMut(WarpEngine<SyncCheckMode>) -> WarpFuture + Send,
    WarpFuture: Future<Output = Result<(), EngineError>> + Send + 'static,
{
    let selected_warp_count = selection.selected_warp_count(topology);
    let global_write_allocations =
        crate::runtime::python::extract_written_allocation_ids(inputs, allocation_ids)?;
    let mode_state = Arc::new(
        SyncCheckLaunchState::with_topology_transition_log_and_global_write_allocations(
            topology,
            ResolvedTransitionLog::with_warp_operation_shards(topology.warp_count()),
            global_write_allocations,
            options.resource_limits.max_diagnostic_bytes,
        ),
    );
    profile_reset();
    let execution_started = Instant::now();
    // The launch runs off the GIL (as the racecheck phase does): a worker
    // thread that drops a host-backed allocation reacquires the GIL for the
    // buffer release, which would deadlock against a phase that held it.
    let execution_mode_state = Arc::clone(&mode_state);
    let execution = py.detach(move || {
        crate::runtime::launch::run_kernel_engine_launch_report_with_poll_limit::<
            SyncCheckMode,
            _,
            _,
        >(
            physical,
            phase_index,
            execution_mode_state,
            selection,
            max_workers,
            options.max_polls,
            execution_policy,
            move |warp| make_warp(warp),
        )
    });
    let execution_wall_time = execution_started.elapsed();
    let execution_profile = profile_snapshot();
    let sync_state = mode_state.as_ref();
    let result = sync_state.result_for_execution(&execution);
    let (fixed_verification, fixed_verification_wall_time) = if execution.is_success()
        && result.findings().is_empty()
        && result.incomplete_reasons().is_empty()
    {
        let started = Instant::now();
        let verification = py.detach(|| {
            verify_fixed_sync_programs(
                sync_state.transition_log(),
                options.fixed_sync_search_limits,
                max_workers,
            )
        });
        (Some(verification), started.elapsed())
    } else {
        (None, Duration::ZERO)
    };
    build_native_sync_check_phase_result(
        py,
        phase_index,
        phase_name,
        topology,
        selected_warp_count,
        selected_warp_count,
        result,
        execution,
        execution_profile,
        fixed_trace_eligible,
        fixed_verification,
        options.coverage_bounds,
        options.resource_limits,
        execution_wall_time,
        fixed_verification_wall_time,
        options.max_polls,
        options.max_transitions,
        execution_policy.native_loop_iteration_budget(),
        execution_policy.native_loop_reschedule_quantum(),
    )
}

fn native_sync_check_search_resource_usage(
    visited_states: u64,
    event_count: u64,
    fixed_transitions: u64,
    _execution_wall_time: Duration,
    fixed_verification_wall_time: Duration,
    diagnostic_bytes: u64,
) -> ResourceUsage {
    ResourceUsage {
        schedules: 1,
        backtrack_nodes: visited_states,
        events_in_current_run: event_count,
        total_events: event_count,
        loop_steps: fixed_transitions,
        // ResourceLimits bound the fixed synchronization search.  The
        // concrete NumSim execution is a completed prerequisite, not pending
        // search work: charging its elapsed time here would retroactively
        // discard a fully executed launch without preventing any work.
        // Preserve end-to-end time in the separate `timing` payload.
        wall_time: fixed_verification_wall_time,
        diagnostic_bytes,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_native_sync_check_phase_result(
    py: Python<'_>,
    phase_index: usize,
    phase_name: &str,
    topology: LaunchTopology,
    cluster_selected_warp_count: usize,
    selected_warp_count: usize,
    result: SyncCheckResult,
    execution: ExecutionReport,
    execution_profile: Vec<(&'static str, u64, u64)>,
    fixed_trace_eligible: bool,
    fixed_verification: Option<FixedSyncVerificationResult>,
    coverage_bounds: crate::CoverageBounds,
    resource_limits: ResourceLimits,
    execution_wall_time: std::time::Duration,
    fixed_verification_wall_time: std::time::Duration,
    max_polls: Option<usize>,
    max_transitions: Option<usize>,
    native_loop_iteration_budget: usize,
    native_loop_reschedule_quantum: usize,
) -> PyResult<Py<PyAny>> {
    let wall_time = execution_wall_time.saturating_add(fixed_verification_wall_time);
    let execution_error = execution.error();
    let partial_domain_setmaxnreg_deadlock = execution_error.is_some_and(|error| {
        setmaxnreg_deadlock_is_partial_domain_incomplete(
            cluster_selected_warp_count,
            topology.warps_per_cluster(),
            error,
        )
    });
    let execution_incomplete = execution_error.is_some_and(engine_error_is_incomplete)
        || has_cluster_barrier_warp_exit_incomplete(&result)
        || partial_domain_setmaxnreg_deadlock;
    let fixed_verification_required =
        result.status() == SyncCheckStatus::Clean && execution.is_success();
    let fixed_error = fixed_verification
        .as_ref()
        .is_some_and(|verification| !verification.errors().is_empty());
    let fixed_incomplete = fixed_verification
        .as_ref()
        .is_some_and(|verification| !verification.incomplete().is_empty());
    let fixed_eligibility_incomplete =
        fixed_verification_required && !fixed_trace_eligible && !fixed_error;
    let fixed_verification_missing =
        fixed_verification_required && fixed_trace_eligible && fixed_verification.is_none();
    let has_error = result.status() == SyncCheckStatus::Error
        || execution_error.is_some() && !execution_incomplete
        || fixed_error;
    let mut analysis_incomplete = result.status() == SyncCheckStatus::Incomplete
        || result.status() == SyncCheckStatus::Clean && execution_incomplete
        || fixed_incomplete
        || fixed_eligibility_incomplete
        || fixed_verification_missing;

    let fixed_stats = fixed_verification
        .as_ref()
        .map(FixedSyncVerificationResult::stats)
        .unwrap_or_default();
    let visited_states = u64::try_from(fixed_stats.visited_states()).unwrap_or(u64::MAX);
    let fixed_transitions = u64::try_from(fixed_stats.explored_transitions()).unwrap_or(u64::MAX);
    let event_count = result
        .total_effect_count()
        .saturating_add(fixed_transitions);
    let effect_payload_plan = SyncCheckEffectPayloadPlan::new(
        &result,
        execution_error,
        fixed_sync_verification_diagnostic_bytes(
            fixed_trace_eligible,
            fixed_verification.as_ref(),
            fixed_verification_required,
        ),
        resource_limits.max_diagnostic_bytes,
    );
    let resource_usage = native_sync_check_search_resource_usage(
        visited_states,
        event_count,
        fixed_transitions,
        execution_wall_time,
        fixed_verification_wall_time,
        effect_payload_plan.diagnostic_bytes,
    );
    let fixed_resource_limit = fixed_verification.as_ref().and_then(|verification| {
        verification
            .incomplete()
            .iter()
            .find_map(|incomplete| match incomplete {
                FixedSyncVerificationIncomplete::StateLimit { .. } => {
                    Some(resource_limits.hit(ResourceLimitKind::BacktrackNodes, resource_usage))
                }
                FixedSyncVerificationIncomplete::TransitionLimit { .. } => {
                    Some(resource_limits.hit(ResourceLimitKind::LoopSteps, resource_usage))
                }
                FixedSyncVerificationIncomplete::ProgramBuild { .. }
                | FixedSyncVerificationIncomplete::ProgramModel { .. }
                | FixedSyncVerificationIncomplete::FirstFailureStop { .. } => None,
            })
    });
    let coverage_resource_limit = fixed_resource_limit
        .or_else(|| first_exceeded_resource_limit(resource_limits, resource_usage));
    analysis_incomplete |= coverage_resource_limit.is_some();
    let termination = if has_error {
        SearchTermination::Finding
    } else if let Some(hit) = coverage_resource_limit {
        SearchTermination::ResourceLimit(hit)
    } else if analysis_incomplete {
        SearchTermination::Unsupported
    } else {
        SearchTermination::WorklistExhausted
    };
    let incomplete_reason = if has_error {
        None
    } else if result.status() == SyncCheckStatus::Incomplete {
        Some(format!(
            "native fixed synchronization verification is incomplete: {:?}",
            result.incomplete_reasons()
        ))
    } else if result.status() == SyncCheckStatus::Clean && execution_incomplete {
        execution_error.map(ToString::to_string)
    } else if fixed_eligibility_incomplete {
        Some(
            "native synchronization control depends on a protocol return that is not fixed by one execution"
                .to_string(),
        )
    } else if let Some(incomplete) = fixed_verification
        .as_ref()
        .and_then(|verification| verification.incomplete().first())
    {
        Some(incomplete.to_string())
    } else if fixed_verification_missing {
        Some("native fixed synchronization verification result is missing".to_string())
    } else if let Some(hit) = coverage_resource_limit {
        Some(format!(
            "native fixed synchronization verification exceeded its {} resource limit",
            resource_limit_kind_name(hit.kind),
        ))
    } else {
        None
    };
    let coverage = CoverageSummary {
        bounds: coverage_bounds,
        maximum_observed_usage: CoverageUsage::default(),
        resource_limits,
        resource_usage,
        pending_work_items: 0,
        pending_backtracks: 0,
        termination,
    };

    let output = build_native_sync_check_phase_dict(
        py,
        phase_index,
        phase_name,
        topology,
        &result,
        &execution,
        &execution_profile,
        cluster_selected_warp_count,
        selected_warp_count,
        max_polls,
        max_transitions,
        native_loop_iteration_budget,
        native_loop_reschedule_quantum,
        &effect_payload_plan,
    )?;
    let timing = PyDict::new(py);
    timing.set_item(
        "execution_wall_time_us",
        duration_micros(execution_wall_time),
    )?;
    timing.set_item(
        "fixed_verification_wall_time_us",
        duration_micros(fixed_verification_wall_time),
    )?;
    timing.set_item("total_wall_time_us", duration_micros(wall_time))?;
    output.set_item("timing", timing)?;
    output.set_item(
        "verdict",
        if has_error {
            "error"
        } else if selected_warp_count != topology.warp_count() || analysis_incomplete {
            "incomplete"
        } else {
            "clean"
        },
    )?;
    append_fixed_sync_verification_payload(
        py,
        &output,
        fixed_trace_eligible,
        fixed_verification.as_ref(),
        fixed_verification_required,
        coverage_resource_limit,
    )?;
    output.set_item("coverage", coverage_summary_to_py(py, &coverage)?)?;
    output.set_item(
        "search",
        fixed_sync_state_search_to_py(
            py,
            termination,
            incomplete_reason.as_deref(),
            fixed_verification.as_ref(),
        )?,
    )?;
    output.set_item("counterexample", py.None())?;

    let replay_limits = PyDict::new(py);
    set_optional_usize(py, &replay_limits, "max_polls", max_polls)?;
    set_optional_usize(py, &replay_limits, "max_transitions", max_transitions)?;
    replay_limits.set_item("native_loop_iteration_budget", native_loop_iteration_budget)?;
    replay_limits.set_item(
        "native_loop_reschedule_quantum",
        native_loop_reschedule_quantum,
    )?;
    output.set_item("replay_resource_limits", replay_limits)?;
    Ok(output.into_any().unbind())
}

#[allow(clippy::too_many_arguments)]
fn build_native_sync_check_phase_dict<'py>(
    py: Python<'py>,
    phase_index: usize,
    phase_name: &str,
    topology: LaunchTopology,
    result: &SyncCheckResult,
    execution: &ExecutionReport,
    execution_profile: &[(&'static str, u64, u64)],
    cluster_selected_warp_count: usize,
    selected_warp_count: usize,
    max_polls: Option<usize>,
    max_transitions: Option<usize>,
    native_loop_iteration_budget: usize,
    native_loop_reschedule_quantum: usize,
    effect_payload_plan: &SyncCheckEffectPayloadPlan,
) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("schema_version", 3_u32)?;
    output.set_item("execution_model", "direct_fixed_sync_state")?;

    let phase = PyDict::new(py);
    phase.set_item("index", phase_index)?;
    phase.set_item("name", phase_name)?;
    phase.set_item("topology", topology_to_py(py, topology)?)?;
    output.set_item("phase", phase)?;
    output.set_item(
        "analysis_scope",
        analysis_scope_to_py(py, selected_warp_count, topology.warp_count())?,
    )?;

    let execution_error = execution.error();
    let partial_domain_setmaxnreg_deadlock = execution_error.is_some_and(|error| {
        setmaxnreg_deadlock_is_partial_domain_incomplete(
            cluster_selected_warp_count,
            topology.warps_per_cluster(),
            error,
        )
    });
    let report_execution_error = reported_sync_execution_error(result, execution_error)
        .filter(|_| !partial_domain_setmaxnreg_deadlock);
    output.set_item("findings", findings_to_py(py, result)?)?;
    output.set_item(
        "incomplete",
        incomplete_to_py(
            py,
            result,
            report_execution_error,
            (selected_warp_count != topology.warp_count())
                .then_some((selected_warp_count, topology.warp_count())),
        )?,
    )?;
    if effect_payload_plan.retain_full_journal {
        output.set_item("effects", effects_to_py(py, result)?)?;
        output.set_item("effect_summary", py.None())?;
    } else {
        output.set_item("effects", PyList::empty(py))?;
        output.set_item(
            "effect_summary",
            effect_summary_to_py(py, effect_payload_plan)?,
        )?;
    }
    output.set_item(
        "stats",
        execution_stats_to_py(py, Some(&execution.stats), execution_profile)?,
    )?;
    match report_execution_error {
        Some(error) => output.set_item(
            "execution_error",
            engine_error_to_py_with_sync_evidence(py, error, result)?,
        )?,
        None => output.set_item("execution_error", py.None())?,
    }

    let limits = PyDict::new(py);
    set_optional_usize(py, &limits, "max_polls", max_polls)?;
    set_optional_usize(py, &limits, "max_transitions", max_transitions)?;
    limits.set_item("native_loop_iteration_budget", native_loop_iteration_budget)?;
    limits.set_item(
        "native_loop_reschedule_quantum",
        native_loop_reschedule_quantum,
    )?;
    output.set_item("resource_limits", limits)?;
    Ok(output)
}

fn first_exceeded_resource_limit(
    limits: ResourceLimits,
    usage: ResourceUsage,
) -> Option<ResourceLimitHit> {
    if usage.schedules > limits.max_schedules {
        Some(limits.hit(ResourceLimitKind::Schedules, usage))
    } else if usage.backtrack_nodes > limits.max_backtrack_nodes {
        Some(limits.hit(ResourceLimitKind::BacktrackNodes, usage))
    } else if usage.events_in_current_run > limits.max_events_per_run {
        Some(limits.hit(ResourceLimitKind::EventsPerRun, usage))
    } else if usage.total_events > limits.max_total_events {
        Some(limits.hit(ResourceLimitKind::TotalEvents, usage))
    } else if usage.loop_steps > limits.max_loop_steps {
        Some(limits.hit(ResourceLimitKind::LoopSteps, usage))
    } else if usage.wall_time > limits.max_wall_time {
        Some(limits.hit(ResourceLimitKind::WallTime, usage))
    } else if usage.diagnostic_bytes > limits.max_diagnostic_bytes {
        Some(limits.hit(ResourceLimitKind::DiagnosticBytes, usage))
    } else {
        None
    }
}

struct SyncCheckEffectPayloadPlan {
    retain_full_journal: bool,
    diagnostic_bytes: u64,
    full_effect_diagnostic_bytes: u64,
    total_effect_count: u64,
    effect_counts: BTreeMap<&'static str, u64>,
}

impl SyncCheckEffectPayloadPlan {
    fn new(
        result: &SyncCheckResult,
        execution_error: Option<&EngineError>,
        fixed_verification_diagnostic_bytes: u64,
        max_diagnostic_bytes: u64,
    ) -> Self {
        let base_diagnostic_bytes = sync_check_non_effect_diagnostic_bytes(result, execution_error)
            .saturating_add(fixed_verification_diagnostic_bytes);
        let full_effect_diagnostic_bytes = result.full_effect_diagnostic_bytes();
        let full_diagnostic_bytes =
            base_diagnostic_bytes.saturating_add(full_effect_diagnostic_bytes);
        let total_effect_count = result.total_effect_count();
        if result.effect_journal_complete()
            && (full_diagnostic_bytes <= max_diagnostic_bytes || total_effect_count == 0)
        {
            return Self {
                retain_full_journal: true,
                diagnostic_bytes: full_diagnostic_bytes,
                full_effect_diagnostic_bytes,
                total_effect_count,
                effect_counts: BTreeMap::new(),
            };
        }

        let mut effect_counts = BTreeMap::new();
        for (effect, count) in result.effect_counts() {
            effect_counts.insert(effect.name(), count);
        }
        let summary_diagnostic_bytes = sync_check_effect_summary_diagnostic_bytes(&effect_counts);
        Self {
            retain_full_journal: false,
            diagnostic_bytes: base_diagnostic_bytes.saturating_add(summary_diagnostic_bytes),
            full_effect_diagnostic_bytes,
            total_effect_count,
            effect_counts,
        }
    }
}

fn sync_check_non_effect_diagnostic_bytes(
    result: &SyncCheckResult,
    execution_error: Option<&EngineError>,
) -> u64 {
    let mut bytes = STRUCTURED_RECORD_OVERHEAD
        .saturating_add(text_bytes(sync_check_status_name(result.status())))
        .saturating_add(STRUCTURED_SEQUENCE_OVERHEAD.saturating_mul(3));
    for finding in result.findings() {
        bytes = bytes.saturating_add(sync_check_finding_diagnostic_bytes(finding));
    }
    for reason in result.incomplete_reasons() {
        bytes = bytes.saturating_add(sync_check_incomplete_diagnostic_bytes(reason));
    }
    if let Some(error) = execution_error {
        bytes = bytes
            .saturating_add(STRUCTURED_RECORD_OVERHEAD)
            .saturating_add(text_bytes(engine_error_kind(error)))
            .saturating_add(display_bytes(error));
    }
    bytes
}

#[cfg(test)]
fn sync_check_diagnostic_bytes(
    result: &SyncCheckResult,
    execution_error: Option<&EngineError>,
) -> u64 {
    sync_check_non_effect_diagnostic_bytes(result, execution_error)
        .saturating_add(result.full_effect_diagnostic_bytes())
}

fn sync_check_effect_summary_diagnostic_bytes(effect_counts: &BTreeMap<&'static str, u64>) -> u64 {
    effect_counts.iter().fold(
        STRUCTURED_RECORD_OVERHEAD
            .saturating_add(text_bytes("bounded_summary"))
            .saturating_add(STRUCTURED_SCALAR_BYTES.saturating_mul(4))
            .saturating_add(STRUCTURED_SEQUENCE_OVERHEAD),
        |bytes, (name, _count)| {
            bytes
                .saturating_add(text_bytes(name))
                .saturating_add(STRUCTURED_SCALAR_BYTES)
        },
    )
}

fn fixed_sync_verification_diagnostic_bytes(
    fixed_trace_eligible: bool,
    verification: Option<&FixedSyncVerificationResult>,
    verification_required: bool,
) -> u64 {
    if !verification_required {
        return 0;
    }
    if !fixed_trace_eligible
        && !verification.is_some_and(|verification| !verification.errors().is_empty())
    {
        return text_bytes(
            "native synchronization control depends on a protocol return that is not fixed by one execution",
        );
    }
    let Some(verification) = verification else {
        return text_bytes("native fixed synchronization verification result is missing");
    };
    verification
        .errors()
        .iter()
        .map(display_bytes)
        .chain(verification.incomplete().iter().map(display_bytes))
        .fold(0_u64, u64::saturating_add)
}

fn append_fixed_sync_verification_payload(
    py: Python<'_>,
    output: &Bound<'_, PyDict>,
    fixed_trace_eligible: bool,
    verification: Option<&FixedSyncVerificationResult>,
    verification_required: bool,
    coverage_resource_limit: Option<ResourceLimitHit>,
) -> PyResult<()> {
    let findings = output
        .get_item("findings")?
        .expect("native synccheck phase payload has findings");
    let findings = findings.cast::<PyList>()?;
    let incomplete = output
        .get_item("incomplete")?
        .expect("native synccheck phase payload has incomplete reasons");
    let incomplete = incomplete.cast::<PyList>()?;
    let fixed_limit_recorded = verification.is_some_and(|verification| {
        verification.incomplete().iter().any(|reason| {
            matches!(
                reason,
                FixedSyncVerificationIncomplete::StateLimit { .. }
                    | FixedSyncVerificationIncomplete::TransitionLimit { .. }
            )
        })
    });
    if let Some(hit) = coverage_resource_limit.filter(|_| !fixed_limit_recorded) {
        let value = PyDict::new(py);
        value.set_item("kind", "analysis_incomplete")?;
        value.set_item("reason", "resource_limit")?;
        value.set_item("resource", resource_limit_kind_name(hit.kind))?;
        value.set_item("limit", resource_amount_to_py(py, hit.limit)?)?;
        value.set_item("usage", resource_amount_to_py(py, hit.usage)?)?;
        incomplete.append(value)?;
    }
    if !verification_required {
        return Ok(());
    }

    if !fixed_trace_eligible
        && !verification.is_some_and(|verification| !verification.errors().is_empty())
    {
        let value = PyDict::new(py);
        value.set_item("kind", "analysis_incomplete")?;
        value.set_item("reason", "fixed_sync_state_ineligible")?;
        value.set_item(
            "message",
            "native synchronization control depends on a protocol return that is not fixed by one execution",
        )?;
        incomplete.append(value)?;
        return Ok(());
    }
    let Some(verification) = verification else {
        let value = PyDict::new(py);
        value.set_item("kind", "analysis_incomplete")?;
        value.set_item("reason", "fixed_sync_state_verification_missing")?;
        value.set_item(
            "message",
            "native fixed synchronization verification result is missing",
        )?;
        incomplete.append(value)?;
        return Ok(());
    };
    for error in verification.errors() {
        findings.append(fixed_sync_verification_error_to_py(py, error)?)?;
    }
    for reason in verification.incomplete() {
        incomplete.append(fixed_sync_verification_incomplete_to_py(py, reason)?)?;
    }
    Ok(())
}

fn fixed_sync_verification_error_to_py<'py>(
    py: Python<'py>,
    error: &FixedSyncVerificationError,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("message", error.to_string())?;
    match error.operation() {
        Some(operation) => value.set_item("operation", dynamic_op_to_py(py, operation)?)?,
        None => value.set_item("operation", py.None())?,
    }
    match error {
        FixedSyncVerificationError::Protocol {
            transition,
            source,
            witness,
            witness_evidence,
            ..
        } => {
            value.set_item("kind", "fixed_sync_protocol_error")?;
            match source.protocol_kind() {
                Some(kind) => value.set_item("protocol", format!("{kind:?}"))?,
                None => value.set_item("protocol", py.None())?,
            }
            value.set_item("transition", format!("{transition:?}"))?;
            value.set_item("source", source.to_string())?;
            value.set_item(
                "related_operations",
                source
                    .related_operations()
                    .iter()
                    .map(|operation| dynamic_op_to_py(py, operation))
                    .collect::<PyResult<Vec<_>>>()?,
            )?;
            value.set_item(
                "witness",
                witness
                    .iter()
                    .map(|transition| format!("{transition:?}"))
                    .collect::<Vec<_>>(),
            )?;
            value.set_item(
                "witness_evidence",
                fixed_sync_witness_evidence_to_py(py, witness_evidence)?,
            )?;
        }
        FixedSyncVerificationError::Deadlock {
            deadlock,
            witness,
            witness_evidence,
            ..
        } => {
            value.set_item("kind", "deadlock")?;
            value.set_item("verification", "fixed_sync")?;
            value.set_item("deadlock", format!("{deadlock:?}"))?;
            value.set_item(
                "witness",
                witness
                    .iter()
                    .map(|transition| format!("{transition:?}"))
                    .collect::<Vec<_>>(),
            )?;
            value.set_item(
                "witness_evidence",
                fixed_sync_witness_evidence_to_py(py, witness_evidence)?,
            )?;
        }
        FixedSyncVerificationError::NonConfluent {
            complete_states,
            witnesses,
            witnesses_evidence,
            ..
        } => {
            value.set_item("kind", "fixed_sync_nonconfluent")?;
            value.set_item("complete_states", complete_states)?;
            let rendered = witnesses
                .iter()
                .map(|witness| {
                    witness
                        .iter()
                        .map(|transition| format!("{transition:?}"))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            value.set_item("witness", rendered.last().cloned().unwrap_or_default())?;
            value.set_item("witnesses", rendered)?;
            let evidence = witnesses_evidence
                .iter()
                .map(|witness| fixed_sync_witness_evidence_to_py(py, witness))
                .collect::<PyResult<Vec<_>>>()?;
            value.set_item("witnesses_evidence", evidence)?;
        }
    }
    Ok(value)
}

fn fixed_sync_witness_evidence_to_py<'py>(
    py: Python<'py>,
    evidence: &[FixedSyncTransitionEvidence],
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    evidence
        .iter()
        .map(|step| {
            let value = PyDict::new(py);
            value.set_item("transition", format!("{:?}", step.transition()))?;
            value.set_item("description", step.description())?;
            match step.operation() {
                Some(operation) => value.set_item("operation", dynamic_op_to_py(py, operation)?)?,
                None => value.set_item("operation", py.None())?,
            }
            Ok(value)
        })
        .collect()
}

fn fixed_sync_verification_incomplete_to_py<'py>(
    py: Python<'py>,
    reason: &FixedSyncVerificationIncomplete,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("kind", "analysis_incomplete")?;
    value.set_item("message", reason.to_string())?;
    match reason {
        FixedSyncVerificationIncomplete::ProgramBuild { source } => {
            value.set_item("reason", "fixed_sync_program_build")?;
            value.set_item("source", source.to_string())?;
        }
        FixedSyncVerificationIncomplete::ProgramModel {
            operation,
            transition,
            source,
            witness,
        } => {
            value.set_item("reason", "fixed_sync_program_model_incomplete")?;
            match operation {
                Some(operation) => value.set_item("operation", dynamic_op_to_py(py, operation)?)?,
                None => value.set_item("operation", py.None())?,
            }
            value.set_item("transition", format!("{transition:?}"))?;
            value.set_item("source", source.to_string())?;
            value.set_item(
                "witness",
                witness
                    .iter()
                    .map(|transition| format!("{transition:?}"))
                    .collect::<Vec<_>>(),
            )?;
        }
        FixedSyncVerificationIncomplete::StateLimit { operation, limit } => {
            value.set_item("reason", "resource_limit")?;
            value.set_item("resource", "fixed_sync_states")?;
            match operation {
                Some(operation) => value.set_item("operation", dynamic_op_to_py(py, operation)?)?,
                None => value.set_item("operation", py.None())?,
            }
            value.set_item("limit", limit)?;
        }
        FixedSyncVerificationIncomplete::TransitionLimit { operation, limit } => {
            value.set_item("reason", "resource_limit")?;
            value.set_item("resource", "fixed_sync_transitions")?;
            match operation {
                Some(operation) => value.set_item("operation", dynamic_op_to_py(py, operation)?)?,
                None => value.set_item("operation", py.None())?,
            }
            value.set_item("limit", limit)?;
        }
        FixedSyncVerificationIncomplete::FirstFailureStop { operation } => {
            value.set_item("reason", "fixed_sync_first_failure_unretained")?;
            match operation {
                Some(operation) => value.set_item("operation", dynamic_op_to_py(py, operation)?)?,
                None => value.set_item("operation", py.None())?,
            }
        }
    }
    Ok(value)
}

// The explorer needs a stable resource charge, not a rendered Rust Debug dump.
// These constants conservatively account for compact structured record/list
// metadata while the helpers below charge every scalar and variable payload.
const STRUCTURED_RECORD_OVERHEAD: u64 = 16;
const STRUCTURED_SEQUENCE_OVERHEAD: u64 = 8;
const STRUCTURED_SCALAR_BYTES: u64 = 8;

fn sync_check_status_name(status: SyncCheckStatus) -> &'static str {
    match status {
        SyncCheckStatus::Clean => "clean",
        SyncCheckStatus::Incomplete => "incomplete",
        SyncCheckStatus::Error => "error",
    }
}

fn sync_check_finding_diagnostic_bytes(finding: &crate::SyncCheckFinding) -> u64 {
    STRUCTURED_RECORD_OVERHEAD
        .saturating_add(dynamic_op_diagnostic_bytes(finding.operation()))
        .saturating_add(text_bytes(finding.effect().name()))
        .saturating_add(text_bytes(strict_error_kind(finding.error())))
        .saturating_add(display_bytes(finding.error()))
}

fn sync_check_incomplete_diagnostic_bytes(reason: &SyncCheckIncompleteReason) -> u64 {
    let bytes = STRUCTURED_RECORD_OVERHEAD;
    match reason {
        SyncCheckIncompleteReason::AnalysisGap {
            operation,
            kind,
        } => bytes
            .saturating_add(text_bytes("analysis_gap"))
            .saturating_add(dynamic_op_diagnostic_bytes(operation))
            .saturating_add(text_bytes(kind.domain().name()))
            .saturating_add(text_bytes(kind.name())),
        SyncCheckIncompleteReason::CompletionActionUnobserved { barrier_id, .. } => bytes
            .saturating_add(text_bytes("completion_action_unobserved"))
            .saturating_add(STRUCTURED_SCALAR_BYTES.saturating_mul(2))
            .saturating_add(physical_barrier_diagnostic_bytes(*barrier_id)),
        SyncCheckIncompleteReason::CompletionTransitionUnobserved {
            operation,
            barrier_id,
            ..
        } => bytes
            .saturating_add(text_bytes("completion_transition_unobserved"))
            .saturating_add(dynamic_op_diagnostic_bytes(operation))
            .saturating_add(physical_barrier_diagnostic_bytes(*barrier_id))
            .saturating_add(STRUCTURED_SCALAR_BYTES),
        SyncCheckIncompleteReason::EffectCommitUnobserved { operation, effect } => bytes
            .saturating_add(text_bytes("effect_commit_unobserved"))
            .saturating_add(dynamic_op_diagnostic_bytes(operation))
            .saturating_add(text_bytes(effect.name())),
        SyncCheckIncompleteReason::ClusterBarrierParticipantExitUnmodeled {
            barrier_id,
            missing_warps,
            ..
        } => bytes
            .saturating_add(text_bytes("cluster_barrier_warp_exit_unmodeled"))
            .saturating_add(value_bytes(barrier_id))
            .saturating_add(STRUCTURED_SCALAR_BYTES)
            .saturating_add(usize_slice_diagnostic_bytes(missing_warps)),
        SyncCheckIncompleteReason::ClusterBarrierUnalignedUnmodeled { operation, effect } => bytes
            .saturating_add(text_bytes("cluster_barrier_unaligned_unmodeled"))
            .saturating_add(dynamic_op_diagnostic_bytes(operation))
            .saturating_add(text_bytes(effect.name())),
        SyncCheckIncompleteReason::ClusterBarrierRearrivalWithoutWaitUnmodeled {
            operation,
            barrier_id,
            ..
        } => bytes
            .saturating_add(text_bytes(
                "cluster_barrier_rearrival_without_wait_unmodeled",
            ))
            .saturating_add(dynamic_op_diagnostic_bytes(operation))
            .saturating_add(value_bytes(barrier_id))
            .saturating_add(STRUCTURED_SCALAR_BYTES.saturating_mul(2)),
    }
}

fn dynamic_op_diagnostic_bytes(operation: &DynamicOpId) -> u64 {
    STRUCTURED_RECORD_OVERHEAD
        .saturating_add(STRUCTURED_SCALAR_BYTES.saturating_mul(4))
        .saturating_add(STRUCTURED_SEQUENCE_OVERHEAD)
        .saturating_add(
            count_bytes(operation.loop_frames().len())
                .saturating_mul(STRUCTURED_RECORD_OVERHEAD + STRUCTURED_SCALAR_BYTES * 2),
        )
}

fn physical_barrier_diagnostic_bytes(_barrier: PhysicalBarrierId) -> u64 {
    STRUCTURED_RECORD_OVERHEAD.saturating_add(STRUCTURED_SCALAR_BYTES.saturating_mul(3))
}

fn usize_slice_diagnostic_bytes(values: &[usize]) -> u64 {
    STRUCTURED_SEQUENCE_OVERHEAD
        .saturating_add(count_bytes(values.len()).saturating_mul(STRUCTURED_SCALAR_BYTES))
}

fn value_bytes<T>(value: &T) -> u64 {
    count_bytes(std::mem::size_of_val(value))
}

fn text_bytes(value: &str) -> u64 {
    count_bytes(value.len())
}

fn count_bytes(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn display_bytes(value: &impl fmt::Display) -> u64 {
    let mut counter = DiagnosticByteCounter::default();
    fmt::write(&mut counter, format_args!("{value}")).expect("diagnostic byte counter cannot fail");
    counter.bytes
}

#[derive(Default)]
struct DiagnosticByteCounter {
    bytes: u64,
}

impl fmt::Write for DiagnosticByteCounter {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        self.bytes = self.bytes.saturating_add(text_bytes(value));
        Ok(())
    }
}

pub(crate) fn analysis_scope_to_py<'py>(
    py: Python<'py>,
    selected_warp_count: usize,
    total_warp_count: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item(
        "kind",
        if selected_warp_count == total_warp_count {
            "full_launch"
        } else {
            "subset"
        },
    )?;
    value.set_item("selected_warp_count", selected_warp_count)?;
    value.set_item("total_warp_count", total_warp_count)?;
    Ok(value)
}

fn fixed_sync_state_search_to_py<'py>(
    py: Python<'py>,
    termination: SearchTermination,
    incomplete_reason: Option<&str>,
    verification: Option<&FixedSyncVerificationResult>,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    let stats = verification
        .map(FixedSyncVerificationResult::stats)
        .unwrap_or_default();
    value.set_item("algorithm", "fixed_sync_state")?;
    value.set_item("run_count", 1_u64)?;
    value.set_item("backtrack_count", 0_u64)?;
    value.set_item("sleep_pruned_branch_count", 0_u64)?;
    value.set_item("program_count", stats.programs())?;
    value.set_item("reused_clean_program_count", stats.reused_clean_programs())?;
    value.set_item("visited_state_count", stats.visited_states())?;
    value.set_item("explored_transition_count", stats.explored_transitions())?;
    value.set_item(
        "strong_diamond_pruned_transition_count",
        stats.strong_diamond_pruned_transitions(),
    )?;
    match incomplete_reason {
        Some(reason) => value.set_item("incomplete_reason", reason)?,
        None => value.set_item("incomplete_reason", py.None())?,
    }

    let run = PyDict::new(py);
    run.set_item("prefix", PyList::empty(py))?;
    run.set_item("warp_preemption_bound", 0_u64)?;
    run.set_item("trace_digest", py.None())?;
    run.set_item("trace_digest_hex", py.None())?;
    run.set_item(
        "coverage_usage",
        coverage_usage_to_py(py, CoverageUsage::default())?,
    )?;
    run.set_item(
        "status",
        match termination {
            SearchTermination::Finding => "finding",
            SearchTermination::WorklistExhausted => "complete",
            SearchTermination::ResourceLimit(_)
            | SearchTermination::Unsupported
            | SearchTermination::Cancelled => "incomplete",
        },
    )?;
    let runs = PyList::empty(py);
    runs.append(run)?;
    value.set_item("runs", runs)?;
    Ok(value)
}

pub(crate) fn coverage_summary_to_py<'py>(
    py: Python<'py>,
    coverage: &CoverageSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item(
        "status",
        match coverage.status() {
            CoverageStatus::CompleteWithinBounds => "complete_within_bounds",
            CoverageStatus::Finding => "finding",
            CoverageStatus::Incomplete => "incomplete",
        },
    )?;
    value.set_item("eligible_for_clean", coverage.eligible_for_clean())?;

    let bounds = PyDict::new(py);
    bounds.set_item("max_warp_preemptions", coverage.bounds.max_warp_preemptions)?;
    bounds.set_item(
        "max_completion_schedule_deviations",
        coverage.bounds.max_completion_schedule_deviations,
    )?;
    value.set_item("bounds", bounds)?;
    value.set_item(
        "maximum_observed_usage",
        coverage_usage_to_py(py, coverage.maximum_observed_usage)?,
    )?;
    value.set_item(
        "resource_limits",
        resource_limits_to_py(py, coverage.resource_limits)?,
    )?;
    value.set_item(
        "resource_usage",
        resource_usage_to_py(py, coverage.resource_usage)?,
    )?;
    value.set_item("pending_work_items", coverage.pending_work_items)?;
    value.set_item("pending_backtracks", coverage.pending_backtracks)?;
    value.set_item("termination", termination_to_py(py, coverage.termination)?)?;
    Ok(value)
}

fn coverage_usage_to_py<'py>(
    py: Python<'py>,
    usage: CoverageUsage,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("warp_preemptions", usage.warp_preemptions)?;
    value.set_item(
        "completion_schedule_deviations",
        usage.completion_schedule_deviations,
    )?;
    Ok(value)
}

fn resource_limits_to_py<'py>(
    py: Python<'py>,
    limits: ResourceLimits,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("max_schedules", limits.max_schedules)?;
    value.set_item("max_backtrack_nodes", limits.max_backtrack_nodes)?;
    value.set_item("max_events_per_run", limits.max_events_per_run)?;
    value.set_item("max_total_events", limits.max_total_events)?;
    value.set_item("max_loop_steps", limits.max_loop_steps)?;
    value.set_item("max_wall_time_ms", duration_millis(limits.max_wall_time))?;
    value.set_item("max_diagnostic_bytes", limits.max_diagnostic_bytes)?;
    Ok(value)
}

fn resource_usage_to_py<'py>(
    py: Python<'py>,
    usage: ResourceUsage,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("schedules", usage.schedules)?;
    value.set_item("backtrack_nodes", usage.backtrack_nodes)?;
    value.set_item("events_in_current_run", usage.events_in_current_run)?;
    value.set_item("total_events", usage.total_events)?;
    value.set_item("loop_steps", usage.loop_steps)?;
    value.set_item("wall_time_ms", duration_millis(usage.wall_time))?;
    value.set_item("diagnostic_bytes", usage.diagnostic_bytes)?;
    Ok(value)
}

fn termination_to_py<'py>(
    py: Python<'py>,
    termination: SearchTermination,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("kind", termination_kind(termination))?;
    match termination {
        SearchTermination::ResourceLimit(hit) => {
            value.set_item("resource_limit", resource_limit_to_py(py, hit)?)?;
        }
        _ => value.set_item("resource_limit", py.None())?,
    }
    Ok(value)
}

fn termination_kind(termination: SearchTermination) -> &'static str {
    match termination {
        SearchTermination::WorklistExhausted => "worklist_exhausted",
        SearchTermination::Finding => "finding",
        SearchTermination::ResourceLimit(_) => "resource_limit",
        SearchTermination::Unsupported => "unsupported",
        SearchTermination::Cancelled => "cancelled",
    }
}

fn resource_limit_to_py<'py>(
    py: Python<'py>,
    hit: ResourceLimitHit,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("resource", resource_limit_kind_name(hit.kind))?;
    value.set_item("limit", resource_amount_to_py(py, hit.limit)?)?;
    value.set_item("usage", resource_amount_to_py(py, hit.usage)?)?;
    Ok(value)
}

fn resource_limit_kind_name(kind: ResourceLimitKind) -> &'static str {
    match kind {
        ResourceLimitKind::Schedules => "schedules",
        ResourceLimitKind::BacktrackNodes => "backtrack_nodes",
        ResourceLimitKind::EventsPerRun => "events_per_run",
        ResourceLimitKind::TotalEvents => "total_events",
        ResourceLimitKind::LoopSteps => "loop_steps",
        ResourceLimitKind::WallTime => "wall_time",
        ResourceLimitKind::DiagnosticBytes => "diagnostic_bytes",
    }
}

fn resource_amount_to_py<'py>(
    py: Python<'py>,
    amount: ResourceAmount,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    match amount {
        ResourceAmount::Count(count) => {
            value.set_item("kind", "count")?;
            value.set_item("value", count)?;
        }
        ResourceAmount::Time(duration) => {
            value.set_item("kind", "milliseconds")?;
            value.set_item("value", duration_millis(duration))?;
        }
    }
    Ok(value)
}

fn duration_millis(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn duration_micros(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

pub(crate) fn sync_result_to_py<'py>(
    py: Python<'py>,
    result: &SyncCheckResult,
    execution_error: Option<&EngineError>,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    let execution_incomplete = execution_error.is_some_and(engine_error_is_incomplete)
        || has_cluster_barrier_warp_exit_incomplete(result);
    value.set_item(
        "verdict",
        match result.status() {
            SyncCheckStatus::Error => "error",
            _ if execution_error.is_some() && !execution_incomplete => "error",
            SyncCheckStatus::Incomplete => "incomplete",
            SyncCheckStatus::Clean if execution_incomplete => "incomplete",
            SyncCheckStatus::Clean => "clean",
        },
    )?;
    value.set_item("findings", findings_to_py(py, result)?)?;
    value.set_item(
        "incomplete",
        incomplete_to_py(py, result, execution_error, None)?,
    )?;
    let effect_payload_plan = SyncCheckEffectPayloadPlan::new(result, execution_error, 0, u64::MAX);
    if effect_payload_plan.retain_full_journal {
        value.set_item("effects", effects_to_py(py, result)?)?;
    } else {
        value.set_item("effects", PyList::empty(py))?;
        value.set_item(
            "effect_summary",
            effect_summary_to_py(py, &effect_payload_plan)?,
        )?;
    }
    Ok(value)
}

fn findings_to_py<'py>(py: Python<'py>, result: &SyncCheckResult) -> PyResult<Bound<'py, PyList>> {
    let findings = PyList::empty(py);
    for finding in result.findings() {
        let value = PyDict::new(py);
        value.set_item("kind", strict_error_kind(finding.error()))?;
        value.set_item("effect", finding.effect().name())?;
        value.set_item("operation", dynamic_op_to_py(py, finding.operation())?)?;
        value.set_item("message", finding.error().to_string())?;
        let related_operations = strict_error_related_operations(finding.error());
        if !related_operations.is_empty() {
            value.set_item(
                "related_operations",
                related_operations
                    .into_iter()
                    .map(|operation| dynamic_op_to_py(py, operation))
                    .collect::<PyResult<Vec<_>>>()?,
            )?;
        }
        findings.append(value)?;
    }
    Ok(findings)
}

fn incomplete_to_py<'py>(
    py: Python<'py>,
    result: &SyncCheckResult,
    execution_error: Option<&EngineError>,
    subset_execution: Option<(usize, usize)>,
) -> PyResult<Bound<'py, PyList>> {
    let incomplete = PyList::empty(py);
    let semantic_terminal_error = result.status() == SyncCheckStatus::Error
        || execution_error.is_some_and(|error| !engine_error_is_incomplete(error));
    for reason in result.incomplete_reasons() {
        if semantic_terminal_error && is_irrelevant_after_terminal_error(reason) {
            continue;
        }
        incomplete.append(sync_incomplete_to_py(py, reason)?)?;
    }
    if let Some(error) = execution_error.filter(|error| {
        engine_error_is_incomplete(error) && !result.incomplete_reasons().iter().any(|reason| {
            matches!(
                reason,
                SyncCheckIncompleteReason::AnalysisGap { operation, kind, .. }
                    if Some(kind.name()) == engine_error_analysis_kind(error)
                        && error.operation_context().is_none_or(|context| context.id() == operation)
            )
        })
    }) {
        incomplete.append(engine_incomplete_to_py(py, error)?)?;
    }
    if let Some((selected_warp_count, total_warp_count)) = subset_execution {
        let value = PyDict::new(py);
        value.set_item("kind", "analysis_incomplete")?;
        value.set_item("reason", "subset_execution")?;
        value.set_item("selected_warp_count", selected_warp_count)?;
        value.set_item("total_warp_count", total_warp_count)?;
        incomplete.append(value)?;
    }
    Ok(incomplete)
}

fn is_irrelevant_after_terminal_error(reason: &SyncCheckIncompleteReason) -> bool {
    matches!(
        reason,
        SyncCheckIncompleteReason::CompletionActionUnobserved { .. }
            | SyncCheckIncompleteReason::CompletionTransitionUnobserved { .. }
            | SyncCheckIncompleteReason::EffectCommitUnobserved { .. }
    )
}

pub(crate) fn sync_incomplete_to_py<'py>(
    py: Python<'py>,
    reason: &SyncCheckIncompleteReason,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("kind", "analysis_incomplete")?;
    match reason {
        SyncCheckIncompleteReason::AnalysisGap {
            operation,
            kind,
        } => {
            value.set_item(
                "reason",
                match kind.domain() {
                    AnalysisGapDomain::Tcgen => "tcgen_protocol_unmodeled",
                    AnalysisGapDomain::ClusterBarrier => "cluster_barrier_unaligned_unmodeled",
                    AnalysisGapDomain::Atomic => "atomic_lane_serialization_unmodeled",
                },
            )?;
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
            value.set_item("domain", kind.domain().name())?;
            value.set_item("effect", kind.name())?;
        }
        SyncCheckIncompleteReason::CompletionActionUnobserved {
            action_id,
            barrier_id,
            generation,
        } => {
            value.set_item("reason", "completion_action_unobserved")?;
            value.set_item("action_id", action_id.get())?;
            value.set_item("barrier", physical_barrier_to_py(py, *barrier_id)?)?;
            value.set_item("generation", generation)?;
        }
        SyncCheckIncompleteReason::CompletionTransitionUnobserved {
            operation,
            barrier_id,
            generation,
        } => {
            value.set_item("reason", "completion_transition_unobserved")?;
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
            value.set_item("barrier", physical_barrier_to_py(py, *barrier_id)?)?;
            value.set_item("generation", generation)?;
        }
        SyncCheckIncompleteReason::EffectCommitUnobserved { operation, effect } => {
            value.set_item("reason", "effect_commit_unobserved")?;
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
            value.set_item("effect", effect.name())?;
        }
        SyncCheckIncompleteReason::ClusterBarrierParticipantExitUnmodeled {
            barrier_id,
            generation,
            missing_warps,
        } => {
            value.set_item("reason", "cluster_barrier_warp_exit_unmodeled")?;
            value.set_item("kernel_index", barrier_id.kernel_index())?;
            value.set_item("cluster_id", barrier_id.cluster_id())?;
            value.set_item("generation", generation)?;
            value.set_item("missing_warps", missing_warps.as_ref())?;
        }
        SyncCheckIncompleteReason::ClusterBarrierUnalignedUnmodeled { operation, effect } => {
            value.set_item("reason", "cluster_barrier_unaligned_unmodeled")?;
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
            value.set_item("effect", effect.name())?;
        }
        SyncCheckIncompleteReason::ClusterBarrierRearrivalWithoutWaitUnmodeled {
            operation,
            barrier_id,
            generation,
            warp_id,
        } => {
            value.set_item("reason", "cluster_barrier_rearrival_without_wait_unmodeled")?;
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
            value.set_item("kernel_index", barrier_id.kernel_index())?;
            value.set_item("cluster_id", barrier_id.cluster_id())?;
            value.set_item("generation", generation)?;
            value.set_item("warp_id", warp_id)?;
            value.set_item("verification", "[VERIFY]")?;
        }
    }
    Ok(value)
}

pub(crate) fn has_cluster_barrier_warp_exit_incomplete(result: &SyncCheckResult) -> bool {
    result.incomplete_reasons().iter().any(|reason| {
        matches!(
            reason,
            SyncCheckIncompleteReason::ClusterBarrierParticipantExitUnmodeled { .. }
        )
    })
}

pub(crate) fn reported_sync_execution_error<'a>(
    result: &SyncCheckResult,
    execution_error: Option<&'a EngineError>,
) -> Option<&'a EngineError> {
    execution_error.filter(|error| {
        let participant_exits = error.cluster_barrier_participant_exit_evidence();
        participant_exits.is_empty()
            || !participant_exits.iter().all(|participant_exit| {
                result.incomplete_reasons().iter().any(|reason| {
                    matches!(
                        reason,
                        SyncCheckIncompleteReason::ClusterBarrierParticipantExitUnmodeled {
                            barrier_id,
                            generation,
                            missing_warps,
                        } if barrier_id.cluster_id() == participant_exit.cluster_id()
                            && *generation == participant_exit.generation()
                            && missing_warps.as_ref() == participant_exit.exited_warps()
                    )
                })
            })
    })
}

fn effects_to_py<'py>(py: Python<'py>, result: &SyncCheckResult) -> PyResult<Bound<'py, PyList>> {
    let effects = PyList::empty(py);
    for effect in result.effects() {
        let value = PyDict::new(py);
        value.set_item("operation", dynamic_op_to_py(py, effect.operation())?)?;
        value.set_item("effect", effect.effect().name())?;
        value.set_item("outcome", effect_outcome_to_py(py, effect.outcome())?)?;
        effects.append(value)?;
    }
    Ok(effects)
}

fn effect_summary_to_py<'py>(
    py: Python<'py>,
    plan: &SyncCheckEffectPayloadPlan,
) -> PyResult<Bound<'py, PyDict>> {
    debug_assert!(!plan.retain_full_journal);
    let value = PyDict::new(py);
    value.set_item("retention", "bounded_summary")?;
    value.set_item("total_count", plan.total_effect_count)?;
    value.set_item("retained_count", 0_u64)?;
    value.set_item("omitted_count", plan.total_effect_count)?;
    value.set_item("full_diagnostic_bytes", plan.full_effect_diagnostic_bytes)?;
    let counts = PyDict::new(py);
    for (name, count) in &plan.effect_counts {
        counts.set_item(name, count)?;
    }
    value.set_item("counts", counts)?;
    Ok(value)
}

fn effect_outcome_to_py<'py>(
    py: Python<'py>,
    outcome: &SyncCheckEffectOutcome,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    match outcome {
        SyncCheckEffectOutcome::Applied => value.set_item("kind", "applied")?,
        SyncCheckEffectOutcome::MbarrierArrive {
            completed_generation,
            consumed_generation,
            ready_warps,
        } => {
            value.set_item("kind", "mbarrier_arrive")?;
            set_optional_u64(py, &value, "completed_generation", *completed_generation)?;
            set_optional_u64(py, &value, "consumed_generation", *consumed_generation)?;
            value.set_item("ready_warps", ready_warps.as_ref())?;
        }
        SyncCheckEffectOutcome::MbarrierArriveBatch { arrivals } => {
            value.set_item("kind", "mbarrier_arrive_batch")?;
            let arrival_values = PyList::empty(py);
            for arrival in arrivals {
                let arrival_value = PyDict::new(py);
                arrival_value
                    .set_item("barrier", physical_barrier_to_py(py, arrival.barrier_id())?)?;
                arrival_value.set_item("generation", arrival.generation())?;
                set_optional_u64(
                    py,
                    &arrival_value,
                    "completed_generation",
                    arrival.completed_generation(),
                )?;
                set_optional_u64(
                    py,
                    &arrival_value,
                    "consumed_generation",
                    arrival.consumed_generation(),
                )?;
                arrival_value.set_item("ready_warps", arrival.ready_warps())?;
                arrival_values.append(arrival_value)?;
            }
            value.set_item("arrivals", arrival_values)?;
        }
        SyncCheckEffectOutcome::MbarrierWait { staged, committed } => {
            value.set_item("kind", "mbarrier_wait")?;
            value.set_item("staged", wait_state_to_py(py, staged)?)?;
            match committed {
                Some(committed) => value.set_item("committed", wait_state_to_py(py, committed)?)?,
                None => value.set_item("committed", py.None())?,
            }
        }
        SyncCheckEffectOutcome::MbarrierCompletionIssue { actions } => {
            value.set_item("kind", "mbarrier_completion_issue")?;
            let action_values = PyList::empty(py);
            for action in actions {
                let action_value = PyDict::new(py);
                action_value.set_item("action_id", action.action_id().get())?;
                action_value
                    .set_item("barrier", physical_barrier_to_py(py, action.barrier_id())?)?;
                set_optional_u64(py, &action_value, "generation", action.generation())?;
                action_value.set_item("transactions", action.transactions())?;
                action_values.append(action_value)?;
            }
            value.set_item("actions", action_values)?;
        }
        SyncCheckEffectOutcome::MbarrierCompletion {
            action_id,
            barrier_id,
            generation,
            transactions,
            completion_kind,
            completed_generation,
            ready_warps,
        } => {
            value.set_item("kind", "mbarrier_completion")?;
            value.set_item("action_id", action_id.get())?;
            value.set_item("barrier", physical_barrier_to_py(py, *barrier_id)?)?;
            value.set_item("generation", generation)?;
            value.set_item("transactions", transactions)?;
            value.set_item(
                "completion_kind",
                match completion_kind {
                    PhysicalCompletionKind::Transaction { .. } => "transaction",
                    PhysicalCompletionKind::Arrival { .. } => "arrival",
                },
            )?;
            set_optional_u64(py, &value, "completed_generation", *completed_generation)?;
            value.set_item("ready_warps", ready_warps.as_ref())?;
        }
        SyncCheckEffectOutcome::NamedBarrierArrive {
            generation,
            completed_now,
        } => {
            value.set_item("kind", "named_barrier_arrive")?;
            value.set_item("generation", generation)?;
            value.set_item("completed_now", completed_now)?;
        }
        SyncCheckEffectOutcome::ClusterBarrier {
            generation,
            completed_now,
        } => {
            value.set_item("kind", "cluster_barrier")?;
            value.set_item("generation", generation)?;
            value.set_item("completed_now", completed_now)?;
        }
    }
    Ok(value)
}

fn wait_state_to_py<'py>(
    py: Python<'py>,
    state: &SyncCheckWaitState,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    match state {
        SyncCheckWaitState::Ready {
            generation,
            consumed_now,
        } => {
            value.set_item("kind", "ready")?;
            set_optional_u64(py, &value, "generation", *generation)?;
            value.set_item("consumed_now", consumed_now)?;
        }
        SyncCheckWaitState::Registered { generation } => {
            value.set_item("kind", "registered")?;
            value.set_item("generation", generation)?;
        }
    }
    Ok(value)
}

pub(crate) fn execution_stats_to_py<'py>(
    py: Python<'py>,
    stats: Option<&ExecutionStats>,
    profile: &[(&'static str, u64, u64)],
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    match stats {
        Some(stats) => {
            value.set_item("available", true)?;
            value.set_item("task_count", stats.task_count)?;
            value.set_item("completed_task_count", stats.completed_task_count)?;
            value.set_item("poll_count", stats.poll_count)?;
            value.set_item("normal_poll_count", stats.normal_poll_count)?;
            value.set_item("poll_recheck_poll_count", stats.poll_recheck_poll_count)?;
            value.set_item("completion_pump_count", stats.completion_pump_count)?;
            value.set_item(
                "completion_operation_count",
                stats.completion_operation_count,
            )?;
            value.set_item("worker_count", stats.worker_count)?;
            value.set_item("scheduling_domain_count", stats.scheduling_domain_count)?;
            #[cfg(feature = "profile")]
            value.set_item(
                "executed_instruction_variants",
                crate::runtime::python::instruction_variants_to_py(py, stats)?,
            )?;
        }
        None => {
            value.set_item("available", false)?;
            for key in [
                "task_count",
                "completed_task_count",
                "poll_count",
                "normal_poll_count",
                "poll_recheck_poll_count",
                "completion_pump_count",
                "completion_operation_count",
                "worker_count",
                "scheduling_domain_count",
            ] {
                value.set_item(key, py.None())?;
            }
        }
    }
    let profile_value = PyDict::new(py);
    for &(name, count, nanos) in profile {
        let entry = PyDict::new(py);
        entry.set_item("count", count)?;
        entry.set_item("nanos", nanos)?;
        profile_value.set_item(name, entry)?;
    }
    value.set_item("profile", profile_value)?;
    Ok(value)
}

pub(crate) fn engine_error_to_py<'py>(
    py: Python<'py>,
    error: &EngineError,
) -> PyResult<Bound<'py, PyDict>> {
    if engine_error_is_incomplete(error) {
        let value = engine_incomplete_to_py(py, error)?;
        value.set_item("message", error.to_string())?;
        if let Some(operation) = error.operation_context() {
            value.set_item("operation", dynamic_op_to_py(py, operation.id())?)?;
        }
        return Ok(value);
    }
    let value = PyDict::new(py);
    value.set_item("kind", engine_error_kind(error))?;
    value.set_item("message", error.to_string())?;
    if let Some(operation) = error.operation_context() {
        value.set_item("operation", dynamic_op_to_py(py, operation.id())?)?;
    }
    Ok(value)
}

const MAX_RETAINED_DEADLOCK_OPERATIONS: usize = 64;

fn engine_error_to_py_with_sync_evidence<'py>(
    py: Python<'py>,
    error: &EngineError,
    result: &SyncCheckResult,
) -> PyResult<Bound<'py, PyDict>> {
    let Some((blocked_warps, blocked_operations, poll_count)) = engine_deadlock(error) else {
        return engine_error_to_py(py, error);
    };

    let bounded = blocked_operations.len() > MAX_RETAINED_DEADLOCK_OPERATIONS
        || blocked_warps.len() > MAX_RETAINED_DEADLOCK_OPERATIONS;
    let value = if bounded {
        let value = PyDict::new(py);
        value.set_item("kind", engine_error_kind(error))?;
        value.set_item(
            "message",
            format!(
                "executor deadlock after {poll_count} polls; {} blocked warps and {} \
                 synchronization waits; structured details are retained as a bounded summary",
                blocked_warps.len(),
                blocked_operations.len(),
            ),
        )?;
        value
    } else {
        engine_error_to_py(py, error)?
    };

    let representative = blocked_operations
        .iter()
        .find_map(BlockedOperation::operation);
    if let Some(operation) = representative {
        value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
    }

    let blocked_values = PyList::empty(py);
    let mut described_warps = BTreeSet::new();
    for blocked in blocked_operations
        .iter()
        .take(MAX_RETAINED_DEADLOCK_OPERATIONS)
    {
        let item = PyDict::new(py);
        item.set_item("warp_id", blocked.warp_id)?;
        item.set_item("awaited_operation", blocked.awaited_operation().to_string())?;
        item.set_item("phase", blocked.phase)?;
        item.set_item("description", blocked.to_string())?;
        if let Some(operation) = blocked.operation() {
            item.set_item("operation", dynamic_op_to_py(py, operation)?)?;
        }
        described_warps.insert(blocked.warp_id);
        blocked_values.append(item)?;
    }
    described_warps.extend(blocked_operations.iter().map(|blocked| blocked.warp_id));
    value.set_item("blocked_operations", blocked_values)?;
    value.set_item("blocked_warp_count", blocked_warps.len())?;
    value.set_item("blocked_operation_count", blocked_operations.len())?;
    value.set_item(
        "retained_blocked_operation_count",
        blocked_operations
            .len()
            .min(MAX_RETAINED_DEADLOCK_OPERATIONS),
    )?;
    value.set_item(
        "omitted_blocked_operation_count",
        blocked_operations
            .len()
            .saturating_sub(MAX_RETAINED_DEADLOCK_OPERATIONS),
    )?;

    // An engine-level stuttering-loop checkpoint is scheduler state, not a
    // synchronization completion source, so it is absent from
    // `blocked_operations`. Preserve the latest exact operation for every such
    // stalled warp; for scheduler loops this is the acquire/scoped-load site
    // that explains why the warp failed to join its following barrier.
    let stalled_values = PyList::empty(py);
    let stalled_warps = blocked_warps
        .iter()
        .copied()
        .filter(|warp_id| !described_warps.contains(warp_id))
        .collect::<BTreeSet<_>>();
    let latest_operations = latest_reported_operations(result, &stalled_warps);
    let mut retained_stalled = 0usize;
    for &warp_id in blocked_warps {
        if described_warps.contains(&warp_id) {
            continue;
        }
        let Some(&operation) = latest_operations.get(&warp_id) else {
            continue;
        };
        if representative.is_none() && value.get_item("operation")?.is_none() {
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
        }
        if retained_stalled >= MAX_RETAINED_DEADLOCK_OPERATIONS {
            continue;
        }
        let item = PyDict::new(py);
        item.set_item("warp_id", warp_id)?;
        item.set_item("operation", dynamic_op_to_py(py, operation)?)?;
        stalled_values.append(item)?;
        retained_stalled += 1;
    }
    value.set_item("stalled_operations", stalled_values)?;
    value.set_item("stalled_operation_count", latest_operations.len())?;
    value.set_item("retained_stalled_operation_count", retained_stalled)?;
    value.set_item(
        "omitted_stalled_operation_count",
        latest_operations.len().saturating_sub(retained_stalled),
    )?;
    value.set_item(
        "diagnostic_retention",
        if bounded { "bounded_summary" } else { "full" },
    )?;
    Ok(value)
}

fn engine_deadlock(error: &EngineError) -> Option<(&[usize], &[BlockedOperation], usize)> {
    match error.kind() {
        EngineErrorKind::Deadlock {
            blocked_warps,
            blocked_operations,
            poll_count,
        } => Some((blocked_warps, blocked_operations, *poll_count)),
        EngineErrorKind::Context { source, .. } | EngineErrorKind::WarpFailed { source, .. } => {
            engine_deadlock(source)
        }
        _ => None,
    }
}

fn latest_reported_operations<'a>(
    result: &'a SyncCheckResult,
    warp_ids: &BTreeSet<usize>,
) -> BTreeMap<usize, &'a DynamicOpId> {
    let effects = result.effects().iter().map(|effect| effect.operation());
    let findings = result.findings().iter().map(|finding| finding.operation());
    let analysis_gaps = result
        .incomplete_reasons()
        .iter()
        .filter_map(|reason| match reason {
            SyncCheckIncompleteReason::AnalysisGap { operation, .. } => Some(operation),
            _ => None,
        });
    effects
        .chain(findings)
        .chain(analysis_gaps)
        .filter(|operation| warp_ids.contains(&operation.global_warp_id()))
        .fold(BTreeMap::new(), |mut latest, operation| {
            let warp_id = operation.global_warp_id();
            let replace = latest.get(&warp_id).is_none_or(|prior: &&DynamicOpId| {
                prior.per_warp_sequence() < operation.per_warp_sequence()
            });
            if replace {
                latest.insert(warp_id, operation);
            }
            latest
        })
}

pub(crate) fn engine_incomplete_to_py<'py>(
    py: Python<'py>,
    error: &EngineError,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("kind", "analysis_incomplete")?;
    if let Some((label, limit)) = engine_native_loop_iteration_limit(error) {
        value.set_item("reason", "resource_limit")?;
        value.set_item("resource", "native_loop_iterations")?;
        value.set_item("loop", label)?;
        value.set_item("limit", limit)?;
        return Ok(value);
    }
    match error.kind() {
        EngineErrorKind::PollLimitExceeded {
            limit,
            pending_warps,
        } => {
            value.set_item("reason", "resource_limit")?;
            value.set_item("resource", "polls")?;
            value.set_item("limit", limit)?;
            value.set_item("pending_warps", pending_warps)?;
        }
        _ => {
            let kind = engine_error_analysis_kind(error)
                .expect("analysis-incomplete engine errors carry a gap kind");
            value.set_item("reason", kind)?;
            value.set_item("effect", kind)?;
            value.set_item("message", error.to_string())?;
            if let Some(operation) = error.operation_context() {
                value.set_item("operation", dynamic_op_to_py(py, operation.id())?)?;
            }
        }
    }
    Ok(value)
}

pub(crate) fn engine_error_is_incomplete(error: &EngineError) -> bool {
    matches!(error.kind(), EngineErrorKind::PollLimitExceeded { .. })
        || engine_native_loop_iteration_limit(error).is_some()
        || engine_error_analysis_kind(error).is_some()
}

fn engine_native_loop_iteration_limit(error: &EngineError) -> Option<(&str, usize)> {
    match error.kind() {
        EngineErrorKind::NativeLoopIterationLimitExceeded { label, limit } => {
            Some((label.as_str(), *limit))
        }
        EngineErrorKind::Context { source, .. } | EngineErrorKind::WarpFailed { source, .. } => {
            engine_native_loop_iteration_limit(source)
        }
        _ => None,
    }
}

fn engine_error_analysis_kind(error: &EngineError) -> Option<&'static str> {
    match error.kind() {
        EngineErrorKind::AnalysisIncomplete { kind } => Some(*kind),
        EngineErrorKind::Context { source, .. } | EngineErrorKind::WarpFailed { source, .. } => {
            engine_error_analysis_kind(source)
        }
        _ => None,
    }
}

pub(crate) fn setmaxnreg_deadlock_is_partial_domain_incomplete(
    cluster_selected_warp_count: usize,
    cluster_total_warp_count: usize,
    error: &EngineError,
) -> bool {
    cluster_selected_warp_count != cluster_total_warp_count
        && engine_error_is_setmaxnreg_deadlock(error)
}

fn engine_error_is_setmaxnreg_deadlock(error: &EngineError) -> bool {
    match error.kind() {
        EngineErrorKind::Deadlock {
            blocked_operations, ..
        } => blocked_operations.iter().any(|blocked| {
            matches!(
                blocked.awaited_operation(),
                crate::AwaitedOperation::Setmaxnreg | crate::AwaitedOperation::SetmaxnregPool
            )
        }),
        EngineErrorKind::Context { source, .. } | EngineErrorKind::WarpFailed { source, .. } => {
            engine_error_is_setmaxnreg_deadlock(source)
        }
        _ => false,
    }
}

fn strict_error_kind(error: &SyncCheckProtocolError) -> &'static str {
    match error {
        SyncCheckProtocolError::Mbarrier(error) => mbarrier_error_kind(error),
        SyncCheckProtocolError::NamedBarrier(error) => named_barrier_error_kind(error),
        SyncCheckProtocolError::ClusterBarrier(error) => cluster_barrier_error_kind(error),
        SyncCheckProtocolError::Causality(error) => causality_error_kind(error),
    }
}

fn strict_error_related_operations(error: &SyncCheckProtocolError) -> Vec<&DynamicOpId> {
    match error {
        SyncCheckProtocolError::Causality(
            crate::SyncCausalityError::PriorGenerationConsumptionNotHappensBefore {
                consumption_operation: Some(operation),
                ..
            },
        ) => vec![operation],
        _ => Vec::new(),
    }
}

fn causality_error_kind(error: &crate::SyncCausalityError) -> &'static str {
    match error {
        crate::SyncCausalityError::InvalidWarp { .. } => "sync_causality_invalid_warp",
        crate::SyncCausalityError::ClockOverflow { .. } => "sync_causality_clock_overflow",
        crate::SyncCausalityError::ClockDimensionMismatch { .. } => {
            "sync_causality_clock_dimension_mismatch"
        }
        crate::SyncCausalityError::DuplicateSyncParticipant { .. } => {
            "sync_causality_duplicate_participant"
        }
        crate::SyncCausalityError::BarrierAlreadyInitialized { .. } => {
            "mbarrier_causal_already_initialized"
        }
        crate::SyncCausalityError::BarrierUninitialized { .. } => "mbarrier_causal_use_before_init",
        crate::SyncCausalityError::InitNotHappensBeforeUse { .. } => {
            "mbarrier_init_not_happens_before_use"
        }
        crate::SyncCausalityError::PriorGenerationMissing { .. } => {
            "mbarrier_prior_generation_missing"
        }
        crate::SyncCausalityError::PriorGenerationNotConsumed { .. } => {
            "mbarrier_prior_generation_not_consumed"
        }
        crate::SyncCausalityError::PriorGenerationConsumptionNotHappensBefore { .. } => {
            "mbarrier_prior_generation_consumption_not_happens_before"
        }
        crate::SyncCausalityError::MissingGenerationRelease { .. } => {
            "mbarrier_generation_release_missing"
        }
        crate::SyncCausalityError::ReleasePayloadRetired { .. } => {
            "barrier_generation_release_retired"
        }
        crate::SyncCausalityError::GenerationAlreadyConsumed { .. } => {
            "mbarrier_generation_already_consumed"
        }
        crate::SyncCausalityError::CompletionTokenSpaceExhausted => {
            "mbarrier_causal_completion_token_space_exhausted"
        }
        crate::SyncCausalityError::UnknownCompletionToken { .. } => {
            "mbarrier_causal_completion_token_unknown"
        }
        crate::SyncCausalityError::StaleCompletionToken { .. } => {
            "mbarrier_causal_completion_token_stale"
        }
        crate::SyncCausalityError::ReinitializeGenerationMismatch { .. } => {
            "mbarrier_causal_reinitialize_generation_mismatch"
        }
        crate::SyncCausalityError::ReinitializeWithOutstandingCompletions { .. } => {
            "mbarrier_causal_reinitialize_with_outstanding_completions"
        }
        crate::SyncCausalityError::EpochOverflow { .. } => "mbarrier_causal_epoch_overflow",
    }
}

fn cluster_barrier_error_kind(error: &StrictClusterBarrierError) -> &'static str {
    match error {
        StrictClusterBarrierError::PartialWarpParticipation { .. } => {
            "cluster_barrier_partial_warp"
        }
        StrictClusterBarrierError::ContractMismatch { .. } => "cluster_barrier_contract_mismatch",
        StrictClusterBarrierError::UnexpectedParticipant { .. } => {
            "cluster_barrier_unexpected_participant"
        }
        StrictClusterBarrierError::EarlyArrival { .. } => "cluster_barrier_early_arrival",
        StrictClusterBarrierError::WaitBeforeArrival { .. } => {
            "cluster_barrier_wait_before_arrival"
        }
        StrictClusterBarrierError::DuplicateWait { .. } => "cluster_barrier_duplicate_wait",
        StrictClusterBarrierError::ResumeWithoutRegistration { .. } => {
            "cluster_barrier_resume_without_registration"
        }
        StrictClusterBarrierError::ResumeBeforeCompletion { .. } => {
            "cluster_barrier_resume_before_completion"
        }
        StrictClusterBarrierError::GenerationOverflow { .. } => {
            "cluster_barrier_generation_overflow"
        }
    }
}

fn mbarrier_error_kind(error: &StrictMbarrierError) -> &'static str {
    match error {
        StrictMbarrierError::InvalidateWithOutstandingWork { .. } => {
            "mbarrier_invalidate_with_outstanding_work"
        }
        StrictMbarrierError::Uninitialized { .. } => "mbarrier_use_before_init",
        StrictMbarrierError::InvalidPhase { .. } => "mbarrier_invalid_phase",
        StrictMbarrierError::AcquireBeforeCompletion { .. } => "mbarrier_acquire_before_completion",
        StrictMbarrierError::InvalidExpectedArrivals { .. } => "mbarrier_invalid_expected_arrivals",
        StrictMbarrierError::ReinitializeBeforeConsumption { .. } => {
            "mbarrier_reinit_before_consumption"
        }
        StrictMbarrierError::ReinitializeWhileActive { .. } => "mbarrier_reinit_while_active",
        StrictMbarrierError::ReinitializeWithoutInvalidation { .. } => {
            "mbarrier_reinit_without_inval"
        }
        StrictMbarrierError::ExpectTxBeforeConsumption { .. } => {
            "mbarrier_expect_tx_before_consumption"
        }
        StrictMbarrierError::ArriveBeforeConsumption { .. } => "mbarrier_arrive_before_consumption",
        StrictMbarrierError::ArrivalOverflow { .. } => "mbarrier_arrival_overflow",
        StrictMbarrierError::CounterOverflow { .. } => "mbarrier_counter_overflow",
        StrictMbarrierError::TransactionOverDelivery { .. } => "mbarrier_transaction_over_delivery",
        StrictMbarrierError::DuplicateWaiter { .. } => "mbarrier_duplicate_waiter",
        StrictMbarrierError::CompletionAfterGenerationComplete { .. } => {
            "mbarrier_completion_after_generation_complete"
        }
        StrictMbarrierError::StaleCompletion { .. } => "mbarrier_stale_completion",
        StrictMbarrierError::FutureCompletionNotBufferable { .. } => {
            "mbarrier_future_completion_not_bufferable"
        }
        StrictMbarrierError::UnknownCompletionToken { .. } => "mbarrier_unknown_completion_token",
        StrictMbarrierError::GenerationOverflow { .. } => "mbarrier_generation_overflow",
        StrictMbarrierError::CompletionTokenSpaceExhausted { .. } => {
            "mbarrier_completion_token_space_exhausted"
        }
    }
}

fn named_barrier_error_kind(error: &StrictNamedBarrierError) -> &'static str {
    match error {
        StrictNamedBarrierError::ElectSyncParticipation { .. } => "elect_sync_named_barrier",
        StrictNamedBarrierError::InvalidExpectedArrivals { .. } => {
            "named_barrier_invalid_expected_arrivals"
        }
        StrictNamedBarrierError::InvalidArrivalCount { .. } => {
            "named_barrier_invalid_arrival_count"
        }
        StrictNamedBarrierError::ContractMismatch { .. } => "named_barrier_contract_mismatch",
        StrictNamedBarrierError::DuplicateContribution { .. } => {
            "named_barrier_duplicate_contribution"
        }
        StrictNamedBarrierError::ArrivalOverflow { .. } => "named_barrier_arrival_overflow",
        StrictNamedBarrierError::CounterOverflow { .. } => "named_barrier_counter_overflow",
        StrictNamedBarrierError::GenerationOverflow { .. } => "named_barrier_generation_overflow",
        StrictNamedBarrierError::ResumeWithoutRegistration { .. } => {
            "named_barrier_resume_without_registration"
        }
        StrictNamedBarrierError::AlignedSyncContractMismatch { .. } => {
            "aligned_sync_contract_mismatch"
        }
        StrictNamedBarrierError::FullCtaAlignedMissingParticipants { .. } => {
            "full_cta_aligned_control"
        }
    }
}

pub(crate) fn dynamic_op_to_py<'py>(
    py: Python<'py>,
    operation: &DynamicOpId,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("kernel_index", operation.kernel_index())?;
    value.set_item("global_warp_id", operation.global_warp_id())?;
    value.set_item("per_warp_sequence", operation.per_warp_sequence())?;
    value.set_item("source_op_id", operation.source_op_id().get())?;
    let loop_frames = PyList::empty(py);
    for frame in operation.loop_frames() {
        let frame_value = PyDict::new(py);
        frame_value.set_item("loop_site_id", frame.loop_site_id().get())?;
        frame_value.set_item("iteration_ordinal", frame.iteration_ordinal())?;
        loop_frames.append(frame_value)?;
    }
    value.set_item("loop_frames", loop_frames)?;
    Ok(value)
}

pub(crate) fn physical_barrier_to_py<'py>(
    py: Python<'py>,
    barrier: PhysicalBarrierId,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("allocation_id", barrier.allocation_id())?;
    value.set_item("byte_offset", barrier.byte_offset())?;
    value.set_item("target_global_cta_id", barrier.target_global_cta_id())?;
    Ok(value)
}

pub(crate) fn topology_to_py<'py>(
    py: Python<'py>,
    topology: LaunchTopology,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("clusters", topology.clusters())?;
    value.set_item("ctas_per_cluster", topology.ctas_per_cluster())?;
    value.set_item("warps_per_cta", topology.warps_per_cta())?;
    value.set_item("warp_count", topology.warp_count())?;
    Ok(value)
}

pub(crate) fn set_optional_usize(
    py: Python<'_>,
    value: &Bound<'_, PyDict>,
    key: &str,
    item: Option<usize>,
) -> PyResult<()> {
    match item {
        Some(item) => value.set_item(key, item),
        None => value.set_item(key, py.None()),
    }
}

fn set_optional_u64(
    py: Python<'_>,
    value: &Bound<'_, PyDict>,
    key: &str,
    item: Option<u64>,
) -> PyResult<()> {
    match item {
        Some(item) => value.set_item(key, item),
        None => value.set_item(key, py.None()),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        AnalysisGapEffect, AnalysisGapKind, BlockedOperation, DynamicOpId, LoopFrame,
        NamedBarrierId, OccurrenceKey, OperationContext, OperationKind, ParticipantState,
        ScopeInstance, StaticOpId, StrictNamedBarrierProtocol, SyncCheckLaunchState, WarpMask,
    };

    use super::*;

    #[test]
    fn named_barrier_protocol_errors_have_stable_finding_kinds() {
        let id = NamedBarrierId::new(0, 3);
        let protocol = StrictNamedBarrierProtocol::new(None);
        protocol
            .sync(id, 16, 0, WarpMask::from_bits(0x0000_00ff), None)
            .unwrap();

        let contract = protocol.sync(id, 32, 1, WarpMask::FULL, None).unwrap_err();
        assert_eq!(
            strict_error_kind(&SyncCheckProtocolError::NamedBarrier(contract)),
            "named_barrier_contract_mismatch"
        );

        let duplicate = protocol
            .sync(id, 16, 0, WarpMask::from_bits(0x0000_000f), None)
            .unwrap_err();
        assert_eq!(
            strict_error_kind(&SyncCheckProtocolError::NamedBarrier(duplicate)),
            "named_barrier_duplicate_contribution"
        );

        let overflow = protocol.sync(id, 16, 1, WarpMask::FULL, None).unwrap_err();
        assert_eq!(
            strict_error_kind(&SyncCheckProtocolError::NamedBarrier(overflow)),
            "named_barrier_arrival_overflow"
        );
    }

    #[test]
    fn only_a_partial_scheduling_domain_suppresses_setmaxnreg_deadlock() {
        let blocked = BlockedOperation::new(
            1,
            crate::AwaitedOperation::Setmaxnreg,
            OccurrenceKey::new(
                0,
                "setmaxnreg",
                std::iter::empty::<i64>(),
                ScopeInstance::WarpGroup {
                    global_cta_id: 0,
                    warpgroup_id: 0,
                },
            ),
            None,
            ParticipantState {
                expected: vec![0, 1, 2, 3],
                arrived: vec![1, 2, 3],
                missing: vec![0],
                expected_arrival_count: None,
                completed_arrival_count: None,
                expected_transactions: None,
                completed_transactions: None,
            },
        );
        let error = EngineError::deadlock(vec![1, 2, 3], vec![blocked], 4);

        assert!(engine_error_is_setmaxnreg_deadlock(&error));
        assert!(setmaxnreg_deadlock_is_partial_domain_incomplete(
            3, 4, &error
        ));
        assert!(!setmaxnreg_deadlock_is_partial_domain_incomplete(
            4, 4, &error
        ));

        let pool_blocked = BlockedOperation::new(
            0,
            crate::AwaitedOperation::SetmaxnregPool,
            OccurrenceKey::new(
                0,
                "setmaxnreg.pool",
                std::iter::empty::<i64>(),
                ScopeInstance::WarpGroup {
                    global_cta_id: 0,
                    warpgroup_id: 0,
                },
            ),
            None,
            ParticipantState {
                expected: vec![0, 1, 2, 3],
                arrived: vec![0, 1, 2, 3],
                missing: vec![],
                expected_arrival_count: None,
                completed_arrival_count: None,
                expected_transactions: None,
                completed_transactions: None,
            },
        );
        let pool_error = EngineError::deadlock(vec![0, 1, 2, 3], vec![pool_blocked], 4);
        assert_eq!(engine_error_kind(&pool_error), "setmaxnreg_pool_deadlock");
        assert!(engine_error_is_setmaxnreg_deadlock(&pool_error));
    }

    #[test]
    fn diagnostic_byte_estimate_tracks_structured_result_payload() {
        let clean = SyncCheckLaunchState::new().result();
        let clean_bytes = sync_check_diagnostic_bytes(&clean, None);
        assert!(clean_bytes > 0);

        let state = SyncCheckLaunchState::new();
        let operation = OperationContext::new(
            DynamicOpId::new(
                0,
                3,
                9,
                StaticOpId::new(17),
                [
                    LoopFrame::new(StaticOpId::new(21), 5),
                    LoopFrame::new(StaticOpId::new(22), 8),
                ],
            ),
            OperationKind::Control,
            WarpMask::FULL,
        );
        state
            .record_external_analysis_gap(
                &operation,
                AnalysisGapEffect::new(AnalysisGapKind::ClusterBarrierUnaligned, 0, 0, None),
            )
            .unwrap();
        let incomplete = state.result();
        let incomplete_bytes = sync_check_diagnostic_bytes(&incomplete, None);
        assert!(incomplete_bytes > clean_bytes);
        assert!(
            dynamic_op_diagnostic_bytes(operation.id())
                > dynamic_op_diagnostic_bytes(&DynamicOpId::new(0, 3, 9, StaticOpId::new(17), [],))
        );
    }

    #[test]
    fn diagnostic_byte_estimate_counts_execution_error_display() {
        let result = SyncCheckLaunchState::new().result();
        let error = EngineError::poll_limit_exceeded(11, vec![1, 7, 19]);
        assert_eq!(display_bytes(&error), text_bytes(&error.to_string()));
        assert!(
            sync_check_diagnostic_bytes(&result, Some(&error))
                > sync_check_diagnostic_bytes(&result, None)
        );
    }

    #[test]
    fn oversized_success_effect_journal_uses_an_exact_bounded_summary() {
        let result = SyncCheckResult::with_test_applied_effects(100);
        let full = SyncCheckEffectPayloadPlan::new(&result, None, 0, u64::MAX);
        assert!(full.retain_full_journal);
        assert_eq!(full.total_effect_count, 100);

        let bounded = SyncCheckEffectPayloadPlan::new(
            &result,
            None,
            0,
            full.diagnostic_bytes.saturating_sub(1),
        );
        assert!(!bounded.retain_full_journal);
        assert_eq!(bounded.total_effect_count, 100);
        assert_eq!(bounded.effect_counts.get("bar.sync.register"), Some(&100));
        assert!(bounded.diagnostic_bytes < full.diagnostic_bytes);
    }

    #[test]
    fn completed_execution_does_not_consume_the_fixed_search_wall_time_limit() {
        let limits = ResourceLimits {
            max_wall_time: Duration::from_secs(30),
            ..ResourceLimits::unbounded()
        };
        let execution_wall_time = Duration::from_secs(31);
        let fixed_verification_wall_time = Duration::from_millis(35);
        let usage = native_sync_check_search_resource_usage(
            7,
            11,
            13,
            execution_wall_time,
            fixed_verification_wall_time,
            17,
        );

        assert!(
            execution_wall_time.saturating_add(fixed_verification_wall_time) > limits.max_wall_time
        );
        assert_eq!(usage.wall_time, fixed_verification_wall_time);
        assert_eq!(usage.backtrack_nodes, 7);
        assert_eq!(usage.total_events, 11);
        assert_eq!(usage.loop_steps, 13);
        assert_eq!(usage.diagnostic_bytes, 17);
        assert_eq!(first_exceeded_resource_limit(limits, usage), None);
    }

    #[test]
    fn large_deadlock_payload_is_bounded_and_source_anchored() {
        let count = MAX_RETAINED_DEADLOCK_OPERATIONS + 7;
        let blocked = (0..count)
            .map(|warp_id| {
                let operation = DynamicOpId::new(0, warp_id, 17, StaticOpId::new(313), []);
                BlockedOperation::new(
                    warp_id,
                    crate::AwaitedOperation::NamedBarrierSync,
                    OccurrenceKey::new(
                        0,
                        "named_barrier",
                        [3_i64],
                        ScopeInstance::Cta { global_cta_id: 0 },
                    ),
                    None,
                    ParticipantState::counted([], 128, 96, None, None),
                )
                .with_operation(Some(operation))
            })
            .collect::<Vec<_>>();
        let error = EngineError::deadlock((0..count).collect(), blocked, 19);
        let result = SyncCheckLaunchState::new().result();

        Python::attach(|py| {
            let value = engine_error_to_py_with_sync_evidence(py, &error, &result).unwrap();
            assert_eq!(
                value
                    .get_item("diagnostic_retention")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "bounded_summary"
            );
            assert_eq!(
                value
                    .get_item("blocked_operation_count")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                count
            );
            assert_eq!(
                value
                    .get_item("omitted_blocked_operation_count")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                7
            );
            assert_eq!(
                value
                    .get_item("blocked_operations")
                    .unwrap()
                    .unwrap()
                    .cast::<PyList>()
                    .unwrap()
                    .len(),
                MAX_RETAINED_DEADLOCK_OPERATIONS
            );
            assert!(value.get_item("operation").unwrap().is_some());
        });
    }

    #[test]
    fn live_analysis_gaps_keep_their_domain_and_terminal_visibility() {
        let operation = DynamicOpId::new(0, 0, 0, StaticOpId::new(9), []);
        Python::attach(|py| {
            for (kind, domain, label) in [
                (
                    AnalysisGapKind::TcgenMma,
                    "tcgen",
                    "tcgen_protocol_unmodeled",
                ),
                (
                    AnalysisGapKind::ClusterBarrierUnaligned,
                    "cluster_barrier",
                    "cluster_barrier_unaligned_unmodeled",
                ),
                (
                    AnalysisGapKind::AtomicLaneSerialization,
                    "atomic",
                    "atomic_lane_serialization_unmodeled",
                ),
            ] {
                let reason = SyncCheckIncompleteReason::AnalysisGap {
                    operation: operation.clone(),
                    kind,
                };
                assert!(!is_irrelevant_after_terminal_error(&reason));
                let payload = sync_incomplete_to_py(py, &reason).unwrap();
                for (key, expected) in [("domain", domain), ("kind", "analysis_incomplete"), ("reason", label)] {
                    assert_eq!(
                        payload
                            .get_item(key)
                            .unwrap()
                            .unwrap()
                            .extract::<String>()
                            .unwrap(),
                        expected
                    );
                }
            }
        });
    }

}
